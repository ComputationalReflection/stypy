
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Test functions for the sparse.linalg._expm_multiply module
2: '''
3: 
4: from __future__ import division, print_function, absolute_import
5: 
6: import numpy as np
7: from numpy.testing import assert_allclose, assert_, assert_equal
8: from scipy._lib._numpy_compat import suppress_warnings
9: 
10: from scipy.sparse import SparseEfficiencyWarning
11: import scipy.linalg
12: from scipy.sparse.linalg._expm_multiply import (_theta, _compute_p_max,
13:         _onenormest_matrix_power, expm_multiply, _expm_multiply_simple,
14:         _expm_multiply_interval)
15: 
16: 
17: def less_than_or_close(a, b):
18:     return np.allclose(a, b) or (a < b)
19: 
20: 
21: class TestExpmActionSimple(object):
22:     '''
23:     These tests do not consider the case of multiple time steps in one call.
24:     '''
25: 
26:     def test_theta_monotonicity(self):
27:         pairs = sorted(_theta.items())
28:         for (m_a, theta_a), (m_b, theta_b) in zip(pairs[:-1], pairs[1:]):
29:             assert_(theta_a < theta_b)
30: 
31:     def test_p_max_default(self):
32:         m_max = 55
33:         expected_p_max = 8
34:         observed_p_max = _compute_p_max(m_max)
35:         assert_equal(observed_p_max, expected_p_max)
36: 
37:     def test_p_max_range(self):
38:         for m_max in range(1, 55+1):
39:             p_max = _compute_p_max(m_max)
40:             assert_(p_max*(p_max - 1) <= m_max + 1)
41:             p_too_big = p_max + 1
42:             assert_(p_too_big*(p_too_big - 1) > m_max + 1)
43: 
44:     def test_onenormest_matrix_power(self):
45:         np.random.seed(1234)
46:         n = 40
47:         nsamples = 10
48:         for i in range(nsamples):
49:             A = scipy.linalg.inv(np.random.randn(n, n))
50:             for p in range(4):
51:                 if not p:
52:                     M = np.identity(n)
53:                 else:
54:                     M = np.dot(M, A)
55:                 estimated = _onenormest_matrix_power(A, p)
56:                 exact = np.linalg.norm(M, 1)
57:                 assert_(less_than_or_close(estimated, exact))
58:                 assert_(less_than_or_close(exact, 3*estimated))
59: 
60:     def test_expm_multiply(self):
61:         np.random.seed(1234)
62:         n = 40
63:         k = 3
64:         nsamples = 10
65:         for i in range(nsamples):
66:             A = scipy.linalg.inv(np.random.randn(n, n))
67:             B = np.random.randn(n, k)
68:             observed = expm_multiply(A, B)
69:             expected = np.dot(scipy.linalg.expm(A), B)
70:             assert_allclose(observed, expected)
71: 
72:     def test_matrix_vector_multiply(self):
73:         np.random.seed(1234)
74:         n = 40
75:         nsamples = 10
76:         for i in range(nsamples):
77:             A = scipy.linalg.inv(np.random.randn(n, n))
78:             v = np.random.randn(n)
79:             observed = expm_multiply(A, v)
80:             expected = np.dot(scipy.linalg.expm(A), v)
81:             assert_allclose(observed, expected)
82: 
83:     def test_scaled_expm_multiply(self):
84:         np.random.seed(1234)
85:         n = 40
86:         k = 3
87:         nsamples = 10
88:         for i in range(nsamples):
89:             for t in (0.2, 1.0, 1.5):
90:                 with np.errstate(invalid='ignore'):
91:                     A = scipy.linalg.inv(np.random.randn(n, n))
92:                     B = np.random.randn(n, k)
93:                     observed = _expm_multiply_simple(A, B, t=t)
94:                     expected = np.dot(scipy.linalg.expm(t*A), B)
95:                     assert_allclose(observed, expected)
96: 
97:     def test_scaled_expm_multiply_single_timepoint(self):
98:         np.random.seed(1234)
99:         t = 0.1
100:         n = 5
101:         k = 2
102:         A = np.random.randn(n, n)
103:         B = np.random.randn(n, k)
104:         observed = _expm_multiply_simple(A, B, t=t)
105:         expected = scipy.linalg.expm(t*A).dot(B)
106:         assert_allclose(observed, expected)
107: 
108:     def test_sparse_expm_multiply(self):
109:         np.random.seed(1234)
110:         n = 40
111:         k = 3
112:         nsamples = 10
113:         for i in range(nsamples):
114:             A = scipy.sparse.rand(n, n, density=0.05)
115:             B = np.random.randn(n, k)
116:             observed = expm_multiply(A, B)
117:             with suppress_warnings() as sup:
118:                 sup.filter(SparseEfficiencyWarning,
119:                            "splu requires CSC matrix format")
120:                 sup.filter(SparseEfficiencyWarning,
121:                            "spsolve is more efficient when sparse b is in the CSC matrix format")
122:                 expected = scipy.linalg.expm(A).dot(B)
123:             assert_allclose(observed, expected)
124: 
125:     def test_complex(self):
126:         A = np.array([
127:             [1j, 1j],
128:             [0, 1j]], dtype=complex)
129:         B = np.array([1j, 1j])
130:         observed = expm_multiply(A, B)
131:         expected = np.array([
132:             1j * np.exp(1j) + 1j * (1j*np.cos(1) - np.sin(1)),
133:             1j * np.exp(1j)], dtype=complex)
134:         assert_allclose(observed, expected)
135: 
136: 
137: class TestExpmActionInterval(object):
138: 
139:     def test_sparse_expm_multiply_interval(self):
140:         np.random.seed(1234)
141:         start = 0.1
142:         stop = 3.2
143:         n = 40
144:         k = 3
145:         endpoint = True
146:         for num in (14, 13, 2):
147:             A = scipy.sparse.rand(n, n, density=0.05)
148:             B = np.random.randn(n, k)
149:             v = np.random.randn(n)
150:             for target in (B, v):
151:                 X = expm_multiply(A, target,
152:                         start=start, stop=stop, num=num, endpoint=endpoint)
153:                 samples = np.linspace(start=start, stop=stop,
154:                         num=num, endpoint=endpoint)
155:                 with suppress_warnings() as sup:
156:                     sup.filter(SparseEfficiencyWarning,
157:                                "splu requires CSC matrix format")
158:                     sup.filter(SparseEfficiencyWarning,
159:                                "spsolve is more efficient when sparse b is in the CSC matrix format")
160:                     for solution, t in zip(X, samples):
161:                         assert_allclose(solution,
162:                                 scipy.linalg.expm(t*A).dot(target))
163: 
164:     def test_expm_multiply_interval_vector(self):
165:         np.random.seed(1234)
166:         start = 0.1
167:         stop = 3.2
168:         endpoint = True
169:         for num in (14, 13, 2):
170:             for n in (1, 2, 5, 20, 40):
171:                 A = scipy.linalg.inv(np.random.randn(n, n))
172:                 v = np.random.randn(n)
173:                 X = expm_multiply(A, v,
174:                         start=start, stop=stop, num=num, endpoint=endpoint)
175:                 samples = np.linspace(start=start, stop=stop,
176:                         num=num, endpoint=endpoint)
177:                 for solution, t in zip(X, samples):
178:                     assert_allclose(solution, scipy.linalg.expm(t*A).dot(v))
179: 
180:     def test_expm_multiply_interval_matrix(self):
181:         np.random.seed(1234)
182:         start = 0.1
183:         stop = 3.2
184:         endpoint = True
185:         for num in (14, 13, 2):
186:             for n in (1, 2, 5, 20, 40):
187:                 for k in (1, 2):
188:                     A = scipy.linalg.inv(np.random.randn(n, n))
189:                     B = np.random.randn(n, k)
190:                     X = expm_multiply(A, B,
191:                             start=start, stop=stop, num=num, endpoint=endpoint)
192:                     samples = np.linspace(start=start, stop=stop,
193:                             num=num, endpoint=endpoint)
194:                     for solution, t in zip(X, samples):
195:                         assert_allclose(solution, scipy.linalg.expm(t*A).dot(B))
196: 
197:     def test_sparse_expm_multiply_interval_dtypes(self):
198:         # Test A & B int
199:         A = scipy.sparse.diags(np.arange(5),format='csr', dtype=int)
200:         B = np.ones(5, dtype=int)
201:         Aexpm = scipy.sparse.diags(np.exp(np.arange(5)),format='csr')
202:         assert_allclose(expm_multiply(A,B,0,1)[-1], Aexpm.dot(B))
203:     
204:         # Test A complex, B int
205:         A = scipy.sparse.diags(-1j*np.arange(5),format='csr', dtype=complex)
206:         B = np.ones(5, dtype=int)
207:         Aexpm = scipy.sparse.diags(np.exp(-1j*np.arange(5)),format='csr')
208:         assert_allclose(expm_multiply(A,B,0,1)[-1], Aexpm.dot(B))
209:     
210:         # Test A int, B complex
211:         A = scipy.sparse.diags(np.arange(5),format='csr', dtype=int)
212:         B = 1j*np.ones(5, dtype=complex)
213:         Aexpm = scipy.sparse.diags(np.exp(np.arange(5)),format='csr')
214:         assert_allclose(expm_multiply(A,B,0,1)[-1], Aexpm.dot(B))
215: 
216:     def test_expm_multiply_interval_status_0(self):
217:         self._help_test_specific_expm_interval_status(0)
218: 
219:     def test_expm_multiply_interval_status_1(self):
220:         self._help_test_specific_expm_interval_status(1)
221: 
222:     def test_expm_multiply_interval_status_2(self):
223:         self._help_test_specific_expm_interval_status(2)
224: 
225:     def _help_test_specific_expm_interval_status(self, target_status):
226:         np.random.seed(1234)
227:         start = 0.1
228:         stop = 3.2
229:         num = 13
230:         endpoint = True
231:         n = 5
232:         k = 2
233:         nrepeats = 10
234:         nsuccesses = 0
235:         for num in [14, 13, 2] * nrepeats:
236:             A = np.random.randn(n, n)
237:             B = np.random.randn(n, k)
238:             status = _expm_multiply_interval(A, B,
239:                     start=start, stop=stop, num=num, endpoint=endpoint,
240:                     status_only=True)
241:             if status == target_status:
242:                 X, status = _expm_multiply_interval(A, B,
243:                         start=start, stop=stop, num=num, endpoint=endpoint,
244:                         status_only=False)
245:                 assert_equal(X.shape, (num, n, k))
246:                 samples = np.linspace(start=start, stop=stop,
247:                         num=num, endpoint=endpoint)
248:                 for solution, t in zip(X, samples):
249:                     assert_allclose(solution, scipy.linalg.expm(t*A).dot(B))
250:                 nsuccesses += 1
251:         if not nsuccesses:
252:             msg = 'failed to find a status-' + str(target_status) + ' interval'
253:             raise Exception(msg)
254: 
255: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_422376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', 'Test functions for the sparse.linalg._expm_multiply module\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_422377 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_422377) is not StypyTypeError):

    if (import_422377 != 'pyd_module'):
        __import__(import_422377)
        sys_modules_422378 = sys.modules[import_422377]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_422378.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_422377)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_allclose, assert_, assert_equal' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_422379 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_422379) is not StypyTypeError):

    if (import_422379 != 'pyd_module'):
        __import__(import_422379)
        sys_modules_422380 = sys.modules[import_422379]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_422380.module_type_store, module_type_store, ['assert_allclose', 'assert_', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_422380, sys_modules_422380.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_allclose, assert_, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_allclose', 'assert_', 'assert_equal'], [assert_allclose, assert_, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_422379)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_422381 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib._numpy_compat')

if (type(import_422381) is not StypyTypeError):

    if (import_422381 != 'pyd_module'):
        __import__(import_422381)
        sys_modules_422382 = sys.modules[import_422381]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib._numpy_compat', sys_modules_422382.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_422382, sys_modules_422382.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib._numpy_compat', import_422381)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.sparse import SparseEfficiencyWarning' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_422383 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse')

if (type(import_422383) is not StypyTypeError):

    if (import_422383 != 'pyd_module'):
        __import__(import_422383)
        sys_modules_422384 = sys.modules[import_422383]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse', sys_modules_422384.module_type_store, module_type_store, ['SparseEfficiencyWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_422384, sys_modules_422384.module_type_store, module_type_store)
    else:
        from scipy.sparse import SparseEfficiencyWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse', None, module_type_store, ['SparseEfficiencyWarning'], [SparseEfficiencyWarning])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse', import_422383)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import scipy.linalg' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_422385 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg')

if (type(import_422385) is not StypyTypeError):

    if (import_422385 != 'pyd_module'):
        __import__(import_422385)
        sys_modules_422386 = sys.modules[import_422385]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg', sys_modules_422386.module_type_store, module_type_store)
    else:
        import scipy.linalg

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg', scipy.linalg, module_type_store)

else:
    # Assigning a type to the variable 'scipy.linalg' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg', import_422385)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.sparse.linalg._expm_multiply import _theta, _compute_p_max, _onenormest_matrix_power, expm_multiply, _expm_multiply_simple, _expm_multiply_interval' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_422387 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse.linalg._expm_multiply')

if (type(import_422387) is not StypyTypeError):

    if (import_422387 != 'pyd_module'):
        __import__(import_422387)
        sys_modules_422388 = sys.modules[import_422387]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse.linalg._expm_multiply', sys_modules_422388.module_type_store, module_type_store, ['_theta', '_compute_p_max', '_onenormest_matrix_power', 'expm_multiply', '_expm_multiply_simple', '_expm_multiply_interval'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_422388, sys_modules_422388.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg._expm_multiply import _theta, _compute_p_max, _onenormest_matrix_power, expm_multiply, _expm_multiply_simple, _expm_multiply_interval

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse.linalg._expm_multiply', None, module_type_store, ['_theta', '_compute_p_max', '_onenormest_matrix_power', 'expm_multiply', '_expm_multiply_simple', '_expm_multiply_interval'], [_theta, _compute_p_max, _onenormest_matrix_power, expm_multiply, _expm_multiply_simple, _expm_multiply_interval])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg._expm_multiply' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse.linalg._expm_multiply', import_422387)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')


@norecursion
def less_than_or_close(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'less_than_or_close'
    module_type_store = module_type_store.open_function_context('less_than_or_close', 17, 0, False)
    
    # Passed parameters checking function
    less_than_or_close.stypy_localization = localization
    less_than_or_close.stypy_type_of_self = None
    less_than_or_close.stypy_type_store = module_type_store
    less_than_or_close.stypy_function_name = 'less_than_or_close'
    less_than_or_close.stypy_param_names_list = ['a', 'b']
    less_than_or_close.stypy_varargs_param_name = None
    less_than_or_close.stypy_kwargs_param_name = None
    less_than_or_close.stypy_call_defaults = defaults
    less_than_or_close.stypy_call_varargs = varargs
    less_than_or_close.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'less_than_or_close', ['a', 'b'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'less_than_or_close', localization, ['a', 'b'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'less_than_or_close(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Call to allclose(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'a' (line 18)
    a_422391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 23), 'a', False)
    # Getting the type of 'b' (line 18)
    b_422392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 26), 'b', False)
    # Processing the call keyword arguments (line 18)
    kwargs_422393 = {}
    # Getting the type of 'np' (line 18)
    np_422389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'np', False)
    # Obtaining the member 'allclose' of a type (line 18)
    allclose_422390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 11), np_422389, 'allclose')
    # Calling allclose(args, kwargs) (line 18)
    allclose_call_result_422394 = invoke(stypy.reporting.localization.Localization(__file__, 18, 11), allclose_422390, *[a_422391, b_422392], **kwargs_422393)
    
    
    # Getting the type of 'a' (line 18)
    a_422395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 33), 'a')
    # Getting the type of 'b' (line 18)
    b_422396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 37), 'b')
    # Applying the binary operator '<' (line 18)
    result_lt_422397 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 33), '<', a_422395, b_422396)
    
    # Applying the binary operator 'or' (line 18)
    result_or_keyword_422398 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 11), 'or', allclose_call_result_422394, result_lt_422397)
    
    # Assigning a type to the variable 'stypy_return_type' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type', result_or_keyword_422398)
    
    # ################# End of 'less_than_or_close(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'less_than_or_close' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_422399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_422399)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'less_than_or_close'
    return stypy_return_type_422399

# Assigning a type to the variable 'less_than_or_close' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'less_than_or_close', less_than_or_close)
# Declaration of the 'TestExpmActionSimple' class

class TestExpmActionSimple(object, ):
    str_422400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, (-1)), 'str', '\n    These tests do not consider the case of multiple time steps in one call.\n    ')

    @norecursion
    def test_theta_monotonicity(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_theta_monotonicity'
        module_type_store = module_type_store.open_function_context('test_theta_monotonicity', 26, 4, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpmActionSimple.test_theta_monotonicity.__dict__.__setitem__('stypy_localization', localization)
        TestExpmActionSimple.test_theta_monotonicity.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpmActionSimple.test_theta_monotonicity.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpmActionSimple.test_theta_monotonicity.__dict__.__setitem__('stypy_function_name', 'TestExpmActionSimple.test_theta_monotonicity')
        TestExpmActionSimple.test_theta_monotonicity.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpmActionSimple.test_theta_monotonicity.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpmActionSimple.test_theta_monotonicity.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpmActionSimple.test_theta_monotonicity.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpmActionSimple.test_theta_monotonicity.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpmActionSimple.test_theta_monotonicity.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpmActionSimple.test_theta_monotonicity.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpmActionSimple.test_theta_monotonicity', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_theta_monotonicity', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_theta_monotonicity(...)' code ##################

        
        # Assigning a Call to a Name (line 27):
        
        # Assigning a Call to a Name (line 27):
        
        # Call to sorted(...): (line 27)
        # Processing the call arguments (line 27)
        
        # Call to items(...): (line 27)
        # Processing the call keyword arguments (line 27)
        kwargs_422404 = {}
        # Getting the type of '_theta' (line 27)
        _theta_422402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 23), '_theta', False)
        # Obtaining the member 'items' of a type (line 27)
        items_422403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 23), _theta_422402, 'items')
        # Calling items(args, kwargs) (line 27)
        items_call_result_422405 = invoke(stypy.reporting.localization.Localization(__file__, 27, 23), items_422403, *[], **kwargs_422404)
        
        # Processing the call keyword arguments (line 27)
        kwargs_422406 = {}
        # Getting the type of 'sorted' (line 27)
        sorted_422401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 16), 'sorted', False)
        # Calling sorted(args, kwargs) (line 27)
        sorted_call_result_422407 = invoke(stypy.reporting.localization.Localization(__file__, 27, 16), sorted_422401, *[items_call_result_422405], **kwargs_422406)
        
        # Assigning a type to the variable 'pairs' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'pairs', sorted_call_result_422407)
        
        
        # Call to zip(...): (line 28)
        # Processing the call arguments (line 28)
        
        # Obtaining the type of the subscript
        int_422409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 57), 'int')
        slice_422410 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 28, 50), None, int_422409, None)
        # Getting the type of 'pairs' (line 28)
        pairs_422411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 50), 'pairs', False)
        # Obtaining the member '__getitem__' of a type (line 28)
        getitem___422412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 50), pairs_422411, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 28)
        subscript_call_result_422413 = invoke(stypy.reporting.localization.Localization(__file__, 28, 50), getitem___422412, slice_422410)
        
        
        # Obtaining the type of the subscript
        int_422414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 68), 'int')
        slice_422415 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 28, 62), int_422414, None, None)
        # Getting the type of 'pairs' (line 28)
        pairs_422416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 62), 'pairs', False)
        # Obtaining the member '__getitem__' of a type (line 28)
        getitem___422417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 62), pairs_422416, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 28)
        subscript_call_result_422418 = invoke(stypy.reporting.localization.Localization(__file__, 28, 62), getitem___422417, slice_422415)
        
        # Processing the call keyword arguments (line 28)
        kwargs_422419 = {}
        # Getting the type of 'zip' (line 28)
        zip_422408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 46), 'zip', False)
        # Calling zip(args, kwargs) (line 28)
        zip_call_result_422420 = invoke(stypy.reporting.localization.Localization(__file__, 28, 46), zip_422408, *[subscript_call_result_422413, subscript_call_result_422418], **kwargs_422419)
        
        # Testing the type of a for loop iterable (line 28)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 28, 8), zip_call_result_422420)
        # Getting the type of the for loop variable (line 28)
        for_loop_var_422421 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 28, 8), zip_call_result_422420)
        # Assigning a type to the variable 'm_a' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'm_a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 8), for_loop_var_422421))
        # Assigning a type to the variable 'theta_a' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'theta_a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 8), for_loop_var_422421))
        # Assigning a type to the variable 'm_b' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'm_b', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 8), for_loop_var_422421))
        # Assigning a type to the variable 'theta_b' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'theta_b', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 8), for_loop_var_422421))
        # SSA begins for a for statement (line 28)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_(...): (line 29)
        # Processing the call arguments (line 29)
        
        # Getting the type of 'theta_a' (line 29)
        theta_a_422423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 20), 'theta_a', False)
        # Getting the type of 'theta_b' (line 29)
        theta_b_422424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 30), 'theta_b', False)
        # Applying the binary operator '<' (line 29)
        result_lt_422425 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 20), '<', theta_a_422423, theta_b_422424)
        
        # Processing the call keyword arguments (line 29)
        kwargs_422426 = {}
        # Getting the type of 'assert_' (line 29)
        assert__422422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 29)
        assert__call_result_422427 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), assert__422422, *[result_lt_422425], **kwargs_422426)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_theta_monotonicity(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_theta_monotonicity' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_422428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_422428)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_theta_monotonicity'
        return stypy_return_type_422428


    @norecursion
    def test_p_max_default(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_p_max_default'
        module_type_store = module_type_store.open_function_context('test_p_max_default', 31, 4, False)
        # Assigning a type to the variable 'self' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpmActionSimple.test_p_max_default.__dict__.__setitem__('stypy_localization', localization)
        TestExpmActionSimple.test_p_max_default.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpmActionSimple.test_p_max_default.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpmActionSimple.test_p_max_default.__dict__.__setitem__('stypy_function_name', 'TestExpmActionSimple.test_p_max_default')
        TestExpmActionSimple.test_p_max_default.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpmActionSimple.test_p_max_default.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpmActionSimple.test_p_max_default.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpmActionSimple.test_p_max_default.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpmActionSimple.test_p_max_default.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpmActionSimple.test_p_max_default.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpmActionSimple.test_p_max_default.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpmActionSimple.test_p_max_default', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_p_max_default', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_p_max_default(...)' code ##################

        
        # Assigning a Num to a Name (line 32):
        
        # Assigning a Num to a Name (line 32):
        int_422429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 16), 'int')
        # Assigning a type to the variable 'm_max' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'm_max', int_422429)
        
        # Assigning a Num to a Name (line 33):
        
        # Assigning a Num to a Name (line 33):
        int_422430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 25), 'int')
        # Assigning a type to the variable 'expected_p_max' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'expected_p_max', int_422430)
        
        # Assigning a Call to a Name (line 34):
        
        # Assigning a Call to a Name (line 34):
        
        # Call to _compute_p_max(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'm_max' (line 34)
        m_max_422432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 40), 'm_max', False)
        # Processing the call keyword arguments (line 34)
        kwargs_422433 = {}
        # Getting the type of '_compute_p_max' (line 34)
        _compute_p_max_422431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 25), '_compute_p_max', False)
        # Calling _compute_p_max(args, kwargs) (line 34)
        _compute_p_max_call_result_422434 = invoke(stypy.reporting.localization.Localization(__file__, 34, 25), _compute_p_max_422431, *[m_max_422432], **kwargs_422433)
        
        # Assigning a type to the variable 'observed_p_max' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'observed_p_max', _compute_p_max_call_result_422434)
        
        # Call to assert_equal(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'observed_p_max' (line 35)
        observed_p_max_422436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 21), 'observed_p_max', False)
        # Getting the type of 'expected_p_max' (line 35)
        expected_p_max_422437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 37), 'expected_p_max', False)
        # Processing the call keyword arguments (line 35)
        kwargs_422438 = {}
        # Getting the type of 'assert_equal' (line 35)
        assert_equal_422435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 35)
        assert_equal_call_result_422439 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), assert_equal_422435, *[observed_p_max_422436, expected_p_max_422437], **kwargs_422438)
        
        
        # ################# End of 'test_p_max_default(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_p_max_default' in the type store
        # Getting the type of 'stypy_return_type' (line 31)
        stypy_return_type_422440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_422440)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_p_max_default'
        return stypy_return_type_422440


    @norecursion
    def test_p_max_range(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_p_max_range'
        module_type_store = module_type_store.open_function_context('test_p_max_range', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpmActionSimple.test_p_max_range.__dict__.__setitem__('stypy_localization', localization)
        TestExpmActionSimple.test_p_max_range.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpmActionSimple.test_p_max_range.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpmActionSimple.test_p_max_range.__dict__.__setitem__('stypy_function_name', 'TestExpmActionSimple.test_p_max_range')
        TestExpmActionSimple.test_p_max_range.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpmActionSimple.test_p_max_range.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpmActionSimple.test_p_max_range.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpmActionSimple.test_p_max_range.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpmActionSimple.test_p_max_range.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpmActionSimple.test_p_max_range.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpmActionSimple.test_p_max_range.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpmActionSimple.test_p_max_range', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_p_max_range', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_p_max_range(...)' code ##################

        
        
        # Call to range(...): (line 38)
        # Processing the call arguments (line 38)
        int_422442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 27), 'int')
        int_422443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 30), 'int')
        int_422444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 33), 'int')
        # Applying the binary operator '+' (line 38)
        result_add_422445 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 30), '+', int_422443, int_422444)
        
        # Processing the call keyword arguments (line 38)
        kwargs_422446 = {}
        # Getting the type of 'range' (line 38)
        range_422441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 21), 'range', False)
        # Calling range(args, kwargs) (line 38)
        range_call_result_422447 = invoke(stypy.reporting.localization.Localization(__file__, 38, 21), range_422441, *[int_422442, result_add_422445], **kwargs_422446)
        
        # Testing the type of a for loop iterable (line 38)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 38, 8), range_call_result_422447)
        # Getting the type of the for loop variable (line 38)
        for_loop_var_422448 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 38, 8), range_call_result_422447)
        # Assigning a type to the variable 'm_max' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'm_max', for_loop_var_422448)
        # SSA begins for a for statement (line 38)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 39):
        
        # Assigning a Call to a Name (line 39):
        
        # Call to _compute_p_max(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'm_max' (line 39)
        m_max_422450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 35), 'm_max', False)
        # Processing the call keyword arguments (line 39)
        kwargs_422451 = {}
        # Getting the type of '_compute_p_max' (line 39)
        _compute_p_max_422449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), '_compute_p_max', False)
        # Calling _compute_p_max(args, kwargs) (line 39)
        _compute_p_max_call_result_422452 = invoke(stypy.reporting.localization.Localization(__file__, 39, 20), _compute_p_max_422449, *[m_max_422450], **kwargs_422451)
        
        # Assigning a type to the variable 'p_max' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'p_max', _compute_p_max_call_result_422452)
        
        # Call to assert_(...): (line 40)
        # Processing the call arguments (line 40)
        
        # Getting the type of 'p_max' (line 40)
        p_max_422454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'p_max', False)
        # Getting the type of 'p_max' (line 40)
        p_max_422455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 27), 'p_max', False)
        int_422456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 35), 'int')
        # Applying the binary operator '-' (line 40)
        result_sub_422457 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 27), '-', p_max_422455, int_422456)
        
        # Applying the binary operator '*' (line 40)
        result_mul_422458 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 20), '*', p_max_422454, result_sub_422457)
        
        # Getting the type of 'm_max' (line 40)
        m_max_422459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 41), 'm_max', False)
        int_422460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 49), 'int')
        # Applying the binary operator '+' (line 40)
        result_add_422461 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 41), '+', m_max_422459, int_422460)
        
        # Applying the binary operator '<=' (line 40)
        result_le_422462 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 20), '<=', result_mul_422458, result_add_422461)
        
        # Processing the call keyword arguments (line 40)
        kwargs_422463 = {}
        # Getting the type of 'assert_' (line 40)
        assert__422453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 40)
        assert__call_result_422464 = invoke(stypy.reporting.localization.Localization(__file__, 40, 12), assert__422453, *[result_le_422462], **kwargs_422463)
        
        
        # Assigning a BinOp to a Name (line 41):
        
        # Assigning a BinOp to a Name (line 41):
        # Getting the type of 'p_max' (line 41)
        p_max_422465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 24), 'p_max')
        int_422466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 32), 'int')
        # Applying the binary operator '+' (line 41)
        result_add_422467 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 24), '+', p_max_422465, int_422466)
        
        # Assigning a type to the variable 'p_too_big' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'p_too_big', result_add_422467)
        
        # Call to assert_(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Getting the type of 'p_too_big' (line 42)
        p_too_big_422469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'p_too_big', False)
        # Getting the type of 'p_too_big' (line 42)
        p_too_big_422470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 31), 'p_too_big', False)
        int_422471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 43), 'int')
        # Applying the binary operator '-' (line 42)
        result_sub_422472 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 31), '-', p_too_big_422470, int_422471)
        
        # Applying the binary operator '*' (line 42)
        result_mul_422473 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 20), '*', p_too_big_422469, result_sub_422472)
        
        # Getting the type of 'm_max' (line 42)
        m_max_422474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 48), 'm_max', False)
        int_422475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 56), 'int')
        # Applying the binary operator '+' (line 42)
        result_add_422476 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 48), '+', m_max_422474, int_422475)
        
        # Applying the binary operator '>' (line 42)
        result_gt_422477 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 20), '>', result_mul_422473, result_add_422476)
        
        # Processing the call keyword arguments (line 42)
        kwargs_422478 = {}
        # Getting the type of 'assert_' (line 42)
        assert__422468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 42)
        assert__call_result_422479 = invoke(stypy.reporting.localization.Localization(__file__, 42, 12), assert__422468, *[result_gt_422477], **kwargs_422478)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_p_max_range(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_p_max_range' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_422480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_422480)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_p_max_range'
        return stypy_return_type_422480


    @norecursion
    def test_onenormest_matrix_power(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_onenormest_matrix_power'
        module_type_store = module_type_store.open_function_context('test_onenormest_matrix_power', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpmActionSimple.test_onenormest_matrix_power.__dict__.__setitem__('stypy_localization', localization)
        TestExpmActionSimple.test_onenormest_matrix_power.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpmActionSimple.test_onenormest_matrix_power.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpmActionSimple.test_onenormest_matrix_power.__dict__.__setitem__('stypy_function_name', 'TestExpmActionSimple.test_onenormest_matrix_power')
        TestExpmActionSimple.test_onenormest_matrix_power.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpmActionSimple.test_onenormest_matrix_power.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpmActionSimple.test_onenormest_matrix_power.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpmActionSimple.test_onenormest_matrix_power.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpmActionSimple.test_onenormest_matrix_power.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpmActionSimple.test_onenormest_matrix_power.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpmActionSimple.test_onenormest_matrix_power.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpmActionSimple.test_onenormest_matrix_power', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_onenormest_matrix_power', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_onenormest_matrix_power(...)' code ##################

        
        # Call to seed(...): (line 45)
        # Processing the call arguments (line 45)
        int_422484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 23), 'int')
        # Processing the call keyword arguments (line 45)
        kwargs_422485 = {}
        # Getting the type of 'np' (line 45)
        np_422481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 45)
        random_422482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), np_422481, 'random')
        # Obtaining the member 'seed' of a type (line 45)
        seed_422483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), random_422482, 'seed')
        # Calling seed(args, kwargs) (line 45)
        seed_call_result_422486 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), seed_422483, *[int_422484], **kwargs_422485)
        
        
        # Assigning a Num to a Name (line 46):
        
        # Assigning a Num to a Name (line 46):
        int_422487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 12), 'int')
        # Assigning a type to the variable 'n' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'n', int_422487)
        
        # Assigning a Num to a Name (line 47):
        
        # Assigning a Num to a Name (line 47):
        int_422488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 19), 'int')
        # Assigning a type to the variable 'nsamples' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'nsamples', int_422488)
        
        
        # Call to range(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'nsamples' (line 48)
        nsamples_422490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 23), 'nsamples', False)
        # Processing the call keyword arguments (line 48)
        kwargs_422491 = {}
        # Getting the type of 'range' (line 48)
        range_422489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 17), 'range', False)
        # Calling range(args, kwargs) (line 48)
        range_call_result_422492 = invoke(stypy.reporting.localization.Localization(__file__, 48, 17), range_422489, *[nsamples_422490], **kwargs_422491)
        
        # Testing the type of a for loop iterable (line 48)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 48, 8), range_call_result_422492)
        # Getting the type of the for loop variable (line 48)
        for_loop_var_422493 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 48, 8), range_call_result_422492)
        # Assigning a type to the variable 'i' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'i', for_loop_var_422493)
        # SSA begins for a for statement (line 48)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 49):
        
        # Assigning a Call to a Name (line 49):
        
        # Call to inv(...): (line 49)
        # Processing the call arguments (line 49)
        
        # Call to randn(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'n' (line 49)
        n_422500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 49), 'n', False)
        # Getting the type of 'n' (line 49)
        n_422501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 52), 'n', False)
        # Processing the call keyword arguments (line 49)
        kwargs_422502 = {}
        # Getting the type of 'np' (line 49)
        np_422497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 33), 'np', False)
        # Obtaining the member 'random' of a type (line 49)
        random_422498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 33), np_422497, 'random')
        # Obtaining the member 'randn' of a type (line 49)
        randn_422499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 33), random_422498, 'randn')
        # Calling randn(args, kwargs) (line 49)
        randn_call_result_422503 = invoke(stypy.reporting.localization.Localization(__file__, 49, 33), randn_422499, *[n_422500, n_422501], **kwargs_422502)
        
        # Processing the call keyword arguments (line 49)
        kwargs_422504 = {}
        # Getting the type of 'scipy' (line 49)
        scipy_422494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 49)
        linalg_422495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 16), scipy_422494, 'linalg')
        # Obtaining the member 'inv' of a type (line 49)
        inv_422496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 16), linalg_422495, 'inv')
        # Calling inv(args, kwargs) (line 49)
        inv_call_result_422505 = invoke(stypy.reporting.localization.Localization(__file__, 49, 16), inv_422496, *[randn_call_result_422503], **kwargs_422504)
        
        # Assigning a type to the variable 'A' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'A', inv_call_result_422505)
        
        
        # Call to range(...): (line 50)
        # Processing the call arguments (line 50)
        int_422507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 27), 'int')
        # Processing the call keyword arguments (line 50)
        kwargs_422508 = {}
        # Getting the type of 'range' (line 50)
        range_422506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 21), 'range', False)
        # Calling range(args, kwargs) (line 50)
        range_call_result_422509 = invoke(stypy.reporting.localization.Localization(__file__, 50, 21), range_422506, *[int_422507], **kwargs_422508)
        
        # Testing the type of a for loop iterable (line 50)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 50, 12), range_call_result_422509)
        # Getting the type of the for loop variable (line 50)
        for_loop_var_422510 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 50, 12), range_call_result_422509)
        # Assigning a type to the variable 'p' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'p', for_loop_var_422510)
        # SSA begins for a for statement (line 50)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'p' (line 51)
        p_422511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 23), 'p')
        # Applying the 'not' unary operator (line 51)
        result_not__422512 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 19), 'not', p_422511)
        
        # Testing the type of an if condition (line 51)
        if_condition_422513 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 51, 16), result_not__422512)
        # Assigning a type to the variable 'if_condition_422513' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'if_condition_422513', if_condition_422513)
        # SSA begins for if statement (line 51)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 52):
        
        # Assigning a Call to a Name (line 52):
        
        # Call to identity(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'n' (line 52)
        n_422516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 36), 'n', False)
        # Processing the call keyword arguments (line 52)
        kwargs_422517 = {}
        # Getting the type of 'np' (line 52)
        np_422514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 24), 'np', False)
        # Obtaining the member 'identity' of a type (line 52)
        identity_422515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 24), np_422514, 'identity')
        # Calling identity(args, kwargs) (line 52)
        identity_call_result_422518 = invoke(stypy.reporting.localization.Localization(__file__, 52, 24), identity_422515, *[n_422516], **kwargs_422517)
        
        # Assigning a type to the variable 'M' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 20), 'M', identity_call_result_422518)
        # SSA branch for the else part of an if statement (line 51)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 54):
        
        # Assigning a Call to a Name (line 54):
        
        # Call to dot(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'M' (line 54)
        M_422521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 31), 'M', False)
        # Getting the type of 'A' (line 54)
        A_422522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 34), 'A', False)
        # Processing the call keyword arguments (line 54)
        kwargs_422523 = {}
        # Getting the type of 'np' (line 54)
        np_422519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'np', False)
        # Obtaining the member 'dot' of a type (line 54)
        dot_422520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 24), np_422519, 'dot')
        # Calling dot(args, kwargs) (line 54)
        dot_call_result_422524 = invoke(stypy.reporting.localization.Localization(__file__, 54, 24), dot_422520, *[M_422521, A_422522], **kwargs_422523)
        
        # Assigning a type to the variable 'M' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'M', dot_call_result_422524)
        # SSA join for if statement (line 51)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 55):
        
        # Assigning a Call to a Name (line 55):
        
        # Call to _onenormest_matrix_power(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'A' (line 55)
        A_422526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 53), 'A', False)
        # Getting the type of 'p' (line 55)
        p_422527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 56), 'p', False)
        # Processing the call keyword arguments (line 55)
        kwargs_422528 = {}
        # Getting the type of '_onenormest_matrix_power' (line 55)
        _onenormest_matrix_power_422525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 28), '_onenormest_matrix_power', False)
        # Calling _onenormest_matrix_power(args, kwargs) (line 55)
        _onenormest_matrix_power_call_result_422529 = invoke(stypy.reporting.localization.Localization(__file__, 55, 28), _onenormest_matrix_power_422525, *[A_422526, p_422527], **kwargs_422528)
        
        # Assigning a type to the variable 'estimated' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'estimated', _onenormest_matrix_power_call_result_422529)
        
        # Assigning a Call to a Name (line 56):
        
        # Assigning a Call to a Name (line 56):
        
        # Call to norm(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'M' (line 56)
        M_422533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 39), 'M', False)
        int_422534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 42), 'int')
        # Processing the call keyword arguments (line 56)
        kwargs_422535 = {}
        # Getting the type of 'np' (line 56)
        np_422530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 24), 'np', False)
        # Obtaining the member 'linalg' of a type (line 56)
        linalg_422531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 24), np_422530, 'linalg')
        # Obtaining the member 'norm' of a type (line 56)
        norm_422532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 24), linalg_422531, 'norm')
        # Calling norm(args, kwargs) (line 56)
        norm_call_result_422536 = invoke(stypy.reporting.localization.Localization(__file__, 56, 24), norm_422532, *[M_422533, int_422534], **kwargs_422535)
        
        # Assigning a type to the variable 'exact' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'exact', norm_call_result_422536)
        
        # Call to assert_(...): (line 57)
        # Processing the call arguments (line 57)
        
        # Call to less_than_or_close(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'estimated' (line 57)
        estimated_422539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 43), 'estimated', False)
        # Getting the type of 'exact' (line 57)
        exact_422540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 54), 'exact', False)
        # Processing the call keyword arguments (line 57)
        kwargs_422541 = {}
        # Getting the type of 'less_than_or_close' (line 57)
        less_than_or_close_422538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 24), 'less_than_or_close', False)
        # Calling less_than_or_close(args, kwargs) (line 57)
        less_than_or_close_call_result_422542 = invoke(stypy.reporting.localization.Localization(__file__, 57, 24), less_than_or_close_422538, *[estimated_422539, exact_422540], **kwargs_422541)
        
        # Processing the call keyword arguments (line 57)
        kwargs_422543 = {}
        # Getting the type of 'assert_' (line 57)
        assert__422537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'assert_', False)
        # Calling assert_(args, kwargs) (line 57)
        assert__call_result_422544 = invoke(stypy.reporting.localization.Localization(__file__, 57, 16), assert__422537, *[less_than_or_close_call_result_422542], **kwargs_422543)
        
        
        # Call to assert_(...): (line 58)
        # Processing the call arguments (line 58)
        
        # Call to less_than_or_close(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'exact' (line 58)
        exact_422547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 43), 'exact', False)
        int_422548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 50), 'int')
        # Getting the type of 'estimated' (line 58)
        estimated_422549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 52), 'estimated', False)
        # Applying the binary operator '*' (line 58)
        result_mul_422550 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 50), '*', int_422548, estimated_422549)
        
        # Processing the call keyword arguments (line 58)
        kwargs_422551 = {}
        # Getting the type of 'less_than_or_close' (line 58)
        less_than_or_close_422546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 24), 'less_than_or_close', False)
        # Calling less_than_or_close(args, kwargs) (line 58)
        less_than_or_close_call_result_422552 = invoke(stypy.reporting.localization.Localization(__file__, 58, 24), less_than_or_close_422546, *[exact_422547, result_mul_422550], **kwargs_422551)
        
        # Processing the call keyword arguments (line 58)
        kwargs_422553 = {}
        # Getting the type of 'assert_' (line 58)
        assert__422545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'assert_', False)
        # Calling assert_(args, kwargs) (line 58)
        assert__call_result_422554 = invoke(stypy.reporting.localization.Localization(__file__, 58, 16), assert__422545, *[less_than_or_close_call_result_422552], **kwargs_422553)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_onenormest_matrix_power(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_onenormest_matrix_power' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_422555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_422555)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_onenormest_matrix_power'
        return stypy_return_type_422555


    @norecursion
    def test_expm_multiply(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_expm_multiply'
        module_type_store = module_type_store.open_function_context('test_expm_multiply', 60, 4, False)
        # Assigning a type to the variable 'self' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpmActionSimple.test_expm_multiply.__dict__.__setitem__('stypy_localization', localization)
        TestExpmActionSimple.test_expm_multiply.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpmActionSimple.test_expm_multiply.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpmActionSimple.test_expm_multiply.__dict__.__setitem__('stypy_function_name', 'TestExpmActionSimple.test_expm_multiply')
        TestExpmActionSimple.test_expm_multiply.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpmActionSimple.test_expm_multiply.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpmActionSimple.test_expm_multiply.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpmActionSimple.test_expm_multiply.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpmActionSimple.test_expm_multiply.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpmActionSimple.test_expm_multiply.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpmActionSimple.test_expm_multiply.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpmActionSimple.test_expm_multiply', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_expm_multiply', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_expm_multiply(...)' code ##################

        
        # Call to seed(...): (line 61)
        # Processing the call arguments (line 61)
        int_422559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 23), 'int')
        # Processing the call keyword arguments (line 61)
        kwargs_422560 = {}
        # Getting the type of 'np' (line 61)
        np_422556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 61)
        random_422557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), np_422556, 'random')
        # Obtaining the member 'seed' of a type (line 61)
        seed_422558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), random_422557, 'seed')
        # Calling seed(args, kwargs) (line 61)
        seed_call_result_422561 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), seed_422558, *[int_422559], **kwargs_422560)
        
        
        # Assigning a Num to a Name (line 62):
        
        # Assigning a Num to a Name (line 62):
        int_422562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 12), 'int')
        # Assigning a type to the variable 'n' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'n', int_422562)
        
        # Assigning a Num to a Name (line 63):
        
        # Assigning a Num to a Name (line 63):
        int_422563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 12), 'int')
        # Assigning a type to the variable 'k' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'k', int_422563)
        
        # Assigning a Num to a Name (line 64):
        
        # Assigning a Num to a Name (line 64):
        int_422564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 19), 'int')
        # Assigning a type to the variable 'nsamples' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'nsamples', int_422564)
        
        
        # Call to range(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'nsamples' (line 65)
        nsamples_422566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 23), 'nsamples', False)
        # Processing the call keyword arguments (line 65)
        kwargs_422567 = {}
        # Getting the type of 'range' (line 65)
        range_422565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 17), 'range', False)
        # Calling range(args, kwargs) (line 65)
        range_call_result_422568 = invoke(stypy.reporting.localization.Localization(__file__, 65, 17), range_422565, *[nsamples_422566], **kwargs_422567)
        
        # Testing the type of a for loop iterable (line 65)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 65, 8), range_call_result_422568)
        # Getting the type of the for loop variable (line 65)
        for_loop_var_422569 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 65, 8), range_call_result_422568)
        # Assigning a type to the variable 'i' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'i', for_loop_var_422569)
        # SSA begins for a for statement (line 65)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 66):
        
        # Assigning a Call to a Name (line 66):
        
        # Call to inv(...): (line 66)
        # Processing the call arguments (line 66)
        
        # Call to randn(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'n' (line 66)
        n_422576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 49), 'n', False)
        # Getting the type of 'n' (line 66)
        n_422577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 52), 'n', False)
        # Processing the call keyword arguments (line 66)
        kwargs_422578 = {}
        # Getting the type of 'np' (line 66)
        np_422573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 33), 'np', False)
        # Obtaining the member 'random' of a type (line 66)
        random_422574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 33), np_422573, 'random')
        # Obtaining the member 'randn' of a type (line 66)
        randn_422575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 33), random_422574, 'randn')
        # Calling randn(args, kwargs) (line 66)
        randn_call_result_422579 = invoke(stypy.reporting.localization.Localization(__file__, 66, 33), randn_422575, *[n_422576, n_422577], **kwargs_422578)
        
        # Processing the call keyword arguments (line 66)
        kwargs_422580 = {}
        # Getting the type of 'scipy' (line 66)
        scipy_422570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 66)
        linalg_422571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 16), scipy_422570, 'linalg')
        # Obtaining the member 'inv' of a type (line 66)
        inv_422572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 16), linalg_422571, 'inv')
        # Calling inv(args, kwargs) (line 66)
        inv_call_result_422581 = invoke(stypy.reporting.localization.Localization(__file__, 66, 16), inv_422572, *[randn_call_result_422579], **kwargs_422580)
        
        # Assigning a type to the variable 'A' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'A', inv_call_result_422581)
        
        # Assigning a Call to a Name (line 67):
        
        # Assigning a Call to a Name (line 67):
        
        # Call to randn(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'n' (line 67)
        n_422585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 32), 'n', False)
        # Getting the type of 'k' (line 67)
        k_422586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 35), 'k', False)
        # Processing the call keyword arguments (line 67)
        kwargs_422587 = {}
        # Getting the type of 'np' (line 67)
        np_422582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 16), 'np', False)
        # Obtaining the member 'random' of a type (line 67)
        random_422583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 16), np_422582, 'random')
        # Obtaining the member 'randn' of a type (line 67)
        randn_422584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 16), random_422583, 'randn')
        # Calling randn(args, kwargs) (line 67)
        randn_call_result_422588 = invoke(stypy.reporting.localization.Localization(__file__, 67, 16), randn_422584, *[n_422585, k_422586], **kwargs_422587)
        
        # Assigning a type to the variable 'B' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'B', randn_call_result_422588)
        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Call to expm_multiply(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'A' (line 68)
        A_422590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 37), 'A', False)
        # Getting the type of 'B' (line 68)
        B_422591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 40), 'B', False)
        # Processing the call keyword arguments (line 68)
        kwargs_422592 = {}
        # Getting the type of 'expm_multiply' (line 68)
        expm_multiply_422589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 23), 'expm_multiply', False)
        # Calling expm_multiply(args, kwargs) (line 68)
        expm_multiply_call_result_422593 = invoke(stypy.reporting.localization.Localization(__file__, 68, 23), expm_multiply_422589, *[A_422590, B_422591], **kwargs_422592)
        
        # Assigning a type to the variable 'observed' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'observed', expm_multiply_call_result_422593)
        
        # Assigning a Call to a Name (line 69):
        
        # Assigning a Call to a Name (line 69):
        
        # Call to dot(...): (line 69)
        # Processing the call arguments (line 69)
        
        # Call to expm(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'A' (line 69)
        A_422599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 48), 'A', False)
        # Processing the call keyword arguments (line 69)
        kwargs_422600 = {}
        # Getting the type of 'scipy' (line 69)
        scipy_422596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 30), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 69)
        linalg_422597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 30), scipy_422596, 'linalg')
        # Obtaining the member 'expm' of a type (line 69)
        expm_422598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 30), linalg_422597, 'expm')
        # Calling expm(args, kwargs) (line 69)
        expm_call_result_422601 = invoke(stypy.reporting.localization.Localization(__file__, 69, 30), expm_422598, *[A_422599], **kwargs_422600)
        
        # Getting the type of 'B' (line 69)
        B_422602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 52), 'B', False)
        # Processing the call keyword arguments (line 69)
        kwargs_422603 = {}
        # Getting the type of 'np' (line 69)
        np_422594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 23), 'np', False)
        # Obtaining the member 'dot' of a type (line 69)
        dot_422595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 23), np_422594, 'dot')
        # Calling dot(args, kwargs) (line 69)
        dot_call_result_422604 = invoke(stypy.reporting.localization.Localization(__file__, 69, 23), dot_422595, *[expm_call_result_422601, B_422602], **kwargs_422603)
        
        # Assigning a type to the variable 'expected' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'expected', dot_call_result_422604)
        
        # Call to assert_allclose(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'observed' (line 70)
        observed_422606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 28), 'observed', False)
        # Getting the type of 'expected' (line 70)
        expected_422607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 38), 'expected', False)
        # Processing the call keyword arguments (line 70)
        kwargs_422608 = {}
        # Getting the type of 'assert_allclose' (line 70)
        assert_allclose_422605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 70)
        assert_allclose_call_result_422609 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), assert_allclose_422605, *[observed_422606, expected_422607], **kwargs_422608)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_expm_multiply(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_expm_multiply' in the type store
        # Getting the type of 'stypy_return_type' (line 60)
        stypy_return_type_422610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_422610)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_expm_multiply'
        return stypy_return_type_422610


    @norecursion
    def test_matrix_vector_multiply(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_matrix_vector_multiply'
        module_type_store = module_type_store.open_function_context('test_matrix_vector_multiply', 72, 4, False)
        # Assigning a type to the variable 'self' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpmActionSimple.test_matrix_vector_multiply.__dict__.__setitem__('stypy_localization', localization)
        TestExpmActionSimple.test_matrix_vector_multiply.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpmActionSimple.test_matrix_vector_multiply.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpmActionSimple.test_matrix_vector_multiply.__dict__.__setitem__('stypy_function_name', 'TestExpmActionSimple.test_matrix_vector_multiply')
        TestExpmActionSimple.test_matrix_vector_multiply.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpmActionSimple.test_matrix_vector_multiply.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpmActionSimple.test_matrix_vector_multiply.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpmActionSimple.test_matrix_vector_multiply.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpmActionSimple.test_matrix_vector_multiply.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpmActionSimple.test_matrix_vector_multiply.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpmActionSimple.test_matrix_vector_multiply.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpmActionSimple.test_matrix_vector_multiply', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_matrix_vector_multiply', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_matrix_vector_multiply(...)' code ##################

        
        # Call to seed(...): (line 73)
        # Processing the call arguments (line 73)
        int_422614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 23), 'int')
        # Processing the call keyword arguments (line 73)
        kwargs_422615 = {}
        # Getting the type of 'np' (line 73)
        np_422611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 73)
        random_422612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), np_422611, 'random')
        # Obtaining the member 'seed' of a type (line 73)
        seed_422613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), random_422612, 'seed')
        # Calling seed(args, kwargs) (line 73)
        seed_call_result_422616 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), seed_422613, *[int_422614], **kwargs_422615)
        
        
        # Assigning a Num to a Name (line 74):
        
        # Assigning a Num to a Name (line 74):
        int_422617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 12), 'int')
        # Assigning a type to the variable 'n' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'n', int_422617)
        
        # Assigning a Num to a Name (line 75):
        
        # Assigning a Num to a Name (line 75):
        int_422618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 19), 'int')
        # Assigning a type to the variable 'nsamples' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'nsamples', int_422618)
        
        
        # Call to range(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'nsamples' (line 76)
        nsamples_422620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'nsamples', False)
        # Processing the call keyword arguments (line 76)
        kwargs_422621 = {}
        # Getting the type of 'range' (line 76)
        range_422619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 17), 'range', False)
        # Calling range(args, kwargs) (line 76)
        range_call_result_422622 = invoke(stypy.reporting.localization.Localization(__file__, 76, 17), range_422619, *[nsamples_422620], **kwargs_422621)
        
        # Testing the type of a for loop iterable (line 76)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 76, 8), range_call_result_422622)
        # Getting the type of the for loop variable (line 76)
        for_loop_var_422623 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 76, 8), range_call_result_422622)
        # Assigning a type to the variable 'i' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'i', for_loop_var_422623)
        # SSA begins for a for statement (line 76)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 77):
        
        # Assigning a Call to a Name (line 77):
        
        # Call to inv(...): (line 77)
        # Processing the call arguments (line 77)
        
        # Call to randn(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'n' (line 77)
        n_422630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 49), 'n', False)
        # Getting the type of 'n' (line 77)
        n_422631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 52), 'n', False)
        # Processing the call keyword arguments (line 77)
        kwargs_422632 = {}
        # Getting the type of 'np' (line 77)
        np_422627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 33), 'np', False)
        # Obtaining the member 'random' of a type (line 77)
        random_422628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 33), np_422627, 'random')
        # Obtaining the member 'randn' of a type (line 77)
        randn_422629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 33), random_422628, 'randn')
        # Calling randn(args, kwargs) (line 77)
        randn_call_result_422633 = invoke(stypy.reporting.localization.Localization(__file__, 77, 33), randn_422629, *[n_422630, n_422631], **kwargs_422632)
        
        # Processing the call keyword arguments (line 77)
        kwargs_422634 = {}
        # Getting the type of 'scipy' (line 77)
        scipy_422624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 77)
        linalg_422625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 16), scipy_422624, 'linalg')
        # Obtaining the member 'inv' of a type (line 77)
        inv_422626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 16), linalg_422625, 'inv')
        # Calling inv(args, kwargs) (line 77)
        inv_call_result_422635 = invoke(stypy.reporting.localization.Localization(__file__, 77, 16), inv_422626, *[randn_call_result_422633], **kwargs_422634)
        
        # Assigning a type to the variable 'A' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'A', inv_call_result_422635)
        
        # Assigning a Call to a Name (line 78):
        
        # Assigning a Call to a Name (line 78):
        
        # Call to randn(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'n' (line 78)
        n_422639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 32), 'n', False)
        # Processing the call keyword arguments (line 78)
        kwargs_422640 = {}
        # Getting the type of 'np' (line 78)
        np_422636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'np', False)
        # Obtaining the member 'random' of a type (line 78)
        random_422637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 16), np_422636, 'random')
        # Obtaining the member 'randn' of a type (line 78)
        randn_422638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 16), random_422637, 'randn')
        # Calling randn(args, kwargs) (line 78)
        randn_call_result_422641 = invoke(stypy.reporting.localization.Localization(__file__, 78, 16), randn_422638, *[n_422639], **kwargs_422640)
        
        # Assigning a type to the variable 'v' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'v', randn_call_result_422641)
        
        # Assigning a Call to a Name (line 79):
        
        # Assigning a Call to a Name (line 79):
        
        # Call to expm_multiply(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'A' (line 79)
        A_422643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 37), 'A', False)
        # Getting the type of 'v' (line 79)
        v_422644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 40), 'v', False)
        # Processing the call keyword arguments (line 79)
        kwargs_422645 = {}
        # Getting the type of 'expm_multiply' (line 79)
        expm_multiply_422642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 23), 'expm_multiply', False)
        # Calling expm_multiply(args, kwargs) (line 79)
        expm_multiply_call_result_422646 = invoke(stypy.reporting.localization.Localization(__file__, 79, 23), expm_multiply_422642, *[A_422643, v_422644], **kwargs_422645)
        
        # Assigning a type to the variable 'observed' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'observed', expm_multiply_call_result_422646)
        
        # Assigning a Call to a Name (line 80):
        
        # Assigning a Call to a Name (line 80):
        
        # Call to dot(...): (line 80)
        # Processing the call arguments (line 80)
        
        # Call to expm(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'A' (line 80)
        A_422652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 48), 'A', False)
        # Processing the call keyword arguments (line 80)
        kwargs_422653 = {}
        # Getting the type of 'scipy' (line 80)
        scipy_422649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 30), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 80)
        linalg_422650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 30), scipy_422649, 'linalg')
        # Obtaining the member 'expm' of a type (line 80)
        expm_422651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 30), linalg_422650, 'expm')
        # Calling expm(args, kwargs) (line 80)
        expm_call_result_422654 = invoke(stypy.reporting.localization.Localization(__file__, 80, 30), expm_422651, *[A_422652], **kwargs_422653)
        
        # Getting the type of 'v' (line 80)
        v_422655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 52), 'v', False)
        # Processing the call keyword arguments (line 80)
        kwargs_422656 = {}
        # Getting the type of 'np' (line 80)
        np_422647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 23), 'np', False)
        # Obtaining the member 'dot' of a type (line 80)
        dot_422648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 23), np_422647, 'dot')
        # Calling dot(args, kwargs) (line 80)
        dot_call_result_422657 = invoke(stypy.reporting.localization.Localization(__file__, 80, 23), dot_422648, *[expm_call_result_422654, v_422655], **kwargs_422656)
        
        # Assigning a type to the variable 'expected' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'expected', dot_call_result_422657)
        
        # Call to assert_allclose(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'observed' (line 81)
        observed_422659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 28), 'observed', False)
        # Getting the type of 'expected' (line 81)
        expected_422660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 38), 'expected', False)
        # Processing the call keyword arguments (line 81)
        kwargs_422661 = {}
        # Getting the type of 'assert_allclose' (line 81)
        assert_allclose_422658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 81)
        assert_allclose_call_result_422662 = invoke(stypy.reporting.localization.Localization(__file__, 81, 12), assert_allclose_422658, *[observed_422659, expected_422660], **kwargs_422661)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_matrix_vector_multiply(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_matrix_vector_multiply' in the type store
        # Getting the type of 'stypy_return_type' (line 72)
        stypy_return_type_422663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_422663)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_matrix_vector_multiply'
        return stypy_return_type_422663


    @norecursion
    def test_scaled_expm_multiply(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_scaled_expm_multiply'
        module_type_store = module_type_store.open_function_context('test_scaled_expm_multiply', 83, 4, False)
        # Assigning a type to the variable 'self' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpmActionSimple.test_scaled_expm_multiply.__dict__.__setitem__('stypy_localization', localization)
        TestExpmActionSimple.test_scaled_expm_multiply.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpmActionSimple.test_scaled_expm_multiply.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpmActionSimple.test_scaled_expm_multiply.__dict__.__setitem__('stypy_function_name', 'TestExpmActionSimple.test_scaled_expm_multiply')
        TestExpmActionSimple.test_scaled_expm_multiply.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpmActionSimple.test_scaled_expm_multiply.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpmActionSimple.test_scaled_expm_multiply.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpmActionSimple.test_scaled_expm_multiply.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpmActionSimple.test_scaled_expm_multiply.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpmActionSimple.test_scaled_expm_multiply.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpmActionSimple.test_scaled_expm_multiply.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpmActionSimple.test_scaled_expm_multiply', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_scaled_expm_multiply', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_scaled_expm_multiply(...)' code ##################

        
        # Call to seed(...): (line 84)
        # Processing the call arguments (line 84)
        int_422667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 23), 'int')
        # Processing the call keyword arguments (line 84)
        kwargs_422668 = {}
        # Getting the type of 'np' (line 84)
        np_422664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 84)
        random_422665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), np_422664, 'random')
        # Obtaining the member 'seed' of a type (line 84)
        seed_422666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), random_422665, 'seed')
        # Calling seed(args, kwargs) (line 84)
        seed_call_result_422669 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), seed_422666, *[int_422667], **kwargs_422668)
        
        
        # Assigning a Num to a Name (line 85):
        
        # Assigning a Num to a Name (line 85):
        int_422670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 12), 'int')
        # Assigning a type to the variable 'n' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'n', int_422670)
        
        # Assigning a Num to a Name (line 86):
        
        # Assigning a Num to a Name (line 86):
        int_422671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 12), 'int')
        # Assigning a type to the variable 'k' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'k', int_422671)
        
        # Assigning a Num to a Name (line 87):
        
        # Assigning a Num to a Name (line 87):
        int_422672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 19), 'int')
        # Assigning a type to the variable 'nsamples' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'nsamples', int_422672)
        
        
        # Call to range(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'nsamples' (line 88)
        nsamples_422674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 23), 'nsamples', False)
        # Processing the call keyword arguments (line 88)
        kwargs_422675 = {}
        # Getting the type of 'range' (line 88)
        range_422673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 17), 'range', False)
        # Calling range(args, kwargs) (line 88)
        range_call_result_422676 = invoke(stypy.reporting.localization.Localization(__file__, 88, 17), range_422673, *[nsamples_422674], **kwargs_422675)
        
        # Testing the type of a for loop iterable (line 88)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 88, 8), range_call_result_422676)
        # Getting the type of the for loop variable (line 88)
        for_loop_var_422677 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 88, 8), range_call_result_422676)
        # Assigning a type to the variable 'i' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'i', for_loop_var_422677)
        # SSA begins for a for statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 89)
        tuple_422678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 89)
        # Adding element type (line 89)
        float_422679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 22), tuple_422678, float_422679)
        # Adding element type (line 89)
        float_422680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 22), tuple_422678, float_422680)
        # Adding element type (line 89)
        float_422681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 22), tuple_422678, float_422681)
        
        # Testing the type of a for loop iterable (line 89)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 89, 12), tuple_422678)
        # Getting the type of the for loop variable (line 89)
        for_loop_var_422682 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 89, 12), tuple_422678)
        # Assigning a type to the variable 't' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 't', for_loop_var_422682)
        # SSA begins for a for statement (line 89)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to errstate(...): (line 90)
        # Processing the call keyword arguments (line 90)
        str_422685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 41), 'str', 'ignore')
        keyword_422686 = str_422685
        kwargs_422687 = {'invalid': keyword_422686}
        # Getting the type of 'np' (line 90)
        np_422683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 21), 'np', False)
        # Obtaining the member 'errstate' of a type (line 90)
        errstate_422684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 21), np_422683, 'errstate')
        # Calling errstate(args, kwargs) (line 90)
        errstate_call_result_422688 = invoke(stypy.reporting.localization.Localization(__file__, 90, 21), errstate_422684, *[], **kwargs_422687)
        
        with_422689 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 90, 21), errstate_call_result_422688, 'with parameter', '__enter__', '__exit__')

        if with_422689:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 90)
            enter___422690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 21), errstate_call_result_422688, '__enter__')
            with_enter_422691 = invoke(stypy.reporting.localization.Localization(__file__, 90, 21), enter___422690)
            
            # Assigning a Call to a Name (line 91):
            
            # Assigning a Call to a Name (line 91):
            
            # Call to inv(...): (line 91)
            # Processing the call arguments (line 91)
            
            # Call to randn(...): (line 91)
            # Processing the call arguments (line 91)
            # Getting the type of 'n' (line 91)
            n_422698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 57), 'n', False)
            # Getting the type of 'n' (line 91)
            n_422699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 60), 'n', False)
            # Processing the call keyword arguments (line 91)
            kwargs_422700 = {}
            # Getting the type of 'np' (line 91)
            np_422695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 41), 'np', False)
            # Obtaining the member 'random' of a type (line 91)
            random_422696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 41), np_422695, 'random')
            # Obtaining the member 'randn' of a type (line 91)
            randn_422697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 41), random_422696, 'randn')
            # Calling randn(args, kwargs) (line 91)
            randn_call_result_422701 = invoke(stypy.reporting.localization.Localization(__file__, 91, 41), randn_422697, *[n_422698, n_422699], **kwargs_422700)
            
            # Processing the call keyword arguments (line 91)
            kwargs_422702 = {}
            # Getting the type of 'scipy' (line 91)
            scipy_422692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 24), 'scipy', False)
            # Obtaining the member 'linalg' of a type (line 91)
            linalg_422693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 24), scipy_422692, 'linalg')
            # Obtaining the member 'inv' of a type (line 91)
            inv_422694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 24), linalg_422693, 'inv')
            # Calling inv(args, kwargs) (line 91)
            inv_call_result_422703 = invoke(stypy.reporting.localization.Localization(__file__, 91, 24), inv_422694, *[randn_call_result_422701], **kwargs_422702)
            
            # Assigning a type to the variable 'A' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'A', inv_call_result_422703)
            
            # Assigning a Call to a Name (line 92):
            
            # Assigning a Call to a Name (line 92):
            
            # Call to randn(...): (line 92)
            # Processing the call arguments (line 92)
            # Getting the type of 'n' (line 92)
            n_422707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 40), 'n', False)
            # Getting the type of 'k' (line 92)
            k_422708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 43), 'k', False)
            # Processing the call keyword arguments (line 92)
            kwargs_422709 = {}
            # Getting the type of 'np' (line 92)
            np_422704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'np', False)
            # Obtaining the member 'random' of a type (line 92)
            random_422705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 24), np_422704, 'random')
            # Obtaining the member 'randn' of a type (line 92)
            randn_422706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 24), random_422705, 'randn')
            # Calling randn(args, kwargs) (line 92)
            randn_call_result_422710 = invoke(stypy.reporting.localization.Localization(__file__, 92, 24), randn_422706, *[n_422707, k_422708], **kwargs_422709)
            
            # Assigning a type to the variable 'B' (line 92)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 20), 'B', randn_call_result_422710)
            
            # Assigning a Call to a Name (line 93):
            
            # Assigning a Call to a Name (line 93):
            
            # Call to _expm_multiply_simple(...): (line 93)
            # Processing the call arguments (line 93)
            # Getting the type of 'A' (line 93)
            A_422712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 53), 'A', False)
            # Getting the type of 'B' (line 93)
            B_422713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 56), 'B', False)
            # Processing the call keyword arguments (line 93)
            # Getting the type of 't' (line 93)
            t_422714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 61), 't', False)
            keyword_422715 = t_422714
            kwargs_422716 = {'t': keyword_422715}
            # Getting the type of '_expm_multiply_simple' (line 93)
            _expm_multiply_simple_422711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 31), '_expm_multiply_simple', False)
            # Calling _expm_multiply_simple(args, kwargs) (line 93)
            _expm_multiply_simple_call_result_422717 = invoke(stypy.reporting.localization.Localization(__file__, 93, 31), _expm_multiply_simple_422711, *[A_422712, B_422713], **kwargs_422716)
            
            # Assigning a type to the variable 'observed' (line 93)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 20), 'observed', _expm_multiply_simple_call_result_422717)
            
            # Assigning a Call to a Name (line 94):
            
            # Assigning a Call to a Name (line 94):
            
            # Call to dot(...): (line 94)
            # Processing the call arguments (line 94)
            
            # Call to expm(...): (line 94)
            # Processing the call arguments (line 94)
            # Getting the type of 't' (line 94)
            t_422723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 56), 't', False)
            # Getting the type of 'A' (line 94)
            A_422724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 58), 'A', False)
            # Applying the binary operator '*' (line 94)
            result_mul_422725 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 56), '*', t_422723, A_422724)
            
            # Processing the call keyword arguments (line 94)
            kwargs_422726 = {}
            # Getting the type of 'scipy' (line 94)
            scipy_422720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 38), 'scipy', False)
            # Obtaining the member 'linalg' of a type (line 94)
            linalg_422721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 38), scipy_422720, 'linalg')
            # Obtaining the member 'expm' of a type (line 94)
            expm_422722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 38), linalg_422721, 'expm')
            # Calling expm(args, kwargs) (line 94)
            expm_call_result_422727 = invoke(stypy.reporting.localization.Localization(__file__, 94, 38), expm_422722, *[result_mul_422725], **kwargs_422726)
            
            # Getting the type of 'B' (line 94)
            B_422728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 62), 'B', False)
            # Processing the call keyword arguments (line 94)
            kwargs_422729 = {}
            # Getting the type of 'np' (line 94)
            np_422718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 31), 'np', False)
            # Obtaining the member 'dot' of a type (line 94)
            dot_422719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 31), np_422718, 'dot')
            # Calling dot(args, kwargs) (line 94)
            dot_call_result_422730 = invoke(stypy.reporting.localization.Localization(__file__, 94, 31), dot_422719, *[expm_call_result_422727, B_422728], **kwargs_422729)
            
            # Assigning a type to the variable 'expected' (line 94)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'expected', dot_call_result_422730)
            
            # Call to assert_allclose(...): (line 95)
            # Processing the call arguments (line 95)
            # Getting the type of 'observed' (line 95)
            observed_422732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 36), 'observed', False)
            # Getting the type of 'expected' (line 95)
            expected_422733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 46), 'expected', False)
            # Processing the call keyword arguments (line 95)
            kwargs_422734 = {}
            # Getting the type of 'assert_allclose' (line 95)
            assert_allclose_422731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 20), 'assert_allclose', False)
            # Calling assert_allclose(args, kwargs) (line 95)
            assert_allclose_call_result_422735 = invoke(stypy.reporting.localization.Localization(__file__, 95, 20), assert_allclose_422731, *[observed_422732, expected_422733], **kwargs_422734)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 90)
            exit___422736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 21), errstate_call_result_422688, '__exit__')
            with_exit_422737 = invoke(stypy.reporting.localization.Localization(__file__, 90, 21), exit___422736, None, None, None)

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_scaled_expm_multiply(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_scaled_expm_multiply' in the type store
        # Getting the type of 'stypy_return_type' (line 83)
        stypy_return_type_422738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_422738)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_scaled_expm_multiply'
        return stypy_return_type_422738


    @norecursion
    def test_scaled_expm_multiply_single_timepoint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_scaled_expm_multiply_single_timepoint'
        module_type_store = module_type_store.open_function_context('test_scaled_expm_multiply_single_timepoint', 97, 4, False)
        # Assigning a type to the variable 'self' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpmActionSimple.test_scaled_expm_multiply_single_timepoint.__dict__.__setitem__('stypy_localization', localization)
        TestExpmActionSimple.test_scaled_expm_multiply_single_timepoint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpmActionSimple.test_scaled_expm_multiply_single_timepoint.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpmActionSimple.test_scaled_expm_multiply_single_timepoint.__dict__.__setitem__('stypy_function_name', 'TestExpmActionSimple.test_scaled_expm_multiply_single_timepoint')
        TestExpmActionSimple.test_scaled_expm_multiply_single_timepoint.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpmActionSimple.test_scaled_expm_multiply_single_timepoint.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpmActionSimple.test_scaled_expm_multiply_single_timepoint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpmActionSimple.test_scaled_expm_multiply_single_timepoint.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpmActionSimple.test_scaled_expm_multiply_single_timepoint.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpmActionSimple.test_scaled_expm_multiply_single_timepoint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpmActionSimple.test_scaled_expm_multiply_single_timepoint.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpmActionSimple.test_scaled_expm_multiply_single_timepoint', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_scaled_expm_multiply_single_timepoint', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_scaled_expm_multiply_single_timepoint(...)' code ##################

        
        # Call to seed(...): (line 98)
        # Processing the call arguments (line 98)
        int_422742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 23), 'int')
        # Processing the call keyword arguments (line 98)
        kwargs_422743 = {}
        # Getting the type of 'np' (line 98)
        np_422739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 98)
        random_422740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), np_422739, 'random')
        # Obtaining the member 'seed' of a type (line 98)
        seed_422741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), random_422740, 'seed')
        # Calling seed(args, kwargs) (line 98)
        seed_call_result_422744 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), seed_422741, *[int_422742], **kwargs_422743)
        
        
        # Assigning a Num to a Name (line 99):
        
        # Assigning a Num to a Name (line 99):
        float_422745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 12), 'float')
        # Assigning a type to the variable 't' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 't', float_422745)
        
        # Assigning a Num to a Name (line 100):
        
        # Assigning a Num to a Name (line 100):
        int_422746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 12), 'int')
        # Assigning a type to the variable 'n' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'n', int_422746)
        
        # Assigning a Num to a Name (line 101):
        
        # Assigning a Num to a Name (line 101):
        int_422747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 12), 'int')
        # Assigning a type to the variable 'k' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'k', int_422747)
        
        # Assigning a Call to a Name (line 102):
        
        # Assigning a Call to a Name (line 102):
        
        # Call to randn(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'n' (line 102)
        n_422751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 28), 'n', False)
        # Getting the type of 'n' (line 102)
        n_422752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 31), 'n', False)
        # Processing the call keyword arguments (line 102)
        kwargs_422753 = {}
        # Getting the type of 'np' (line 102)
        np_422748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 102)
        random_422749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), np_422748, 'random')
        # Obtaining the member 'randn' of a type (line 102)
        randn_422750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), random_422749, 'randn')
        # Calling randn(args, kwargs) (line 102)
        randn_call_result_422754 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), randn_422750, *[n_422751, n_422752], **kwargs_422753)
        
        # Assigning a type to the variable 'A' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'A', randn_call_result_422754)
        
        # Assigning a Call to a Name (line 103):
        
        # Assigning a Call to a Name (line 103):
        
        # Call to randn(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'n' (line 103)
        n_422758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 28), 'n', False)
        # Getting the type of 'k' (line 103)
        k_422759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 31), 'k', False)
        # Processing the call keyword arguments (line 103)
        kwargs_422760 = {}
        # Getting the type of 'np' (line 103)
        np_422755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 103)
        random_422756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), np_422755, 'random')
        # Obtaining the member 'randn' of a type (line 103)
        randn_422757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), random_422756, 'randn')
        # Calling randn(args, kwargs) (line 103)
        randn_call_result_422761 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), randn_422757, *[n_422758, k_422759], **kwargs_422760)
        
        # Assigning a type to the variable 'B' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'B', randn_call_result_422761)
        
        # Assigning a Call to a Name (line 104):
        
        # Assigning a Call to a Name (line 104):
        
        # Call to _expm_multiply_simple(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'A' (line 104)
        A_422763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 41), 'A', False)
        # Getting the type of 'B' (line 104)
        B_422764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 44), 'B', False)
        # Processing the call keyword arguments (line 104)
        # Getting the type of 't' (line 104)
        t_422765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 49), 't', False)
        keyword_422766 = t_422765
        kwargs_422767 = {'t': keyword_422766}
        # Getting the type of '_expm_multiply_simple' (line 104)
        _expm_multiply_simple_422762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 19), '_expm_multiply_simple', False)
        # Calling _expm_multiply_simple(args, kwargs) (line 104)
        _expm_multiply_simple_call_result_422768 = invoke(stypy.reporting.localization.Localization(__file__, 104, 19), _expm_multiply_simple_422762, *[A_422763, B_422764], **kwargs_422767)
        
        # Assigning a type to the variable 'observed' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'observed', _expm_multiply_simple_call_result_422768)
        
        # Assigning a Call to a Name (line 105):
        
        # Assigning a Call to a Name (line 105):
        
        # Call to dot(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'B' (line 105)
        B_422778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 46), 'B', False)
        # Processing the call keyword arguments (line 105)
        kwargs_422779 = {}
        
        # Call to expm(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 't' (line 105)
        t_422772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 37), 't', False)
        # Getting the type of 'A' (line 105)
        A_422773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 39), 'A', False)
        # Applying the binary operator '*' (line 105)
        result_mul_422774 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 37), '*', t_422772, A_422773)
        
        # Processing the call keyword arguments (line 105)
        kwargs_422775 = {}
        # Getting the type of 'scipy' (line 105)
        scipy_422769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 19), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 105)
        linalg_422770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 19), scipy_422769, 'linalg')
        # Obtaining the member 'expm' of a type (line 105)
        expm_422771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 19), linalg_422770, 'expm')
        # Calling expm(args, kwargs) (line 105)
        expm_call_result_422776 = invoke(stypy.reporting.localization.Localization(__file__, 105, 19), expm_422771, *[result_mul_422774], **kwargs_422775)
        
        # Obtaining the member 'dot' of a type (line 105)
        dot_422777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 19), expm_call_result_422776, 'dot')
        # Calling dot(args, kwargs) (line 105)
        dot_call_result_422780 = invoke(stypy.reporting.localization.Localization(__file__, 105, 19), dot_422777, *[B_422778], **kwargs_422779)
        
        # Assigning a type to the variable 'expected' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'expected', dot_call_result_422780)
        
        # Call to assert_allclose(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'observed' (line 106)
        observed_422782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 24), 'observed', False)
        # Getting the type of 'expected' (line 106)
        expected_422783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 34), 'expected', False)
        # Processing the call keyword arguments (line 106)
        kwargs_422784 = {}
        # Getting the type of 'assert_allclose' (line 106)
        assert_allclose_422781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 106)
        assert_allclose_call_result_422785 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), assert_allclose_422781, *[observed_422782, expected_422783], **kwargs_422784)
        
        
        # ################# End of 'test_scaled_expm_multiply_single_timepoint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_scaled_expm_multiply_single_timepoint' in the type store
        # Getting the type of 'stypy_return_type' (line 97)
        stypy_return_type_422786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_422786)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_scaled_expm_multiply_single_timepoint'
        return stypy_return_type_422786


    @norecursion
    def test_sparse_expm_multiply(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sparse_expm_multiply'
        module_type_store = module_type_store.open_function_context('test_sparse_expm_multiply', 108, 4, False)
        # Assigning a type to the variable 'self' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpmActionSimple.test_sparse_expm_multiply.__dict__.__setitem__('stypy_localization', localization)
        TestExpmActionSimple.test_sparse_expm_multiply.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpmActionSimple.test_sparse_expm_multiply.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpmActionSimple.test_sparse_expm_multiply.__dict__.__setitem__('stypy_function_name', 'TestExpmActionSimple.test_sparse_expm_multiply')
        TestExpmActionSimple.test_sparse_expm_multiply.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpmActionSimple.test_sparse_expm_multiply.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpmActionSimple.test_sparse_expm_multiply.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpmActionSimple.test_sparse_expm_multiply.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpmActionSimple.test_sparse_expm_multiply.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpmActionSimple.test_sparse_expm_multiply.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpmActionSimple.test_sparse_expm_multiply.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpmActionSimple.test_sparse_expm_multiply', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sparse_expm_multiply', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sparse_expm_multiply(...)' code ##################

        
        # Call to seed(...): (line 109)
        # Processing the call arguments (line 109)
        int_422790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 23), 'int')
        # Processing the call keyword arguments (line 109)
        kwargs_422791 = {}
        # Getting the type of 'np' (line 109)
        np_422787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 109)
        random_422788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), np_422787, 'random')
        # Obtaining the member 'seed' of a type (line 109)
        seed_422789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), random_422788, 'seed')
        # Calling seed(args, kwargs) (line 109)
        seed_call_result_422792 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), seed_422789, *[int_422790], **kwargs_422791)
        
        
        # Assigning a Num to a Name (line 110):
        
        # Assigning a Num to a Name (line 110):
        int_422793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 12), 'int')
        # Assigning a type to the variable 'n' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'n', int_422793)
        
        # Assigning a Num to a Name (line 111):
        
        # Assigning a Num to a Name (line 111):
        int_422794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 12), 'int')
        # Assigning a type to the variable 'k' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'k', int_422794)
        
        # Assigning a Num to a Name (line 112):
        
        # Assigning a Num to a Name (line 112):
        int_422795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 19), 'int')
        # Assigning a type to the variable 'nsamples' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'nsamples', int_422795)
        
        
        # Call to range(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'nsamples' (line 113)
        nsamples_422797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 23), 'nsamples', False)
        # Processing the call keyword arguments (line 113)
        kwargs_422798 = {}
        # Getting the type of 'range' (line 113)
        range_422796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 17), 'range', False)
        # Calling range(args, kwargs) (line 113)
        range_call_result_422799 = invoke(stypy.reporting.localization.Localization(__file__, 113, 17), range_422796, *[nsamples_422797], **kwargs_422798)
        
        # Testing the type of a for loop iterable (line 113)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 113, 8), range_call_result_422799)
        # Getting the type of the for loop variable (line 113)
        for_loop_var_422800 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 113, 8), range_call_result_422799)
        # Assigning a type to the variable 'i' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'i', for_loop_var_422800)
        # SSA begins for a for statement (line 113)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 114):
        
        # Assigning a Call to a Name (line 114):
        
        # Call to rand(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'n' (line 114)
        n_422804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 34), 'n', False)
        # Getting the type of 'n' (line 114)
        n_422805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 37), 'n', False)
        # Processing the call keyword arguments (line 114)
        float_422806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 48), 'float')
        keyword_422807 = float_422806
        kwargs_422808 = {'density': keyword_422807}
        # Getting the type of 'scipy' (line 114)
        scipy_422801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 114)
        sparse_422802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 16), scipy_422801, 'sparse')
        # Obtaining the member 'rand' of a type (line 114)
        rand_422803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 16), sparse_422802, 'rand')
        # Calling rand(args, kwargs) (line 114)
        rand_call_result_422809 = invoke(stypy.reporting.localization.Localization(__file__, 114, 16), rand_422803, *[n_422804, n_422805], **kwargs_422808)
        
        # Assigning a type to the variable 'A' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'A', rand_call_result_422809)
        
        # Assigning a Call to a Name (line 115):
        
        # Assigning a Call to a Name (line 115):
        
        # Call to randn(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'n' (line 115)
        n_422813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 32), 'n', False)
        # Getting the type of 'k' (line 115)
        k_422814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 35), 'k', False)
        # Processing the call keyword arguments (line 115)
        kwargs_422815 = {}
        # Getting the type of 'np' (line 115)
        np_422810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'np', False)
        # Obtaining the member 'random' of a type (line 115)
        random_422811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 16), np_422810, 'random')
        # Obtaining the member 'randn' of a type (line 115)
        randn_422812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 16), random_422811, 'randn')
        # Calling randn(args, kwargs) (line 115)
        randn_call_result_422816 = invoke(stypy.reporting.localization.Localization(__file__, 115, 16), randn_422812, *[n_422813, k_422814], **kwargs_422815)
        
        # Assigning a type to the variable 'B' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'B', randn_call_result_422816)
        
        # Assigning a Call to a Name (line 116):
        
        # Assigning a Call to a Name (line 116):
        
        # Call to expm_multiply(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'A' (line 116)
        A_422818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 37), 'A', False)
        # Getting the type of 'B' (line 116)
        B_422819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 40), 'B', False)
        # Processing the call keyword arguments (line 116)
        kwargs_422820 = {}
        # Getting the type of 'expm_multiply' (line 116)
        expm_multiply_422817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 23), 'expm_multiply', False)
        # Calling expm_multiply(args, kwargs) (line 116)
        expm_multiply_call_result_422821 = invoke(stypy.reporting.localization.Localization(__file__, 116, 23), expm_multiply_422817, *[A_422818, B_422819], **kwargs_422820)
        
        # Assigning a type to the variable 'observed' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'observed', expm_multiply_call_result_422821)
        
        # Call to suppress_warnings(...): (line 117)
        # Processing the call keyword arguments (line 117)
        kwargs_422823 = {}
        # Getting the type of 'suppress_warnings' (line 117)
        suppress_warnings_422822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 17), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 117)
        suppress_warnings_call_result_422824 = invoke(stypy.reporting.localization.Localization(__file__, 117, 17), suppress_warnings_422822, *[], **kwargs_422823)
        
        with_422825 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 117, 17), suppress_warnings_call_result_422824, 'with parameter', '__enter__', '__exit__')

        if with_422825:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 117)
            enter___422826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 17), suppress_warnings_call_result_422824, '__enter__')
            with_enter_422827 = invoke(stypy.reporting.localization.Localization(__file__, 117, 17), enter___422826)
            # Assigning a type to the variable 'sup' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 17), 'sup', with_enter_422827)
            
            # Call to filter(...): (line 118)
            # Processing the call arguments (line 118)
            # Getting the type of 'SparseEfficiencyWarning' (line 118)
            SparseEfficiencyWarning_422830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 27), 'SparseEfficiencyWarning', False)
            str_422831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 27), 'str', 'splu requires CSC matrix format')
            # Processing the call keyword arguments (line 118)
            kwargs_422832 = {}
            # Getting the type of 'sup' (line 118)
            sup_422828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'sup', False)
            # Obtaining the member 'filter' of a type (line 118)
            filter_422829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 16), sup_422828, 'filter')
            # Calling filter(args, kwargs) (line 118)
            filter_call_result_422833 = invoke(stypy.reporting.localization.Localization(__file__, 118, 16), filter_422829, *[SparseEfficiencyWarning_422830, str_422831], **kwargs_422832)
            
            
            # Call to filter(...): (line 120)
            # Processing the call arguments (line 120)
            # Getting the type of 'SparseEfficiencyWarning' (line 120)
            SparseEfficiencyWarning_422836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 27), 'SparseEfficiencyWarning', False)
            str_422837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 27), 'str', 'spsolve is more efficient when sparse b is in the CSC matrix format')
            # Processing the call keyword arguments (line 120)
            kwargs_422838 = {}
            # Getting the type of 'sup' (line 120)
            sup_422834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'sup', False)
            # Obtaining the member 'filter' of a type (line 120)
            filter_422835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 16), sup_422834, 'filter')
            # Calling filter(args, kwargs) (line 120)
            filter_call_result_422839 = invoke(stypy.reporting.localization.Localization(__file__, 120, 16), filter_422835, *[SparseEfficiencyWarning_422836, str_422837], **kwargs_422838)
            
            
            # Assigning a Call to a Name (line 122):
            
            # Assigning a Call to a Name (line 122):
            
            # Call to dot(...): (line 122)
            # Processing the call arguments (line 122)
            # Getting the type of 'B' (line 122)
            B_422847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 52), 'B', False)
            # Processing the call keyword arguments (line 122)
            kwargs_422848 = {}
            
            # Call to expm(...): (line 122)
            # Processing the call arguments (line 122)
            # Getting the type of 'A' (line 122)
            A_422843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 45), 'A', False)
            # Processing the call keyword arguments (line 122)
            kwargs_422844 = {}
            # Getting the type of 'scipy' (line 122)
            scipy_422840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 27), 'scipy', False)
            # Obtaining the member 'linalg' of a type (line 122)
            linalg_422841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 27), scipy_422840, 'linalg')
            # Obtaining the member 'expm' of a type (line 122)
            expm_422842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 27), linalg_422841, 'expm')
            # Calling expm(args, kwargs) (line 122)
            expm_call_result_422845 = invoke(stypy.reporting.localization.Localization(__file__, 122, 27), expm_422842, *[A_422843], **kwargs_422844)
            
            # Obtaining the member 'dot' of a type (line 122)
            dot_422846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 27), expm_call_result_422845, 'dot')
            # Calling dot(args, kwargs) (line 122)
            dot_call_result_422849 = invoke(stypy.reporting.localization.Localization(__file__, 122, 27), dot_422846, *[B_422847], **kwargs_422848)
            
            # Assigning a type to the variable 'expected' (line 122)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'expected', dot_call_result_422849)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 117)
            exit___422850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 17), suppress_warnings_call_result_422824, '__exit__')
            with_exit_422851 = invoke(stypy.reporting.localization.Localization(__file__, 117, 17), exit___422850, None, None, None)

        
        # Call to assert_allclose(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'observed' (line 123)
        observed_422853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 28), 'observed', False)
        # Getting the type of 'expected' (line 123)
        expected_422854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 38), 'expected', False)
        # Processing the call keyword arguments (line 123)
        kwargs_422855 = {}
        # Getting the type of 'assert_allclose' (line 123)
        assert_allclose_422852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 123)
        assert_allclose_call_result_422856 = invoke(stypy.reporting.localization.Localization(__file__, 123, 12), assert_allclose_422852, *[observed_422853, expected_422854], **kwargs_422855)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_sparse_expm_multiply(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sparse_expm_multiply' in the type store
        # Getting the type of 'stypy_return_type' (line 108)
        stypy_return_type_422857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_422857)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sparse_expm_multiply'
        return stypy_return_type_422857


    @norecursion
    def test_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_complex'
        module_type_store = module_type_store.open_function_context('test_complex', 125, 4, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpmActionSimple.test_complex.__dict__.__setitem__('stypy_localization', localization)
        TestExpmActionSimple.test_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpmActionSimple.test_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpmActionSimple.test_complex.__dict__.__setitem__('stypy_function_name', 'TestExpmActionSimple.test_complex')
        TestExpmActionSimple.test_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpmActionSimple.test_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpmActionSimple.test_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpmActionSimple.test_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpmActionSimple.test_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpmActionSimple.test_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpmActionSimple.test_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpmActionSimple.test_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_complex(...)' code ##################

        
        # Assigning a Call to a Name (line 126):
        
        # Assigning a Call to a Name (line 126):
        
        # Call to array(...): (line 126)
        # Processing the call arguments (line 126)
        
        # Obtaining an instance of the builtin type 'list' (line 126)
        list_422860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 126)
        # Adding element type (line 126)
        
        # Obtaining an instance of the builtin type 'list' (line 127)
        list_422861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 127)
        # Adding element type (line 127)
        complex_422862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 13), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 12), list_422861, complex_422862)
        # Adding element type (line 127)
        complex_422863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 17), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 12), list_422861, complex_422863)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 21), list_422860, list_422861)
        # Adding element type (line 126)
        
        # Obtaining an instance of the builtin type 'list' (line 128)
        list_422864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 128)
        # Adding element type (line 128)
        int_422865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 12), list_422864, int_422865)
        # Adding element type (line 128)
        complex_422866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 16), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 12), list_422864, complex_422866)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 21), list_422860, list_422864)
        
        # Processing the call keyword arguments (line 126)
        # Getting the type of 'complex' (line 128)
        complex_422867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 28), 'complex', False)
        keyword_422868 = complex_422867
        kwargs_422869 = {'dtype': keyword_422868}
        # Getting the type of 'np' (line 126)
        np_422858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 126)
        array_422859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 12), np_422858, 'array')
        # Calling array(args, kwargs) (line 126)
        array_call_result_422870 = invoke(stypy.reporting.localization.Localization(__file__, 126, 12), array_422859, *[list_422860], **kwargs_422869)
        
        # Assigning a type to the variable 'A' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'A', array_call_result_422870)
        
        # Assigning a Call to a Name (line 129):
        
        # Assigning a Call to a Name (line 129):
        
        # Call to array(...): (line 129)
        # Processing the call arguments (line 129)
        
        # Obtaining an instance of the builtin type 'list' (line 129)
        list_422873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 129)
        # Adding element type (line 129)
        complex_422874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 22), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 21), list_422873, complex_422874)
        # Adding element type (line 129)
        complex_422875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 26), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 21), list_422873, complex_422875)
        
        # Processing the call keyword arguments (line 129)
        kwargs_422876 = {}
        # Getting the type of 'np' (line 129)
        np_422871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 129)
        array_422872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 12), np_422871, 'array')
        # Calling array(args, kwargs) (line 129)
        array_call_result_422877 = invoke(stypy.reporting.localization.Localization(__file__, 129, 12), array_422872, *[list_422873], **kwargs_422876)
        
        # Assigning a type to the variable 'B' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'B', array_call_result_422877)
        
        # Assigning a Call to a Name (line 130):
        
        # Assigning a Call to a Name (line 130):
        
        # Call to expm_multiply(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'A' (line 130)
        A_422879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 33), 'A', False)
        # Getting the type of 'B' (line 130)
        B_422880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 36), 'B', False)
        # Processing the call keyword arguments (line 130)
        kwargs_422881 = {}
        # Getting the type of 'expm_multiply' (line 130)
        expm_multiply_422878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 19), 'expm_multiply', False)
        # Calling expm_multiply(args, kwargs) (line 130)
        expm_multiply_call_result_422882 = invoke(stypy.reporting.localization.Localization(__file__, 130, 19), expm_multiply_422878, *[A_422879, B_422880], **kwargs_422881)
        
        # Assigning a type to the variable 'observed' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'observed', expm_multiply_call_result_422882)
        
        # Assigning a Call to a Name (line 131):
        
        # Assigning a Call to a Name (line 131):
        
        # Call to array(...): (line 131)
        # Processing the call arguments (line 131)
        
        # Obtaining an instance of the builtin type 'list' (line 131)
        list_422885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 131)
        # Adding element type (line 131)
        complex_422886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 12), 'complex')
        
        # Call to exp(...): (line 132)
        # Processing the call arguments (line 132)
        complex_422889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 24), 'complex')
        # Processing the call keyword arguments (line 132)
        kwargs_422890 = {}
        # Getting the type of 'np' (line 132)
        np_422887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 17), 'np', False)
        # Obtaining the member 'exp' of a type (line 132)
        exp_422888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 17), np_422887, 'exp')
        # Calling exp(args, kwargs) (line 132)
        exp_call_result_422891 = invoke(stypy.reporting.localization.Localization(__file__, 132, 17), exp_422888, *[complex_422889], **kwargs_422890)
        
        # Applying the binary operator '*' (line 132)
        result_mul_422892 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 12), '*', complex_422886, exp_call_result_422891)
        
        complex_422893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 30), 'complex')
        complex_422894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 36), 'complex')
        
        # Call to cos(...): (line 132)
        # Processing the call arguments (line 132)
        int_422897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 46), 'int')
        # Processing the call keyword arguments (line 132)
        kwargs_422898 = {}
        # Getting the type of 'np' (line 132)
        np_422895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 39), 'np', False)
        # Obtaining the member 'cos' of a type (line 132)
        cos_422896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 39), np_422895, 'cos')
        # Calling cos(args, kwargs) (line 132)
        cos_call_result_422899 = invoke(stypy.reporting.localization.Localization(__file__, 132, 39), cos_422896, *[int_422897], **kwargs_422898)
        
        # Applying the binary operator '*' (line 132)
        result_mul_422900 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 36), '*', complex_422894, cos_call_result_422899)
        
        
        # Call to sin(...): (line 132)
        # Processing the call arguments (line 132)
        int_422903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 58), 'int')
        # Processing the call keyword arguments (line 132)
        kwargs_422904 = {}
        # Getting the type of 'np' (line 132)
        np_422901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 51), 'np', False)
        # Obtaining the member 'sin' of a type (line 132)
        sin_422902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 51), np_422901, 'sin')
        # Calling sin(args, kwargs) (line 132)
        sin_call_result_422905 = invoke(stypy.reporting.localization.Localization(__file__, 132, 51), sin_422902, *[int_422903], **kwargs_422904)
        
        # Applying the binary operator '-' (line 132)
        result_sub_422906 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 36), '-', result_mul_422900, sin_call_result_422905)
        
        # Applying the binary operator '*' (line 132)
        result_mul_422907 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 30), '*', complex_422893, result_sub_422906)
        
        # Applying the binary operator '+' (line 132)
        result_add_422908 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 12), '+', result_mul_422892, result_mul_422907)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 28), list_422885, result_add_422908)
        # Adding element type (line 131)
        complex_422909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 12), 'complex')
        
        # Call to exp(...): (line 133)
        # Processing the call arguments (line 133)
        complex_422912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 24), 'complex')
        # Processing the call keyword arguments (line 133)
        kwargs_422913 = {}
        # Getting the type of 'np' (line 133)
        np_422910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 17), 'np', False)
        # Obtaining the member 'exp' of a type (line 133)
        exp_422911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 17), np_422910, 'exp')
        # Calling exp(args, kwargs) (line 133)
        exp_call_result_422914 = invoke(stypy.reporting.localization.Localization(__file__, 133, 17), exp_422911, *[complex_422912], **kwargs_422913)
        
        # Applying the binary operator '*' (line 133)
        result_mul_422915 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 12), '*', complex_422909, exp_call_result_422914)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 28), list_422885, result_mul_422915)
        
        # Processing the call keyword arguments (line 131)
        # Getting the type of 'complex' (line 133)
        complex_422916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 36), 'complex', False)
        keyword_422917 = complex_422916
        kwargs_422918 = {'dtype': keyword_422917}
        # Getting the type of 'np' (line 131)
        np_422883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 131)
        array_422884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 19), np_422883, 'array')
        # Calling array(args, kwargs) (line 131)
        array_call_result_422919 = invoke(stypy.reporting.localization.Localization(__file__, 131, 19), array_422884, *[list_422885], **kwargs_422918)
        
        # Assigning a type to the variable 'expected' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'expected', array_call_result_422919)
        
        # Call to assert_allclose(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'observed' (line 134)
        observed_422921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 24), 'observed', False)
        # Getting the type of 'expected' (line 134)
        expected_422922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 34), 'expected', False)
        # Processing the call keyword arguments (line 134)
        kwargs_422923 = {}
        # Getting the type of 'assert_allclose' (line 134)
        assert_allclose_422920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 134)
        assert_allclose_call_result_422924 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), assert_allclose_422920, *[observed_422921, expected_422922], **kwargs_422923)
        
        
        # ################# End of 'test_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 125)
        stypy_return_type_422925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_422925)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_complex'
        return stypy_return_type_422925


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpmActionSimple.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestExpmActionSimple' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'TestExpmActionSimple', TestExpmActionSimple)
# Declaration of the 'TestExpmActionInterval' class

class TestExpmActionInterval(object, ):

    @norecursion
    def test_sparse_expm_multiply_interval(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sparse_expm_multiply_interval'
        module_type_store = module_type_store.open_function_context('test_sparse_expm_multiply_interval', 139, 4, False)
        # Assigning a type to the variable 'self' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpmActionInterval.test_sparse_expm_multiply_interval.__dict__.__setitem__('stypy_localization', localization)
        TestExpmActionInterval.test_sparse_expm_multiply_interval.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpmActionInterval.test_sparse_expm_multiply_interval.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpmActionInterval.test_sparse_expm_multiply_interval.__dict__.__setitem__('stypy_function_name', 'TestExpmActionInterval.test_sparse_expm_multiply_interval')
        TestExpmActionInterval.test_sparse_expm_multiply_interval.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpmActionInterval.test_sparse_expm_multiply_interval.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpmActionInterval.test_sparse_expm_multiply_interval.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpmActionInterval.test_sparse_expm_multiply_interval.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpmActionInterval.test_sparse_expm_multiply_interval.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpmActionInterval.test_sparse_expm_multiply_interval.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpmActionInterval.test_sparse_expm_multiply_interval.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpmActionInterval.test_sparse_expm_multiply_interval', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sparse_expm_multiply_interval', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sparse_expm_multiply_interval(...)' code ##################

        
        # Call to seed(...): (line 140)
        # Processing the call arguments (line 140)
        int_422929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 23), 'int')
        # Processing the call keyword arguments (line 140)
        kwargs_422930 = {}
        # Getting the type of 'np' (line 140)
        np_422926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 140)
        random_422927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), np_422926, 'random')
        # Obtaining the member 'seed' of a type (line 140)
        seed_422928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), random_422927, 'seed')
        # Calling seed(args, kwargs) (line 140)
        seed_call_result_422931 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), seed_422928, *[int_422929], **kwargs_422930)
        
        
        # Assigning a Num to a Name (line 141):
        
        # Assigning a Num to a Name (line 141):
        float_422932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 16), 'float')
        # Assigning a type to the variable 'start' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'start', float_422932)
        
        # Assigning a Num to a Name (line 142):
        
        # Assigning a Num to a Name (line 142):
        float_422933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 15), 'float')
        # Assigning a type to the variable 'stop' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'stop', float_422933)
        
        # Assigning a Num to a Name (line 143):
        
        # Assigning a Num to a Name (line 143):
        int_422934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 12), 'int')
        # Assigning a type to the variable 'n' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'n', int_422934)
        
        # Assigning a Num to a Name (line 144):
        
        # Assigning a Num to a Name (line 144):
        int_422935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 12), 'int')
        # Assigning a type to the variable 'k' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'k', int_422935)
        
        # Assigning a Name to a Name (line 145):
        
        # Assigning a Name to a Name (line 145):
        # Getting the type of 'True' (line 145)
        True_422936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 19), 'True')
        # Assigning a type to the variable 'endpoint' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'endpoint', True_422936)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 146)
        tuple_422937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 146)
        # Adding element type (line 146)
        int_422938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 20), tuple_422937, int_422938)
        # Adding element type (line 146)
        int_422939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 20), tuple_422937, int_422939)
        # Adding element type (line 146)
        int_422940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 20), tuple_422937, int_422940)
        
        # Testing the type of a for loop iterable (line 146)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 146, 8), tuple_422937)
        # Getting the type of the for loop variable (line 146)
        for_loop_var_422941 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 146, 8), tuple_422937)
        # Assigning a type to the variable 'num' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'num', for_loop_var_422941)
        # SSA begins for a for statement (line 146)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 147):
        
        # Assigning a Call to a Name (line 147):
        
        # Call to rand(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'n' (line 147)
        n_422945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 34), 'n', False)
        # Getting the type of 'n' (line 147)
        n_422946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 37), 'n', False)
        # Processing the call keyword arguments (line 147)
        float_422947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 48), 'float')
        keyword_422948 = float_422947
        kwargs_422949 = {'density': keyword_422948}
        # Getting the type of 'scipy' (line 147)
        scipy_422942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 147)
        sparse_422943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 16), scipy_422942, 'sparse')
        # Obtaining the member 'rand' of a type (line 147)
        rand_422944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 16), sparse_422943, 'rand')
        # Calling rand(args, kwargs) (line 147)
        rand_call_result_422950 = invoke(stypy.reporting.localization.Localization(__file__, 147, 16), rand_422944, *[n_422945, n_422946], **kwargs_422949)
        
        # Assigning a type to the variable 'A' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'A', rand_call_result_422950)
        
        # Assigning a Call to a Name (line 148):
        
        # Assigning a Call to a Name (line 148):
        
        # Call to randn(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'n' (line 148)
        n_422954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 32), 'n', False)
        # Getting the type of 'k' (line 148)
        k_422955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 35), 'k', False)
        # Processing the call keyword arguments (line 148)
        kwargs_422956 = {}
        # Getting the type of 'np' (line 148)
        np_422951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 16), 'np', False)
        # Obtaining the member 'random' of a type (line 148)
        random_422952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 16), np_422951, 'random')
        # Obtaining the member 'randn' of a type (line 148)
        randn_422953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 16), random_422952, 'randn')
        # Calling randn(args, kwargs) (line 148)
        randn_call_result_422957 = invoke(stypy.reporting.localization.Localization(__file__, 148, 16), randn_422953, *[n_422954, k_422955], **kwargs_422956)
        
        # Assigning a type to the variable 'B' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'B', randn_call_result_422957)
        
        # Assigning a Call to a Name (line 149):
        
        # Assigning a Call to a Name (line 149):
        
        # Call to randn(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'n' (line 149)
        n_422961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 32), 'n', False)
        # Processing the call keyword arguments (line 149)
        kwargs_422962 = {}
        # Getting the type of 'np' (line 149)
        np_422958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'np', False)
        # Obtaining the member 'random' of a type (line 149)
        random_422959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 16), np_422958, 'random')
        # Obtaining the member 'randn' of a type (line 149)
        randn_422960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 16), random_422959, 'randn')
        # Calling randn(args, kwargs) (line 149)
        randn_call_result_422963 = invoke(stypy.reporting.localization.Localization(__file__, 149, 16), randn_422960, *[n_422961], **kwargs_422962)
        
        # Assigning a type to the variable 'v' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'v', randn_call_result_422963)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 150)
        tuple_422964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 150)
        # Adding element type (line 150)
        # Getting the type of 'B' (line 150)
        B_422965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 27), 'B')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 27), tuple_422964, B_422965)
        # Adding element type (line 150)
        # Getting the type of 'v' (line 150)
        v_422966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 30), 'v')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 27), tuple_422964, v_422966)
        
        # Testing the type of a for loop iterable (line 150)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 150, 12), tuple_422964)
        # Getting the type of the for loop variable (line 150)
        for_loop_var_422967 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 150, 12), tuple_422964)
        # Assigning a type to the variable 'target' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'target', for_loop_var_422967)
        # SSA begins for a for statement (line 150)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 151):
        
        # Assigning a Call to a Name (line 151):
        
        # Call to expm_multiply(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'A' (line 151)
        A_422969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 34), 'A', False)
        # Getting the type of 'target' (line 151)
        target_422970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 37), 'target', False)
        # Processing the call keyword arguments (line 151)
        # Getting the type of 'start' (line 152)
        start_422971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 30), 'start', False)
        keyword_422972 = start_422971
        # Getting the type of 'stop' (line 152)
        stop_422973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 42), 'stop', False)
        keyword_422974 = stop_422973
        # Getting the type of 'num' (line 152)
        num_422975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 52), 'num', False)
        keyword_422976 = num_422975
        # Getting the type of 'endpoint' (line 152)
        endpoint_422977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 66), 'endpoint', False)
        keyword_422978 = endpoint_422977
        kwargs_422979 = {'start': keyword_422972, 'num': keyword_422976, 'stop': keyword_422974, 'endpoint': keyword_422978}
        # Getting the type of 'expm_multiply' (line 151)
        expm_multiply_422968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 20), 'expm_multiply', False)
        # Calling expm_multiply(args, kwargs) (line 151)
        expm_multiply_call_result_422980 = invoke(stypy.reporting.localization.Localization(__file__, 151, 20), expm_multiply_422968, *[A_422969, target_422970], **kwargs_422979)
        
        # Assigning a type to the variable 'X' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'X', expm_multiply_call_result_422980)
        
        # Assigning a Call to a Name (line 153):
        
        # Assigning a Call to a Name (line 153):
        
        # Call to linspace(...): (line 153)
        # Processing the call keyword arguments (line 153)
        # Getting the type of 'start' (line 153)
        start_422983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 44), 'start', False)
        keyword_422984 = start_422983
        # Getting the type of 'stop' (line 153)
        stop_422985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 56), 'stop', False)
        keyword_422986 = stop_422985
        # Getting the type of 'num' (line 154)
        num_422987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 28), 'num', False)
        keyword_422988 = num_422987
        # Getting the type of 'endpoint' (line 154)
        endpoint_422989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 42), 'endpoint', False)
        keyword_422990 = endpoint_422989
        kwargs_422991 = {'start': keyword_422984, 'num': keyword_422988, 'stop': keyword_422986, 'endpoint': keyword_422990}
        # Getting the type of 'np' (line 153)
        np_422981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 26), 'np', False)
        # Obtaining the member 'linspace' of a type (line 153)
        linspace_422982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 26), np_422981, 'linspace')
        # Calling linspace(args, kwargs) (line 153)
        linspace_call_result_422992 = invoke(stypy.reporting.localization.Localization(__file__, 153, 26), linspace_422982, *[], **kwargs_422991)
        
        # Assigning a type to the variable 'samples' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'samples', linspace_call_result_422992)
        
        # Call to suppress_warnings(...): (line 155)
        # Processing the call keyword arguments (line 155)
        kwargs_422994 = {}
        # Getting the type of 'suppress_warnings' (line 155)
        suppress_warnings_422993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 21), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 155)
        suppress_warnings_call_result_422995 = invoke(stypy.reporting.localization.Localization(__file__, 155, 21), suppress_warnings_422993, *[], **kwargs_422994)
        
        with_422996 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 155, 21), suppress_warnings_call_result_422995, 'with parameter', '__enter__', '__exit__')

        if with_422996:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 155)
            enter___422997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 21), suppress_warnings_call_result_422995, '__enter__')
            with_enter_422998 = invoke(stypy.reporting.localization.Localization(__file__, 155, 21), enter___422997)
            # Assigning a type to the variable 'sup' (line 155)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 21), 'sup', with_enter_422998)
            
            # Call to filter(...): (line 156)
            # Processing the call arguments (line 156)
            # Getting the type of 'SparseEfficiencyWarning' (line 156)
            SparseEfficiencyWarning_423001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 31), 'SparseEfficiencyWarning', False)
            str_423002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 31), 'str', 'splu requires CSC matrix format')
            # Processing the call keyword arguments (line 156)
            kwargs_423003 = {}
            # Getting the type of 'sup' (line 156)
            sup_422999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 20), 'sup', False)
            # Obtaining the member 'filter' of a type (line 156)
            filter_423000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 20), sup_422999, 'filter')
            # Calling filter(args, kwargs) (line 156)
            filter_call_result_423004 = invoke(stypy.reporting.localization.Localization(__file__, 156, 20), filter_423000, *[SparseEfficiencyWarning_423001, str_423002], **kwargs_423003)
            
            
            # Call to filter(...): (line 158)
            # Processing the call arguments (line 158)
            # Getting the type of 'SparseEfficiencyWarning' (line 158)
            SparseEfficiencyWarning_423007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 31), 'SparseEfficiencyWarning', False)
            str_423008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 31), 'str', 'spsolve is more efficient when sparse b is in the CSC matrix format')
            # Processing the call keyword arguments (line 158)
            kwargs_423009 = {}
            # Getting the type of 'sup' (line 158)
            sup_423005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 20), 'sup', False)
            # Obtaining the member 'filter' of a type (line 158)
            filter_423006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 20), sup_423005, 'filter')
            # Calling filter(args, kwargs) (line 158)
            filter_call_result_423010 = invoke(stypy.reporting.localization.Localization(__file__, 158, 20), filter_423006, *[SparseEfficiencyWarning_423007, str_423008], **kwargs_423009)
            
            
            
            # Call to zip(...): (line 160)
            # Processing the call arguments (line 160)
            # Getting the type of 'X' (line 160)
            X_423012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 43), 'X', False)
            # Getting the type of 'samples' (line 160)
            samples_423013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 46), 'samples', False)
            # Processing the call keyword arguments (line 160)
            kwargs_423014 = {}
            # Getting the type of 'zip' (line 160)
            zip_423011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 39), 'zip', False)
            # Calling zip(args, kwargs) (line 160)
            zip_call_result_423015 = invoke(stypy.reporting.localization.Localization(__file__, 160, 39), zip_423011, *[X_423012, samples_423013], **kwargs_423014)
            
            # Testing the type of a for loop iterable (line 160)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 160, 20), zip_call_result_423015)
            # Getting the type of the for loop variable (line 160)
            for_loop_var_423016 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 160, 20), zip_call_result_423015)
            # Assigning a type to the variable 'solution' (line 160)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'solution', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 20), for_loop_var_423016))
            # Assigning a type to the variable 't' (line 160)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 't', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 20), for_loop_var_423016))
            # SSA begins for a for statement (line 160)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to assert_allclose(...): (line 161)
            # Processing the call arguments (line 161)
            # Getting the type of 'solution' (line 161)
            solution_423018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 40), 'solution', False)
            
            # Call to dot(...): (line 162)
            # Processing the call arguments (line 162)
            # Getting the type of 'target' (line 162)
            target_423028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 59), 'target', False)
            # Processing the call keyword arguments (line 162)
            kwargs_423029 = {}
            
            # Call to expm(...): (line 162)
            # Processing the call arguments (line 162)
            # Getting the type of 't' (line 162)
            t_423022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 50), 't', False)
            # Getting the type of 'A' (line 162)
            A_423023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 52), 'A', False)
            # Applying the binary operator '*' (line 162)
            result_mul_423024 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 50), '*', t_423022, A_423023)
            
            # Processing the call keyword arguments (line 162)
            kwargs_423025 = {}
            # Getting the type of 'scipy' (line 162)
            scipy_423019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 32), 'scipy', False)
            # Obtaining the member 'linalg' of a type (line 162)
            linalg_423020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 32), scipy_423019, 'linalg')
            # Obtaining the member 'expm' of a type (line 162)
            expm_423021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 32), linalg_423020, 'expm')
            # Calling expm(args, kwargs) (line 162)
            expm_call_result_423026 = invoke(stypy.reporting.localization.Localization(__file__, 162, 32), expm_423021, *[result_mul_423024], **kwargs_423025)
            
            # Obtaining the member 'dot' of a type (line 162)
            dot_423027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 32), expm_call_result_423026, 'dot')
            # Calling dot(args, kwargs) (line 162)
            dot_call_result_423030 = invoke(stypy.reporting.localization.Localization(__file__, 162, 32), dot_423027, *[target_423028], **kwargs_423029)
            
            # Processing the call keyword arguments (line 161)
            kwargs_423031 = {}
            # Getting the type of 'assert_allclose' (line 161)
            assert_allclose_423017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 24), 'assert_allclose', False)
            # Calling assert_allclose(args, kwargs) (line 161)
            assert_allclose_call_result_423032 = invoke(stypy.reporting.localization.Localization(__file__, 161, 24), assert_allclose_423017, *[solution_423018, dot_call_result_423030], **kwargs_423031)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 155)
            exit___423033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 21), suppress_warnings_call_result_422995, '__exit__')
            with_exit_423034 = invoke(stypy.reporting.localization.Localization(__file__, 155, 21), exit___423033, None, None, None)

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_sparse_expm_multiply_interval(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sparse_expm_multiply_interval' in the type store
        # Getting the type of 'stypy_return_type' (line 139)
        stypy_return_type_423035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_423035)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sparse_expm_multiply_interval'
        return stypy_return_type_423035


    @norecursion
    def test_expm_multiply_interval_vector(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_expm_multiply_interval_vector'
        module_type_store = module_type_store.open_function_context('test_expm_multiply_interval_vector', 164, 4, False)
        # Assigning a type to the variable 'self' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpmActionInterval.test_expm_multiply_interval_vector.__dict__.__setitem__('stypy_localization', localization)
        TestExpmActionInterval.test_expm_multiply_interval_vector.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpmActionInterval.test_expm_multiply_interval_vector.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpmActionInterval.test_expm_multiply_interval_vector.__dict__.__setitem__('stypy_function_name', 'TestExpmActionInterval.test_expm_multiply_interval_vector')
        TestExpmActionInterval.test_expm_multiply_interval_vector.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpmActionInterval.test_expm_multiply_interval_vector.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpmActionInterval.test_expm_multiply_interval_vector.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpmActionInterval.test_expm_multiply_interval_vector.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpmActionInterval.test_expm_multiply_interval_vector.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpmActionInterval.test_expm_multiply_interval_vector.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpmActionInterval.test_expm_multiply_interval_vector.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpmActionInterval.test_expm_multiply_interval_vector', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_expm_multiply_interval_vector', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_expm_multiply_interval_vector(...)' code ##################

        
        # Call to seed(...): (line 165)
        # Processing the call arguments (line 165)
        int_423039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 23), 'int')
        # Processing the call keyword arguments (line 165)
        kwargs_423040 = {}
        # Getting the type of 'np' (line 165)
        np_423036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 165)
        random_423037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), np_423036, 'random')
        # Obtaining the member 'seed' of a type (line 165)
        seed_423038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), random_423037, 'seed')
        # Calling seed(args, kwargs) (line 165)
        seed_call_result_423041 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), seed_423038, *[int_423039], **kwargs_423040)
        
        
        # Assigning a Num to a Name (line 166):
        
        # Assigning a Num to a Name (line 166):
        float_423042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 16), 'float')
        # Assigning a type to the variable 'start' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'start', float_423042)
        
        # Assigning a Num to a Name (line 167):
        
        # Assigning a Num to a Name (line 167):
        float_423043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 15), 'float')
        # Assigning a type to the variable 'stop' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'stop', float_423043)
        
        # Assigning a Name to a Name (line 168):
        
        # Assigning a Name to a Name (line 168):
        # Getting the type of 'True' (line 168)
        True_423044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 19), 'True')
        # Assigning a type to the variable 'endpoint' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'endpoint', True_423044)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 169)
        tuple_423045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 169)
        # Adding element type (line 169)
        int_423046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 20), tuple_423045, int_423046)
        # Adding element type (line 169)
        int_423047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 20), tuple_423045, int_423047)
        # Adding element type (line 169)
        int_423048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 20), tuple_423045, int_423048)
        
        # Testing the type of a for loop iterable (line 169)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 169, 8), tuple_423045)
        # Getting the type of the for loop variable (line 169)
        for_loop_var_423049 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 169, 8), tuple_423045)
        # Assigning a type to the variable 'num' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'num', for_loop_var_423049)
        # SSA begins for a for statement (line 169)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 170)
        tuple_423050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 170)
        # Adding element type (line 170)
        int_423051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 22), tuple_423050, int_423051)
        # Adding element type (line 170)
        int_423052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 22), tuple_423050, int_423052)
        # Adding element type (line 170)
        int_423053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 22), tuple_423050, int_423053)
        # Adding element type (line 170)
        int_423054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 22), tuple_423050, int_423054)
        # Adding element type (line 170)
        int_423055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 22), tuple_423050, int_423055)
        
        # Testing the type of a for loop iterable (line 170)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 170, 12), tuple_423050)
        # Getting the type of the for loop variable (line 170)
        for_loop_var_423056 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 170, 12), tuple_423050)
        # Assigning a type to the variable 'n' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'n', for_loop_var_423056)
        # SSA begins for a for statement (line 170)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 171):
        
        # Assigning a Call to a Name (line 171):
        
        # Call to inv(...): (line 171)
        # Processing the call arguments (line 171)
        
        # Call to randn(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'n' (line 171)
        n_423063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 53), 'n', False)
        # Getting the type of 'n' (line 171)
        n_423064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 56), 'n', False)
        # Processing the call keyword arguments (line 171)
        kwargs_423065 = {}
        # Getting the type of 'np' (line 171)
        np_423060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 37), 'np', False)
        # Obtaining the member 'random' of a type (line 171)
        random_423061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 37), np_423060, 'random')
        # Obtaining the member 'randn' of a type (line 171)
        randn_423062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 37), random_423061, 'randn')
        # Calling randn(args, kwargs) (line 171)
        randn_call_result_423066 = invoke(stypy.reporting.localization.Localization(__file__, 171, 37), randn_423062, *[n_423063, n_423064], **kwargs_423065)
        
        # Processing the call keyword arguments (line 171)
        kwargs_423067 = {}
        # Getting the type of 'scipy' (line 171)
        scipy_423057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 20), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 171)
        linalg_423058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 20), scipy_423057, 'linalg')
        # Obtaining the member 'inv' of a type (line 171)
        inv_423059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 20), linalg_423058, 'inv')
        # Calling inv(args, kwargs) (line 171)
        inv_call_result_423068 = invoke(stypy.reporting.localization.Localization(__file__, 171, 20), inv_423059, *[randn_call_result_423066], **kwargs_423067)
        
        # Assigning a type to the variable 'A' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'A', inv_call_result_423068)
        
        # Assigning a Call to a Name (line 172):
        
        # Assigning a Call to a Name (line 172):
        
        # Call to randn(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'n' (line 172)
        n_423072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 36), 'n', False)
        # Processing the call keyword arguments (line 172)
        kwargs_423073 = {}
        # Getting the type of 'np' (line 172)
        np_423069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 20), 'np', False)
        # Obtaining the member 'random' of a type (line 172)
        random_423070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 20), np_423069, 'random')
        # Obtaining the member 'randn' of a type (line 172)
        randn_423071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 20), random_423070, 'randn')
        # Calling randn(args, kwargs) (line 172)
        randn_call_result_423074 = invoke(stypy.reporting.localization.Localization(__file__, 172, 20), randn_423071, *[n_423072], **kwargs_423073)
        
        # Assigning a type to the variable 'v' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'v', randn_call_result_423074)
        
        # Assigning a Call to a Name (line 173):
        
        # Assigning a Call to a Name (line 173):
        
        # Call to expm_multiply(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'A' (line 173)
        A_423076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 34), 'A', False)
        # Getting the type of 'v' (line 173)
        v_423077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 37), 'v', False)
        # Processing the call keyword arguments (line 173)
        # Getting the type of 'start' (line 174)
        start_423078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 30), 'start', False)
        keyword_423079 = start_423078
        # Getting the type of 'stop' (line 174)
        stop_423080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 42), 'stop', False)
        keyword_423081 = stop_423080
        # Getting the type of 'num' (line 174)
        num_423082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 52), 'num', False)
        keyword_423083 = num_423082
        # Getting the type of 'endpoint' (line 174)
        endpoint_423084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 66), 'endpoint', False)
        keyword_423085 = endpoint_423084
        kwargs_423086 = {'start': keyword_423079, 'num': keyword_423083, 'stop': keyword_423081, 'endpoint': keyword_423085}
        # Getting the type of 'expm_multiply' (line 173)
        expm_multiply_423075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 20), 'expm_multiply', False)
        # Calling expm_multiply(args, kwargs) (line 173)
        expm_multiply_call_result_423087 = invoke(stypy.reporting.localization.Localization(__file__, 173, 20), expm_multiply_423075, *[A_423076, v_423077], **kwargs_423086)
        
        # Assigning a type to the variable 'X' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'X', expm_multiply_call_result_423087)
        
        # Assigning a Call to a Name (line 175):
        
        # Assigning a Call to a Name (line 175):
        
        # Call to linspace(...): (line 175)
        # Processing the call keyword arguments (line 175)
        # Getting the type of 'start' (line 175)
        start_423090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 44), 'start', False)
        keyword_423091 = start_423090
        # Getting the type of 'stop' (line 175)
        stop_423092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 56), 'stop', False)
        keyword_423093 = stop_423092
        # Getting the type of 'num' (line 176)
        num_423094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 28), 'num', False)
        keyword_423095 = num_423094
        # Getting the type of 'endpoint' (line 176)
        endpoint_423096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 42), 'endpoint', False)
        keyword_423097 = endpoint_423096
        kwargs_423098 = {'start': keyword_423091, 'num': keyword_423095, 'stop': keyword_423093, 'endpoint': keyword_423097}
        # Getting the type of 'np' (line 175)
        np_423088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 26), 'np', False)
        # Obtaining the member 'linspace' of a type (line 175)
        linspace_423089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 26), np_423088, 'linspace')
        # Calling linspace(args, kwargs) (line 175)
        linspace_call_result_423099 = invoke(stypy.reporting.localization.Localization(__file__, 175, 26), linspace_423089, *[], **kwargs_423098)
        
        # Assigning a type to the variable 'samples' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 16), 'samples', linspace_call_result_423099)
        
        
        # Call to zip(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'X' (line 177)
        X_423101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 39), 'X', False)
        # Getting the type of 'samples' (line 177)
        samples_423102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 42), 'samples', False)
        # Processing the call keyword arguments (line 177)
        kwargs_423103 = {}
        # Getting the type of 'zip' (line 177)
        zip_423100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 35), 'zip', False)
        # Calling zip(args, kwargs) (line 177)
        zip_call_result_423104 = invoke(stypy.reporting.localization.Localization(__file__, 177, 35), zip_423100, *[X_423101, samples_423102], **kwargs_423103)
        
        # Testing the type of a for loop iterable (line 177)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 177, 16), zip_call_result_423104)
        # Getting the type of the for loop variable (line 177)
        for_loop_var_423105 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 177, 16), zip_call_result_423104)
        # Assigning a type to the variable 'solution' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 16), 'solution', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 16), for_loop_var_423105))
        # Assigning a type to the variable 't' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 16), 't', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 16), for_loop_var_423105))
        # SSA begins for a for statement (line 177)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_allclose(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'solution' (line 178)
        solution_423107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 36), 'solution', False)
        
        # Call to dot(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'v' (line 178)
        v_423117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 73), 'v', False)
        # Processing the call keyword arguments (line 178)
        kwargs_423118 = {}
        
        # Call to expm(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 't' (line 178)
        t_423111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 64), 't', False)
        # Getting the type of 'A' (line 178)
        A_423112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 66), 'A', False)
        # Applying the binary operator '*' (line 178)
        result_mul_423113 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 64), '*', t_423111, A_423112)
        
        # Processing the call keyword arguments (line 178)
        kwargs_423114 = {}
        # Getting the type of 'scipy' (line 178)
        scipy_423108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 46), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 178)
        linalg_423109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 46), scipy_423108, 'linalg')
        # Obtaining the member 'expm' of a type (line 178)
        expm_423110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 46), linalg_423109, 'expm')
        # Calling expm(args, kwargs) (line 178)
        expm_call_result_423115 = invoke(stypy.reporting.localization.Localization(__file__, 178, 46), expm_423110, *[result_mul_423113], **kwargs_423114)
        
        # Obtaining the member 'dot' of a type (line 178)
        dot_423116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 46), expm_call_result_423115, 'dot')
        # Calling dot(args, kwargs) (line 178)
        dot_call_result_423119 = invoke(stypy.reporting.localization.Localization(__file__, 178, 46), dot_423116, *[v_423117], **kwargs_423118)
        
        # Processing the call keyword arguments (line 178)
        kwargs_423120 = {}
        # Getting the type of 'assert_allclose' (line 178)
        assert_allclose_423106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 20), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 178)
        assert_allclose_call_result_423121 = invoke(stypy.reporting.localization.Localization(__file__, 178, 20), assert_allclose_423106, *[solution_423107, dot_call_result_423119], **kwargs_423120)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_expm_multiply_interval_vector(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_expm_multiply_interval_vector' in the type store
        # Getting the type of 'stypy_return_type' (line 164)
        stypy_return_type_423122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_423122)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_expm_multiply_interval_vector'
        return stypy_return_type_423122


    @norecursion
    def test_expm_multiply_interval_matrix(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_expm_multiply_interval_matrix'
        module_type_store = module_type_store.open_function_context('test_expm_multiply_interval_matrix', 180, 4, False)
        # Assigning a type to the variable 'self' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpmActionInterval.test_expm_multiply_interval_matrix.__dict__.__setitem__('stypy_localization', localization)
        TestExpmActionInterval.test_expm_multiply_interval_matrix.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpmActionInterval.test_expm_multiply_interval_matrix.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpmActionInterval.test_expm_multiply_interval_matrix.__dict__.__setitem__('stypy_function_name', 'TestExpmActionInterval.test_expm_multiply_interval_matrix')
        TestExpmActionInterval.test_expm_multiply_interval_matrix.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpmActionInterval.test_expm_multiply_interval_matrix.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpmActionInterval.test_expm_multiply_interval_matrix.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpmActionInterval.test_expm_multiply_interval_matrix.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpmActionInterval.test_expm_multiply_interval_matrix.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpmActionInterval.test_expm_multiply_interval_matrix.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpmActionInterval.test_expm_multiply_interval_matrix.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpmActionInterval.test_expm_multiply_interval_matrix', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_expm_multiply_interval_matrix', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_expm_multiply_interval_matrix(...)' code ##################

        
        # Call to seed(...): (line 181)
        # Processing the call arguments (line 181)
        int_423126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 23), 'int')
        # Processing the call keyword arguments (line 181)
        kwargs_423127 = {}
        # Getting the type of 'np' (line 181)
        np_423123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 181)
        random_423124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), np_423123, 'random')
        # Obtaining the member 'seed' of a type (line 181)
        seed_423125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), random_423124, 'seed')
        # Calling seed(args, kwargs) (line 181)
        seed_call_result_423128 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), seed_423125, *[int_423126], **kwargs_423127)
        
        
        # Assigning a Num to a Name (line 182):
        
        # Assigning a Num to a Name (line 182):
        float_423129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 16), 'float')
        # Assigning a type to the variable 'start' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'start', float_423129)
        
        # Assigning a Num to a Name (line 183):
        
        # Assigning a Num to a Name (line 183):
        float_423130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 15), 'float')
        # Assigning a type to the variable 'stop' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'stop', float_423130)
        
        # Assigning a Name to a Name (line 184):
        
        # Assigning a Name to a Name (line 184):
        # Getting the type of 'True' (line 184)
        True_423131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 19), 'True')
        # Assigning a type to the variable 'endpoint' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'endpoint', True_423131)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 185)
        tuple_423132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 185)
        # Adding element type (line 185)
        int_423133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 20), tuple_423132, int_423133)
        # Adding element type (line 185)
        int_423134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 20), tuple_423132, int_423134)
        # Adding element type (line 185)
        int_423135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 20), tuple_423132, int_423135)
        
        # Testing the type of a for loop iterable (line 185)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 185, 8), tuple_423132)
        # Getting the type of the for loop variable (line 185)
        for_loop_var_423136 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 185, 8), tuple_423132)
        # Assigning a type to the variable 'num' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'num', for_loop_var_423136)
        # SSA begins for a for statement (line 185)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 186)
        tuple_423137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 186)
        # Adding element type (line 186)
        int_423138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 22), tuple_423137, int_423138)
        # Adding element type (line 186)
        int_423139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 22), tuple_423137, int_423139)
        # Adding element type (line 186)
        int_423140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 22), tuple_423137, int_423140)
        # Adding element type (line 186)
        int_423141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 22), tuple_423137, int_423141)
        # Adding element type (line 186)
        int_423142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 22), tuple_423137, int_423142)
        
        # Testing the type of a for loop iterable (line 186)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 186, 12), tuple_423137)
        # Getting the type of the for loop variable (line 186)
        for_loop_var_423143 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 186, 12), tuple_423137)
        # Assigning a type to the variable 'n' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'n', for_loop_var_423143)
        # SSA begins for a for statement (line 186)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 187)
        tuple_423144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 187)
        # Adding element type (line 187)
        int_423145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 26), tuple_423144, int_423145)
        # Adding element type (line 187)
        int_423146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 26), tuple_423144, int_423146)
        
        # Testing the type of a for loop iterable (line 187)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 187, 16), tuple_423144)
        # Getting the type of the for loop variable (line 187)
        for_loop_var_423147 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 187, 16), tuple_423144)
        # Assigning a type to the variable 'k' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 16), 'k', for_loop_var_423147)
        # SSA begins for a for statement (line 187)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 188):
        
        # Assigning a Call to a Name (line 188):
        
        # Call to inv(...): (line 188)
        # Processing the call arguments (line 188)
        
        # Call to randn(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'n' (line 188)
        n_423154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 57), 'n', False)
        # Getting the type of 'n' (line 188)
        n_423155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 60), 'n', False)
        # Processing the call keyword arguments (line 188)
        kwargs_423156 = {}
        # Getting the type of 'np' (line 188)
        np_423151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 41), 'np', False)
        # Obtaining the member 'random' of a type (line 188)
        random_423152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 41), np_423151, 'random')
        # Obtaining the member 'randn' of a type (line 188)
        randn_423153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 41), random_423152, 'randn')
        # Calling randn(args, kwargs) (line 188)
        randn_call_result_423157 = invoke(stypy.reporting.localization.Localization(__file__, 188, 41), randn_423153, *[n_423154, n_423155], **kwargs_423156)
        
        # Processing the call keyword arguments (line 188)
        kwargs_423158 = {}
        # Getting the type of 'scipy' (line 188)
        scipy_423148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 24), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 188)
        linalg_423149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 24), scipy_423148, 'linalg')
        # Obtaining the member 'inv' of a type (line 188)
        inv_423150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 24), linalg_423149, 'inv')
        # Calling inv(args, kwargs) (line 188)
        inv_call_result_423159 = invoke(stypy.reporting.localization.Localization(__file__, 188, 24), inv_423150, *[randn_call_result_423157], **kwargs_423158)
        
        # Assigning a type to the variable 'A' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 20), 'A', inv_call_result_423159)
        
        # Assigning a Call to a Name (line 189):
        
        # Assigning a Call to a Name (line 189):
        
        # Call to randn(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'n' (line 189)
        n_423163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 40), 'n', False)
        # Getting the type of 'k' (line 189)
        k_423164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 43), 'k', False)
        # Processing the call keyword arguments (line 189)
        kwargs_423165 = {}
        # Getting the type of 'np' (line 189)
        np_423160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 24), 'np', False)
        # Obtaining the member 'random' of a type (line 189)
        random_423161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 24), np_423160, 'random')
        # Obtaining the member 'randn' of a type (line 189)
        randn_423162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 24), random_423161, 'randn')
        # Calling randn(args, kwargs) (line 189)
        randn_call_result_423166 = invoke(stypy.reporting.localization.Localization(__file__, 189, 24), randn_423162, *[n_423163, k_423164], **kwargs_423165)
        
        # Assigning a type to the variable 'B' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 20), 'B', randn_call_result_423166)
        
        # Assigning a Call to a Name (line 190):
        
        # Assigning a Call to a Name (line 190):
        
        # Call to expm_multiply(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'A' (line 190)
        A_423168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 38), 'A', False)
        # Getting the type of 'B' (line 190)
        B_423169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 41), 'B', False)
        # Processing the call keyword arguments (line 190)
        # Getting the type of 'start' (line 191)
        start_423170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 34), 'start', False)
        keyword_423171 = start_423170
        # Getting the type of 'stop' (line 191)
        stop_423172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 46), 'stop', False)
        keyword_423173 = stop_423172
        # Getting the type of 'num' (line 191)
        num_423174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 56), 'num', False)
        keyword_423175 = num_423174
        # Getting the type of 'endpoint' (line 191)
        endpoint_423176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 70), 'endpoint', False)
        keyword_423177 = endpoint_423176
        kwargs_423178 = {'start': keyword_423171, 'num': keyword_423175, 'stop': keyword_423173, 'endpoint': keyword_423177}
        # Getting the type of 'expm_multiply' (line 190)
        expm_multiply_423167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 24), 'expm_multiply', False)
        # Calling expm_multiply(args, kwargs) (line 190)
        expm_multiply_call_result_423179 = invoke(stypy.reporting.localization.Localization(__file__, 190, 24), expm_multiply_423167, *[A_423168, B_423169], **kwargs_423178)
        
        # Assigning a type to the variable 'X' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 20), 'X', expm_multiply_call_result_423179)
        
        # Assigning a Call to a Name (line 192):
        
        # Assigning a Call to a Name (line 192):
        
        # Call to linspace(...): (line 192)
        # Processing the call keyword arguments (line 192)
        # Getting the type of 'start' (line 192)
        start_423182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 48), 'start', False)
        keyword_423183 = start_423182
        # Getting the type of 'stop' (line 192)
        stop_423184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 60), 'stop', False)
        keyword_423185 = stop_423184
        # Getting the type of 'num' (line 193)
        num_423186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 32), 'num', False)
        keyword_423187 = num_423186
        # Getting the type of 'endpoint' (line 193)
        endpoint_423188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 46), 'endpoint', False)
        keyword_423189 = endpoint_423188
        kwargs_423190 = {'start': keyword_423183, 'num': keyword_423187, 'stop': keyword_423185, 'endpoint': keyword_423189}
        # Getting the type of 'np' (line 192)
        np_423180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 30), 'np', False)
        # Obtaining the member 'linspace' of a type (line 192)
        linspace_423181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 30), np_423180, 'linspace')
        # Calling linspace(args, kwargs) (line 192)
        linspace_call_result_423191 = invoke(stypy.reporting.localization.Localization(__file__, 192, 30), linspace_423181, *[], **kwargs_423190)
        
        # Assigning a type to the variable 'samples' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 20), 'samples', linspace_call_result_423191)
        
        
        # Call to zip(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'X' (line 194)
        X_423193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 43), 'X', False)
        # Getting the type of 'samples' (line 194)
        samples_423194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 46), 'samples', False)
        # Processing the call keyword arguments (line 194)
        kwargs_423195 = {}
        # Getting the type of 'zip' (line 194)
        zip_423192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 39), 'zip', False)
        # Calling zip(args, kwargs) (line 194)
        zip_call_result_423196 = invoke(stypy.reporting.localization.Localization(__file__, 194, 39), zip_423192, *[X_423193, samples_423194], **kwargs_423195)
        
        # Testing the type of a for loop iterable (line 194)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 194, 20), zip_call_result_423196)
        # Getting the type of the for loop variable (line 194)
        for_loop_var_423197 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 194, 20), zip_call_result_423196)
        # Assigning a type to the variable 'solution' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 20), 'solution', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 20), for_loop_var_423197))
        # Assigning a type to the variable 't' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 20), 't', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 20), for_loop_var_423197))
        # SSA begins for a for statement (line 194)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_allclose(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'solution' (line 195)
        solution_423199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 40), 'solution', False)
        
        # Call to dot(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'B' (line 195)
        B_423209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 77), 'B', False)
        # Processing the call keyword arguments (line 195)
        kwargs_423210 = {}
        
        # Call to expm(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 't' (line 195)
        t_423203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 68), 't', False)
        # Getting the type of 'A' (line 195)
        A_423204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 70), 'A', False)
        # Applying the binary operator '*' (line 195)
        result_mul_423205 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 68), '*', t_423203, A_423204)
        
        # Processing the call keyword arguments (line 195)
        kwargs_423206 = {}
        # Getting the type of 'scipy' (line 195)
        scipy_423200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 50), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 195)
        linalg_423201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 50), scipy_423200, 'linalg')
        # Obtaining the member 'expm' of a type (line 195)
        expm_423202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 50), linalg_423201, 'expm')
        # Calling expm(args, kwargs) (line 195)
        expm_call_result_423207 = invoke(stypy.reporting.localization.Localization(__file__, 195, 50), expm_423202, *[result_mul_423205], **kwargs_423206)
        
        # Obtaining the member 'dot' of a type (line 195)
        dot_423208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 50), expm_call_result_423207, 'dot')
        # Calling dot(args, kwargs) (line 195)
        dot_call_result_423211 = invoke(stypy.reporting.localization.Localization(__file__, 195, 50), dot_423208, *[B_423209], **kwargs_423210)
        
        # Processing the call keyword arguments (line 195)
        kwargs_423212 = {}
        # Getting the type of 'assert_allclose' (line 195)
        assert_allclose_423198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 24), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 195)
        assert_allclose_call_result_423213 = invoke(stypy.reporting.localization.Localization(__file__, 195, 24), assert_allclose_423198, *[solution_423199, dot_call_result_423211], **kwargs_423212)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_expm_multiply_interval_matrix(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_expm_multiply_interval_matrix' in the type store
        # Getting the type of 'stypy_return_type' (line 180)
        stypy_return_type_423214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_423214)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_expm_multiply_interval_matrix'
        return stypy_return_type_423214


    @norecursion
    def test_sparse_expm_multiply_interval_dtypes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sparse_expm_multiply_interval_dtypes'
        module_type_store = module_type_store.open_function_context('test_sparse_expm_multiply_interval_dtypes', 197, 4, False)
        # Assigning a type to the variable 'self' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpmActionInterval.test_sparse_expm_multiply_interval_dtypes.__dict__.__setitem__('stypy_localization', localization)
        TestExpmActionInterval.test_sparse_expm_multiply_interval_dtypes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpmActionInterval.test_sparse_expm_multiply_interval_dtypes.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpmActionInterval.test_sparse_expm_multiply_interval_dtypes.__dict__.__setitem__('stypy_function_name', 'TestExpmActionInterval.test_sparse_expm_multiply_interval_dtypes')
        TestExpmActionInterval.test_sparse_expm_multiply_interval_dtypes.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpmActionInterval.test_sparse_expm_multiply_interval_dtypes.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpmActionInterval.test_sparse_expm_multiply_interval_dtypes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpmActionInterval.test_sparse_expm_multiply_interval_dtypes.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpmActionInterval.test_sparse_expm_multiply_interval_dtypes.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpmActionInterval.test_sparse_expm_multiply_interval_dtypes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpmActionInterval.test_sparse_expm_multiply_interval_dtypes.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpmActionInterval.test_sparse_expm_multiply_interval_dtypes', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sparse_expm_multiply_interval_dtypes', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sparse_expm_multiply_interval_dtypes(...)' code ##################

        
        # Assigning a Call to a Name (line 199):
        
        # Assigning a Call to a Name (line 199):
        
        # Call to diags(...): (line 199)
        # Processing the call arguments (line 199)
        
        # Call to arange(...): (line 199)
        # Processing the call arguments (line 199)
        int_423220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 41), 'int')
        # Processing the call keyword arguments (line 199)
        kwargs_423221 = {}
        # Getting the type of 'np' (line 199)
        np_423218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 31), 'np', False)
        # Obtaining the member 'arange' of a type (line 199)
        arange_423219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 31), np_423218, 'arange')
        # Calling arange(args, kwargs) (line 199)
        arange_call_result_423222 = invoke(stypy.reporting.localization.Localization(__file__, 199, 31), arange_423219, *[int_423220], **kwargs_423221)
        
        # Processing the call keyword arguments (line 199)
        str_423223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 51), 'str', 'csr')
        keyword_423224 = str_423223
        # Getting the type of 'int' (line 199)
        int_423225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 64), 'int', False)
        keyword_423226 = int_423225
        kwargs_423227 = {'dtype': keyword_423226, 'format': keyword_423224}
        # Getting the type of 'scipy' (line 199)
        scipy_423215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 199)
        sparse_423216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 12), scipy_423215, 'sparse')
        # Obtaining the member 'diags' of a type (line 199)
        diags_423217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 12), sparse_423216, 'diags')
        # Calling diags(args, kwargs) (line 199)
        diags_call_result_423228 = invoke(stypy.reporting.localization.Localization(__file__, 199, 12), diags_423217, *[arange_call_result_423222], **kwargs_423227)
        
        # Assigning a type to the variable 'A' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'A', diags_call_result_423228)
        
        # Assigning a Call to a Name (line 200):
        
        # Assigning a Call to a Name (line 200):
        
        # Call to ones(...): (line 200)
        # Processing the call arguments (line 200)
        int_423231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 20), 'int')
        # Processing the call keyword arguments (line 200)
        # Getting the type of 'int' (line 200)
        int_423232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 29), 'int', False)
        keyword_423233 = int_423232
        kwargs_423234 = {'dtype': keyword_423233}
        # Getting the type of 'np' (line 200)
        np_423229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'np', False)
        # Obtaining the member 'ones' of a type (line 200)
        ones_423230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 12), np_423229, 'ones')
        # Calling ones(args, kwargs) (line 200)
        ones_call_result_423235 = invoke(stypy.reporting.localization.Localization(__file__, 200, 12), ones_423230, *[int_423231], **kwargs_423234)
        
        # Assigning a type to the variable 'B' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'B', ones_call_result_423235)
        
        # Assigning a Call to a Name (line 201):
        
        # Assigning a Call to a Name (line 201):
        
        # Call to diags(...): (line 201)
        # Processing the call arguments (line 201)
        
        # Call to exp(...): (line 201)
        # Processing the call arguments (line 201)
        
        # Call to arange(...): (line 201)
        # Processing the call arguments (line 201)
        int_423243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 52), 'int')
        # Processing the call keyword arguments (line 201)
        kwargs_423244 = {}
        # Getting the type of 'np' (line 201)
        np_423241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 42), 'np', False)
        # Obtaining the member 'arange' of a type (line 201)
        arange_423242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 42), np_423241, 'arange')
        # Calling arange(args, kwargs) (line 201)
        arange_call_result_423245 = invoke(stypy.reporting.localization.Localization(__file__, 201, 42), arange_423242, *[int_423243], **kwargs_423244)
        
        # Processing the call keyword arguments (line 201)
        kwargs_423246 = {}
        # Getting the type of 'np' (line 201)
        np_423239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 35), 'np', False)
        # Obtaining the member 'exp' of a type (line 201)
        exp_423240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 35), np_423239, 'exp')
        # Calling exp(args, kwargs) (line 201)
        exp_call_result_423247 = invoke(stypy.reporting.localization.Localization(__file__, 201, 35), exp_423240, *[arange_call_result_423245], **kwargs_423246)
        
        # Processing the call keyword arguments (line 201)
        str_423248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 63), 'str', 'csr')
        keyword_423249 = str_423248
        kwargs_423250 = {'format': keyword_423249}
        # Getting the type of 'scipy' (line 201)
        scipy_423236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 201)
        sparse_423237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 16), scipy_423236, 'sparse')
        # Obtaining the member 'diags' of a type (line 201)
        diags_423238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 16), sparse_423237, 'diags')
        # Calling diags(args, kwargs) (line 201)
        diags_call_result_423251 = invoke(stypy.reporting.localization.Localization(__file__, 201, 16), diags_423238, *[exp_call_result_423247], **kwargs_423250)
        
        # Assigning a type to the variable 'Aexpm' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'Aexpm', diags_call_result_423251)
        
        # Call to assert_allclose(...): (line 202)
        # Processing the call arguments (line 202)
        
        # Obtaining the type of the subscript
        int_423253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 47), 'int')
        
        # Call to expm_multiply(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'A' (line 202)
        A_423255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 38), 'A', False)
        # Getting the type of 'B' (line 202)
        B_423256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 40), 'B', False)
        int_423257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 42), 'int')
        int_423258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 44), 'int')
        # Processing the call keyword arguments (line 202)
        kwargs_423259 = {}
        # Getting the type of 'expm_multiply' (line 202)
        expm_multiply_423254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 24), 'expm_multiply', False)
        # Calling expm_multiply(args, kwargs) (line 202)
        expm_multiply_call_result_423260 = invoke(stypy.reporting.localization.Localization(__file__, 202, 24), expm_multiply_423254, *[A_423255, B_423256, int_423257, int_423258], **kwargs_423259)
        
        # Obtaining the member '__getitem__' of a type (line 202)
        getitem___423261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 24), expm_multiply_call_result_423260, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 202)
        subscript_call_result_423262 = invoke(stypy.reporting.localization.Localization(__file__, 202, 24), getitem___423261, int_423253)
        
        
        # Call to dot(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'B' (line 202)
        B_423265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 62), 'B', False)
        # Processing the call keyword arguments (line 202)
        kwargs_423266 = {}
        # Getting the type of 'Aexpm' (line 202)
        Aexpm_423263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 52), 'Aexpm', False)
        # Obtaining the member 'dot' of a type (line 202)
        dot_423264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 52), Aexpm_423263, 'dot')
        # Calling dot(args, kwargs) (line 202)
        dot_call_result_423267 = invoke(stypy.reporting.localization.Localization(__file__, 202, 52), dot_423264, *[B_423265], **kwargs_423266)
        
        # Processing the call keyword arguments (line 202)
        kwargs_423268 = {}
        # Getting the type of 'assert_allclose' (line 202)
        assert_allclose_423252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 202)
        assert_allclose_call_result_423269 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), assert_allclose_423252, *[subscript_call_result_423262, dot_call_result_423267], **kwargs_423268)
        
        
        # Assigning a Call to a Name (line 205):
        
        # Assigning a Call to a Name (line 205):
        
        # Call to diags(...): (line 205)
        # Processing the call arguments (line 205)
        complex_423273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 31), 'complex')
        
        # Call to arange(...): (line 205)
        # Processing the call arguments (line 205)
        int_423276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 45), 'int')
        # Processing the call keyword arguments (line 205)
        kwargs_423277 = {}
        # Getting the type of 'np' (line 205)
        np_423274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 35), 'np', False)
        # Obtaining the member 'arange' of a type (line 205)
        arange_423275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 35), np_423274, 'arange')
        # Calling arange(args, kwargs) (line 205)
        arange_call_result_423278 = invoke(stypy.reporting.localization.Localization(__file__, 205, 35), arange_423275, *[int_423276], **kwargs_423277)
        
        # Applying the binary operator '*' (line 205)
        result_mul_423279 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 31), '*', complex_423273, arange_call_result_423278)
        
        # Processing the call keyword arguments (line 205)
        str_423280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 55), 'str', 'csr')
        keyword_423281 = str_423280
        # Getting the type of 'complex' (line 205)
        complex_423282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 68), 'complex', False)
        keyword_423283 = complex_423282
        kwargs_423284 = {'dtype': keyword_423283, 'format': keyword_423281}
        # Getting the type of 'scipy' (line 205)
        scipy_423270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 205)
        sparse_423271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 12), scipy_423270, 'sparse')
        # Obtaining the member 'diags' of a type (line 205)
        diags_423272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 12), sparse_423271, 'diags')
        # Calling diags(args, kwargs) (line 205)
        diags_call_result_423285 = invoke(stypy.reporting.localization.Localization(__file__, 205, 12), diags_423272, *[result_mul_423279], **kwargs_423284)
        
        # Assigning a type to the variable 'A' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'A', diags_call_result_423285)
        
        # Assigning a Call to a Name (line 206):
        
        # Assigning a Call to a Name (line 206):
        
        # Call to ones(...): (line 206)
        # Processing the call arguments (line 206)
        int_423288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 20), 'int')
        # Processing the call keyword arguments (line 206)
        # Getting the type of 'int' (line 206)
        int_423289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 29), 'int', False)
        keyword_423290 = int_423289
        kwargs_423291 = {'dtype': keyword_423290}
        # Getting the type of 'np' (line 206)
        np_423286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'np', False)
        # Obtaining the member 'ones' of a type (line 206)
        ones_423287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 12), np_423286, 'ones')
        # Calling ones(args, kwargs) (line 206)
        ones_call_result_423292 = invoke(stypy.reporting.localization.Localization(__file__, 206, 12), ones_423287, *[int_423288], **kwargs_423291)
        
        # Assigning a type to the variable 'B' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'B', ones_call_result_423292)
        
        # Assigning a Call to a Name (line 207):
        
        # Assigning a Call to a Name (line 207):
        
        # Call to diags(...): (line 207)
        # Processing the call arguments (line 207)
        
        # Call to exp(...): (line 207)
        # Processing the call arguments (line 207)
        complex_423298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 42), 'complex')
        
        # Call to arange(...): (line 207)
        # Processing the call arguments (line 207)
        int_423301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 56), 'int')
        # Processing the call keyword arguments (line 207)
        kwargs_423302 = {}
        # Getting the type of 'np' (line 207)
        np_423299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 46), 'np', False)
        # Obtaining the member 'arange' of a type (line 207)
        arange_423300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 46), np_423299, 'arange')
        # Calling arange(args, kwargs) (line 207)
        arange_call_result_423303 = invoke(stypy.reporting.localization.Localization(__file__, 207, 46), arange_423300, *[int_423301], **kwargs_423302)
        
        # Applying the binary operator '*' (line 207)
        result_mul_423304 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 42), '*', complex_423298, arange_call_result_423303)
        
        # Processing the call keyword arguments (line 207)
        kwargs_423305 = {}
        # Getting the type of 'np' (line 207)
        np_423296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 35), 'np', False)
        # Obtaining the member 'exp' of a type (line 207)
        exp_423297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 35), np_423296, 'exp')
        # Calling exp(args, kwargs) (line 207)
        exp_call_result_423306 = invoke(stypy.reporting.localization.Localization(__file__, 207, 35), exp_423297, *[result_mul_423304], **kwargs_423305)
        
        # Processing the call keyword arguments (line 207)
        str_423307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 67), 'str', 'csr')
        keyword_423308 = str_423307
        kwargs_423309 = {'format': keyword_423308}
        # Getting the type of 'scipy' (line 207)
        scipy_423293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 207)
        sparse_423294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 16), scipy_423293, 'sparse')
        # Obtaining the member 'diags' of a type (line 207)
        diags_423295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 16), sparse_423294, 'diags')
        # Calling diags(args, kwargs) (line 207)
        diags_call_result_423310 = invoke(stypy.reporting.localization.Localization(__file__, 207, 16), diags_423295, *[exp_call_result_423306], **kwargs_423309)
        
        # Assigning a type to the variable 'Aexpm' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'Aexpm', diags_call_result_423310)
        
        # Call to assert_allclose(...): (line 208)
        # Processing the call arguments (line 208)
        
        # Obtaining the type of the subscript
        int_423312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 47), 'int')
        
        # Call to expm_multiply(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'A' (line 208)
        A_423314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 38), 'A', False)
        # Getting the type of 'B' (line 208)
        B_423315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 40), 'B', False)
        int_423316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 42), 'int')
        int_423317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 44), 'int')
        # Processing the call keyword arguments (line 208)
        kwargs_423318 = {}
        # Getting the type of 'expm_multiply' (line 208)
        expm_multiply_423313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 24), 'expm_multiply', False)
        # Calling expm_multiply(args, kwargs) (line 208)
        expm_multiply_call_result_423319 = invoke(stypy.reporting.localization.Localization(__file__, 208, 24), expm_multiply_423313, *[A_423314, B_423315, int_423316, int_423317], **kwargs_423318)
        
        # Obtaining the member '__getitem__' of a type (line 208)
        getitem___423320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 24), expm_multiply_call_result_423319, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 208)
        subscript_call_result_423321 = invoke(stypy.reporting.localization.Localization(__file__, 208, 24), getitem___423320, int_423312)
        
        
        # Call to dot(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'B' (line 208)
        B_423324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 62), 'B', False)
        # Processing the call keyword arguments (line 208)
        kwargs_423325 = {}
        # Getting the type of 'Aexpm' (line 208)
        Aexpm_423322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 52), 'Aexpm', False)
        # Obtaining the member 'dot' of a type (line 208)
        dot_423323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 52), Aexpm_423322, 'dot')
        # Calling dot(args, kwargs) (line 208)
        dot_call_result_423326 = invoke(stypy.reporting.localization.Localization(__file__, 208, 52), dot_423323, *[B_423324], **kwargs_423325)
        
        # Processing the call keyword arguments (line 208)
        kwargs_423327 = {}
        # Getting the type of 'assert_allclose' (line 208)
        assert_allclose_423311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 208)
        assert_allclose_call_result_423328 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), assert_allclose_423311, *[subscript_call_result_423321, dot_call_result_423326], **kwargs_423327)
        
        
        # Assigning a Call to a Name (line 211):
        
        # Assigning a Call to a Name (line 211):
        
        # Call to diags(...): (line 211)
        # Processing the call arguments (line 211)
        
        # Call to arange(...): (line 211)
        # Processing the call arguments (line 211)
        int_423334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 41), 'int')
        # Processing the call keyword arguments (line 211)
        kwargs_423335 = {}
        # Getting the type of 'np' (line 211)
        np_423332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 31), 'np', False)
        # Obtaining the member 'arange' of a type (line 211)
        arange_423333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 31), np_423332, 'arange')
        # Calling arange(args, kwargs) (line 211)
        arange_call_result_423336 = invoke(stypy.reporting.localization.Localization(__file__, 211, 31), arange_423333, *[int_423334], **kwargs_423335)
        
        # Processing the call keyword arguments (line 211)
        str_423337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 51), 'str', 'csr')
        keyword_423338 = str_423337
        # Getting the type of 'int' (line 211)
        int_423339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 64), 'int', False)
        keyword_423340 = int_423339
        kwargs_423341 = {'dtype': keyword_423340, 'format': keyword_423338}
        # Getting the type of 'scipy' (line 211)
        scipy_423329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 211)
        sparse_423330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 12), scipy_423329, 'sparse')
        # Obtaining the member 'diags' of a type (line 211)
        diags_423331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 12), sparse_423330, 'diags')
        # Calling diags(args, kwargs) (line 211)
        diags_call_result_423342 = invoke(stypy.reporting.localization.Localization(__file__, 211, 12), diags_423331, *[arange_call_result_423336], **kwargs_423341)
        
        # Assigning a type to the variable 'A' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'A', diags_call_result_423342)
        
        # Assigning a BinOp to a Name (line 212):
        
        # Assigning a BinOp to a Name (line 212):
        complex_423343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 12), 'complex')
        
        # Call to ones(...): (line 212)
        # Processing the call arguments (line 212)
        int_423346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 23), 'int')
        # Processing the call keyword arguments (line 212)
        # Getting the type of 'complex' (line 212)
        complex_423347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 32), 'complex', False)
        keyword_423348 = complex_423347
        kwargs_423349 = {'dtype': keyword_423348}
        # Getting the type of 'np' (line 212)
        np_423344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 15), 'np', False)
        # Obtaining the member 'ones' of a type (line 212)
        ones_423345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 15), np_423344, 'ones')
        # Calling ones(args, kwargs) (line 212)
        ones_call_result_423350 = invoke(stypy.reporting.localization.Localization(__file__, 212, 15), ones_423345, *[int_423346], **kwargs_423349)
        
        # Applying the binary operator '*' (line 212)
        result_mul_423351 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 12), '*', complex_423343, ones_call_result_423350)
        
        # Assigning a type to the variable 'B' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'B', result_mul_423351)
        
        # Assigning a Call to a Name (line 213):
        
        # Assigning a Call to a Name (line 213):
        
        # Call to diags(...): (line 213)
        # Processing the call arguments (line 213)
        
        # Call to exp(...): (line 213)
        # Processing the call arguments (line 213)
        
        # Call to arange(...): (line 213)
        # Processing the call arguments (line 213)
        int_423359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 52), 'int')
        # Processing the call keyword arguments (line 213)
        kwargs_423360 = {}
        # Getting the type of 'np' (line 213)
        np_423357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 42), 'np', False)
        # Obtaining the member 'arange' of a type (line 213)
        arange_423358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 42), np_423357, 'arange')
        # Calling arange(args, kwargs) (line 213)
        arange_call_result_423361 = invoke(stypy.reporting.localization.Localization(__file__, 213, 42), arange_423358, *[int_423359], **kwargs_423360)
        
        # Processing the call keyword arguments (line 213)
        kwargs_423362 = {}
        # Getting the type of 'np' (line 213)
        np_423355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 35), 'np', False)
        # Obtaining the member 'exp' of a type (line 213)
        exp_423356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 35), np_423355, 'exp')
        # Calling exp(args, kwargs) (line 213)
        exp_call_result_423363 = invoke(stypy.reporting.localization.Localization(__file__, 213, 35), exp_423356, *[arange_call_result_423361], **kwargs_423362)
        
        # Processing the call keyword arguments (line 213)
        str_423364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 63), 'str', 'csr')
        keyword_423365 = str_423364
        kwargs_423366 = {'format': keyword_423365}
        # Getting the type of 'scipy' (line 213)
        scipy_423352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 213)
        sparse_423353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 16), scipy_423352, 'sparse')
        # Obtaining the member 'diags' of a type (line 213)
        diags_423354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 16), sparse_423353, 'diags')
        # Calling diags(args, kwargs) (line 213)
        diags_call_result_423367 = invoke(stypy.reporting.localization.Localization(__file__, 213, 16), diags_423354, *[exp_call_result_423363], **kwargs_423366)
        
        # Assigning a type to the variable 'Aexpm' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'Aexpm', diags_call_result_423367)
        
        # Call to assert_allclose(...): (line 214)
        # Processing the call arguments (line 214)
        
        # Obtaining the type of the subscript
        int_423369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 47), 'int')
        
        # Call to expm_multiply(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'A' (line 214)
        A_423371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 38), 'A', False)
        # Getting the type of 'B' (line 214)
        B_423372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 40), 'B', False)
        int_423373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 42), 'int')
        int_423374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 44), 'int')
        # Processing the call keyword arguments (line 214)
        kwargs_423375 = {}
        # Getting the type of 'expm_multiply' (line 214)
        expm_multiply_423370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 24), 'expm_multiply', False)
        # Calling expm_multiply(args, kwargs) (line 214)
        expm_multiply_call_result_423376 = invoke(stypy.reporting.localization.Localization(__file__, 214, 24), expm_multiply_423370, *[A_423371, B_423372, int_423373, int_423374], **kwargs_423375)
        
        # Obtaining the member '__getitem__' of a type (line 214)
        getitem___423377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 24), expm_multiply_call_result_423376, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 214)
        subscript_call_result_423378 = invoke(stypy.reporting.localization.Localization(__file__, 214, 24), getitem___423377, int_423369)
        
        
        # Call to dot(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'B' (line 214)
        B_423381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 62), 'B', False)
        # Processing the call keyword arguments (line 214)
        kwargs_423382 = {}
        # Getting the type of 'Aexpm' (line 214)
        Aexpm_423379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 52), 'Aexpm', False)
        # Obtaining the member 'dot' of a type (line 214)
        dot_423380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 52), Aexpm_423379, 'dot')
        # Calling dot(args, kwargs) (line 214)
        dot_call_result_423383 = invoke(stypy.reporting.localization.Localization(__file__, 214, 52), dot_423380, *[B_423381], **kwargs_423382)
        
        # Processing the call keyword arguments (line 214)
        kwargs_423384 = {}
        # Getting the type of 'assert_allclose' (line 214)
        assert_allclose_423368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 214)
        assert_allclose_call_result_423385 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), assert_allclose_423368, *[subscript_call_result_423378, dot_call_result_423383], **kwargs_423384)
        
        
        # ################# End of 'test_sparse_expm_multiply_interval_dtypes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sparse_expm_multiply_interval_dtypes' in the type store
        # Getting the type of 'stypy_return_type' (line 197)
        stypy_return_type_423386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_423386)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sparse_expm_multiply_interval_dtypes'
        return stypy_return_type_423386


    @norecursion
    def test_expm_multiply_interval_status_0(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_expm_multiply_interval_status_0'
        module_type_store = module_type_store.open_function_context('test_expm_multiply_interval_status_0', 216, 4, False)
        # Assigning a type to the variable 'self' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpmActionInterval.test_expm_multiply_interval_status_0.__dict__.__setitem__('stypy_localization', localization)
        TestExpmActionInterval.test_expm_multiply_interval_status_0.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpmActionInterval.test_expm_multiply_interval_status_0.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpmActionInterval.test_expm_multiply_interval_status_0.__dict__.__setitem__('stypy_function_name', 'TestExpmActionInterval.test_expm_multiply_interval_status_0')
        TestExpmActionInterval.test_expm_multiply_interval_status_0.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpmActionInterval.test_expm_multiply_interval_status_0.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpmActionInterval.test_expm_multiply_interval_status_0.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpmActionInterval.test_expm_multiply_interval_status_0.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpmActionInterval.test_expm_multiply_interval_status_0.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpmActionInterval.test_expm_multiply_interval_status_0.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpmActionInterval.test_expm_multiply_interval_status_0.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpmActionInterval.test_expm_multiply_interval_status_0', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_expm_multiply_interval_status_0', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_expm_multiply_interval_status_0(...)' code ##################

        
        # Call to _help_test_specific_expm_interval_status(...): (line 217)
        # Processing the call arguments (line 217)
        int_423389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 54), 'int')
        # Processing the call keyword arguments (line 217)
        kwargs_423390 = {}
        # Getting the type of 'self' (line 217)
        self_423387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'self', False)
        # Obtaining the member '_help_test_specific_expm_interval_status' of a type (line 217)
        _help_test_specific_expm_interval_status_423388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 8), self_423387, '_help_test_specific_expm_interval_status')
        # Calling _help_test_specific_expm_interval_status(args, kwargs) (line 217)
        _help_test_specific_expm_interval_status_call_result_423391 = invoke(stypy.reporting.localization.Localization(__file__, 217, 8), _help_test_specific_expm_interval_status_423388, *[int_423389], **kwargs_423390)
        
        
        # ################# End of 'test_expm_multiply_interval_status_0(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_expm_multiply_interval_status_0' in the type store
        # Getting the type of 'stypy_return_type' (line 216)
        stypy_return_type_423392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_423392)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_expm_multiply_interval_status_0'
        return stypy_return_type_423392


    @norecursion
    def test_expm_multiply_interval_status_1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_expm_multiply_interval_status_1'
        module_type_store = module_type_store.open_function_context('test_expm_multiply_interval_status_1', 219, 4, False)
        # Assigning a type to the variable 'self' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpmActionInterval.test_expm_multiply_interval_status_1.__dict__.__setitem__('stypy_localization', localization)
        TestExpmActionInterval.test_expm_multiply_interval_status_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpmActionInterval.test_expm_multiply_interval_status_1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpmActionInterval.test_expm_multiply_interval_status_1.__dict__.__setitem__('stypy_function_name', 'TestExpmActionInterval.test_expm_multiply_interval_status_1')
        TestExpmActionInterval.test_expm_multiply_interval_status_1.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpmActionInterval.test_expm_multiply_interval_status_1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpmActionInterval.test_expm_multiply_interval_status_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpmActionInterval.test_expm_multiply_interval_status_1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpmActionInterval.test_expm_multiply_interval_status_1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpmActionInterval.test_expm_multiply_interval_status_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpmActionInterval.test_expm_multiply_interval_status_1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpmActionInterval.test_expm_multiply_interval_status_1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_expm_multiply_interval_status_1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_expm_multiply_interval_status_1(...)' code ##################

        
        # Call to _help_test_specific_expm_interval_status(...): (line 220)
        # Processing the call arguments (line 220)
        int_423395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 54), 'int')
        # Processing the call keyword arguments (line 220)
        kwargs_423396 = {}
        # Getting the type of 'self' (line 220)
        self_423393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'self', False)
        # Obtaining the member '_help_test_specific_expm_interval_status' of a type (line 220)
        _help_test_specific_expm_interval_status_423394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 8), self_423393, '_help_test_specific_expm_interval_status')
        # Calling _help_test_specific_expm_interval_status(args, kwargs) (line 220)
        _help_test_specific_expm_interval_status_call_result_423397 = invoke(stypy.reporting.localization.Localization(__file__, 220, 8), _help_test_specific_expm_interval_status_423394, *[int_423395], **kwargs_423396)
        
        
        # ################# End of 'test_expm_multiply_interval_status_1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_expm_multiply_interval_status_1' in the type store
        # Getting the type of 'stypy_return_type' (line 219)
        stypy_return_type_423398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_423398)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_expm_multiply_interval_status_1'
        return stypy_return_type_423398


    @norecursion
    def test_expm_multiply_interval_status_2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_expm_multiply_interval_status_2'
        module_type_store = module_type_store.open_function_context('test_expm_multiply_interval_status_2', 222, 4, False)
        # Assigning a type to the variable 'self' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpmActionInterval.test_expm_multiply_interval_status_2.__dict__.__setitem__('stypy_localization', localization)
        TestExpmActionInterval.test_expm_multiply_interval_status_2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpmActionInterval.test_expm_multiply_interval_status_2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpmActionInterval.test_expm_multiply_interval_status_2.__dict__.__setitem__('stypy_function_name', 'TestExpmActionInterval.test_expm_multiply_interval_status_2')
        TestExpmActionInterval.test_expm_multiply_interval_status_2.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpmActionInterval.test_expm_multiply_interval_status_2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpmActionInterval.test_expm_multiply_interval_status_2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpmActionInterval.test_expm_multiply_interval_status_2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpmActionInterval.test_expm_multiply_interval_status_2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpmActionInterval.test_expm_multiply_interval_status_2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpmActionInterval.test_expm_multiply_interval_status_2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpmActionInterval.test_expm_multiply_interval_status_2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_expm_multiply_interval_status_2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_expm_multiply_interval_status_2(...)' code ##################

        
        # Call to _help_test_specific_expm_interval_status(...): (line 223)
        # Processing the call arguments (line 223)
        int_423401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 54), 'int')
        # Processing the call keyword arguments (line 223)
        kwargs_423402 = {}
        # Getting the type of 'self' (line 223)
        self_423399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'self', False)
        # Obtaining the member '_help_test_specific_expm_interval_status' of a type (line 223)
        _help_test_specific_expm_interval_status_423400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), self_423399, '_help_test_specific_expm_interval_status')
        # Calling _help_test_specific_expm_interval_status(args, kwargs) (line 223)
        _help_test_specific_expm_interval_status_call_result_423403 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), _help_test_specific_expm_interval_status_423400, *[int_423401], **kwargs_423402)
        
        
        # ################# End of 'test_expm_multiply_interval_status_2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_expm_multiply_interval_status_2' in the type store
        # Getting the type of 'stypy_return_type' (line 222)
        stypy_return_type_423404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_423404)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_expm_multiply_interval_status_2'
        return stypy_return_type_423404


    @norecursion
    def _help_test_specific_expm_interval_status(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_help_test_specific_expm_interval_status'
        module_type_store = module_type_store.open_function_context('_help_test_specific_expm_interval_status', 225, 4, False)
        # Assigning a type to the variable 'self' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpmActionInterval._help_test_specific_expm_interval_status.__dict__.__setitem__('stypy_localization', localization)
        TestExpmActionInterval._help_test_specific_expm_interval_status.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpmActionInterval._help_test_specific_expm_interval_status.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpmActionInterval._help_test_specific_expm_interval_status.__dict__.__setitem__('stypy_function_name', 'TestExpmActionInterval._help_test_specific_expm_interval_status')
        TestExpmActionInterval._help_test_specific_expm_interval_status.__dict__.__setitem__('stypy_param_names_list', ['target_status'])
        TestExpmActionInterval._help_test_specific_expm_interval_status.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpmActionInterval._help_test_specific_expm_interval_status.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpmActionInterval._help_test_specific_expm_interval_status.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpmActionInterval._help_test_specific_expm_interval_status.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpmActionInterval._help_test_specific_expm_interval_status.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpmActionInterval._help_test_specific_expm_interval_status.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpmActionInterval._help_test_specific_expm_interval_status', ['target_status'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_help_test_specific_expm_interval_status', localization, ['target_status'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_help_test_specific_expm_interval_status(...)' code ##################

        
        # Call to seed(...): (line 226)
        # Processing the call arguments (line 226)
        int_423408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 23), 'int')
        # Processing the call keyword arguments (line 226)
        kwargs_423409 = {}
        # Getting the type of 'np' (line 226)
        np_423405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 226)
        random_423406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), np_423405, 'random')
        # Obtaining the member 'seed' of a type (line 226)
        seed_423407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), random_423406, 'seed')
        # Calling seed(args, kwargs) (line 226)
        seed_call_result_423410 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), seed_423407, *[int_423408], **kwargs_423409)
        
        
        # Assigning a Num to a Name (line 227):
        
        # Assigning a Num to a Name (line 227):
        float_423411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 16), 'float')
        # Assigning a type to the variable 'start' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'start', float_423411)
        
        # Assigning a Num to a Name (line 228):
        
        # Assigning a Num to a Name (line 228):
        float_423412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 15), 'float')
        # Assigning a type to the variable 'stop' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'stop', float_423412)
        
        # Assigning a Num to a Name (line 229):
        
        # Assigning a Num to a Name (line 229):
        int_423413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 14), 'int')
        # Assigning a type to the variable 'num' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'num', int_423413)
        
        # Assigning a Name to a Name (line 230):
        
        # Assigning a Name to a Name (line 230):
        # Getting the type of 'True' (line 230)
        True_423414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 19), 'True')
        # Assigning a type to the variable 'endpoint' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'endpoint', True_423414)
        
        # Assigning a Num to a Name (line 231):
        
        # Assigning a Num to a Name (line 231):
        int_423415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 12), 'int')
        # Assigning a type to the variable 'n' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'n', int_423415)
        
        # Assigning a Num to a Name (line 232):
        
        # Assigning a Num to a Name (line 232):
        int_423416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 12), 'int')
        # Assigning a type to the variable 'k' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'k', int_423416)
        
        # Assigning a Num to a Name (line 233):
        
        # Assigning a Num to a Name (line 233):
        int_423417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 19), 'int')
        # Assigning a type to the variable 'nrepeats' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'nrepeats', int_423417)
        
        # Assigning a Num to a Name (line 234):
        
        # Assigning a Num to a Name (line 234):
        int_423418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 21), 'int')
        # Assigning a type to the variable 'nsuccesses' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'nsuccesses', int_423418)
        
        
        # Obtaining an instance of the builtin type 'list' (line 235)
        list_423419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 235)
        # Adding element type (line 235)
        int_423420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 19), list_423419, int_423420)
        # Adding element type (line 235)
        int_423421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 19), list_423419, int_423421)
        # Adding element type (line 235)
        int_423422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 19), list_423419, int_423422)
        
        # Getting the type of 'nrepeats' (line 235)
        nrepeats_423423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 33), 'nrepeats')
        # Applying the binary operator '*' (line 235)
        result_mul_423424 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 19), '*', list_423419, nrepeats_423423)
        
        # Testing the type of a for loop iterable (line 235)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 235, 8), result_mul_423424)
        # Getting the type of the for loop variable (line 235)
        for_loop_var_423425 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 235, 8), result_mul_423424)
        # Assigning a type to the variable 'num' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'num', for_loop_var_423425)
        # SSA begins for a for statement (line 235)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 236):
        
        # Assigning a Call to a Name (line 236):
        
        # Call to randn(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'n' (line 236)
        n_423429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 32), 'n', False)
        # Getting the type of 'n' (line 236)
        n_423430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 35), 'n', False)
        # Processing the call keyword arguments (line 236)
        kwargs_423431 = {}
        # Getting the type of 'np' (line 236)
        np_423426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 16), 'np', False)
        # Obtaining the member 'random' of a type (line 236)
        random_423427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 16), np_423426, 'random')
        # Obtaining the member 'randn' of a type (line 236)
        randn_423428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 16), random_423427, 'randn')
        # Calling randn(args, kwargs) (line 236)
        randn_call_result_423432 = invoke(stypy.reporting.localization.Localization(__file__, 236, 16), randn_423428, *[n_423429, n_423430], **kwargs_423431)
        
        # Assigning a type to the variable 'A' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'A', randn_call_result_423432)
        
        # Assigning a Call to a Name (line 237):
        
        # Assigning a Call to a Name (line 237):
        
        # Call to randn(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'n' (line 237)
        n_423436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 32), 'n', False)
        # Getting the type of 'k' (line 237)
        k_423437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 35), 'k', False)
        # Processing the call keyword arguments (line 237)
        kwargs_423438 = {}
        # Getting the type of 'np' (line 237)
        np_423433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 16), 'np', False)
        # Obtaining the member 'random' of a type (line 237)
        random_423434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 16), np_423433, 'random')
        # Obtaining the member 'randn' of a type (line 237)
        randn_423435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 16), random_423434, 'randn')
        # Calling randn(args, kwargs) (line 237)
        randn_call_result_423439 = invoke(stypy.reporting.localization.Localization(__file__, 237, 16), randn_423435, *[n_423436, k_423437], **kwargs_423438)
        
        # Assigning a type to the variable 'B' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'B', randn_call_result_423439)
        
        # Assigning a Call to a Name (line 238):
        
        # Assigning a Call to a Name (line 238):
        
        # Call to _expm_multiply_interval(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'A' (line 238)
        A_423441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 45), 'A', False)
        # Getting the type of 'B' (line 238)
        B_423442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 48), 'B', False)
        # Processing the call keyword arguments (line 238)
        # Getting the type of 'start' (line 239)
        start_423443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 26), 'start', False)
        keyword_423444 = start_423443
        # Getting the type of 'stop' (line 239)
        stop_423445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 38), 'stop', False)
        keyword_423446 = stop_423445
        # Getting the type of 'num' (line 239)
        num_423447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 48), 'num', False)
        keyword_423448 = num_423447
        # Getting the type of 'endpoint' (line 239)
        endpoint_423449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 62), 'endpoint', False)
        keyword_423450 = endpoint_423449
        # Getting the type of 'True' (line 240)
        True_423451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 32), 'True', False)
        keyword_423452 = True_423451
        kwargs_423453 = {'status_only': keyword_423452, 'start': keyword_423444, 'num': keyword_423448, 'stop': keyword_423446, 'endpoint': keyword_423450}
        # Getting the type of '_expm_multiply_interval' (line 238)
        _expm_multiply_interval_423440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 21), '_expm_multiply_interval', False)
        # Calling _expm_multiply_interval(args, kwargs) (line 238)
        _expm_multiply_interval_call_result_423454 = invoke(stypy.reporting.localization.Localization(__file__, 238, 21), _expm_multiply_interval_423440, *[A_423441, B_423442], **kwargs_423453)
        
        # Assigning a type to the variable 'status' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'status', _expm_multiply_interval_call_result_423454)
        
        
        # Getting the type of 'status' (line 241)
        status_423455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 15), 'status')
        # Getting the type of 'target_status' (line 241)
        target_status_423456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 25), 'target_status')
        # Applying the binary operator '==' (line 241)
        result_eq_423457 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 15), '==', status_423455, target_status_423456)
        
        # Testing the type of an if condition (line 241)
        if_condition_423458 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 12), result_eq_423457)
        # Assigning a type to the variable 'if_condition_423458' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'if_condition_423458', if_condition_423458)
        # SSA begins for if statement (line 241)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 242):
        
        # Assigning a Subscript to a Name (line 242):
        
        # Obtaining the type of the subscript
        int_423459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 16), 'int')
        
        # Call to _expm_multiply_interval(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'A' (line 242)
        A_423461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 52), 'A', False)
        # Getting the type of 'B' (line 242)
        B_423462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 55), 'B', False)
        # Processing the call keyword arguments (line 242)
        # Getting the type of 'start' (line 243)
        start_423463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 30), 'start', False)
        keyword_423464 = start_423463
        # Getting the type of 'stop' (line 243)
        stop_423465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 42), 'stop', False)
        keyword_423466 = stop_423465
        # Getting the type of 'num' (line 243)
        num_423467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 52), 'num', False)
        keyword_423468 = num_423467
        # Getting the type of 'endpoint' (line 243)
        endpoint_423469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 66), 'endpoint', False)
        keyword_423470 = endpoint_423469
        # Getting the type of 'False' (line 244)
        False_423471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 36), 'False', False)
        keyword_423472 = False_423471
        kwargs_423473 = {'status_only': keyword_423472, 'start': keyword_423464, 'num': keyword_423468, 'stop': keyword_423466, 'endpoint': keyword_423470}
        # Getting the type of '_expm_multiply_interval' (line 242)
        _expm_multiply_interval_423460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 28), '_expm_multiply_interval', False)
        # Calling _expm_multiply_interval(args, kwargs) (line 242)
        _expm_multiply_interval_call_result_423474 = invoke(stypy.reporting.localization.Localization(__file__, 242, 28), _expm_multiply_interval_423460, *[A_423461, B_423462], **kwargs_423473)
        
        # Obtaining the member '__getitem__' of a type (line 242)
        getitem___423475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 16), _expm_multiply_interval_call_result_423474, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 242)
        subscript_call_result_423476 = invoke(stypy.reporting.localization.Localization(__file__, 242, 16), getitem___423475, int_423459)
        
        # Assigning a type to the variable 'tuple_var_assignment_422374' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'tuple_var_assignment_422374', subscript_call_result_423476)
        
        # Assigning a Subscript to a Name (line 242):
        
        # Obtaining the type of the subscript
        int_423477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 16), 'int')
        
        # Call to _expm_multiply_interval(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'A' (line 242)
        A_423479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 52), 'A', False)
        # Getting the type of 'B' (line 242)
        B_423480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 55), 'B', False)
        # Processing the call keyword arguments (line 242)
        # Getting the type of 'start' (line 243)
        start_423481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 30), 'start', False)
        keyword_423482 = start_423481
        # Getting the type of 'stop' (line 243)
        stop_423483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 42), 'stop', False)
        keyword_423484 = stop_423483
        # Getting the type of 'num' (line 243)
        num_423485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 52), 'num', False)
        keyword_423486 = num_423485
        # Getting the type of 'endpoint' (line 243)
        endpoint_423487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 66), 'endpoint', False)
        keyword_423488 = endpoint_423487
        # Getting the type of 'False' (line 244)
        False_423489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 36), 'False', False)
        keyword_423490 = False_423489
        kwargs_423491 = {'status_only': keyword_423490, 'start': keyword_423482, 'num': keyword_423486, 'stop': keyword_423484, 'endpoint': keyword_423488}
        # Getting the type of '_expm_multiply_interval' (line 242)
        _expm_multiply_interval_423478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 28), '_expm_multiply_interval', False)
        # Calling _expm_multiply_interval(args, kwargs) (line 242)
        _expm_multiply_interval_call_result_423492 = invoke(stypy.reporting.localization.Localization(__file__, 242, 28), _expm_multiply_interval_423478, *[A_423479, B_423480], **kwargs_423491)
        
        # Obtaining the member '__getitem__' of a type (line 242)
        getitem___423493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 16), _expm_multiply_interval_call_result_423492, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 242)
        subscript_call_result_423494 = invoke(stypy.reporting.localization.Localization(__file__, 242, 16), getitem___423493, int_423477)
        
        # Assigning a type to the variable 'tuple_var_assignment_422375' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'tuple_var_assignment_422375', subscript_call_result_423494)
        
        # Assigning a Name to a Name (line 242):
        # Getting the type of 'tuple_var_assignment_422374' (line 242)
        tuple_var_assignment_422374_423495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'tuple_var_assignment_422374')
        # Assigning a type to the variable 'X' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'X', tuple_var_assignment_422374_423495)
        
        # Assigning a Name to a Name (line 242):
        # Getting the type of 'tuple_var_assignment_422375' (line 242)
        tuple_var_assignment_422375_423496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'tuple_var_assignment_422375')
        # Assigning a type to the variable 'status' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 19), 'status', tuple_var_assignment_422375_423496)
        
        # Call to assert_equal(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'X' (line 245)
        X_423498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 29), 'X', False)
        # Obtaining the member 'shape' of a type (line 245)
        shape_423499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 29), X_423498, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 245)
        tuple_423500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 245)
        # Adding element type (line 245)
        # Getting the type of 'num' (line 245)
        num_423501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 39), 'num', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 39), tuple_423500, num_423501)
        # Adding element type (line 245)
        # Getting the type of 'n' (line 245)
        n_423502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 44), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 39), tuple_423500, n_423502)
        # Adding element type (line 245)
        # Getting the type of 'k' (line 245)
        k_423503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 47), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 39), tuple_423500, k_423503)
        
        # Processing the call keyword arguments (line 245)
        kwargs_423504 = {}
        # Getting the type of 'assert_equal' (line 245)
        assert_equal_423497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 16), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 245)
        assert_equal_call_result_423505 = invoke(stypy.reporting.localization.Localization(__file__, 245, 16), assert_equal_423497, *[shape_423499, tuple_423500], **kwargs_423504)
        
        
        # Assigning a Call to a Name (line 246):
        
        # Assigning a Call to a Name (line 246):
        
        # Call to linspace(...): (line 246)
        # Processing the call keyword arguments (line 246)
        # Getting the type of 'start' (line 246)
        start_423508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 44), 'start', False)
        keyword_423509 = start_423508
        # Getting the type of 'stop' (line 246)
        stop_423510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 56), 'stop', False)
        keyword_423511 = stop_423510
        # Getting the type of 'num' (line 247)
        num_423512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 28), 'num', False)
        keyword_423513 = num_423512
        # Getting the type of 'endpoint' (line 247)
        endpoint_423514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 42), 'endpoint', False)
        keyword_423515 = endpoint_423514
        kwargs_423516 = {'start': keyword_423509, 'num': keyword_423513, 'stop': keyword_423511, 'endpoint': keyword_423515}
        # Getting the type of 'np' (line 246)
        np_423506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 26), 'np', False)
        # Obtaining the member 'linspace' of a type (line 246)
        linspace_423507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 26), np_423506, 'linspace')
        # Calling linspace(args, kwargs) (line 246)
        linspace_call_result_423517 = invoke(stypy.reporting.localization.Localization(__file__, 246, 26), linspace_423507, *[], **kwargs_423516)
        
        # Assigning a type to the variable 'samples' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'samples', linspace_call_result_423517)
        
        
        # Call to zip(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'X' (line 248)
        X_423519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 39), 'X', False)
        # Getting the type of 'samples' (line 248)
        samples_423520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 42), 'samples', False)
        # Processing the call keyword arguments (line 248)
        kwargs_423521 = {}
        # Getting the type of 'zip' (line 248)
        zip_423518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 35), 'zip', False)
        # Calling zip(args, kwargs) (line 248)
        zip_call_result_423522 = invoke(stypy.reporting.localization.Localization(__file__, 248, 35), zip_423518, *[X_423519, samples_423520], **kwargs_423521)
        
        # Testing the type of a for loop iterable (line 248)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 248, 16), zip_call_result_423522)
        # Getting the type of the for loop variable (line 248)
        for_loop_var_423523 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 248, 16), zip_call_result_423522)
        # Assigning a type to the variable 'solution' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'solution', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 16), for_loop_var_423523))
        # Assigning a type to the variable 't' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 't', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 16), for_loop_var_423523))
        # SSA begins for a for statement (line 248)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_allclose(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 'solution' (line 249)
        solution_423525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 36), 'solution', False)
        
        # Call to dot(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 'B' (line 249)
        B_423535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 73), 'B', False)
        # Processing the call keyword arguments (line 249)
        kwargs_423536 = {}
        
        # Call to expm(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 't' (line 249)
        t_423529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 64), 't', False)
        # Getting the type of 'A' (line 249)
        A_423530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 66), 'A', False)
        # Applying the binary operator '*' (line 249)
        result_mul_423531 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 64), '*', t_423529, A_423530)
        
        # Processing the call keyword arguments (line 249)
        kwargs_423532 = {}
        # Getting the type of 'scipy' (line 249)
        scipy_423526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 46), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 249)
        linalg_423527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 46), scipy_423526, 'linalg')
        # Obtaining the member 'expm' of a type (line 249)
        expm_423528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 46), linalg_423527, 'expm')
        # Calling expm(args, kwargs) (line 249)
        expm_call_result_423533 = invoke(stypy.reporting.localization.Localization(__file__, 249, 46), expm_423528, *[result_mul_423531], **kwargs_423532)
        
        # Obtaining the member 'dot' of a type (line 249)
        dot_423534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 46), expm_call_result_423533, 'dot')
        # Calling dot(args, kwargs) (line 249)
        dot_call_result_423537 = invoke(stypy.reporting.localization.Localization(__file__, 249, 46), dot_423534, *[B_423535], **kwargs_423536)
        
        # Processing the call keyword arguments (line 249)
        kwargs_423538 = {}
        # Getting the type of 'assert_allclose' (line 249)
        assert_allclose_423524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 20), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 249)
        assert_allclose_call_result_423539 = invoke(stypy.reporting.localization.Localization(__file__, 249, 20), assert_allclose_423524, *[solution_423525, dot_call_result_423537], **kwargs_423538)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'nsuccesses' (line 250)
        nsuccesses_423540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'nsuccesses')
        int_423541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 30), 'int')
        # Applying the binary operator '+=' (line 250)
        result_iadd_423542 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 16), '+=', nsuccesses_423540, int_423541)
        # Assigning a type to the variable 'nsuccesses' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'nsuccesses', result_iadd_423542)
        
        # SSA join for if statement (line 241)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'nsuccesses' (line 251)
        nsuccesses_423543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 15), 'nsuccesses')
        # Applying the 'not' unary operator (line 251)
        result_not__423544 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 11), 'not', nsuccesses_423543)
        
        # Testing the type of an if condition (line 251)
        if_condition_423545 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 251, 8), result_not__423544)
        # Assigning a type to the variable 'if_condition_423545' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'if_condition_423545', if_condition_423545)
        # SSA begins for if statement (line 251)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 252):
        
        # Assigning a BinOp to a Name (line 252):
        str_423546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 18), 'str', 'failed to find a status-')
        
        # Call to str(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'target_status' (line 252)
        target_status_423548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 51), 'target_status', False)
        # Processing the call keyword arguments (line 252)
        kwargs_423549 = {}
        # Getting the type of 'str' (line 252)
        str_423547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 47), 'str', False)
        # Calling str(args, kwargs) (line 252)
        str_call_result_423550 = invoke(stypy.reporting.localization.Localization(__file__, 252, 47), str_423547, *[target_status_423548], **kwargs_423549)
        
        # Applying the binary operator '+' (line 252)
        result_add_423551 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 18), '+', str_423546, str_call_result_423550)
        
        str_423552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 68), 'str', ' interval')
        # Applying the binary operator '+' (line 252)
        result_add_423553 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 66), '+', result_add_423551, str_423552)
        
        # Assigning a type to the variable 'msg' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'msg', result_add_423553)
        
        # Call to Exception(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'msg' (line 253)
        msg_423555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 28), 'msg', False)
        # Processing the call keyword arguments (line 253)
        kwargs_423556 = {}
        # Getting the type of 'Exception' (line 253)
        Exception_423554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 18), 'Exception', False)
        # Calling Exception(args, kwargs) (line 253)
        Exception_call_result_423557 = invoke(stypy.reporting.localization.Localization(__file__, 253, 18), Exception_423554, *[msg_423555], **kwargs_423556)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 253, 12), Exception_call_result_423557, 'raise parameter', BaseException)
        # SSA join for if statement (line 251)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_help_test_specific_expm_interval_status(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_help_test_specific_expm_interval_status' in the type store
        # Getting the type of 'stypy_return_type' (line 225)
        stypy_return_type_423558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_423558)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_help_test_specific_expm_interval_status'
        return stypy_return_type_423558


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 137, 0, False)
        # Assigning a type to the variable 'self' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpmActionInterval.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestExpmActionInterval' (line 137)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 0), 'TestExpmActionInterval', TestExpmActionInterval)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
