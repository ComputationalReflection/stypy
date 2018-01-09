
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Test functions for the sparse.linalg.interface module
2: '''
3: 
4: from __future__ import division, print_function, absolute_import
5: 
6: from functools import partial
7: from itertools import product
8: import operator
9: import pytest
10: from pytest import raises as assert_raises
11: from numpy.testing import assert_, assert_equal
12: 
13: import numpy as np
14: import scipy.sparse as sparse
15: 
16: from scipy.sparse.linalg import interface
17: 
18: 
19: # Only test matmul operator (A @ B) when available (Python 3.5+)
20: TEST_MATMUL = hasattr(operator, 'matmul')
21: 
22: 
23: class TestLinearOperator(object):
24:     def setup_method(self):
25:         self.A = np.array([[1,2,3],
26:                            [4,5,6]])
27:         self.B = np.array([[1,2],
28:                            [3,4],
29:                            [5,6]])
30:         self.C = np.array([[1,2],
31:                            [3,4]])
32: 
33:     def test_matvec(self):
34:         def get_matvecs(A):
35:             return [{
36:                         'shape': A.shape,
37:                         'matvec': lambda x: np.dot(A, x).reshape(A.shape[0]),
38:                         'rmatvec': lambda x: np.dot(A.T.conj(),
39:                                                     x).reshape(A.shape[1])
40:                     },
41:                     {
42:                         'shape': A.shape,
43:                         'matvec': lambda x: np.dot(A, x),
44:                         'rmatvec': lambda x: np.dot(A.T.conj(), x),
45:                         'matmat': lambda x: np.dot(A, x)
46:                     }]
47: 
48:         for matvecs in get_matvecs(self.A):
49:             A = interface.LinearOperator(**matvecs)
50: 
51:             assert_(A.args == ())
52: 
53:             assert_equal(A.matvec(np.array([1,2,3])), [14,32])
54:             assert_equal(A.matvec(np.array([[1],[2],[3]])), [[14],[32]])
55:             assert_equal(A * np.array([1,2,3]), [14,32])
56:             assert_equal(A * np.array([[1],[2],[3]]), [[14],[32]])
57:             assert_equal(A.dot(np.array([1,2,3])), [14,32])
58:             assert_equal(A.dot(np.array([[1],[2],[3]])), [[14],[32]])
59: 
60:             assert_equal(A.matvec(np.matrix([[1],[2],[3]])), [[14],[32]])
61:             assert_equal(A * np.matrix([[1],[2],[3]]), [[14],[32]])
62:             assert_equal(A.dot(np.matrix([[1],[2],[3]])), [[14],[32]])
63: 
64:             assert_equal((2*A)*[1,1,1], [12,30])
65:             assert_equal((2*A).rmatvec([1,1]), [10, 14, 18])
66:             assert_equal((2*A).H.matvec([1,1]), [10, 14, 18])
67:             assert_equal((2*A)*[[1],[1],[1]], [[12],[30]])
68:             assert_equal((2*A).matmat([[1],[1],[1]]), [[12],[30]])
69:             assert_equal((A*2)*[1,1,1], [12,30])
70:             assert_equal((A*2)*[[1],[1],[1]], [[12],[30]])
71:             assert_equal((2j*A)*[1,1,1], [12j,30j])
72:             assert_equal((A+A)*[1,1,1], [12, 30])
73:             assert_equal((A+A).rmatvec([1,1]), [10, 14, 18])
74:             assert_equal((A+A).H.matvec([1,1]), [10, 14, 18])
75:             assert_equal((A+A)*[[1],[1],[1]], [[12], [30]])
76:             assert_equal((A+A).matmat([[1],[1],[1]]), [[12], [30]])
77:             assert_equal((-A)*[1,1,1], [-6,-15])
78:             assert_equal((-A)*[[1],[1],[1]], [[-6],[-15]])
79:             assert_equal((A-A)*[1,1,1], [0,0])
80:             assert_equal((A-A)*[[1],[1],[1]], [[0],[0]])
81: 
82:             z = A+A
83:             assert_(len(z.args) == 2 and z.args[0] is A and z.args[1] is A)
84:             z = 2*A
85:             assert_(len(z.args) == 2 and z.args[0] is A and z.args[1] == 2)
86: 
87:             assert_(isinstance(A.matvec([1, 2, 3]), np.ndarray))
88:             assert_(isinstance(A.matvec(np.array([[1],[2],[3]])), np.ndarray))
89:             assert_(isinstance(A * np.array([1,2,3]), np.ndarray))
90:             assert_(isinstance(A * np.array([[1],[2],[3]]), np.ndarray))
91:             assert_(isinstance(A.dot(np.array([1,2,3])), np.ndarray))
92:             assert_(isinstance(A.dot(np.array([[1],[2],[3]])), np.ndarray))
93: 
94:             assert_(isinstance(A.matvec(np.matrix([[1],[2],[3]])), np.ndarray))
95:             assert_(isinstance(A * np.matrix([[1],[2],[3]]), np.ndarray))
96:             assert_(isinstance(A.dot(np.matrix([[1],[2],[3]])), np.ndarray))
97: 
98:             assert_(isinstance(2*A, interface._ScaledLinearOperator))
99:             assert_(isinstance(2j*A, interface._ScaledLinearOperator))
100:             assert_(isinstance(A+A, interface._SumLinearOperator))
101:             assert_(isinstance(-A, interface._ScaledLinearOperator))
102:             assert_(isinstance(A-A, interface._SumLinearOperator))
103: 
104:             assert_((2j*A).dtype == np.complex_)
105: 
106:             assert_raises(ValueError, A.matvec, np.array([1,2]))
107:             assert_raises(ValueError, A.matvec, np.array([1,2,3,4]))
108:             assert_raises(ValueError, A.matvec, np.array([[1],[2]]))
109:             assert_raises(ValueError, A.matvec, np.array([[1],[2],[3],[4]]))
110: 
111:             assert_raises(ValueError, lambda: A*A)
112:             assert_raises(ValueError, lambda: A**2)
113: 
114:         for matvecsA, matvecsB in product(get_matvecs(self.A),
115:                                           get_matvecs(self.B)):
116:             A = interface.LinearOperator(**matvecsA)
117:             B = interface.LinearOperator(**matvecsB)
118: 
119:             assert_equal((A*B)*[1,1], [50,113])
120:             assert_equal((A*B)*[[1],[1]], [[50],[113]])
121:             assert_equal((A*B).matmat([[1],[1]]), [[50],[113]])
122: 
123:             assert_equal((A*B).rmatvec([1,1]), [71,92])
124:             assert_equal((A*B).H.matvec([1,1]), [71,92])
125: 
126:             assert_(isinstance(A*B, interface._ProductLinearOperator))
127: 
128:             assert_raises(ValueError, lambda: A+B)
129:             assert_raises(ValueError, lambda: A**2)
130: 
131:             z = A*B
132:             assert_(len(z.args) == 2 and z.args[0] is A and z.args[1] is B)
133: 
134:         for matvecsC in get_matvecs(self.C):
135:             C = interface.LinearOperator(**matvecsC)
136: 
137:             assert_equal((C**2)*[1,1], [17,37])
138:             assert_equal((C**2).rmatvec([1,1]), [22,32])
139:             assert_equal((C**2).H.matvec([1,1]), [22,32])
140:             assert_equal((C**2).matmat([[1],[1]]), [[17],[37]])
141: 
142:             assert_(isinstance(C**2, interface._PowerLinearOperator))
143: 
144:     def test_matmul(self):
145:         if not TEST_MATMUL:
146:             pytest.skip("matmul is only tested in Python 3.5+")
147: 
148:         D = {'shape': self.A.shape,
149:              'matvec': lambda x: np.dot(self.A, x).reshape(self.A.shape[0]),
150:              'rmatvec': lambda x: np.dot(self.A.T.conj(),
151:                                          x).reshape(self.A.shape[1]),
152:              'matmat': lambda x: np.dot(self.A, x)}
153:         A = interface.LinearOperator(**D)
154:         B = np.array([[1, 2, 3],
155:                       [4, 5, 6],
156:                       [7, 8, 9]])
157:         b = B[0]
158: 
159:         assert_equal(operator.matmul(A, b), A * b)
160:         assert_equal(operator.matmul(A, B), A * B)
161:         assert_raises(ValueError, operator.matmul, A, 2)
162:         assert_raises(ValueError, operator.matmul, 2, A)
163: 
164: 
165: class TestAsLinearOperator(object):
166:     def setup_method(self):
167:         self.cases = []
168: 
169:         def make_cases(dtype):
170:             self.cases.append(np.matrix([[1,2,3],[4,5,6]], dtype=dtype))
171:             self.cases.append(np.array([[1,2,3],[4,5,6]], dtype=dtype))
172:             self.cases.append(sparse.csr_matrix([[1,2,3],[4,5,6]], dtype=dtype))
173: 
174:             # Test default implementations of _adjoint and _rmatvec, which
175:             # refer to each other.
176:             def mv(x, dtype):
177:                 y = np.array([1 * x[0] + 2 * x[1] + 3 * x[2],
178:                               4 * x[0] + 5 * x[1] + 6 * x[2]], dtype=dtype)
179:                 if len(x.shape) == 2:
180:                     y = y.reshape(-1, 1)
181:                 return y
182: 
183:             def rmv(x, dtype):
184:                 return np.array([1 * x[0] + 4 * x[1],
185:                                  2 * x[0] + 5 * x[1],
186:                                  3 * x[0] + 6 * x[1]], dtype=dtype)
187: 
188:             class BaseMatlike(interface.LinearOperator):
189:                 def __init__(self, dtype):
190:                     self.dtype = np.dtype(dtype)
191:                     self.shape = (2,3)
192: 
193:                 def _matvec(self, x):
194:                     return mv(x, self.dtype)
195: 
196:             class HasRmatvec(BaseMatlike):
197:                 def _rmatvec(self,x):
198:                     return rmv(x, self.dtype)
199: 
200:             class HasAdjoint(BaseMatlike):
201:                 def _adjoint(self):
202:                     shape = self.shape[1], self.shape[0]
203:                     matvec = partial(rmv, dtype=self.dtype)
204:                     rmatvec = partial(mv, dtype=self.dtype)
205:                     return interface.LinearOperator(matvec=matvec,
206:                                                     rmatvec=rmatvec,
207:                                                     dtype=self.dtype,
208:                                                     shape=shape)
209: 
210:             self.cases.append(HasRmatvec(dtype))
211:             self.cases.append(HasAdjoint(dtype))
212: 
213:         make_cases('int32')
214:         make_cases('float32')
215:         make_cases('float64')
216: 
217:     def test_basic(self):
218: 
219:         for M in self.cases:
220:             A = interface.aslinearoperator(M)
221:             M,N = A.shape
222: 
223:             assert_equal(A.matvec(np.array([1,2,3])), [14,32])
224:             assert_equal(A.matvec(np.array([[1],[2],[3]])), [[14],[32]])
225: 
226:             assert_equal(A * np.array([1,2,3]), [14,32])
227:             assert_equal(A * np.array([[1],[2],[3]]), [[14],[32]])
228: 
229:             assert_equal(A.rmatvec(np.array([1,2])), [9,12,15])
230:             assert_equal(A.rmatvec(np.array([[1],[2]])), [[9],[12],[15]])
231:             assert_equal(A.H.matvec(np.array([1,2])), [9,12,15])
232:             assert_equal(A.H.matvec(np.array([[1],[2]])), [[9],[12],[15]])
233: 
234:             assert_equal(
235:                     A.matmat(np.array([[1,4],[2,5],[3,6]])),
236:                     [[14,32],[32,77]])
237: 
238:             assert_equal(A * np.array([[1,4],[2,5],[3,6]]), [[14,32],[32,77]])
239: 
240:             if hasattr(M,'dtype'):
241:                 assert_equal(A.dtype, M.dtype)
242: 
243:     def test_dot(self):
244: 
245:         for M in self.cases:
246:             A = interface.aslinearoperator(M)
247:             M,N = A.shape
248: 
249:             assert_equal(A.dot(np.array([1,2,3])), [14,32])
250:             assert_equal(A.dot(np.array([[1],[2],[3]])), [[14],[32]])
251: 
252:             assert_equal(
253:                     A.dot(np.array([[1,4],[2,5],[3,6]])),
254:                     [[14,32],[32,77]])
255: 
256: 
257: def test_repr():
258:     A = interface.LinearOperator(shape=(1, 1), matvec=lambda x: 1)
259:     repr_A = repr(A)
260:     assert_('unspecified dtype' not in repr_A, repr_A)
261: 
262: 
263: def test_identity():
264:     ident = interface.IdentityOperator((3, 3))
265:     assert_equal(ident * [1, 2, 3], [1, 2, 3])
266:     assert_equal(ident.dot(np.arange(9).reshape(3, 3)).ravel(), np.arange(9))
267: 
268:     assert_raises(ValueError, ident.matvec, [1, 2, 3, 4])
269: 
270: 
271: def test_attributes():
272:     A = interface.aslinearoperator(np.arange(16).reshape(4, 4))
273: 
274:     def always_four_ones(x):
275:         x = np.asarray(x)
276:         assert_(x.shape == (3,) or x.shape == (3, 1))
277:         return np.ones(4)
278: 
279:     B = interface.LinearOperator(shape=(4, 3), matvec=always_four_ones)
280: 
281:     for op in [A, B, A * B, A.H, A + A, B + B, A ** 4]:
282:         assert_(hasattr(op, "dtype"))
283:         assert_(hasattr(op, "shape"))
284:         assert_(hasattr(op, "_matvec"))
285: 
286: def matvec(x):
287:     ''' Needed for test_pickle as local functions are not pickleable '''
288:     return np.zeros(3)
289: 
290: def test_pickle():
291:     import pickle
292: 
293:     for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
294:         A = interface.LinearOperator((3, 3), matvec)
295:         s = pickle.dumps(A, protocol=protocol)
296:         B = pickle.loads(s)
297: 
298:         for k in A.__dict__:
299:             assert_equal(getattr(A, k), getattr(B, k))
300: 
301: def test_inheritance():
302:     class Empty(interface.LinearOperator):
303:         pass
304: 
305:     assert_raises(TypeError, Empty)
306: 
307:     class Identity(interface.LinearOperator):
308:         def __init__(self, n):
309:             super(Identity, self).__init__(dtype=None, shape=(n, n))
310: 
311:         def _matvec(self, x):
312:             return x
313: 
314:     id3 = Identity(3)
315:     assert_equal(id3.matvec([1, 2, 3]), [1, 2, 3])
316:     assert_raises(NotImplementedError, id3.rmatvec, [4, 5, 6])
317: 
318:     class MatmatOnly(interface.LinearOperator):
319:         def __init__(self, A):
320:             super(MatmatOnly, self).__init__(A.dtype, A.shape)
321:             self.A = A
322: 
323:         def _matmat(self, x):
324:             return self.A.dot(x)
325: 
326:     mm = MatmatOnly(np.random.randn(5, 3))
327:     assert_equal(mm.matvec(np.random.randn(3)).shape, (5,))
328: 
329: def test_dtypes_of_operator_sum():
330:     # gh-6078
331: 
332:     mat_complex = np.random.rand(2,2) + 1j * np.random.rand(2,2)
333:     mat_real = np.random.rand(2,2)
334: 
335:     complex_operator = interface.aslinearoperator(mat_complex)
336:     real_operator = interface.aslinearoperator(mat_real)
337: 
338:     sum_complex = complex_operator + complex_operator
339:     sum_real = real_operator + real_operator
340: 
341:     assert_equal(sum_real.dtype, np.float64)
342:     assert_equal(sum_complex.dtype, np.complex128)
343: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_423563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', 'Test functions for the sparse.linalg.interface module\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from functools import partial' statement (line 6)
try:
    from functools import partial

except:
    partial = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'functools', None, module_type_store, ['partial'], [partial])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from itertools import product' statement (line 7)
try:
    from itertools import product

except:
    product = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'itertools', None, module_type_store, ['product'], [product])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import operator' statement (line 8)
import operator

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'operator', operator, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import pytest' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_423564 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest')

if (type(import_423564) is not StypyTypeError):

    if (import_423564 != 'pyd_module'):
        __import__(import_423564)
        sys_modules_423565 = sys.modules[import_423564]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', sys_modules_423565.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', import_423564)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from pytest import assert_raises' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_423566 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest')

if (type(import_423566) is not StypyTypeError):

    if (import_423566 != 'pyd_module'):
        __import__(import_423566)
        sys_modules_423567 = sys.modules[import_423566]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest', sys_modules_423567.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_423567, sys_modules_423567.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest', import_423566)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from numpy.testing import assert_, assert_equal' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_423568 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.testing')

if (type(import_423568) is not StypyTypeError):

    if (import_423568 != 'pyd_module'):
        __import__(import_423568)
        sys_modules_423569 = sys.modules[import_423568]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.testing', sys_modules_423569.module_type_store, module_type_store, ['assert_', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_423569, sys_modules_423569.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_equal'], [assert_, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.testing', import_423568)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import numpy' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_423570 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy')

if (type(import_423570) is not StypyTypeError):

    if (import_423570 != 'pyd_module'):
        __import__(import_423570)
        sys_modules_423571 = sys.modules[import_423570]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'np', sys_modules_423571.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy', import_423570)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import scipy.sparse' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_423572 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse')

if (type(import_423572) is not StypyTypeError):

    if (import_423572 != 'pyd_module'):
        __import__(import_423572)
        sys_modules_423573 = sys.modules[import_423572]
        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'sparse', sys_modules_423573.module_type_store, module_type_store)
    else:
        import scipy.sparse as sparse

        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'sparse', scipy.sparse, module_type_store)

else:
    # Assigning a type to the variable 'scipy.sparse' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse', import_423572)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy.sparse.linalg import interface' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_423574 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse.linalg')

if (type(import_423574) is not StypyTypeError):

    if (import_423574 != 'pyd_module'):
        __import__(import_423574)
        sys_modules_423575 = sys.modules[import_423574]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse.linalg', sys_modules_423575.module_type_store, module_type_store, ['interface'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_423575, sys_modules_423575.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import interface

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse.linalg', None, module_type_store, ['interface'], [interface])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse.linalg', import_423574)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')


# Assigning a Call to a Name (line 20):

# Assigning a Call to a Name (line 20):

# Call to hasattr(...): (line 20)
# Processing the call arguments (line 20)
# Getting the type of 'operator' (line 20)
operator_423577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 22), 'operator', False)
str_423578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 32), 'str', 'matmul')
# Processing the call keyword arguments (line 20)
kwargs_423579 = {}
# Getting the type of 'hasattr' (line 20)
hasattr_423576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 14), 'hasattr', False)
# Calling hasattr(args, kwargs) (line 20)
hasattr_call_result_423580 = invoke(stypy.reporting.localization.Localization(__file__, 20, 14), hasattr_423576, *[operator_423577, str_423578], **kwargs_423579)

# Assigning a type to the variable 'TEST_MATMUL' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'TEST_MATMUL', hasattr_call_result_423580)
# Declaration of the 'TestLinearOperator' class

class TestLinearOperator(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinearOperator.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestLinearOperator.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinearOperator.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinearOperator.setup_method.__dict__.__setitem__('stypy_function_name', 'TestLinearOperator.setup_method')
        TestLinearOperator.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinearOperator.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinearOperator.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinearOperator.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinearOperator.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinearOperator.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinearOperator.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinearOperator.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Attribute (line 25):
        
        # Assigning a Call to a Attribute (line 25):
        
        # Call to array(...): (line 25)
        # Processing the call arguments (line 25)
        
        # Obtaining an instance of the builtin type 'list' (line 25)
        list_423583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 25)
        # Adding element type (line 25)
        
        # Obtaining an instance of the builtin type 'list' (line 25)
        list_423584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 25)
        # Adding element type (line 25)
        int_423585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 27), list_423584, int_423585)
        # Adding element type (line 25)
        int_423586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 27), list_423584, int_423586)
        # Adding element type (line 25)
        int_423587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 27), list_423584, int_423587)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 26), list_423583, list_423584)
        # Adding element type (line 25)
        
        # Obtaining an instance of the builtin type 'list' (line 26)
        list_423588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 26)
        # Adding element type (line 26)
        int_423589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 27), list_423588, int_423589)
        # Adding element type (line 26)
        int_423590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 27), list_423588, int_423590)
        # Adding element type (line 26)
        int_423591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 27), list_423588, int_423591)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 26), list_423583, list_423588)
        
        # Processing the call keyword arguments (line 25)
        kwargs_423592 = {}
        # Getting the type of 'np' (line 25)
        np_423581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 25)
        array_423582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 17), np_423581, 'array')
        # Calling array(args, kwargs) (line 25)
        array_call_result_423593 = invoke(stypy.reporting.localization.Localization(__file__, 25, 17), array_423582, *[list_423583], **kwargs_423592)
        
        # Getting the type of 'self' (line 25)
        self_423594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self')
        # Setting the type of the member 'A' of a type (line 25)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_423594, 'A', array_call_result_423593)
        
        # Assigning a Call to a Attribute (line 27):
        
        # Assigning a Call to a Attribute (line 27):
        
        # Call to array(...): (line 27)
        # Processing the call arguments (line 27)
        
        # Obtaining an instance of the builtin type 'list' (line 27)
        list_423597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 27)
        # Adding element type (line 27)
        
        # Obtaining an instance of the builtin type 'list' (line 27)
        list_423598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 27)
        # Adding element type (line 27)
        int_423599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 27), list_423598, int_423599)
        # Adding element type (line 27)
        int_423600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 27), list_423598, int_423600)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 26), list_423597, list_423598)
        # Adding element type (line 27)
        
        # Obtaining an instance of the builtin type 'list' (line 28)
        list_423601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 28)
        # Adding element type (line 28)
        int_423602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 27), list_423601, int_423602)
        # Adding element type (line 28)
        int_423603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 27), list_423601, int_423603)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 26), list_423597, list_423601)
        # Adding element type (line 27)
        
        # Obtaining an instance of the builtin type 'list' (line 29)
        list_423604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 29)
        # Adding element type (line 29)
        int_423605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 27), list_423604, int_423605)
        # Adding element type (line 29)
        int_423606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 27), list_423604, int_423606)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 26), list_423597, list_423604)
        
        # Processing the call keyword arguments (line 27)
        kwargs_423607 = {}
        # Getting the type of 'np' (line 27)
        np_423595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 27)
        array_423596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 17), np_423595, 'array')
        # Calling array(args, kwargs) (line 27)
        array_call_result_423608 = invoke(stypy.reporting.localization.Localization(__file__, 27, 17), array_423596, *[list_423597], **kwargs_423607)
        
        # Getting the type of 'self' (line 27)
        self_423609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'self')
        # Setting the type of the member 'B' of a type (line 27)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), self_423609, 'B', array_call_result_423608)
        
        # Assigning a Call to a Attribute (line 30):
        
        # Assigning a Call to a Attribute (line 30):
        
        # Call to array(...): (line 30)
        # Processing the call arguments (line 30)
        
        # Obtaining an instance of the builtin type 'list' (line 30)
        list_423612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 30)
        # Adding element type (line 30)
        
        # Obtaining an instance of the builtin type 'list' (line 30)
        list_423613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 30)
        # Adding element type (line 30)
        int_423614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 27), list_423613, int_423614)
        # Adding element type (line 30)
        int_423615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 27), list_423613, int_423615)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 26), list_423612, list_423613)
        # Adding element type (line 30)
        
        # Obtaining an instance of the builtin type 'list' (line 31)
        list_423616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 31)
        # Adding element type (line 31)
        int_423617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 27), list_423616, int_423617)
        # Adding element type (line 31)
        int_423618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 27), list_423616, int_423618)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 26), list_423612, list_423616)
        
        # Processing the call keyword arguments (line 30)
        kwargs_423619 = {}
        # Getting the type of 'np' (line 30)
        np_423610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 30)
        array_423611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 17), np_423610, 'array')
        # Calling array(args, kwargs) (line 30)
        array_call_result_423620 = invoke(stypy.reporting.localization.Localization(__file__, 30, 17), array_423611, *[list_423612], **kwargs_423619)
        
        # Getting the type of 'self' (line 30)
        self_423621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        # Setting the type of the member 'C' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_423621, 'C', array_call_result_423620)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_423622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_423622)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_423622


    @norecursion
    def test_matvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_matvec'
        module_type_store = module_type_store.open_function_context('test_matvec', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinearOperator.test_matvec.__dict__.__setitem__('stypy_localization', localization)
        TestLinearOperator.test_matvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinearOperator.test_matvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinearOperator.test_matvec.__dict__.__setitem__('stypy_function_name', 'TestLinearOperator.test_matvec')
        TestLinearOperator.test_matvec.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinearOperator.test_matvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinearOperator.test_matvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinearOperator.test_matvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinearOperator.test_matvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinearOperator.test_matvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinearOperator.test_matvec.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinearOperator.test_matvec', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_matvec', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_matvec(...)' code ##################


        @norecursion
        def get_matvecs(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'get_matvecs'
            module_type_store = module_type_store.open_function_context('get_matvecs', 34, 8, False)
            
            # Passed parameters checking function
            get_matvecs.stypy_localization = localization
            get_matvecs.stypy_type_of_self = None
            get_matvecs.stypy_type_store = module_type_store
            get_matvecs.stypy_function_name = 'get_matvecs'
            get_matvecs.stypy_param_names_list = ['A']
            get_matvecs.stypy_varargs_param_name = None
            get_matvecs.stypy_kwargs_param_name = None
            get_matvecs.stypy_call_defaults = defaults
            get_matvecs.stypy_call_varargs = varargs
            get_matvecs.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'get_matvecs', ['A'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'get_matvecs', localization, ['A'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'get_matvecs(...)' code ##################

            
            # Obtaining an instance of the builtin type 'list' (line 35)
            list_423623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 19), 'list')
            # Adding type elements to the builtin type 'list' instance (line 35)
            # Adding element type (line 35)
            
            # Obtaining an instance of the builtin type 'dict' (line 35)
            dict_423624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 20), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 35)
            # Adding element type (key, value) (line 35)
            str_423625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 24), 'str', 'shape')
            # Getting the type of 'A' (line 36)
            A_423626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 33), 'A')
            # Obtaining the member 'shape' of a type (line 36)
            shape_423627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 33), A_423626, 'shape')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 20), dict_423624, (str_423625, shape_423627))
            # Adding element type (key, value) (line 35)
            str_423628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 24), 'str', 'matvec')

            @norecursion
            def _stypy_temp_lambda_226(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_226'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_226', 37, 34, True)
                # Passed parameters checking function
                _stypy_temp_lambda_226.stypy_localization = localization
                _stypy_temp_lambda_226.stypy_type_of_self = None
                _stypy_temp_lambda_226.stypy_type_store = module_type_store
                _stypy_temp_lambda_226.stypy_function_name = '_stypy_temp_lambda_226'
                _stypy_temp_lambda_226.stypy_param_names_list = ['x']
                _stypy_temp_lambda_226.stypy_varargs_param_name = None
                _stypy_temp_lambda_226.stypy_kwargs_param_name = None
                _stypy_temp_lambda_226.stypy_call_defaults = defaults
                _stypy_temp_lambda_226.stypy_call_varargs = varargs
                _stypy_temp_lambda_226.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_226', ['x'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_226', ['x'], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                # Call to reshape(...): (line 37)
                # Processing the call arguments (line 37)
                
                # Obtaining the type of the subscript
                int_423636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 73), 'int')
                # Getting the type of 'A' (line 37)
                A_423637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 65), 'A', False)
                # Obtaining the member 'shape' of a type (line 37)
                shape_423638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 65), A_423637, 'shape')
                # Obtaining the member '__getitem__' of a type (line 37)
                getitem___423639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 65), shape_423638, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 37)
                subscript_call_result_423640 = invoke(stypy.reporting.localization.Localization(__file__, 37, 65), getitem___423639, int_423636)
                
                # Processing the call keyword arguments (line 37)
                kwargs_423641 = {}
                
                # Call to dot(...): (line 37)
                # Processing the call arguments (line 37)
                # Getting the type of 'A' (line 37)
                A_423631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 51), 'A', False)
                # Getting the type of 'x' (line 37)
                x_423632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 54), 'x', False)
                # Processing the call keyword arguments (line 37)
                kwargs_423633 = {}
                # Getting the type of 'np' (line 37)
                np_423629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 44), 'np', False)
                # Obtaining the member 'dot' of a type (line 37)
                dot_423630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 44), np_423629, 'dot')
                # Calling dot(args, kwargs) (line 37)
                dot_call_result_423634 = invoke(stypy.reporting.localization.Localization(__file__, 37, 44), dot_423630, *[A_423631, x_423632], **kwargs_423633)
                
                # Obtaining the member 'reshape' of a type (line 37)
                reshape_423635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 44), dot_call_result_423634, 'reshape')
                # Calling reshape(args, kwargs) (line 37)
                reshape_call_result_423642 = invoke(stypy.reporting.localization.Localization(__file__, 37, 44), reshape_423635, *[subscript_call_result_423640], **kwargs_423641)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 37)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 34), 'stypy_return_type', reshape_call_result_423642)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_226' in the type store
                # Getting the type of 'stypy_return_type' (line 37)
                stypy_return_type_423643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 34), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_423643)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_226'
                return stypy_return_type_423643

            # Assigning a type to the variable '_stypy_temp_lambda_226' (line 37)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 34), '_stypy_temp_lambda_226', _stypy_temp_lambda_226)
            # Getting the type of '_stypy_temp_lambda_226' (line 37)
            _stypy_temp_lambda_226_423644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 34), '_stypy_temp_lambda_226')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 20), dict_423624, (str_423628, _stypy_temp_lambda_226_423644))
            # Adding element type (key, value) (line 35)
            str_423645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 24), 'str', 'rmatvec')

            @norecursion
            def _stypy_temp_lambda_227(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_227'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_227', 38, 35, True)
                # Passed parameters checking function
                _stypy_temp_lambda_227.stypy_localization = localization
                _stypy_temp_lambda_227.stypy_type_of_self = None
                _stypy_temp_lambda_227.stypy_type_store = module_type_store
                _stypy_temp_lambda_227.stypy_function_name = '_stypy_temp_lambda_227'
                _stypy_temp_lambda_227.stypy_param_names_list = ['x']
                _stypy_temp_lambda_227.stypy_varargs_param_name = None
                _stypy_temp_lambda_227.stypy_kwargs_param_name = None
                _stypy_temp_lambda_227.stypy_call_defaults = defaults
                _stypy_temp_lambda_227.stypy_call_varargs = varargs
                _stypy_temp_lambda_227.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_227', ['x'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_227', ['x'], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                # Call to reshape(...): (line 38)
                # Processing the call arguments (line 38)
                
                # Obtaining the type of the subscript
                int_423657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 71), 'int')
                # Getting the type of 'A' (line 39)
                A_423658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 63), 'A', False)
                # Obtaining the member 'shape' of a type (line 39)
                shape_423659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 63), A_423658, 'shape')
                # Obtaining the member '__getitem__' of a type (line 39)
                getitem___423660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 63), shape_423659, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 39)
                subscript_call_result_423661 = invoke(stypy.reporting.localization.Localization(__file__, 39, 63), getitem___423660, int_423657)
                
                # Processing the call keyword arguments (line 38)
                kwargs_423662 = {}
                
                # Call to dot(...): (line 38)
                # Processing the call arguments (line 38)
                
                # Call to conj(...): (line 38)
                # Processing the call keyword arguments (line 38)
                kwargs_423651 = {}
                # Getting the type of 'A' (line 38)
                A_423648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 52), 'A', False)
                # Obtaining the member 'T' of a type (line 38)
                T_423649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 52), A_423648, 'T')
                # Obtaining the member 'conj' of a type (line 38)
                conj_423650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 52), T_423649, 'conj')
                # Calling conj(args, kwargs) (line 38)
                conj_call_result_423652 = invoke(stypy.reporting.localization.Localization(__file__, 38, 52), conj_423650, *[], **kwargs_423651)
                
                # Getting the type of 'x' (line 39)
                x_423653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 52), 'x', False)
                # Processing the call keyword arguments (line 38)
                kwargs_423654 = {}
                # Getting the type of 'np' (line 38)
                np_423646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 45), 'np', False)
                # Obtaining the member 'dot' of a type (line 38)
                dot_423647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 45), np_423646, 'dot')
                # Calling dot(args, kwargs) (line 38)
                dot_call_result_423655 = invoke(stypy.reporting.localization.Localization(__file__, 38, 45), dot_423647, *[conj_call_result_423652, x_423653], **kwargs_423654)
                
                # Obtaining the member 'reshape' of a type (line 38)
                reshape_423656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 45), dot_call_result_423655, 'reshape')
                # Calling reshape(args, kwargs) (line 38)
                reshape_call_result_423663 = invoke(stypy.reporting.localization.Localization(__file__, 38, 45), reshape_423656, *[subscript_call_result_423661], **kwargs_423662)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 38)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 35), 'stypy_return_type', reshape_call_result_423663)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_227' in the type store
                # Getting the type of 'stypy_return_type' (line 38)
                stypy_return_type_423664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 35), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_423664)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_227'
                return stypy_return_type_423664

            # Assigning a type to the variable '_stypy_temp_lambda_227' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 35), '_stypy_temp_lambda_227', _stypy_temp_lambda_227)
            # Getting the type of '_stypy_temp_lambda_227' (line 38)
            _stypy_temp_lambda_227_423665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 35), '_stypy_temp_lambda_227')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 20), dict_423624, (str_423645, _stypy_temp_lambda_227_423665))
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 19), list_423623, dict_423624)
            # Adding element type (line 35)
            
            # Obtaining an instance of the builtin type 'dict' (line 41)
            dict_423666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 20), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 41)
            # Adding element type (key, value) (line 41)
            str_423667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 24), 'str', 'shape')
            # Getting the type of 'A' (line 42)
            A_423668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 33), 'A')
            # Obtaining the member 'shape' of a type (line 42)
            shape_423669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 33), A_423668, 'shape')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 20), dict_423666, (str_423667, shape_423669))
            # Adding element type (key, value) (line 41)
            str_423670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 24), 'str', 'matvec')

            @norecursion
            def _stypy_temp_lambda_228(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_228'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_228', 43, 34, True)
                # Passed parameters checking function
                _stypy_temp_lambda_228.stypy_localization = localization
                _stypy_temp_lambda_228.stypy_type_of_self = None
                _stypy_temp_lambda_228.stypy_type_store = module_type_store
                _stypy_temp_lambda_228.stypy_function_name = '_stypy_temp_lambda_228'
                _stypy_temp_lambda_228.stypy_param_names_list = ['x']
                _stypy_temp_lambda_228.stypy_varargs_param_name = None
                _stypy_temp_lambda_228.stypy_kwargs_param_name = None
                _stypy_temp_lambda_228.stypy_call_defaults = defaults
                _stypy_temp_lambda_228.stypy_call_varargs = varargs
                _stypy_temp_lambda_228.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_228', ['x'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_228', ['x'], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                # Call to dot(...): (line 43)
                # Processing the call arguments (line 43)
                # Getting the type of 'A' (line 43)
                A_423673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 51), 'A', False)
                # Getting the type of 'x' (line 43)
                x_423674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 54), 'x', False)
                # Processing the call keyword arguments (line 43)
                kwargs_423675 = {}
                # Getting the type of 'np' (line 43)
                np_423671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 44), 'np', False)
                # Obtaining the member 'dot' of a type (line 43)
                dot_423672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 44), np_423671, 'dot')
                # Calling dot(args, kwargs) (line 43)
                dot_call_result_423676 = invoke(stypy.reporting.localization.Localization(__file__, 43, 44), dot_423672, *[A_423673, x_423674], **kwargs_423675)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 43)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 34), 'stypy_return_type', dot_call_result_423676)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_228' in the type store
                # Getting the type of 'stypy_return_type' (line 43)
                stypy_return_type_423677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 34), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_423677)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_228'
                return stypy_return_type_423677

            # Assigning a type to the variable '_stypy_temp_lambda_228' (line 43)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 34), '_stypy_temp_lambda_228', _stypy_temp_lambda_228)
            # Getting the type of '_stypy_temp_lambda_228' (line 43)
            _stypy_temp_lambda_228_423678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 34), '_stypy_temp_lambda_228')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 20), dict_423666, (str_423670, _stypy_temp_lambda_228_423678))
            # Adding element type (key, value) (line 41)
            str_423679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 24), 'str', 'rmatvec')

            @norecursion
            def _stypy_temp_lambda_229(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_229'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_229', 44, 35, True)
                # Passed parameters checking function
                _stypy_temp_lambda_229.stypy_localization = localization
                _stypy_temp_lambda_229.stypy_type_of_self = None
                _stypy_temp_lambda_229.stypy_type_store = module_type_store
                _stypy_temp_lambda_229.stypy_function_name = '_stypy_temp_lambda_229'
                _stypy_temp_lambda_229.stypy_param_names_list = ['x']
                _stypy_temp_lambda_229.stypy_varargs_param_name = None
                _stypy_temp_lambda_229.stypy_kwargs_param_name = None
                _stypy_temp_lambda_229.stypy_call_defaults = defaults
                _stypy_temp_lambda_229.stypy_call_varargs = varargs
                _stypy_temp_lambda_229.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_229', ['x'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_229', ['x'], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                # Call to dot(...): (line 44)
                # Processing the call arguments (line 44)
                
                # Call to conj(...): (line 44)
                # Processing the call keyword arguments (line 44)
                kwargs_423685 = {}
                # Getting the type of 'A' (line 44)
                A_423682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 52), 'A', False)
                # Obtaining the member 'T' of a type (line 44)
                T_423683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 52), A_423682, 'T')
                # Obtaining the member 'conj' of a type (line 44)
                conj_423684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 52), T_423683, 'conj')
                # Calling conj(args, kwargs) (line 44)
                conj_call_result_423686 = invoke(stypy.reporting.localization.Localization(__file__, 44, 52), conj_423684, *[], **kwargs_423685)
                
                # Getting the type of 'x' (line 44)
                x_423687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 64), 'x', False)
                # Processing the call keyword arguments (line 44)
                kwargs_423688 = {}
                # Getting the type of 'np' (line 44)
                np_423680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 45), 'np', False)
                # Obtaining the member 'dot' of a type (line 44)
                dot_423681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 45), np_423680, 'dot')
                # Calling dot(args, kwargs) (line 44)
                dot_call_result_423689 = invoke(stypy.reporting.localization.Localization(__file__, 44, 45), dot_423681, *[conj_call_result_423686, x_423687], **kwargs_423688)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 44)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 35), 'stypy_return_type', dot_call_result_423689)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_229' in the type store
                # Getting the type of 'stypy_return_type' (line 44)
                stypy_return_type_423690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 35), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_423690)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_229'
                return stypy_return_type_423690

            # Assigning a type to the variable '_stypy_temp_lambda_229' (line 44)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 35), '_stypy_temp_lambda_229', _stypy_temp_lambda_229)
            # Getting the type of '_stypy_temp_lambda_229' (line 44)
            _stypy_temp_lambda_229_423691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 35), '_stypy_temp_lambda_229')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 20), dict_423666, (str_423679, _stypy_temp_lambda_229_423691))
            # Adding element type (key, value) (line 41)
            str_423692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 24), 'str', 'matmat')

            @norecursion
            def _stypy_temp_lambda_230(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_230'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_230', 45, 34, True)
                # Passed parameters checking function
                _stypy_temp_lambda_230.stypy_localization = localization
                _stypy_temp_lambda_230.stypy_type_of_self = None
                _stypy_temp_lambda_230.stypy_type_store = module_type_store
                _stypy_temp_lambda_230.stypy_function_name = '_stypy_temp_lambda_230'
                _stypy_temp_lambda_230.stypy_param_names_list = ['x']
                _stypy_temp_lambda_230.stypy_varargs_param_name = None
                _stypy_temp_lambda_230.stypy_kwargs_param_name = None
                _stypy_temp_lambda_230.stypy_call_defaults = defaults
                _stypy_temp_lambda_230.stypy_call_varargs = varargs
                _stypy_temp_lambda_230.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_230', ['x'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_230', ['x'], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                # Call to dot(...): (line 45)
                # Processing the call arguments (line 45)
                # Getting the type of 'A' (line 45)
                A_423695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 51), 'A', False)
                # Getting the type of 'x' (line 45)
                x_423696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 54), 'x', False)
                # Processing the call keyword arguments (line 45)
                kwargs_423697 = {}
                # Getting the type of 'np' (line 45)
                np_423693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 44), 'np', False)
                # Obtaining the member 'dot' of a type (line 45)
                dot_423694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 44), np_423693, 'dot')
                # Calling dot(args, kwargs) (line 45)
                dot_call_result_423698 = invoke(stypy.reporting.localization.Localization(__file__, 45, 44), dot_423694, *[A_423695, x_423696], **kwargs_423697)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 45)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 34), 'stypy_return_type', dot_call_result_423698)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_230' in the type store
                # Getting the type of 'stypy_return_type' (line 45)
                stypy_return_type_423699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 34), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_423699)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_230'
                return stypy_return_type_423699

            # Assigning a type to the variable '_stypy_temp_lambda_230' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 34), '_stypy_temp_lambda_230', _stypy_temp_lambda_230)
            # Getting the type of '_stypy_temp_lambda_230' (line 45)
            _stypy_temp_lambda_230_423700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 34), '_stypy_temp_lambda_230')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 20), dict_423666, (str_423692, _stypy_temp_lambda_230_423700))
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 19), list_423623, dict_423666)
            
            # Assigning a type to the variable 'stypy_return_type' (line 35)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'stypy_return_type', list_423623)
            
            # ################# End of 'get_matvecs(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'get_matvecs' in the type store
            # Getting the type of 'stypy_return_type' (line 34)
            stypy_return_type_423701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_423701)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'get_matvecs'
            return stypy_return_type_423701

        # Assigning a type to the variable 'get_matvecs' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'get_matvecs', get_matvecs)
        
        
        # Call to get_matvecs(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'self' (line 48)
        self_423703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 35), 'self', False)
        # Obtaining the member 'A' of a type (line 48)
        A_423704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 35), self_423703, 'A')
        # Processing the call keyword arguments (line 48)
        kwargs_423705 = {}
        # Getting the type of 'get_matvecs' (line 48)
        get_matvecs_423702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 23), 'get_matvecs', False)
        # Calling get_matvecs(args, kwargs) (line 48)
        get_matvecs_call_result_423706 = invoke(stypy.reporting.localization.Localization(__file__, 48, 23), get_matvecs_423702, *[A_423704], **kwargs_423705)
        
        # Testing the type of a for loop iterable (line 48)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 48, 8), get_matvecs_call_result_423706)
        # Getting the type of the for loop variable (line 48)
        for_loop_var_423707 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 48, 8), get_matvecs_call_result_423706)
        # Assigning a type to the variable 'matvecs' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'matvecs', for_loop_var_423707)
        # SSA begins for a for statement (line 48)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 49):
        
        # Assigning a Call to a Name (line 49):
        
        # Call to LinearOperator(...): (line 49)
        # Processing the call keyword arguments (line 49)
        # Getting the type of 'matvecs' (line 49)
        matvecs_423710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 43), 'matvecs', False)
        kwargs_423711 = {'matvecs_423710': matvecs_423710}
        # Getting the type of 'interface' (line 49)
        interface_423708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'interface', False)
        # Obtaining the member 'LinearOperator' of a type (line 49)
        LinearOperator_423709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 16), interface_423708, 'LinearOperator')
        # Calling LinearOperator(args, kwargs) (line 49)
        LinearOperator_call_result_423712 = invoke(stypy.reporting.localization.Localization(__file__, 49, 16), LinearOperator_423709, *[], **kwargs_423711)
        
        # Assigning a type to the variable 'A' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'A', LinearOperator_call_result_423712)
        
        # Call to assert_(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Getting the type of 'A' (line 51)
        A_423714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 20), 'A', False)
        # Obtaining the member 'args' of a type (line 51)
        args_423715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 20), A_423714, 'args')
        
        # Obtaining an instance of the builtin type 'tuple' (line 51)
        tuple_423716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 51)
        
        # Applying the binary operator '==' (line 51)
        result_eq_423717 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 20), '==', args_423715, tuple_423716)
        
        # Processing the call keyword arguments (line 51)
        kwargs_423718 = {}
        # Getting the type of 'assert_' (line 51)
        assert__423713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 51)
        assert__call_result_423719 = invoke(stypy.reporting.localization.Localization(__file__, 51, 12), assert__423713, *[result_eq_423717], **kwargs_423718)
        
        
        # Call to assert_equal(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Call to matvec(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Call to array(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Obtaining an instance of the builtin type 'list' (line 53)
        list_423725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 53)
        # Adding element type (line 53)
        int_423726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 43), list_423725, int_423726)
        # Adding element type (line 53)
        int_423727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 43), list_423725, int_423727)
        # Adding element type (line 53)
        int_423728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 43), list_423725, int_423728)
        
        # Processing the call keyword arguments (line 53)
        kwargs_423729 = {}
        # Getting the type of 'np' (line 53)
        np_423723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 34), 'np', False)
        # Obtaining the member 'array' of a type (line 53)
        array_423724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 34), np_423723, 'array')
        # Calling array(args, kwargs) (line 53)
        array_call_result_423730 = invoke(stypy.reporting.localization.Localization(__file__, 53, 34), array_423724, *[list_423725], **kwargs_423729)
        
        # Processing the call keyword arguments (line 53)
        kwargs_423731 = {}
        # Getting the type of 'A' (line 53)
        A_423721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 25), 'A', False)
        # Obtaining the member 'matvec' of a type (line 53)
        matvec_423722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 25), A_423721, 'matvec')
        # Calling matvec(args, kwargs) (line 53)
        matvec_call_result_423732 = invoke(stypy.reporting.localization.Localization(__file__, 53, 25), matvec_423722, *[array_call_result_423730], **kwargs_423731)
        
        
        # Obtaining an instance of the builtin type 'list' (line 53)
        list_423733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 54), 'list')
        # Adding type elements to the builtin type 'list' instance (line 53)
        # Adding element type (line 53)
        int_423734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 54), list_423733, int_423734)
        # Adding element type (line 53)
        int_423735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 54), list_423733, int_423735)
        
        # Processing the call keyword arguments (line 53)
        kwargs_423736 = {}
        # Getting the type of 'assert_equal' (line 53)
        assert_equal_423720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 53)
        assert_equal_call_result_423737 = invoke(stypy.reporting.localization.Localization(__file__, 53, 12), assert_equal_423720, *[matvec_call_result_423732, list_423733], **kwargs_423736)
        
        
        # Call to assert_equal(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Call to matvec(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Call to array(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_423743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        # Adding element type (line 54)
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_423744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        # Adding element type (line 54)
        int_423745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 44), list_423744, int_423745)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 43), list_423743, list_423744)
        # Adding element type (line 54)
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_423746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        # Adding element type (line 54)
        int_423747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 48), list_423746, int_423747)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 43), list_423743, list_423746)
        # Adding element type (line 54)
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_423748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        # Adding element type (line 54)
        int_423749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 52), list_423748, int_423749)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 43), list_423743, list_423748)
        
        # Processing the call keyword arguments (line 54)
        kwargs_423750 = {}
        # Getting the type of 'np' (line 54)
        np_423741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 34), 'np', False)
        # Obtaining the member 'array' of a type (line 54)
        array_423742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 34), np_423741, 'array')
        # Calling array(args, kwargs) (line 54)
        array_call_result_423751 = invoke(stypy.reporting.localization.Localization(__file__, 54, 34), array_423742, *[list_423743], **kwargs_423750)
        
        # Processing the call keyword arguments (line 54)
        kwargs_423752 = {}
        # Getting the type of 'A' (line 54)
        A_423739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 25), 'A', False)
        # Obtaining the member 'matvec' of a type (line 54)
        matvec_423740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 25), A_423739, 'matvec')
        # Calling matvec(args, kwargs) (line 54)
        matvec_call_result_423753 = invoke(stypy.reporting.localization.Localization(__file__, 54, 25), matvec_423740, *[array_call_result_423751], **kwargs_423752)
        
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_423754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 60), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        # Adding element type (line 54)
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_423755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 61), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        # Adding element type (line 54)
        int_423756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 61), list_423755, int_423756)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 60), list_423754, list_423755)
        # Adding element type (line 54)
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_423757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 66), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        # Adding element type (line 54)
        int_423758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 67), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 66), list_423757, int_423758)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 60), list_423754, list_423757)
        
        # Processing the call keyword arguments (line 54)
        kwargs_423759 = {}
        # Getting the type of 'assert_equal' (line 54)
        assert_equal_423738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 54)
        assert_equal_call_result_423760 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), assert_equal_423738, *[matvec_call_result_423753, list_423754], **kwargs_423759)
        
        
        # Call to assert_equal(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'A' (line 55)
        A_423762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 25), 'A', False)
        
        # Call to array(...): (line 55)
        # Processing the call arguments (line 55)
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_423765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        # Adding element type (line 55)
        int_423766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 38), list_423765, int_423766)
        # Adding element type (line 55)
        int_423767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 38), list_423765, int_423767)
        # Adding element type (line 55)
        int_423768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 38), list_423765, int_423768)
        
        # Processing the call keyword arguments (line 55)
        kwargs_423769 = {}
        # Getting the type of 'np' (line 55)
        np_423763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 29), 'np', False)
        # Obtaining the member 'array' of a type (line 55)
        array_423764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 29), np_423763, 'array')
        # Calling array(args, kwargs) (line 55)
        array_call_result_423770 = invoke(stypy.reporting.localization.Localization(__file__, 55, 29), array_423764, *[list_423765], **kwargs_423769)
        
        # Applying the binary operator '*' (line 55)
        result_mul_423771 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 25), '*', A_423762, array_call_result_423770)
        
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_423772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        # Adding element type (line 55)
        int_423773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 48), list_423772, int_423773)
        # Adding element type (line 55)
        int_423774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 48), list_423772, int_423774)
        
        # Processing the call keyword arguments (line 55)
        kwargs_423775 = {}
        # Getting the type of 'assert_equal' (line 55)
        assert_equal_423761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 55)
        assert_equal_call_result_423776 = invoke(stypy.reporting.localization.Localization(__file__, 55, 12), assert_equal_423761, *[result_mul_423771, list_423772], **kwargs_423775)
        
        
        # Call to assert_equal(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'A' (line 56)
        A_423778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 25), 'A', False)
        
        # Call to array(...): (line 56)
        # Processing the call arguments (line 56)
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_423781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        # Adding element type (line 56)
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_423782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        # Adding element type (line 56)
        int_423783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 39), list_423782, int_423783)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 38), list_423781, list_423782)
        # Adding element type (line 56)
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_423784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        # Adding element type (line 56)
        int_423785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 43), list_423784, int_423785)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 38), list_423781, list_423784)
        # Adding element type (line 56)
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_423786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        # Adding element type (line 56)
        int_423787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 47), list_423786, int_423787)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 38), list_423781, list_423786)
        
        # Processing the call keyword arguments (line 56)
        kwargs_423788 = {}
        # Getting the type of 'np' (line 56)
        np_423779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 29), 'np', False)
        # Obtaining the member 'array' of a type (line 56)
        array_423780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 29), np_423779, 'array')
        # Calling array(args, kwargs) (line 56)
        array_call_result_423789 = invoke(stypy.reporting.localization.Localization(__file__, 56, 29), array_423780, *[list_423781], **kwargs_423788)
        
        # Applying the binary operator '*' (line 56)
        result_mul_423790 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 25), '*', A_423778, array_call_result_423789)
        
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_423791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 54), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        # Adding element type (line 56)
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_423792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        # Adding element type (line 56)
        int_423793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 55), list_423792, int_423793)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 54), list_423791, list_423792)
        # Adding element type (line 56)
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_423794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 60), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        # Adding element type (line 56)
        int_423795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 60), list_423794, int_423795)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 54), list_423791, list_423794)
        
        # Processing the call keyword arguments (line 56)
        kwargs_423796 = {}
        # Getting the type of 'assert_equal' (line 56)
        assert_equal_423777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 56)
        assert_equal_call_result_423797 = invoke(stypy.reporting.localization.Localization(__file__, 56, 12), assert_equal_423777, *[result_mul_423790, list_423791], **kwargs_423796)
        
        
        # Call to assert_equal(...): (line 57)
        # Processing the call arguments (line 57)
        
        # Call to dot(...): (line 57)
        # Processing the call arguments (line 57)
        
        # Call to array(...): (line 57)
        # Processing the call arguments (line 57)
        
        # Obtaining an instance of the builtin type 'list' (line 57)
        list_423803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 57)
        # Adding element type (line 57)
        int_423804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 40), list_423803, int_423804)
        # Adding element type (line 57)
        int_423805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 40), list_423803, int_423805)
        # Adding element type (line 57)
        int_423806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 40), list_423803, int_423806)
        
        # Processing the call keyword arguments (line 57)
        kwargs_423807 = {}
        # Getting the type of 'np' (line 57)
        np_423801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 31), 'np', False)
        # Obtaining the member 'array' of a type (line 57)
        array_423802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 31), np_423801, 'array')
        # Calling array(args, kwargs) (line 57)
        array_call_result_423808 = invoke(stypy.reporting.localization.Localization(__file__, 57, 31), array_423802, *[list_423803], **kwargs_423807)
        
        # Processing the call keyword arguments (line 57)
        kwargs_423809 = {}
        # Getting the type of 'A' (line 57)
        A_423799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'A', False)
        # Obtaining the member 'dot' of a type (line 57)
        dot_423800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 25), A_423799, 'dot')
        # Calling dot(args, kwargs) (line 57)
        dot_call_result_423810 = invoke(stypy.reporting.localization.Localization(__file__, 57, 25), dot_423800, *[array_call_result_423808], **kwargs_423809)
        
        
        # Obtaining an instance of the builtin type 'list' (line 57)
        list_423811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 57)
        # Adding element type (line 57)
        int_423812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 51), list_423811, int_423812)
        # Adding element type (line 57)
        int_423813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 51), list_423811, int_423813)
        
        # Processing the call keyword arguments (line 57)
        kwargs_423814 = {}
        # Getting the type of 'assert_equal' (line 57)
        assert_equal_423798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 57)
        assert_equal_call_result_423815 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), assert_equal_423798, *[dot_call_result_423810, list_423811], **kwargs_423814)
        
        
        # Call to assert_equal(...): (line 58)
        # Processing the call arguments (line 58)
        
        # Call to dot(...): (line 58)
        # Processing the call arguments (line 58)
        
        # Call to array(...): (line 58)
        # Processing the call arguments (line 58)
        
        # Obtaining an instance of the builtin type 'list' (line 58)
        list_423821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 58)
        # Adding element type (line 58)
        
        # Obtaining an instance of the builtin type 'list' (line 58)
        list_423822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 58)
        # Adding element type (line 58)
        int_423823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 41), list_423822, int_423823)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 40), list_423821, list_423822)
        # Adding element type (line 58)
        
        # Obtaining an instance of the builtin type 'list' (line 58)
        list_423824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 58)
        # Adding element type (line 58)
        int_423825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 45), list_423824, int_423825)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 40), list_423821, list_423824)
        # Adding element type (line 58)
        
        # Obtaining an instance of the builtin type 'list' (line 58)
        list_423826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 58)
        # Adding element type (line 58)
        int_423827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 49), list_423826, int_423827)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 40), list_423821, list_423826)
        
        # Processing the call keyword arguments (line 58)
        kwargs_423828 = {}
        # Getting the type of 'np' (line 58)
        np_423819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 31), 'np', False)
        # Obtaining the member 'array' of a type (line 58)
        array_423820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 31), np_423819, 'array')
        # Calling array(args, kwargs) (line 58)
        array_call_result_423829 = invoke(stypy.reporting.localization.Localization(__file__, 58, 31), array_423820, *[list_423821], **kwargs_423828)
        
        # Processing the call keyword arguments (line 58)
        kwargs_423830 = {}
        # Getting the type of 'A' (line 58)
        A_423817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 25), 'A', False)
        # Obtaining the member 'dot' of a type (line 58)
        dot_423818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 25), A_423817, 'dot')
        # Calling dot(args, kwargs) (line 58)
        dot_call_result_423831 = invoke(stypy.reporting.localization.Localization(__file__, 58, 25), dot_423818, *[array_call_result_423829], **kwargs_423830)
        
        
        # Obtaining an instance of the builtin type 'list' (line 58)
        list_423832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 58)
        # Adding element type (line 58)
        
        # Obtaining an instance of the builtin type 'list' (line 58)
        list_423833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 58)
        # Adding element type (line 58)
        int_423834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 58), list_423833, int_423834)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 57), list_423832, list_423833)
        # Adding element type (line 58)
        
        # Obtaining an instance of the builtin type 'list' (line 58)
        list_423835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 63), 'list')
        # Adding type elements to the builtin type 'list' instance (line 58)
        # Adding element type (line 58)
        int_423836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 64), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 63), list_423835, int_423836)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 57), list_423832, list_423835)
        
        # Processing the call keyword arguments (line 58)
        kwargs_423837 = {}
        # Getting the type of 'assert_equal' (line 58)
        assert_equal_423816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 58)
        assert_equal_call_result_423838 = invoke(stypy.reporting.localization.Localization(__file__, 58, 12), assert_equal_423816, *[dot_call_result_423831, list_423832], **kwargs_423837)
        
        
        # Call to assert_equal(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Call to matvec(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Call to matrix(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Obtaining an instance of the builtin type 'list' (line 60)
        list_423844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 60)
        # Adding element type (line 60)
        
        # Obtaining an instance of the builtin type 'list' (line 60)
        list_423845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 60)
        # Adding element type (line 60)
        int_423846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 45), list_423845, int_423846)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 44), list_423844, list_423845)
        # Adding element type (line 60)
        
        # Obtaining an instance of the builtin type 'list' (line 60)
        list_423847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 60)
        # Adding element type (line 60)
        int_423848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 49), list_423847, int_423848)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 44), list_423844, list_423847)
        # Adding element type (line 60)
        
        # Obtaining an instance of the builtin type 'list' (line 60)
        list_423849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 60)
        # Adding element type (line 60)
        int_423850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 53), list_423849, int_423850)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 44), list_423844, list_423849)
        
        # Processing the call keyword arguments (line 60)
        kwargs_423851 = {}
        # Getting the type of 'np' (line 60)
        np_423842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 34), 'np', False)
        # Obtaining the member 'matrix' of a type (line 60)
        matrix_423843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 34), np_423842, 'matrix')
        # Calling matrix(args, kwargs) (line 60)
        matrix_call_result_423852 = invoke(stypy.reporting.localization.Localization(__file__, 60, 34), matrix_423843, *[list_423844], **kwargs_423851)
        
        # Processing the call keyword arguments (line 60)
        kwargs_423853 = {}
        # Getting the type of 'A' (line 60)
        A_423840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 'A', False)
        # Obtaining the member 'matvec' of a type (line 60)
        matvec_423841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 25), A_423840, 'matvec')
        # Calling matvec(args, kwargs) (line 60)
        matvec_call_result_423854 = invoke(stypy.reporting.localization.Localization(__file__, 60, 25), matvec_423841, *[matrix_call_result_423852], **kwargs_423853)
        
        
        # Obtaining an instance of the builtin type 'list' (line 60)
        list_423855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 61), 'list')
        # Adding type elements to the builtin type 'list' instance (line 60)
        # Adding element type (line 60)
        
        # Obtaining an instance of the builtin type 'list' (line 60)
        list_423856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 62), 'list')
        # Adding type elements to the builtin type 'list' instance (line 60)
        # Adding element type (line 60)
        int_423857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 62), list_423856, int_423857)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 61), list_423855, list_423856)
        # Adding element type (line 60)
        
        # Obtaining an instance of the builtin type 'list' (line 60)
        list_423858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 67), 'list')
        # Adding type elements to the builtin type 'list' instance (line 60)
        # Adding element type (line 60)
        int_423859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 67), list_423858, int_423859)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 61), list_423855, list_423858)
        
        # Processing the call keyword arguments (line 60)
        kwargs_423860 = {}
        # Getting the type of 'assert_equal' (line 60)
        assert_equal_423839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 60)
        assert_equal_call_result_423861 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), assert_equal_423839, *[matvec_call_result_423854, list_423855], **kwargs_423860)
        
        
        # Call to assert_equal(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'A' (line 61)
        A_423863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 25), 'A', False)
        
        # Call to matrix(...): (line 61)
        # Processing the call arguments (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_423866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_423867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        int_423868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 40), list_423867, int_423868)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 39), list_423866, list_423867)
        # Adding element type (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_423869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        int_423870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 44), list_423869, int_423870)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 39), list_423866, list_423869)
        # Adding element type (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_423871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        int_423872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 48), list_423871, int_423872)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 39), list_423866, list_423871)
        
        # Processing the call keyword arguments (line 61)
        kwargs_423873 = {}
        # Getting the type of 'np' (line 61)
        np_423864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 29), 'np', False)
        # Obtaining the member 'matrix' of a type (line 61)
        matrix_423865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 29), np_423864, 'matrix')
        # Calling matrix(args, kwargs) (line 61)
        matrix_call_result_423874 = invoke(stypy.reporting.localization.Localization(__file__, 61, 29), matrix_423865, *[list_423866], **kwargs_423873)
        
        # Applying the binary operator '*' (line 61)
        result_mul_423875 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 25), '*', A_423863, matrix_call_result_423874)
        
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_423876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_423877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        int_423878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 56), list_423877, int_423878)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 55), list_423876, list_423877)
        # Adding element type (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_423879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 61), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        int_423880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 61), list_423879, int_423880)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 55), list_423876, list_423879)
        
        # Processing the call keyword arguments (line 61)
        kwargs_423881 = {}
        # Getting the type of 'assert_equal' (line 61)
        assert_equal_423862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 61)
        assert_equal_call_result_423882 = invoke(stypy.reporting.localization.Localization(__file__, 61, 12), assert_equal_423862, *[result_mul_423875, list_423876], **kwargs_423881)
        
        
        # Call to assert_equal(...): (line 62)
        # Processing the call arguments (line 62)
        
        # Call to dot(...): (line 62)
        # Processing the call arguments (line 62)
        
        # Call to matrix(...): (line 62)
        # Processing the call arguments (line 62)
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_423888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        # Adding element type (line 62)
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_423889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        # Adding element type (line 62)
        int_423890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 42), list_423889, int_423890)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 41), list_423888, list_423889)
        # Adding element type (line 62)
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_423891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        # Adding element type (line 62)
        int_423892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 46), list_423891, int_423892)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 41), list_423888, list_423891)
        # Adding element type (line 62)
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_423893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        # Adding element type (line 62)
        int_423894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 50), list_423893, int_423894)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 41), list_423888, list_423893)
        
        # Processing the call keyword arguments (line 62)
        kwargs_423895 = {}
        # Getting the type of 'np' (line 62)
        np_423886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 31), 'np', False)
        # Obtaining the member 'matrix' of a type (line 62)
        matrix_423887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 31), np_423886, 'matrix')
        # Calling matrix(args, kwargs) (line 62)
        matrix_call_result_423896 = invoke(stypy.reporting.localization.Localization(__file__, 62, 31), matrix_423887, *[list_423888], **kwargs_423895)
        
        # Processing the call keyword arguments (line 62)
        kwargs_423897 = {}
        # Getting the type of 'A' (line 62)
        A_423884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 25), 'A', False)
        # Obtaining the member 'dot' of a type (line 62)
        dot_423885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 25), A_423884, 'dot')
        # Calling dot(args, kwargs) (line 62)
        dot_call_result_423898 = invoke(stypy.reporting.localization.Localization(__file__, 62, 25), dot_423885, *[matrix_call_result_423896], **kwargs_423897)
        
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_423899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        # Adding element type (line 62)
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_423900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 59), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        # Adding element type (line 62)
        int_423901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 59), list_423900, int_423901)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 58), list_423899, list_423900)
        # Adding element type (line 62)
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_423902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 64), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        # Adding element type (line 62)
        int_423903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 64), list_423902, int_423903)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 58), list_423899, list_423902)
        
        # Processing the call keyword arguments (line 62)
        kwargs_423904 = {}
        # Getting the type of 'assert_equal' (line 62)
        assert_equal_423883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 62)
        assert_equal_call_result_423905 = invoke(stypy.reporting.localization.Localization(__file__, 62, 12), assert_equal_423883, *[dot_call_result_423898, list_423899], **kwargs_423904)
        
        
        # Call to assert_equal(...): (line 64)
        # Processing the call arguments (line 64)
        int_423907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 26), 'int')
        # Getting the type of 'A' (line 64)
        A_423908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 28), 'A', False)
        # Applying the binary operator '*' (line 64)
        result_mul_423909 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 26), '*', int_423907, A_423908)
        
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_423910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        # Adding element type (line 64)
        int_423911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 31), list_423910, int_423911)
        # Adding element type (line 64)
        int_423912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 31), list_423910, int_423912)
        # Adding element type (line 64)
        int_423913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 31), list_423910, int_423913)
        
        # Applying the binary operator '*' (line 64)
        result_mul_423914 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 25), '*', result_mul_423909, list_423910)
        
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_423915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        # Adding element type (line 64)
        int_423916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 40), list_423915, int_423916)
        # Adding element type (line 64)
        int_423917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 40), list_423915, int_423917)
        
        # Processing the call keyword arguments (line 64)
        kwargs_423918 = {}
        # Getting the type of 'assert_equal' (line 64)
        assert_equal_423906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 64)
        assert_equal_call_result_423919 = invoke(stypy.reporting.localization.Localization(__file__, 64, 12), assert_equal_423906, *[result_mul_423914, list_423915], **kwargs_423918)
        
        
        # Call to assert_equal(...): (line 65)
        # Processing the call arguments (line 65)
        
        # Call to rmatvec(...): (line 65)
        # Processing the call arguments (line 65)
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_423925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        # Adding element type (line 65)
        int_423926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 39), list_423925, int_423926)
        # Adding element type (line 65)
        int_423927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 39), list_423925, int_423927)
        
        # Processing the call keyword arguments (line 65)
        kwargs_423928 = {}
        int_423921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 26), 'int')
        # Getting the type of 'A' (line 65)
        A_423922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 28), 'A', False)
        # Applying the binary operator '*' (line 65)
        result_mul_423923 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 26), '*', int_423921, A_423922)
        
        # Obtaining the member 'rmatvec' of a type (line 65)
        rmatvec_423924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 26), result_mul_423923, 'rmatvec')
        # Calling rmatvec(args, kwargs) (line 65)
        rmatvec_call_result_423929 = invoke(stypy.reporting.localization.Localization(__file__, 65, 26), rmatvec_423924, *[list_423925], **kwargs_423928)
        
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_423930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        # Adding element type (line 65)
        int_423931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 47), list_423930, int_423931)
        # Adding element type (line 65)
        int_423932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 47), list_423930, int_423932)
        # Adding element type (line 65)
        int_423933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 47), list_423930, int_423933)
        
        # Processing the call keyword arguments (line 65)
        kwargs_423934 = {}
        # Getting the type of 'assert_equal' (line 65)
        assert_equal_423920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 65)
        assert_equal_call_result_423935 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), assert_equal_423920, *[rmatvec_call_result_423929, list_423930], **kwargs_423934)
        
        
        # Call to assert_equal(...): (line 66)
        # Processing the call arguments (line 66)
        
        # Call to matvec(...): (line 66)
        # Processing the call arguments (line 66)
        
        # Obtaining an instance of the builtin type 'list' (line 66)
        list_423942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 66)
        # Adding element type (line 66)
        int_423943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 40), list_423942, int_423943)
        # Adding element type (line 66)
        int_423944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 40), list_423942, int_423944)
        
        # Processing the call keyword arguments (line 66)
        kwargs_423945 = {}
        int_423937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 26), 'int')
        # Getting the type of 'A' (line 66)
        A_423938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 28), 'A', False)
        # Applying the binary operator '*' (line 66)
        result_mul_423939 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 26), '*', int_423937, A_423938)
        
        # Obtaining the member 'H' of a type (line 66)
        H_423940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 26), result_mul_423939, 'H')
        # Obtaining the member 'matvec' of a type (line 66)
        matvec_423941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 26), H_423940, 'matvec')
        # Calling matvec(args, kwargs) (line 66)
        matvec_call_result_423946 = invoke(stypy.reporting.localization.Localization(__file__, 66, 26), matvec_423941, *[list_423942], **kwargs_423945)
        
        
        # Obtaining an instance of the builtin type 'list' (line 66)
        list_423947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 66)
        # Adding element type (line 66)
        int_423948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 48), list_423947, int_423948)
        # Adding element type (line 66)
        int_423949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 48), list_423947, int_423949)
        # Adding element type (line 66)
        int_423950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 48), list_423947, int_423950)
        
        # Processing the call keyword arguments (line 66)
        kwargs_423951 = {}
        # Getting the type of 'assert_equal' (line 66)
        assert_equal_423936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 66)
        assert_equal_call_result_423952 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), assert_equal_423936, *[matvec_call_result_423946, list_423947], **kwargs_423951)
        
        
        # Call to assert_equal(...): (line 67)
        # Processing the call arguments (line 67)
        int_423954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 26), 'int')
        # Getting the type of 'A' (line 67)
        A_423955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 28), 'A', False)
        # Applying the binary operator '*' (line 67)
        result_mul_423956 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 26), '*', int_423954, A_423955)
        
        
        # Obtaining an instance of the builtin type 'list' (line 67)
        list_423957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 67)
        # Adding element type (line 67)
        
        # Obtaining an instance of the builtin type 'list' (line 67)
        list_423958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 67)
        # Adding element type (line 67)
        int_423959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 32), list_423958, int_423959)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 31), list_423957, list_423958)
        # Adding element type (line 67)
        
        # Obtaining an instance of the builtin type 'list' (line 67)
        list_423960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 67)
        # Adding element type (line 67)
        int_423961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 36), list_423960, int_423961)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 31), list_423957, list_423960)
        # Adding element type (line 67)
        
        # Obtaining an instance of the builtin type 'list' (line 67)
        list_423962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 67)
        # Adding element type (line 67)
        int_423963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 40), list_423962, int_423963)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 31), list_423957, list_423962)
        
        # Applying the binary operator '*' (line 67)
        result_mul_423964 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 25), '*', result_mul_423956, list_423957)
        
        
        # Obtaining an instance of the builtin type 'list' (line 67)
        list_423965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 67)
        # Adding element type (line 67)
        
        # Obtaining an instance of the builtin type 'list' (line 67)
        list_423966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 67)
        # Adding element type (line 67)
        int_423967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 47), list_423966, int_423967)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 46), list_423965, list_423966)
        # Adding element type (line 67)
        
        # Obtaining an instance of the builtin type 'list' (line 67)
        list_423968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 67)
        # Adding element type (line 67)
        int_423969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 52), list_423968, int_423969)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 46), list_423965, list_423968)
        
        # Processing the call keyword arguments (line 67)
        kwargs_423970 = {}
        # Getting the type of 'assert_equal' (line 67)
        assert_equal_423953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 67)
        assert_equal_call_result_423971 = invoke(stypy.reporting.localization.Localization(__file__, 67, 12), assert_equal_423953, *[result_mul_423964, list_423965], **kwargs_423970)
        
        
        # Call to assert_equal(...): (line 68)
        # Processing the call arguments (line 68)
        
        # Call to matmat(...): (line 68)
        # Processing the call arguments (line 68)
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_423977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        # Adding element type (line 68)
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_423978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        # Adding element type (line 68)
        int_423979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 39), list_423978, int_423979)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 38), list_423977, list_423978)
        # Adding element type (line 68)
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_423980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        # Adding element type (line 68)
        int_423981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 43), list_423980, int_423981)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 38), list_423977, list_423980)
        # Adding element type (line 68)
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_423982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        # Adding element type (line 68)
        int_423983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 47), list_423982, int_423983)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 38), list_423977, list_423982)
        
        # Processing the call keyword arguments (line 68)
        kwargs_423984 = {}
        int_423973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 26), 'int')
        # Getting the type of 'A' (line 68)
        A_423974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 28), 'A', False)
        # Applying the binary operator '*' (line 68)
        result_mul_423975 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 26), '*', int_423973, A_423974)
        
        # Obtaining the member 'matmat' of a type (line 68)
        matmat_423976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 26), result_mul_423975, 'matmat')
        # Calling matmat(args, kwargs) (line 68)
        matmat_call_result_423985 = invoke(stypy.reporting.localization.Localization(__file__, 68, 26), matmat_423976, *[list_423977], **kwargs_423984)
        
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_423986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 54), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        # Adding element type (line 68)
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_423987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        # Adding element type (line 68)
        int_423988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 55), list_423987, int_423988)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 54), list_423986, list_423987)
        # Adding element type (line 68)
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_423989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 60), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        # Adding element type (line 68)
        int_423990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 60), list_423989, int_423990)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 54), list_423986, list_423989)
        
        # Processing the call keyword arguments (line 68)
        kwargs_423991 = {}
        # Getting the type of 'assert_equal' (line 68)
        assert_equal_423972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 68)
        assert_equal_call_result_423992 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), assert_equal_423972, *[matmat_call_result_423985, list_423986], **kwargs_423991)
        
        
        # Call to assert_equal(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'A' (line 69)
        A_423994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 26), 'A', False)
        int_423995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 28), 'int')
        # Applying the binary operator '*' (line 69)
        result_mul_423996 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 26), '*', A_423994, int_423995)
        
        
        # Obtaining an instance of the builtin type 'list' (line 69)
        list_423997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 69)
        # Adding element type (line 69)
        int_423998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 31), list_423997, int_423998)
        # Adding element type (line 69)
        int_423999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 31), list_423997, int_423999)
        # Adding element type (line 69)
        int_424000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 31), list_423997, int_424000)
        
        # Applying the binary operator '*' (line 69)
        result_mul_424001 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 25), '*', result_mul_423996, list_423997)
        
        
        # Obtaining an instance of the builtin type 'list' (line 69)
        list_424002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 69)
        # Adding element type (line 69)
        int_424003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 40), list_424002, int_424003)
        # Adding element type (line 69)
        int_424004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 40), list_424002, int_424004)
        
        # Processing the call keyword arguments (line 69)
        kwargs_424005 = {}
        # Getting the type of 'assert_equal' (line 69)
        assert_equal_423993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 69)
        assert_equal_call_result_424006 = invoke(stypy.reporting.localization.Localization(__file__, 69, 12), assert_equal_423993, *[result_mul_424001, list_424002], **kwargs_424005)
        
        
        # Call to assert_equal(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'A' (line 70)
        A_424008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 26), 'A', False)
        int_424009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 28), 'int')
        # Applying the binary operator '*' (line 70)
        result_mul_424010 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 26), '*', A_424008, int_424009)
        
        
        # Obtaining an instance of the builtin type 'list' (line 70)
        list_424011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 70)
        # Adding element type (line 70)
        
        # Obtaining an instance of the builtin type 'list' (line 70)
        list_424012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 70)
        # Adding element type (line 70)
        int_424013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 32), list_424012, int_424013)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 31), list_424011, list_424012)
        # Adding element type (line 70)
        
        # Obtaining an instance of the builtin type 'list' (line 70)
        list_424014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 70)
        # Adding element type (line 70)
        int_424015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 36), list_424014, int_424015)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 31), list_424011, list_424014)
        # Adding element type (line 70)
        
        # Obtaining an instance of the builtin type 'list' (line 70)
        list_424016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 70)
        # Adding element type (line 70)
        int_424017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 40), list_424016, int_424017)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 31), list_424011, list_424016)
        
        # Applying the binary operator '*' (line 70)
        result_mul_424018 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 25), '*', result_mul_424010, list_424011)
        
        
        # Obtaining an instance of the builtin type 'list' (line 70)
        list_424019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 70)
        # Adding element type (line 70)
        
        # Obtaining an instance of the builtin type 'list' (line 70)
        list_424020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 70)
        # Adding element type (line 70)
        int_424021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 47), list_424020, int_424021)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 46), list_424019, list_424020)
        # Adding element type (line 70)
        
        # Obtaining an instance of the builtin type 'list' (line 70)
        list_424022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 70)
        # Adding element type (line 70)
        int_424023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 52), list_424022, int_424023)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 46), list_424019, list_424022)
        
        # Processing the call keyword arguments (line 70)
        kwargs_424024 = {}
        # Getting the type of 'assert_equal' (line 70)
        assert_equal_424007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 70)
        assert_equal_call_result_424025 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), assert_equal_424007, *[result_mul_424018, list_424019], **kwargs_424024)
        
        
        # Call to assert_equal(...): (line 71)
        # Processing the call arguments (line 71)
        complex_424027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 26), 'complex')
        # Getting the type of 'A' (line 71)
        A_424028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 29), 'A', False)
        # Applying the binary operator '*' (line 71)
        result_mul_424029 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 26), '*', complex_424027, A_424028)
        
        
        # Obtaining an instance of the builtin type 'list' (line 71)
        list_424030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 71)
        # Adding element type (line 71)
        int_424031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 32), list_424030, int_424031)
        # Adding element type (line 71)
        int_424032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 32), list_424030, int_424032)
        # Adding element type (line 71)
        int_424033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 32), list_424030, int_424033)
        
        # Applying the binary operator '*' (line 71)
        result_mul_424034 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 25), '*', result_mul_424029, list_424030)
        
        
        # Obtaining an instance of the builtin type 'list' (line 71)
        list_424035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 71)
        # Adding element type (line 71)
        complex_424036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 42), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 41), list_424035, complex_424036)
        # Adding element type (line 71)
        complex_424037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 46), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 41), list_424035, complex_424037)
        
        # Processing the call keyword arguments (line 71)
        kwargs_424038 = {}
        # Getting the type of 'assert_equal' (line 71)
        assert_equal_424026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 71)
        assert_equal_call_result_424039 = invoke(stypy.reporting.localization.Localization(__file__, 71, 12), assert_equal_424026, *[result_mul_424034, list_424035], **kwargs_424038)
        
        
        # Call to assert_equal(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'A' (line 72)
        A_424041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 26), 'A', False)
        # Getting the type of 'A' (line 72)
        A_424042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 28), 'A', False)
        # Applying the binary operator '+' (line 72)
        result_add_424043 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 26), '+', A_424041, A_424042)
        
        
        # Obtaining an instance of the builtin type 'list' (line 72)
        list_424044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 72)
        # Adding element type (line 72)
        int_424045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 31), list_424044, int_424045)
        # Adding element type (line 72)
        int_424046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 31), list_424044, int_424046)
        # Adding element type (line 72)
        int_424047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 31), list_424044, int_424047)
        
        # Applying the binary operator '*' (line 72)
        result_mul_424048 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 25), '*', result_add_424043, list_424044)
        
        
        # Obtaining an instance of the builtin type 'list' (line 72)
        list_424049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 72)
        # Adding element type (line 72)
        int_424050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 40), list_424049, int_424050)
        # Adding element type (line 72)
        int_424051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 40), list_424049, int_424051)
        
        # Processing the call keyword arguments (line 72)
        kwargs_424052 = {}
        # Getting the type of 'assert_equal' (line 72)
        assert_equal_424040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 72)
        assert_equal_call_result_424053 = invoke(stypy.reporting.localization.Localization(__file__, 72, 12), assert_equal_424040, *[result_mul_424048, list_424049], **kwargs_424052)
        
        
        # Call to assert_equal(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Call to rmatvec(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Obtaining an instance of the builtin type 'list' (line 73)
        list_424059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 73)
        # Adding element type (line 73)
        int_424060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 39), list_424059, int_424060)
        # Adding element type (line 73)
        int_424061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 39), list_424059, int_424061)
        
        # Processing the call keyword arguments (line 73)
        kwargs_424062 = {}
        # Getting the type of 'A' (line 73)
        A_424055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 26), 'A', False)
        # Getting the type of 'A' (line 73)
        A_424056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 28), 'A', False)
        # Applying the binary operator '+' (line 73)
        result_add_424057 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 26), '+', A_424055, A_424056)
        
        # Obtaining the member 'rmatvec' of a type (line 73)
        rmatvec_424058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 26), result_add_424057, 'rmatvec')
        # Calling rmatvec(args, kwargs) (line 73)
        rmatvec_call_result_424063 = invoke(stypy.reporting.localization.Localization(__file__, 73, 26), rmatvec_424058, *[list_424059], **kwargs_424062)
        
        
        # Obtaining an instance of the builtin type 'list' (line 73)
        list_424064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 73)
        # Adding element type (line 73)
        int_424065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 47), list_424064, int_424065)
        # Adding element type (line 73)
        int_424066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 47), list_424064, int_424066)
        # Adding element type (line 73)
        int_424067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 47), list_424064, int_424067)
        
        # Processing the call keyword arguments (line 73)
        kwargs_424068 = {}
        # Getting the type of 'assert_equal' (line 73)
        assert_equal_424054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 73)
        assert_equal_call_result_424069 = invoke(stypy.reporting.localization.Localization(__file__, 73, 12), assert_equal_424054, *[rmatvec_call_result_424063, list_424064], **kwargs_424068)
        
        
        # Call to assert_equal(...): (line 74)
        # Processing the call arguments (line 74)
        
        # Call to matvec(...): (line 74)
        # Processing the call arguments (line 74)
        
        # Obtaining an instance of the builtin type 'list' (line 74)
        list_424076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 74)
        # Adding element type (line 74)
        int_424077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 40), list_424076, int_424077)
        # Adding element type (line 74)
        int_424078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 40), list_424076, int_424078)
        
        # Processing the call keyword arguments (line 74)
        kwargs_424079 = {}
        # Getting the type of 'A' (line 74)
        A_424071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 26), 'A', False)
        # Getting the type of 'A' (line 74)
        A_424072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'A', False)
        # Applying the binary operator '+' (line 74)
        result_add_424073 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 26), '+', A_424071, A_424072)
        
        # Obtaining the member 'H' of a type (line 74)
        H_424074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 26), result_add_424073, 'H')
        # Obtaining the member 'matvec' of a type (line 74)
        matvec_424075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 26), H_424074, 'matvec')
        # Calling matvec(args, kwargs) (line 74)
        matvec_call_result_424080 = invoke(stypy.reporting.localization.Localization(__file__, 74, 26), matvec_424075, *[list_424076], **kwargs_424079)
        
        
        # Obtaining an instance of the builtin type 'list' (line 74)
        list_424081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 74)
        # Adding element type (line 74)
        int_424082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 48), list_424081, int_424082)
        # Adding element type (line 74)
        int_424083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 48), list_424081, int_424083)
        # Adding element type (line 74)
        int_424084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 48), list_424081, int_424084)
        
        # Processing the call keyword arguments (line 74)
        kwargs_424085 = {}
        # Getting the type of 'assert_equal' (line 74)
        assert_equal_424070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 74)
        assert_equal_call_result_424086 = invoke(stypy.reporting.localization.Localization(__file__, 74, 12), assert_equal_424070, *[matvec_call_result_424080, list_424081], **kwargs_424085)
        
        
        # Call to assert_equal(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'A' (line 75)
        A_424088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 26), 'A', False)
        # Getting the type of 'A' (line 75)
        A_424089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 28), 'A', False)
        # Applying the binary operator '+' (line 75)
        result_add_424090 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 26), '+', A_424088, A_424089)
        
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_424091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        # Adding element type (line 75)
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_424092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        # Adding element type (line 75)
        int_424093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 32), list_424092, int_424093)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 31), list_424091, list_424092)
        # Adding element type (line 75)
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_424094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        # Adding element type (line 75)
        int_424095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 36), list_424094, int_424095)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 31), list_424091, list_424094)
        # Adding element type (line 75)
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_424096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        # Adding element type (line 75)
        int_424097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 40), list_424096, int_424097)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 31), list_424091, list_424096)
        
        # Applying the binary operator '*' (line 75)
        result_mul_424098 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 25), '*', result_add_424090, list_424091)
        
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_424099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        # Adding element type (line 75)
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_424100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        # Adding element type (line 75)
        int_424101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 47), list_424100, int_424101)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 46), list_424099, list_424100)
        # Adding element type (line 75)
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_424102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        # Adding element type (line 75)
        int_424103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 53), list_424102, int_424103)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 46), list_424099, list_424102)
        
        # Processing the call keyword arguments (line 75)
        kwargs_424104 = {}
        # Getting the type of 'assert_equal' (line 75)
        assert_equal_424087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 75)
        assert_equal_call_result_424105 = invoke(stypy.reporting.localization.Localization(__file__, 75, 12), assert_equal_424087, *[result_mul_424098, list_424099], **kwargs_424104)
        
        
        # Call to assert_equal(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Call to matmat(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_424111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        # Adding element type (line 76)
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_424112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        # Adding element type (line 76)
        int_424113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 39), list_424112, int_424113)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 38), list_424111, list_424112)
        # Adding element type (line 76)
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_424114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        # Adding element type (line 76)
        int_424115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 43), list_424114, int_424115)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 38), list_424111, list_424114)
        # Adding element type (line 76)
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_424116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        # Adding element type (line 76)
        int_424117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 47), list_424116, int_424117)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 38), list_424111, list_424116)
        
        # Processing the call keyword arguments (line 76)
        kwargs_424118 = {}
        # Getting the type of 'A' (line 76)
        A_424107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 26), 'A', False)
        # Getting the type of 'A' (line 76)
        A_424108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 28), 'A', False)
        # Applying the binary operator '+' (line 76)
        result_add_424109 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 26), '+', A_424107, A_424108)
        
        # Obtaining the member 'matmat' of a type (line 76)
        matmat_424110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 26), result_add_424109, 'matmat')
        # Calling matmat(args, kwargs) (line 76)
        matmat_call_result_424119 = invoke(stypy.reporting.localization.Localization(__file__, 76, 26), matmat_424110, *[list_424111], **kwargs_424118)
        
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_424120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 54), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        # Adding element type (line 76)
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_424121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        # Adding element type (line 76)
        int_424122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 55), list_424121, int_424122)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 54), list_424120, list_424121)
        # Adding element type (line 76)
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_424123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 61), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        # Adding element type (line 76)
        int_424124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 61), list_424123, int_424124)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 54), list_424120, list_424123)
        
        # Processing the call keyword arguments (line 76)
        kwargs_424125 = {}
        # Getting the type of 'assert_equal' (line 76)
        assert_equal_424106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 76)
        assert_equal_call_result_424126 = invoke(stypy.reporting.localization.Localization(__file__, 76, 12), assert_equal_424106, *[matmat_call_result_424119, list_424120], **kwargs_424125)
        
        
        # Call to assert_equal(...): (line 77)
        # Processing the call arguments (line 77)
        
        # Getting the type of 'A' (line 77)
        A_424128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 27), 'A', False)
        # Applying the 'usub' unary operator (line 77)
        result___neg___424129 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 26), 'usub', A_424128)
        
        
        # Obtaining an instance of the builtin type 'list' (line 77)
        list_424130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 77)
        # Adding element type (line 77)
        int_424131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 30), list_424130, int_424131)
        # Adding element type (line 77)
        int_424132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 30), list_424130, int_424132)
        # Adding element type (line 77)
        int_424133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 30), list_424130, int_424133)
        
        # Applying the binary operator '*' (line 77)
        result_mul_424134 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 25), '*', result___neg___424129, list_424130)
        
        
        # Obtaining an instance of the builtin type 'list' (line 77)
        list_424135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 77)
        # Adding element type (line 77)
        int_424136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 39), list_424135, int_424136)
        # Adding element type (line 77)
        int_424137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 39), list_424135, int_424137)
        
        # Processing the call keyword arguments (line 77)
        kwargs_424138 = {}
        # Getting the type of 'assert_equal' (line 77)
        assert_equal_424127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 77)
        assert_equal_call_result_424139 = invoke(stypy.reporting.localization.Localization(__file__, 77, 12), assert_equal_424127, *[result_mul_424134, list_424135], **kwargs_424138)
        
        
        # Call to assert_equal(...): (line 78)
        # Processing the call arguments (line 78)
        
        # Getting the type of 'A' (line 78)
        A_424141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 27), 'A', False)
        # Applying the 'usub' unary operator (line 78)
        result___neg___424142 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 26), 'usub', A_424141)
        
        
        # Obtaining an instance of the builtin type 'list' (line 78)
        list_424143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 78)
        # Adding element type (line 78)
        
        # Obtaining an instance of the builtin type 'list' (line 78)
        list_424144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 78)
        # Adding element type (line 78)
        int_424145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 31), list_424144, int_424145)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 30), list_424143, list_424144)
        # Adding element type (line 78)
        
        # Obtaining an instance of the builtin type 'list' (line 78)
        list_424146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 78)
        # Adding element type (line 78)
        int_424147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 35), list_424146, int_424147)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 30), list_424143, list_424146)
        # Adding element type (line 78)
        
        # Obtaining an instance of the builtin type 'list' (line 78)
        list_424148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 78)
        # Adding element type (line 78)
        int_424149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 39), list_424148, int_424149)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 30), list_424143, list_424148)
        
        # Applying the binary operator '*' (line 78)
        result_mul_424150 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 25), '*', result___neg___424142, list_424143)
        
        
        # Obtaining an instance of the builtin type 'list' (line 78)
        list_424151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 78)
        # Adding element type (line 78)
        
        # Obtaining an instance of the builtin type 'list' (line 78)
        list_424152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 78)
        # Adding element type (line 78)
        int_424153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 46), list_424152, int_424153)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 45), list_424151, list_424152)
        # Adding element type (line 78)
        
        # Obtaining an instance of the builtin type 'list' (line 78)
        list_424154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 78)
        # Adding element type (line 78)
        int_424155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 51), list_424154, int_424155)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 45), list_424151, list_424154)
        
        # Processing the call keyword arguments (line 78)
        kwargs_424156 = {}
        # Getting the type of 'assert_equal' (line 78)
        assert_equal_424140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 78)
        assert_equal_call_result_424157 = invoke(stypy.reporting.localization.Localization(__file__, 78, 12), assert_equal_424140, *[result_mul_424150, list_424151], **kwargs_424156)
        
        
        # Call to assert_equal(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'A' (line 79)
        A_424159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 26), 'A', False)
        # Getting the type of 'A' (line 79)
        A_424160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 28), 'A', False)
        # Applying the binary operator '-' (line 79)
        result_sub_424161 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 26), '-', A_424159, A_424160)
        
        
        # Obtaining an instance of the builtin type 'list' (line 79)
        list_424162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 79)
        # Adding element type (line 79)
        int_424163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 31), list_424162, int_424163)
        # Adding element type (line 79)
        int_424164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 31), list_424162, int_424164)
        # Adding element type (line 79)
        int_424165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 31), list_424162, int_424165)
        
        # Applying the binary operator '*' (line 79)
        result_mul_424166 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 25), '*', result_sub_424161, list_424162)
        
        
        # Obtaining an instance of the builtin type 'list' (line 79)
        list_424167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 79)
        # Adding element type (line 79)
        int_424168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 40), list_424167, int_424168)
        # Adding element type (line 79)
        int_424169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 40), list_424167, int_424169)
        
        # Processing the call keyword arguments (line 79)
        kwargs_424170 = {}
        # Getting the type of 'assert_equal' (line 79)
        assert_equal_424158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 79)
        assert_equal_call_result_424171 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), assert_equal_424158, *[result_mul_424166, list_424167], **kwargs_424170)
        
        
        # Call to assert_equal(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'A' (line 80)
        A_424173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 26), 'A', False)
        # Getting the type of 'A' (line 80)
        A_424174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 28), 'A', False)
        # Applying the binary operator '-' (line 80)
        result_sub_424175 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 26), '-', A_424173, A_424174)
        
        
        # Obtaining an instance of the builtin type 'list' (line 80)
        list_424176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 80)
        # Adding element type (line 80)
        
        # Obtaining an instance of the builtin type 'list' (line 80)
        list_424177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 80)
        # Adding element type (line 80)
        int_424178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 32), list_424177, int_424178)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 31), list_424176, list_424177)
        # Adding element type (line 80)
        
        # Obtaining an instance of the builtin type 'list' (line 80)
        list_424179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 80)
        # Adding element type (line 80)
        int_424180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 36), list_424179, int_424180)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 31), list_424176, list_424179)
        # Adding element type (line 80)
        
        # Obtaining an instance of the builtin type 'list' (line 80)
        list_424181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 80)
        # Adding element type (line 80)
        int_424182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 40), list_424181, int_424182)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 31), list_424176, list_424181)
        
        # Applying the binary operator '*' (line 80)
        result_mul_424183 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 25), '*', result_sub_424175, list_424176)
        
        
        # Obtaining an instance of the builtin type 'list' (line 80)
        list_424184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 80)
        # Adding element type (line 80)
        
        # Obtaining an instance of the builtin type 'list' (line 80)
        list_424185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 80)
        # Adding element type (line 80)
        int_424186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 47), list_424185, int_424186)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 46), list_424184, list_424185)
        # Adding element type (line 80)
        
        # Obtaining an instance of the builtin type 'list' (line 80)
        list_424187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 80)
        # Adding element type (line 80)
        int_424188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 51), list_424187, int_424188)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 46), list_424184, list_424187)
        
        # Processing the call keyword arguments (line 80)
        kwargs_424189 = {}
        # Getting the type of 'assert_equal' (line 80)
        assert_equal_424172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 80)
        assert_equal_call_result_424190 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), assert_equal_424172, *[result_mul_424183, list_424184], **kwargs_424189)
        
        
        # Assigning a BinOp to a Name (line 82):
        
        # Assigning a BinOp to a Name (line 82):
        # Getting the type of 'A' (line 82)
        A_424191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'A')
        # Getting the type of 'A' (line 82)
        A_424192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 18), 'A')
        # Applying the binary operator '+' (line 82)
        result_add_424193 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 16), '+', A_424191, A_424192)
        
        # Assigning a type to the variable 'z' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'z', result_add_424193)
        
        # Call to assert_(...): (line 83)
        # Processing the call arguments (line 83)
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'z' (line 83)
        z_424196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), 'z', False)
        # Obtaining the member 'args' of a type (line 83)
        args_424197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 24), z_424196, 'args')
        # Processing the call keyword arguments (line 83)
        kwargs_424198 = {}
        # Getting the type of 'len' (line 83)
        len_424195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'len', False)
        # Calling len(args, kwargs) (line 83)
        len_call_result_424199 = invoke(stypy.reporting.localization.Localization(__file__, 83, 20), len_424195, *[args_424197], **kwargs_424198)
        
        int_424200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 35), 'int')
        # Applying the binary operator '==' (line 83)
        result_eq_424201 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 20), '==', len_call_result_424199, int_424200)
        
        
        
        # Obtaining the type of the subscript
        int_424202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 48), 'int')
        # Getting the type of 'z' (line 83)
        z_424203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 41), 'z', False)
        # Obtaining the member 'args' of a type (line 83)
        args_424204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 41), z_424203, 'args')
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___424205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 41), args_424204, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_424206 = invoke(stypy.reporting.localization.Localization(__file__, 83, 41), getitem___424205, int_424202)
        
        # Getting the type of 'A' (line 83)
        A_424207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 54), 'A', False)
        # Applying the binary operator 'is' (line 83)
        result_is__424208 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 41), 'is', subscript_call_result_424206, A_424207)
        
        # Applying the binary operator 'and' (line 83)
        result_and_keyword_424209 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 20), 'and', result_eq_424201, result_is__424208)
        
        
        # Obtaining the type of the subscript
        int_424210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 67), 'int')
        # Getting the type of 'z' (line 83)
        z_424211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 60), 'z', False)
        # Obtaining the member 'args' of a type (line 83)
        args_424212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 60), z_424211, 'args')
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___424213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 60), args_424212, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_424214 = invoke(stypy.reporting.localization.Localization(__file__, 83, 60), getitem___424213, int_424210)
        
        # Getting the type of 'A' (line 83)
        A_424215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 73), 'A', False)
        # Applying the binary operator 'is' (line 83)
        result_is__424216 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 60), 'is', subscript_call_result_424214, A_424215)
        
        # Applying the binary operator 'and' (line 83)
        result_and_keyword_424217 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 20), 'and', result_and_keyword_424209, result_is__424216)
        
        # Processing the call keyword arguments (line 83)
        kwargs_424218 = {}
        # Getting the type of 'assert_' (line 83)
        assert__424194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 83)
        assert__call_result_424219 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), assert__424194, *[result_and_keyword_424217], **kwargs_424218)
        
        
        # Assigning a BinOp to a Name (line 84):
        
        # Assigning a BinOp to a Name (line 84):
        int_424220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 16), 'int')
        # Getting the type of 'A' (line 84)
        A_424221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 18), 'A')
        # Applying the binary operator '*' (line 84)
        result_mul_424222 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 16), '*', int_424220, A_424221)
        
        # Assigning a type to the variable 'z' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'z', result_mul_424222)
        
        # Call to assert_(...): (line 85)
        # Processing the call arguments (line 85)
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'z' (line 85)
        z_424225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 24), 'z', False)
        # Obtaining the member 'args' of a type (line 85)
        args_424226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 24), z_424225, 'args')
        # Processing the call keyword arguments (line 85)
        kwargs_424227 = {}
        # Getting the type of 'len' (line 85)
        len_424224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 20), 'len', False)
        # Calling len(args, kwargs) (line 85)
        len_call_result_424228 = invoke(stypy.reporting.localization.Localization(__file__, 85, 20), len_424224, *[args_424226], **kwargs_424227)
        
        int_424229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 35), 'int')
        # Applying the binary operator '==' (line 85)
        result_eq_424230 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 20), '==', len_call_result_424228, int_424229)
        
        
        
        # Obtaining the type of the subscript
        int_424231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 48), 'int')
        # Getting the type of 'z' (line 85)
        z_424232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 41), 'z', False)
        # Obtaining the member 'args' of a type (line 85)
        args_424233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 41), z_424232, 'args')
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___424234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 41), args_424233, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_424235 = invoke(stypy.reporting.localization.Localization(__file__, 85, 41), getitem___424234, int_424231)
        
        # Getting the type of 'A' (line 85)
        A_424236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 54), 'A', False)
        # Applying the binary operator 'is' (line 85)
        result_is__424237 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 41), 'is', subscript_call_result_424235, A_424236)
        
        # Applying the binary operator 'and' (line 85)
        result_and_keyword_424238 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 20), 'and', result_eq_424230, result_is__424237)
        
        
        # Obtaining the type of the subscript
        int_424239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 67), 'int')
        # Getting the type of 'z' (line 85)
        z_424240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 60), 'z', False)
        # Obtaining the member 'args' of a type (line 85)
        args_424241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 60), z_424240, 'args')
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___424242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 60), args_424241, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_424243 = invoke(stypy.reporting.localization.Localization(__file__, 85, 60), getitem___424242, int_424239)
        
        int_424244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 73), 'int')
        # Applying the binary operator '==' (line 85)
        result_eq_424245 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 60), '==', subscript_call_result_424243, int_424244)
        
        # Applying the binary operator 'and' (line 85)
        result_and_keyword_424246 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 20), 'and', result_and_keyword_424238, result_eq_424245)
        
        # Processing the call keyword arguments (line 85)
        kwargs_424247 = {}
        # Getting the type of 'assert_' (line 85)
        assert__424223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 85)
        assert__call_result_424248 = invoke(stypy.reporting.localization.Localization(__file__, 85, 12), assert__424223, *[result_and_keyword_424246], **kwargs_424247)
        
        
        # Call to assert_(...): (line 87)
        # Processing the call arguments (line 87)
        
        # Call to isinstance(...): (line 87)
        # Processing the call arguments (line 87)
        
        # Call to matvec(...): (line 87)
        # Processing the call arguments (line 87)
        
        # Obtaining an instance of the builtin type 'list' (line 87)
        list_424253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 87)
        # Adding element type (line 87)
        int_424254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 40), list_424253, int_424254)
        # Adding element type (line 87)
        int_424255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 40), list_424253, int_424255)
        # Adding element type (line 87)
        int_424256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 40), list_424253, int_424256)
        
        # Processing the call keyword arguments (line 87)
        kwargs_424257 = {}
        # Getting the type of 'A' (line 87)
        A_424251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 31), 'A', False)
        # Obtaining the member 'matvec' of a type (line 87)
        matvec_424252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 31), A_424251, 'matvec')
        # Calling matvec(args, kwargs) (line 87)
        matvec_call_result_424258 = invoke(stypy.reporting.localization.Localization(__file__, 87, 31), matvec_424252, *[list_424253], **kwargs_424257)
        
        # Getting the type of 'np' (line 87)
        np_424259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 52), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 87)
        ndarray_424260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 52), np_424259, 'ndarray')
        # Processing the call keyword arguments (line 87)
        kwargs_424261 = {}
        # Getting the type of 'isinstance' (line 87)
        isinstance_424250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 87)
        isinstance_call_result_424262 = invoke(stypy.reporting.localization.Localization(__file__, 87, 20), isinstance_424250, *[matvec_call_result_424258, ndarray_424260], **kwargs_424261)
        
        # Processing the call keyword arguments (line 87)
        kwargs_424263 = {}
        # Getting the type of 'assert_' (line 87)
        assert__424249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 87)
        assert__call_result_424264 = invoke(stypy.reporting.localization.Localization(__file__, 87, 12), assert__424249, *[isinstance_call_result_424262], **kwargs_424263)
        
        
        # Call to assert_(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Call to isinstance(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Call to matvec(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Call to array(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Obtaining an instance of the builtin type 'list' (line 88)
        list_424271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 88)
        # Adding element type (line 88)
        
        # Obtaining an instance of the builtin type 'list' (line 88)
        list_424272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 88)
        # Adding element type (line 88)
        int_424273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 50), list_424272, int_424273)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 49), list_424271, list_424272)
        # Adding element type (line 88)
        
        # Obtaining an instance of the builtin type 'list' (line 88)
        list_424274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 54), 'list')
        # Adding type elements to the builtin type 'list' instance (line 88)
        # Adding element type (line 88)
        int_424275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 54), list_424274, int_424275)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 49), list_424271, list_424274)
        # Adding element type (line 88)
        
        # Obtaining an instance of the builtin type 'list' (line 88)
        list_424276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 88)
        # Adding element type (line 88)
        int_424277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 58), list_424276, int_424277)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 49), list_424271, list_424276)
        
        # Processing the call keyword arguments (line 88)
        kwargs_424278 = {}
        # Getting the type of 'np' (line 88)
        np_424269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 40), 'np', False)
        # Obtaining the member 'array' of a type (line 88)
        array_424270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 40), np_424269, 'array')
        # Calling array(args, kwargs) (line 88)
        array_call_result_424279 = invoke(stypy.reporting.localization.Localization(__file__, 88, 40), array_424270, *[list_424271], **kwargs_424278)
        
        # Processing the call keyword arguments (line 88)
        kwargs_424280 = {}
        # Getting the type of 'A' (line 88)
        A_424267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 31), 'A', False)
        # Obtaining the member 'matvec' of a type (line 88)
        matvec_424268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 31), A_424267, 'matvec')
        # Calling matvec(args, kwargs) (line 88)
        matvec_call_result_424281 = invoke(stypy.reporting.localization.Localization(__file__, 88, 31), matvec_424268, *[array_call_result_424279], **kwargs_424280)
        
        # Getting the type of 'np' (line 88)
        np_424282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 66), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 88)
        ndarray_424283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 66), np_424282, 'ndarray')
        # Processing the call keyword arguments (line 88)
        kwargs_424284 = {}
        # Getting the type of 'isinstance' (line 88)
        isinstance_424266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 88)
        isinstance_call_result_424285 = invoke(stypy.reporting.localization.Localization(__file__, 88, 20), isinstance_424266, *[matvec_call_result_424281, ndarray_424283], **kwargs_424284)
        
        # Processing the call keyword arguments (line 88)
        kwargs_424286 = {}
        # Getting the type of 'assert_' (line 88)
        assert__424265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 88)
        assert__call_result_424287 = invoke(stypy.reporting.localization.Localization(__file__, 88, 12), assert__424265, *[isinstance_call_result_424285], **kwargs_424286)
        
        
        # Call to assert_(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Call to isinstance(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'A' (line 89)
        A_424290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 31), 'A', False)
        
        # Call to array(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Obtaining an instance of the builtin type 'list' (line 89)
        list_424293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 89)
        # Adding element type (line 89)
        int_424294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 44), list_424293, int_424294)
        # Adding element type (line 89)
        int_424295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 44), list_424293, int_424295)
        # Adding element type (line 89)
        int_424296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 44), list_424293, int_424296)
        
        # Processing the call keyword arguments (line 89)
        kwargs_424297 = {}
        # Getting the type of 'np' (line 89)
        np_424291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 35), 'np', False)
        # Obtaining the member 'array' of a type (line 89)
        array_424292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 35), np_424291, 'array')
        # Calling array(args, kwargs) (line 89)
        array_call_result_424298 = invoke(stypy.reporting.localization.Localization(__file__, 89, 35), array_424292, *[list_424293], **kwargs_424297)
        
        # Applying the binary operator '*' (line 89)
        result_mul_424299 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 31), '*', A_424290, array_call_result_424298)
        
        # Getting the type of 'np' (line 89)
        np_424300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 54), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 89)
        ndarray_424301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 54), np_424300, 'ndarray')
        # Processing the call keyword arguments (line 89)
        kwargs_424302 = {}
        # Getting the type of 'isinstance' (line 89)
        isinstance_424289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 89)
        isinstance_call_result_424303 = invoke(stypy.reporting.localization.Localization(__file__, 89, 20), isinstance_424289, *[result_mul_424299, ndarray_424301], **kwargs_424302)
        
        # Processing the call keyword arguments (line 89)
        kwargs_424304 = {}
        # Getting the type of 'assert_' (line 89)
        assert__424288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 89)
        assert__call_result_424305 = invoke(stypy.reporting.localization.Localization(__file__, 89, 12), assert__424288, *[isinstance_call_result_424303], **kwargs_424304)
        
        
        # Call to assert_(...): (line 90)
        # Processing the call arguments (line 90)
        
        # Call to isinstance(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'A' (line 90)
        A_424308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 31), 'A', False)
        
        # Call to array(...): (line 90)
        # Processing the call arguments (line 90)
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_424311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_424312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        int_424313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 45), list_424312, int_424313)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 44), list_424311, list_424312)
        # Adding element type (line 90)
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_424314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        int_424315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 49), list_424314, int_424315)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 44), list_424311, list_424314)
        # Adding element type (line 90)
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_424316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        int_424317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 53), list_424316, int_424317)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 44), list_424311, list_424316)
        
        # Processing the call keyword arguments (line 90)
        kwargs_424318 = {}
        # Getting the type of 'np' (line 90)
        np_424309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 35), 'np', False)
        # Obtaining the member 'array' of a type (line 90)
        array_424310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 35), np_424309, 'array')
        # Calling array(args, kwargs) (line 90)
        array_call_result_424319 = invoke(stypy.reporting.localization.Localization(__file__, 90, 35), array_424310, *[list_424311], **kwargs_424318)
        
        # Applying the binary operator '*' (line 90)
        result_mul_424320 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 31), '*', A_424308, array_call_result_424319)
        
        # Getting the type of 'np' (line 90)
        np_424321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 60), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 90)
        ndarray_424322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 60), np_424321, 'ndarray')
        # Processing the call keyword arguments (line 90)
        kwargs_424323 = {}
        # Getting the type of 'isinstance' (line 90)
        isinstance_424307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 90)
        isinstance_call_result_424324 = invoke(stypy.reporting.localization.Localization(__file__, 90, 20), isinstance_424307, *[result_mul_424320, ndarray_424322], **kwargs_424323)
        
        # Processing the call keyword arguments (line 90)
        kwargs_424325 = {}
        # Getting the type of 'assert_' (line 90)
        assert__424306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 90)
        assert__call_result_424326 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), assert__424306, *[isinstance_call_result_424324], **kwargs_424325)
        
        
        # Call to assert_(...): (line 91)
        # Processing the call arguments (line 91)
        
        # Call to isinstance(...): (line 91)
        # Processing the call arguments (line 91)
        
        # Call to dot(...): (line 91)
        # Processing the call arguments (line 91)
        
        # Call to array(...): (line 91)
        # Processing the call arguments (line 91)
        
        # Obtaining an instance of the builtin type 'list' (line 91)
        list_424333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 91)
        # Adding element type (line 91)
        int_424334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 46), list_424333, int_424334)
        # Adding element type (line 91)
        int_424335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 46), list_424333, int_424335)
        # Adding element type (line 91)
        int_424336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 46), list_424333, int_424336)
        
        # Processing the call keyword arguments (line 91)
        kwargs_424337 = {}
        # Getting the type of 'np' (line 91)
        np_424331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 37), 'np', False)
        # Obtaining the member 'array' of a type (line 91)
        array_424332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 37), np_424331, 'array')
        # Calling array(args, kwargs) (line 91)
        array_call_result_424338 = invoke(stypy.reporting.localization.Localization(__file__, 91, 37), array_424332, *[list_424333], **kwargs_424337)
        
        # Processing the call keyword arguments (line 91)
        kwargs_424339 = {}
        # Getting the type of 'A' (line 91)
        A_424329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 31), 'A', False)
        # Obtaining the member 'dot' of a type (line 91)
        dot_424330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 31), A_424329, 'dot')
        # Calling dot(args, kwargs) (line 91)
        dot_call_result_424340 = invoke(stypy.reporting.localization.Localization(__file__, 91, 31), dot_424330, *[array_call_result_424338], **kwargs_424339)
        
        # Getting the type of 'np' (line 91)
        np_424341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 57), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 91)
        ndarray_424342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 57), np_424341, 'ndarray')
        # Processing the call keyword arguments (line 91)
        kwargs_424343 = {}
        # Getting the type of 'isinstance' (line 91)
        isinstance_424328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 91)
        isinstance_call_result_424344 = invoke(stypy.reporting.localization.Localization(__file__, 91, 20), isinstance_424328, *[dot_call_result_424340, ndarray_424342], **kwargs_424343)
        
        # Processing the call keyword arguments (line 91)
        kwargs_424345 = {}
        # Getting the type of 'assert_' (line 91)
        assert__424327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 91)
        assert__call_result_424346 = invoke(stypy.reporting.localization.Localization(__file__, 91, 12), assert__424327, *[isinstance_call_result_424344], **kwargs_424345)
        
        
        # Call to assert_(...): (line 92)
        # Processing the call arguments (line 92)
        
        # Call to isinstance(...): (line 92)
        # Processing the call arguments (line 92)
        
        # Call to dot(...): (line 92)
        # Processing the call arguments (line 92)
        
        # Call to array(...): (line 92)
        # Processing the call arguments (line 92)
        
        # Obtaining an instance of the builtin type 'list' (line 92)
        list_424353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 92)
        # Adding element type (line 92)
        
        # Obtaining an instance of the builtin type 'list' (line 92)
        list_424354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 92)
        # Adding element type (line 92)
        int_424355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 47), list_424354, int_424355)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 46), list_424353, list_424354)
        # Adding element type (line 92)
        
        # Obtaining an instance of the builtin type 'list' (line 92)
        list_424356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 92)
        # Adding element type (line 92)
        int_424357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 51), list_424356, int_424357)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 46), list_424353, list_424356)
        # Adding element type (line 92)
        
        # Obtaining an instance of the builtin type 'list' (line 92)
        list_424358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 92)
        # Adding element type (line 92)
        int_424359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 55), list_424358, int_424359)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 46), list_424353, list_424358)
        
        # Processing the call keyword arguments (line 92)
        kwargs_424360 = {}
        # Getting the type of 'np' (line 92)
        np_424351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 37), 'np', False)
        # Obtaining the member 'array' of a type (line 92)
        array_424352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 37), np_424351, 'array')
        # Calling array(args, kwargs) (line 92)
        array_call_result_424361 = invoke(stypy.reporting.localization.Localization(__file__, 92, 37), array_424352, *[list_424353], **kwargs_424360)
        
        # Processing the call keyword arguments (line 92)
        kwargs_424362 = {}
        # Getting the type of 'A' (line 92)
        A_424349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 31), 'A', False)
        # Obtaining the member 'dot' of a type (line 92)
        dot_424350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 31), A_424349, 'dot')
        # Calling dot(args, kwargs) (line 92)
        dot_call_result_424363 = invoke(stypy.reporting.localization.Localization(__file__, 92, 31), dot_424350, *[array_call_result_424361], **kwargs_424362)
        
        # Getting the type of 'np' (line 92)
        np_424364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 63), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 92)
        ndarray_424365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 63), np_424364, 'ndarray')
        # Processing the call keyword arguments (line 92)
        kwargs_424366 = {}
        # Getting the type of 'isinstance' (line 92)
        isinstance_424348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 92)
        isinstance_call_result_424367 = invoke(stypy.reporting.localization.Localization(__file__, 92, 20), isinstance_424348, *[dot_call_result_424363, ndarray_424365], **kwargs_424366)
        
        # Processing the call keyword arguments (line 92)
        kwargs_424368 = {}
        # Getting the type of 'assert_' (line 92)
        assert__424347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 92)
        assert__call_result_424369 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), assert__424347, *[isinstance_call_result_424367], **kwargs_424368)
        
        
        # Call to assert_(...): (line 94)
        # Processing the call arguments (line 94)
        
        # Call to isinstance(...): (line 94)
        # Processing the call arguments (line 94)
        
        # Call to matvec(...): (line 94)
        # Processing the call arguments (line 94)
        
        # Call to matrix(...): (line 94)
        # Processing the call arguments (line 94)
        
        # Obtaining an instance of the builtin type 'list' (line 94)
        list_424376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 94)
        # Adding element type (line 94)
        
        # Obtaining an instance of the builtin type 'list' (line 94)
        list_424377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 94)
        # Adding element type (line 94)
        int_424378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 51), list_424377, int_424378)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 50), list_424376, list_424377)
        # Adding element type (line 94)
        
        # Obtaining an instance of the builtin type 'list' (line 94)
        list_424379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 94)
        # Adding element type (line 94)
        int_424380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 55), list_424379, int_424380)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 50), list_424376, list_424379)
        # Adding element type (line 94)
        
        # Obtaining an instance of the builtin type 'list' (line 94)
        list_424381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 59), 'list')
        # Adding type elements to the builtin type 'list' instance (line 94)
        # Adding element type (line 94)
        int_424382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 59), list_424381, int_424382)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 50), list_424376, list_424381)
        
        # Processing the call keyword arguments (line 94)
        kwargs_424383 = {}
        # Getting the type of 'np' (line 94)
        np_424374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 40), 'np', False)
        # Obtaining the member 'matrix' of a type (line 94)
        matrix_424375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 40), np_424374, 'matrix')
        # Calling matrix(args, kwargs) (line 94)
        matrix_call_result_424384 = invoke(stypy.reporting.localization.Localization(__file__, 94, 40), matrix_424375, *[list_424376], **kwargs_424383)
        
        # Processing the call keyword arguments (line 94)
        kwargs_424385 = {}
        # Getting the type of 'A' (line 94)
        A_424372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 31), 'A', False)
        # Obtaining the member 'matvec' of a type (line 94)
        matvec_424373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 31), A_424372, 'matvec')
        # Calling matvec(args, kwargs) (line 94)
        matvec_call_result_424386 = invoke(stypy.reporting.localization.Localization(__file__, 94, 31), matvec_424373, *[matrix_call_result_424384], **kwargs_424385)
        
        # Getting the type of 'np' (line 94)
        np_424387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 67), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 94)
        ndarray_424388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 67), np_424387, 'ndarray')
        # Processing the call keyword arguments (line 94)
        kwargs_424389 = {}
        # Getting the type of 'isinstance' (line 94)
        isinstance_424371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 94)
        isinstance_call_result_424390 = invoke(stypy.reporting.localization.Localization(__file__, 94, 20), isinstance_424371, *[matvec_call_result_424386, ndarray_424388], **kwargs_424389)
        
        # Processing the call keyword arguments (line 94)
        kwargs_424391 = {}
        # Getting the type of 'assert_' (line 94)
        assert__424370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 94)
        assert__call_result_424392 = invoke(stypy.reporting.localization.Localization(__file__, 94, 12), assert__424370, *[isinstance_call_result_424390], **kwargs_424391)
        
        
        # Call to assert_(...): (line 95)
        # Processing the call arguments (line 95)
        
        # Call to isinstance(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'A' (line 95)
        A_424395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 31), 'A', False)
        
        # Call to matrix(...): (line 95)
        # Processing the call arguments (line 95)
        
        # Obtaining an instance of the builtin type 'list' (line 95)
        list_424398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 95)
        # Adding element type (line 95)
        
        # Obtaining an instance of the builtin type 'list' (line 95)
        list_424399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 95)
        # Adding element type (line 95)
        int_424400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 46), list_424399, int_424400)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 45), list_424398, list_424399)
        # Adding element type (line 95)
        
        # Obtaining an instance of the builtin type 'list' (line 95)
        list_424401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 95)
        # Adding element type (line 95)
        int_424402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 50), list_424401, int_424402)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 45), list_424398, list_424401)
        # Adding element type (line 95)
        
        # Obtaining an instance of the builtin type 'list' (line 95)
        list_424403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 54), 'list')
        # Adding type elements to the builtin type 'list' instance (line 95)
        # Adding element type (line 95)
        int_424404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 54), list_424403, int_424404)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 45), list_424398, list_424403)
        
        # Processing the call keyword arguments (line 95)
        kwargs_424405 = {}
        # Getting the type of 'np' (line 95)
        np_424396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 35), 'np', False)
        # Obtaining the member 'matrix' of a type (line 95)
        matrix_424397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 35), np_424396, 'matrix')
        # Calling matrix(args, kwargs) (line 95)
        matrix_call_result_424406 = invoke(stypy.reporting.localization.Localization(__file__, 95, 35), matrix_424397, *[list_424398], **kwargs_424405)
        
        # Applying the binary operator '*' (line 95)
        result_mul_424407 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 31), '*', A_424395, matrix_call_result_424406)
        
        # Getting the type of 'np' (line 95)
        np_424408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 61), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 95)
        ndarray_424409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 61), np_424408, 'ndarray')
        # Processing the call keyword arguments (line 95)
        kwargs_424410 = {}
        # Getting the type of 'isinstance' (line 95)
        isinstance_424394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 95)
        isinstance_call_result_424411 = invoke(stypy.reporting.localization.Localization(__file__, 95, 20), isinstance_424394, *[result_mul_424407, ndarray_424409], **kwargs_424410)
        
        # Processing the call keyword arguments (line 95)
        kwargs_424412 = {}
        # Getting the type of 'assert_' (line 95)
        assert__424393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 95)
        assert__call_result_424413 = invoke(stypy.reporting.localization.Localization(__file__, 95, 12), assert__424393, *[isinstance_call_result_424411], **kwargs_424412)
        
        
        # Call to assert_(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Call to isinstance(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Call to dot(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Call to matrix(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_424420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_424421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        int_424422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 48), list_424421, int_424422)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 47), list_424420, list_424421)
        # Adding element type (line 96)
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_424423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        int_424424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 52), list_424423, int_424424)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 47), list_424420, list_424423)
        # Adding element type (line 96)
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_424425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        int_424426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 56), list_424425, int_424426)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 47), list_424420, list_424425)
        
        # Processing the call keyword arguments (line 96)
        kwargs_424427 = {}
        # Getting the type of 'np' (line 96)
        np_424418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 37), 'np', False)
        # Obtaining the member 'matrix' of a type (line 96)
        matrix_424419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 37), np_424418, 'matrix')
        # Calling matrix(args, kwargs) (line 96)
        matrix_call_result_424428 = invoke(stypy.reporting.localization.Localization(__file__, 96, 37), matrix_424419, *[list_424420], **kwargs_424427)
        
        # Processing the call keyword arguments (line 96)
        kwargs_424429 = {}
        # Getting the type of 'A' (line 96)
        A_424416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 31), 'A', False)
        # Obtaining the member 'dot' of a type (line 96)
        dot_424417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 31), A_424416, 'dot')
        # Calling dot(args, kwargs) (line 96)
        dot_call_result_424430 = invoke(stypy.reporting.localization.Localization(__file__, 96, 31), dot_424417, *[matrix_call_result_424428], **kwargs_424429)
        
        # Getting the type of 'np' (line 96)
        np_424431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 64), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 96)
        ndarray_424432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 64), np_424431, 'ndarray')
        # Processing the call keyword arguments (line 96)
        kwargs_424433 = {}
        # Getting the type of 'isinstance' (line 96)
        isinstance_424415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 96)
        isinstance_call_result_424434 = invoke(stypy.reporting.localization.Localization(__file__, 96, 20), isinstance_424415, *[dot_call_result_424430, ndarray_424432], **kwargs_424433)
        
        # Processing the call keyword arguments (line 96)
        kwargs_424435 = {}
        # Getting the type of 'assert_' (line 96)
        assert__424414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 96)
        assert__call_result_424436 = invoke(stypy.reporting.localization.Localization(__file__, 96, 12), assert__424414, *[isinstance_call_result_424434], **kwargs_424435)
        
        
        # Call to assert_(...): (line 98)
        # Processing the call arguments (line 98)
        
        # Call to isinstance(...): (line 98)
        # Processing the call arguments (line 98)
        int_424439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 31), 'int')
        # Getting the type of 'A' (line 98)
        A_424440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 33), 'A', False)
        # Applying the binary operator '*' (line 98)
        result_mul_424441 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 31), '*', int_424439, A_424440)
        
        # Getting the type of 'interface' (line 98)
        interface_424442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 36), 'interface', False)
        # Obtaining the member '_ScaledLinearOperator' of a type (line 98)
        _ScaledLinearOperator_424443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 36), interface_424442, '_ScaledLinearOperator')
        # Processing the call keyword arguments (line 98)
        kwargs_424444 = {}
        # Getting the type of 'isinstance' (line 98)
        isinstance_424438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 98)
        isinstance_call_result_424445 = invoke(stypy.reporting.localization.Localization(__file__, 98, 20), isinstance_424438, *[result_mul_424441, _ScaledLinearOperator_424443], **kwargs_424444)
        
        # Processing the call keyword arguments (line 98)
        kwargs_424446 = {}
        # Getting the type of 'assert_' (line 98)
        assert__424437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 98)
        assert__call_result_424447 = invoke(stypy.reporting.localization.Localization(__file__, 98, 12), assert__424437, *[isinstance_call_result_424445], **kwargs_424446)
        
        
        # Call to assert_(...): (line 99)
        # Processing the call arguments (line 99)
        
        # Call to isinstance(...): (line 99)
        # Processing the call arguments (line 99)
        complex_424450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 31), 'complex')
        # Getting the type of 'A' (line 99)
        A_424451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 34), 'A', False)
        # Applying the binary operator '*' (line 99)
        result_mul_424452 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 31), '*', complex_424450, A_424451)
        
        # Getting the type of 'interface' (line 99)
        interface_424453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 37), 'interface', False)
        # Obtaining the member '_ScaledLinearOperator' of a type (line 99)
        _ScaledLinearOperator_424454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 37), interface_424453, '_ScaledLinearOperator')
        # Processing the call keyword arguments (line 99)
        kwargs_424455 = {}
        # Getting the type of 'isinstance' (line 99)
        isinstance_424449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 99)
        isinstance_call_result_424456 = invoke(stypy.reporting.localization.Localization(__file__, 99, 20), isinstance_424449, *[result_mul_424452, _ScaledLinearOperator_424454], **kwargs_424455)
        
        # Processing the call keyword arguments (line 99)
        kwargs_424457 = {}
        # Getting the type of 'assert_' (line 99)
        assert__424448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 99)
        assert__call_result_424458 = invoke(stypy.reporting.localization.Localization(__file__, 99, 12), assert__424448, *[isinstance_call_result_424456], **kwargs_424457)
        
        
        # Call to assert_(...): (line 100)
        # Processing the call arguments (line 100)
        
        # Call to isinstance(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'A' (line 100)
        A_424461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 31), 'A', False)
        # Getting the type of 'A' (line 100)
        A_424462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 33), 'A', False)
        # Applying the binary operator '+' (line 100)
        result_add_424463 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 31), '+', A_424461, A_424462)
        
        # Getting the type of 'interface' (line 100)
        interface_424464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 36), 'interface', False)
        # Obtaining the member '_SumLinearOperator' of a type (line 100)
        _SumLinearOperator_424465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 36), interface_424464, '_SumLinearOperator')
        # Processing the call keyword arguments (line 100)
        kwargs_424466 = {}
        # Getting the type of 'isinstance' (line 100)
        isinstance_424460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 100)
        isinstance_call_result_424467 = invoke(stypy.reporting.localization.Localization(__file__, 100, 20), isinstance_424460, *[result_add_424463, _SumLinearOperator_424465], **kwargs_424466)
        
        # Processing the call keyword arguments (line 100)
        kwargs_424468 = {}
        # Getting the type of 'assert_' (line 100)
        assert__424459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 100)
        assert__call_result_424469 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), assert__424459, *[isinstance_call_result_424467], **kwargs_424468)
        
        
        # Call to assert_(...): (line 101)
        # Processing the call arguments (line 101)
        
        # Call to isinstance(...): (line 101)
        # Processing the call arguments (line 101)
        
        # Getting the type of 'A' (line 101)
        A_424472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 32), 'A', False)
        # Applying the 'usub' unary operator (line 101)
        result___neg___424473 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 31), 'usub', A_424472)
        
        # Getting the type of 'interface' (line 101)
        interface_424474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 35), 'interface', False)
        # Obtaining the member '_ScaledLinearOperator' of a type (line 101)
        _ScaledLinearOperator_424475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 35), interface_424474, '_ScaledLinearOperator')
        # Processing the call keyword arguments (line 101)
        kwargs_424476 = {}
        # Getting the type of 'isinstance' (line 101)
        isinstance_424471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 101)
        isinstance_call_result_424477 = invoke(stypy.reporting.localization.Localization(__file__, 101, 20), isinstance_424471, *[result___neg___424473, _ScaledLinearOperator_424475], **kwargs_424476)
        
        # Processing the call keyword arguments (line 101)
        kwargs_424478 = {}
        # Getting the type of 'assert_' (line 101)
        assert__424470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 101)
        assert__call_result_424479 = invoke(stypy.reporting.localization.Localization(__file__, 101, 12), assert__424470, *[isinstance_call_result_424477], **kwargs_424478)
        
        
        # Call to assert_(...): (line 102)
        # Processing the call arguments (line 102)
        
        # Call to isinstance(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'A' (line 102)
        A_424482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 31), 'A', False)
        # Getting the type of 'A' (line 102)
        A_424483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 33), 'A', False)
        # Applying the binary operator '-' (line 102)
        result_sub_424484 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 31), '-', A_424482, A_424483)
        
        # Getting the type of 'interface' (line 102)
        interface_424485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 36), 'interface', False)
        # Obtaining the member '_SumLinearOperator' of a type (line 102)
        _SumLinearOperator_424486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 36), interface_424485, '_SumLinearOperator')
        # Processing the call keyword arguments (line 102)
        kwargs_424487 = {}
        # Getting the type of 'isinstance' (line 102)
        isinstance_424481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 102)
        isinstance_call_result_424488 = invoke(stypy.reporting.localization.Localization(__file__, 102, 20), isinstance_424481, *[result_sub_424484, _SumLinearOperator_424486], **kwargs_424487)
        
        # Processing the call keyword arguments (line 102)
        kwargs_424489 = {}
        # Getting the type of 'assert_' (line 102)
        assert__424480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 102)
        assert__call_result_424490 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), assert__424480, *[isinstance_call_result_424488], **kwargs_424489)
        
        
        # Call to assert_(...): (line 104)
        # Processing the call arguments (line 104)
        
        complex_424492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 21), 'complex')
        # Getting the type of 'A' (line 104)
        A_424493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 24), 'A', False)
        # Applying the binary operator '*' (line 104)
        result_mul_424494 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 21), '*', complex_424492, A_424493)
        
        # Obtaining the member 'dtype' of a type (line 104)
        dtype_424495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 21), result_mul_424494, 'dtype')
        # Getting the type of 'np' (line 104)
        np_424496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 36), 'np', False)
        # Obtaining the member 'complex_' of a type (line 104)
        complex__424497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 36), np_424496, 'complex_')
        # Applying the binary operator '==' (line 104)
        result_eq_424498 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 20), '==', dtype_424495, complex__424497)
        
        # Processing the call keyword arguments (line 104)
        kwargs_424499 = {}
        # Getting the type of 'assert_' (line 104)
        assert__424491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 104)
        assert__call_result_424500 = invoke(stypy.reporting.localization.Localization(__file__, 104, 12), assert__424491, *[result_eq_424498], **kwargs_424499)
        
        
        # Call to assert_raises(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'ValueError' (line 106)
        ValueError_424502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 26), 'ValueError', False)
        # Getting the type of 'A' (line 106)
        A_424503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 38), 'A', False)
        # Obtaining the member 'matvec' of a type (line 106)
        matvec_424504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 38), A_424503, 'matvec')
        
        # Call to array(...): (line 106)
        # Processing the call arguments (line 106)
        
        # Obtaining an instance of the builtin type 'list' (line 106)
        list_424507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 106)
        # Adding element type (line 106)
        int_424508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 57), list_424507, int_424508)
        # Adding element type (line 106)
        int_424509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 57), list_424507, int_424509)
        
        # Processing the call keyword arguments (line 106)
        kwargs_424510 = {}
        # Getting the type of 'np' (line 106)
        np_424505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 48), 'np', False)
        # Obtaining the member 'array' of a type (line 106)
        array_424506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 48), np_424505, 'array')
        # Calling array(args, kwargs) (line 106)
        array_call_result_424511 = invoke(stypy.reporting.localization.Localization(__file__, 106, 48), array_424506, *[list_424507], **kwargs_424510)
        
        # Processing the call keyword arguments (line 106)
        kwargs_424512 = {}
        # Getting the type of 'assert_raises' (line 106)
        assert_raises_424501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 106)
        assert_raises_call_result_424513 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), assert_raises_424501, *[ValueError_424502, matvec_424504, array_call_result_424511], **kwargs_424512)
        
        
        # Call to assert_raises(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'ValueError' (line 107)
        ValueError_424515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 26), 'ValueError', False)
        # Getting the type of 'A' (line 107)
        A_424516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 38), 'A', False)
        # Obtaining the member 'matvec' of a type (line 107)
        matvec_424517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 38), A_424516, 'matvec')
        
        # Call to array(...): (line 107)
        # Processing the call arguments (line 107)
        
        # Obtaining an instance of the builtin type 'list' (line 107)
        list_424520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 107)
        # Adding element type (line 107)
        int_424521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 57), list_424520, int_424521)
        # Adding element type (line 107)
        int_424522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 57), list_424520, int_424522)
        # Adding element type (line 107)
        int_424523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 57), list_424520, int_424523)
        # Adding element type (line 107)
        int_424524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 64), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 57), list_424520, int_424524)
        
        # Processing the call keyword arguments (line 107)
        kwargs_424525 = {}
        # Getting the type of 'np' (line 107)
        np_424518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 48), 'np', False)
        # Obtaining the member 'array' of a type (line 107)
        array_424519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 48), np_424518, 'array')
        # Calling array(args, kwargs) (line 107)
        array_call_result_424526 = invoke(stypy.reporting.localization.Localization(__file__, 107, 48), array_424519, *[list_424520], **kwargs_424525)
        
        # Processing the call keyword arguments (line 107)
        kwargs_424527 = {}
        # Getting the type of 'assert_raises' (line 107)
        assert_raises_424514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 107)
        assert_raises_call_result_424528 = invoke(stypy.reporting.localization.Localization(__file__, 107, 12), assert_raises_424514, *[ValueError_424515, matvec_424517, array_call_result_424526], **kwargs_424527)
        
        
        # Call to assert_raises(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'ValueError' (line 108)
        ValueError_424530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 26), 'ValueError', False)
        # Getting the type of 'A' (line 108)
        A_424531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 38), 'A', False)
        # Obtaining the member 'matvec' of a type (line 108)
        matvec_424532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 38), A_424531, 'matvec')
        
        # Call to array(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_424535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_424536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        int_424537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 58), list_424536, int_424537)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 57), list_424535, list_424536)
        # Adding element type (line 108)
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_424538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 62), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        int_424539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 62), list_424538, int_424539)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 57), list_424535, list_424538)
        
        # Processing the call keyword arguments (line 108)
        kwargs_424540 = {}
        # Getting the type of 'np' (line 108)
        np_424533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 48), 'np', False)
        # Obtaining the member 'array' of a type (line 108)
        array_424534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 48), np_424533, 'array')
        # Calling array(args, kwargs) (line 108)
        array_call_result_424541 = invoke(stypy.reporting.localization.Localization(__file__, 108, 48), array_424534, *[list_424535], **kwargs_424540)
        
        # Processing the call keyword arguments (line 108)
        kwargs_424542 = {}
        # Getting the type of 'assert_raises' (line 108)
        assert_raises_424529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 108)
        assert_raises_call_result_424543 = invoke(stypy.reporting.localization.Localization(__file__, 108, 12), assert_raises_424529, *[ValueError_424530, matvec_424532, array_call_result_424541], **kwargs_424542)
        
        
        # Call to assert_raises(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'ValueError' (line 109)
        ValueError_424545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 26), 'ValueError', False)
        # Getting the type of 'A' (line 109)
        A_424546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 38), 'A', False)
        # Obtaining the member 'matvec' of a type (line 109)
        matvec_424547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 38), A_424546, 'matvec')
        
        # Call to array(...): (line 109)
        # Processing the call arguments (line 109)
        
        # Obtaining an instance of the builtin type 'list' (line 109)
        list_424550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 109)
        # Adding element type (line 109)
        
        # Obtaining an instance of the builtin type 'list' (line 109)
        list_424551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 109)
        # Adding element type (line 109)
        int_424552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 58), list_424551, int_424552)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 57), list_424550, list_424551)
        # Adding element type (line 109)
        
        # Obtaining an instance of the builtin type 'list' (line 109)
        list_424553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 62), 'list')
        # Adding type elements to the builtin type 'list' instance (line 109)
        # Adding element type (line 109)
        int_424554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 62), list_424553, int_424554)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 57), list_424550, list_424553)
        # Adding element type (line 109)
        
        # Obtaining an instance of the builtin type 'list' (line 109)
        list_424555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 66), 'list')
        # Adding type elements to the builtin type 'list' instance (line 109)
        # Adding element type (line 109)
        int_424556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 67), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 66), list_424555, int_424556)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 57), list_424550, list_424555)
        # Adding element type (line 109)
        
        # Obtaining an instance of the builtin type 'list' (line 109)
        list_424557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 70), 'list')
        # Adding type elements to the builtin type 'list' instance (line 109)
        # Adding element type (line 109)
        int_424558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 71), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 70), list_424557, int_424558)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 57), list_424550, list_424557)
        
        # Processing the call keyword arguments (line 109)
        kwargs_424559 = {}
        # Getting the type of 'np' (line 109)
        np_424548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 48), 'np', False)
        # Obtaining the member 'array' of a type (line 109)
        array_424549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 48), np_424548, 'array')
        # Calling array(args, kwargs) (line 109)
        array_call_result_424560 = invoke(stypy.reporting.localization.Localization(__file__, 109, 48), array_424549, *[list_424550], **kwargs_424559)
        
        # Processing the call keyword arguments (line 109)
        kwargs_424561 = {}
        # Getting the type of 'assert_raises' (line 109)
        assert_raises_424544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 109)
        assert_raises_call_result_424562 = invoke(stypy.reporting.localization.Localization(__file__, 109, 12), assert_raises_424544, *[ValueError_424545, matvec_424547, array_call_result_424560], **kwargs_424561)
        
        
        # Call to assert_raises(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'ValueError' (line 111)
        ValueError_424564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 26), 'ValueError', False)

        @norecursion
        def _stypy_temp_lambda_231(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_231'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_231', 111, 38, True)
            # Passed parameters checking function
            _stypy_temp_lambda_231.stypy_localization = localization
            _stypy_temp_lambda_231.stypy_type_of_self = None
            _stypy_temp_lambda_231.stypy_type_store = module_type_store
            _stypy_temp_lambda_231.stypy_function_name = '_stypy_temp_lambda_231'
            _stypy_temp_lambda_231.stypy_param_names_list = []
            _stypy_temp_lambda_231.stypy_varargs_param_name = None
            _stypy_temp_lambda_231.stypy_kwargs_param_name = None
            _stypy_temp_lambda_231.stypy_call_defaults = defaults
            _stypy_temp_lambda_231.stypy_call_varargs = varargs
            _stypy_temp_lambda_231.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_231', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_231', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'A' (line 111)
            A_424565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 46), 'A', False)
            # Getting the type of 'A' (line 111)
            A_424566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 48), 'A', False)
            # Applying the binary operator '*' (line 111)
            result_mul_424567 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 46), '*', A_424565, A_424566)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 111)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 38), 'stypy_return_type', result_mul_424567)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_231' in the type store
            # Getting the type of 'stypy_return_type' (line 111)
            stypy_return_type_424568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 38), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_424568)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_231'
            return stypy_return_type_424568

        # Assigning a type to the variable '_stypy_temp_lambda_231' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 38), '_stypy_temp_lambda_231', _stypy_temp_lambda_231)
        # Getting the type of '_stypy_temp_lambda_231' (line 111)
        _stypy_temp_lambda_231_424569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 38), '_stypy_temp_lambda_231')
        # Processing the call keyword arguments (line 111)
        kwargs_424570 = {}
        # Getting the type of 'assert_raises' (line 111)
        assert_raises_424563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 111)
        assert_raises_call_result_424571 = invoke(stypy.reporting.localization.Localization(__file__, 111, 12), assert_raises_424563, *[ValueError_424564, _stypy_temp_lambda_231_424569], **kwargs_424570)
        
        
        # Call to assert_raises(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'ValueError' (line 112)
        ValueError_424573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 26), 'ValueError', False)

        @norecursion
        def _stypy_temp_lambda_232(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_232'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_232', 112, 38, True)
            # Passed parameters checking function
            _stypy_temp_lambda_232.stypy_localization = localization
            _stypy_temp_lambda_232.stypy_type_of_self = None
            _stypy_temp_lambda_232.stypy_type_store = module_type_store
            _stypy_temp_lambda_232.stypy_function_name = '_stypy_temp_lambda_232'
            _stypy_temp_lambda_232.stypy_param_names_list = []
            _stypy_temp_lambda_232.stypy_varargs_param_name = None
            _stypy_temp_lambda_232.stypy_kwargs_param_name = None
            _stypy_temp_lambda_232.stypy_call_defaults = defaults
            _stypy_temp_lambda_232.stypy_call_varargs = varargs
            _stypy_temp_lambda_232.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_232', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_232', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'A' (line 112)
            A_424574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 46), 'A', False)
            int_424575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 49), 'int')
            # Applying the binary operator '**' (line 112)
            result_pow_424576 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 46), '**', A_424574, int_424575)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 112)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 38), 'stypy_return_type', result_pow_424576)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_232' in the type store
            # Getting the type of 'stypy_return_type' (line 112)
            stypy_return_type_424577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 38), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_424577)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_232'
            return stypy_return_type_424577

        # Assigning a type to the variable '_stypy_temp_lambda_232' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 38), '_stypy_temp_lambda_232', _stypy_temp_lambda_232)
        # Getting the type of '_stypy_temp_lambda_232' (line 112)
        _stypy_temp_lambda_232_424578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 38), '_stypy_temp_lambda_232')
        # Processing the call keyword arguments (line 112)
        kwargs_424579 = {}
        # Getting the type of 'assert_raises' (line 112)
        assert_raises_424572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 112)
        assert_raises_call_result_424580 = invoke(stypy.reporting.localization.Localization(__file__, 112, 12), assert_raises_424572, *[ValueError_424573, _stypy_temp_lambda_232_424578], **kwargs_424579)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to product(...): (line 114)
        # Processing the call arguments (line 114)
        
        # Call to get_matvecs(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'self' (line 114)
        self_424583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 54), 'self', False)
        # Obtaining the member 'A' of a type (line 114)
        A_424584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 54), self_424583, 'A')
        # Processing the call keyword arguments (line 114)
        kwargs_424585 = {}
        # Getting the type of 'get_matvecs' (line 114)
        get_matvecs_424582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 42), 'get_matvecs', False)
        # Calling get_matvecs(args, kwargs) (line 114)
        get_matvecs_call_result_424586 = invoke(stypy.reporting.localization.Localization(__file__, 114, 42), get_matvecs_424582, *[A_424584], **kwargs_424585)
        
        
        # Call to get_matvecs(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'self' (line 115)
        self_424588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 54), 'self', False)
        # Obtaining the member 'B' of a type (line 115)
        B_424589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 54), self_424588, 'B')
        # Processing the call keyword arguments (line 115)
        kwargs_424590 = {}
        # Getting the type of 'get_matvecs' (line 115)
        get_matvecs_424587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 42), 'get_matvecs', False)
        # Calling get_matvecs(args, kwargs) (line 115)
        get_matvecs_call_result_424591 = invoke(stypy.reporting.localization.Localization(__file__, 115, 42), get_matvecs_424587, *[B_424589], **kwargs_424590)
        
        # Processing the call keyword arguments (line 114)
        kwargs_424592 = {}
        # Getting the type of 'product' (line 114)
        product_424581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 34), 'product', False)
        # Calling product(args, kwargs) (line 114)
        product_call_result_424593 = invoke(stypy.reporting.localization.Localization(__file__, 114, 34), product_424581, *[get_matvecs_call_result_424586, get_matvecs_call_result_424591], **kwargs_424592)
        
        # Testing the type of a for loop iterable (line 114)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 114, 8), product_call_result_424593)
        # Getting the type of the for loop variable (line 114)
        for_loop_var_424594 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 114, 8), product_call_result_424593)
        # Assigning a type to the variable 'matvecsA' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'matvecsA', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 8), for_loop_var_424594))
        # Assigning a type to the variable 'matvecsB' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'matvecsB', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 8), for_loop_var_424594))
        # SSA begins for a for statement (line 114)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 116):
        
        # Assigning a Call to a Name (line 116):
        
        # Call to LinearOperator(...): (line 116)
        # Processing the call keyword arguments (line 116)
        # Getting the type of 'matvecsA' (line 116)
        matvecsA_424597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 43), 'matvecsA', False)
        kwargs_424598 = {'matvecsA_424597': matvecsA_424597}
        # Getting the type of 'interface' (line 116)
        interface_424595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'interface', False)
        # Obtaining the member 'LinearOperator' of a type (line 116)
        LinearOperator_424596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 16), interface_424595, 'LinearOperator')
        # Calling LinearOperator(args, kwargs) (line 116)
        LinearOperator_call_result_424599 = invoke(stypy.reporting.localization.Localization(__file__, 116, 16), LinearOperator_424596, *[], **kwargs_424598)
        
        # Assigning a type to the variable 'A' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'A', LinearOperator_call_result_424599)
        
        # Assigning a Call to a Name (line 117):
        
        # Assigning a Call to a Name (line 117):
        
        # Call to LinearOperator(...): (line 117)
        # Processing the call keyword arguments (line 117)
        # Getting the type of 'matvecsB' (line 117)
        matvecsB_424602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 43), 'matvecsB', False)
        kwargs_424603 = {'matvecsB_424602': matvecsB_424602}
        # Getting the type of 'interface' (line 117)
        interface_424600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'interface', False)
        # Obtaining the member 'LinearOperator' of a type (line 117)
        LinearOperator_424601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 16), interface_424600, 'LinearOperator')
        # Calling LinearOperator(args, kwargs) (line 117)
        LinearOperator_call_result_424604 = invoke(stypy.reporting.localization.Localization(__file__, 117, 16), LinearOperator_424601, *[], **kwargs_424603)
        
        # Assigning a type to the variable 'B' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'B', LinearOperator_call_result_424604)
        
        # Call to assert_equal(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'A' (line 119)
        A_424606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 26), 'A', False)
        # Getting the type of 'B' (line 119)
        B_424607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 28), 'B', False)
        # Applying the binary operator '*' (line 119)
        result_mul_424608 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 26), '*', A_424606, B_424607)
        
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_424609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        int_424610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 31), list_424609, int_424610)
        # Adding element type (line 119)
        int_424611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 31), list_424609, int_424611)
        
        # Applying the binary operator '*' (line 119)
        result_mul_424612 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 25), '*', result_mul_424608, list_424609)
        
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_424613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        int_424614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 38), list_424613, int_424614)
        # Adding element type (line 119)
        int_424615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 38), list_424613, int_424615)
        
        # Processing the call keyword arguments (line 119)
        kwargs_424616 = {}
        # Getting the type of 'assert_equal' (line 119)
        assert_equal_424605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 119)
        assert_equal_call_result_424617 = invoke(stypy.reporting.localization.Localization(__file__, 119, 12), assert_equal_424605, *[result_mul_424612, list_424613], **kwargs_424616)
        
        
        # Call to assert_equal(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'A' (line 120)
        A_424619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 26), 'A', False)
        # Getting the type of 'B' (line 120)
        B_424620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 28), 'B', False)
        # Applying the binary operator '*' (line 120)
        result_mul_424621 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 26), '*', A_424619, B_424620)
        
        
        # Obtaining an instance of the builtin type 'list' (line 120)
        list_424622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 120)
        # Adding element type (line 120)
        
        # Obtaining an instance of the builtin type 'list' (line 120)
        list_424623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 120)
        # Adding element type (line 120)
        int_424624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 32), list_424623, int_424624)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 31), list_424622, list_424623)
        # Adding element type (line 120)
        
        # Obtaining an instance of the builtin type 'list' (line 120)
        list_424625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 120)
        # Adding element type (line 120)
        int_424626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 36), list_424625, int_424626)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 31), list_424622, list_424625)
        
        # Applying the binary operator '*' (line 120)
        result_mul_424627 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 25), '*', result_mul_424621, list_424622)
        
        
        # Obtaining an instance of the builtin type 'list' (line 120)
        list_424628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 120)
        # Adding element type (line 120)
        
        # Obtaining an instance of the builtin type 'list' (line 120)
        list_424629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 120)
        # Adding element type (line 120)
        int_424630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 43), list_424629, int_424630)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 42), list_424628, list_424629)
        # Adding element type (line 120)
        
        # Obtaining an instance of the builtin type 'list' (line 120)
        list_424631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 120)
        # Adding element type (line 120)
        int_424632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 48), list_424631, int_424632)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 42), list_424628, list_424631)
        
        # Processing the call keyword arguments (line 120)
        kwargs_424633 = {}
        # Getting the type of 'assert_equal' (line 120)
        assert_equal_424618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 120)
        assert_equal_call_result_424634 = invoke(stypy.reporting.localization.Localization(__file__, 120, 12), assert_equal_424618, *[result_mul_424627, list_424628], **kwargs_424633)
        
        
        # Call to assert_equal(...): (line 121)
        # Processing the call arguments (line 121)
        
        # Call to matmat(...): (line 121)
        # Processing the call arguments (line 121)
        
        # Obtaining an instance of the builtin type 'list' (line 121)
        list_424640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 121)
        # Adding element type (line 121)
        
        # Obtaining an instance of the builtin type 'list' (line 121)
        list_424641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 121)
        # Adding element type (line 121)
        int_424642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 39), list_424641, int_424642)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 38), list_424640, list_424641)
        # Adding element type (line 121)
        
        # Obtaining an instance of the builtin type 'list' (line 121)
        list_424643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 121)
        # Adding element type (line 121)
        int_424644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 43), list_424643, int_424644)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 38), list_424640, list_424643)
        
        # Processing the call keyword arguments (line 121)
        kwargs_424645 = {}
        # Getting the type of 'A' (line 121)
        A_424636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 26), 'A', False)
        # Getting the type of 'B' (line 121)
        B_424637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 28), 'B', False)
        # Applying the binary operator '*' (line 121)
        result_mul_424638 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 26), '*', A_424636, B_424637)
        
        # Obtaining the member 'matmat' of a type (line 121)
        matmat_424639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 26), result_mul_424638, 'matmat')
        # Calling matmat(args, kwargs) (line 121)
        matmat_call_result_424646 = invoke(stypy.reporting.localization.Localization(__file__, 121, 26), matmat_424639, *[list_424640], **kwargs_424645)
        
        
        # Obtaining an instance of the builtin type 'list' (line 121)
        list_424647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 121)
        # Adding element type (line 121)
        
        # Obtaining an instance of the builtin type 'list' (line 121)
        list_424648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 121)
        # Adding element type (line 121)
        int_424649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 51), list_424648, int_424649)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 50), list_424647, list_424648)
        # Adding element type (line 121)
        
        # Obtaining an instance of the builtin type 'list' (line 121)
        list_424650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 121)
        # Adding element type (line 121)
        int_424651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 56), list_424650, int_424651)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 50), list_424647, list_424650)
        
        # Processing the call keyword arguments (line 121)
        kwargs_424652 = {}
        # Getting the type of 'assert_equal' (line 121)
        assert_equal_424635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 121)
        assert_equal_call_result_424653 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), assert_equal_424635, *[matmat_call_result_424646, list_424647], **kwargs_424652)
        
        
        # Call to assert_equal(...): (line 123)
        # Processing the call arguments (line 123)
        
        # Call to rmatvec(...): (line 123)
        # Processing the call arguments (line 123)
        
        # Obtaining an instance of the builtin type 'list' (line 123)
        list_424659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 123)
        # Adding element type (line 123)
        int_424660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 39), list_424659, int_424660)
        # Adding element type (line 123)
        int_424661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 39), list_424659, int_424661)
        
        # Processing the call keyword arguments (line 123)
        kwargs_424662 = {}
        # Getting the type of 'A' (line 123)
        A_424655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 26), 'A', False)
        # Getting the type of 'B' (line 123)
        B_424656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 28), 'B', False)
        # Applying the binary operator '*' (line 123)
        result_mul_424657 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 26), '*', A_424655, B_424656)
        
        # Obtaining the member 'rmatvec' of a type (line 123)
        rmatvec_424658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 26), result_mul_424657, 'rmatvec')
        # Calling rmatvec(args, kwargs) (line 123)
        rmatvec_call_result_424663 = invoke(stypy.reporting.localization.Localization(__file__, 123, 26), rmatvec_424658, *[list_424659], **kwargs_424662)
        
        
        # Obtaining an instance of the builtin type 'list' (line 123)
        list_424664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 123)
        # Adding element type (line 123)
        int_424665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 47), list_424664, int_424665)
        # Adding element type (line 123)
        int_424666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 47), list_424664, int_424666)
        
        # Processing the call keyword arguments (line 123)
        kwargs_424667 = {}
        # Getting the type of 'assert_equal' (line 123)
        assert_equal_424654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 123)
        assert_equal_call_result_424668 = invoke(stypy.reporting.localization.Localization(__file__, 123, 12), assert_equal_424654, *[rmatvec_call_result_424663, list_424664], **kwargs_424667)
        
        
        # Call to assert_equal(...): (line 124)
        # Processing the call arguments (line 124)
        
        # Call to matvec(...): (line 124)
        # Processing the call arguments (line 124)
        
        # Obtaining an instance of the builtin type 'list' (line 124)
        list_424675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 124)
        # Adding element type (line 124)
        int_424676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 40), list_424675, int_424676)
        # Adding element type (line 124)
        int_424677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 40), list_424675, int_424677)
        
        # Processing the call keyword arguments (line 124)
        kwargs_424678 = {}
        # Getting the type of 'A' (line 124)
        A_424670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 26), 'A', False)
        # Getting the type of 'B' (line 124)
        B_424671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 28), 'B', False)
        # Applying the binary operator '*' (line 124)
        result_mul_424672 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 26), '*', A_424670, B_424671)
        
        # Obtaining the member 'H' of a type (line 124)
        H_424673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 26), result_mul_424672, 'H')
        # Obtaining the member 'matvec' of a type (line 124)
        matvec_424674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 26), H_424673, 'matvec')
        # Calling matvec(args, kwargs) (line 124)
        matvec_call_result_424679 = invoke(stypy.reporting.localization.Localization(__file__, 124, 26), matvec_424674, *[list_424675], **kwargs_424678)
        
        
        # Obtaining an instance of the builtin type 'list' (line 124)
        list_424680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 124)
        # Adding element type (line 124)
        int_424681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 48), list_424680, int_424681)
        # Adding element type (line 124)
        int_424682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 48), list_424680, int_424682)
        
        # Processing the call keyword arguments (line 124)
        kwargs_424683 = {}
        # Getting the type of 'assert_equal' (line 124)
        assert_equal_424669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 124)
        assert_equal_call_result_424684 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), assert_equal_424669, *[matvec_call_result_424679, list_424680], **kwargs_424683)
        
        
        # Call to assert_(...): (line 126)
        # Processing the call arguments (line 126)
        
        # Call to isinstance(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'A' (line 126)
        A_424687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 31), 'A', False)
        # Getting the type of 'B' (line 126)
        B_424688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 33), 'B', False)
        # Applying the binary operator '*' (line 126)
        result_mul_424689 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 31), '*', A_424687, B_424688)
        
        # Getting the type of 'interface' (line 126)
        interface_424690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 36), 'interface', False)
        # Obtaining the member '_ProductLinearOperator' of a type (line 126)
        _ProductLinearOperator_424691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 36), interface_424690, '_ProductLinearOperator')
        # Processing the call keyword arguments (line 126)
        kwargs_424692 = {}
        # Getting the type of 'isinstance' (line 126)
        isinstance_424686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 126)
        isinstance_call_result_424693 = invoke(stypy.reporting.localization.Localization(__file__, 126, 20), isinstance_424686, *[result_mul_424689, _ProductLinearOperator_424691], **kwargs_424692)
        
        # Processing the call keyword arguments (line 126)
        kwargs_424694 = {}
        # Getting the type of 'assert_' (line 126)
        assert__424685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 126)
        assert__call_result_424695 = invoke(stypy.reporting.localization.Localization(__file__, 126, 12), assert__424685, *[isinstance_call_result_424693], **kwargs_424694)
        
        
        # Call to assert_raises(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'ValueError' (line 128)
        ValueError_424697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 26), 'ValueError', False)

        @norecursion
        def _stypy_temp_lambda_233(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_233'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_233', 128, 38, True)
            # Passed parameters checking function
            _stypy_temp_lambda_233.stypy_localization = localization
            _stypy_temp_lambda_233.stypy_type_of_self = None
            _stypy_temp_lambda_233.stypy_type_store = module_type_store
            _stypy_temp_lambda_233.stypy_function_name = '_stypy_temp_lambda_233'
            _stypy_temp_lambda_233.stypy_param_names_list = []
            _stypy_temp_lambda_233.stypy_varargs_param_name = None
            _stypy_temp_lambda_233.stypy_kwargs_param_name = None
            _stypy_temp_lambda_233.stypy_call_defaults = defaults
            _stypy_temp_lambda_233.stypy_call_varargs = varargs
            _stypy_temp_lambda_233.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_233', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_233', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'A' (line 128)
            A_424698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 46), 'A', False)
            # Getting the type of 'B' (line 128)
            B_424699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 48), 'B', False)
            # Applying the binary operator '+' (line 128)
            result_add_424700 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 46), '+', A_424698, B_424699)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 128)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 38), 'stypy_return_type', result_add_424700)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_233' in the type store
            # Getting the type of 'stypy_return_type' (line 128)
            stypy_return_type_424701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 38), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_424701)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_233'
            return stypy_return_type_424701

        # Assigning a type to the variable '_stypy_temp_lambda_233' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 38), '_stypy_temp_lambda_233', _stypy_temp_lambda_233)
        # Getting the type of '_stypy_temp_lambda_233' (line 128)
        _stypy_temp_lambda_233_424702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 38), '_stypy_temp_lambda_233')
        # Processing the call keyword arguments (line 128)
        kwargs_424703 = {}
        # Getting the type of 'assert_raises' (line 128)
        assert_raises_424696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 128)
        assert_raises_call_result_424704 = invoke(stypy.reporting.localization.Localization(__file__, 128, 12), assert_raises_424696, *[ValueError_424697, _stypy_temp_lambda_233_424702], **kwargs_424703)
        
        
        # Call to assert_raises(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'ValueError' (line 129)
        ValueError_424706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 26), 'ValueError', False)

        @norecursion
        def _stypy_temp_lambda_234(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_234'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_234', 129, 38, True)
            # Passed parameters checking function
            _stypy_temp_lambda_234.stypy_localization = localization
            _stypy_temp_lambda_234.stypy_type_of_self = None
            _stypy_temp_lambda_234.stypy_type_store = module_type_store
            _stypy_temp_lambda_234.stypy_function_name = '_stypy_temp_lambda_234'
            _stypy_temp_lambda_234.stypy_param_names_list = []
            _stypy_temp_lambda_234.stypy_varargs_param_name = None
            _stypy_temp_lambda_234.stypy_kwargs_param_name = None
            _stypy_temp_lambda_234.stypy_call_defaults = defaults
            _stypy_temp_lambda_234.stypy_call_varargs = varargs
            _stypy_temp_lambda_234.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_234', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_234', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'A' (line 129)
            A_424707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 46), 'A', False)
            int_424708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 49), 'int')
            # Applying the binary operator '**' (line 129)
            result_pow_424709 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 46), '**', A_424707, int_424708)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 129)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 38), 'stypy_return_type', result_pow_424709)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_234' in the type store
            # Getting the type of 'stypy_return_type' (line 129)
            stypy_return_type_424710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 38), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_424710)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_234'
            return stypy_return_type_424710

        # Assigning a type to the variable '_stypy_temp_lambda_234' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 38), '_stypy_temp_lambda_234', _stypy_temp_lambda_234)
        # Getting the type of '_stypy_temp_lambda_234' (line 129)
        _stypy_temp_lambda_234_424711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 38), '_stypy_temp_lambda_234')
        # Processing the call keyword arguments (line 129)
        kwargs_424712 = {}
        # Getting the type of 'assert_raises' (line 129)
        assert_raises_424705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 129)
        assert_raises_call_result_424713 = invoke(stypy.reporting.localization.Localization(__file__, 129, 12), assert_raises_424705, *[ValueError_424706, _stypy_temp_lambda_234_424711], **kwargs_424712)
        
        
        # Assigning a BinOp to a Name (line 131):
        
        # Assigning a BinOp to a Name (line 131):
        # Getting the type of 'A' (line 131)
        A_424714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'A')
        # Getting the type of 'B' (line 131)
        B_424715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 18), 'B')
        # Applying the binary operator '*' (line 131)
        result_mul_424716 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 16), '*', A_424714, B_424715)
        
        # Assigning a type to the variable 'z' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'z', result_mul_424716)
        
        # Call to assert_(...): (line 132)
        # Processing the call arguments (line 132)
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'z' (line 132)
        z_424719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 24), 'z', False)
        # Obtaining the member 'args' of a type (line 132)
        args_424720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 24), z_424719, 'args')
        # Processing the call keyword arguments (line 132)
        kwargs_424721 = {}
        # Getting the type of 'len' (line 132)
        len_424718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 20), 'len', False)
        # Calling len(args, kwargs) (line 132)
        len_call_result_424722 = invoke(stypy.reporting.localization.Localization(__file__, 132, 20), len_424718, *[args_424720], **kwargs_424721)
        
        int_424723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 35), 'int')
        # Applying the binary operator '==' (line 132)
        result_eq_424724 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 20), '==', len_call_result_424722, int_424723)
        
        
        
        # Obtaining the type of the subscript
        int_424725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 48), 'int')
        # Getting the type of 'z' (line 132)
        z_424726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 41), 'z', False)
        # Obtaining the member 'args' of a type (line 132)
        args_424727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 41), z_424726, 'args')
        # Obtaining the member '__getitem__' of a type (line 132)
        getitem___424728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 41), args_424727, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
        subscript_call_result_424729 = invoke(stypy.reporting.localization.Localization(__file__, 132, 41), getitem___424728, int_424725)
        
        # Getting the type of 'A' (line 132)
        A_424730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 54), 'A', False)
        # Applying the binary operator 'is' (line 132)
        result_is__424731 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 41), 'is', subscript_call_result_424729, A_424730)
        
        # Applying the binary operator 'and' (line 132)
        result_and_keyword_424732 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 20), 'and', result_eq_424724, result_is__424731)
        
        
        # Obtaining the type of the subscript
        int_424733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 67), 'int')
        # Getting the type of 'z' (line 132)
        z_424734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 60), 'z', False)
        # Obtaining the member 'args' of a type (line 132)
        args_424735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 60), z_424734, 'args')
        # Obtaining the member '__getitem__' of a type (line 132)
        getitem___424736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 60), args_424735, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
        subscript_call_result_424737 = invoke(stypy.reporting.localization.Localization(__file__, 132, 60), getitem___424736, int_424733)
        
        # Getting the type of 'B' (line 132)
        B_424738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 73), 'B', False)
        # Applying the binary operator 'is' (line 132)
        result_is__424739 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 60), 'is', subscript_call_result_424737, B_424738)
        
        # Applying the binary operator 'and' (line 132)
        result_and_keyword_424740 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 20), 'and', result_and_keyword_424732, result_is__424739)
        
        # Processing the call keyword arguments (line 132)
        kwargs_424741 = {}
        # Getting the type of 'assert_' (line 132)
        assert__424717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 132)
        assert__call_result_424742 = invoke(stypy.reporting.localization.Localization(__file__, 132, 12), assert__424717, *[result_and_keyword_424740], **kwargs_424741)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to get_matvecs(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'self' (line 134)
        self_424744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 36), 'self', False)
        # Obtaining the member 'C' of a type (line 134)
        C_424745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 36), self_424744, 'C')
        # Processing the call keyword arguments (line 134)
        kwargs_424746 = {}
        # Getting the type of 'get_matvecs' (line 134)
        get_matvecs_424743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 24), 'get_matvecs', False)
        # Calling get_matvecs(args, kwargs) (line 134)
        get_matvecs_call_result_424747 = invoke(stypy.reporting.localization.Localization(__file__, 134, 24), get_matvecs_424743, *[C_424745], **kwargs_424746)
        
        # Testing the type of a for loop iterable (line 134)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 134, 8), get_matvecs_call_result_424747)
        # Getting the type of the for loop variable (line 134)
        for_loop_var_424748 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 134, 8), get_matvecs_call_result_424747)
        # Assigning a type to the variable 'matvecsC' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'matvecsC', for_loop_var_424748)
        # SSA begins for a for statement (line 134)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 135):
        
        # Assigning a Call to a Name (line 135):
        
        # Call to LinearOperator(...): (line 135)
        # Processing the call keyword arguments (line 135)
        # Getting the type of 'matvecsC' (line 135)
        matvecsC_424751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 43), 'matvecsC', False)
        kwargs_424752 = {'matvecsC_424751': matvecsC_424751}
        # Getting the type of 'interface' (line 135)
        interface_424749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'interface', False)
        # Obtaining the member 'LinearOperator' of a type (line 135)
        LinearOperator_424750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 16), interface_424749, 'LinearOperator')
        # Calling LinearOperator(args, kwargs) (line 135)
        LinearOperator_call_result_424753 = invoke(stypy.reporting.localization.Localization(__file__, 135, 16), LinearOperator_424750, *[], **kwargs_424752)
        
        # Assigning a type to the variable 'C' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'C', LinearOperator_call_result_424753)
        
        # Call to assert_equal(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'C' (line 137)
        C_424755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 26), 'C', False)
        int_424756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 29), 'int')
        # Applying the binary operator '**' (line 137)
        result_pow_424757 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 26), '**', C_424755, int_424756)
        
        
        # Obtaining an instance of the builtin type 'list' (line 137)
        list_424758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 137)
        # Adding element type (line 137)
        int_424759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 32), list_424758, int_424759)
        # Adding element type (line 137)
        int_424760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 32), list_424758, int_424760)
        
        # Applying the binary operator '*' (line 137)
        result_mul_424761 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 25), '*', result_pow_424757, list_424758)
        
        
        # Obtaining an instance of the builtin type 'list' (line 137)
        list_424762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 137)
        # Adding element type (line 137)
        int_424763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 39), list_424762, int_424763)
        # Adding element type (line 137)
        int_424764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 39), list_424762, int_424764)
        
        # Processing the call keyword arguments (line 137)
        kwargs_424765 = {}
        # Getting the type of 'assert_equal' (line 137)
        assert_equal_424754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 137)
        assert_equal_call_result_424766 = invoke(stypy.reporting.localization.Localization(__file__, 137, 12), assert_equal_424754, *[result_mul_424761, list_424762], **kwargs_424765)
        
        
        # Call to assert_equal(...): (line 138)
        # Processing the call arguments (line 138)
        
        # Call to rmatvec(...): (line 138)
        # Processing the call arguments (line 138)
        
        # Obtaining an instance of the builtin type 'list' (line 138)
        list_424772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 138)
        # Adding element type (line 138)
        int_424773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 40), list_424772, int_424773)
        # Adding element type (line 138)
        int_424774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 40), list_424772, int_424774)
        
        # Processing the call keyword arguments (line 138)
        kwargs_424775 = {}
        # Getting the type of 'C' (line 138)
        C_424768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 26), 'C', False)
        int_424769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 29), 'int')
        # Applying the binary operator '**' (line 138)
        result_pow_424770 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 26), '**', C_424768, int_424769)
        
        # Obtaining the member 'rmatvec' of a type (line 138)
        rmatvec_424771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 26), result_pow_424770, 'rmatvec')
        # Calling rmatvec(args, kwargs) (line 138)
        rmatvec_call_result_424776 = invoke(stypy.reporting.localization.Localization(__file__, 138, 26), rmatvec_424771, *[list_424772], **kwargs_424775)
        
        
        # Obtaining an instance of the builtin type 'list' (line 138)
        list_424777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 138)
        # Adding element type (line 138)
        int_424778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 48), list_424777, int_424778)
        # Adding element type (line 138)
        int_424779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 48), list_424777, int_424779)
        
        # Processing the call keyword arguments (line 138)
        kwargs_424780 = {}
        # Getting the type of 'assert_equal' (line 138)
        assert_equal_424767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 138)
        assert_equal_call_result_424781 = invoke(stypy.reporting.localization.Localization(__file__, 138, 12), assert_equal_424767, *[rmatvec_call_result_424776, list_424777], **kwargs_424780)
        
        
        # Call to assert_equal(...): (line 139)
        # Processing the call arguments (line 139)
        
        # Call to matvec(...): (line 139)
        # Processing the call arguments (line 139)
        
        # Obtaining an instance of the builtin type 'list' (line 139)
        list_424788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 139)
        # Adding element type (line 139)
        int_424789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 41), list_424788, int_424789)
        # Adding element type (line 139)
        int_424790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 41), list_424788, int_424790)
        
        # Processing the call keyword arguments (line 139)
        kwargs_424791 = {}
        # Getting the type of 'C' (line 139)
        C_424783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 26), 'C', False)
        int_424784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 29), 'int')
        # Applying the binary operator '**' (line 139)
        result_pow_424785 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 26), '**', C_424783, int_424784)
        
        # Obtaining the member 'H' of a type (line 139)
        H_424786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 26), result_pow_424785, 'H')
        # Obtaining the member 'matvec' of a type (line 139)
        matvec_424787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 26), H_424786, 'matvec')
        # Calling matvec(args, kwargs) (line 139)
        matvec_call_result_424792 = invoke(stypy.reporting.localization.Localization(__file__, 139, 26), matvec_424787, *[list_424788], **kwargs_424791)
        
        
        # Obtaining an instance of the builtin type 'list' (line 139)
        list_424793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 139)
        # Adding element type (line 139)
        int_424794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 49), list_424793, int_424794)
        # Adding element type (line 139)
        int_424795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 49), list_424793, int_424795)
        
        # Processing the call keyword arguments (line 139)
        kwargs_424796 = {}
        # Getting the type of 'assert_equal' (line 139)
        assert_equal_424782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 139)
        assert_equal_call_result_424797 = invoke(stypy.reporting.localization.Localization(__file__, 139, 12), assert_equal_424782, *[matvec_call_result_424792, list_424793], **kwargs_424796)
        
        
        # Call to assert_equal(...): (line 140)
        # Processing the call arguments (line 140)
        
        # Call to matmat(...): (line 140)
        # Processing the call arguments (line 140)
        
        # Obtaining an instance of the builtin type 'list' (line 140)
        list_424803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 140)
        # Adding element type (line 140)
        
        # Obtaining an instance of the builtin type 'list' (line 140)
        list_424804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 140)
        # Adding element type (line 140)
        int_424805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 40), list_424804, int_424805)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 39), list_424803, list_424804)
        # Adding element type (line 140)
        
        # Obtaining an instance of the builtin type 'list' (line 140)
        list_424806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 140)
        # Adding element type (line 140)
        int_424807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 44), list_424806, int_424807)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 39), list_424803, list_424806)
        
        # Processing the call keyword arguments (line 140)
        kwargs_424808 = {}
        # Getting the type of 'C' (line 140)
        C_424799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 26), 'C', False)
        int_424800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 29), 'int')
        # Applying the binary operator '**' (line 140)
        result_pow_424801 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 26), '**', C_424799, int_424800)
        
        # Obtaining the member 'matmat' of a type (line 140)
        matmat_424802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 26), result_pow_424801, 'matmat')
        # Calling matmat(args, kwargs) (line 140)
        matmat_call_result_424809 = invoke(stypy.reporting.localization.Localization(__file__, 140, 26), matmat_424802, *[list_424803], **kwargs_424808)
        
        
        # Obtaining an instance of the builtin type 'list' (line 140)
        list_424810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 140)
        # Adding element type (line 140)
        
        # Obtaining an instance of the builtin type 'list' (line 140)
        list_424811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 140)
        # Adding element type (line 140)
        int_424812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 52), list_424811, int_424812)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 51), list_424810, list_424811)
        # Adding element type (line 140)
        
        # Obtaining an instance of the builtin type 'list' (line 140)
        list_424813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 140)
        # Adding element type (line 140)
        int_424814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 57), list_424813, int_424814)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 51), list_424810, list_424813)
        
        # Processing the call keyword arguments (line 140)
        kwargs_424815 = {}
        # Getting the type of 'assert_equal' (line 140)
        assert_equal_424798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 140)
        assert_equal_call_result_424816 = invoke(stypy.reporting.localization.Localization(__file__, 140, 12), assert_equal_424798, *[matmat_call_result_424809, list_424810], **kwargs_424815)
        
        
        # Call to assert_(...): (line 142)
        # Processing the call arguments (line 142)
        
        # Call to isinstance(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'C' (line 142)
        C_424819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 31), 'C', False)
        int_424820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 34), 'int')
        # Applying the binary operator '**' (line 142)
        result_pow_424821 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 31), '**', C_424819, int_424820)
        
        # Getting the type of 'interface' (line 142)
        interface_424822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 37), 'interface', False)
        # Obtaining the member '_PowerLinearOperator' of a type (line 142)
        _PowerLinearOperator_424823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 37), interface_424822, '_PowerLinearOperator')
        # Processing the call keyword arguments (line 142)
        kwargs_424824 = {}
        # Getting the type of 'isinstance' (line 142)
        isinstance_424818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 142)
        isinstance_call_result_424825 = invoke(stypy.reporting.localization.Localization(__file__, 142, 20), isinstance_424818, *[result_pow_424821, _PowerLinearOperator_424823], **kwargs_424824)
        
        # Processing the call keyword arguments (line 142)
        kwargs_424826 = {}
        # Getting the type of 'assert_' (line 142)
        assert__424817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 142)
        assert__call_result_424827 = invoke(stypy.reporting.localization.Localization(__file__, 142, 12), assert__424817, *[isinstance_call_result_424825], **kwargs_424826)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_matvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_matvec' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_424828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_424828)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_matvec'
        return stypy_return_type_424828


    @norecursion
    def test_matmul(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_matmul'
        module_type_store = module_type_store.open_function_context('test_matmul', 144, 4, False)
        # Assigning a type to the variable 'self' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinearOperator.test_matmul.__dict__.__setitem__('stypy_localization', localization)
        TestLinearOperator.test_matmul.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinearOperator.test_matmul.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinearOperator.test_matmul.__dict__.__setitem__('stypy_function_name', 'TestLinearOperator.test_matmul')
        TestLinearOperator.test_matmul.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinearOperator.test_matmul.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinearOperator.test_matmul.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinearOperator.test_matmul.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinearOperator.test_matmul.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinearOperator.test_matmul.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinearOperator.test_matmul.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinearOperator.test_matmul', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_matmul', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_matmul(...)' code ##################

        
        
        # Getting the type of 'TEST_MATMUL' (line 145)
        TEST_MATMUL_424829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 15), 'TEST_MATMUL')
        # Applying the 'not' unary operator (line 145)
        result_not__424830 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 11), 'not', TEST_MATMUL_424829)
        
        # Testing the type of an if condition (line 145)
        if_condition_424831 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 8), result_not__424830)
        # Assigning a type to the variable 'if_condition_424831' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'if_condition_424831', if_condition_424831)
        # SSA begins for if statement (line 145)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to skip(...): (line 146)
        # Processing the call arguments (line 146)
        str_424834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 24), 'str', 'matmul is only tested in Python 3.5+')
        # Processing the call keyword arguments (line 146)
        kwargs_424835 = {}
        # Getting the type of 'pytest' (line 146)
        pytest_424832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'pytest', False)
        # Obtaining the member 'skip' of a type (line 146)
        skip_424833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 12), pytest_424832, 'skip')
        # Calling skip(args, kwargs) (line 146)
        skip_call_result_424836 = invoke(stypy.reporting.localization.Localization(__file__, 146, 12), skip_424833, *[str_424834], **kwargs_424835)
        
        # SSA join for if statement (line 145)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Dict to a Name (line 148):
        
        # Assigning a Dict to a Name (line 148):
        
        # Obtaining an instance of the builtin type 'dict' (line 148)
        dict_424837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 12), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 148)
        # Adding element type (key, value) (line 148)
        str_424838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 13), 'str', 'shape')
        # Getting the type of 'self' (line 148)
        self_424839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 22), 'self')
        # Obtaining the member 'A' of a type (line 148)
        A_424840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 22), self_424839, 'A')
        # Obtaining the member 'shape' of a type (line 148)
        shape_424841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 22), A_424840, 'shape')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 12), dict_424837, (str_424838, shape_424841))
        # Adding element type (key, value) (line 148)
        str_424842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 13), 'str', 'matvec')

        @norecursion
        def _stypy_temp_lambda_235(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_235'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_235', 149, 23, True)
            # Passed parameters checking function
            _stypy_temp_lambda_235.stypy_localization = localization
            _stypy_temp_lambda_235.stypy_type_of_self = None
            _stypy_temp_lambda_235.stypy_type_store = module_type_store
            _stypy_temp_lambda_235.stypy_function_name = '_stypy_temp_lambda_235'
            _stypy_temp_lambda_235.stypy_param_names_list = ['x']
            _stypy_temp_lambda_235.stypy_varargs_param_name = None
            _stypy_temp_lambda_235.stypy_kwargs_param_name = None
            _stypy_temp_lambda_235.stypy_call_defaults = defaults
            _stypy_temp_lambda_235.stypy_call_varargs = varargs
            _stypy_temp_lambda_235.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_235', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_235', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to reshape(...): (line 149)
            # Processing the call arguments (line 149)
            
            # Obtaining the type of the subscript
            int_424851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 72), 'int')
            # Getting the type of 'self' (line 149)
            self_424852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 59), 'self', False)
            # Obtaining the member 'A' of a type (line 149)
            A_424853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 59), self_424852, 'A')
            # Obtaining the member 'shape' of a type (line 149)
            shape_424854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 59), A_424853, 'shape')
            # Obtaining the member '__getitem__' of a type (line 149)
            getitem___424855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 59), shape_424854, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 149)
            subscript_call_result_424856 = invoke(stypy.reporting.localization.Localization(__file__, 149, 59), getitem___424855, int_424851)
            
            # Processing the call keyword arguments (line 149)
            kwargs_424857 = {}
            
            # Call to dot(...): (line 149)
            # Processing the call arguments (line 149)
            # Getting the type of 'self' (line 149)
            self_424845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 40), 'self', False)
            # Obtaining the member 'A' of a type (line 149)
            A_424846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 40), self_424845, 'A')
            # Getting the type of 'x' (line 149)
            x_424847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 48), 'x', False)
            # Processing the call keyword arguments (line 149)
            kwargs_424848 = {}
            # Getting the type of 'np' (line 149)
            np_424843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 33), 'np', False)
            # Obtaining the member 'dot' of a type (line 149)
            dot_424844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 33), np_424843, 'dot')
            # Calling dot(args, kwargs) (line 149)
            dot_call_result_424849 = invoke(stypy.reporting.localization.Localization(__file__, 149, 33), dot_424844, *[A_424846, x_424847], **kwargs_424848)
            
            # Obtaining the member 'reshape' of a type (line 149)
            reshape_424850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 33), dot_call_result_424849, 'reshape')
            # Calling reshape(args, kwargs) (line 149)
            reshape_call_result_424858 = invoke(stypy.reporting.localization.Localization(__file__, 149, 33), reshape_424850, *[subscript_call_result_424856], **kwargs_424857)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 149)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 23), 'stypy_return_type', reshape_call_result_424858)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_235' in the type store
            # Getting the type of 'stypy_return_type' (line 149)
            stypy_return_type_424859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 23), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_424859)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_235'
            return stypy_return_type_424859

        # Assigning a type to the variable '_stypy_temp_lambda_235' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 23), '_stypy_temp_lambda_235', _stypy_temp_lambda_235)
        # Getting the type of '_stypy_temp_lambda_235' (line 149)
        _stypy_temp_lambda_235_424860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 23), '_stypy_temp_lambda_235')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 12), dict_424837, (str_424842, _stypy_temp_lambda_235_424860))
        # Adding element type (key, value) (line 148)
        str_424861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 13), 'str', 'rmatvec')

        @norecursion
        def _stypy_temp_lambda_236(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_236'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_236', 150, 24, True)
            # Passed parameters checking function
            _stypy_temp_lambda_236.stypy_localization = localization
            _stypy_temp_lambda_236.stypy_type_of_self = None
            _stypy_temp_lambda_236.stypy_type_store = module_type_store
            _stypy_temp_lambda_236.stypy_function_name = '_stypy_temp_lambda_236'
            _stypy_temp_lambda_236.stypy_param_names_list = ['x']
            _stypy_temp_lambda_236.stypy_varargs_param_name = None
            _stypy_temp_lambda_236.stypy_kwargs_param_name = None
            _stypy_temp_lambda_236.stypy_call_defaults = defaults
            _stypy_temp_lambda_236.stypy_call_varargs = varargs
            _stypy_temp_lambda_236.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_236', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_236', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to reshape(...): (line 150)
            # Processing the call arguments (line 150)
            
            # Obtaining the type of the subscript
            int_424874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 65), 'int')
            # Getting the type of 'self' (line 151)
            self_424875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 52), 'self', False)
            # Obtaining the member 'A' of a type (line 151)
            A_424876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 52), self_424875, 'A')
            # Obtaining the member 'shape' of a type (line 151)
            shape_424877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 52), A_424876, 'shape')
            # Obtaining the member '__getitem__' of a type (line 151)
            getitem___424878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 52), shape_424877, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 151)
            subscript_call_result_424879 = invoke(stypy.reporting.localization.Localization(__file__, 151, 52), getitem___424878, int_424874)
            
            # Processing the call keyword arguments (line 150)
            kwargs_424880 = {}
            
            # Call to dot(...): (line 150)
            # Processing the call arguments (line 150)
            
            # Call to conj(...): (line 150)
            # Processing the call keyword arguments (line 150)
            kwargs_424868 = {}
            # Getting the type of 'self' (line 150)
            self_424864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 41), 'self', False)
            # Obtaining the member 'A' of a type (line 150)
            A_424865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 41), self_424864, 'A')
            # Obtaining the member 'T' of a type (line 150)
            T_424866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 41), A_424865, 'T')
            # Obtaining the member 'conj' of a type (line 150)
            conj_424867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 41), T_424866, 'conj')
            # Calling conj(args, kwargs) (line 150)
            conj_call_result_424869 = invoke(stypy.reporting.localization.Localization(__file__, 150, 41), conj_424867, *[], **kwargs_424868)
            
            # Getting the type of 'x' (line 151)
            x_424870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 41), 'x', False)
            # Processing the call keyword arguments (line 150)
            kwargs_424871 = {}
            # Getting the type of 'np' (line 150)
            np_424862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 34), 'np', False)
            # Obtaining the member 'dot' of a type (line 150)
            dot_424863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 34), np_424862, 'dot')
            # Calling dot(args, kwargs) (line 150)
            dot_call_result_424872 = invoke(stypy.reporting.localization.Localization(__file__, 150, 34), dot_424863, *[conj_call_result_424869, x_424870], **kwargs_424871)
            
            # Obtaining the member 'reshape' of a type (line 150)
            reshape_424873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 34), dot_call_result_424872, 'reshape')
            # Calling reshape(args, kwargs) (line 150)
            reshape_call_result_424881 = invoke(stypy.reporting.localization.Localization(__file__, 150, 34), reshape_424873, *[subscript_call_result_424879], **kwargs_424880)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 150)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 24), 'stypy_return_type', reshape_call_result_424881)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_236' in the type store
            # Getting the type of 'stypy_return_type' (line 150)
            stypy_return_type_424882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 24), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_424882)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_236'
            return stypy_return_type_424882

        # Assigning a type to the variable '_stypy_temp_lambda_236' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 24), '_stypy_temp_lambda_236', _stypy_temp_lambda_236)
        # Getting the type of '_stypy_temp_lambda_236' (line 150)
        _stypy_temp_lambda_236_424883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 24), '_stypy_temp_lambda_236')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 12), dict_424837, (str_424861, _stypy_temp_lambda_236_424883))
        # Adding element type (key, value) (line 148)
        str_424884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 13), 'str', 'matmat')

        @norecursion
        def _stypy_temp_lambda_237(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_237'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_237', 152, 23, True)
            # Passed parameters checking function
            _stypy_temp_lambda_237.stypy_localization = localization
            _stypy_temp_lambda_237.stypy_type_of_self = None
            _stypy_temp_lambda_237.stypy_type_store = module_type_store
            _stypy_temp_lambda_237.stypy_function_name = '_stypy_temp_lambda_237'
            _stypy_temp_lambda_237.stypy_param_names_list = ['x']
            _stypy_temp_lambda_237.stypy_varargs_param_name = None
            _stypy_temp_lambda_237.stypy_kwargs_param_name = None
            _stypy_temp_lambda_237.stypy_call_defaults = defaults
            _stypy_temp_lambda_237.stypy_call_varargs = varargs
            _stypy_temp_lambda_237.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_237', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_237', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to dot(...): (line 152)
            # Processing the call arguments (line 152)
            # Getting the type of 'self' (line 152)
            self_424887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 40), 'self', False)
            # Obtaining the member 'A' of a type (line 152)
            A_424888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 40), self_424887, 'A')
            # Getting the type of 'x' (line 152)
            x_424889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 48), 'x', False)
            # Processing the call keyword arguments (line 152)
            kwargs_424890 = {}
            # Getting the type of 'np' (line 152)
            np_424885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 33), 'np', False)
            # Obtaining the member 'dot' of a type (line 152)
            dot_424886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 33), np_424885, 'dot')
            # Calling dot(args, kwargs) (line 152)
            dot_call_result_424891 = invoke(stypy.reporting.localization.Localization(__file__, 152, 33), dot_424886, *[A_424888, x_424889], **kwargs_424890)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 152)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 23), 'stypy_return_type', dot_call_result_424891)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_237' in the type store
            # Getting the type of 'stypy_return_type' (line 152)
            stypy_return_type_424892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 23), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_424892)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_237'
            return stypy_return_type_424892

        # Assigning a type to the variable '_stypy_temp_lambda_237' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 23), '_stypy_temp_lambda_237', _stypy_temp_lambda_237)
        # Getting the type of '_stypy_temp_lambda_237' (line 152)
        _stypy_temp_lambda_237_424893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 23), '_stypy_temp_lambda_237')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 12), dict_424837, (str_424884, _stypy_temp_lambda_237_424893))
        
        # Assigning a type to the variable 'D' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'D', dict_424837)
        
        # Assigning a Call to a Name (line 153):
        
        # Assigning a Call to a Name (line 153):
        
        # Call to LinearOperator(...): (line 153)
        # Processing the call keyword arguments (line 153)
        # Getting the type of 'D' (line 153)
        D_424896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 39), 'D', False)
        kwargs_424897 = {'D_424896': D_424896}
        # Getting the type of 'interface' (line 153)
        interface_424894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'interface', False)
        # Obtaining the member 'LinearOperator' of a type (line 153)
        LinearOperator_424895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 12), interface_424894, 'LinearOperator')
        # Calling LinearOperator(args, kwargs) (line 153)
        LinearOperator_call_result_424898 = invoke(stypy.reporting.localization.Localization(__file__, 153, 12), LinearOperator_424895, *[], **kwargs_424897)
        
        # Assigning a type to the variable 'A' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'A', LinearOperator_call_result_424898)
        
        # Assigning a Call to a Name (line 154):
        
        # Assigning a Call to a Name (line 154):
        
        # Call to array(...): (line 154)
        # Processing the call arguments (line 154)
        
        # Obtaining an instance of the builtin type 'list' (line 154)
        list_424901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 154)
        # Adding element type (line 154)
        
        # Obtaining an instance of the builtin type 'list' (line 154)
        list_424902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 154)
        # Adding element type (line 154)
        int_424903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 22), list_424902, int_424903)
        # Adding element type (line 154)
        int_424904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 22), list_424902, int_424904)
        # Adding element type (line 154)
        int_424905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 22), list_424902, int_424905)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 21), list_424901, list_424902)
        # Adding element type (line 154)
        
        # Obtaining an instance of the builtin type 'list' (line 155)
        list_424906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 155)
        # Adding element type (line 155)
        int_424907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 22), list_424906, int_424907)
        # Adding element type (line 155)
        int_424908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 22), list_424906, int_424908)
        # Adding element type (line 155)
        int_424909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 22), list_424906, int_424909)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 21), list_424901, list_424906)
        # Adding element type (line 154)
        
        # Obtaining an instance of the builtin type 'list' (line 156)
        list_424910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 156)
        # Adding element type (line 156)
        int_424911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 22), list_424910, int_424911)
        # Adding element type (line 156)
        int_424912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 22), list_424910, int_424912)
        # Adding element type (line 156)
        int_424913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 22), list_424910, int_424913)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 21), list_424901, list_424910)
        
        # Processing the call keyword arguments (line 154)
        kwargs_424914 = {}
        # Getting the type of 'np' (line 154)
        np_424899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 154)
        array_424900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 12), np_424899, 'array')
        # Calling array(args, kwargs) (line 154)
        array_call_result_424915 = invoke(stypy.reporting.localization.Localization(__file__, 154, 12), array_424900, *[list_424901], **kwargs_424914)
        
        # Assigning a type to the variable 'B' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'B', array_call_result_424915)
        
        # Assigning a Subscript to a Name (line 157):
        
        # Assigning a Subscript to a Name (line 157):
        
        # Obtaining the type of the subscript
        int_424916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 14), 'int')
        # Getting the type of 'B' (line 157)
        B_424917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'B')
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___424918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 12), B_424917, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_424919 = invoke(stypy.reporting.localization.Localization(__file__, 157, 12), getitem___424918, int_424916)
        
        # Assigning a type to the variable 'b' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'b', subscript_call_result_424919)
        
        # Call to assert_equal(...): (line 159)
        # Processing the call arguments (line 159)
        
        # Call to matmul(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'A' (line 159)
        A_424923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 37), 'A', False)
        # Getting the type of 'b' (line 159)
        b_424924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 40), 'b', False)
        # Processing the call keyword arguments (line 159)
        kwargs_424925 = {}
        # Getting the type of 'operator' (line 159)
        operator_424921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 21), 'operator', False)
        # Obtaining the member 'matmul' of a type (line 159)
        matmul_424922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 21), operator_424921, 'matmul')
        # Calling matmul(args, kwargs) (line 159)
        matmul_call_result_424926 = invoke(stypy.reporting.localization.Localization(__file__, 159, 21), matmul_424922, *[A_424923, b_424924], **kwargs_424925)
        
        # Getting the type of 'A' (line 159)
        A_424927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 44), 'A', False)
        # Getting the type of 'b' (line 159)
        b_424928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 48), 'b', False)
        # Applying the binary operator '*' (line 159)
        result_mul_424929 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 44), '*', A_424927, b_424928)
        
        # Processing the call keyword arguments (line 159)
        kwargs_424930 = {}
        # Getting the type of 'assert_equal' (line 159)
        assert_equal_424920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 159)
        assert_equal_call_result_424931 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), assert_equal_424920, *[matmul_call_result_424926, result_mul_424929], **kwargs_424930)
        
        
        # Call to assert_equal(...): (line 160)
        # Processing the call arguments (line 160)
        
        # Call to matmul(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'A' (line 160)
        A_424935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 37), 'A', False)
        # Getting the type of 'B' (line 160)
        B_424936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 40), 'B', False)
        # Processing the call keyword arguments (line 160)
        kwargs_424937 = {}
        # Getting the type of 'operator' (line 160)
        operator_424933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 21), 'operator', False)
        # Obtaining the member 'matmul' of a type (line 160)
        matmul_424934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 21), operator_424933, 'matmul')
        # Calling matmul(args, kwargs) (line 160)
        matmul_call_result_424938 = invoke(stypy.reporting.localization.Localization(__file__, 160, 21), matmul_424934, *[A_424935, B_424936], **kwargs_424937)
        
        # Getting the type of 'A' (line 160)
        A_424939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 44), 'A', False)
        # Getting the type of 'B' (line 160)
        B_424940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 48), 'B', False)
        # Applying the binary operator '*' (line 160)
        result_mul_424941 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 44), '*', A_424939, B_424940)
        
        # Processing the call keyword arguments (line 160)
        kwargs_424942 = {}
        # Getting the type of 'assert_equal' (line 160)
        assert_equal_424932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 160)
        assert_equal_call_result_424943 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), assert_equal_424932, *[matmul_call_result_424938, result_mul_424941], **kwargs_424942)
        
        
        # Call to assert_raises(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'ValueError' (line 161)
        ValueError_424945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 22), 'ValueError', False)
        # Getting the type of 'operator' (line 161)
        operator_424946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 34), 'operator', False)
        # Obtaining the member 'matmul' of a type (line 161)
        matmul_424947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 34), operator_424946, 'matmul')
        # Getting the type of 'A' (line 161)
        A_424948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 51), 'A', False)
        int_424949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 54), 'int')
        # Processing the call keyword arguments (line 161)
        kwargs_424950 = {}
        # Getting the type of 'assert_raises' (line 161)
        assert_raises_424944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 161)
        assert_raises_call_result_424951 = invoke(stypy.reporting.localization.Localization(__file__, 161, 8), assert_raises_424944, *[ValueError_424945, matmul_424947, A_424948, int_424949], **kwargs_424950)
        
        
        # Call to assert_raises(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'ValueError' (line 162)
        ValueError_424953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 22), 'ValueError', False)
        # Getting the type of 'operator' (line 162)
        operator_424954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 34), 'operator', False)
        # Obtaining the member 'matmul' of a type (line 162)
        matmul_424955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 34), operator_424954, 'matmul')
        int_424956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 51), 'int')
        # Getting the type of 'A' (line 162)
        A_424957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 54), 'A', False)
        # Processing the call keyword arguments (line 162)
        kwargs_424958 = {}
        # Getting the type of 'assert_raises' (line 162)
        assert_raises_424952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 162)
        assert_raises_call_result_424959 = invoke(stypy.reporting.localization.Localization(__file__, 162, 8), assert_raises_424952, *[ValueError_424953, matmul_424955, int_424956, A_424957], **kwargs_424958)
        
        
        # ################# End of 'test_matmul(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_matmul' in the type store
        # Getting the type of 'stypy_return_type' (line 144)
        stypy_return_type_424960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_424960)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_matmul'
        return stypy_return_type_424960


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 23, 0, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinearOperator.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestLinearOperator' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'TestLinearOperator', TestLinearOperator)
# Declaration of the 'TestAsLinearOperator' class

class TestAsLinearOperator(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 166, 4, False)
        # Assigning a type to the variable 'self' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestAsLinearOperator.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestAsLinearOperator.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestAsLinearOperator.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestAsLinearOperator.setup_method.__dict__.__setitem__('stypy_function_name', 'TestAsLinearOperator.setup_method')
        TestAsLinearOperator.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestAsLinearOperator.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestAsLinearOperator.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestAsLinearOperator.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestAsLinearOperator.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestAsLinearOperator.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestAsLinearOperator.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestAsLinearOperator.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Attribute (line 167):
        
        # Assigning a List to a Attribute (line 167):
        
        # Obtaining an instance of the builtin type 'list' (line 167)
        list_424961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 167)
        
        # Getting the type of 'self' (line 167)
        self_424962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'self')
        # Setting the type of the member 'cases' of a type (line 167)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), self_424962, 'cases', list_424961)

        @norecursion
        def make_cases(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'make_cases'
            module_type_store = module_type_store.open_function_context('make_cases', 169, 8, False)
            
            # Passed parameters checking function
            make_cases.stypy_localization = localization
            make_cases.stypy_type_of_self = None
            make_cases.stypy_type_store = module_type_store
            make_cases.stypy_function_name = 'make_cases'
            make_cases.stypy_param_names_list = ['dtype']
            make_cases.stypy_varargs_param_name = None
            make_cases.stypy_kwargs_param_name = None
            make_cases.stypy_call_defaults = defaults
            make_cases.stypy_call_varargs = varargs
            make_cases.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'make_cases', ['dtype'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'make_cases', localization, ['dtype'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'make_cases(...)' code ##################

            
            # Call to append(...): (line 170)
            # Processing the call arguments (line 170)
            
            # Call to matrix(...): (line 170)
            # Processing the call arguments (line 170)
            
            # Obtaining an instance of the builtin type 'list' (line 170)
            list_424968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 40), 'list')
            # Adding type elements to the builtin type 'list' instance (line 170)
            # Adding element type (line 170)
            
            # Obtaining an instance of the builtin type 'list' (line 170)
            list_424969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 41), 'list')
            # Adding type elements to the builtin type 'list' instance (line 170)
            # Adding element type (line 170)
            int_424970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 42), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 41), list_424969, int_424970)
            # Adding element type (line 170)
            int_424971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 44), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 41), list_424969, int_424971)
            # Adding element type (line 170)
            int_424972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 46), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 41), list_424969, int_424972)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 40), list_424968, list_424969)
            # Adding element type (line 170)
            
            # Obtaining an instance of the builtin type 'list' (line 170)
            list_424973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 49), 'list')
            # Adding type elements to the builtin type 'list' instance (line 170)
            # Adding element type (line 170)
            int_424974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 50), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 49), list_424973, int_424974)
            # Adding element type (line 170)
            int_424975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 52), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 49), list_424973, int_424975)
            # Adding element type (line 170)
            int_424976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 54), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 49), list_424973, int_424976)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 40), list_424968, list_424973)
            
            # Processing the call keyword arguments (line 170)
            # Getting the type of 'dtype' (line 170)
            dtype_424977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 65), 'dtype', False)
            keyword_424978 = dtype_424977
            kwargs_424979 = {'dtype': keyword_424978}
            # Getting the type of 'np' (line 170)
            np_424966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 30), 'np', False)
            # Obtaining the member 'matrix' of a type (line 170)
            matrix_424967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 30), np_424966, 'matrix')
            # Calling matrix(args, kwargs) (line 170)
            matrix_call_result_424980 = invoke(stypy.reporting.localization.Localization(__file__, 170, 30), matrix_424967, *[list_424968], **kwargs_424979)
            
            # Processing the call keyword arguments (line 170)
            kwargs_424981 = {}
            # Getting the type of 'self' (line 170)
            self_424963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'self', False)
            # Obtaining the member 'cases' of a type (line 170)
            cases_424964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 12), self_424963, 'cases')
            # Obtaining the member 'append' of a type (line 170)
            append_424965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 12), cases_424964, 'append')
            # Calling append(args, kwargs) (line 170)
            append_call_result_424982 = invoke(stypy.reporting.localization.Localization(__file__, 170, 12), append_424965, *[matrix_call_result_424980], **kwargs_424981)
            
            
            # Call to append(...): (line 171)
            # Processing the call arguments (line 171)
            
            # Call to array(...): (line 171)
            # Processing the call arguments (line 171)
            
            # Obtaining an instance of the builtin type 'list' (line 171)
            list_424988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 39), 'list')
            # Adding type elements to the builtin type 'list' instance (line 171)
            # Adding element type (line 171)
            
            # Obtaining an instance of the builtin type 'list' (line 171)
            list_424989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 40), 'list')
            # Adding type elements to the builtin type 'list' instance (line 171)
            # Adding element type (line 171)
            int_424990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 41), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 40), list_424989, int_424990)
            # Adding element type (line 171)
            int_424991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 43), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 40), list_424989, int_424991)
            # Adding element type (line 171)
            int_424992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 45), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 40), list_424989, int_424992)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 39), list_424988, list_424989)
            # Adding element type (line 171)
            
            # Obtaining an instance of the builtin type 'list' (line 171)
            list_424993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 48), 'list')
            # Adding type elements to the builtin type 'list' instance (line 171)
            # Adding element type (line 171)
            int_424994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 49), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 48), list_424993, int_424994)
            # Adding element type (line 171)
            int_424995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 51), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 48), list_424993, int_424995)
            # Adding element type (line 171)
            int_424996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 53), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 48), list_424993, int_424996)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 39), list_424988, list_424993)
            
            # Processing the call keyword arguments (line 171)
            # Getting the type of 'dtype' (line 171)
            dtype_424997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 64), 'dtype', False)
            keyword_424998 = dtype_424997
            kwargs_424999 = {'dtype': keyword_424998}
            # Getting the type of 'np' (line 171)
            np_424986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 30), 'np', False)
            # Obtaining the member 'array' of a type (line 171)
            array_424987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 30), np_424986, 'array')
            # Calling array(args, kwargs) (line 171)
            array_call_result_425000 = invoke(stypy.reporting.localization.Localization(__file__, 171, 30), array_424987, *[list_424988], **kwargs_424999)
            
            # Processing the call keyword arguments (line 171)
            kwargs_425001 = {}
            # Getting the type of 'self' (line 171)
            self_424983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'self', False)
            # Obtaining the member 'cases' of a type (line 171)
            cases_424984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 12), self_424983, 'cases')
            # Obtaining the member 'append' of a type (line 171)
            append_424985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 12), cases_424984, 'append')
            # Calling append(args, kwargs) (line 171)
            append_call_result_425002 = invoke(stypy.reporting.localization.Localization(__file__, 171, 12), append_424985, *[array_call_result_425000], **kwargs_425001)
            
            
            # Call to append(...): (line 172)
            # Processing the call arguments (line 172)
            
            # Call to csr_matrix(...): (line 172)
            # Processing the call arguments (line 172)
            
            # Obtaining an instance of the builtin type 'list' (line 172)
            list_425008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 48), 'list')
            # Adding type elements to the builtin type 'list' instance (line 172)
            # Adding element type (line 172)
            
            # Obtaining an instance of the builtin type 'list' (line 172)
            list_425009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 49), 'list')
            # Adding type elements to the builtin type 'list' instance (line 172)
            # Adding element type (line 172)
            int_425010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 50), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 49), list_425009, int_425010)
            # Adding element type (line 172)
            int_425011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 52), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 49), list_425009, int_425011)
            # Adding element type (line 172)
            int_425012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 54), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 49), list_425009, int_425012)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 48), list_425008, list_425009)
            # Adding element type (line 172)
            
            # Obtaining an instance of the builtin type 'list' (line 172)
            list_425013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 57), 'list')
            # Adding type elements to the builtin type 'list' instance (line 172)
            # Adding element type (line 172)
            int_425014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 58), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 57), list_425013, int_425014)
            # Adding element type (line 172)
            int_425015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 60), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 57), list_425013, int_425015)
            # Adding element type (line 172)
            int_425016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 62), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 57), list_425013, int_425016)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 48), list_425008, list_425013)
            
            # Processing the call keyword arguments (line 172)
            # Getting the type of 'dtype' (line 172)
            dtype_425017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 73), 'dtype', False)
            keyword_425018 = dtype_425017
            kwargs_425019 = {'dtype': keyword_425018}
            # Getting the type of 'sparse' (line 172)
            sparse_425006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 30), 'sparse', False)
            # Obtaining the member 'csr_matrix' of a type (line 172)
            csr_matrix_425007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 30), sparse_425006, 'csr_matrix')
            # Calling csr_matrix(args, kwargs) (line 172)
            csr_matrix_call_result_425020 = invoke(stypy.reporting.localization.Localization(__file__, 172, 30), csr_matrix_425007, *[list_425008], **kwargs_425019)
            
            # Processing the call keyword arguments (line 172)
            kwargs_425021 = {}
            # Getting the type of 'self' (line 172)
            self_425003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'self', False)
            # Obtaining the member 'cases' of a type (line 172)
            cases_425004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 12), self_425003, 'cases')
            # Obtaining the member 'append' of a type (line 172)
            append_425005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 12), cases_425004, 'append')
            # Calling append(args, kwargs) (line 172)
            append_call_result_425022 = invoke(stypy.reporting.localization.Localization(__file__, 172, 12), append_425005, *[csr_matrix_call_result_425020], **kwargs_425021)
            

            @norecursion
            def mv(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'mv'
                module_type_store = module_type_store.open_function_context('mv', 176, 12, False)
                
                # Passed parameters checking function
                mv.stypy_localization = localization
                mv.stypy_type_of_self = None
                mv.stypy_type_store = module_type_store
                mv.stypy_function_name = 'mv'
                mv.stypy_param_names_list = ['x', 'dtype']
                mv.stypy_varargs_param_name = None
                mv.stypy_kwargs_param_name = None
                mv.stypy_call_defaults = defaults
                mv.stypy_call_varargs = varargs
                mv.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, 'mv', ['x', 'dtype'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'mv', localization, ['x', 'dtype'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'mv(...)' code ##################

                
                # Assigning a Call to a Name (line 177):
                
                # Assigning a Call to a Name (line 177):
                
                # Call to array(...): (line 177)
                # Processing the call arguments (line 177)
                
                # Obtaining an instance of the builtin type 'list' (line 177)
                list_425025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 29), 'list')
                # Adding type elements to the builtin type 'list' instance (line 177)
                # Adding element type (line 177)
                int_425026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 30), 'int')
                
                # Obtaining the type of the subscript
                int_425027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 36), 'int')
                # Getting the type of 'x' (line 177)
                x_425028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 34), 'x', False)
                # Obtaining the member '__getitem__' of a type (line 177)
                getitem___425029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 34), x_425028, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 177)
                subscript_call_result_425030 = invoke(stypy.reporting.localization.Localization(__file__, 177, 34), getitem___425029, int_425027)
                
                # Applying the binary operator '*' (line 177)
                result_mul_425031 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 30), '*', int_425026, subscript_call_result_425030)
                
                int_425032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 41), 'int')
                
                # Obtaining the type of the subscript
                int_425033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 47), 'int')
                # Getting the type of 'x' (line 177)
                x_425034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 45), 'x', False)
                # Obtaining the member '__getitem__' of a type (line 177)
                getitem___425035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 45), x_425034, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 177)
                subscript_call_result_425036 = invoke(stypy.reporting.localization.Localization(__file__, 177, 45), getitem___425035, int_425033)
                
                # Applying the binary operator '*' (line 177)
                result_mul_425037 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 41), '*', int_425032, subscript_call_result_425036)
                
                # Applying the binary operator '+' (line 177)
                result_add_425038 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 30), '+', result_mul_425031, result_mul_425037)
                
                int_425039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 52), 'int')
                
                # Obtaining the type of the subscript
                int_425040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 58), 'int')
                # Getting the type of 'x' (line 177)
                x_425041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 56), 'x', False)
                # Obtaining the member '__getitem__' of a type (line 177)
                getitem___425042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 56), x_425041, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 177)
                subscript_call_result_425043 = invoke(stypy.reporting.localization.Localization(__file__, 177, 56), getitem___425042, int_425040)
                
                # Applying the binary operator '*' (line 177)
                result_mul_425044 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 52), '*', int_425039, subscript_call_result_425043)
                
                # Applying the binary operator '+' (line 177)
                result_add_425045 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 50), '+', result_add_425038, result_mul_425044)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 29), list_425025, result_add_425045)
                # Adding element type (line 177)
                int_425046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 30), 'int')
                
                # Obtaining the type of the subscript
                int_425047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 36), 'int')
                # Getting the type of 'x' (line 178)
                x_425048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 34), 'x', False)
                # Obtaining the member '__getitem__' of a type (line 178)
                getitem___425049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 34), x_425048, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 178)
                subscript_call_result_425050 = invoke(stypy.reporting.localization.Localization(__file__, 178, 34), getitem___425049, int_425047)
                
                # Applying the binary operator '*' (line 178)
                result_mul_425051 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 30), '*', int_425046, subscript_call_result_425050)
                
                int_425052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 41), 'int')
                
                # Obtaining the type of the subscript
                int_425053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 47), 'int')
                # Getting the type of 'x' (line 178)
                x_425054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 45), 'x', False)
                # Obtaining the member '__getitem__' of a type (line 178)
                getitem___425055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 45), x_425054, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 178)
                subscript_call_result_425056 = invoke(stypy.reporting.localization.Localization(__file__, 178, 45), getitem___425055, int_425053)
                
                # Applying the binary operator '*' (line 178)
                result_mul_425057 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 41), '*', int_425052, subscript_call_result_425056)
                
                # Applying the binary operator '+' (line 178)
                result_add_425058 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 30), '+', result_mul_425051, result_mul_425057)
                
                int_425059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 52), 'int')
                
                # Obtaining the type of the subscript
                int_425060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 58), 'int')
                # Getting the type of 'x' (line 178)
                x_425061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 56), 'x', False)
                # Obtaining the member '__getitem__' of a type (line 178)
                getitem___425062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 56), x_425061, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 178)
                subscript_call_result_425063 = invoke(stypy.reporting.localization.Localization(__file__, 178, 56), getitem___425062, int_425060)
                
                # Applying the binary operator '*' (line 178)
                result_mul_425064 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 52), '*', int_425059, subscript_call_result_425063)
                
                # Applying the binary operator '+' (line 178)
                result_add_425065 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 50), '+', result_add_425058, result_mul_425064)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 29), list_425025, result_add_425065)
                
                # Processing the call keyword arguments (line 177)
                # Getting the type of 'dtype' (line 178)
                dtype_425066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 69), 'dtype', False)
                keyword_425067 = dtype_425066
                kwargs_425068 = {'dtype': keyword_425067}
                # Getting the type of 'np' (line 177)
                np_425023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 20), 'np', False)
                # Obtaining the member 'array' of a type (line 177)
                array_425024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 20), np_425023, 'array')
                # Calling array(args, kwargs) (line 177)
                array_call_result_425069 = invoke(stypy.reporting.localization.Localization(__file__, 177, 20), array_425024, *[list_425025], **kwargs_425068)
                
                # Assigning a type to the variable 'y' (line 177)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 16), 'y', array_call_result_425069)
                
                
                
                # Call to len(...): (line 179)
                # Processing the call arguments (line 179)
                # Getting the type of 'x' (line 179)
                x_425071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 23), 'x', False)
                # Obtaining the member 'shape' of a type (line 179)
                shape_425072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 23), x_425071, 'shape')
                # Processing the call keyword arguments (line 179)
                kwargs_425073 = {}
                # Getting the type of 'len' (line 179)
                len_425070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 19), 'len', False)
                # Calling len(args, kwargs) (line 179)
                len_call_result_425074 = invoke(stypy.reporting.localization.Localization(__file__, 179, 19), len_425070, *[shape_425072], **kwargs_425073)
                
                int_425075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 35), 'int')
                # Applying the binary operator '==' (line 179)
                result_eq_425076 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 19), '==', len_call_result_425074, int_425075)
                
                # Testing the type of an if condition (line 179)
                if_condition_425077 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 179, 16), result_eq_425076)
                # Assigning a type to the variable 'if_condition_425077' (line 179)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'if_condition_425077', if_condition_425077)
                # SSA begins for if statement (line 179)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 180):
                
                # Assigning a Call to a Name (line 180):
                
                # Call to reshape(...): (line 180)
                # Processing the call arguments (line 180)
                int_425080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 34), 'int')
                int_425081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 38), 'int')
                # Processing the call keyword arguments (line 180)
                kwargs_425082 = {}
                # Getting the type of 'y' (line 180)
                y_425078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 24), 'y', False)
                # Obtaining the member 'reshape' of a type (line 180)
                reshape_425079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 24), y_425078, 'reshape')
                # Calling reshape(args, kwargs) (line 180)
                reshape_call_result_425083 = invoke(stypy.reporting.localization.Localization(__file__, 180, 24), reshape_425079, *[int_425080, int_425081], **kwargs_425082)
                
                # Assigning a type to the variable 'y' (line 180)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 20), 'y', reshape_call_result_425083)
                # SSA join for if statement (line 179)
                module_type_store = module_type_store.join_ssa_context()
                
                # Getting the type of 'y' (line 181)
                y_425084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 23), 'y')
                # Assigning a type to the variable 'stypy_return_type' (line 181)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'stypy_return_type', y_425084)
                
                # ################# End of 'mv(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'mv' in the type store
                # Getting the type of 'stypy_return_type' (line 176)
                stypy_return_type_425085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_425085)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'mv'
                return stypy_return_type_425085

            # Assigning a type to the variable 'mv' (line 176)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'mv', mv)

            @norecursion
            def rmv(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'rmv'
                module_type_store = module_type_store.open_function_context('rmv', 183, 12, False)
                
                # Passed parameters checking function
                rmv.stypy_localization = localization
                rmv.stypy_type_of_self = None
                rmv.stypy_type_store = module_type_store
                rmv.stypy_function_name = 'rmv'
                rmv.stypy_param_names_list = ['x', 'dtype']
                rmv.stypy_varargs_param_name = None
                rmv.stypy_kwargs_param_name = None
                rmv.stypy_call_defaults = defaults
                rmv.stypy_call_varargs = varargs
                rmv.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, 'rmv', ['x', 'dtype'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'rmv', localization, ['x', 'dtype'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'rmv(...)' code ##################

                
                # Call to array(...): (line 184)
                # Processing the call arguments (line 184)
                
                # Obtaining an instance of the builtin type 'list' (line 184)
                list_425088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 32), 'list')
                # Adding type elements to the builtin type 'list' instance (line 184)
                # Adding element type (line 184)
                int_425089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 33), 'int')
                
                # Obtaining the type of the subscript
                int_425090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 39), 'int')
                # Getting the type of 'x' (line 184)
                x_425091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 37), 'x', False)
                # Obtaining the member '__getitem__' of a type (line 184)
                getitem___425092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 37), x_425091, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 184)
                subscript_call_result_425093 = invoke(stypy.reporting.localization.Localization(__file__, 184, 37), getitem___425092, int_425090)
                
                # Applying the binary operator '*' (line 184)
                result_mul_425094 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 33), '*', int_425089, subscript_call_result_425093)
                
                int_425095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 44), 'int')
                
                # Obtaining the type of the subscript
                int_425096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 50), 'int')
                # Getting the type of 'x' (line 184)
                x_425097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 48), 'x', False)
                # Obtaining the member '__getitem__' of a type (line 184)
                getitem___425098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 48), x_425097, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 184)
                subscript_call_result_425099 = invoke(stypy.reporting.localization.Localization(__file__, 184, 48), getitem___425098, int_425096)
                
                # Applying the binary operator '*' (line 184)
                result_mul_425100 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 44), '*', int_425095, subscript_call_result_425099)
                
                # Applying the binary operator '+' (line 184)
                result_add_425101 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 33), '+', result_mul_425094, result_mul_425100)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 32), list_425088, result_add_425101)
                # Adding element type (line 184)
                int_425102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 33), 'int')
                
                # Obtaining the type of the subscript
                int_425103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 39), 'int')
                # Getting the type of 'x' (line 185)
                x_425104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 37), 'x', False)
                # Obtaining the member '__getitem__' of a type (line 185)
                getitem___425105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 37), x_425104, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 185)
                subscript_call_result_425106 = invoke(stypy.reporting.localization.Localization(__file__, 185, 37), getitem___425105, int_425103)
                
                # Applying the binary operator '*' (line 185)
                result_mul_425107 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 33), '*', int_425102, subscript_call_result_425106)
                
                int_425108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 44), 'int')
                
                # Obtaining the type of the subscript
                int_425109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 50), 'int')
                # Getting the type of 'x' (line 185)
                x_425110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 48), 'x', False)
                # Obtaining the member '__getitem__' of a type (line 185)
                getitem___425111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 48), x_425110, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 185)
                subscript_call_result_425112 = invoke(stypy.reporting.localization.Localization(__file__, 185, 48), getitem___425111, int_425109)
                
                # Applying the binary operator '*' (line 185)
                result_mul_425113 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 44), '*', int_425108, subscript_call_result_425112)
                
                # Applying the binary operator '+' (line 185)
                result_add_425114 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 33), '+', result_mul_425107, result_mul_425113)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 32), list_425088, result_add_425114)
                # Adding element type (line 184)
                int_425115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 33), 'int')
                
                # Obtaining the type of the subscript
                int_425116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 39), 'int')
                # Getting the type of 'x' (line 186)
                x_425117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 37), 'x', False)
                # Obtaining the member '__getitem__' of a type (line 186)
                getitem___425118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 37), x_425117, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 186)
                subscript_call_result_425119 = invoke(stypy.reporting.localization.Localization(__file__, 186, 37), getitem___425118, int_425116)
                
                # Applying the binary operator '*' (line 186)
                result_mul_425120 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 33), '*', int_425115, subscript_call_result_425119)
                
                int_425121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 44), 'int')
                
                # Obtaining the type of the subscript
                int_425122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 50), 'int')
                # Getting the type of 'x' (line 186)
                x_425123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 48), 'x', False)
                # Obtaining the member '__getitem__' of a type (line 186)
                getitem___425124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 48), x_425123, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 186)
                subscript_call_result_425125 = invoke(stypy.reporting.localization.Localization(__file__, 186, 48), getitem___425124, int_425122)
                
                # Applying the binary operator '*' (line 186)
                result_mul_425126 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 44), '*', int_425121, subscript_call_result_425125)
                
                # Applying the binary operator '+' (line 186)
                result_add_425127 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 33), '+', result_mul_425120, result_mul_425126)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 32), list_425088, result_add_425127)
                
                # Processing the call keyword arguments (line 184)
                # Getting the type of 'dtype' (line 186)
                dtype_425128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 61), 'dtype', False)
                keyword_425129 = dtype_425128
                kwargs_425130 = {'dtype': keyword_425129}
                # Getting the type of 'np' (line 184)
                np_425086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 23), 'np', False)
                # Obtaining the member 'array' of a type (line 184)
                array_425087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 23), np_425086, 'array')
                # Calling array(args, kwargs) (line 184)
                array_call_result_425131 = invoke(stypy.reporting.localization.Localization(__file__, 184, 23), array_425087, *[list_425088], **kwargs_425130)
                
                # Assigning a type to the variable 'stypy_return_type' (line 184)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'stypy_return_type', array_call_result_425131)
                
                # ################# End of 'rmv(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'rmv' in the type store
                # Getting the type of 'stypy_return_type' (line 183)
                stypy_return_type_425132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_425132)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'rmv'
                return stypy_return_type_425132

            # Assigning a type to the variable 'rmv' (line 183)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'rmv', rmv)
            # Declaration of the 'BaseMatlike' class
            # Getting the type of 'interface' (line 188)
            interface_425133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 30), 'interface')
            # Obtaining the member 'LinearOperator' of a type (line 188)
            LinearOperator_425134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 30), interface_425133, 'LinearOperator')

            class BaseMatlike(LinearOperator_425134, ):

                @norecursion
                def __init__(type_of_self, localization, *varargs, **kwargs):
                    global module_type_store
                    # Assign values to the parameters with defaults
                    defaults = []
                    # Create a new context for function '__init__'
                    module_type_store = module_type_store.open_function_context('__init__', 189, 16, False)
                    # Assigning a type to the variable 'self' (line 190)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), 'self', type_of_self)
                    
                    # Passed parameters checking function
                    arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMatlike.__init__', ['dtype'], None, None, defaults, varargs, kwargs)

                    if is_error_type(arguments):
                        # Destroy the current context
                        module_type_store = module_type_store.close_function_context()
                        return

                    # Initialize method data
                    init_call_information(module_type_store, '__init__', localization, ['dtype'], arguments)
                    
                    # Default return type storage variable (SSA)
                    # Assigning a type to the variable 'stypy_return_type'
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                    
                    
                    # ################# Begin of '__init__(...)' code ##################

                    
                    # Assigning a Call to a Attribute (line 190):
                    
                    # Assigning a Call to a Attribute (line 190):
                    
                    # Call to dtype(...): (line 190)
                    # Processing the call arguments (line 190)
                    # Getting the type of 'dtype' (line 190)
                    dtype_425137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 42), 'dtype', False)
                    # Processing the call keyword arguments (line 190)
                    kwargs_425138 = {}
                    # Getting the type of 'np' (line 190)
                    np_425135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 33), 'np', False)
                    # Obtaining the member 'dtype' of a type (line 190)
                    dtype_425136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 33), np_425135, 'dtype')
                    # Calling dtype(args, kwargs) (line 190)
                    dtype_call_result_425139 = invoke(stypy.reporting.localization.Localization(__file__, 190, 33), dtype_425136, *[dtype_425137], **kwargs_425138)
                    
                    # Getting the type of 'self' (line 190)
                    self_425140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 20), 'self')
                    # Setting the type of the member 'dtype' of a type (line 190)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 20), self_425140, 'dtype', dtype_call_result_425139)
                    
                    # Assigning a Tuple to a Attribute (line 191):
                    
                    # Assigning a Tuple to a Attribute (line 191):
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 191)
                    tuple_425141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 34), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 191)
                    # Adding element type (line 191)
                    int_425142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 34), 'int')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 34), tuple_425141, int_425142)
                    # Adding element type (line 191)
                    int_425143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 36), 'int')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 34), tuple_425141, int_425143)
                    
                    # Getting the type of 'self' (line 191)
                    self_425144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 20), 'self')
                    # Setting the type of the member 'shape' of a type (line 191)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 20), self_425144, 'shape', tuple_425141)
                    
                    # ################# End of '__init__(...)' code ##################

                    # Teardown call information
                    teardown_call_information(localization, arguments)
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()


                @norecursion
                def _matvec(type_of_self, localization, *varargs, **kwargs):
                    global module_type_store
                    # Assign values to the parameters with defaults
                    defaults = []
                    # Create a new context for function '_matvec'
                    module_type_store = module_type_store.open_function_context('_matvec', 193, 16, False)
                    # Assigning a type to the variable 'self' (line 194)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 16), 'self', type_of_self)
                    
                    # Passed parameters checking function
                    BaseMatlike._matvec.__dict__.__setitem__('stypy_localization', localization)
                    BaseMatlike._matvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                    BaseMatlike._matvec.__dict__.__setitem__('stypy_type_store', module_type_store)
                    BaseMatlike._matvec.__dict__.__setitem__('stypy_function_name', 'BaseMatlike._matvec')
                    BaseMatlike._matvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
                    BaseMatlike._matvec.__dict__.__setitem__('stypy_varargs_param_name', None)
                    BaseMatlike._matvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
                    BaseMatlike._matvec.__dict__.__setitem__('stypy_call_defaults', defaults)
                    BaseMatlike._matvec.__dict__.__setitem__('stypy_call_varargs', varargs)
                    BaseMatlike._matvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                    BaseMatlike._matvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
                    arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMatlike._matvec', ['x'], None, None, defaults, varargs, kwargs)

                    if is_error_type(arguments):
                        # Destroy the current context
                        module_type_store = module_type_store.close_function_context()
                        return arguments

                    # Initialize method data
                    init_call_information(module_type_store, '_matvec', localization, ['x'], arguments)
                    
                    # Default return type storage variable (SSA)
                    # Assigning a type to the variable 'stypy_return_type'
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                    
                    
                    # ################# Begin of '_matvec(...)' code ##################

                    
                    # Call to mv(...): (line 194)
                    # Processing the call arguments (line 194)
                    # Getting the type of 'x' (line 194)
                    x_425146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 30), 'x', False)
                    # Getting the type of 'self' (line 194)
                    self_425147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 33), 'self', False)
                    # Obtaining the member 'dtype' of a type (line 194)
                    dtype_425148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 33), self_425147, 'dtype')
                    # Processing the call keyword arguments (line 194)
                    kwargs_425149 = {}
                    # Getting the type of 'mv' (line 194)
                    mv_425145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 27), 'mv', False)
                    # Calling mv(args, kwargs) (line 194)
                    mv_call_result_425150 = invoke(stypy.reporting.localization.Localization(__file__, 194, 27), mv_425145, *[x_425146, dtype_425148], **kwargs_425149)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 194)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 20), 'stypy_return_type', mv_call_result_425150)
                    
                    # ################# End of '_matvec(...)' code ##################

                    # Teardown call information
                    teardown_call_information(localization, arguments)
                    
                    # Storing the return type of function '_matvec' in the type store
                    # Getting the type of 'stypy_return_type' (line 193)
                    stypy_return_type_425151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 16), 'stypy_return_type')
                    module_type_store.store_return_type_of_current_context(stypy_return_type_425151)
                    
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    
                    # Return type of the function '_matvec'
                    return stypy_return_type_425151

            
            # Assigning a type to the variable 'BaseMatlike' (line 188)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'BaseMatlike', BaseMatlike)
            # Declaration of the 'HasRmatvec' class
            # Getting the type of 'BaseMatlike' (line 196)
            BaseMatlike_425152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 29), 'BaseMatlike')

            class HasRmatvec(BaseMatlike_425152, ):

                @norecursion
                def _rmatvec(type_of_self, localization, *varargs, **kwargs):
                    global module_type_store
                    # Assign values to the parameters with defaults
                    defaults = []
                    # Create a new context for function '_rmatvec'
                    module_type_store = module_type_store.open_function_context('_rmatvec', 197, 16, False)
                    # Assigning a type to the variable 'self' (line 198)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'self', type_of_self)
                    
                    # Passed parameters checking function
                    HasRmatvec._rmatvec.__dict__.__setitem__('stypy_localization', localization)
                    HasRmatvec._rmatvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                    HasRmatvec._rmatvec.__dict__.__setitem__('stypy_type_store', module_type_store)
                    HasRmatvec._rmatvec.__dict__.__setitem__('stypy_function_name', 'HasRmatvec._rmatvec')
                    HasRmatvec._rmatvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
                    HasRmatvec._rmatvec.__dict__.__setitem__('stypy_varargs_param_name', None)
                    HasRmatvec._rmatvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
                    HasRmatvec._rmatvec.__dict__.__setitem__('stypy_call_defaults', defaults)
                    HasRmatvec._rmatvec.__dict__.__setitem__('stypy_call_varargs', varargs)
                    HasRmatvec._rmatvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                    HasRmatvec._rmatvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
                    arguments = process_argument_values(localization, type_of_self, module_type_store, 'HasRmatvec._rmatvec', ['x'], None, None, defaults, varargs, kwargs)

                    if is_error_type(arguments):
                        # Destroy the current context
                        module_type_store = module_type_store.close_function_context()
                        return arguments

                    # Initialize method data
                    init_call_information(module_type_store, '_rmatvec', localization, ['x'], arguments)
                    
                    # Default return type storage variable (SSA)
                    # Assigning a type to the variable 'stypy_return_type'
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                    
                    
                    # ################# Begin of '_rmatvec(...)' code ##################

                    
                    # Call to rmv(...): (line 198)
                    # Processing the call arguments (line 198)
                    # Getting the type of 'x' (line 198)
                    x_425154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 31), 'x', False)
                    # Getting the type of 'self' (line 198)
                    self_425155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 34), 'self', False)
                    # Obtaining the member 'dtype' of a type (line 198)
                    dtype_425156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 34), self_425155, 'dtype')
                    # Processing the call keyword arguments (line 198)
                    kwargs_425157 = {}
                    # Getting the type of 'rmv' (line 198)
                    rmv_425153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 27), 'rmv', False)
                    # Calling rmv(args, kwargs) (line 198)
                    rmv_call_result_425158 = invoke(stypy.reporting.localization.Localization(__file__, 198, 27), rmv_425153, *[x_425154, dtype_425156], **kwargs_425157)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 198)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 20), 'stypy_return_type', rmv_call_result_425158)
                    
                    # ################# End of '_rmatvec(...)' code ##################

                    # Teardown call information
                    teardown_call_information(localization, arguments)
                    
                    # Storing the return type of function '_rmatvec' in the type store
                    # Getting the type of 'stypy_return_type' (line 197)
                    stypy_return_type_425159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 16), 'stypy_return_type')
                    module_type_store.store_return_type_of_current_context(stypy_return_type_425159)
                    
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    
                    # Return type of the function '_rmatvec'
                    return stypy_return_type_425159

            
            # Assigning a type to the variable 'HasRmatvec' (line 196)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'HasRmatvec', HasRmatvec)
            # Declaration of the 'HasAdjoint' class
            # Getting the type of 'BaseMatlike' (line 200)
            BaseMatlike_425160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 29), 'BaseMatlike')

            class HasAdjoint(BaseMatlike_425160, ):

                @norecursion
                def _adjoint(type_of_self, localization, *varargs, **kwargs):
                    global module_type_store
                    # Assign values to the parameters with defaults
                    defaults = []
                    # Create a new context for function '_adjoint'
                    module_type_store = module_type_store.open_function_context('_adjoint', 201, 16, False)
                    # Assigning a type to the variable 'self' (line 202)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'self', type_of_self)
                    
                    # Passed parameters checking function
                    HasAdjoint._adjoint.__dict__.__setitem__('stypy_localization', localization)
                    HasAdjoint._adjoint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                    HasAdjoint._adjoint.__dict__.__setitem__('stypy_type_store', module_type_store)
                    HasAdjoint._adjoint.__dict__.__setitem__('stypy_function_name', 'HasAdjoint._adjoint')
                    HasAdjoint._adjoint.__dict__.__setitem__('stypy_param_names_list', [])
                    HasAdjoint._adjoint.__dict__.__setitem__('stypy_varargs_param_name', None)
                    HasAdjoint._adjoint.__dict__.__setitem__('stypy_kwargs_param_name', None)
                    HasAdjoint._adjoint.__dict__.__setitem__('stypy_call_defaults', defaults)
                    HasAdjoint._adjoint.__dict__.__setitem__('stypy_call_varargs', varargs)
                    HasAdjoint._adjoint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                    HasAdjoint._adjoint.__dict__.__setitem__('stypy_declared_arg_number', 1)
                    arguments = process_argument_values(localization, type_of_self, module_type_store, 'HasAdjoint._adjoint', [], None, None, defaults, varargs, kwargs)

                    if is_error_type(arguments):
                        # Destroy the current context
                        module_type_store = module_type_store.close_function_context()
                        return arguments

                    # Initialize method data
                    init_call_information(module_type_store, '_adjoint', localization, [], arguments)
                    
                    # Default return type storage variable (SSA)
                    # Assigning a type to the variable 'stypy_return_type'
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                    
                    
                    # ################# Begin of '_adjoint(...)' code ##################

                    
                    # Assigning a Tuple to a Name (line 202):
                    
                    # Assigning a Tuple to a Name (line 202):
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 202)
                    tuple_425161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 28), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 202)
                    # Adding element type (line 202)
                    
                    # Obtaining the type of the subscript
                    int_425162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 39), 'int')
                    # Getting the type of 'self' (line 202)
                    self_425163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 28), 'self')
                    # Obtaining the member 'shape' of a type (line 202)
                    shape_425164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 28), self_425163, 'shape')
                    # Obtaining the member '__getitem__' of a type (line 202)
                    getitem___425165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 28), shape_425164, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 202)
                    subscript_call_result_425166 = invoke(stypy.reporting.localization.Localization(__file__, 202, 28), getitem___425165, int_425162)
                    
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 28), tuple_425161, subscript_call_result_425166)
                    # Adding element type (line 202)
                    
                    # Obtaining the type of the subscript
                    int_425167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 54), 'int')
                    # Getting the type of 'self' (line 202)
                    self_425168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 43), 'self')
                    # Obtaining the member 'shape' of a type (line 202)
                    shape_425169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 43), self_425168, 'shape')
                    # Obtaining the member '__getitem__' of a type (line 202)
                    getitem___425170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 43), shape_425169, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 202)
                    subscript_call_result_425171 = invoke(stypy.reporting.localization.Localization(__file__, 202, 43), getitem___425170, int_425167)
                    
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 28), tuple_425161, subscript_call_result_425171)
                    
                    # Assigning a type to the variable 'shape' (line 202)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 20), 'shape', tuple_425161)
                    
                    # Assigning a Call to a Name (line 203):
                    
                    # Assigning a Call to a Name (line 203):
                    
                    # Call to partial(...): (line 203)
                    # Processing the call arguments (line 203)
                    # Getting the type of 'rmv' (line 203)
                    rmv_425173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 37), 'rmv', False)
                    # Processing the call keyword arguments (line 203)
                    # Getting the type of 'self' (line 203)
                    self_425174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 48), 'self', False)
                    # Obtaining the member 'dtype' of a type (line 203)
                    dtype_425175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 48), self_425174, 'dtype')
                    keyword_425176 = dtype_425175
                    kwargs_425177 = {'dtype': keyword_425176}
                    # Getting the type of 'partial' (line 203)
                    partial_425172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 29), 'partial', False)
                    # Calling partial(args, kwargs) (line 203)
                    partial_call_result_425178 = invoke(stypy.reporting.localization.Localization(__file__, 203, 29), partial_425172, *[rmv_425173], **kwargs_425177)
                    
                    # Assigning a type to the variable 'matvec' (line 203)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 20), 'matvec', partial_call_result_425178)
                    
                    # Assigning a Call to a Name (line 204):
                    
                    # Assigning a Call to a Name (line 204):
                    
                    # Call to partial(...): (line 204)
                    # Processing the call arguments (line 204)
                    # Getting the type of 'mv' (line 204)
                    mv_425180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 38), 'mv', False)
                    # Processing the call keyword arguments (line 204)
                    # Getting the type of 'self' (line 204)
                    self_425181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 48), 'self', False)
                    # Obtaining the member 'dtype' of a type (line 204)
                    dtype_425182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 48), self_425181, 'dtype')
                    keyword_425183 = dtype_425182
                    kwargs_425184 = {'dtype': keyword_425183}
                    # Getting the type of 'partial' (line 204)
                    partial_425179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 30), 'partial', False)
                    # Calling partial(args, kwargs) (line 204)
                    partial_call_result_425185 = invoke(stypy.reporting.localization.Localization(__file__, 204, 30), partial_425179, *[mv_425180], **kwargs_425184)
                    
                    # Assigning a type to the variable 'rmatvec' (line 204)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 20), 'rmatvec', partial_call_result_425185)
                    
                    # Call to LinearOperator(...): (line 205)
                    # Processing the call keyword arguments (line 205)
                    # Getting the type of 'matvec' (line 205)
                    matvec_425188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 59), 'matvec', False)
                    keyword_425189 = matvec_425188
                    # Getting the type of 'rmatvec' (line 206)
                    rmatvec_425190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 60), 'rmatvec', False)
                    keyword_425191 = rmatvec_425190
                    # Getting the type of 'self' (line 207)
                    self_425192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 58), 'self', False)
                    # Obtaining the member 'dtype' of a type (line 207)
                    dtype_425193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 58), self_425192, 'dtype')
                    keyword_425194 = dtype_425193
                    # Getting the type of 'shape' (line 208)
                    shape_425195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 58), 'shape', False)
                    keyword_425196 = shape_425195
                    kwargs_425197 = {'dtype': keyword_425194, 'shape': keyword_425196, 'rmatvec': keyword_425191, 'matvec': keyword_425189}
                    # Getting the type of 'interface' (line 205)
                    interface_425186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 27), 'interface', False)
                    # Obtaining the member 'LinearOperator' of a type (line 205)
                    LinearOperator_425187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 27), interface_425186, 'LinearOperator')
                    # Calling LinearOperator(args, kwargs) (line 205)
                    LinearOperator_call_result_425198 = invoke(stypy.reporting.localization.Localization(__file__, 205, 27), LinearOperator_425187, *[], **kwargs_425197)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 205)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'stypy_return_type', LinearOperator_call_result_425198)
                    
                    # ################# End of '_adjoint(...)' code ##################

                    # Teardown call information
                    teardown_call_information(localization, arguments)
                    
                    # Storing the return type of function '_adjoint' in the type store
                    # Getting the type of 'stypy_return_type' (line 201)
                    stypy_return_type_425199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'stypy_return_type')
                    module_type_store.store_return_type_of_current_context(stypy_return_type_425199)
                    
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    
                    # Return type of the function '_adjoint'
                    return stypy_return_type_425199

            
            # Assigning a type to the variable 'HasAdjoint' (line 200)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'HasAdjoint', HasAdjoint)
            
            # Call to append(...): (line 210)
            # Processing the call arguments (line 210)
            
            # Call to HasRmatvec(...): (line 210)
            # Processing the call arguments (line 210)
            # Getting the type of 'dtype' (line 210)
            dtype_425204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 41), 'dtype', False)
            # Processing the call keyword arguments (line 210)
            kwargs_425205 = {}
            # Getting the type of 'HasRmatvec' (line 210)
            HasRmatvec_425203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 30), 'HasRmatvec', False)
            # Calling HasRmatvec(args, kwargs) (line 210)
            HasRmatvec_call_result_425206 = invoke(stypy.reporting.localization.Localization(__file__, 210, 30), HasRmatvec_425203, *[dtype_425204], **kwargs_425205)
            
            # Processing the call keyword arguments (line 210)
            kwargs_425207 = {}
            # Getting the type of 'self' (line 210)
            self_425200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'self', False)
            # Obtaining the member 'cases' of a type (line 210)
            cases_425201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 12), self_425200, 'cases')
            # Obtaining the member 'append' of a type (line 210)
            append_425202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 12), cases_425201, 'append')
            # Calling append(args, kwargs) (line 210)
            append_call_result_425208 = invoke(stypy.reporting.localization.Localization(__file__, 210, 12), append_425202, *[HasRmatvec_call_result_425206], **kwargs_425207)
            
            
            # Call to append(...): (line 211)
            # Processing the call arguments (line 211)
            
            # Call to HasAdjoint(...): (line 211)
            # Processing the call arguments (line 211)
            # Getting the type of 'dtype' (line 211)
            dtype_425213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 41), 'dtype', False)
            # Processing the call keyword arguments (line 211)
            kwargs_425214 = {}
            # Getting the type of 'HasAdjoint' (line 211)
            HasAdjoint_425212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 30), 'HasAdjoint', False)
            # Calling HasAdjoint(args, kwargs) (line 211)
            HasAdjoint_call_result_425215 = invoke(stypy.reporting.localization.Localization(__file__, 211, 30), HasAdjoint_425212, *[dtype_425213], **kwargs_425214)
            
            # Processing the call keyword arguments (line 211)
            kwargs_425216 = {}
            # Getting the type of 'self' (line 211)
            self_425209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'self', False)
            # Obtaining the member 'cases' of a type (line 211)
            cases_425210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 12), self_425209, 'cases')
            # Obtaining the member 'append' of a type (line 211)
            append_425211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 12), cases_425210, 'append')
            # Calling append(args, kwargs) (line 211)
            append_call_result_425217 = invoke(stypy.reporting.localization.Localization(__file__, 211, 12), append_425211, *[HasAdjoint_call_result_425215], **kwargs_425216)
            
            
            # ################# End of 'make_cases(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'make_cases' in the type store
            # Getting the type of 'stypy_return_type' (line 169)
            stypy_return_type_425218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_425218)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'make_cases'
            return stypy_return_type_425218

        # Assigning a type to the variable 'make_cases' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'make_cases', make_cases)
        
        # Call to make_cases(...): (line 213)
        # Processing the call arguments (line 213)
        str_425220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 19), 'str', 'int32')
        # Processing the call keyword arguments (line 213)
        kwargs_425221 = {}
        # Getting the type of 'make_cases' (line 213)
        make_cases_425219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'make_cases', False)
        # Calling make_cases(args, kwargs) (line 213)
        make_cases_call_result_425222 = invoke(stypy.reporting.localization.Localization(__file__, 213, 8), make_cases_425219, *[str_425220], **kwargs_425221)
        
        
        # Call to make_cases(...): (line 214)
        # Processing the call arguments (line 214)
        str_425224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 19), 'str', 'float32')
        # Processing the call keyword arguments (line 214)
        kwargs_425225 = {}
        # Getting the type of 'make_cases' (line 214)
        make_cases_425223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'make_cases', False)
        # Calling make_cases(args, kwargs) (line 214)
        make_cases_call_result_425226 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), make_cases_425223, *[str_425224], **kwargs_425225)
        
        
        # Call to make_cases(...): (line 215)
        # Processing the call arguments (line 215)
        str_425228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 19), 'str', 'float64')
        # Processing the call keyword arguments (line 215)
        kwargs_425229 = {}
        # Getting the type of 'make_cases' (line 215)
        make_cases_425227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'make_cases', False)
        # Calling make_cases(args, kwargs) (line 215)
        make_cases_call_result_425230 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), make_cases_425227, *[str_425228], **kwargs_425229)
        
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 166)
        stypy_return_type_425231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_425231)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_425231


    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 217, 4, False)
        # Assigning a type to the variable 'self' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestAsLinearOperator.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestAsLinearOperator.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestAsLinearOperator.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestAsLinearOperator.test_basic.__dict__.__setitem__('stypy_function_name', 'TestAsLinearOperator.test_basic')
        TestAsLinearOperator.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestAsLinearOperator.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestAsLinearOperator.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestAsLinearOperator.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestAsLinearOperator.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestAsLinearOperator.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestAsLinearOperator.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestAsLinearOperator.test_basic', [], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'self' (line 219)
        self_425232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 17), 'self')
        # Obtaining the member 'cases' of a type (line 219)
        cases_425233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 17), self_425232, 'cases')
        # Testing the type of a for loop iterable (line 219)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 219, 8), cases_425233)
        # Getting the type of the for loop variable (line 219)
        for_loop_var_425234 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 219, 8), cases_425233)
        # Assigning a type to the variable 'M' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'M', for_loop_var_425234)
        # SSA begins for a for statement (line 219)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 220):
        
        # Assigning a Call to a Name (line 220):
        
        # Call to aslinearoperator(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'M' (line 220)
        M_425237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 43), 'M', False)
        # Processing the call keyword arguments (line 220)
        kwargs_425238 = {}
        # Getting the type of 'interface' (line 220)
        interface_425235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'interface', False)
        # Obtaining the member 'aslinearoperator' of a type (line 220)
        aslinearoperator_425236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 16), interface_425235, 'aslinearoperator')
        # Calling aslinearoperator(args, kwargs) (line 220)
        aslinearoperator_call_result_425239 = invoke(stypy.reporting.localization.Localization(__file__, 220, 16), aslinearoperator_425236, *[M_425237], **kwargs_425238)
        
        # Assigning a type to the variable 'A' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'A', aslinearoperator_call_result_425239)
        
        # Assigning a Attribute to a Tuple (line 221):
        
        # Assigning a Subscript to a Name (line 221):
        
        # Obtaining the type of the subscript
        int_425240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 12), 'int')
        # Getting the type of 'A' (line 221)
        A_425241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 18), 'A')
        # Obtaining the member 'shape' of a type (line 221)
        shape_425242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 18), A_425241, 'shape')
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___425243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 12), shape_425242, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 221)
        subscript_call_result_425244 = invoke(stypy.reporting.localization.Localization(__file__, 221, 12), getitem___425243, int_425240)
        
        # Assigning a type to the variable 'tuple_var_assignment_423559' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'tuple_var_assignment_423559', subscript_call_result_425244)
        
        # Assigning a Subscript to a Name (line 221):
        
        # Obtaining the type of the subscript
        int_425245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 12), 'int')
        # Getting the type of 'A' (line 221)
        A_425246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 18), 'A')
        # Obtaining the member 'shape' of a type (line 221)
        shape_425247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 18), A_425246, 'shape')
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___425248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 12), shape_425247, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 221)
        subscript_call_result_425249 = invoke(stypy.reporting.localization.Localization(__file__, 221, 12), getitem___425248, int_425245)
        
        # Assigning a type to the variable 'tuple_var_assignment_423560' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'tuple_var_assignment_423560', subscript_call_result_425249)
        
        # Assigning a Name to a Name (line 221):
        # Getting the type of 'tuple_var_assignment_423559' (line 221)
        tuple_var_assignment_423559_425250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'tuple_var_assignment_423559')
        # Assigning a type to the variable 'M' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'M', tuple_var_assignment_423559_425250)
        
        # Assigning a Name to a Name (line 221):
        # Getting the type of 'tuple_var_assignment_423560' (line 221)
        tuple_var_assignment_423560_425251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'tuple_var_assignment_423560')
        # Assigning a type to the variable 'N' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 14), 'N', tuple_var_assignment_423560_425251)
        
        # Call to assert_equal(...): (line 223)
        # Processing the call arguments (line 223)
        
        # Call to matvec(...): (line 223)
        # Processing the call arguments (line 223)
        
        # Call to array(...): (line 223)
        # Processing the call arguments (line 223)
        
        # Obtaining an instance of the builtin type 'list' (line 223)
        list_425257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 223)
        # Adding element type (line 223)
        int_425258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 43), list_425257, int_425258)
        # Adding element type (line 223)
        int_425259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 43), list_425257, int_425259)
        # Adding element type (line 223)
        int_425260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 43), list_425257, int_425260)
        
        # Processing the call keyword arguments (line 223)
        kwargs_425261 = {}
        # Getting the type of 'np' (line 223)
        np_425255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 34), 'np', False)
        # Obtaining the member 'array' of a type (line 223)
        array_425256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 34), np_425255, 'array')
        # Calling array(args, kwargs) (line 223)
        array_call_result_425262 = invoke(stypy.reporting.localization.Localization(__file__, 223, 34), array_425256, *[list_425257], **kwargs_425261)
        
        # Processing the call keyword arguments (line 223)
        kwargs_425263 = {}
        # Getting the type of 'A' (line 223)
        A_425253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 25), 'A', False)
        # Obtaining the member 'matvec' of a type (line 223)
        matvec_425254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 25), A_425253, 'matvec')
        # Calling matvec(args, kwargs) (line 223)
        matvec_call_result_425264 = invoke(stypy.reporting.localization.Localization(__file__, 223, 25), matvec_425254, *[array_call_result_425262], **kwargs_425263)
        
        
        # Obtaining an instance of the builtin type 'list' (line 223)
        list_425265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 54), 'list')
        # Adding type elements to the builtin type 'list' instance (line 223)
        # Adding element type (line 223)
        int_425266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 54), list_425265, int_425266)
        # Adding element type (line 223)
        int_425267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 54), list_425265, int_425267)
        
        # Processing the call keyword arguments (line 223)
        kwargs_425268 = {}
        # Getting the type of 'assert_equal' (line 223)
        assert_equal_425252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 223)
        assert_equal_call_result_425269 = invoke(stypy.reporting.localization.Localization(__file__, 223, 12), assert_equal_425252, *[matvec_call_result_425264, list_425265], **kwargs_425268)
        
        
        # Call to assert_equal(...): (line 224)
        # Processing the call arguments (line 224)
        
        # Call to matvec(...): (line 224)
        # Processing the call arguments (line 224)
        
        # Call to array(...): (line 224)
        # Processing the call arguments (line 224)
        
        # Obtaining an instance of the builtin type 'list' (line 224)
        list_425275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 224)
        # Adding element type (line 224)
        
        # Obtaining an instance of the builtin type 'list' (line 224)
        list_425276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 224)
        # Adding element type (line 224)
        int_425277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 44), list_425276, int_425277)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 43), list_425275, list_425276)
        # Adding element type (line 224)
        
        # Obtaining an instance of the builtin type 'list' (line 224)
        list_425278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 224)
        # Adding element type (line 224)
        int_425279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 48), list_425278, int_425279)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 43), list_425275, list_425278)
        # Adding element type (line 224)
        
        # Obtaining an instance of the builtin type 'list' (line 224)
        list_425280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 224)
        # Adding element type (line 224)
        int_425281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 52), list_425280, int_425281)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 43), list_425275, list_425280)
        
        # Processing the call keyword arguments (line 224)
        kwargs_425282 = {}
        # Getting the type of 'np' (line 224)
        np_425273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 34), 'np', False)
        # Obtaining the member 'array' of a type (line 224)
        array_425274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 34), np_425273, 'array')
        # Calling array(args, kwargs) (line 224)
        array_call_result_425283 = invoke(stypy.reporting.localization.Localization(__file__, 224, 34), array_425274, *[list_425275], **kwargs_425282)
        
        # Processing the call keyword arguments (line 224)
        kwargs_425284 = {}
        # Getting the type of 'A' (line 224)
        A_425271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 25), 'A', False)
        # Obtaining the member 'matvec' of a type (line 224)
        matvec_425272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 25), A_425271, 'matvec')
        # Calling matvec(args, kwargs) (line 224)
        matvec_call_result_425285 = invoke(stypy.reporting.localization.Localization(__file__, 224, 25), matvec_425272, *[array_call_result_425283], **kwargs_425284)
        
        
        # Obtaining an instance of the builtin type 'list' (line 224)
        list_425286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 60), 'list')
        # Adding type elements to the builtin type 'list' instance (line 224)
        # Adding element type (line 224)
        
        # Obtaining an instance of the builtin type 'list' (line 224)
        list_425287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 61), 'list')
        # Adding type elements to the builtin type 'list' instance (line 224)
        # Adding element type (line 224)
        int_425288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 61), list_425287, int_425288)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 60), list_425286, list_425287)
        # Adding element type (line 224)
        
        # Obtaining an instance of the builtin type 'list' (line 224)
        list_425289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 66), 'list')
        # Adding type elements to the builtin type 'list' instance (line 224)
        # Adding element type (line 224)
        int_425290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 67), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 66), list_425289, int_425290)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 60), list_425286, list_425289)
        
        # Processing the call keyword arguments (line 224)
        kwargs_425291 = {}
        # Getting the type of 'assert_equal' (line 224)
        assert_equal_425270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 224)
        assert_equal_call_result_425292 = invoke(stypy.reporting.localization.Localization(__file__, 224, 12), assert_equal_425270, *[matvec_call_result_425285, list_425286], **kwargs_425291)
        
        
        # Call to assert_equal(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'A' (line 226)
        A_425294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 25), 'A', False)
        
        # Call to array(...): (line 226)
        # Processing the call arguments (line 226)
        
        # Obtaining an instance of the builtin type 'list' (line 226)
        list_425297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 226)
        # Adding element type (line 226)
        int_425298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 38), list_425297, int_425298)
        # Adding element type (line 226)
        int_425299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 38), list_425297, int_425299)
        # Adding element type (line 226)
        int_425300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 38), list_425297, int_425300)
        
        # Processing the call keyword arguments (line 226)
        kwargs_425301 = {}
        # Getting the type of 'np' (line 226)
        np_425295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 29), 'np', False)
        # Obtaining the member 'array' of a type (line 226)
        array_425296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 29), np_425295, 'array')
        # Calling array(args, kwargs) (line 226)
        array_call_result_425302 = invoke(stypy.reporting.localization.Localization(__file__, 226, 29), array_425296, *[list_425297], **kwargs_425301)
        
        # Applying the binary operator '*' (line 226)
        result_mul_425303 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 25), '*', A_425294, array_call_result_425302)
        
        
        # Obtaining an instance of the builtin type 'list' (line 226)
        list_425304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 226)
        # Adding element type (line 226)
        int_425305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 48), list_425304, int_425305)
        # Adding element type (line 226)
        int_425306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 48), list_425304, int_425306)
        
        # Processing the call keyword arguments (line 226)
        kwargs_425307 = {}
        # Getting the type of 'assert_equal' (line 226)
        assert_equal_425293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 226)
        assert_equal_call_result_425308 = invoke(stypy.reporting.localization.Localization(__file__, 226, 12), assert_equal_425293, *[result_mul_425303, list_425304], **kwargs_425307)
        
        
        # Call to assert_equal(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'A' (line 227)
        A_425310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 25), 'A', False)
        
        # Call to array(...): (line 227)
        # Processing the call arguments (line 227)
        
        # Obtaining an instance of the builtin type 'list' (line 227)
        list_425313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 227)
        # Adding element type (line 227)
        
        # Obtaining an instance of the builtin type 'list' (line 227)
        list_425314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 227)
        # Adding element type (line 227)
        int_425315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 39), list_425314, int_425315)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 38), list_425313, list_425314)
        # Adding element type (line 227)
        
        # Obtaining an instance of the builtin type 'list' (line 227)
        list_425316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 227)
        # Adding element type (line 227)
        int_425317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 43), list_425316, int_425317)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 38), list_425313, list_425316)
        # Adding element type (line 227)
        
        # Obtaining an instance of the builtin type 'list' (line 227)
        list_425318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 227)
        # Adding element type (line 227)
        int_425319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 47), list_425318, int_425319)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 38), list_425313, list_425318)
        
        # Processing the call keyword arguments (line 227)
        kwargs_425320 = {}
        # Getting the type of 'np' (line 227)
        np_425311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 29), 'np', False)
        # Obtaining the member 'array' of a type (line 227)
        array_425312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 29), np_425311, 'array')
        # Calling array(args, kwargs) (line 227)
        array_call_result_425321 = invoke(stypy.reporting.localization.Localization(__file__, 227, 29), array_425312, *[list_425313], **kwargs_425320)
        
        # Applying the binary operator '*' (line 227)
        result_mul_425322 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 25), '*', A_425310, array_call_result_425321)
        
        
        # Obtaining an instance of the builtin type 'list' (line 227)
        list_425323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 54), 'list')
        # Adding type elements to the builtin type 'list' instance (line 227)
        # Adding element type (line 227)
        
        # Obtaining an instance of the builtin type 'list' (line 227)
        list_425324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 227)
        # Adding element type (line 227)
        int_425325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 55), list_425324, int_425325)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 54), list_425323, list_425324)
        # Adding element type (line 227)
        
        # Obtaining an instance of the builtin type 'list' (line 227)
        list_425326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 60), 'list')
        # Adding type elements to the builtin type 'list' instance (line 227)
        # Adding element type (line 227)
        int_425327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 60), list_425326, int_425327)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 54), list_425323, list_425326)
        
        # Processing the call keyword arguments (line 227)
        kwargs_425328 = {}
        # Getting the type of 'assert_equal' (line 227)
        assert_equal_425309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 227)
        assert_equal_call_result_425329 = invoke(stypy.reporting.localization.Localization(__file__, 227, 12), assert_equal_425309, *[result_mul_425322, list_425323], **kwargs_425328)
        
        
        # Call to assert_equal(...): (line 229)
        # Processing the call arguments (line 229)
        
        # Call to rmatvec(...): (line 229)
        # Processing the call arguments (line 229)
        
        # Call to array(...): (line 229)
        # Processing the call arguments (line 229)
        
        # Obtaining an instance of the builtin type 'list' (line 229)
        list_425335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 229)
        # Adding element type (line 229)
        int_425336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 44), list_425335, int_425336)
        # Adding element type (line 229)
        int_425337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 44), list_425335, int_425337)
        
        # Processing the call keyword arguments (line 229)
        kwargs_425338 = {}
        # Getting the type of 'np' (line 229)
        np_425333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 35), 'np', False)
        # Obtaining the member 'array' of a type (line 229)
        array_425334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 35), np_425333, 'array')
        # Calling array(args, kwargs) (line 229)
        array_call_result_425339 = invoke(stypy.reporting.localization.Localization(__file__, 229, 35), array_425334, *[list_425335], **kwargs_425338)
        
        # Processing the call keyword arguments (line 229)
        kwargs_425340 = {}
        # Getting the type of 'A' (line 229)
        A_425331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 25), 'A', False)
        # Obtaining the member 'rmatvec' of a type (line 229)
        rmatvec_425332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 25), A_425331, 'rmatvec')
        # Calling rmatvec(args, kwargs) (line 229)
        rmatvec_call_result_425341 = invoke(stypy.reporting.localization.Localization(__file__, 229, 25), rmatvec_425332, *[array_call_result_425339], **kwargs_425340)
        
        
        # Obtaining an instance of the builtin type 'list' (line 229)
        list_425342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 229)
        # Adding element type (line 229)
        int_425343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 53), list_425342, int_425343)
        # Adding element type (line 229)
        int_425344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 53), list_425342, int_425344)
        # Adding element type (line 229)
        int_425345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 53), list_425342, int_425345)
        
        # Processing the call keyword arguments (line 229)
        kwargs_425346 = {}
        # Getting the type of 'assert_equal' (line 229)
        assert_equal_425330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 229)
        assert_equal_call_result_425347 = invoke(stypy.reporting.localization.Localization(__file__, 229, 12), assert_equal_425330, *[rmatvec_call_result_425341, list_425342], **kwargs_425346)
        
        
        # Call to assert_equal(...): (line 230)
        # Processing the call arguments (line 230)
        
        # Call to rmatvec(...): (line 230)
        # Processing the call arguments (line 230)
        
        # Call to array(...): (line 230)
        # Processing the call arguments (line 230)
        
        # Obtaining an instance of the builtin type 'list' (line 230)
        list_425353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 230)
        # Adding element type (line 230)
        
        # Obtaining an instance of the builtin type 'list' (line 230)
        list_425354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 230)
        # Adding element type (line 230)
        int_425355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 45), list_425354, int_425355)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 44), list_425353, list_425354)
        # Adding element type (line 230)
        
        # Obtaining an instance of the builtin type 'list' (line 230)
        list_425356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 230)
        # Adding element type (line 230)
        int_425357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 49), list_425356, int_425357)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 44), list_425353, list_425356)
        
        # Processing the call keyword arguments (line 230)
        kwargs_425358 = {}
        # Getting the type of 'np' (line 230)
        np_425351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 35), 'np', False)
        # Obtaining the member 'array' of a type (line 230)
        array_425352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 35), np_425351, 'array')
        # Calling array(args, kwargs) (line 230)
        array_call_result_425359 = invoke(stypy.reporting.localization.Localization(__file__, 230, 35), array_425352, *[list_425353], **kwargs_425358)
        
        # Processing the call keyword arguments (line 230)
        kwargs_425360 = {}
        # Getting the type of 'A' (line 230)
        A_425349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 25), 'A', False)
        # Obtaining the member 'rmatvec' of a type (line 230)
        rmatvec_425350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 25), A_425349, 'rmatvec')
        # Calling rmatvec(args, kwargs) (line 230)
        rmatvec_call_result_425361 = invoke(stypy.reporting.localization.Localization(__file__, 230, 25), rmatvec_425350, *[array_call_result_425359], **kwargs_425360)
        
        
        # Obtaining an instance of the builtin type 'list' (line 230)
        list_425362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 230)
        # Adding element type (line 230)
        
        # Obtaining an instance of the builtin type 'list' (line 230)
        list_425363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 230)
        # Adding element type (line 230)
        int_425364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 58), list_425363, int_425364)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 57), list_425362, list_425363)
        # Adding element type (line 230)
        
        # Obtaining an instance of the builtin type 'list' (line 230)
        list_425365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 62), 'list')
        # Adding type elements to the builtin type 'list' instance (line 230)
        # Adding element type (line 230)
        int_425366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 62), list_425365, int_425366)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 57), list_425362, list_425365)
        # Adding element type (line 230)
        
        # Obtaining an instance of the builtin type 'list' (line 230)
        list_425367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 67), 'list')
        # Adding type elements to the builtin type 'list' instance (line 230)
        # Adding element type (line 230)
        int_425368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 67), list_425367, int_425368)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 57), list_425362, list_425367)
        
        # Processing the call keyword arguments (line 230)
        kwargs_425369 = {}
        # Getting the type of 'assert_equal' (line 230)
        assert_equal_425348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 230)
        assert_equal_call_result_425370 = invoke(stypy.reporting.localization.Localization(__file__, 230, 12), assert_equal_425348, *[rmatvec_call_result_425361, list_425362], **kwargs_425369)
        
        
        # Call to assert_equal(...): (line 231)
        # Processing the call arguments (line 231)
        
        # Call to matvec(...): (line 231)
        # Processing the call arguments (line 231)
        
        # Call to array(...): (line 231)
        # Processing the call arguments (line 231)
        
        # Obtaining an instance of the builtin type 'list' (line 231)
        list_425377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 231)
        # Adding element type (line 231)
        int_425378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 45), list_425377, int_425378)
        # Adding element type (line 231)
        int_425379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 45), list_425377, int_425379)
        
        # Processing the call keyword arguments (line 231)
        kwargs_425380 = {}
        # Getting the type of 'np' (line 231)
        np_425375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 36), 'np', False)
        # Obtaining the member 'array' of a type (line 231)
        array_425376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 36), np_425375, 'array')
        # Calling array(args, kwargs) (line 231)
        array_call_result_425381 = invoke(stypy.reporting.localization.Localization(__file__, 231, 36), array_425376, *[list_425377], **kwargs_425380)
        
        # Processing the call keyword arguments (line 231)
        kwargs_425382 = {}
        # Getting the type of 'A' (line 231)
        A_425372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 25), 'A', False)
        # Obtaining the member 'H' of a type (line 231)
        H_425373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 25), A_425372, 'H')
        # Obtaining the member 'matvec' of a type (line 231)
        matvec_425374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 25), H_425373, 'matvec')
        # Calling matvec(args, kwargs) (line 231)
        matvec_call_result_425383 = invoke(stypy.reporting.localization.Localization(__file__, 231, 25), matvec_425374, *[array_call_result_425381], **kwargs_425382)
        
        
        # Obtaining an instance of the builtin type 'list' (line 231)
        list_425384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 54), 'list')
        # Adding type elements to the builtin type 'list' instance (line 231)
        # Adding element type (line 231)
        int_425385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 54), list_425384, int_425385)
        # Adding element type (line 231)
        int_425386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 54), list_425384, int_425386)
        # Adding element type (line 231)
        int_425387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 54), list_425384, int_425387)
        
        # Processing the call keyword arguments (line 231)
        kwargs_425388 = {}
        # Getting the type of 'assert_equal' (line 231)
        assert_equal_425371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 231)
        assert_equal_call_result_425389 = invoke(stypy.reporting.localization.Localization(__file__, 231, 12), assert_equal_425371, *[matvec_call_result_425383, list_425384], **kwargs_425388)
        
        
        # Call to assert_equal(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Call to matvec(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Call to array(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Obtaining an instance of the builtin type 'list' (line 232)
        list_425396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 232)
        # Adding element type (line 232)
        
        # Obtaining an instance of the builtin type 'list' (line 232)
        list_425397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 232)
        # Adding element type (line 232)
        int_425398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 46), list_425397, int_425398)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 45), list_425396, list_425397)
        # Adding element type (line 232)
        
        # Obtaining an instance of the builtin type 'list' (line 232)
        list_425399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 232)
        # Adding element type (line 232)
        int_425400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 50), list_425399, int_425400)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 45), list_425396, list_425399)
        
        # Processing the call keyword arguments (line 232)
        kwargs_425401 = {}
        # Getting the type of 'np' (line 232)
        np_425394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 36), 'np', False)
        # Obtaining the member 'array' of a type (line 232)
        array_425395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 36), np_425394, 'array')
        # Calling array(args, kwargs) (line 232)
        array_call_result_425402 = invoke(stypy.reporting.localization.Localization(__file__, 232, 36), array_425395, *[list_425396], **kwargs_425401)
        
        # Processing the call keyword arguments (line 232)
        kwargs_425403 = {}
        # Getting the type of 'A' (line 232)
        A_425391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 25), 'A', False)
        # Obtaining the member 'H' of a type (line 232)
        H_425392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 25), A_425391, 'H')
        # Obtaining the member 'matvec' of a type (line 232)
        matvec_425393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 25), H_425392, 'matvec')
        # Calling matvec(args, kwargs) (line 232)
        matvec_call_result_425404 = invoke(stypy.reporting.localization.Localization(__file__, 232, 25), matvec_425393, *[array_call_result_425402], **kwargs_425403)
        
        
        # Obtaining an instance of the builtin type 'list' (line 232)
        list_425405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 232)
        # Adding element type (line 232)
        
        # Obtaining an instance of the builtin type 'list' (line 232)
        list_425406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 59), 'list')
        # Adding type elements to the builtin type 'list' instance (line 232)
        # Adding element type (line 232)
        int_425407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 59), list_425406, int_425407)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 58), list_425405, list_425406)
        # Adding element type (line 232)
        
        # Obtaining an instance of the builtin type 'list' (line 232)
        list_425408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 63), 'list')
        # Adding type elements to the builtin type 'list' instance (line 232)
        # Adding element type (line 232)
        int_425409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 64), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 63), list_425408, int_425409)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 58), list_425405, list_425408)
        # Adding element type (line 232)
        
        # Obtaining an instance of the builtin type 'list' (line 232)
        list_425410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 68), 'list')
        # Adding type elements to the builtin type 'list' instance (line 232)
        # Adding element type (line 232)
        int_425411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 69), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 68), list_425410, int_425411)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 58), list_425405, list_425410)
        
        # Processing the call keyword arguments (line 232)
        kwargs_425412 = {}
        # Getting the type of 'assert_equal' (line 232)
        assert_equal_425390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 232)
        assert_equal_call_result_425413 = invoke(stypy.reporting.localization.Localization(__file__, 232, 12), assert_equal_425390, *[matvec_call_result_425404, list_425405], **kwargs_425412)
        
        
        # Call to assert_equal(...): (line 234)
        # Processing the call arguments (line 234)
        
        # Call to matmat(...): (line 235)
        # Processing the call arguments (line 235)
        
        # Call to array(...): (line 235)
        # Processing the call arguments (line 235)
        
        # Obtaining an instance of the builtin type 'list' (line 235)
        list_425419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 235)
        # Adding element type (line 235)
        
        # Obtaining an instance of the builtin type 'list' (line 235)
        list_425420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 235)
        # Adding element type (line 235)
        int_425421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 39), list_425420, int_425421)
        # Adding element type (line 235)
        int_425422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 39), list_425420, int_425422)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 38), list_425419, list_425420)
        # Adding element type (line 235)
        
        # Obtaining an instance of the builtin type 'list' (line 235)
        list_425423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 235)
        # Adding element type (line 235)
        int_425424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 45), list_425423, int_425424)
        # Adding element type (line 235)
        int_425425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 45), list_425423, int_425425)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 38), list_425419, list_425423)
        # Adding element type (line 235)
        
        # Obtaining an instance of the builtin type 'list' (line 235)
        list_425426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 235)
        # Adding element type (line 235)
        int_425427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 51), list_425426, int_425427)
        # Adding element type (line 235)
        int_425428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 51), list_425426, int_425428)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 38), list_425419, list_425426)
        
        # Processing the call keyword arguments (line 235)
        kwargs_425429 = {}
        # Getting the type of 'np' (line 235)
        np_425417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 29), 'np', False)
        # Obtaining the member 'array' of a type (line 235)
        array_425418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 29), np_425417, 'array')
        # Calling array(args, kwargs) (line 235)
        array_call_result_425430 = invoke(stypy.reporting.localization.Localization(__file__, 235, 29), array_425418, *[list_425419], **kwargs_425429)
        
        # Processing the call keyword arguments (line 235)
        kwargs_425431 = {}
        # Getting the type of 'A' (line 235)
        A_425415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'A', False)
        # Obtaining the member 'matmat' of a type (line 235)
        matmat_425416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 20), A_425415, 'matmat')
        # Calling matmat(args, kwargs) (line 235)
        matmat_call_result_425432 = invoke(stypy.reporting.localization.Localization(__file__, 235, 20), matmat_425416, *[array_call_result_425430], **kwargs_425431)
        
        
        # Obtaining an instance of the builtin type 'list' (line 236)
        list_425433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 236)
        # Adding element type (line 236)
        
        # Obtaining an instance of the builtin type 'list' (line 236)
        list_425434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 236)
        # Adding element type (line 236)
        int_425435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 21), list_425434, int_425435)
        # Adding element type (line 236)
        int_425436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 21), list_425434, int_425436)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 20), list_425433, list_425434)
        # Adding element type (line 236)
        
        # Obtaining an instance of the builtin type 'list' (line 236)
        list_425437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 236)
        # Adding element type (line 236)
        int_425438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 29), list_425437, int_425438)
        # Adding element type (line 236)
        int_425439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 29), list_425437, int_425439)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 20), list_425433, list_425437)
        
        # Processing the call keyword arguments (line 234)
        kwargs_425440 = {}
        # Getting the type of 'assert_equal' (line 234)
        assert_equal_425414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 234)
        assert_equal_call_result_425441 = invoke(stypy.reporting.localization.Localization(__file__, 234, 12), assert_equal_425414, *[matmat_call_result_425432, list_425433], **kwargs_425440)
        
        
        # Call to assert_equal(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'A' (line 238)
        A_425443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 25), 'A', False)
        
        # Call to array(...): (line 238)
        # Processing the call arguments (line 238)
        
        # Obtaining an instance of the builtin type 'list' (line 238)
        list_425446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 238)
        # Adding element type (line 238)
        
        # Obtaining an instance of the builtin type 'list' (line 238)
        list_425447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 238)
        # Adding element type (line 238)
        int_425448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 39), list_425447, int_425448)
        # Adding element type (line 238)
        int_425449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 39), list_425447, int_425449)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 38), list_425446, list_425447)
        # Adding element type (line 238)
        
        # Obtaining an instance of the builtin type 'list' (line 238)
        list_425450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 238)
        # Adding element type (line 238)
        int_425451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 45), list_425450, int_425451)
        # Adding element type (line 238)
        int_425452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 45), list_425450, int_425452)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 38), list_425446, list_425450)
        # Adding element type (line 238)
        
        # Obtaining an instance of the builtin type 'list' (line 238)
        list_425453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 238)
        # Adding element type (line 238)
        int_425454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 51), list_425453, int_425454)
        # Adding element type (line 238)
        int_425455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 51), list_425453, int_425455)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 38), list_425446, list_425453)
        
        # Processing the call keyword arguments (line 238)
        kwargs_425456 = {}
        # Getting the type of 'np' (line 238)
        np_425444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 29), 'np', False)
        # Obtaining the member 'array' of a type (line 238)
        array_425445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 29), np_425444, 'array')
        # Calling array(args, kwargs) (line 238)
        array_call_result_425457 = invoke(stypy.reporting.localization.Localization(__file__, 238, 29), array_425445, *[list_425446], **kwargs_425456)
        
        # Applying the binary operator '*' (line 238)
        result_mul_425458 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 25), '*', A_425443, array_call_result_425457)
        
        
        # Obtaining an instance of the builtin type 'list' (line 238)
        list_425459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 60), 'list')
        # Adding type elements to the builtin type 'list' instance (line 238)
        # Adding element type (line 238)
        
        # Obtaining an instance of the builtin type 'list' (line 238)
        list_425460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 61), 'list')
        # Adding type elements to the builtin type 'list' instance (line 238)
        # Adding element type (line 238)
        int_425461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 61), list_425460, int_425461)
        # Adding element type (line 238)
        int_425462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 61), list_425460, int_425462)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 60), list_425459, list_425460)
        # Adding element type (line 238)
        
        # Obtaining an instance of the builtin type 'list' (line 238)
        list_425463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 69), 'list')
        # Adding type elements to the builtin type 'list' instance (line 238)
        # Adding element type (line 238)
        int_425464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 70), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 69), list_425463, int_425464)
        # Adding element type (line 238)
        int_425465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 73), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 69), list_425463, int_425465)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 60), list_425459, list_425463)
        
        # Processing the call keyword arguments (line 238)
        kwargs_425466 = {}
        # Getting the type of 'assert_equal' (line 238)
        assert_equal_425442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 238)
        assert_equal_call_result_425467 = invoke(stypy.reporting.localization.Localization(__file__, 238, 12), assert_equal_425442, *[result_mul_425458, list_425459], **kwargs_425466)
        
        
        # Type idiom detected: calculating its left and rigth part (line 240)
        str_425468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 25), 'str', 'dtype')
        # Getting the type of 'M' (line 240)
        M_425469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 23), 'M')
        
        (may_be_425470, more_types_in_union_425471) = may_provide_member(str_425468, M_425469)

        if may_be_425470:

            if more_types_in_union_425471:
                # Runtime conditional SSA (line 240)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'M' (line 240)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'M', remove_not_member_provider_from_union(M_425469, 'dtype'))
            
            # Call to assert_equal(...): (line 241)
            # Processing the call arguments (line 241)
            # Getting the type of 'A' (line 241)
            A_425473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 29), 'A', False)
            # Obtaining the member 'dtype' of a type (line 241)
            dtype_425474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 29), A_425473, 'dtype')
            # Getting the type of 'M' (line 241)
            M_425475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 38), 'M', False)
            # Obtaining the member 'dtype' of a type (line 241)
            dtype_425476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 38), M_425475, 'dtype')
            # Processing the call keyword arguments (line 241)
            kwargs_425477 = {}
            # Getting the type of 'assert_equal' (line 241)
            assert_equal_425472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 16), 'assert_equal', False)
            # Calling assert_equal(args, kwargs) (line 241)
            assert_equal_call_result_425478 = invoke(stypy.reporting.localization.Localization(__file__, 241, 16), assert_equal_425472, *[dtype_425474, dtype_425476], **kwargs_425477)
            

            if more_types_in_union_425471:
                # SSA join for if statement (line 240)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 217)
        stypy_return_type_425479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_425479)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_425479


    @norecursion
    def test_dot(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dot'
        module_type_store = module_type_store.open_function_context('test_dot', 243, 4, False)
        # Assigning a type to the variable 'self' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestAsLinearOperator.test_dot.__dict__.__setitem__('stypy_localization', localization)
        TestAsLinearOperator.test_dot.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestAsLinearOperator.test_dot.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestAsLinearOperator.test_dot.__dict__.__setitem__('stypy_function_name', 'TestAsLinearOperator.test_dot')
        TestAsLinearOperator.test_dot.__dict__.__setitem__('stypy_param_names_list', [])
        TestAsLinearOperator.test_dot.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestAsLinearOperator.test_dot.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestAsLinearOperator.test_dot.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestAsLinearOperator.test_dot.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestAsLinearOperator.test_dot.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestAsLinearOperator.test_dot.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestAsLinearOperator.test_dot', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dot', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dot(...)' code ##################

        
        # Getting the type of 'self' (line 245)
        self_425480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 17), 'self')
        # Obtaining the member 'cases' of a type (line 245)
        cases_425481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 17), self_425480, 'cases')
        # Testing the type of a for loop iterable (line 245)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 245, 8), cases_425481)
        # Getting the type of the for loop variable (line 245)
        for_loop_var_425482 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 245, 8), cases_425481)
        # Assigning a type to the variable 'M' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'M', for_loop_var_425482)
        # SSA begins for a for statement (line 245)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 246):
        
        # Assigning a Call to a Name (line 246):
        
        # Call to aslinearoperator(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'M' (line 246)
        M_425485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 43), 'M', False)
        # Processing the call keyword arguments (line 246)
        kwargs_425486 = {}
        # Getting the type of 'interface' (line 246)
        interface_425483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'interface', False)
        # Obtaining the member 'aslinearoperator' of a type (line 246)
        aslinearoperator_425484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 16), interface_425483, 'aslinearoperator')
        # Calling aslinearoperator(args, kwargs) (line 246)
        aslinearoperator_call_result_425487 = invoke(stypy.reporting.localization.Localization(__file__, 246, 16), aslinearoperator_425484, *[M_425485], **kwargs_425486)
        
        # Assigning a type to the variable 'A' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'A', aslinearoperator_call_result_425487)
        
        # Assigning a Attribute to a Tuple (line 247):
        
        # Assigning a Subscript to a Name (line 247):
        
        # Obtaining the type of the subscript
        int_425488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 12), 'int')
        # Getting the type of 'A' (line 247)
        A_425489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 18), 'A')
        # Obtaining the member 'shape' of a type (line 247)
        shape_425490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 18), A_425489, 'shape')
        # Obtaining the member '__getitem__' of a type (line 247)
        getitem___425491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 12), shape_425490, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 247)
        subscript_call_result_425492 = invoke(stypy.reporting.localization.Localization(__file__, 247, 12), getitem___425491, int_425488)
        
        # Assigning a type to the variable 'tuple_var_assignment_423561' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'tuple_var_assignment_423561', subscript_call_result_425492)
        
        # Assigning a Subscript to a Name (line 247):
        
        # Obtaining the type of the subscript
        int_425493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 12), 'int')
        # Getting the type of 'A' (line 247)
        A_425494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 18), 'A')
        # Obtaining the member 'shape' of a type (line 247)
        shape_425495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 18), A_425494, 'shape')
        # Obtaining the member '__getitem__' of a type (line 247)
        getitem___425496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 12), shape_425495, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 247)
        subscript_call_result_425497 = invoke(stypy.reporting.localization.Localization(__file__, 247, 12), getitem___425496, int_425493)
        
        # Assigning a type to the variable 'tuple_var_assignment_423562' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'tuple_var_assignment_423562', subscript_call_result_425497)
        
        # Assigning a Name to a Name (line 247):
        # Getting the type of 'tuple_var_assignment_423561' (line 247)
        tuple_var_assignment_423561_425498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'tuple_var_assignment_423561')
        # Assigning a type to the variable 'M' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'M', tuple_var_assignment_423561_425498)
        
        # Assigning a Name to a Name (line 247):
        # Getting the type of 'tuple_var_assignment_423562' (line 247)
        tuple_var_assignment_423562_425499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'tuple_var_assignment_423562')
        # Assigning a type to the variable 'N' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 14), 'N', tuple_var_assignment_423562_425499)
        
        # Call to assert_equal(...): (line 249)
        # Processing the call arguments (line 249)
        
        # Call to dot(...): (line 249)
        # Processing the call arguments (line 249)
        
        # Call to array(...): (line 249)
        # Processing the call arguments (line 249)
        
        # Obtaining an instance of the builtin type 'list' (line 249)
        list_425505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 249)
        # Adding element type (line 249)
        int_425506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 40), list_425505, int_425506)
        # Adding element type (line 249)
        int_425507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 40), list_425505, int_425507)
        # Adding element type (line 249)
        int_425508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 40), list_425505, int_425508)
        
        # Processing the call keyword arguments (line 249)
        kwargs_425509 = {}
        # Getting the type of 'np' (line 249)
        np_425503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 31), 'np', False)
        # Obtaining the member 'array' of a type (line 249)
        array_425504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 31), np_425503, 'array')
        # Calling array(args, kwargs) (line 249)
        array_call_result_425510 = invoke(stypy.reporting.localization.Localization(__file__, 249, 31), array_425504, *[list_425505], **kwargs_425509)
        
        # Processing the call keyword arguments (line 249)
        kwargs_425511 = {}
        # Getting the type of 'A' (line 249)
        A_425501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 25), 'A', False)
        # Obtaining the member 'dot' of a type (line 249)
        dot_425502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 25), A_425501, 'dot')
        # Calling dot(args, kwargs) (line 249)
        dot_call_result_425512 = invoke(stypy.reporting.localization.Localization(__file__, 249, 25), dot_425502, *[array_call_result_425510], **kwargs_425511)
        
        
        # Obtaining an instance of the builtin type 'list' (line 249)
        list_425513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 249)
        # Adding element type (line 249)
        int_425514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 51), list_425513, int_425514)
        # Adding element type (line 249)
        int_425515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 51), list_425513, int_425515)
        
        # Processing the call keyword arguments (line 249)
        kwargs_425516 = {}
        # Getting the type of 'assert_equal' (line 249)
        assert_equal_425500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 249)
        assert_equal_call_result_425517 = invoke(stypy.reporting.localization.Localization(__file__, 249, 12), assert_equal_425500, *[dot_call_result_425512, list_425513], **kwargs_425516)
        
        
        # Call to assert_equal(...): (line 250)
        # Processing the call arguments (line 250)
        
        # Call to dot(...): (line 250)
        # Processing the call arguments (line 250)
        
        # Call to array(...): (line 250)
        # Processing the call arguments (line 250)
        
        # Obtaining an instance of the builtin type 'list' (line 250)
        list_425523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 250)
        # Adding element type (line 250)
        
        # Obtaining an instance of the builtin type 'list' (line 250)
        list_425524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 250)
        # Adding element type (line 250)
        int_425525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 41), list_425524, int_425525)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 40), list_425523, list_425524)
        # Adding element type (line 250)
        
        # Obtaining an instance of the builtin type 'list' (line 250)
        list_425526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 250)
        # Adding element type (line 250)
        int_425527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 45), list_425526, int_425527)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 40), list_425523, list_425526)
        # Adding element type (line 250)
        
        # Obtaining an instance of the builtin type 'list' (line 250)
        list_425528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 250)
        # Adding element type (line 250)
        int_425529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 49), list_425528, int_425529)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 40), list_425523, list_425528)
        
        # Processing the call keyword arguments (line 250)
        kwargs_425530 = {}
        # Getting the type of 'np' (line 250)
        np_425521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 31), 'np', False)
        # Obtaining the member 'array' of a type (line 250)
        array_425522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 31), np_425521, 'array')
        # Calling array(args, kwargs) (line 250)
        array_call_result_425531 = invoke(stypy.reporting.localization.Localization(__file__, 250, 31), array_425522, *[list_425523], **kwargs_425530)
        
        # Processing the call keyword arguments (line 250)
        kwargs_425532 = {}
        # Getting the type of 'A' (line 250)
        A_425519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 25), 'A', False)
        # Obtaining the member 'dot' of a type (line 250)
        dot_425520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 25), A_425519, 'dot')
        # Calling dot(args, kwargs) (line 250)
        dot_call_result_425533 = invoke(stypy.reporting.localization.Localization(__file__, 250, 25), dot_425520, *[array_call_result_425531], **kwargs_425532)
        
        
        # Obtaining an instance of the builtin type 'list' (line 250)
        list_425534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 250)
        # Adding element type (line 250)
        
        # Obtaining an instance of the builtin type 'list' (line 250)
        list_425535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 250)
        # Adding element type (line 250)
        int_425536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 58), list_425535, int_425536)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 57), list_425534, list_425535)
        # Adding element type (line 250)
        
        # Obtaining an instance of the builtin type 'list' (line 250)
        list_425537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 63), 'list')
        # Adding type elements to the builtin type 'list' instance (line 250)
        # Adding element type (line 250)
        int_425538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 64), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 63), list_425537, int_425538)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 57), list_425534, list_425537)
        
        # Processing the call keyword arguments (line 250)
        kwargs_425539 = {}
        # Getting the type of 'assert_equal' (line 250)
        assert_equal_425518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 250)
        assert_equal_call_result_425540 = invoke(stypy.reporting.localization.Localization(__file__, 250, 12), assert_equal_425518, *[dot_call_result_425533, list_425534], **kwargs_425539)
        
        
        # Call to assert_equal(...): (line 252)
        # Processing the call arguments (line 252)
        
        # Call to dot(...): (line 253)
        # Processing the call arguments (line 253)
        
        # Call to array(...): (line 253)
        # Processing the call arguments (line 253)
        
        # Obtaining an instance of the builtin type 'list' (line 253)
        list_425546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 253)
        # Adding element type (line 253)
        
        # Obtaining an instance of the builtin type 'list' (line 253)
        list_425547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 253)
        # Adding element type (line 253)
        int_425548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 36), list_425547, int_425548)
        # Adding element type (line 253)
        int_425549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 36), list_425547, int_425549)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 35), list_425546, list_425547)
        # Adding element type (line 253)
        
        # Obtaining an instance of the builtin type 'list' (line 253)
        list_425550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 253)
        # Adding element type (line 253)
        int_425551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 42), list_425550, int_425551)
        # Adding element type (line 253)
        int_425552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 42), list_425550, int_425552)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 35), list_425546, list_425550)
        # Adding element type (line 253)
        
        # Obtaining an instance of the builtin type 'list' (line 253)
        list_425553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 253)
        # Adding element type (line 253)
        int_425554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 48), list_425553, int_425554)
        # Adding element type (line 253)
        int_425555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 48), list_425553, int_425555)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 35), list_425546, list_425553)
        
        # Processing the call keyword arguments (line 253)
        kwargs_425556 = {}
        # Getting the type of 'np' (line 253)
        np_425544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 26), 'np', False)
        # Obtaining the member 'array' of a type (line 253)
        array_425545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 26), np_425544, 'array')
        # Calling array(args, kwargs) (line 253)
        array_call_result_425557 = invoke(stypy.reporting.localization.Localization(__file__, 253, 26), array_425545, *[list_425546], **kwargs_425556)
        
        # Processing the call keyword arguments (line 253)
        kwargs_425558 = {}
        # Getting the type of 'A' (line 253)
        A_425542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 20), 'A', False)
        # Obtaining the member 'dot' of a type (line 253)
        dot_425543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 20), A_425542, 'dot')
        # Calling dot(args, kwargs) (line 253)
        dot_call_result_425559 = invoke(stypy.reporting.localization.Localization(__file__, 253, 20), dot_425543, *[array_call_result_425557], **kwargs_425558)
        
        
        # Obtaining an instance of the builtin type 'list' (line 254)
        list_425560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 254)
        # Adding element type (line 254)
        
        # Obtaining an instance of the builtin type 'list' (line 254)
        list_425561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 254)
        # Adding element type (line 254)
        int_425562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 21), list_425561, int_425562)
        # Adding element type (line 254)
        int_425563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 21), list_425561, int_425563)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 20), list_425560, list_425561)
        # Adding element type (line 254)
        
        # Obtaining an instance of the builtin type 'list' (line 254)
        list_425564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 254)
        # Adding element type (line 254)
        int_425565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 29), list_425564, int_425565)
        # Adding element type (line 254)
        int_425566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 29), list_425564, int_425566)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 20), list_425560, list_425564)
        
        # Processing the call keyword arguments (line 252)
        kwargs_425567 = {}
        # Getting the type of 'assert_equal' (line 252)
        assert_equal_425541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 252)
        assert_equal_call_result_425568 = invoke(stypy.reporting.localization.Localization(__file__, 252, 12), assert_equal_425541, *[dot_call_result_425559, list_425560], **kwargs_425567)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_dot(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dot' in the type store
        # Getting the type of 'stypy_return_type' (line 243)
        stypy_return_type_425569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_425569)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dot'
        return stypy_return_type_425569


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 165, 0, False)
        # Assigning a type to the variable 'self' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestAsLinearOperator.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestAsLinearOperator' (line 165)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 0), 'TestAsLinearOperator', TestAsLinearOperator)

@norecursion
def test_repr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_repr'
    module_type_store = module_type_store.open_function_context('test_repr', 257, 0, False)
    
    # Passed parameters checking function
    test_repr.stypy_localization = localization
    test_repr.stypy_type_of_self = None
    test_repr.stypy_type_store = module_type_store
    test_repr.stypy_function_name = 'test_repr'
    test_repr.stypy_param_names_list = []
    test_repr.stypy_varargs_param_name = None
    test_repr.stypy_kwargs_param_name = None
    test_repr.stypy_call_defaults = defaults
    test_repr.stypy_call_varargs = varargs
    test_repr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_repr', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_repr', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_repr(...)' code ##################

    
    # Assigning a Call to a Name (line 258):
    
    # Assigning a Call to a Name (line 258):
    
    # Call to LinearOperator(...): (line 258)
    # Processing the call keyword arguments (line 258)
    
    # Obtaining an instance of the builtin type 'tuple' (line 258)
    tuple_425572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 258)
    # Adding element type (line 258)
    int_425573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 40), tuple_425572, int_425573)
    # Adding element type (line 258)
    int_425574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 40), tuple_425572, int_425574)
    
    keyword_425575 = tuple_425572

    @norecursion
    def _stypy_temp_lambda_238(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_238'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_238', 258, 54, True)
        # Passed parameters checking function
        _stypy_temp_lambda_238.stypy_localization = localization
        _stypy_temp_lambda_238.stypy_type_of_self = None
        _stypy_temp_lambda_238.stypy_type_store = module_type_store
        _stypy_temp_lambda_238.stypy_function_name = '_stypy_temp_lambda_238'
        _stypy_temp_lambda_238.stypy_param_names_list = ['x']
        _stypy_temp_lambda_238.stypy_varargs_param_name = None
        _stypy_temp_lambda_238.stypy_kwargs_param_name = None
        _stypy_temp_lambda_238.stypy_call_defaults = defaults
        _stypy_temp_lambda_238.stypy_call_varargs = varargs
        _stypy_temp_lambda_238.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_238', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_238', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        int_425576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 64), 'int')
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 54), 'stypy_return_type', int_425576)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_238' in the type store
        # Getting the type of 'stypy_return_type' (line 258)
        stypy_return_type_425577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 54), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_425577)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_238'
        return stypy_return_type_425577

    # Assigning a type to the variable '_stypy_temp_lambda_238' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 54), '_stypy_temp_lambda_238', _stypy_temp_lambda_238)
    # Getting the type of '_stypy_temp_lambda_238' (line 258)
    _stypy_temp_lambda_238_425578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 54), '_stypy_temp_lambda_238')
    keyword_425579 = _stypy_temp_lambda_238_425578
    kwargs_425580 = {'shape': keyword_425575, 'matvec': keyword_425579}
    # Getting the type of 'interface' (line 258)
    interface_425570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'interface', False)
    # Obtaining the member 'LinearOperator' of a type (line 258)
    LinearOperator_425571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 8), interface_425570, 'LinearOperator')
    # Calling LinearOperator(args, kwargs) (line 258)
    LinearOperator_call_result_425581 = invoke(stypy.reporting.localization.Localization(__file__, 258, 8), LinearOperator_425571, *[], **kwargs_425580)
    
    # Assigning a type to the variable 'A' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'A', LinearOperator_call_result_425581)
    
    # Assigning a Call to a Name (line 259):
    
    # Assigning a Call to a Name (line 259):
    
    # Call to repr(...): (line 259)
    # Processing the call arguments (line 259)
    # Getting the type of 'A' (line 259)
    A_425583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 18), 'A', False)
    # Processing the call keyword arguments (line 259)
    kwargs_425584 = {}
    # Getting the type of 'repr' (line 259)
    repr_425582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 13), 'repr', False)
    # Calling repr(args, kwargs) (line 259)
    repr_call_result_425585 = invoke(stypy.reporting.localization.Localization(__file__, 259, 13), repr_425582, *[A_425583], **kwargs_425584)
    
    # Assigning a type to the variable 'repr_A' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'repr_A', repr_call_result_425585)
    
    # Call to assert_(...): (line 260)
    # Processing the call arguments (line 260)
    
    str_425587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 12), 'str', 'unspecified dtype')
    # Getting the type of 'repr_A' (line 260)
    repr_A_425588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 39), 'repr_A', False)
    # Applying the binary operator 'notin' (line 260)
    result_contains_425589 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 12), 'notin', str_425587, repr_A_425588)
    
    # Getting the type of 'repr_A' (line 260)
    repr_A_425590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 47), 'repr_A', False)
    # Processing the call keyword arguments (line 260)
    kwargs_425591 = {}
    # Getting the type of 'assert_' (line 260)
    assert__425586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 260)
    assert__call_result_425592 = invoke(stypy.reporting.localization.Localization(__file__, 260, 4), assert__425586, *[result_contains_425589, repr_A_425590], **kwargs_425591)
    
    
    # ################# End of 'test_repr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_repr' in the type store
    # Getting the type of 'stypy_return_type' (line 257)
    stypy_return_type_425593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_425593)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_repr'
    return stypy_return_type_425593

# Assigning a type to the variable 'test_repr' (line 257)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 0), 'test_repr', test_repr)

@norecursion
def test_identity(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_identity'
    module_type_store = module_type_store.open_function_context('test_identity', 263, 0, False)
    
    # Passed parameters checking function
    test_identity.stypy_localization = localization
    test_identity.stypy_type_of_self = None
    test_identity.stypy_type_store = module_type_store
    test_identity.stypy_function_name = 'test_identity'
    test_identity.stypy_param_names_list = []
    test_identity.stypy_varargs_param_name = None
    test_identity.stypy_kwargs_param_name = None
    test_identity.stypy_call_defaults = defaults
    test_identity.stypy_call_varargs = varargs
    test_identity.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_identity', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_identity', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_identity(...)' code ##################

    
    # Assigning a Call to a Name (line 264):
    
    # Assigning a Call to a Name (line 264):
    
    # Call to IdentityOperator(...): (line 264)
    # Processing the call arguments (line 264)
    
    # Obtaining an instance of the builtin type 'tuple' (line 264)
    tuple_425596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 264)
    # Adding element type (line 264)
    int_425597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 40), tuple_425596, int_425597)
    # Adding element type (line 264)
    int_425598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 40), tuple_425596, int_425598)
    
    # Processing the call keyword arguments (line 264)
    kwargs_425599 = {}
    # Getting the type of 'interface' (line 264)
    interface_425594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'interface', False)
    # Obtaining the member 'IdentityOperator' of a type (line 264)
    IdentityOperator_425595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 12), interface_425594, 'IdentityOperator')
    # Calling IdentityOperator(args, kwargs) (line 264)
    IdentityOperator_call_result_425600 = invoke(stypy.reporting.localization.Localization(__file__, 264, 12), IdentityOperator_425595, *[tuple_425596], **kwargs_425599)
    
    # Assigning a type to the variable 'ident' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'ident', IdentityOperator_call_result_425600)
    
    # Call to assert_equal(...): (line 265)
    # Processing the call arguments (line 265)
    # Getting the type of 'ident' (line 265)
    ident_425602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 17), 'ident', False)
    
    # Obtaining an instance of the builtin type 'list' (line 265)
    list_425603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 265)
    # Adding element type (line 265)
    int_425604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 25), list_425603, int_425604)
    # Adding element type (line 265)
    int_425605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 25), list_425603, int_425605)
    # Adding element type (line 265)
    int_425606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 25), list_425603, int_425606)
    
    # Applying the binary operator '*' (line 265)
    result_mul_425607 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 17), '*', ident_425602, list_425603)
    
    
    # Obtaining an instance of the builtin type 'list' (line 265)
    list_425608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 265)
    # Adding element type (line 265)
    int_425609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 36), list_425608, int_425609)
    # Adding element type (line 265)
    int_425610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 36), list_425608, int_425610)
    # Adding element type (line 265)
    int_425611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 36), list_425608, int_425611)
    
    # Processing the call keyword arguments (line 265)
    kwargs_425612 = {}
    # Getting the type of 'assert_equal' (line 265)
    assert_equal_425601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 265)
    assert_equal_call_result_425613 = invoke(stypy.reporting.localization.Localization(__file__, 265, 4), assert_equal_425601, *[result_mul_425607, list_425608], **kwargs_425612)
    
    
    # Call to assert_equal(...): (line 266)
    # Processing the call arguments (line 266)
    
    # Call to ravel(...): (line 266)
    # Processing the call keyword arguments (line 266)
    kwargs_425630 = {}
    
    # Call to dot(...): (line 266)
    # Processing the call arguments (line 266)
    
    # Call to reshape(...): (line 266)
    # Processing the call arguments (line 266)
    int_425623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 48), 'int')
    int_425624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 51), 'int')
    # Processing the call keyword arguments (line 266)
    kwargs_425625 = {}
    
    # Call to arange(...): (line 266)
    # Processing the call arguments (line 266)
    int_425619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 37), 'int')
    # Processing the call keyword arguments (line 266)
    kwargs_425620 = {}
    # Getting the type of 'np' (line 266)
    np_425617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 27), 'np', False)
    # Obtaining the member 'arange' of a type (line 266)
    arange_425618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 27), np_425617, 'arange')
    # Calling arange(args, kwargs) (line 266)
    arange_call_result_425621 = invoke(stypy.reporting.localization.Localization(__file__, 266, 27), arange_425618, *[int_425619], **kwargs_425620)
    
    # Obtaining the member 'reshape' of a type (line 266)
    reshape_425622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 27), arange_call_result_425621, 'reshape')
    # Calling reshape(args, kwargs) (line 266)
    reshape_call_result_425626 = invoke(stypy.reporting.localization.Localization(__file__, 266, 27), reshape_425622, *[int_425623, int_425624], **kwargs_425625)
    
    # Processing the call keyword arguments (line 266)
    kwargs_425627 = {}
    # Getting the type of 'ident' (line 266)
    ident_425615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 17), 'ident', False)
    # Obtaining the member 'dot' of a type (line 266)
    dot_425616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 17), ident_425615, 'dot')
    # Calling dot(args, kwargs) (line 266)
    dot_call_result_425628 = invoke(stypy.reporting.localization.Localization(__file__, 266, 17), dot_425616, *[reshape_call_result_425626], **kwargs_425627)
    
    # Obtaining the member 'ravel' of a type (line 266)
    ravel_425629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 17), dot_call_result_425628, 'ravel')
    # Calling ravel(args, kwargs) (line 266)
    ravel_call_result_425631 = invoke(stypy.reporting.localization.Localization(__file__, 266, 17), ravel_425629, *[], **kwargs_425630)
    
    
    # Call to arange(...): (line 266)
    # Processing the call arguments (line 266)
    int_425634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 74), 'int')
    # Processing the call keyword arguments (line 266)
    kwargs_425635 = {}
    # Getting the type of 'np' (line 266)
    np_425632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 64), 'np', False)
    # Obtaining the member 'arange' of a type (line 266)
    arange_425633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 64), np_425632, 'arange')
    # Calling arange(args, kwargs) (line 266)
    arange_call_result_425636 = invoke(stypy.reporting.localization.Localization(__file__, 266, 64), arange_425633, *[int_425634], **kwargs_425635)
    
    # Processing the call keyword arguments (line 266)
    kwargs_425637 = {}
    # Getting the type of 'assert_equal' (line 266)
    assert_equal_425614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 266)
    assert_equal_call_result_425638 = invoke(stypy.reporting.localization.Localization(__file__, 266, 4), assert_equal_425614, *[ravel_call_result_425631, arange_call_result_425636], **kwargs_425637)
    
    
    # Call to assert_raises(...): (line 268)
    # Processing the call arguments (line 268)
    # Getting the type of 'ValueError' (line 268)
    ValueError_425640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 18), 'ValueError', False)
    # Getting the type of 'ident' (line 268)
    ident_425641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 30), 'ident', False)
    # Obtaining the member 'matvec' of a type (line 268)
    matvec_425642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 30), ident_425641, 'matvec')
    
    # Obtaining an instance of the builtin type 'list' (line 268)
    list_425643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 44), 'list')
    # Adding type elements to the builtin type 'list' instance (line 268)
    # Adding element type (line 268)
    int_425644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 44), list_425643, int_425644)
    # Adding element type (line 268)
    int_425645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 44), list_425643, int_425645)
    # Adding element type (line 268)
    int_425646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 44), list_425643, int_425646)
    # Adding element type (line 268)
    int_425647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 44), list_425643, int_425647)
    
    # Processing the call keyword arguments (line 268)
    kwargs_425648 = {}
    # Getting the type of 'assert_raises' (line 268)
    assert_raises_425639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 268)
    assert_raises_call_result_425649 = invoke(stypy.reporting.localization.Localization(__file__, 268, 4), assert_raises_425639, *[ValueError_425640, matvec_425642, list_425643], **kwargs_425648)
    
    
    # ################# End of 'test_identity(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_identity' in the type store
    # Getting the type of 'stypy_return_type' (line 263)
    stypy_return_type_425650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_425650)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_identity'
    return stypy_return_type_425650

# Assigning a type to the variable 'test_identity' (line 263)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'test_identity', test_identity)

@norecursion
def test_attributes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_attributes'
    module_type_store = module_type_store.open_function_context('test_attributes', 271, 0, False)
    
    # Passed parameters checking function
    test_attributes.stypy_localization = localization
    test_attributes.stypy_type_of_self = None
    test_attributes.stypy_type_store = module_type_store
    test_attributes.stypy_function_name = 'test_attributes'
    test_attributes.stypy_param_names_list = []
    test_attributes.stypy_varargs_param_name = None
    test_attributes.stypy_kwargs_param_name = None
    test_attributes.stypy_call_defaults = defaults
    test_attributes.stypy_call_varargs = varargs
    test_attributes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_attributes', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_attributes', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_attributes(...)' code ##################

    
    # Assigning a Call to a Name (line 272):
    
    # Assigning a Call to a Name (line 272):
    
    # Call to aslinearoperator(...): (line 272)
    # Processing the call arguments (line 272)
    
    # Call to reshape(...): (line 272)
    # Processing the call arguments (line 272)
    int_425659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 57), 'int')
    int_425660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 60), 'int')
    # Processing the call keyword arguments (line 272)
    kwargs_425661 = {}
    
    # Call to arange(...): (line 272)
    # Processing the call arguments (line 272)
    int_425655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 45), 'int')
    # Processing the call keyword arguments (line 272)
    kwargs_425656 = {}
    # Getting the type of 'np' (line 272)
    np_425653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 35), 'np', False)
    # Obtaining the member 'arange' of a type (line 272)
    arange_425654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 35), np_425653, 'arange')
    # Calling arange(args, kwargs) (line 272)
    arange_call_result_425657 = invoke(stypy.reporting.localization.Localization(__file__, 272, 35), arange_425654, *[int_425655], **kwargs_425656)
    
    # Obtaining the member 'reshape' of a type (line 272)
    reshape_425658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 35), arange_call_result_425657, 'reshape')
    # Calling reshape(args, kwargs) (line 272)
    reshape_call_result_425662 = invoke(stypy.reporting.localization.Localization(__file__, 272, 35), reshape_425658, *[int_425659, int_425660], **kwargs_425661)
    
    # Processing the call keyword arguments (line 272)
    kwargs_425663 = {}
    # Getting the type of 'interface' (line 272)
    interface_425651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'interface', False)
    # Obtaining the member 'aslinearoperator' of a type (line 272)
    aslinearoperator_425652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), interface_425651, 'aslinearoperator')
    # Calling aslinearoperator(args, kwargs) (line 272)
    aslinearoperator_call_result_425664 = invoke(stypy.reporting.localization.Localization(__file__, 272, 8), aslinearoperator_425652, *[reshape_call_result_425662], **kwargs_425663)
    
    # Assigning a type to the variable 'A' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'A', aslinearoperator_call_result_425664)

    @norecursion
    def always_four_ones(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'always_four_ones'
        module_type_store = module_type_store.open_function_context('always_four_ones', 274, 4, False)
        
        # Passed parameters checking function
        always_four_ones.stypy_localization = localization
        always_four_ones.stypy_type_of_self = None
        always_four_ones.stypy_type_store = module_type_store
        always_four_ones.stypy_function_name = 'always_four_ones'
        always_four_ones.stypy_param_names_list = ['x']
        always_four_ones.stypy_varargs_param_name = None
        always_four_ones.stypy_kwargs_param_name = None
        always_four_ones.stypy_call_defaults = defaults
        always_four_ones.stypy_call_varargs = varargs
        always_four_ones.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'always_four_ones', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'always_four_ones', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'always_four_ones(...)' code ##################

        
        # Assigning a Call to a Name (line 275):
        
        # Assigning a Call to a Name (line 275):
        
        # Call to asarray(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'x' (line 275)
        x_425667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 23), 'x', False)
        # Processing the call keyword arguments (line 275)
        kwargs_425668 = {}
        # Getting the type of 'np' (line 275)
        np_425665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 275)
        asarray_425666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 12), np_425665, 'asarray')
        # Calling asarray(args, kwargs) (line 275)
        asarray_call_result_425669 = invoke(stypy.reporting.localization.Localization(__file__, 275, 12), asarray_425666, *[x_425667], **kwargs_425668)
        
        # Assigning a type to the variable 'x' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'x', asarray_call_result_425669)
        
        # Call to assert_(...): (line 276)
        # Processing the call arguments (line 276)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'x' (line 276)
        x_425671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 16), 'x', False)
        # Obtaining the member 'shape' of a type (line 276)
        shape_425672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 16), x_425671, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 276)
        tuple_425673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 276)
        # Adding element type (line 276)
        int_425674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 28), tuple_425673, int_425674)
        
        # Applying the binary operator '==' (line 276)
        result_eq_425675 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 16), '==', shape_425672, tuple_425673)
        
        
        # Getting the type of 'x' (line 276)
        x_425676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 35), 'x', False)
        # Obtaining the member 'shape' of a type (line 276)
        shape_425677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 35), x_425676, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 276)
        tuple_425678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 276)
        # Adding element type (line 276)
        int_425679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 47), tuple_425678, int_425679)
        # Adding element type (line 276)
        int_425680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 47), tuple_425678, int_425680)
        
        # Applying the binary operator '==' (line 276)
        result_eq_425681 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 35), '==', shape_425677, tuple_425678)
        
        # Applying the binary operator 'or' (line 276)
        result_or_keyword_425682 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 16), 'or', result_eq_425675, result_eq_425681)
        
        # Processing the call keyword arguments (line 276)
        kwargs_425683 = {}
        # Getting the type of 'assert_' (line 276)
        assert__425670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 276)
        assert__call_result_425684 = invoke(stypy.reporting.localization.Localization(__file__, 276, 8), assert__425670, *[result_or_keyword_425682], **kwargs_425683)
        
        
        # Call to ones(...): (line 277)
        # Processing the call arguments (line 277)
        int_425687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 23), 'int')
        # Processing the call keyword arguments (line 277)
        kwargs_425688 = {}
        # Getting the type of 'np' (line 277)
        np_425685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 15), 'np', False)
        # Obtaining the member 'ones' of a type (line 277)
        ones_425686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 15), np_425685, 'ones')
        # Calling ones(args, kwargs) (line 277)
        ones_call_result_425689 = invoke(stypy.reporting.localization.Localization(__file__, 277, 15), ones_425686, *[int_425687], **kwargs_425688)
        
        # Assigning a type to the variable 'stypy_return_type' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'stypy_return_type', ones_call_result_425689)
        
        # ################# End of 'always_four_ones(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'always_four_ones' in the type store
        # Getting the type of 'stypy_return_type' (line 274)
        stypy_return_type_425690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_425690)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'always_four_ones'
        return stypy_return_type_425690

    # Assigning a type to the variable 'always_four_ones' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'always_four_ones', always_four_ones)
    
    # Assigning a Call to a Name (line 279):
    
    # Assigning a Call to a Name (line 279):
    
    # Call to LinearOperator(...): (line 279)
    # Processing the call keyword arguments (line 279)
    
    # Obtaining an instance of the builtin type 'tuple' (line 279)
    tuple_425693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 279)
    # Adding element type (line 279)
    int_425694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 40), tuple_425693, int_425694)
    # Adding element type (line 279)
    int_425695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 40), tuple_425693, int_425695)
    
    keyword_425696 = tuple_425693
    # Getting the type of 'always_four_ones' (line 279)
    always_four_ones_425697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 54), 'always_four_ones', False)
    keyword_425698 = always_four_ones_425697
    kwargs_425699 = {'shape': keyword_425696, 'matvec': keyword_425698}
    # Getting the type of 'interface' (line 279)
    interface_425691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'interface', False)
    # Obtaining the member 'LinearOperator' of a type (line 279)
    LinearOperator_425692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 8), interface_425691, 'LinearOperator')
    # Calling LinearOperator(args, kwargs) (line 279)
    LinearOperator_call_result_425700 = invoke(stypy.reporting.localization.Localization(__file__, 279, 8), LinearOperator_425692, *[], **kwargs_425699)
    
    # Assigning a type to the variable 'B' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'B', LinearOperator_call_result_425700)
    
    
    # Obtaining an instance of the builtin type 'list' (line 281)
    list_425701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 281)
    # Adding element type (line 281)
    # Getting the type of 'A' (line 281)
    A_425702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 15), 'A')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 14), list_425701, A_425702)
    # Adding element type (line 281)
    # Getting the type of 'B' (line 281)
    B_425703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 18), 'B')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 14), list_425701, B_425703)
    # Adding element type (line 281)
    # Getting the type of 'A' (line 281)
    A_425704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 21), 'A')
    # Getting the type of 'B' (line 281)
    B_425705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 25), 'B')
    # Applying the binary operator '*' (line 281)
    result_mul_425706 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 21), '*', A_425704, B_425705)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 14), list_425701, result_mul_425706)
    # Adding element type (line 281)
    # Getting the type of 'A' (line 281)
    A_425707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 28), 'A')
    # Obtaining the member 'H' of a type (line 281)
    H_425708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 28), A_425707, 'H')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 14), list_425701, H_425708)
    # Adding element type (line 281)
    # Getting the type of 'A' (line 281)
    A_425709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 33), 'A')
    # Getting the type of 'A' (line 281)
    A_425710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 37), 'A')
    # Applying the binary operator '+' (line 281)
    result_add_425711 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 33), '+', A_425709, A_425710)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 14), list_425701, result_add_425711)
    # Adding element type (line 281)
    # Getting the type of 'B' (line 281)
    B_425712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 40), 'B')
    # Getting the type of 'B' (line 281)
    B_425713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 44), 'B')
    # Applying the binary operator '+' (line 281)
    result_add_425714 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 40), '+', B_425712, B_425713)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 14), list_425701, result_add_425714)
    # Adding element type (line 281)
    # Getting the type of 'A' (line 281)
    A_425715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 47), 'A')
    int_425716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 52), 'int')
    # Applying the binary operator '**' (line 281)
    result_pow_425717 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 47), '**', A_425715, int_425716)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 14), list_425701, result_pow_425717)
    
    # Testing the type of a for loop iterable (line 281)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 281, 4), list_425701)
    # Getting the type of the for loop variable (line 281)
    for_loop_var_425718 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 281, 4), list_425701)
    # Assigning a type to the variable 'op' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'op', for_loop_var_425718)
    # SSA begins for a for statement (line 281)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_(...): (line 282)
    # Processing the call arguments (line 282)
    
    # Call to hasattr(...): (line 282)
    # Processing the call arguments (line 282)
    # Getting the type of 'op' (line 282)
    op_425721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 24), 'op', False)
    str_425722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 28), 'str', 'dtype')
    # Processing the call keyword arguments (line 282)
    kwargs_425723 = {}
    # Getting the type of 'hasattr' (line 282)
    hasattr_425720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 16), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 282)
    hasattr_call_result_425724 = invoke(stypy.reporting.localization.Localization(__file__, 282, 16), hasattr_425720, *[op_425721, str_425722], **kwargs_425723)
    
    # Processing the call keyword arguments (line 282)
    kwargs_425725 = {}
    # Getting the type of 'assert_' (line 282)
    assert__425719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 282)
    assert__call_result_425726 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), assert__425719, *[hasattr_call_result_425724], **kwargs_425725)
    
    
    # Call to assert_(...): (line 283)
    # Processing the call arguments (line 283)
    
    # Call to hasattr(...): (line 283)
    # Processing the call arguments (line 283)
    # Getting the type of 'op' (line 283)
    op_425729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 24), 'op', False)
    str_425730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 28), 'str', 'shape')
    # Processing the call keyword arguments (line 283)
    kwargs_425731 = {}
    # Getting the type of 'hasattr' (line 283)
    hasattr_425728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 283)
    hasattr_call_result_425732 = invoke(stypy.reporting.localization.Localization(__file__, 283, 16), hasattr_425728, *[op_425729, str_425730], **kwargs_425731)
    
    # Processing the call keyword arguments (line 283)
    kwargs_425733 = {}
    # Getting the type of 'assert_' (line 283)
    assert__425727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 283)
    assert__call_result_425734 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), assert__425727, *[hasattr_call_result_425732], **kwargs_425733)
    
    
    # Call to assert_(...): (line 284)
    # Processing the call arguments (line 284)
    
    # Call to hasattr(...): (line 284)
    # Processing the call arguments (line 284)
    # Getting the type of 'op' (line 284)
    op_425737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 24), 'op', False)
    str_425738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 28), 'str', '_matvec')
    # Processing the call keyword arguments (line 284)
    kwargs_425739 = {}
    # Getting the type of 'hasattr' (line 284)
    hasattr_425736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 284)
    hasattr_call_result_425740 = invoke(stypy.reporting.localization.Localization(__file__, 284, 16), hasattr_425736, *[op_425737, str_425738], **kwargs_425739)
    
    # Processing the call keyword arguments (line 284)
    kwargs_425741 = {}
    # Getting the type of 'assert_' (line 284)
    assert__425735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 284)
    assert__call_result_425742 = invoke(stypy.reporting.localization.Localization(__file__, 284, 8), assert__425735, *[hasattr_call_result_425740], **kwargs_425741)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_attributes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_attributes' in the type store
    # Getting the type of 'stypy_return_type' (line 271)
    stypy_return_type_425743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_425743)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_attributes'
    return stypy_return_type_425743

# Assigning a type to the variable 'test_attributes' (line 271)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 0), 'test_attributes', test_attributes)

@norecursion
def matvec(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'matvec'
    module_type_store = module_type_store.open_function_context('matvec', 286, 0, False)
    
    # Passed parameters checking function
    matvec.stypy_localization = localization
    matvec.stypy_type_of_self = None
    matvec.stypy_type_store = module_type_store
    matvec.stypy_function_name = 'matvec'
    matvec.stypy_param_names_list = ['x']
    matvec.stypy_varargs_param_name = None
    matvec.stypy_kwargs_param_name = None
    matvec.stypy_call_defaults = defaults
    matvec.stypy_call_varargs = varargs
    matvec.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'matvec', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'matvec', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'matvec(...)' code ##################

    str_425744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 4), 'str', ' Needed for test_pickle as local functions are not pickleable ')
    
    # Call to zeros(...): (line 288)
    # Processing the call arguments (line 288)
    int_425747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 20), 'int')
    # Processing the call keyword arguments (line 288)
    kwargs_425748 = {}
    # Getting the type of 'np' (line 288)
    np_425745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 11), 'np', False)
    # Obtaining the member 'zeros' of a type (line 288)
    zeros_425746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 11), np_425745, 'zeros')
    # Calling zeros(args, kwargs) (line 288)
    zeros_call_result_425749 = invoke(stypy.reporting.localization.Localization(__file__, 288, 11), zeros_425746, *[int_425747], **kwargs_425748)
    
    # Assigning a type to the variable 'stypy_return_type' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'stypy_return_type', zeros_call_result_425749)
    
    # ################# End of 'matvec(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'matvec' in the type store
    # Getting the type of 'stypy_return_type' (line 286)
    stypy_return_type_425750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_425750)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'matvec'
    return stypy_return_type_425750

# Assigning a type to the variable 'matvec' (line 286)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 0), 'matvec', matvec)

@norecursion
def test_pickle(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_pickle'
    module_type_store = module_type_store.open_function_context('test_pickle', 290, 0, False)
    
    # Passed parameters checking function
    test_pickle.stypy_localization = localization
    test_pickle.stypy_type_of_self = None
    test_pickle.stypy_type_store = module_type_store
    test_pickle.stypy_function_name = 'test_pickle'
    test_pickle.stypy_param_names_list = []
    test_pickle.stypy_varargs_param_name = None
    test_pickle.stypy_kwargs_param_name = None
    test_pickle.stypy_call_defaults = defaults
    test_pickle.stypy_call_varargs = varargs
    test_pickle.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_pickle', [], None, None, defaults, varargs, kwargs)

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

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 291, 4))
    
    # 'import pickle' statement (line 291)
    import pickle

    import_module(stypy.reporting.localization.Localization(__file__, 291, 4), 'pickle', pickle, module_type_store)
    
    
    
    # Call to range(...): (line 293)
    # Processing the call arguments (line 293)
    # Getting the type of 'pickle' (line 293)
    pickle_425752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 26), 'pickle', False)
    # Obtaining the member 'HIGHEST_PROTOCOL' of a type (line 293)
    HIGHEST_PROTOCOL_425753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 26), pickle_425752, 'HIGHEST_PROTOCOL')
    int_425754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 52), 'int')
    # Applying the binary operator '+' (line 293)
    result_add_425755 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 26), '+', HIGHEST_PROTOCOL_425753, int_425754)
    
    # Processing the call keyword arguments (line 293)
    kwargs_425756 = {}
    # Getting the type of 'range' (line 293)
    range_425751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 20), 'range', False)
    # Calling range(args, kwargs) (line 293)
    range_call_result_425757 = invoke(stypy.reporting.localization.Localization(__file__, 293, 20), range_425751, *[result_add_425755], **kwargs_425756)
    
    # Testing the type of a for loop iterable (line 293)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 293, 4), range_call_result_425757)
    # Getting the type of the for loop variable (line 293)
    for_loop_var_425758 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 293, 4), range_call_result_425757)
    # Assigning a type to the variable 'protocol' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'protocol', for_loop_var_425758)
    # SSA begins for a for statement (line 293)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 294):
    
    # Assigning a Call to a Name (line 294):
    
    # Call to LinearOperator(...): (line 294)
    # Processing the call arguments (line 294)
    
    # Obtaining an instance of the builtin type 'tuple' (line 294)
    tuple_425761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 294)
    # Adding element type (line 294)
    int_425762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 38), tuple_425761, int_425762)
    # Adding element type (line 294)
    int_425763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 38), tuple_425761, int_425763)
    
    # Getting the type of 'matvec' (line 294)
    matvec_425764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 45), 'matvec', False)
    # Processing the call keyword arguments (line 294)
    kwargs_425765 = {}
    # Getting the type of 'interface' (line 294)
    interface_425759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'interface', False)
    # Obtaining the member 'LinearOperator' of a type (line 294)
    LinearOperator_425760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 12), interface_425759, 'LinearOperator')
    # Calling LinearOperator(args, kwargs) (line 294)
    LinearOperator_call_result_425766 = invoke(stypy.reporting.localization.Localization(__file__, 294, 12), LinearOperator_425760, *[tuple_425761, matvec_425764], **kwargs_425765)
    
    # Assigning a type to the variable 'A' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'A', LinearOperator_call_result_425766)
    
    # Assigning a Call to a Name (line 295):
    
    # Assigning a Call to a Name (line 295):
    
    # Call to dumps(...): (line 295)
    # Processing the call arguments (line 295)
    # Getting the type of 'A' (line 295)
    A_425769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 25), 'A', False)
    # Processing the call keyword arguments (line 295)
    # Getting the type of 'protocol' (line 295)
    protocol_425770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 37), 'protocol', False)
    keyword_425771 = protocol_425770
    kwargs_425772 = {'protocol': keyword_425771}
    # Getting the type of 'pickle' (line 295)
    pickle_425767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'pickle', False)
    # Obtaining the member 'dumps' of a type (line 295)
    dumps_425768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 12), pickle_425767, 'dumps')
    # Calling dumps(args, kwargs) (line 295)
    dumps_call_result_425773 = invoke(stypy.reporting.localization.Localization(__file__, 295, 12), dumps_425768, *[A_425769], **kwargs_425772)
    
    # Assigning a type to the variable 's' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 's', dumps_call_result_425773)
    
    # Assigning a Call to a Name (line 296):
    
    # Assigning a Call to a Name (line 296):
    
    # Call to loads(...): (line 296)
    # Processing the call arguments (line 296)
    # Getting the type of 's' (line 296)
    s_425776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 25), 's', False)
    # Processing the call keyword arguments (line 296)
    kwargs_425777 = {}
    # Getting the type of 'pickle' (line 296)
    pickle_425774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'pickle', False)
    # Obtaining the member 'loads' of a type (line 296)
    loads_425775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 12), pickle_425774, 'loads')
    # Calling loads(args, kwargs) (line 296)
    loads_call_result_425778 = invoke(stypy.reporting.localization.Localization(__file__, 296, 12), loads_425775, *[s_425776], **kwargs_425777)
    
    # Assigning a type to the variable 'B' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'B', loads_call_result_425778)
    
    # Getting the type of 'A' (line 298)
    A_425779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 17), 'A')
    # Obtaining the member '__dict__' of a type (line 298)
    dict___425780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 17), A_425779, '__dict__')
    # Testing the type of a for loop iterable (line 298)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 298, 8), dict___425780)
    # Getting the type of the for loop variable (line 298)
    for_loop_var_425781 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 298, 8), dict___425780)
    # Assigning a type to the variable 'k' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'k', for_loop_var_425781)
    # SSA begins for a for statement (line 298)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_equal(...): (line 299)
    # Processing the call arguments (line 299)
    
    # Call to getattr(...): (line 299)
    # Processing the call arguments (line 299)
    # Getting the type of 'A' (line 299)
    A_425784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 33), 'A', False)
    # Getting the type of 'k' (line 299)
    k_425785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 36), 'k', False)
    # Processing the call keyword arguments (line 299)
    kwargs_425786 = {}
    # Getting the type of 'getattr' (line 299)
    getattr_425783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 25), 'getattr', False)
    # Calling getattr(args, kwargs) (line 299)
    getattr_call_result_425787 = invoke(stypy.reporting.localization.Localization(__file__, 299, 25), getattr_425783, *[A_425784, k_425785], **kwargs_425786)
    
    
    # Call to getattr(...): (line 299)
    # Processing the call arguments (line 299)
    # Getting the type of 'B' (line 299)
    B_425789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 48), 'B', False)
    # Getting the type of 'k' (line 299)
    k_425790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 51), 'k', False)
    # Processing the call keyword arguments (line 299)
    kwargs_425791 = {}
    # Getting the type of 'getattr' (line 299)
    getattr_425788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 40), 'getattr', False)
    # Calling getattr(args, kwargs) (line 299)
    getattr_call_result_425792 = invoke(stypy.reporting.localization.Localization(__file__, 299, 40), getattr_425788, *[B_425789, k_425790], **kwargs_425791)
    
    # Processing the call keyword arguments (line 299)
    kwargs_425793 = {}
    # Getting the type of 'assert_equal' (line 299)
    assert_equal_425782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 299)
    assert_equal_call_result_425794 = invoke(stypy.reporting.localization.Localization(__file__, 299, 12), assert_equal_425782, *[getattr_call_result_425787, getattr_call_result_425792], **kwargs_425793)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_pickle(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_pickle' in the type store
    # Getting the type of 'stypy_return_type' (line 290)
    stypy_return_type_425795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_425795)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_pickle'
    return stypy_return_type_425795

# Assigning a type to the variable 'test_pickle' (line 290)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 0), 'test_pickle', test_pickle)

@norecursion
def test_inheritance(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_inheritance'
    module_type_store = module_type_store.open_function_context('test_inheritance', 301, 0, False)
    
    # Passed parameters checking function
    test_inheritance.stypy_localization = localization
    test_inheritance.stypy_type_of_self = None
    test_inheritance.stypy_type_store = module_type_store
    test_inheritance.stypy_function_name = 'test_inheritance'
    test_inheritance.stypy_param_names_list = []
    test_inheritance.stypy_varargs_param_name = None
    test_inheritance.stypy_kwargs_param_name = None
    test_inheritance.stypy_call_defaults = defaults
    test_inheritance.stypy_call_varargs = varargs
    test_inheritance.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_inheritance', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_inheritance', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_inheritance(...)' code ##################

    # Declaration of the 'Empty' class
    # Getting the type of 'interface' (line 302)
    interface_425796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 16), 'interface')
    # Obtaining the member 'LinearOperator' of a type (line 302)
    LinearOperator_425797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 16), interface_425796, 'LinearOperator')

    class Empty(LinearOperator_425797, ):
        pass

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 302, 4, False)
            # Assigning a type to the variable 'self' (line 303)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Empty.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Empty' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'Empty', Empty)
    
    # Call to assert_raises(...): (line 305)
    # Processing the call arguments (line 305)
    # Getting the type of 'TypeError' (line 305)
    TypeError_425799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 18), 'TypeError', False)
    # Getting the type of 'Empty' (line 305)
    Empty_425800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 29), 'Empty', False)
    # Processing the call keyword arguments (line 305)
    kwargs_425801 = {}
    # Getting the type of 'assert_raises' (line 305)
    assert_raises_425798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 305)
    assert_raises_call_result_425802 = invoke(stypy.reporting.localization.Localization(__file__, 305, 4), assert_raises_425798, *[TypeError_425799, Empty_425800], **kwargs_425801)
    
    # Declaration of the 'Identity' class
    # Getting the type of 'interface' (line 307)
    interface_425803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 19), 'interface')
    # Obtaining the member 'LinearOperator' of a type (line 307)
    LinearOperator_425804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 19), interface_425803, 'LinearOperator')

    class Identity(LinearOperator_425804, ):

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 308, 8, False)
            # Assigning a type to the variable 'self' (line 309)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Identity.__init__', ['n'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return

            # Initialize method data
            init_call_information(module_type_store, '__init__', localization, ['n'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__init__(...)' code ##################

            
            # Call to __init__(...): (line 309)
            # Processing the call keyword arguments (line 309)
            # Getting the type of 'None' (line 309)
            None_425811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 49), 'None', False)
            keyword_425812 = None_425811
            
            # Obtaining an instance of the builtin type 'tuple' (line 309)
            tuple_425813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 62), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 309)
            # Adding element type (line 309)
            # Getting the type of 'n' (line 309)
            n_425814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 62), 'n', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 62), tuple_425813, n_425814)
            # Adding element type (line 309)
            # Getting the type of 'n' (line 309)
            n_425815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 65), 'n', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 62), tuple_425813, n_425815)
            
            keyword_425816 = tuple_425813
            kwargs_425817 = {'dtype': keyword_425812, 'shape': keyword_425816}
            
            # Call to super(...): (line 309)
            # Processing the call arguments (line 309)
            # Getting the type of 'Identity' (line 309)
            Identity_425806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 18), 'Identity', False)
            # Getting the type of 'self' (line 309)
            self_425807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 28), 'self', False)
            # Processing the call keyword arguments (line 309)
            kwargs_425808 = {}
            # Getting the type of 'super' (line 309)
            super_425805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'super', False)
            # Calling super(args, kwargs) (line 309)
            super_call_result_425809 = invoke(stypy.reporting.localization.Localization(__file__, 309, 12), super_425805, *[Identity_425806, self_425807], **kwargs_425808)
            
            # Obtaining the member '__init__' of a type (line 309)
            init___425810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 12), super_call_result_425809, '__init__')
            # Calling __init__(args, kwargs) (line 309)
            init___call_result_425818 = invoke(stypy.reporting.localization.Localization(__file__, 309, 12), init___425810, *[], **kwargs_425817)
            
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()


        @norecursion
        def _matvec(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_matvec'
            module_type_store = module_type_store.open_function_context('_matvec', 311, 8, False)
            # Assigning a type to the variable 'self' (line 312)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Identity._matvec.__dict__.__setitem__('stypy_localization', localization)
            Identity._matvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Identity._matvec.__dict__.__setitem__('stypy_type_store', module_type_store)
            Identity._matvec.__dict__.__setitem__('stypy_function_name', 'Identity._matvec')
            Identity._matvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
            Identity._matvec.__dict__.__setitem__('stypy_varargs_param_name', None)
            Identity._matvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Identity._matvec.__dict__.__setitem__('stypy_call_defaults', defaults)
            Identity._matvec.__dict__.__setitem__('stypy_call_varargs', varargs)
            Identity._matvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Identity._matvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Identity._matvec', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_matvec', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_matvec(...)' code ##################

            # Getting the type of 'x' (line 312)
            x_425819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 19), 'x')
            # Assigning a type to the variable 'stypy_return_type' (line 312)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'stypy_return_type', x_425819)
            
            # ################# End of '_matvec(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_matvec' in the type store
            # Getting the type of 'stypy_return_type' (line 311)
            stypy_return_type_425820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_425820)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_matvec'
            return stypy_return_type_425820

    
    # Assigning a type to the variable 'Identity' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'Identity', Identity)
    
    # Assigning a Call to a Name (line 314):
    
    # Assigning a Call to a Name (line 314):
    
    # Call to Identity(...): (line 314)
    # Processing the call arguments (line 314)
    int_425822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 19), 'int')
    # Processing the call keyword arguments (line 314)
    kwargs_425823 = {}
    # Getting the type of 'Identity' (line 314)
    Identity_425821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 10), 'Identity', False)
    # Calling Identity(args, kwargs) (line 314)
    Identity_call_result_425824 = invoke(stypy.reporting.localization.Localization(__file__, 314, 10), Identity_425821, *[int_425822], **kwargs_425823)
    
    # Assigning a type to the variable 'id3' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'id3', Identity_call_result_425824)
    
    # Call to assert_equal(...): (line 315)
    # Processing the call arguments (line 315)
    
    # Call to matvec(...): (line 315)
    # Processing the call arguments (line 315)
    
    # Obtaining an instance of the builtin type 'list' (line 315)
    list_425828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 315)
    # Adding element type (line 315)
    int_425829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 28), list_425828, int_425829)
    # Adding element type (line 315)
    int_425830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 28), list_425828, int_425830)
    # Adding element type (line 315)
    int_425831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 28), list_425828, int_425831)
    
    # Processing the call keyword arguments (line 315)
    kwargs_425832 = {}
    # Getting the type of 'id3' (line 315)
    id3_425826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 17), 'id3', False)
    # Obtaining the member 'matvec' of a type (line 315)
    matvec_425827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 17), id3_425826, 'matvec')
    # Calling matvec(args, kwargs) (line 315)
    matvec_call_result_425833 = invoke(stypy.reporting.localization.Localization(__file__, 315, 17), matvec_425827, *[list_425828], **kwargs_425832)
    
    
    # Obtaining an instance of the builtin type 'list' (line 315)
    list_425834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 315)
    # Adding element type (line 315)
    int_425835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 40), list_425834, int_425835)
    # Adding element type (line 315)
    int_425836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 40), list_425834, int_425836)
    # Adding element type (line 315)
    int_425837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 40), list_425834, int_425837)
    
    # Processing the call keyword arguments (line 315)
    kwargs_425838 = {}
    # Getting the type of 'assert_equal' (line 315)
    assert_equal_425825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 315)
    assert_equal_call_result_425839 = invoke(stypy.reporting.localization.Localization(__file__, 315, 4), assert_equal_425825, *[matvec_call_result_425833, list_425834], **kwargs_425838)
    
    
    # Call to assert_raises(...): (line 316)
    # Processing the call arguments (line 316)
    # Getting the type of 'NotImplementedError' (line 316)
    NotImplementedError_425841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 18), 'NotImplementedError', False)
    # Getting the type of 'id3' (line 316)
    id3_425842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 39), 'id3', False)
    # Obtaining the member 'rmatvec' of a type (line 316)
    rmatvec_425843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 39), id3_425842, 'rmatvec')
    
    # Obtaining an instance of the builtin type 'list' (line 316)
    list_425844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 52), 'list')
    # Adding type elements to the builtin type 'list' instance (line 316)
    # Adding element type (line 316)
    int_425845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 53), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 52), list_425844, int_425845)
    # Adding element type (line 316)
    int_425846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 56), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 52), list_425844, int_425846)
    # Adding element type (line 316)
    int_425847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 59), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 52), list_425844, int_425847)
    
    # Processing the call keyword arguments (line 316)
    kwargs_425848 = {}
    # Getting the type of 'assert_raises' (line 316)
    assert_raises_425840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 316)
    assert_raises_call_result_425849 = invoke(stypy.reporting.localization.Localization(__file__, 316, 4), assert_raises_425840, *[NotImplementedError_425841, rmatvec_425843, list_425844], **kwargs_425848)
    
    # Declaration of the 'MatmatOnly' class
    # Getting the type of 'interface' (line 318)
    interface_425850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 21), 'interface')
    # Obtaining the member 'LinearOperator' of a type (line 318)
    LinearOperator_425851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 21), interface_425850, 'LinearOperator')

    class MatmatOnly(LinearOperator_425851, ):

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 319, 8, False)
            # Assigning a type to the variable 'self' (line 320)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatmatOnly.__init__', ['A'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return

            # Initialize method data
            init_call_information(module_type_store, '__init__', localization, ['A'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__init__(...)' code ##################

            
            # Call to __init__(...): (line 320)
            # Processing the call arguments (line 320)
            # Getting the type of 'A' (line 320)
            A_425858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 45), 'A', False)
            # Obtaining the member 'dtype' of a type (line 320)
            dtype_425859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 45), A_425858, 'dtype')
            # Getting the type of 'A' (line 320)
            A_425860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 54), 'A', False)
            # Obtaining the member 'shape' of a type (line 320)
            shape_425861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 54), A_425860, 'shape')
            # Processing the call keyword arguments (line 320)
            kwargs_425862 = {}
            
            # Call to super(...): (line 320)
            # Processing the call arguments (line 320)
            # Getting the type of 'MatmatOnly' (line 320)
            MatmatOnly_425853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 18), 'MatmatOnly', False)
            # Getting the type of 'self' (line 320)
            self_425854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 30), 'self', False)
            # Processing the call keyword arguments (line 320)
            kwargs_425855 = {}
            # Getting the type of 'super' (line 320)
            super_425852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'super', False)
            # Calling super(args, kwargs) (line 320)
            super_call_result_425856 = invoke(stypy.reporting.localization.Localization(__file__, 320, 12), super_425852, *[MatmatOnly_425853, self_425854], **kwargs_425855)
            
            # Obtaining the member '__init__' of a type (line 320)
            init___425857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 12), super_call_result_425856, '__init__')
            # Calling __init__(args, kwargs) (line 320)
            init___call_result_425863 = invoke(stypy.reporting.localization.Localization(__file__, 320, 12), init___425857, *[dtype_425859, shape_425861], **kwargs_425862)
            
            
            # Assigning a Name to a Attribute (line 321):
            
            # Assigning a Name to a Attribute (line 321):
            # Getting the type of 'A' (line 321)
            A_425864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 21), 'A')
            # Getting the type of 'self' (line 321)
            self_425865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'self')
            # Setting the type of the member 'A' of a type (line 321)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 12), self_425865, 'A', A_425864)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()


        @norecursion
        def _matmat(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_matmat'
            module_type_store = module_type_store.open_function_context('_matmat', 323, 8, False)
            # Assigning a type to the variable 'self' (line 324)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            MatmatOnly._matmat.__dict__.__setitem__('stypy_localization', localization)
            MatmatOnly._matmat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            MatmatOnly._matmat.__dict__.__setitem__('stypy_type_store', module_type_store)
            MatmatOnly._matmat.__dict__.__setitem__('stypy_function_name', 'MatmatOnly._matmat')
            MatmatOnly._matmat.__dict__.__setitem__('stypy_param_names_list', ['x'])
            MatmatOnly._matmat.__dict__.__setitem__('stypy_varargs_param_name', None)
            MatmatOnly._matmat.__dict__.__setitem__('stypy_kwargs_param_name', None)
            MatmatOnly._matmat.__dict__.__setitem__('stypy_call_defaults', defaults)
            MatmatOnly._matmat.__dict__.__setitem__('stypy_call_varargs', varargs)
            MatmatOnly._matmat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            MatmatOnly._matmat.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatmatOnly._matmat', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_matmat', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_matmat(...)' code ##################

            
            # Call to dot(...): (line 324)
            # Processing the call arguments (line 324)
            # Getting the type of 'x' (line 324)
            x_425869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 30), 'x', False)
            # Processing the call keyword arguments (line 324)
            kwargs_425870 = {}
            # Getting the type of 'self' (line 324)
            self_425866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 19), 'self', False)
            # Obtaining the member 'A' of a type (line 324)
            A_425867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 19), self_425866, 'A')
            # Obtaining the member 'dot' of a type (line 324)
            dot_425868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 19), A_425867, 'dot')
            # Calling dot(args, kwargs) (line 324)
            dot_call_result_425871 = invoke(stypy.reporting.localization.Localization(__file__, 324, 19), dot_425868, *[x_425869], **kwargs_425870)
            
            # Assigning a type to the variable 'stypy_return_type' (line 324)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'stypy_return_type', dot_call_result_425871)
            
            # ################# End of '_matmat(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_matmat' in the type store
            # Getting the type of 'stypy_return_type' (line 323)
            stypy_return_type_425872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_425872)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_matmat'
            return stypy_return_type_425872

    
    # Assigning a type to the variable 'MatmatOnly' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'MatmatOnly', MatmatOnly)
    
    # Assigning a Call to a Name (line 326):
    
    # Assigning a Call to a Name (line 326):
    
    # Call to MatmatOnly(...): (line 326)
    # Processing the call arguments (line 326)
    
    # Call to randn(...): (line 326)
    # Processing the call arguments (line 326)
    int_425877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 36), 'int')
    int_425878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 39), 'int')
    # Processing the call keyword arguments (line 326)
    kwargs_425879 = {}
    # Getting the type of 'np' (line 326)
    np_425874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 20), 'np', False)
    # Obtaining the member 'random' of a type (line 326)
    random_425875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 20), np_425874, 'random')
    # Obtaining the member 'randn' of a type (line 326)
    randn_425876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 20), random_425875, 'randn')
    # Calling randn(args, kwargs) (line 326)
    randn_call_result_425880 = invoke(stypy.reporting.localization.Localization(__file__, 326, 20), randn_425876, *[int_425877, int_425878], **kwargs_425879)
    
    # Processing the call keyword arguments (line 326)
    kwargs_425881 = {}
    # Getting the type of 'MatmatOnly' (line 326)
    MatmatOnly_425873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 9), 'MatmatOnly', False)
    # Calling MatmatOnly(args, kwargs) (line 326)
    MatmatOnly_call_result_425882 = invoke(stypy.reporting.localization.Localization(__file__, 326, 9), MatmatOnly_425873, *[randn_call_result_425880], **kwargs_425881)
    
    # Assigning a type to the variable 'mm' (line 326)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'mm', MatmatOnly_call_result_425882)
    
    # Call to assert_equal(...): (line 327)
    # Processing the call arguments (line 327)
    
    # Call to matvec(...): (line 327)
    # Processing the call arguments (line 327)
    
    # Call to randn(...): (line 327)
    # Processing the call arguments (line 327)
    int_425889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 43), 'int')
    # Processing the call keyword arguments (line 327)
    kwargs_425890 = {}
    # Getting the type of 'np' (line 327)
    np_425886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 27), 'np', False)
    # Obtaining the member 'random' of a type (line 327)
    random_425887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 27), np_425886, 'random')
    # Obtaining the member 'randn' of a type (line 327)
    randn_425888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 27), random_425887, 'randn')
    # Calling randn(args, kwargs) (line 327)
    randn_call_result_425891 = invoke(stypy.reporting.localization.Localization(__file__, 327, 27), randn_425888, *[int_425889], **kwargs_425890)
    
    # Processing the call keyword arguments (line 327)
    kwargs_425892 = {}
    # Getting the type of 'mm' (line 327)
    mm_425884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 17), 'mm', False)
    # Obtaining the member 'matvec' of a type (line 327)
    matvec_425885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 17), mm_425884, 'matvec')
    # Calling matvec(args, kwargs) (line 327)
    matvec_call_result_425893 = invoke(stypy.reporting.localization.Localization(__file__, 327, 17), matvec_425885, *[randn_call_result_425891], **kwargs_425892)
    
    # Obtaining the member 'shape' of a type (line 327)
    shape_425894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 17), matvec_call_result_425893, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 327)
    tuple_425895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 55), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 327)
    # Adding element type (line 327)
    int_425896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 55), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 55), tuple_425895, int_425896)
    
    # Processing the call keyword arguments (line 327)
    kwargs_425897 = {}
    # Getting the type of 'assert_equal' (line 327)
    assert_equal_425883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 327)
    assert_equal_call_result_425898 = invoke(stypy.reporting.localization.Localization(__file__, 327, 4), assert_equal_425883, *[shape_425894, tuple_425895], **kwargs_425897)
    
    
    # ################# End of 'test_inheritance(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_inheritance' in the type store
    # Getting the type of 'stypy_return_type' (line 301)
    stypy_return_type_425899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_425899)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_inheritance'
    return stypy_return_type_425899

# Assigning a type to the variable 'test_inheritance' (line 301)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 0), 'test_inheritance', test_inheritance)

@norecursion
def test_dtypes_of_operator_sum(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_dtypes_of_operator_sum'
    module_type_store = module_type_store.open_function_context('test_dtypes_of_operator_sum', 329, 0, False)
    
    # Passed parameters checking function
    test_dtypes_of_operator_sum.stypy_localization = localization
    test_dtypes_of_operator_sum.stypy_type_of_self = None
    test_dtypes_of_operator_sum.stypy_type_store = module_type_store
    test_dtypes_of_operator_sum.stypy_function_name = 'test_dtypes_of_operator_sum'
    test_dtypes_of_operator_sum.stypy_param_names_list = []
    test_dtypes_of_operator_sum.stypy_varargs_param_name = None
    test_dtypes_of_operator_sum.stypy_kwargs_param_name = None
    test_dtypes_of_operator_sum.stypy_call_defaults = defaults
    test_dtypes_of_operator_sum.stypy_call_varargs = varargs
    test_dtypes_of_operator_sum.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_dtypes_of_operator_sum', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_dtypes_of_operator_sum', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_dtypes_of_operator_sum(...)' code ##################

    
    # Assigning a BinOp to a Name (line 332):
    
    # Assigning a BinOp to a Name (line 332):
    
    # Call to rand(...): (line 332)
    # Processing the call arguments (line 332)
    int_425903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 33), 'int')
    int_425904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 35), 'int')
    # Processing the call keyword arguments (line 332)
    kwargs_425905 = {}
    # Getting the type of 'np' (line 332)
    np_425900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 18), 'np', False)
    # Obtaining the member 'random' of a type (line 332)
    random_425901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 18), np_425900, 'random')
    # Obtaining the member 'rand' of a type (line 332)
    rand_425902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 18), random_425901, 'rand')
    # Calling rand(args, kwargs) (line 332)
    rand_call_result_425906 = invoke(stypy.reporting.localization.Localization(__file__, 332, 18), rand_425902, *[int_425903, int_425904], **kwargs_425905)
    
    complex_425907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 40), 'complex')
    
    # Call to rand(...): (line 332)
    # Processing the call arguments (line 332)
    int_425911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 60), 'int')
    int_425912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 62), 'int')
    # Processing the call keyword arguments (line 332)
    kwargs_425913 = {}
    # Getting the type of 'np' (line 332)
    np_425908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 45), 'np', False)
    # Obtaining the member 'random' of a type (line 332)
    random_425909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 45), np_425908, 'random')
    # Obtaining the member 'rand' of a type (line 332)
    rand_425910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 45), random_425909, 'rand')
    # Calling rand(args, kwargs) (line 332)
    rand_call_result_425914 = invoke(stypy.reporting.localization.Localization(__file__, 332, 45), rand_425910, *[int_425911, int_425912], **kwargs_425913)
    
    # Applying the binary operator '*' (line 332)
    result_mul_425915 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 40), '*', complex_425907, rand_call_result_425914)
    
    # Applying the binary operator '+' (line 332)
    result_add_425916 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 18), '+', rand_call_result_425906, result_mul_425915)
    
    # Assigning a type to the variable 'mat_complex' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'mat_complex', result_add_425916)
    
    # Assigning a Call to a Name (line 333):
    
    # Assigning a Call to a Name (line 333):
    
    # Call to rand(...): (line 333)
    # Processing the call arguments (line 333)
    int_425920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 30), 'int')
    int_425921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 32), 'int')
    # Processing the call keyword arguments (line 333)
    kwargs_425922 = {}
    # Getting the type of 'np' (line 333)
    np_425917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 15), 'np', False)
    # Obtaining the member 'random' of a type (line 333)
    random_425918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 15), np_425917, 'random')
    # Obtaining the member 'rand' of a type (line 333)
    rand_425919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 15), random_425918, 'rand')
    # Calling rand(args, kwargs) (line 333)
    rand_call_result_425923 = invoke(stypy.reporting.localization.Localization(__file__, 333, 15), rand_425919, *[int_425920, int_425921], **kwargs_425922)
    
    # Assigning a type to the variable 'mat_real' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'mat_real', rand_call_result_425923)
    
    # Assigning a Call to a Name (line 335):
    
    # Assigning a Call to a Name (line 335):
    
    # Call to aslinearoperator(...): (line 335)
    # Processing the call arguments (line 335)
    # Getting the type of 'mat_complex' (line 335)
    mat_complex_425926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 50), 'mat_complex', False)
    # Processing the call keyword arguments (line 335)
    kwargs_425927 = {}
    # Getting the type of 'interface' (line 335)
    interface_425924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 23), 'interface', False)
    # Obtaining the member 'aslinearoperator' of a type (line 335)
    aslinearoperator_425925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 23), interface_425924, 'aslinearoperator')
    # Calling aslinearoperator(args, kwargs) (line 335)
    aslinearoperator_call_result_425928 = invoke(stypy.reporting.localization.Localization(__file__, 335, 23), aslinearoperator_425925, *[mat_complex_425926], **kwargs_425927)
    
    # Assigning a type to the variable 'complex_operator' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'complex_operator', aslinearoperator_call_result_425928)
    
    # Assigning a Call to a Name (line 336):
    
    # Assigning a Call to a Name (line 336):
    
    # Call to aslinearoperator(...): (line 336)
    # Processing the call arguments (line 336)
    # Getting the type of 'mat_real' (line 336)
    mat_real_425931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 47), 'mat_real', False)
    # Processing the call keyword arguments (line 336)
    kwargs_425932 = {}
    # Getting the type of 'interface' (line 336)
    interface_425929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 20), 'interface', False)
    # Obtaining the member 'aslinearoperator' of a type (line 336)
    aslinearoperator_425930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 20), interface_425929, 'aslinearoperator')
    # Calling aslinearoperator(args, kwargs) (line 336)
    aslinearoperator_call_result_425933 = invoke(stypy.reporting.localization.Localization(__file__, 336, 20), aslinearoperator_425930, *[mat_real_425931], **kwargs_425932)
    
    # Assigning a type to the variable 'real_operator' (line 336)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'real_operator', aslinearoperator_call_result_425933)
    
    # Assigning a BinOp to a Name (line 338):
    
    # Assigning a BinOp to a Name (line 338):
    # Getting the type of 'complex_operator' (line 338)
    complex_operator_425934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 18), 'complex_operator')
    # Getting the type of 'complex_operator' (line 338)
    complex_operator_425935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 37), 'complex_operator')
    # Applying the binary operator '+' (line 338)
    result_add_425936 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 18), '+', complex_operator_425934, complex_operator_425935)
    
    # Assigning a type to the variable 'sum_complex' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'sum_complex', result_add_425936)
    
    # Assigning a BinOp to a Name (line 339):
    
    # Assigning a BinOp to a Name (line 339):
    # Getting the type of 'real_operator' (line 339)
    real_operator_425937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 15), 'real_operator')
    # Getting the type of 'real_operator' (line 339)
    real_operator_425938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 31), 'real_operator')
    # Applying the binary operator '+' (line 339)
    result_add_425939 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 15), '+', real_operator_425937, real_operator_425938)
    
    # Assigning a type to the variable 'sum_real' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'sum_real', result_add_425939)
    
    # Call to assert_equal(...): (line 341)
    # Processing the call arguments (line 341)
    # Getting the type of 'sum_real' (line 341)
    sum_real_425941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 17), 'sum_real', False)
    # Obtaining the member 'dtype' of a type (line 341)
    dtype_425942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 17), sum_real_425941, 'dtype')
    # Getting the type of 'np' (line 341)
    np_425943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 33), 'np', False)
    # Obtaining the member 'float64' of a type (line 341)
    float64_425944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 33), np_425943, 'float64')
    # Processing the call keyword arguments (line 341)
    kwargs_425945 = {}
    # Getting the type of 'assert_equal' (line 341)
    assert_equal_425940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 341)
    assert_equal_call_result_425946 = invoke(stypy.reporting.localization.Localization(__file__, 341, 4), assert_equal_425940, *[dtype_425942, float64_425944], **kwargs_425945)
    
    
    # Call to assert_equal(...): (line 342)
    # Processing the call arguments (line 342)
    # Getting the type of 'sum_complex' (line 342)
    sum_complex_425948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 17), 'sum_complex', False)
    # Obtaining the member 'dtype' of a type (line 342)
    dtype_425949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 17), sum_complex_425948, 'dtype')
    # Getting the type of 'np' (line 342)
    np_425950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 36), 'np', False)
    # Obtaining the member 'complex128' of a type (line 342)
    complex128_425951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 36), np_425950, 'complex128')
    # Processing the call keyword arguments (line 342)
    kwargs_425952 = {}
    # Getting the type of 'assert_equal' (line 342)
    assert_equal_425947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 342)
    assert_equal_call_result_425953 = invoke(stypy.reporting.localization.Localization(__file__, 342, 4), assert_equal_425947, *[dtype_425949, complex128_425951], **kwargs_425952)
    
    
    # ################# End of 'test_dtypes_of_operator_sum(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_dtypes_of_operator_sum' in the type store
    # Getting the type of 'stypy_return_type' (line 329)
    stypy_return_type_425954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_425954)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_dtypes_of_operator_sum'
    return stypy_return_type_425954

# Assigning a type to the variable 'test_dtypes_of_operator_sum' (line 329)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 0), 'test_dtypes_of_operator_sum', test_dtypes_of_operator_sum)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
