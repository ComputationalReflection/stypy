
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import sys
4: import os
5: import gc
6: import re
7: import threading
8: 
9: import numpy as np
10: from numpy.testing import assert_equal, assert_, assert_allclose
11: from scipy.sparse import (_sparsetools, coo_matrix, csr_matrix, csc_matrix,
12:                           bsr_matrix, dia_matrix)
13: from scipy.sparse.sputils import supported_dtypes
14: from scipy._lib._testutils import check_free_memory
15: 
16: import pytest
17: from pytest import raises as assert_raises
18: 
19: def test_exception():
20:     assert_raises(MemoryError, _sparsetools.test_throw_error)
21: 
22: 
23: def test_threads():
24:     # Smoke test for parallel threaded execution; doesn't actually
25:     # check that code runs in parallel, but just that it produces
26:     # expected results.
27:     nthreads = 10
28:     niter = 100
29: 
30:     n = 20
31:     a = csr_matrix(np.ones([n, n]))
32:     bres = []
33: 
34:     class Worker(threading.Thread):
35:         def run(self):
36:             b = a.copy()
37:             for j in range(niter):
38:                 _sparsetools.csr_plus_csr(n, n,
39:                                           a.indptr, a.indices, a.data,
40:                                           a.indptr, a.indices, a.data,
41:                                           b.indptr, b.indices, b.data)
42:             bres.append(b)
43: 
44:     threads = [Worker() for _ in range(nthreads)]
45:     for thread in threads:
46:         thread.start()
47:     for thread in threads:
48:         thread.join()
49: 
50:     for b in bres:
51:         assert_(np.all(b.toarray() == 2))
52: 
53: 
54: def test_regression_std_vector_dtypes():
55:     # Regression test for gh-3780, checking the std::vector typemaps
56:     # in sparsetools.cxx are complete.
57:     for dtype in supported_dtypes:
58:         ad = np.matrix([[1, 2], [3, 4]]).astype(dtype)
59:         a = csr_matrix(ad, dtype=dtype)
60: 
61:         # getcol is one function using std::vector typemaps, and should not fail
62:         assert_equal(a.getcol(0).todense(), ad[:,0])
63: 
64: 
65: @pytest.mark.skipif(not (sys.platform.startswith('linux') and np.dtype(np.intp).itemsize >= 8),
66:                     reason="test requires 64-bit Linux")
67: class TestInt32Overflow(object):
68:     '''
69:     Some of the sparsetools routines use dense 2D matrices whose
70:     total size is not bounded by the nnz of the sparse matrix. These
71:     routines used to suffer from int32 wraparounds; here, we try to
72:     check that the wraparounds don't occur any more.
73:     '''
74:     # choose n large enough
75:     n = 50000
76: 
77:     def setup_method(self):
78:         assert self.n**2 > np.iinfo(np.int32).max
79: 
80:         # check there's enough memory even if everything is run at the
81:         # same time
82:         try:
83:             parallel_count = int(os.environ.get('PYTEST_XDIST_WORKER_COUNT', '1'))
84:         except ValueError:
85:             parallel_count = np.inf
86: 
87:         check_free_memory(3000 * parallel_count)
88: 
89:     def teardown_method(self):
90:         gc.collect()
91: 
92:     def test_coo_todense(self):
93:         # Check *_todense routines (cf. gh-2179)
94:         #
95:         # All of them in the end call coo_matrix.todense
96: 
97:         n = self.n
98: 
99:         i = np.array([0, n-1])
100:         j = np.array([0, n-1])
101:         data = np.array([1, 2], dtype=np.int8)
102:         m = coo_matrix((data, (i, j)))
103: 
104:         r = m.todense()
105:         assert_equal(r[0,0], 1)
106:         assert_equal(r[-1,-1], 2)
107:         del r
108:         gc.collect()
109: 
110:     @pytest.mark.slow
111:     def test_matvecs(self):
112:         # Check *_matvecs routines
113:         n = self.n
114: 
115:         i = np.array([0, n-1])
116:         j = np.array([0, n-1])
117:         data = np.array([1, 2], dtype=np.int8)
118:         m = coo_matrix((data, (i, j)))
119: 
120:         b = np.ones((n, n), dtype=np.int8)
121:         for sptype in (csr_matrix, csc_matrix, bsr_matrix):
122:             m2 = sptype(m)
123:             r = m2.dot(b)
124:             assert_equal(r[0,0], 1)
125:             assert_equal(r[-1,-1], 2)
126:             del r
127:             gc.collect()
128: 
129:         del b
130:         gc.collect()
131: 
132:     @pytest.mark.slow
133:     def test_dia_matvec(self):
134:         # Check: huge dia_matrix _matvec
135:         n = self.n
136:         data = np.ones((n, n), dtype=np.int8)
137:         offsets = np.arange(n)
138:         m = dia_matrix((data, offsets), shape=(n, n))
139:         v = np.ones(m.shape[1], dtype=np.int8)
140:         r = m.dot(v)
141:         assert_equal(r[0], np.int8(n))
142:         del data, offsets, m, v, r
143:         gc.collect()
144: 
145:     _bsr_ops = [pytest.param("matmat", marks=pytest.mark.xslow),
146:                 pytest.param("matvecs", marks=pytest.mark.xslow),
147:                 "matvec",
148:                 "diagonal",
149:                 "sort_indices",
150:                 pytest.param("transpose", marks=pytest.mark.xslow)]
151: 
152:     @pytest.mark.slow
153:     @pytest.mark.parametrize("op", _bsr_ops)
154:     def test_bsr_1_block(self, op):
155:         # Check: huge bsr_matrix (1-block)
156:         #
157:         # The point here is that indices inside a block may overflow.
158: 
159:         def get_matrix():
160:             n = self.n
161:             data = np.ones((1, n, n), dtype=np.int8)
162:             indptr = np.array([0, 1], dtype=np.int32)
163:             indices = np.array([0], dtype=np.int32)
164:             m = bsr_matrix((data, indices, indptr), blocksize=(n, n), copy=False)
165:             del data, indptr, indices
166:             return m
167: 
168:         gc.collect()
169:         try:
170:             getattr(self, "_check_bsr_" + op)(get_matrix)
171:         finally:
172:             gc.collect()
173: 
174:     @pytest.mark.slow
175:     @pytest.mark.parametrize("op", _bsr_ops)
176:     def test_bsr_n_block(self, op):
177:         # Check: huge bsr_matrix (n-block)
178:         #
179:         # The point here is that while indices within a block don't
180:         # overflow, accumulators across many block may.
181: 
182:         def get_matrix():
183:             n = self.n
184:             data = np.ones((n, n, 1), dtype=np.int8)
185:             indptr = np.array([0, n], dtype=np.int32)
186:             indices = np.arange(n, dtype=np.int32)
187:             m = bsr_matrix((data, indices, indptr), blocksize=(n, 1), copy=False)
188:             del data, indptr, indices
189:             return m
190: 
191:         gc.collect()
192:         try:
193:             getattr(self, "_check_bsr_" + op)(get_matrix)
194:         finally:
195:             gc.collect()
196: 
197:     def _check_bsr_matvecs(self, m):
198:         m = m()
199:         n = self.n
200: 
201:         # _matvecs
202:         r = m.dot(np.ones((n, 2), dtype=np.int8))
203:         assert_equal(r[0,0], np.int8(n))
204: 
205:     def _check_bsr_matvec(self, m):
206:         m = m()
207:         n = self.n
208: 
209:         # _matvec
210:         r = m.dot(np.ones((n,), dtype=np.int8))
211:         assert_equal(r[0], np.int8(n))
212: 
213:     def _check_bsr_diagonal(self, m):
214:         m = m()
215:         n = self.n
216: 
217:         # _diagonal
218:         r = m.diagonal()
219:         assert_equal(r, np.ones(n))
220: 
221:     def _check_bsr_sort_indices(self, m):
222:         # _sort_indices
223:         m = m()
224:         m.sort_indices()
225: 
226:     def _check_bsr_transpose(self, m):
227:         # _transpose
228:         m = m()
229:         m.transpose()
230: 
231:     def _check_bsr_matmat(self, m):
232:         m = m()
233:         n = self.n
234: 
235:         # _bsr_matmat
236:         m2 = bsr_matrix(np.ones((n, 2), dtype=np.int8), blocksize=(m.blocksize[1], 2))
237:         m.dot(m2)  # shouldn't SIGSEGV
238:         del m2
239: 
240:         # _bsr_matmat
241:         m2 = bsr_matrix(np.ones((2, n), dtype=np.int8), blocksize=(2, m.blocksize[0]))
242:         m2.dot(m)  # shouldn't SIGSEGV
243: 
244: 
245: @pytest.mark.skip(reason="64-bit indices in sparse matrices not available")
246: def test_csr_matmat_int64_overflow():
247:     n = 3037000500
248:     assert n**2 > np.iinfo(np.int64).max
249: 
250:     # the test would take crazy amounts of memory
251:     check_free_memory(n * (8*2 + 1) * 3 / 1e6)
252: 
253:     # int64 overflow
254:     data = np.ones((n,), dtype=np.int8)
255:     indptr = np.arange(n+1, dtype=np.int64)
256:     indices = np.zeros(n, dtype=np.int64)
257:     a = csr_matrix((data, indices, indptr))
258:     b = a.T
259: 
260:     assert_raises(RuntimeError, a.dot, b)
261: 
262: 
263: def test_upcast():
264:     a0 = csr_matrix([[np.pi, np.pi*1j], [3, 4]], dtype=complex)
265:     b0 = np.array([256+1j, 2**32], dtype=complex)
266: 
267:     for a_dtype in supported_dtypes:
268:         for b_dtype in supported_dtypes:
269:             msg = "(%r, %r)" % (a_dtype, b_dtype)
270: 
271:             if np.issubdtype(a_dtype, np.complexfloating):
272:                 a = a0.copy().astype(a_dtype)
273:             else:
274:                 a = a0.real.copy().astype(a_dtype)
275: 
276:             if np.issubdtype(b_dtype, np.complexfloating):
277:                 b = b0.copy().astype(b_dtype)
278:             else:
279:                 b = b0.real.copy().astype(b_dtype)
280: 
281:             if not (a_dtype == np.bool_ and b_dtype == np.bool_):
282:                 c = np.zeros((2,), dtype=np.bool_)
283:                 assert_raises(ValueError, _sparsetools.csr_matvec,
284:                               2, 2, a.indptr, a.indices, a.data, b, c)
285: 
286:             if ((np.issubdtype(a_dtype, np.complexfloating) and
287:                  not np.issubdtype(b_dtype, np.complexfloating)) or
288:                 (not np.issubdtype(a_dtype, np.complexfloating) and
289:                  np.issubdtype(b_dtype, np.complexfloating))):
290:                 c = np.zeros((2,), dtype=np.float64)
291:                 assert_raises(ValueError, _sparsetools.csr_matvec,
292:                               2, 2, a.indptr, a.indices, a.data, b, c)
293: 
294:             c = np.zeros((2,), dtype=np.result_type(a_dtype, b_dtype))
295:             _sparsetools.csr_matvec(2, 2, a.indptr, a.indices, a.data, b, c)
296:             assert_allclose(c, np.dot(a.toarray(), b), err_msg=msg)
297: 
298: 
299: def test_endianness():
300:     d = np.ones((3,4))
301:     offsets = [-1,0,1]
302: 
303:     a = dia_matrix((d.astype('<f8'), offsets), (4, 4))
304:     b = dia_matrix((d.astype('>f8'), offsets), (4, 4))
305:     v = np.arange(4)
306: 
307:     assert_allclose(a.dot(v), [1, 3, 6, 5])
308:     assert_allclose(b.dot(v), [1, 3, 6, 5])
309: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import sys' statement (line 3)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import os' statement (line 4)
import os

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import gc' statement (line 5)
import gc

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'gc', gc, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import re' statement (line 6)
import re

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import threading' statement (line 7)
import threading

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'threading', threading, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import numpy' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_460423 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_460423) is not StypyTypeError):

    if (import_460423 != 'pyd_module'):
        __import__(import_460423)
        sys_modules_460424 = sys.modules[import_460423]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', sys_modules_460424.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_460423)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from numpy.testing import assert_equal, assert_, assert_allclose' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_460425 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.testing')

if (type(import_460425) is not StypyTypeError):

    if (import_460425 != 'pyd_module'):
        __import__(import_460425)
        sys_modules_460426 = sys.modules[import_460425]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.testing', sys_modules_460426.module_type_store, module_type_store, ['assert_equal', 'assert_', 'assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_460426, sys_modules_460426.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_, assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_', 'assert_allclose'], [assert_equal, assert_, assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.testing', import_460425)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.sparse import _sparsetools, coo_matrix, csr_matrix, csc_matrix, bsr_matrix, dia_matrix' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_460427 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse')

if (type(import_460427) is not StypyTypeError):

    if (import_460427 != 'pyd_module'):
        __import__(import_460427)
        sys_modules_460428 = sys.modules[import_460427]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse', sys_modules_460428.module_type_store, module_type_store, ['_sparsetools', 'coo_matrix', 'csr_matrix', 'csc_matrix', 'bsr_matrix', 'dia_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_460428, sys_modules_460428.module_type_store, module_type_store)
    else:
        from scipy.sparse import _sparsetools, coo_matrix, csr_matrix, csc_matrix, bsr_matrix, dia_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse', None, module_type_store, ['_sparsetools', 'coo_matrix', 'csr_matrix', 'csc_matrix', 'bsr_matrix', 'dia_matrix'], [_sparsetools, coo_matrix, csr_matrix, csc_matrix, bsr_matrix, dia_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse', import_460427)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.sparse.sputils import supported_dtypes' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_460429 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.sputils')

if (type(import_460429) is not StypyTypeError):

    if (import_460429 != 'pyd_module'):
        __import__(import_460429)
        sys_modules_460430 = sys.modules[import_460429]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.sputils', sys_modules_460430.module_type_store, module_type_store, ['supported_dtypes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_460430, sys_modules_460430.module_type_store, module_type_store)
    else:
        from scipy.sparse.sputils import supported_dtypes

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.sputils', None, module_type_store, ['supported_dtypes'], [supported_dtypes])

else:
    # Assigning a type to the variable 'scipy.sparse.sputils' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.sputils', import_460429)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy._lib._testutils import check_free_memory' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_460431 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy._lib._testutils')

if (type(import_460431) is not StypyTypeError):

    if (import_460431 != 'pyd_module'):
        __import__(import_460431)
        sys_modules_460432 = sys.modules[import_460431]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy._lib._testutils', sys_modules_460432.module_type_store, module_type_store, ['check_free_memory'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_460432, sys_modules_460432.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import check_free_memory

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy._lib._testutils', None, module_type_store, ['check_free_memory'], [check_free_memory])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy._lib._testutils', import_460431)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import pytest' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_460433 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'pytest')

if (type(import_460433) is not StypyTypeError):

    if (import_460433 != 'pyd_module'):
        __import__(import_460433)
        sys_modules_460434 = sys.modules[import_460433]
        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'pytest', sys_modules_460434.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'pytest', import_460433)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from pytest import assert_raises' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_460435 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'pytest')

if (type(import_460435) is not StypyTypeError):

    if (import_460435 != 'pyd_module'):
        __import__(import_460435)
        sys_modules_460436 = sys.modules[import_460435]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'pytest', sys_modules_460436.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_460436, sys_modules_460436.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'pytest', import_460435)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')


@norecursion
def test_exception(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_exception'
    module_type_store = module_type_store.open_function_context('test_exception', 19, 0, False)
    
    # Passed parameters checking function
    test_exception.stypy_localization = localization
    test_exception.stypy_type_of_self = None
    test_exception.stypy_type_store = module_type_store
    test_exception.stypy_function_name = 'test_exception'
    test_exception.stypy_param_names_list = []
    test_exception.stypy_varargs_param_name = None
    test_exception.stypy_kwargs_param_name = None
    test_exception.stypy_call_defaults = defaults
    test_exception.stypy_call_varargs = varargs
    test_exception.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_exception', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_exception', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_exception(...)' code ##################

    
    # Call to assert_raises(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'MemoryError' (line 20)
    MemoryError_460438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 18), 'MemoryError', False)
    # Getting the type of '_sparsetools' (line 20)
    _sparsetools_460439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 31), '_sparsetools', False)
    # Obtaining the member 'test_throw_error' of a type (line 20)
    test_throw_error_460440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 31), _sparsetools_460439, 'test_throw_error')
    # Processing the call keyword arguments (line 20)
    kwargs_460441 = {}
    # Getting the type of 'assert_raises' (line 20)
    assert_raises_460437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 20)
    assert_raises_call_result_460442 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), assert_raises_460437, *[MemoryError_460438, test_throw_error_460440], **kwargs_460441)
    
    
    # ################# End of 'test_exception(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_exception' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_460443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_460443)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_exception'
    return stypy_return_type_460443

# Assigning a type to the variable 'test_exception' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'test_exception', test_exception)

@norecursion
def test_threads(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_threads'
    module_type_store = module_type_store.open_function_context('test_threads', 23, 0, False)
    
    # Passed parameters checking function
    test_threads.stypy_localization = localization
    test_threads.stypy_type_of_self = None
    test_threads.stypy_type_store = module_type_store
    test_threads.stypy_function_name = 'test_threads'
    test_threads.stypy_param_names_list = []
    test_threads.stypy_varargs_param_name = None
    test_threads.stypy_kwargs_param_name = None
    test_threads.stypy_call_defaults = defaults
    test_threads.stypy_call_varargs = varargs
    test_threads.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_threads', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_threads', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_threads(...)' code ##################

    
    # Assigning a Num to a Name (line 27):
    int_460444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 15), 'int')
    # Assigning a type to the variable 'nthreads' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'nthreads', int_460444)
    
    # Assigning a Num to a Name (line 28):
    int_460445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 12), 'int')
    # Assigning a type to the variable 'niter' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'niter', int_460445)
    
    # Assigning a Num to a Name (line 30):
    int_460446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 8), 'int')
    # Assigning a type to the variable 'n' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'n', int_460446)
    
    # Assigning a Call to a Name (line 31):
    
    # Call to csr_matrix(...): (line 31)
    # Processing the call arguments (line 31)
    
    # Call to ones(...): (line 31)
    # Processing the call arguments (line 31)
    
    # Obtaining an instance of the builtin type 'list' (line 31)
    list_460450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 31)
    # Adding element type (line 31)
    # Getting the type of 'n' (line 31)
    n_460451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 28), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 27), list_460450, n_460451)
    # Adding element type (line 31)
    # Getting the type of 'n' (line 31)
    n_460452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 31), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 27), list_460450, n_460452)
    
    # Processing the call keyword arguments (line 31)
    kwargs_460453 = {}
    # Getting the type of 'np' (line 31)
    np_460448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 19), 'np', False)
    # Obtaining the member 'ones' of a type (line 31)
    ones_460449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 19), np_460448, 'ones')
    # Calling ones(args, kwargs) (line 31)
    ones_call_result_460454 = invoke(stypy.reporting.localization.Localization(__file__, 31, 19), ones_460449, *[list_460450], **kwargs_460453)
    
    # Processing the call keyword arguments (line 31)
    kwargs_460455 = {}
    # Getting the type of 'csr_matrix' (line 31)
    csr_matrix_460447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'csr_matrix', False)
    # Calling csr_matrix(args, kwargs) (line 31)
    csr_matrix_call_result_460456 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), csr_matrix_460447, *[ones_call_result_460454], **kwargs_460455)
    
    # Assigning a type to the variable 'a' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'a', csr_matrix_call_result_460456)
    
    # Assigning a List to a Name (line 32):
    
    # Obtaining an instance of the builtin type 'list' (line 32)
    list_460457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 32)
    
    # Assigning a type to the variable 'bres' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'bres', list_460457)
    # Declaration of the 'Worker' class
    # Getting the type of 'threading' (line 34)
    threading_460458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 17), 'threading')
    # Obtaining the member 'Thread' of a type (line 34)
    Thread_460459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 17), threading_460458, 'Thread')

    class Worker(Thread_460459, ):

        @norecursion
        def run(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'run'
            module_type_store = module_type_store.open_function_context('run', 35, 8, False)
            # Assigning a type to the variable 'self' (line 36)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Worker.run.__dict__.__setitem__('stypy_localization', localization)
            Worker.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Worker.run.__dict__.__setitem__('stypy_type_store', module_type_store)
            Worker.run.__dict__.__setitem__('stypy_function_name', 'Worker.run')
            Worker.run.__dict__.__setitem__('stypy_param_names_list', [])
            Worker.run.__dict__.__setitem__('stypy_varargs_param_name', None)
            Worker.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Worker.run.__dict__.__setitem__('stypy_call_defaults', defaults)
            Worker.run.__dict__.__setitem__('stypy_call_varargs', varargs)
            Worker.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Worker.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Worker.run', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'run', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'run(...)' code ##################

            
            # Assigning a Call to a Name (line 36):
            
            # Call to copy(...): (line 36)
            # Processing the call keyword arguments (line 36)
            kwargs_460462 = {}
            # Getting the type of 'a' (line 36)
            a_460460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'a', False)
            # Obtaining the member 'copy' of a type (line 36)
            copy_460461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 16), a_460460, 'copy')
            # Calling copy(args, kwargs) (line 36)
            copy_call_result_460463 = invoke(stypy.reporting.localization.Localization(__file__, 36, 16), copy_460461, *[], **kwargs_460462)
            
            # Assigning a type to the variable 'b' (line 36)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'b', copy_call_result_460463)
            
            
            # Call to range(...): (line 37)
            # Processing the call arguments (line 37)
            # Getting the type of 'niter' (line 37)
            niter_460465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 27), 'niter', False)
            # Processing the call keyword arguments (line 37)
            kwargs_460466 = {}
            # Getting the type of 'range' (line 37)
            range_460464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 21), 'range', False)
            # Calling range(args, kwargs) (line 37)
            range_call_result_460467 = invoke(stypy.reporting.localization.Localization(__file__, 37, 21), range_460464, *[niter_460465], **kwargs_460466)
            
            # Testing the type of a for loop iterable (line 37)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 37, 12), range_call_result_460467)
            # Getting the type of the for loop variable (line 37)
            for_loop_var_460468 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 37, 12), range_call_result_460467)
            # Assigning a type to the variable 'j' (line 37)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'j', for_loop_var_460468)
            # SSA begins for a for statement (line 37)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to csr_plus_csr(...): (line 38)
            # Processing the call arguments (line 38)
            # Getting the type of 'n' (line 38)
            n_460471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 42), 'n', False)
            # Getting the type of 'n' (line 38)
            n_460472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 45), 'n', False)
            # Getting the type of 'a' (line 39)
            a_460473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 42), 'a', False)
            # Obtaining the member 'indptr' of a type (line 39)
            indptr_460474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 42), a_460473, 'indptr')
            # Getting the type of 'a' (line 39)
            a_460475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 52), 'a', False)
            # Obtaining the member 'indices' of a type (line 39)
            indices_460476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 52), a_460475, 'indices')
            # Getting the type of 'a' (line 39)
            a_460477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 63), 'a', False)
            # Obtaining the member 'data' of a type (line 39)
            data_460478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 63), a_460477, 'data')
            # Getting the type of 'a' (line 40)
            a_460479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 42), 'a', False)
            # Obtaining the member 'indptr' of a type (line 40)
            indptr_460480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 42), a_460479, 'indptr')
            # Getting the type of 'a' (line 40)
            a_460481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 52), 'a', False)
            # Obtaining the member 'indices' of a type (line 40)
            indices_460482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 52), a_460481, 'indices')
            # Getting the type of 'a' (line 40)
            a_460483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 63), 'a', False)
            # Obtaining the member 'data' of a type (line 40)
            data_460484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 63), a_460483, 'data')
            # Getting the type of 'b' (line 41)
            b_460485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 42), 'b', False)
            # Obtaining the member 'indptr' of a type (line 41)
            indptr_460486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 42), b_460485, 'indptr')
            # Getting the type of 'b' (line 41)
            b_460487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 52), 'b', False)
            # Obtaining the member 'indices' of a type (line 41)
            indices_460488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 52), b_460487, 'indices')
            # Getting the type of 'b' (line 41)
            b_460489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 63), 'b', False)
            # Obtaining the member 'data' of a type (line 41)
            data_460490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 63), b_460489, 'data')
            # Processing the call keyword arguments (line 38)
            kwargs_460491 = {}
            # Getting the type of '_sparsetools' (line 38)
            _sparsetools_460469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), '_sparsetools', False)
            # Obtaining the member 'csr_plus_csr' of a type (line 38)
            csr_plus_csr_460470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 16), _sparsetools_460469, 'csr_plus_csr')
            # Calling csr_plus_csr(args, kwargs) (line 38)
            csr_plus_csr_call_result_460492 = invoke(stypy.reporting.localization.Localization(__file__, 38, 16), csr_plus_csr_460470, *[n_460471, n_460472, indptr_460474, indices_460476, data_460478, indptr_460480, indices_460482, data_460484, indptr_460486, indices_460488, data_460490], **kwargs_460491)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to append(...): (line 42)
            # Processing the call arguments (line 42)
            # Getting the type of 'b' (line 42)
            b_460495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'b', False)
            # Processing the call keyword arguments (line 42)
            kwargs_460496 = {}
            # Getting the type of 'bres' (line 42)
            bres_460493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'bres', False)
            # Obtaining the member 'append' of a type (line 42)
            append_460494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 12), bres_460493, 'append')
            # Calling append(args, kwargs) (line 42)
            append_call_result_460497 = invoke(stypy.reporting.localization.Localization(__file__, 42, 12), append_460494, *[b_460495], **kwargs_460496)
            
            
            # ################# End of 'run(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'run' in the type store
            # Getting the type of 'stypy_return_type' (line 35)
            stypy_return_type_460498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_460498)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'run'
            return stypy_return_type_460498


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 34, 4, False)
            # Assigning a type to the variable 'self' (line 35)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Worker.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Worker' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'Worker', Worker)
    
    # Assigning a ListComp to a Name (line 44):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'nthreads' (line 44)
    nthreads_460503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 39), 'nthreads', False)
    # Processing the call keyword arguments (line 44)
    kwargs_460504 = {}
    # Getting the type of 'range' (line 44)
    range_460502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 33), 'range', False)
    # Calling range(args, kwargs) (line 44)
    range_call_result_460505 = invoke(stypy.reporting.localization.Localization(__file__, 44, 33), range_460502, *[nthreads_460503], **kwargs_460504)
    
    comprehension_460506 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 15), range_call_result_460505)
    # Assigning a type to the variable '_' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), '_', comprehension_460506)
    
    # Call to Worker(...): (line 44)
    # Processing the call keyword arguments (line 44)
    kwargs_460500 = {}
    # Getting the type of 'Worker' (line 44)
    Worker_460499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'Worker', False)
    # Calling Worker(args, kwargs) (line 44)
    Worker_call_result_460501 = invoke(stypy.reporting.localization.Localization(__file__, 44, 15), Worker_460499, *[], **kwargs_460500)
    
    list_460507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 15), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 15), list_460507, Worker_call_result_460501)
    # Assigning a type to the variable 'threads' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'threads', list_460507)
    
    # Getting the type of 'threads' (line 45)
    threads_460508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 18), 'threads')
    # Testing the type of a for loop iterable (line 45)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 45, 4), threads_460508)
    # Getting the type of the for loop variable (line 45)
    for_loop_var_460509 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 45, 4), threads_460508)
    # Assigning a type to the variable 'thread' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'thread', for_loop_var_460509)
    # SSA begins for a for statement (line 45)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to start(...): (line 46)
    # Processing the call keyword arguments (line 46)
    kwargs_460512 = {}
    # Getting the type of 'thread' (line 46)
    thread_460510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'thread', False)
    # Obtaining the member 'start' of a type (line 46)
    start_460511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), thread_460510, 'start')
    # Calling start(args, kwargs) (line 46)
    start_call_result_460513 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), start_460511, *[], **kwargs_460512)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'threads' (line 47)
    threads_460514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 18), 'threads')
    # Testing the type of a for loop iterable (line 47)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 47, 4), threads_460514)
    # Getting the type of the for loop variable (line 47)
    for_loop_var_460515 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 47, 4), threads_460514)
    # Assigning a type to the variable 'thread' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'thread', for_loop_var_460515)
    # SSA begins for a for statement (line 47)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to join(...): (line 48)
    # Processing the call keyword arguments (line 48)
    kwargs_460518 = {}
    # Getting the type of 'thread' (line 48)
    thread_460516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'thread', False)
    # Obtaining the member 'join' of a type (line 48)
    join_460517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), thread_460516, 'join')
    # Calling join(args, kwargs) (line 48)
    join_call_result_460519 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), join_460517, *[], **kwargs_460518)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'bres' (line 50)
    bres_460520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 13), 'bres')
    # Testing the type of a for loop iterable (line 50)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 50, 4), bres_460520)
    # Getting the type of the for loop variable (line 50)
    for_loop_var_460521 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 50, 4), bres_460520)
    # Assigning a type to the variable 'b' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'b', for_loop_var_460521)
    # SSA begins for a for statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_(...): (line 51)
    # Processing the call arguments (line 51)
    
    # Call to all(...): (line 51)
    # Processing the call arguments (line 51)
    
    
    # Call to toarray(...): (line 51)
    # Processing the call keyword arguments (line 51)
    kwargs_460527 = {}
    # Getting the type of 'b' (line 51)
    b_460525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 23), 'b', False)
    # Obtaining the member 'toarray' of a type (line 51)
    toarray_460526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 23), b_460525, 'toarray')
    # Calling toarray(args, kwargs) (line 51)
    toarray_call_result_460528 = invoke(stypy.reporting.localization.Localization(__file__, 51, 23), toarray_460526, *[], **kwargs_460527)
    
    int_460529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 38), 'int')
    # Applying the binary operator '==' (line 51)
    result_eq_460530 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 23), '==', toarray_call_result_460528, int_460529)
    
    # Processing the call keyword arguments (line 51)
    kwargs_460531 = {}
    # Getting the type of 'np' (line 51)
    np_460523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'np', False)
    # Obtaining the member 'all' of a type (line 51)
    all_460524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 16), np_460523, 'all')
    # Calling all(args, kwargs) (line 51)
    all_call_result_460532 = invoke(stypy.reporting.localization.Localization(__file__, 51, 16), all_460524, *[result_eq_460530], **kwargs_460531)
    
    # Processing the call keyword arguments (line 51)
    kwargs_460533 = {}
    # Getting the type of 'assert_' (line 51)
    assert__460522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 51)
    assert__call_result_460534 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), assert__460522, *[all_call_result_460532], **kwargs_460533)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_threads(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_threads' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_460535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_460535)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_threads'
    return stypy_return_type_460535

# Assigning a type to the variable 'test_threads' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'test_threads', test_threads)

@norecursion
def test_regression_std_vector_dtypes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_regression_std_vector_dtypes'
    module_type_store = module_type_store.open_function_context('test_regression_std_vector_dtypes', 54, 0, False)
    
    # Passed parameters checking function
    test_regression_std_vector_dtypes.stypy_localization = localization
    test_regression_std_vector_dtypes.stypy_type_of_self = None
    test_regression_std_vector_dtypes.stypy_type_store = module_type_store
    test_regression_std_vector_dtypes.stypy_function_name = 'test_regression_std_vector_dtypes'
    test_regression_std_vector_dtypes.stypy_param_names_list = []
    test_regression_std_vector_dtypes.stypy_varargs_param_name = None
    test_regression_std_vector_dtypes.stypy_kwargs_param_name = None
    test_regression_std_vector_dtypes.stypy_call_defaults = defaults
    test_regression_std_vector_dtypes.stypy_call_varargs = varargs
    test_regression_std_vector_dtypes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_regression_std_vector_dtypes', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_regression_std_vector_dtypes', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_regression_std_vector_dtypes(...)' code ##################

    
    # Getting the type of 'supported_dtypes' (line 57)
    supported_dtypes_460536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 17), 'supported_dtypes')
    # Testing the type of a for loop iterable (line 57)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 57, 4), supported_dtypes_460536)
    # Getting the type of the for loop variable (line 57)
    for_loop_var_460537 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 57, 4), supported_dtypes_460536)
    # Assigning a type to the variable 'dtype' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'dtype', for_loop_var_460537)
    # SSA begins for a for statement (line 57)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 58):
    
    # Call to astype(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'dtype' (line 58)
    dtype_460550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 48), 'dtype', False)
    # Processing the call keyword arguments (line 58)
    kwargs_460551 = {}
    
    # Call to matrix(...): (line 58)
    # Processing the call arguments (line 58)
    
    # Obtaining an instance of the builtin type 'list' (line 58)
    list_460540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 58)
    # Adding element type (line 58)
    
    # Obtaining an instance of the builtin type 'list' (line 58)
    list_460541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 58)
    # Adding element type (line 58)
    int_460542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 24), list_460541, int_460542)
    # Adding element type (line 58)
    int_460543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 24), list_460541, int_460543)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 23), list_460540, list_460541)
    # Adding element type (line 58)
    
    # Obtaining an instance of the builtin type 'list' (line 58)
    list_460544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 58)
    # Adding element type (line 58)
    int_460545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 32), list_460544, int_460545)
    # Adding element type (line 58)
    int_460546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 32), list_460544, int_460546)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 23), list_460540, list_460544)
    
    # Processing the call keyword arguments (line 58)
    kwargs_460547 = {}
    # Getting the type of 'np' (line 58)
    np_460538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 13), 'np', False)
    # Obtaining the member 'matrix' of a type (line 58)
    matrix_460539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 13), np_460538, 'matrix')
    # Calling matrix(args, kwargs) (line 58)
    matrix_call_result_460548 = invoke(stypy.reporting.localization.Localization(__file__, 58, 13), matrix_460539, *[list_460540], **kwargs_460547)
    
    # Obtaining the member 'astype' of a type (line 58)
    astype_460549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 13), matrix_call_result_460548, 'astype')
    # Calling astype(args, kwargs) (line 58)
    astype_call_result_460552 = invoke(stypy.reporting.localization.Localization(__file__, 58, 13), astype_460549, *[dtype_460550], **kwargs_460551)
    
    # Assigning a type to the variable 'ad' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'ad', astype_call_result_460552)
    
    # Assigning a Call to a Name (line 59):
    
    # Call to csr_matrix(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'ad' (line 59)
    ad_460554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 23), 'ad', False)
    # Processing the call keyword arguments (line 59)
    # Getting the type of 'dtype' (line 59)
    dtype_460555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 33), 'dtype', False)
    keyword_460556 = dtype_460555
    kwargs_460557 = {'dtype': keyword_460556}
    # Getting the type of 'csr_matrix' (line 59)
    csr_matrix_460553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'csr_matrix', False)
    # Calling csr_matrix(args, kwargs) (line 59)
    csr_matrix_call_result_460558 = invoke(stypy.reporting.localization.Localization(__file__, 59, 12), csr_matrix_460553, *[ad_460554], **kwargs_460557)
    
    # Assigning a type to the variable 'a' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'a', csr_matrix_call_result_460558)
    
    # Call to assert_equal(...): (line 62)
    # Processing the call arguments (line 62)
    
    # Call to todense(...): (line 62)
    # Processing the call keyword arguments (line 62)
    kwargs_460566 = {}
    
    # Call to getcol(...): (line 62)
    # Processing the call arguments (line 62)
    int_460562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 30), 'int')
    # Processing the call keyword arguments (line 62)
    kwargs_460563 = {}
    # Getting the type of 'a' (line 62)
    a_460560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 21), 'a', False)
    # Obtaining the member 'getcol' of a type (line 62)
    getcol_460561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 21), a_460560, 'getcol')
    # Calling getcol(args, kwargs) (line 62)
    getcol_call_result_460564 = invoke(stypy.reporting.localization.Localization(__file__, 62, 21), getcol_460561, *[int_460562], **kwargs_460563)
    
    # Obtaining the member 'todense' of a type (line 62)
    todense_460565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 21), getcol_call_result_460564, 'todense')
    # Calling todense(args, kwargs) (line 62)
    todense_call_result_460567 = invoke(stypy.reporting.localization.Localization(__file__, 62, 21), todense_460565, *[], **kwargs_460566)
    
    
    # Obtaining the type of the subscript
    slice_460568 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 62, 44), None, None, None)
    int_460569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 49), 'int')
    # Getting the type of 'ad' (line 62)
    ad_460570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 44), 'ad', False)
    # Obtaining the member '__getitem__' of a type (line 62)
    getitem___460571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 44), ad_460570, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 62)
    subscript_call_result_460572 = invoke(stypy.reporting.localization.Localization(__file__, 62, 44), getitem___460571, (slice_460568, int_460569))
    
    # Processing the call keyword arguments (line 62)
    kwargs_460573 = {}
    # Getting the type of 'assert_equal' (line 62)
    assert_equal_460559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 62)
    assert_equal_call_result_460574 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), assert_equal_460559, *[todense_call_result_460567, subscript_call_result_460572], **kwargs_460573)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_regression_std_vector_dtypes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_regression_std_vector_dtypes' in the type store
    # Getting the type of 'stypy_return_type' (line 54)
    stypy_return_type_460575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_460575)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_regression_std_vector_dtypes'
    return stypy_return_type_460575

# Assigning a type to the variable 'test_regression_std_vector_dtypes' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'test_regression_std_vector_dtypes', test_regression_std_vector_dtypes)
# Declaration of the 'TestInt32Overflow' class

class TestInt32Overflow(object, ):
    str_460576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, (-1)), 'str', "\n    Some of the sparsetools routines use dense 2D matrices whose\n    total size is not bounded by the nnz of the sparse matrix. These\n    routines used to suffer from int32 wraparounds; here, we try to\n    check that the wraparounds don't occur any more.\n    ")

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 77, 4, False)
        # Assigning a type to the variable 'self' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestInt32Overflow.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestInt32Overflow.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestInt32Overflow.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestInt32Overflow.setup_method.__dict__.__setitem__('stypy_function_name', 'TestInt32Overflow.setup_method')
        TestInt32Overflow.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestInt32Overflow.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestInt32Overflow.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestInt32Overflow.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestInt32Overflow.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestInt32Overflow.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestInt32Overflow.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInt32Overflow.setup_method', [], None, None, defaults, varargs, kwargs)

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

        # Evaluating assert statement condition
        
        # Getting the type of 'self' (line 78)
        self_460577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 15), 'self')
        # Obtaining the member 'n' of a type (line 78)
        n_460578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 15), self_460577, 'n')
        int_460579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 23), 'int')
        # Applying the binary operator '**' (line 78)
        result_pow_460580 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 15), '**', n_460578, int_460579)
        
        
        # Call to iinfo(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'np' (line 78)
        np_460583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 36), 'np', False)
        # Obtaining the member 'int32' of a type (line 78)
        int32_460584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 36), np_460583, 'int32')
        # Processing the call keyword arguments (line 78)
        kwargs_460585 = {}
        # Getting the type of 'np' (line 78)
        np_460581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 27), 'np', False)
        # Obtaining the member 'iinfo' of a type (line 78)
        iinfo_460582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 27), np_460581, 'iinfo')
        # Calling iinfo(args, kwargs) (line 78)
        iinfo_call_result_460586 = invoke(stypy.reporting.localization.Localization(__file__, 78, 27), iinfo_460582, *[int32_460584], **kwargs_460585)
        
        # Obtaining the member 'max' of a type (line 78)
        max_460587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 27), iinfo_call_result_460586, 'max')
        # Applying the binary operator '>' (line 78)
        result_gt_460588 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 15), '>', result_pow_460580, max_460587)
        
        
        
        # SSA begins for try-except statement (line 82)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 83):
        
        # Call to int(...): (line 83)
        # Processing the call arguments (line 83)
        
        # Call to get(...): (line 83)
        # Processing the call arguments (line 83)
        str_460593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 48), 'str', 'PYTEST_XDIST_WORKER_COUNT')
        str_460594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 77), 'str', '1')
        # Processing the call keyword arguments (line 83)
        kwargs_460595 = {}
        # Getting the type of 'os' (line 83)
        os_460590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 33), 'os', False)
        # Obtaining the member 'environ' of a type (line 83)
        environ_460591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 33), os_460590, 'environ')
        # Obtaining the member 'get' of a type (line 83)
        get_460592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 33), environ_460591, 'get')
        # Calling get(args, kwargs) (line 83)
        get_call_result_460596 = invoke(stypy.reporting.localization.Localization(__file__, 83, 33), get_460592, *[str_460593, str_460594], **kwargs_460595)
        
        # Processing the call keyword arguments (line 83)
        kwargs_460597 = {}
        # Getting the type of 'int' (line 83)
        int_460589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 29), 'int', False)
        # Calling int(args, kwargs) (line 83)
        int_call_result_460598 = invoke(stypy.reporting.localization.Localization(__file__, 83, 29), int_460589, *[get_call_result_460596], **kwargs_460597)
        
        # Assigning a type to the variable 'parallel_count' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'parallel_count', int_call_result_460598)
        # SSA branch for the except part of a try statement (line 82)
        # SSA branch for the except 'ValueError' branch of a try statement (line 82)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Attribute to a Name (line 85):
        # Getting the type of 'np' (line 85)
        np_460599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 29), 'np')
        # Obtaining the member 'inf' of a type (line 85)
        inf_460600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 29), np_460599, 'inf')
        # Assigning a type to the variable 'parallel_count' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'parallel_count', inf_460600)
        # SSA join for try-except statement (line 82)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to check_free_memory(...): (line 87)
        # Processing the call arguments (line 87)
        int_460602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 26), 'int')
        # Getting the type of 'parallel_count' (line 87)
        parallel_count_460603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 33), 'parallel_count', False)
        # Applying the binary operator '*' (line 87)
        result_mul_460604 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 26), '*', int_460602, parallel_count_460603)
        
        # Processing the call keyword arguments (line 87)
        kwargs_460605 = {}
        # Getting the type of 'check_free_memory' (line 87)
        check_free_memory_460601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'check_free_memory', False)
        # Calling check_free_memory(args, kwargs) (line 87)
        check_free_memory_call_result_460606 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), check_free_memory_460601, *[result_mul_460604], **kwargs_460605)
        
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 77)
        stypy_return_type_460607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_460607)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_460607


    @norecursion
    def teardown_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'teardown_method'
        module_type_store = module_type_store.open_function_context('teardown_method', 89, 4, False)
        # Assigning a type to the variable 'self' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestInt32Overflow.teardown_method.__dict__.__setitem__('stypy_localization', localization)
        TestInt32Overflow.teardown_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestInt32Overflow.teardown_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestInt32Overflow.teardown_method.__dict__.__setitem__('stypy_function_name', 'TestInt32Overflow.teardown_method')
        TestInt32Overflow.teardown_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestInt32Overflow.teardown_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestInt32Overflow.teardown_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestInt32Overflow.teardown_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestInt32Overflow.teardown_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestInt32Overflow.teardown_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestInt32Overflow.teardown_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInt32Overflow.teardown_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'teardown_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'teardown_method(...)' code ##################

        
        # Call to collect(...): (line 90)
        # Processing the call keyword arguments (line 90)
        kwargs_460610 = {}
        # Getting the type of 'gc' (line 90)
        gc_460608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'gc', False)
        # Obtaining the member 'collect' of a type (line 90)
        collect_460609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), gc_460608, 'collect')
        # Calling collect(args, kwargs) (line 90)
        collect_call_result_460611 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), collect_460609, *[], **kwargs_460610)
        
        
        # ################# End of 'teardown_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'teardown_method' in the type store
        # Getting the type of 'stypy_return_type' (line 89)
        stypy_return_type_460612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_460612)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'teardown_method'
        return stypy_return_type_460612


    @norecursion
    def test_coo_todense(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_coo_todense'
        module_type_store = module_type_store.open_function_context('test_coo_todense', 92, 4, False)
        # Assigning a type to the variable 'self' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestInt32Overflow.test_coo_todense.__dict__.__setitem__('stypy_localization', localization)
        TestInt32Overflow.test_coo_todense.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestInt32Overflow.test_coo_todense.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestInt32Overflow.test_coo_todense.__dict__.__setitem__('stypy_function_name', 'TestInt32Overflow.test_coo_todense')
        TestInt32Overflow.test_coo_todense.__dict__.__setitem__('stypy_param_names_list', [])
        TestInt32Overflow.test_coo_todense.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestInt32Overflow.test_coo_todense.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestInt32Overflow.test_coo_todense.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestInt32Overflow.test_coo_todense.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestInt32Overflow.test_coo_todense.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestInt32Overflow.test_coo_todense.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInt32Overflow.test_coo_todense', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_coo_todense', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_coo_todense(...)' code ##################

        
        # Assigning a Attribute to a Name (line 97):
        # Getting the type of 'self' (line 97)
        self_460613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'self')
        # Obtaining the member 'n' of a type (line 97)
        n_460614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 12), self_460613, 'n')
        # Assigning a type to the variable 'n' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'n', n_460614)
        
        # Assigning a Call to a Name (line 99):
        
        # Call to array(...): (line 99)
        # Processing the call arguments (line 99)
        
        # Obtaining an instance of the builtin type 'list' (line 99)
        list_460617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 99)
        # Adding element type (line 99)
        int_460618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 21), list_460617, int_460618)
        # Adding element type (line 99)
        # Getting the type of 'n' (line 99)
        n_460619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 25), 'n', False)
        int_460620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 27), 'int')
        # Applying the binary operator '-' (line 99)
        result_sub_460621 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 25), '-', n_460619, int_460620)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 21), list_460617, result_sub_460621)
        
        # Processing the call keyword arguments (line 99)
        kwargs_460622 = {}
        # Getting the type of 'np' (line 99)
        np_460615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 99)
        array_460616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 12), np_460615, 'array')
        # Calling array(args, kwargs) (line 99)
        array_call_result_460623 = invoke(stypy.reporting.localization.Localization(__file__, 99, 12), array_460616, *[list_460617], **kwargs_460622)
        
        # Assigning a type to the variable 'i' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'i', array_call_result_460623)
        
        # Assigning a Call to a Name (line 100):
        
        # Call to array(...): (line 100)
        # Processing the call arguments (line 100)
        
        # Obtaining an instance of the builtin type 'list' (line 100)
        list_460626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 100)
        # Adding element type (line 100)
        int_460627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 21), list_460626, int_460627)
        # Adding element type (line 100)
        # Getting the type of 'n' (line 100)
        n_460628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 25), 'n', False)
        int_460629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 27), 'int')
        # Applying the binary operator '-' (line 100)
        result_sub_460630 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 25), '-', n_460628, int_460629)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 21), list_460626, result_sub_460630)
        
        # Processing the call keyword arguments (line 100)
        kwargs_460631 = {}
        # Getting the type of 'np' (line 100)
        np_460624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 100)
        array_460625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), np_460624, 'array')
        # Calling array(args, kwargs) (line 100)
        array_call_result_460632 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), array_460625, *[list_460626], **kwargs_460631)
        
        # Assigning a type to the variable 'j' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'j', array_call_result_460632)
        
        # Assigning a Call to a Name (line 101):
        
        # Call to array(...): (line 101)
        # Processing the call arguments (line 101)
        
        # Obtaining an instance of the builtin type 'list' (line 101)
        list_460635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 101)
        # Adding element type (line 101)
        int_460636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 24), list_460635, int_460636)
        # Adding element type (line 101)
        int_460637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 24), list_460635, int_460637)
        
        # Processing the call keyword arguments (line 101)
        # Getting the type of 'np' (line 101)
        np_460638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 38), 'np', False)
        # Obtaining the member 'int8' of a type (line 101)
        int8_460639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 38), np_460638, 'int8')
        keyword_460640 = int8_460639
        kwargs_460641 = {'dtype': keyword_460640}
        # Getting the type of 'np' (line 101)
        np_460633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 101)
        array_460634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 15), np_460633, 'array')
        # Calling array(args, kwargs) (line 101)
        array_call_result_460642 = invoke(stypy.reporting.localization.Localization(__file__, 101, 15), array_460634, *[list_460635], **kwargs_460641)
        
        # Assigning a type to the variable 'data' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'data', array_call_result_460642)
        
        # Assigning a Call to a Name (line 102):
        
        # Call to coo_matrix(...): (line 102)
        # Processing the call arguments (line 102)
        
        # Obtaining an instance of the builtin type 'tuple' (line 102)
        tuple_460644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 102)
        # Adding element type (line 102)
        # Getting the type of 'data' (line 102)
        data_460645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 24), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 24), tuple_460644, data_460645)
        # Adding element type (line 102)
        
        # Obtaining an instance of the builtin type 'tuple' (line 102)
        tuple_460646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 102)
        # Adding element type (line 102)
        # Getting the type of 'i' (line 102)
        i_460647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 31), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 31), tuple_460646, i_460647)
        # Adding element type (line 102)
        # Getting the type of 'j' (line 102)
        j_460648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 34), 'j', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 31), tuple_460646, j_460648)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 24), tuple_460644, tuple_460646)
        
        # Processing the call keyword arguments (line 102)
        kwargs_460649 = {}
        # Getting the type of 'coo_matrix' (line 102)
        coo_matrix_460643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'coo_matrix', False)
        # Calling coo_matrix(args, kwargs) (line 102)
        coo_matrix_call_result_460650 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), coo_matrix_460643, *[tuple_460644], **kwargs_460649)
        
        # Assigning a type to the variable 'm' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'm', coo_matrix_call_result_460650)
        
        # Assigning a Call to a Name (line 104):
        
        # Call to todense(...): (line 104)
        # Processing the call keyword arguments (line 104)
        kwargs_460653 = {}
        # Getting the type of 'm' (line 104)
        m_460651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'm', False)
        # Obtaining the member 'todense' of a type (line 104)
        todense_460652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 12), m_460651, 'todense')
        # Calling todense(args, kwargs) (line 104)
        todense_call_result_460654 = invoke(stypy.reporting.localization.Localization(__file__, 104, 12), todense_460652, *[], **kwargs_460653)
        
        # Assigning a type to the variable 'r' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'r', todense_call_result_460654)
        
        # Call to assert_equal(...): (line 105)
        # Processing the call arguments (line 105)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 105)
        tuple_460656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 105)
        # Adding element type (line 105)
        int_460657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 23), tuple_460656, int_460657)
        # Adding element type (line 105)
        int_460658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 23), tuple_460656, int_460658)
        
        # Getting the type of 'r' (line 105)
        r_460659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 21), 'r', False)
        # Obtaining the member '__getitem__' of a type (line 105)
        getitem___460660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 21), r_460659, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 105)
        subscript_call_result_460661 = invoke(stypy.reporting.localization.Localization(__file__, 105, 21), getitem___460660, tuple_460656)
        
        int_460662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 29), 'int')
        # Processing the call keyword arguments (line 105)
        kwargs_460663 = {}
        # Getting the type of 'assert_equal' (line 105)
        assert_equal_460655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 105)
        assert_equal_call_result_460664 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), assert_equal_460655, *[subscript_call_result_460661, int_460662], **kwargs_460663)
        
        
        # Call to assert_equal(...): (line 106)
        # Processing the call arguments (line 106)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 106)
        tuple_460666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 106)
        # Adding element type (line 106)
        int_460667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 23), tuple_460666, int_460667)
        # Adding element type (line 106)
        int_460668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 23), tuple_460666, int_460668)
        
        # Getting the type of 'r' (line 106)
        r_460669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 21), 'r', False)
        # Obtaining the member '__getitem__' of a type (line 106)
        getitem___460670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 21), r_460669, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 106)
        subscript_call_result_460671 = invoke(stypy.reporting.localization.Localization(__file__, 106, 21), getitem___460670, tuple_460666)
        
        int_460672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 31), 'int')
        # Processing the call keyword arguments (line 106)
        kwargs_460673 = {}
        # Getting the type of 'assert_equal' (line 106)
        assert_equal_460665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 106)
        assert_equal_call_result_460674 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), assert_equal_460665, *[subscript_call_result_460671, int_460672], **kwargs_460673)
        
        # Deleting a member
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 107, 8), module_type_store, 'r')
        
        # Call to collect(...): (line 108)
        # Processing the call keyword arguments (line 108)
        kwargs_460677 = {}
        # Getting the type of 'gc' (line 108)
        gc_460675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'gc', False)
        # Obtaining the member 'collect' of a type (line 108)
        collect_460676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), gc_460675, 'collect')
        # Calling collect(args, kwargs) (line 108)
        collect_call_result_460678 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), collect_460676, *[], **kwargs_460677)
        
        
        # ################# End of 'test_coo_todense(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_coo_todense' in the type store
        # Getting the type of 'stypy_return_type' (line 92)
        stypy_return_type_460679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_460679)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_coo_todense'
        return stypy_return_type_460679


    @norecursion
    def test_matvecs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_matvecs'
        module_type_store = module_type_store.open_function_context('test_matvecs', 110, 4, False)
        # Assigning a type to the variable 'self' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestInt32Overflow.test_matvecs.__dict__.__setitem__('stypy_localization', localization)
        TestInt32Overflow.test_matvecs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestInt32Overflow.test_matvecs.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestInt32Overflow.test_matvecs.__dict__.__setitem__('stypy_function_name', 'TestInt32Overflow.test_matvecs')
        TestInt32Overflow.test_matvecs.__dict__.__setitem__('stypy_param_names_list', [])
        TestInt32Overflow.test_matvecs.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestInt32Overflow.test_matvecs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestInt32Overflow.test_matvecs.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestInt32Overflow.test_matvecs.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestInt32Overflow.test_matvecs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestInt32Overflow.test_matvecs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInt32Overflow.test_matvecs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_matvecs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_matvecs(...)' code ##################

        
        # Assigning a Attribute to a Name (line 113):
        # Getting the type of 'self' (line 113)
        self_460680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'self')
        # Obtaining the member 'n' of a type (line 113)
        n_460681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 12), self_460680, 'n')
        # Assigning a type to the variable 'n' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'n', n_460681)
        
        # Assigning a Call to a Name (line 115):
        
        # Call to array(...): (line 115)
        # Processing the call arguments (line 115)
        
        # Obtaining an instance of the builtin type 'list' (line 115)
        list_460684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 115)
        # Adding element type (line 115)
        int_460685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 21), list_460684, int_460685)
        # Adding element type (line 115)
        # Getting the type of 'n' (line 115)
        n_460686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 25), 'n', False)
        int_460687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 27), 'int')
        # Applying the binary operator '-' (line 115)
        result_sub_460688 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 25), '-', n_460686, int_460687)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 21), list_460684, result_sub_460688)
        
        # Processing the call keyword arguments (line 115)
        kwargs_460689 = {}
        # Getting the type of 'np' (line 115)
        np_460682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 115)
        array_460683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), np_460682, 'array')
        # Calling array(args, kwargs) (line 115)
        array_call_result_460690 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), array_460683, *[list_460684], **kwargs_460689)
        
        # Assigning a type to the variable 'i' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'i', array_call_result_460690)
        
        # Assigning a Call to a Name (line 116):
        
        # Call to array(...): (line 116)
        # Processing the call arguments (line 116)
        
        # Obtaining an instance of the builtin type 'list' (line 116)
        list_460693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 116)
        # Adding element type (line 116)
        int_460694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 21), list_460693, int_460694)
        # Adding element type (line 116)
        # Getting the type of 'n' (line 116)
        n_460695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 25), 'n', False)
        int_460696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 27), 'int')
        # Applying the binary operator '-' (line 116)
        result_sub_460697 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 25), '-', n_460695, int_460696)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 21), list_460693, result_sub_460697)
        
        # Processing the call keyword arguments (line 116)
        kwargs_460698 = {}
        # Getting the type of 'np' (line 116)
        np_460691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 116)
        array_460692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 12), np_460691, 'array')
        # Calling array(args, kwargs) (line 116)
        array_call_result_460699 = invoke(stypy.reporting.localization.Localization(__file__, 116, 12), array_460692, *[list_460693], **kwargs_460698)
        
        # Assigning a type to the variable 'j' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'j', array_call_result_460699)
        
        # Assigning a Call to a Name (line 117):
        
        # Call to array(...): (line 117)
        # Processing the call arguments (line 117)
        
        # Obtaining an instance of the builtin type 'list' (line 117)
        list_460702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 117)
        # Adding element type (line 117)
        int_460703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 24), list_460702, int_460703)
        # Adding element type (line 117)
        int_460704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 24), list_460702, int_460704)
        
        # Processing the call keyword arguments (line 117)
        # Getting the type of 'np' (line 117)
        np_460705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 38), 'np', False)
        # Obtaining the member 'int8' of a type (line 117)
        int8_460706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 38), np_460705, 'int8')
        keyword_460707 = int8_460706
        kwargs_460708 = {'dtype': keyword_460707}
        # Getting the type of 'np' (line 117)
        np_460700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 117)
        array_460701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 15), np_460700, 'array')
        # Calling array(args, kwargs) (line 117)
        array_call_result_460709 = invoke(stypy.reporting.localization.Localization(__file__, 117, 15), array_460701, *[list_460702], **kwargs_460708)
        
        # Assigning a type to the variable 'data' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'data', array_call_result_460709)
        
        # Assigning a Call to a Name (line 118):
        
        # Call to coo_matrix(...): (line 118)
        # Processing the call arguments (line 118)
        
        # Obtaining an instance of the builtin type 'tuple' (line 118)
        tuple_460711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 118)
        # Adding element type (line 118)
        # Getting the type of 'data' (line 118)
        data_460712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 24), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 24), tuple_460711, data_460712)
        # Adding element type (line 118)
        
        # Obtaining an instance of the builtin type 'tuple' (line 118)
        tuple_460713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 118)
        # Adding element type (line 118)
        # Getting the type of 'i' (line 118)
        i_460714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 31), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 31), tuple_460713, i_460714)
        # Adding element type (line 118)
        # Getting the type of 'j' (line 118)
        j_460715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 34), 'j', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 31), tuple_460713, j_460715)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 24), tuple_460711, tuple_460713)
        
        # Processing the call keyword arguments (line 118)
        kwargs_460716 = {}
        # Getting the type of 'coo_matrix' (line 118)
        coo_matrix_460710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'coo_matrix', False)
        # Calling coo_matrix(args, kwargs) (line 118)
        coo_matrix_call_result_460717 = invoke(stypy.reporting.localization.Localization(__file__, 118, 12), coo_matrix_460710, *[tuple_460711], **kwargs_460716)
        
        # Assigning a type to the variable 'm' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'm', coo_matrix_call_result_460717)
        
        # Assigning a Call to a Name (line 120):
        
        # Call to ones(...): (line 120)
        # Processing the call arguments (line 120)
        
        # Obtaining an instance of the builtin type 'tuple' (line 120)
        tuple_460720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 120)
        # Adding element type (line 120)
        # Getting the type of 'n' (line 120)
        n_460721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 21), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 21), tuple_460720, n_460721)
        # Adding element type (line 120)
        # Getting the type of 'n' (line 120)
        n_460722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 24), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 21), tuple_460720, n_460722)
        
        # Processing the call keyword arguments (line 120)
        # Getting the type of 'np' (line 120)
        np_460723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 34), 'np', False)
        # Obtaining the member 'int8' of a type (line 120)
        int8_460724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 34), np_460723, 'int8')
        keyword_460725 = int8_460724
        kwargs_460726 = {'dtype': keyword_460725}
        # Getting the type of 'np' (line 120)
        np_460718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'np', False)
        # Obtaining the member 'ones' of a type (line 120)
        ones_460719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 12), np_460718, 'ones')
        # Calling ones(args, kwargs) (line 120)
        ones_call_result_460727 = invoke(stypy.reporting.localization.Localization(__file__, 120, 12), ones_460719, *[tuple_460720], **kwargs_460726)
        
        # Assigning a type to the variable 'b' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'b', ones_call_result_460727)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 121)
        tuple_460728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 121)
        # Adding element type (line 121)
        # Getting the type of 'csr_matrix' (line 121)
        csr_matrix_460729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'csr_matrix')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 23), tuple_460728, csr_matrix_460729)
        # Adding element type (line 121)
        # Getting the type of 'csc_matrix' (line 121)
        csc_matrix_460730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 35), 'csc_matrix')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 23), tuple_460728, csc_matrix_460730)
        # Adding element type (line 121)
        # Getting the type of 'bsr_matrix' (line 121)
        bsr_matrix_460731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 47), 'bsr_matrix')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 23), tuple_460728, bsr_matrix_460731)
        
        # Testing the type of a for loop iterable (line 121)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 121, 8), tuple_460728)
        # Getting the type of the for loop variable (line 121)
        for_loop_var_460732 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 121, 8), tuple_460728)
        # Assigning a type to the variable 'sptype' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'sptype', for_loop_var_460732)
        # SSA begins for a for statement (line 121)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 122):
        
        # Call to sptype(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'm' (line 122)
        m_460734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 24), 'm', False)
        # Processing the call keyword arguments (line 122)
        kwargs_460735 = {}
        # Getting the type of 'sptype' (line 122)
        sptype_460733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 17), 'sptype', False)
        # Calling sptype(args, kwargs) (line 122)
        sptype_call_result_460736 = invoke(stypy.reporting.localization.Localization(__file__, 122, 17), sptype_460733, *[m_460734], **kwargs_460735)
        
        # Assigning a type to the variable 'm2' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'm2', sptype_call_result_460736)
        
        # Assigning a Call to a Name (line 123):
        
        # Call to dot(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'b' (line 123)
        b_460739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 23), 'b', False)
        # Processing the call keyword arguments (line 123)
        kwargs_460740 = {}
        # Getting the type of 'm2' (line 123)
        m2_460737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'm2', False)
        # Obtaining the member 'dot' of a type (line 123)
        dot_460738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 16), m2_460737, 'dot')
        # Calling dot(args, kwargs) (line 123)
        dot_call_result_460741 = invoke(stypy.reporting.localization.Localization(__file__, 123, 16), dot_460738, *[b_460739], **kwargs_460740)
        
        # Assigning a type to the variable 'r' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'r', dot_call_result_460741)
        
        # Call to assert_equal(...): (line 124)
        # Processing the call arguments (line 124)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 124)
        tuple_460743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 124)
        # Adding element type (line 124)
        int_460744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 27), tuple_460743, int_460744)
        # Adding element type (line 124)
        int_460745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 27), tuple_460743, int_460745)
        
        # Getting the type of 'r' (line 124)
        r_460746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 25), 'r', False)
        # Obtaining the member '__getitem__' of a type (line 124)
        getitem___460747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 25), r_460746, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 124)
        subscript_call_result_460748 = invoke(stypy.reporting.localization.Localization(__file__, 124, 25), getitem___460747, tuple_460743)
        
        int_460749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 33), 'int')
        # Processing the call keyword arguments (line 124)
        kwargs_460750 = {}
        # Getting the type of 'assert_equal' (line 124)
        assert_equal_460742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 124)
        assert_equal_call_result_460751 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), assert_equal_460742, *[subscript_call_result_460748, int_460749], **kwargs_460750)
        
        
        # Call to assert_equal(...): (line 125)
        # Processing the call arguments (line 125)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 125)
        tuple_460753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 125)
        # Adding element type (line 125)
        int_460754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 27), tuple_460753, int_460754)
        # Adding element type (line 125)
        int_460755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 27), tuple_460753, int_460755)
        
        # Getting the type of 'r' (line 125)
        r_460756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 25), 'r', False)
        # Obtaining the member '__getitem__' of a type (line 125)
        getitem___460757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 25), r_460756, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 125)
        subscript_call_result_460758 = invoke(stypy.reporting.localization.Localization(__file__, 125, 25), getitem___460757, tuple_460753)
        
        int_460759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 35), 'int')
        # Processing the call keyword arguments (line 125)
        kwargs_460760 = {}
        # Getting the type of 'assert_equal' (line 125)
        assert_equal_460752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 125)
        assert_equal_call_result_460761 = invoke(stypy.reporting.localization.Localization(__file__, 125, 12), assert_equal_460752, *[subscript_call_result_460758, int_460759], **kwargs_460760)
        
        # Deleting a member
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 126, 12), module_type_store, 'r')
        
        # Call to collect(...): (line 127)
        # Processing the call keyword arguments (line 127)
        kwargs_460764 = {}
        # Getting the type of 'gc' (line 127)
        gc_460762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'gc', False)
        # Obtaining the member 'collect' of a type (line 127)
        collect_460763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), gc_460762, 'collect')
        # Calling collect(args, kwargs) (line 127)
        collect_call_result_460765 = invoke(stypy.reporting.localization.Localization(__file__, 127, 12), collect_460763, *[], **kwargs_460764)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Deleting a member
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 129, 8), module_type_store, 'b')
        
        # Call to collect(...): (line 130)
        # Processing the call keyword arguments (line 130)
        kwargs_460768 = {}
        # Getting the type of 'gc' (line 130)
        gc_460766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'gc', False)
        # Obtaining the member 'collect' of a type (line 130)
        collect_460767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), gc_460766, 'collect')
        # Calling collect(args, kwargs) (line 130)
        collect_call_result_460769 = invoke(stypy.reporting.localization.Localization(__file__, 130, 8), collect_460767, *[], **kwargs_460768)
        
        
        # ################# End of 'test_matvecs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_matvecs' in the type store
        # Getting the type of 'stypy_return_type' (line 110)
        stypy_return_type_460770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_460770)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_matvecs'
        return stypy_return_type_460770


    @norecursion
    def test_dia_matvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dia_matvec'
        module_type_store = module_type_store.open_function_context('test_dia_matvec', 132, 4, False)
        # Assigning a type to the variable 'self' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestInt32Overflow.test_dia_matvec.__dict__.__setitem__('stypy_localization', localization)
        TestInt32Overflow.test_dia_matvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestInt32Overflow.test_dia_matvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestInt32Overflow.test_dia_matvec.__dict__.__setitem__('stypy_function_name', 'TestInt32Overflow.test_dia_matvec')
        TestInt32Overflow.test_dia_matvec.__dict__.__setitem__('stypy_param_names_list', [])
        TestInt32Overflow.test_dia_matvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestInt32Overflow.test_dia_matvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestInt32Overflow.test_dia_matvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestInt32Overflow.test_dia_matvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestInt32Overflow.test_dia_matvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestInt32Overflow.test_dia_matvec.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInt32Overflow.test_dia_matvec', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dia_matvec', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dia_matvec(...)' code ##################

        
        # Assigning a Attribute to a Name (line 135):
        # Getting the type of 'self' (line 135)
        self_460771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'self')
        # Obtaining the member 'n' of a type (line 135)
        n_460772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 12), self_460771, 'n')
        # Assigning a type to the variable 'n' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'n', n_460772)
        
        # Assigning a Call to a Name (line 136):
        
        # Call to ones(...): (line 136)
        # Processing the call arguments (line 136)
        
        # Obtaining an instance of the builtin type 'tuple' (line 136)
        tuple_460775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 136)
        # Adding element type (line 136)
        # Getting the type of 'n' (line 136)
        n_460776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 24), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 24), tuple_460775, n_460776)
        # Adding element type (line 136)
        # Getting the type of 'n' (line 136)
        n_460777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 27), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 24), tuple_460775, n_460777)
        
        # Processing the call keyword arguments (line 136)
        # Getting the type of 'np' (line 136)
        np_460778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 37), 'np', False)
        # Obtaining the member 'int8' of a type (line 136)
        int8_460779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 37), np_460778, 'int8')
        keyword_460780 = int8_460779
        kwargs_460781 = {'dtype': keyword_460780}
        # Getting the type of 'np' (line 136)
        np_460773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 15), 'np', False)
        # Obtaining the member 'ones' of a type (line 136)
        ones_460774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 15), np_460773, 'ones')
        # Calling ones(args, kwargs) (line 136)
        ones_call_result_460782 = invoke(stypy.reporting.localization.Localization(__file__, 136, 15), ones_460774, *[tuple_460775], **kwargs_460781)
        
        # Assigning a type to the variable 'data' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'data', ones_call_result_460782)
        
        # Assigning a Call to a Name (line 137):
        
        # Call to arange(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'n' (line 137)
        n_460785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 28), 'n', False)
        # Processing the call keyword arguments (line 137)
        kwargs_460786 = {}
        # Getting the type of 'np' (line 137)
        np_460783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 18), 'np', False)
        # Obtaining the member 'arange' of a type (line 137)
        arange_460784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 18), np_460783, 'arange')
        # Calling arange(args, kwargs) (line 137)
        arange_call_result_460787 = invoke(stypy.reporting.localization.Localization(__file__, 137, 18), arange_460784, *[n_460785], **kwargs_460786)
        
        # Assigning a type to the variable 'offsets' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'offsets', arange_call_result_460787)
        
        # Assigning a Call to a Name (line 138):
        
        # Call to dia_matrix(...): (line 138)
        # Processing the call arguments (line 138)
        
        # Obtaining an instance of the builtin type 'tuple' (line 138)
        tuple_460789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 138)
        # Adding element type (line 138)
        # Getting the type of 'data' (line 138)
        data_460790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 24), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 24), tuple_460789, data_460790)
        # Adding element type (line 138)
        # Getting the type of 'offsets' (line 138)
        offsets_460791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 30), 'offsets', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 24), tuple_460789, offsets_460791)
        
        # Processing the call keyword arguments (line 138)
        
        # Obtaining an instance of the builtin type 'tuple' (line 138)
        tuple_460792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 138)
        # Adding element type (line 138)
        # Getting the type of 'n' (line 138)
        n_460793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 47), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 47), tuple_460792, n_460793)
        # Adding element type (line 138)
        # Getting the type of 'n' (line 138)
        n_460794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 50), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 47), tuple_460792, n_460794)
        
        keyword_460795 = tuple_460792
        kwargs_460796 = {'shape': keyword_460795}
        # Getting the type of 'dia_matrix' (line 138)
        dia_matrix_460788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'dia_matrix', False)
        # Calling dia_matrix(args, kwargs) (line 138)
        dia_matrix_call_result_460797 = invoke(stypy.reporting.localization.Localization(__file__, 138, 12), dia_matrix_460788, *[tuple_460789], **kwargs_460796)
        
        # Assigning a type to the variable 'm' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'm', dia_matrix_call_result_460797)
        
        # Assigning a Call to a Name (line 139):
        
        # Call to ones(...): (line 139)
        # Processing the call arguments (line 139)
        
        # Obtaining the type of the subscript
        int_460800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 28), 'int')
        # Getting the type of 'm' (line 139)
        m_460801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 20), 'm', False)
        # Obtaining the member 'shape' of a type (line 139)
        shape_460802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 20), m_460801, 'shape')
        # Obtaining the member '__getitem__' of a type (line 139)
        getitem___460803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 20), shape_460802, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 139)
        subscript_call_result_460804 = invoke(stypy.reporting.localization.Localization(__file__, 139, 20), getitem___460803, int_460800)
        
        # Processing the call keyword arguments (line 139)
        # Getting the type of 'np' (line 139)
        np_460805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 38), 'np', False)
        # Obtaining the member 'int8' of a type (line 139)
        int8_460806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 38), np_460805, 'int8')
        keyword_460807 = int8_460806
        kwargs_460808 = {'dtype': keyword_460807}
        # Getting the type of 'np' (line 139)
        np_460798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'np', False)
        # Obtaining the member 'ones' of a type (line 139)
        ones_460799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 12), np_460798, 'ones')
        # Calling ones(args, kwargs) (line 139)
        ones_call_result_460809 = invoke(stypy.reporting.localization.Localization(__file__, 139, 12), ones_460799, *[subscript_call_result_460804], **kwargs_460808)
        
        # Assigning a type to the variable 'v' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'v', ones_call_result_460809)
        
        # Assigning a Call to a Name (line 140):
        
        # Call to dot(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'v' (line 140)
        v_460812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 18), 'v', False)
        # Processing the call keyword arguments (line 140)
        kwargs_460813 = {}
        # Getting the type of 'm' (line 140)
        m_460810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'm', False)
        # Obtaining the member 'dot' of a type (line 140)
        dot_460811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 12), m_460810, 'dot')
        # Calling dot(args, kwargs) (line 140)
        dot_call_result_460814 = invoke(stypy.reporting.localization.Localization(__file__, 140, 12), dot_460811, *[v_460812], **kwargs_460813)
        
        # Assigning a type to the variable 'r' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'r', dot_call_result_460814)
        
        # Call to assert_equal(...): (line 141)
        # Processing the call arguments (line 141)
        
        # Obtaining the type of the subscript
        int_460816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 23), 'int')
        # Getting the type of 'r' (line 141)
        r_460817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 21), 'r', False)
        # Obtaining the member '__getitem__' of a type (line 141)
        getitem___460818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 21), r_460817, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 141)
        subscript_call_result_460819 = invoke(stypy.reporting.localization.Localization(__file__, 141, 21), getitem___460818, int_460816)
        
        
        # Call to int8(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'n' (line 141)
        n_460822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 35), 'n', False)
        # Processing the call keyword arguments (line 141)
        kwargs_460823 = {}
        # Getting the type of 'np' (line 141)
        np_460820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 27), 'np', False)
        # Obtaining the member 'int8' of a type (line 141)
        int8_460821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 27), np_460820, 'int8')
        # Calling int8(args, kwargs) (line 141)
        int8_call_result_460824 = invoke(stypy.reporting.localization.Localization(__file__, 141, 27), int8_460821, *[n_460822], **kwargs_460823)
        
        # Processing the call keyword arguments (line 141)
        kwargs_460825 = {}
        # Getting the type of 'assert_equal' (line 141)
        assert_equal_460815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 141)
        assert_equal_call_result_460826 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), assert_equal_460815, *[subscript_call_result_460819, int8_call_result_460824], **kwargs_460825)
        
        # Deleting a member
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 142, 8), module_type_store, 'data')
        # Deleting a member
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 142, 8), module_type_store, 'offsets')
        # Deleting a member
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 142, 8), module_type_store, 'm')
        # Deleting a member
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 142, 8), module_type_store, 'v')
        # Deleting a member
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 142, 8), module_type_store, 'r')
        
        # Call to collect(...): (line 143)
        # Processing the call keyword arguments (line 143)
        kwargs_460829 = {}
        # Getting the type of 'gc' (line 143)
        gc_460827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'gc', False)
        # Obtaining the member 'collect' of a type (line 143)
        collect_460828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), gc_460827, 'collect')
        # Calling collect(args, kwargs) (line 143)
        collect_call_result_460830 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), collect_460828, *[], **kwargs_460829)
        
        
        # ################# End of 'test_dia_matvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dia_matvec' in the type store
        # Getting the type of 'stypy_return_type' (line 132)
        stypy_return_type_460831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_460831)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dia_matvec'
        return stypy_return_type_460831


    @norecursion
    def test_bsr_1_block(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bsr_1_block'
        module_type_store = module_type_store.open_function_context('test_bsr_1_block', 152, 4, False)
        # Assigning a type to the variable 'self' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestInt32Overflow.test_bsr_1_block.__dict__.__setitem__('stypy_localization', localization)
        TestInt32Overflow.test_bsr_1_block.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestInt32Overflow.test_bsr_1_block.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestInt32Overflow.test_bsr_1_block.__dict__.__setitem__('stypy_function_name', 'TestInt32Overflow.test_bsr_1_block')
        TestInt32Overflow.test_bsr_1_block.__dict__.__setitem__('stypy_param_names_list', ['op'])
        TestInt32Overflow.test_bsr_1_block.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestInt32Overflow.test_bsr_1_block.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestInt32Overflow.test_bsr_1_block.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestInt32Overflow.test_bsr_1_block.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestInt32Overflow.test_bsr_1_block.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestInt32Overflow.test_bsr_1_block.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInt32Overflow.test_bsr_1_block', ['op'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bsr_1_block', localization, ['op'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bsr_1_block(...)' code ##################


        @norecursion
        def get_matrix(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'get_matrix'
            module_type_store = module_type_store.open_function_context('get_matrix', 159, 8, False)
            
            # Passed parameters checking function
            get_matrix.stypy_localization = localization
            get_matrix.stypy_type_of_self = None
            get_matrix.stypy_type_store = module_type_store
            get_matrix.stypy_function_name = 'get_matrix'
            get_matrix.stypy_param_names_list = []
            get_matrix.stypy_varargs_param_name = None
            get_matrix.stypy_kwargs_param_name = None
            get_matrix.stypy_call_defaults = defaults
            get_matrix.stypy_call_varargs = varargs
            get_matrix.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'get_matrix', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'get_matrix', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'get_matrix(...)' code ##################

            
            # Assigning a Attribute to a Name (line 160):
            # Getting the type of 'self' (line 160)
            self_460832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 16), 'self')
            # Obtaining the member 'n' of a type (line 160)
            n_460833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 16), self_460832, 'n')
            # Assigning a type to the variable 'n' (line 160)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'n', n_460833)
            
            # Assigning a Call to a Name (line 161):
            
            # Call to ones(...): (line 161)
            # Processing the call arguments (line 161)
            
            # Obtaining an instance of the builtin type 'tuple' (line 161)
            tuple_460836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 28), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 161)
            # Adding element type (line 161)
            int_460837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 28), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 28), tuple_460836, int_460837)
            # Adding element type (line 161)
            # Getting the type of 'n' (line 161)
            n_460838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 31), 'n', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 28), tuple_460836, n_460838)
            # Adding element type (line 161)
            # Getting the type of 'n' (line 161)
            n_460839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 34), 'n', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 28), tuple_460836, n_460839)
            
            # Processing the call keyword arguments (line 161)
            # Getting the type of 'np' (line 161)
            np_460840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 44), 'np', False)
            # Obtaining the member 'int8' of a type (line 161)
            int8_460841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 44), np_460840, 'int8')
            keyword_460842 = int8_460841
            kwargs_460843 = {'dtype': keyword_460842}
            # Getting the type of 'np' (line 161)
            np_460834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 19), 'np', False)
            # Obtaining the member 'ones' of a type (line 161)
            ones_460835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 19), np_460834, 'ones')
            # Calling ones(args, kwargs) (line 161)
            ones_call_result_460844 = invoke(stypy.reporting.localization.Localization(__file__, 161, 19), ones_460835, *[tuple_460836], **kwargs_460843)
            
            # Assigning a type to the variable 'data' (line 161)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'data', ones_call_result_460844)
            
            # Assigning a Call to a Name (line 162):
            
            # Call to array(...): (line 162)
            # Processing the call arguments (line 162)
            
            # Obtaining an instance of the builtin type 'list' (line 162)
            list_460847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 30), 'list')
            # Adding type elements to the builtin type 'list' instance (line 162)
            # Adding element type (line 162)
            int_460848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 31), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 30), list_460847, int_460848)
            # Adding element type (line 162)
            int_460849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 34), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 30), list_460847, int_460849)
            
            # Processing the call keyword arguments (line 162)
            # Getting the type of 'np' (line 162)
            np_460850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 44), 'np', False)
            # Obtaining the member 'int32' of a type (line 162)
            int32_460851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 44), np_460850, 'int32')
            keyword_460852 = int32_460851
            kwargs_460853 = {'dtype': keyword_460852}
            # Getting the type of 'np' (line 162)
            np_460845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 21), 'np', False)
            # Obtaining the member 'array' of a type (line 162)
            array_460846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 21), np_460845, 'array')
            # Calling array(args, kwargs) (line 162)
            array_call_result_460854 = invoke(stypy.reporting.localization.Localization(__file__, 162, 21), array_460846, *[list_460847], **kwargs_460853)
            
            # Assigning a type to the variable 'indptr' (line 162)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'indptr', array_call_result_460854)
            
            # Assigning a Call to a Name (line 163):
            
            # Call to array(...): (line 163)
            # Processing the call arguments (line 163)
            
            # Obtaining an instance of the builtin type 'list' (line 163)
            list_460857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 31), 'list')
            # Adding type elements to the builtin type 'list' instance (line 163)
            # Adding element type (line 163)
            int_460858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 32), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 31), list_460857, int_460858)
            
            # Processing the call keyword arguments (line 163)
            # Getting the type of 'np' (line 163)
            np_460859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 42), 'np', False)
            # Obtaining the member 'int32' of a type (line 163)
            int32_460860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 42), np_460859, 'int32')
            keyword_460861 = int32_460860
            kwargs_460862 = {'dtype': keyword_460861}
            # Getting the type of 'np' (line 163)
            np_460855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 22), 'np', False)
            # Obtaining the member 'array' of a type (line 163)
            array_460856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 22), np_460855, 'array')
            # Calling array(args, kwargs) (line 163)
            array_call_result_460863 = invoke(stypy.reporting.localization.Localization(__file__, 163, 22), array_460856, *[list_460857], **kwargs_460862)
            
            # Assigning a type to the variable 'indices' (line 163)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'indices', array_call_result_460863)
            
            # Assigning a Call to a Name (line 164):
            
            # Call to bsr_matrix(...): (line 164)
            # Processing the call arguments (line 164)
            
            # Obtaining an instance of the builtin type 'tuple' (line 164)
            tuple_460865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 28), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 164)
            # Adding element type (line 164)
            # Getting the type of 'data' (line 164)
            data_460866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'data', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 28), tuple_460865, data_460866)
            # Adding element type (line 164)
            # Getting the type of 'indices' (line 164)
            indices_460867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 34), 'indices', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 28), tuple_460865, indices_460867)
            # Adding element type (line 164)
            # Getting the type of 'indptr' (line 164)
            indptr_460868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 43), 'indptr', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 28), tuple_460865, indptr_460868)
            
            # Processing the call keyword arguments (line 164)
            
            # Obtaining an instance of the builtin type 'tuple' (line 164)
            tuple_460869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 63), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 164)
            # Adding element type (line 164)
            # Getting the type of 'n' (line 164)
            n_460870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 63), 'n', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 63), tuple_460869, n_460870)
            # Adding element type (line 164)
            # Getting the type of 'n' (line 164)
            n_460871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 66), 'n', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 63), tuple_460869, n_460871)
            
            keyword_460872 = tuple_460869
            # Getting the type of 'False' (line 164)
            False_460873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 75), 'False', False)
            keyword_460874 = False_460873
            kwargs_460875 = {'blocksize': keyword_460872, 'copy': keyword_460874}
            # Getting the type of 'bsr_matrix' (line 164)
            bsr_matrix_460864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'bsr_matrix', False)
            # Calling bsr_matrix(args, kwargs) (line 164)
            bsr_matrix_call_result_460876 = invoke(stypy.reporting.localization.Localization(__file__, 164, 16), bsr_matrix_460864, *[tuple_460865], **kwargs_460875)
            
            # Assigning a type to the variable 'm' (line 164)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'm', bsr_matrix_call_result_460876)
            # Deleting a member
            module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 165, 12), module_type_store, 'data')
            # Deleting a member
            module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 165, 12), module_type_store, 'indptr')
            # Deleting a member
            module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 165, 12), module_type_store, 'indices')
            # Getting the type of 'm' (line 166)
            m_460877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 19), 'm')
            # Assigning a type to the variable 'stypy_return_type' (line 166)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'stypy_return_type', m_460877)
            
            # ################# End of 'get_matrix(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'get_matrix' in the type store
            # Getting the type of 'stypy_return_type' (line 159)
            stypy_return_type_460878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_460878)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'get_matrix'
            return stypy_return_type_460878

        # Assigning a type to the variable 'get_matrix' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'get_matrix', get_matrix)
        
        # Call to collect(...): (line 168)
        # Processing the call keyword arguments (line 168)
        kwargs_460881 = {}
        # Getting the type of 'gc' (line 168)
        gc_460879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'gc', False)
        # Obtaining the member 'collect' of a type (line 168)
        collect_460880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), gc_460879, 'collect')
        # Calling collect(args, kwargs) (line 168)
        collect_call_result_460882 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), collect_460880, *[], **kwargs_460881)
        
        
        # Try-finally block (line 169)
        
        # Call to (...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'get_matrix' (line 170)
        get_matrix_460890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 46), 'get_matrix', False)
        # Processing the call keyword arguments (line 170)
        kwargs_460891 = {}
        
        # Call to getattr(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'self' (line 170)
        self_460884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 20), 'self', False)
        str_460885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 26), 'str', '_check_bsr_')
        # Getting the type of 'op' (line 170)
        op_460886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 42), 'op', False)
        # Applying the binary operator '+' (line 170)
        result_add_460887 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 26), '+', str_460885, op_460886)
        
        # Processing the call keyword arguments (line 170)
        kwargs_460888 = {}
        # Getting the type of 'getattr' (line 170)
        getattr_460883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'getattr', False)
        # Calling getattr(args, kwargs) (line 170)
        getattr_call_result_460889 = invoke(stypy.reporting.localization.Localization(__file__, 170, 12), getattr_460883, *[self_460884, result_add_460887], **kwargs_460888)
        
        # Calling (args, kwargs) (line 170)
        _call_result_460892 = invoke(stypy.reporting.localization.Localization(__file__, 170, 12), getattr_call_result_460889, *[get_matrix_460890], **kwargs_460891)
        
        
        # finally branch of the try-finally block (line 169)
        
        # Call to collect(...): (line 172)
        # Processing the call keyword arguments (line 172)
        kwargs_460895 = {}
        # Getting the type of 'gc' (line 172)
        gc_460893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'gc', False)
        # Obtaining the member 'collect' of a type (line 172)
        collect_460894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 12), gc_460893, 'collect')
        # Calling collect(args, kwargs) (line 172)
        collect_call_result_460896 = invoke(stypy.reporting.localization.Localization(__file__, 172, 12), collect_460894, *[], **kwargs_460895)
        
        
        
        # ################# End of 'test_bsr_1_block(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bsr_1_block' in the type store
        # Getting the type of 'stypy_return_type' (line 152)
        stypy_return_type_460897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_460897)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bsr_1_block'
        return stypy_return_type_460897


    @norecursion
    def test_bsr_n_block(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bsr_n_block'
        module_type_store = module_type_store.open_function_context('test_bsr_n_block', 174, 4, False)
        # Assigning a type to the variable 'self' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestInt32Overflow.test_bsr_n_block.__dict__.__setitem__('stypy_localization', localization)
        TestInt32Overflow.test_bsr_n_block.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestInt32Overflow.test_bsr_n_block.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestInt32Overflow.test_bsr_n_block.__dict__.__setitem__('stypy_function_name', 'TestInt32Overflow.test_bsr_n_block')
        TestInt32Overflow.test_bsr_n_block.__dict__.__setitem__('stypy_param_names_list', ['op'])
        TestInt32Overflow.test_bsr_n_block.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestInt32Overflow.test_bsr_n_block.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestInt32Overflow.test_bsr_n_block.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestInt32Overflow.test_bsr_n_block.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestInt32Overflow.test_bsr_n_block.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestInt32Overflow.test_bsr_n_block.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInt32Overflow.test_bsr_n_block', ['op'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bsr_n_block', localization, ['op'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bsr_n_block(...)' code ##################


        @norecursion
        def get_matrix(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'get_matrix'
            module_type_store = module_type_store.open_function_context('get_matrix', 182, 8, False)
            
            # Passed parameters checking function
            get_matrix.stypy_localization = localization
            get_matrix.stypy_type_of_self = None
            get_matrix.stypy_type_store = module_type_store
            get_matrix.stypy_function_name = 'get_matrix'
            get_matrix.stypy_param_names_list = []
            get_matrix.stypy_varargs_param_name = None
            get_matrix.stypy_kwargs_param_name = None
            get_matrix.stypy_call_defaults = defaults
            get_matrix.stypy_call_varargs = varargs
            get_matrix.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'get_matrix', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'get_matrix', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'get_matrix(...)' code ##################

            
            # Assigning a Attribute to a Name (line 183):
            # Getting the type of 'self' (line 183)
            self_460898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'self')
            # Obtaining the member 'n' of a type (line 183)
            n_460899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 16), self_460898, 'n')
            # Assigning a type to the variable 'n' (line 183)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'n', n_460899)
            
            # Assigning a Call to a Name (line 184):
            
            # Call to ones(...): (line 184)
            # Processing the call arguments (line 184)
            
            # Obtaining an instance of the builtin type 'tuple' (line 184)
            tuple_460902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 28), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 184)
            # Adding element type (line 184)
            # Getting the type of 'n' (line 184)
            n_460903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 28), 'n', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 28), tuple_460902, n_460903)
            # Adding element type (line 184)
            # Getting the type of 'n' (line 184)
            n_460904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 31), 'n', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 28), tuple_460902, n_460904)
            # Adding element type (line 184)
            int_460905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 34), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 28), tuple_460902, int_460905)
            
            # Processing the call keyword arguments (line 184)
            # Getting the type of 'np' (line 184)
            np_460906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 44), 'np', False)
            # Obtaining the member 'int8' of a type (line 184)
            int8_460907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 44), np_460906, 'int8')
            keyword_460908 = int8_460907
            kwargs_460909 = {'dtype': keyword_460908}
            # Getting the type of 'np' (line 184)
            np_460900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 19), 'np', False)
            # Obtaining the member 'ones' of a type (line 184)
            ones_460901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 19), np_460900, 'ones')
            # Calling ones(args, kwargs) (line 184)
            ones_call_result_460910 = invoke(stypy.reporting.localization.Localization(__file__, 184, 19), ones_460901, *[tuple_460902], **kwargs_460909)
            
            # Assigning a type to the variable 'data' (line 184)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'data', ones_call_result_460910)
            
            # Assigning a Call to a Name (line 185):
            
            # Call to array(...): (line 185)
            # Processing the call arguments (line 185)
            
            # Obtaining an instance of the builtin type 'list' (line 185)
            list_460913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 30), 'list')
            # Adding type elements to the builtin type 'list' instance (line 185)
            # Adding element type (line 185)
            int_460914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 31), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 30), list_460913, int_460914)
            # Adding element type (line 185)
            # Getting the type of 'n' (line 185)
            n_460915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 34), 'n', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 30), list_460913, n_460915)
            
            # Processing the call keyword arguments (line 185)
            # Getting the type of 'np' (line 185)
            np_460916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 44), 'np', False)
            # Obtaining the member 'int32' of a type (line 185)
            int32_460917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 44), np_460916, 'int32')
            keyword_460918 = int32_460917
            kwargs_460919 = {'dtype': keyword_460918}
            # Getting the type of 'np' (line 185)
            np_460911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 21), 'np', False)
            # Obtaining the member 'array' of a type (line 185)
            array_460912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 21), np_460911, 'array')
            # Calling array(args, kwargs) (line 185)
            array_call_result_460920 = invoke(stypy.reporting.localization.Localization(__file__, 185, 21), array_460912, *[list_460913], **kwargs_460919)
            
            # Assigning a type to the variable 'indptr' (line 185)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'indptr', array_call_result_460920)
            
            # Assigning a Call to a Name (line 186):
            
            # Call to arange(...): (line 186)
            # Processing the call arguments (line 186)
            # Getting the type of 'n' (line 186)
            n_460923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 32), 'n', False)
            # Processing the call keyword arguments (line 186)
            # Getting the type of 'np' (line 186)
            np_460924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 41), 'np', False)
            # Obtaining the member 'int32' of a type (line 186)
            int32_460925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 41), np_460924, 'int32')
            keyword_460926 = int32_460925
            kwargs_460927 = {'dtype': keyword_460926}
            # Getting the type of 'np' (line 186)
            np_460921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 22), 'np', False)
            # Obtaining the member 'arange' of a type (line 186)
            arange_460922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 22), np_460921, 'arange')
            # Calling arange(args, kwargs) (line 186)
            arange_call_result_460928 = invoke(stypy.reporting.localization.Localization(__file__, 186, 22), arange_460922, *[n_460923], **kwargs_460927)
            
            # Assigning a type to the variable 'indices' (line 186)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'indices', arange_call_result_460928)
            
            # Assigning a Call to a Name (line 187):
            
            # Call to bsr_matrix(...): (line 187)
            # Processing the call arguments (line 187)
            
            # Obtaining an instance of the builtin type 'tuple' (line 187)
            tuple_460930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 28), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 187)
            # Adding element type (line 187)
            # Getting the type of 'data' (line 187)
            data_460931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 28), 'data', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 28), tuple_460930, data_460931)
            # Adding element type (line 187)
            # Getting the type of 'indices' (line 187)
            indices_460932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 34), 'indices', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 28), tuple_460930, indices_460932)
            # Adding element type (line 187)
            # Getting the type of 'indptr' (line 187)
            indptr_460933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 43), 'indptr', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 28), tuple_460930, indptr_460933)
            
            # Processing the call keyword arguments (line 187)
            
            # Obtaining an instance of the builtin type 'tuple' (line 187)
            tuple_460934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 63), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 187)
            # Adding element type (line 187)
            # Getting the type of 'n' (line 187)
            n_460935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 63), 'n', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 63), tuple_460934, n_460935)
            # Adding element type (line 187)
            int_460936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 66), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 63), tuple_460934, int_460936)
            
            keyword_460937 = tuple_460934
            # Getting the type of 'False' (line 187)
            False_460938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 75), 'False', False)
            keyword_460939 = False_460938
            kwargs_460940 = {'blocksize': keyword_460937, 'copy': keyword_460939}
            # Getting the type of 'bsr_matrix' (line 187)
            bsr_matrix_460929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 16), 'bsr_matrix', False)
            # Calling bsr_matrix(args, kwargs) (line 187)
            bsr_matrix_call_result_460941 = invoke(stypy.reporting.localization.Localization(__file__, 187, 16), bsr_matrix_460929, *[tuple_460930], **kwargs_460940)
            
            # Assigning a type to the variable 'm' (line 187)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'm', bsr_matrix_call_result_460941)
            # Deleting a member
            module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 188, 12), module_type_store, 'data')
            # Deleting a member
            module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 188, 12), module_type_store, 'indptr')
            # Deleting a member
            module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 188, 12), module_type_store, 'indices')
            # Getting the type of 'm' (line 189)
            m_460942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 19), 'm')
            # Assigning a type to the variable 'stypy_return_type' (line 189)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'stypy_return_type', m_460942)
            
            # ################# End of 'get_matrix(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'get_matrix' in the type store
            # Getting the type of 'stypy_return_type' (line 182)
            stypy_return_type_460943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_460943)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'get_matrix'
            return stypy_return_type_460943

        # Assigning a type to the variable 'get_matrix' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'get_matrix', get_matrix)
        
        # Call to collect(...): (line 191)
        # Processing the call keyword arguments (line 191)
        kwargs_460946 = {}
        # Getting the type of 'gc' (line 191)
        gc_460944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'gc', False)
        # Obtaining the member 'collect' of a type (line 191)
        collect_460945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), gc_460944, 'collect')
        # Calling collect(args, kwargs) (line 191)
        collect_call_result_460947 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), collect_460945, *[], **kwargs_460946)
        
        
        # Try-finally block (line 192)
        
        # Call to (...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'get_matrix' (line 193)
        get_matrix_460955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 46), 'get_matrix', False)
        # Processing the call keyword arguments (line 193)
        kwargs_460956 = {}
        
        # Call to getattr(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'self' (line 193)
        self_460949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 20), 'self', False)
        str_460950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 26), 'str', '_check_bsr_')
        # Getting the type of 'op' (line 193)
        op_460951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 42), 'op', False)
        # Applying the binary operator '+' (line 193)
        result_add_460952 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 26), '+', str_460950, op_460951)
        
        # Processing the call keyword arguments (line 193)
        kwargs_460953 = {}
        # Getting the type of 'getattr' (line 193)
        getattr_460948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'getattr', False)
        # Calling getattr(args, kwargs) (line 193)
        getattr_call_result_460954 = invoke(stypy.reporting.localization.Localization(__file__, 193, 12), getattr_460948, *[self_460949, result_add_460952], **kwargs_460953)
        
        # Calling (args, kwargs) (line 193)
        _call_result_460957 = invoke(stypy.reporting.localization.Localization(__file__, 193, 12), getattr_call_result_460954, *[get_matrix_460955], **kwargs_460956)
        
        
        # finally branch of the try-finally block (line 192)
        
        # Call to collect(...): (line 195)
        # Processing the call keyword arguments (line 195)
        kwargs_460960 = {}
        # Getting the type of 'gc' (line 195)
        gc_460958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'gc', False)
        # Obtaining the member 'collect' of a type (line 195)
        collect_460959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), gc_460958, 'collect')
        # Calling collect(args, kwargs) (line 195)
        collect_call_result_460961 = invoke(stypy.reporting.localization.Localization(__file__, 195, 12), collect_460959, *[], **kwargs_460960)
        
        
        
        # ################# End of 'test_bsr_n_block(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bsr_n_block' in the type store
        # Getting the type of 'stypy_return_type' (line 174)
        stypy_return_type_460962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_460962)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bsr_n_block'
        return stypy_return_type_460962


    @norecursion
    def _check_bsr_matvecs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check_bsr_matvecs'
        module_type_store = module_type_store.open_function_context('_check_bsr_matvecs', 197, 4, False)
        # Assigning a type to the variable 'self' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestInt32Overflow._check_bsr_matvecs.__dict__.__setitem__('stypy_localization', localization)
        TestInt32Overflow._check_bsr_matvecs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestInt32Overflow._check_bsr_matvecs.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestInt32Overflow._check_bsr_matvecs.__dict__.__setitem__('stypy_function_name', 'TestInt32Overflow._check_bsr_matvecs')
        TestInt32Overflow._check_bsr_matvecs.__dict__.__setitem__('stypy_param_names_list', ['m'])
        TestInt32Overflow._check_bsr_matvecs.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestInt32Overflow._check_bsr_matvecs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestInt32Overflow._check_bsr_matvecs.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestInt32Overflow._check_bsr_matvecs.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestInt32Overflow._check_bsr_matvecs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestInt32Overflow._check_bsr_matvecs.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInt32Overflow._check_bsr_matvecs', ['m'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_bsr_matvecs', localization, ['m'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_bsr_matvecs(...)' code ##################

        
        # Assigning a Call to a Name (line 198):
        
        # Call to m(...): (line 198)
        # Processing the call keyword arguments (line 198)
        kwargs_460964 = {}
        # Getting the type of 'm' (line 198)
        m_460963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'm', False)
        # Calling m(args, kwargs) (line 198)
        m_call_result_460965 = invoke(stypy.reporting.localization.Localization(__file__, 198, 12), m_460963, *[], **kwargs_460964)
        
        # Assigning a type to the variable 'm' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'm', m_call_result_460965)
        
        # Assigning a Attribute to a Name (line 199):
        # Getting the type of 'self' (line 199)
        self_460966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'self')
        # Obtaining the member 'n' of a type (line 199)
        n_460967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 12), self_460966, 'n')
        # Assigning a type to the variable 'n' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'n', n_460967)
        
        # Assigning a Call to a Name (line 202):
        
        # Call to dot(...): (line 202)
        # Processing the call arguments (line 202)
        
        # Call to ones(...): (line 202)
        # Processing the call arguments (line 202)
        
        # Obtaining an instance of the builtin type 'tuple' (line 202)
        tuple_460972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 202)
        # Adding element type (line 202)
        # Getting the type of 'n' (line 202)
        n_460973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 27), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 27), tuple_460972, n_460973)
        # Adding element type (line 202)
        int_460974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 27), tuple_460972, int_460974)
        
        # Processing the call keyword arguments (line 202)
        # Getting the type of 'np' (line 202)
        np_460975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 40), 'np', False)
        # Obtaining the member 'int8' of a type (line 202)
        int8_460976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 40), np_460975, 'int8')
        keyword_460977 = int8_460976
        kwargs_460978 = {'dtype': keyword_460977}
        # Getting the type of 'np' (line 202)
        np_460970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 18), 'np', False)
        # Obtaining the member 'ones' of a type (line 202)
        ones_460971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 18), np_460970, 'ones')
        # Calling ones(args, kwargs) (line 202)
        ones_call_result_460979 = invoke(stypy.reporting.localization.Localization(__file__, 202, 18), ones_460971, *[tuple_460972], **kwargs_460978)
        
        # Processing the call keyword arguments (line 202)
        kwargs_460980 = {}
        # Getting the type of 'm' (line 202)
        m_460968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'm', False)
        # Obtaining the member 'dot' of a type (line 202)
        dot_460969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 12), m_460968, 'dot')
        # Calling dot(args, kwargs) (line 202)
        dot_call_result_460981 = invoke(stypy.reporting.localization.Localization(__file__, 202, 12), dot_460969, *[ones_call_result_460979], **kwargs_460980)
        
        # Assigning a type to the variable 'r' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'r', dot_call_result_460981)
        
        # Call to assert_equal(...): (line 203)
        # Processing the call arguments (line 203)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 203)
        tuple_460983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 203)
        # Adding element type (line 203)
        int_460984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 23), tuple_460983, int_460984)
        # Adding element type (line 203)
        int_460985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 23), tuple_460983, int_460985)
        
        # Getting the type of 'r' (line 203)
        r_460986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 21), 'r', False)
        # Obtaining the member '__getitem__' of a type (line 203)
        getitem___460987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 21), r_460986, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 203)
        subscript_call_result_460988 = invoke(stypy.reporting.localization.Localization(__file__, 203, 21), getitem___460987, tuple_460983)
        
        
        # Call to int8(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'n' (line 203)
        n_460991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 37), 'n', False)
        # Processing the call keyword arguments (line 203)
        kwargs_460992 = {}
        # Getting the type of 'np' (line 203)
        np_460989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 29), 'np', False)
        # Obtaining the member 'int8' of a type (line 203)
        int8_460990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 29), np_460989, 'int8')
        # Calling int8(args, kwargs) (line 203)
        int8_call_result_460993 = invoke(stypy.reporting.localization.Localization(__file__, 203, 29), int8_460990, *[n_460991], **kwargs_460992)
        
        # Processing the call keyword arguments (line 203)
        kwargs_460994 = {}
        # Getting the type of 'assert_equal' (line 203)
        assert_equal_460982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 203)
        assert_equal_call_result_460995 = invoke(stypy.reporting.localization.Localization(__file__, 203, 8), assert_equal_460982, *[subscript_call_result_460988, int8_call_result_460993], **kwargs_460994)
        
        
        # ################# End of '_check_bsr_matvecs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_bsr_matvecs' in the type store
        # Getting the type of 'stypy_return_type' (line 197)
        stypy_return_type_460996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_460996)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_bsr_matvecs'
        return stypy_return_type_460996


    @norecursion
    def _check_bsr_matvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check_bsr_matvec'
        module_type_store = module_type_store.open_function_context('_check_bsr_matvec', 205, 4, False)
        # Assigning a type to the variable 'self' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestInt32Overflow._check_bsr_matvec.__dict__.__setitem__('stypy_localization', localization)
        TestInt32Overflow._check_bsr_matvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestInt32Overflow._check_bsr_matvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestInt32Overflow._check_bsr_matvec.__dict__.__setitem__('stypy_function_name', 'TestInt32Overflow._check_bsr_matvec')
        TestInt32Overflow._check_bsr_matvec.__dict__.__setitem__('stypy_param_names_list', ['m'])
        TestInt32Overflow._check_bsr_matvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestInt32Overflow._check_bsr_matvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestInt32Overflow._check_bsr_matvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestInt32Overflow._check_bsr_matvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestInt32Overflow._check_bsr_matvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestInt32Overflow._check_bsr_matvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInt32Overflow._check_bsr_matvec', ['m'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_bsr_matvec', localization, ['m'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_bsr_matvec(...)' code ##################

        
        # Assigning a Call to a Name (line 206):
        
        # Call to m(...): (line 206)
        # Processing the call keyword arguments (line 206)
        kwargs_460998 = {}
        # Getting the type of 'm' (line 206)
        m_460997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'm', False)
        # Calling m(args, kwargs) (line 206)
        m_call_result_460999 = invoke(stypy.reporting.localization.Localization(__file__, 206, 12), m_460997, *[], **kwargs_460998)
        
        # Assigning a type to the variable 'm' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'm', m_call_result_460999)
        
        # Assigning a Attribute to a Name (line 207):
        # Getting the type of 'self' (line 207)
        self_461000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'self')
        # Obtaining the member 'n' of a type (line 207)
        n_461001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 12), self_461000, 'n')
        # Assigning a type to the variable 'n' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'n', n_461001)
        
        # Assigning a Call to a Name (line 210):
        
        # Call to dot(...): (line 210)
        # Processing the call arguments (line 210)
        
        # Call to ones(...): (line 210)
        # Processing the call arguments (line 210)
        
        # Obtaining an instance of the builtin type 'tuple' (line 210)
        tuple_461006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 210)
        # Adding element type (line 210)
        # Getting the type of 'n' (line 210)
        n_461007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 27), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 27), tuple_461006, n_461007)
        
        # Processing the call keyword arguments (line 210)
        # Getting the type of 'np' (line 210)
        np_461008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 38), 'np', False)
        # Obtaining the member 'int8' of a type (line 210)
        int8_461009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 38), np_461008, 'int8')
        keyword_461010 = int8_461009
        kwargs_461011 = {'dtype': keyword_461010}
        # Getting the type of 'np' (line 210)
        np_461004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 18), 'np', False)
        # Obtaining the member 'ones' of a type (line 210)
        ones_461005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 18), np_461004, 'ones')
        # Calling ones(args, kwargs) (line 210)
        ones_call_result_461012 = invoke(stypy.reporting.localization.Localization(__file__, 210, 18), ones_461005, *[tuple_461006], **kwargs_461011)
        
        # Processing the call keyword arguments (line 210)
        kwargs_461013 = {}
        # Getting the type of 'm' (line 210)
        m_461002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'm', False)
        # Obtaining the member 'dot' of a type (line 210)
        dot_461003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 12), m_461002, 'dot')
        # Calling dot(args, kwargs) (line 210)
        dot_call_result_461014 = invoke(stypy.reporting.localization.Localization(__file__, 210, 12), dot_461003, *[ones_call_result_461012], **kwargs_461013)
        
        # Assigning a type to the variable 'r' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'r', dot_call_result_461014)
        
        # Call to assert_equal(...): (line 211)
        # Processing the call arguments (line 211)
        
        # Obtaining the type of the subscript
        int_461016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 23), 'int')
        # Getting the type of 'r' (line 211)
        r_461017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 21), 'r', False)
        # Obtaining the member '__getitem__' of a type (line 211)
        getitem___461018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 21), r_461017, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 211)
        subscript_call_result_461019 = invoke(stypy.reporting.localization.Localization(__file__, 211, 21), getitem___461018, int_461016)
        
        
        # Call to int8(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'n' (line 211)
        n_461022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 35), 'n', False)
        # Processing the call keyword arguments (line 211)
        kwargs_461023 = {}
        # Getting the type of 'np' (line 211)
        np_461020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 27), 'np', False)
        # Obtaining the member 'int8' of a type (line 211)
        int8_461021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 27), np_461020, 'int8')
        # Calling int8(args, kwargs) (line 211)
        int8_call_result_461024 = invoke(stypy.reporting.localization.Localization(__file__, 211, 27), int8_461021, *[n_461022], **kwargs_461023)
        
        # Processing the call keyword arguments (line 211)
        kwargs_461025 = {}
        # Getting the type of 'assert_equal' (line 211)
        assert_equal_461015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 211)
        assert_equal_call_result_461026 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), assert_equal_461015, *[subscript_call_result_461019, int8_call_result_461024], **kwargs_461025)
        
        
        # ################# End of '_check_bsr_matvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_bsr_matvec' in the type store
        # Getting the type of 'stypy_return_type' (line 205)
        stypy_return_type_461027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_461027)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_bsr_matvec'
        return stypy_return_type_461027


    @norecursion
    def _check_bsr_diagonal(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check_bsr_diagonal'
        module_type_store = module_type_store.open_function_context('_check_bsr_diagonal', 213, 4, False)
        # Assigning a type to the variable 'self' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestInt32Overflow._check_bsr_diagonal.__dict__.__setitem__('stypy_localization', localization)
        TestInt32Overflow._check_bsr_diagonal.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestInt32Overflow._check_bsr_diagonal.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestInt32Overflow._check_bsr_diagonal.__dict__.__setitem__('stypy_function_name', 'TestInt32Overflow._check_bsr_diagonal')
        TestInt32Overflow._check_bsr_diagonal.__dict__.__setitem__('stypy_param_names_list', ['m'])
        TestInt32Overflow._check_bsr_diagonal.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestInt32Overflow._check_bsr_diagonal.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestInt32Overflow._check_bsr_diagonal.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestInt32Overflow._check_bsr_diagonal.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestInt32Overflow._check_bsr_diagonal.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestInt32Overflow._check_bsr_diagonal.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInt32Overflow._check_bsr_diagonal', ['m'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_bsr_diagonal', localization, ['m'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_bsr_diagonal(...)' code ##################

        
        # Assigning a Call to a Name (line 214):
        
        # Call to m(...): (line 214)
        # Processing the call keyword arguments (line 214)
        kwargs_461029 = {}
        # Getting the type of 'm' (line 214)
        m_461028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'm', False)
        # Calling m(args, kwargs) (line 214)
        m_call_result_461030 = invoke(stypy.reporting.localization.Localization(__file__, 214, 12), m_461028, *[], **kwargs_461029)
        
        # Assigning a type to the variable 'm' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'm', m_call_result_461030)
        
        # Assigning a Attribute to a Name (line 215):
        # Getting the type of 'self' (line 215)
        self_461031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'self')
        # Obtaining the member 'n' of a type (line 215)
        n_461032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 12), self_461031, 'n')
        # Assigning a type to the variable 'n' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'n', n_461032)
        
        # Assigning a Call to a Name (line 218):
        
        # Call to diagonal(...): (line 218)
        # Processing the call keyword arguments (line 218)
        kwargs_461035 = {}
        # Getting the type of 'm' (line 218)
        m_461033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'm', False)
        # Obtaining the member 'diagonal' of a type (line 218)
        diagonal_461034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 12), m_461033, 'diagonal')
        # Calling diagonal(args, kwargs) (line 218)
        diagonal_call_result_461036 = invoke(stypy.reporting.localization.Localization(__file__, 218, 12), diagonal_461034, *[], **kwargs_461035)
        
        # Assigning a type to the variable 'r' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'r', diagonal_call_result_461036)
        
        # Call to assert_equal(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'r' (line 219)
        r_461038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 21), 'r', False)
        
        # Call to ones(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'n' (line 219)
        n_461041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 32), 'n', False)
        # Processing the call keyword arguments (line 219)
        kwargs_461042 = {}
        # Getting the type of 'np' (line 219)
        np_461039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 24), 'np', False)
        # Obtaining the member 'ones' of a type (line 219)
        ones_461040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 24), np_461039, 'ones')
        # Calling ones(args, kwargs) (line 219)
        ones_call_result_461043 = invoke(stypy.reporting.localization.Localization(__file__, 219, 24), ones_461040, *[n_461041], **kwargs_461042)
        
        # Processing the call keyword arguments (line 219)
        kwargs_461044 = {}
        # Getting the type of 'assert_equal' (line 219)
        assert_equal_461037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 219)
        assert_equal_call_result_461045 = invoke(stypy.reporting.localization.Localization(__file__, 219, 8), assert_equal_461037, *[r_461038, ones_call_result_461043], **kwargs_461044)
        
        
        # ################# End of '_check_bsr_diagonal(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_bsr_diagonal' in the type store
        # Getting the type of 'stypy_return_type' (line 213)
        stypy_return_type_461046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_461046)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_bsr_diagonal'
        return stypy_return_type_461046


    @norecursion
    def _check_bsr_sort_indices(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check_bsr_sort_indices'
        module_type_store = module_type_store.open_function_context('_check_bsr_sort_indices', 221, 4, False)
        # Assigning a type to the variable 'self' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestInt32Overflow._check_bsr_sort_indices.__dict__.__setitem__('stypy_localization', localization)
        TestInt32Overflow._check_bsr_sort_indices.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestInt32Overflow._check_bsr_sort_indices.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestInt32Overflow._check_bsr_sort_indices.__dict__.__setitem__('stypy_function_name', 'TestInt32Overflow._check_bsr_sort_indices')
        TestInt32Overflow._check_bsr_sort_indices.__dict__.__setitem__('stypy_param_names_list', ['m'])
        TestInt32Overflow._check_bsr_sort_indices.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestInt32Overflow._check_bsr_sort_indices.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestInt32Overflow._check_bsr_sort_indices.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestInt32Overflow._check_bsr_sort_indices.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestInt32Overflow._check_bsr_sort_indices.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestInt32Overflow._check_bsr_sort_indices.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInt32Overflow._check_bsr_sort_indices', ['m'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_bsr_sort_indices', localization, ['m'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_bsr_sort_indices(...)' code ##################

        
        # Assigning a Call to a Name (line 223):
        
        # Call to m(...): (line 223)
        # Processing the call keyword arguments (line 223)
        kwargs_461048 = {}
        # Getting the type of 'm' (line 223)
        m_461047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'm', False)
        # Calling m(args, kwargs) (line 223)
        m_call_result_461049 = invoke(stypy.reporting.localization.Localization(__file__, 223, 12), m_461047, *[], **kwargs_461048)
        
        # Assigning a type to the variable 'm' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'm', m_call_result_461049)
        
        # Call to sort_indices(...): (line 224)
        # Processing the call keyword arguments (line 224)
        kwargs_461052 = {}
        # Getting the type of 'm' (line 224)
        m_461050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'm', False)
        # Obtaining the member 'sort_indices' of a type (line 224)
        sort_indices_461051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), m_461050, 'sort_indices')
        # Calling sort_indices(args, kwargs) (line 224)
        sort_indices_call_result_461053 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), sort_indices_461051, *[], **kwargs_461052)
        
        
        # ################# End of '_check_bsr_sort_indices(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_bsr_sort_indices' in the type store
        # Getting the type of 'stypy_return_type' (line 221)
        stypy_return_type_461054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_461054)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_bsr_sort_indices'
        return stypy_return_type_461054


    @norecursion
    def _check_bsr_transpose(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check_bsr_transpose'
        module_type_store = module_type_store.open_function_context('_check_bsr_transpose', 226, 4, False)
        # Assigning a type to the variable 'self' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestInt32Overflow._check_bsr_transpose.__dict__.__setitem__('stypy_localization', localization)
        TestInt32Overflow._check_bsr_transpose.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestInt32Overflow._check_bsr_transpose.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestInt32Overflow._check_bsr_transpose.__dict__.__setitem__('stypy_function_name', 'TestInt32Overflow._check_bsr_transpose')
        TestInt32Overflow._check_bsr_transpose.__dict__.__setitem__('stypy_param_names_list', ['m'])
        TestInt32Overflow._check_bsr_transpose.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestInt32Overflow._check_bsr_transpose.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestInt32Overflow._check_bsr_transpose.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestInt32Overflow._check_bsr_transpose.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestInt32Overflow._check_bsr_transpose.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestInt32Overflow._check_bsr_transpose.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInt32Overflow._check_bsr_transpose', ['m'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_bsr_transpose', localization, ['m'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_bsr_transpose(...)' code ##################

        
        # Assigning a Call to a Name (line 228):
        
        # Call to m(...): (line 228)
        # Processing the call keyword arguments (line 228)
        kwargs_461056 = {}
        # Getting the type of 'm' (line 228)
        m_461055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'm', False)
        # Calling m(args, kwargs) (line 228)
        m_call_result_461057 = invoke(stypy.reporting.localization.Localization(__file__, 228, 12), m_461055, *[], **kwargs_461056)
        
        # Assigning a type to the variable 'm' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'm', m_call_result_461057)
        
        # Call to transpose(...): (line 229)
        # Processing the call keyword arguments (line 229)
        kwargs_461060 = {}
        # Getting the type of 'm' (line 229)
        m_461058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'm', False)
        # Obtaining the member 'transpose' of a type (line 229)
        transpose_461059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), m_461058, 'transpose')
        # Calling transpose(args, kwargs) (line 229)
        transpose_call_result_461061 = invoke(stypy.reporting.localization.Localization(__file__, 229, 8), transpose_461059, *[], **kwargs_461060)
        
        
        # ################# End of '_check_bsr_transpose(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_bsr_transpose' in the type store
        # Getting the type of 'stypy_return_type' (line 226)
        stypy_return_type_461062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_461062)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_bsr_transpose'
        return stypy_return_type_461062


    @norecursion
    def _check_bsr_matmat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check_bsr_matmat'
        module_type_store = module_type_store.open_function_context('_check_bsr_matmat', 231, 4, False)
        # Assigning a type to the variable 'self' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestInt32Overflow._check_bsr_matmat.__dict__.__setitem__('stypy_localization', localization)
        TestInt32Overflow._check_bsr_matmat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestInt32Overflow._check_bsr_matmat.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestInt32Overflow._check_bsr_matmat.__dict__.__setitem__('stypy_function_name', 'TestInt32Overflow._check_bsr_matmat')
        TestInt32Overflow._check_bsr_matmat.__dict__.__setitem__('stypy_param_names_list', ['m'])
        TestInt32Overflow._check_bsr_matmat.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestInt32Overflow._check_bsr_matmat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestInt32Overflow._check_bsr_matmat.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestInt32Overflow._check_bsr_matmat.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestInt32Overflow._check_bsr_matmat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestInt32Overflow._check_bsr_matmat.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInt32Overflow._check_bsr_matmat', ['m'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_bsr_matmat', localization, ['m'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_bsr_matmat(...)' code ##################

        
        # Assigning a Call to a Name (line 232):
        
        # Call to m(...): (line 232)
        # Processing the call keyword arguments (line 232)
        kwargs_461064 = {}
        # Getting the type of 'm' (line 232)
        m_461063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'm', False)
        # Calling m(args, kwargs) (line 232)
        m_call_result_461065 = invoke(stypy.reporting.localization.Localization(__file__, 232, 12), m_461063, *[], **kwargs_461064)
        
        # Assigning a type to the variable 'm' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'm', m_call_result_461065)
        
        # Assigning a Attribute to a Name (line 233):
        # Getting the type of 'self' (line 233)
        self_461066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'self')
        # Obtaining the member 'n' of a type (line 233)
        n_461067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 12), self_461066, 'n')
        # Assigning a type to the variable 'n' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'n', n_461067)
        
        # Assigning a Call to a Name (line 236):
        
        # Call to bsr_matrix(...): (line 236)
        # Processing the call arguments (line 236)
        
        # Call to ones(...): (line 236)
        # Processing the call arguments (line 236)
        
        # Obtaining an instance of the builtin type 'tuple' (line 236)
        tuple_461071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 236)
        # Adding element type (line 236)
        # Getting the type of 'n' (line 236)
        n_461072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 33), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 33), tuple_461071, n_461072)
        # Adding element type (line 236)
        int_461073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 33), tuple_461071, int_461073)
        
        # Processing the call keyword arguments (line 236)
        # Getting the type of 'np' (line 236)
        np_461074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 46), 'np', False)
        # Obtaining the member 'int8' of a type (line 236)
        int8_461075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 46), np_461074, 'int8')
        keyword_461076 = int8_461075
        kwargs_461077 = {'dtype': keyword_461076}
        # Getting the type of 'np' (line 236)
        np_461069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 24), 'np', False)
        # Obtaining the member 'ones' of a type (line 236)
        ones_461070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 24), np_461069, 'ones')
        # Calling ones(args, kwargs) (line 236)
        ones_call_result_461078 = invoke(stypy.reporting.localization.Localization(__file__, 236, 24), ones_461070, *[tuple_461071], **kwargs_461077)
        
        # Processing the call keyword arguments (line 236)
        
        # Obtaining an instance of the builtin type 'tuple' (line 236)
        tuple_461079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 67), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 236)
        # Adding element type (line 236)
        
        # Obtaining the type of the subscript
        int_461080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 79), 'int')
        # Getting the type of 'm' (line 236)
        m_461081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 67), 'm', False)
        # Obtaining the member 'blocksize' of a type (line 236)
        blocksize_461082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 67), m_461081, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 236)
        getitem___461083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 67), blocksize_461082, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 236)
        subscript_call_result_461084 = invoke(stypy.reporting.localization.Localization(__file__, 236, 67), getitem___461083, int_461080)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 67), tuple_461079, subscript_call_result_461084)
        # Adding element type (line 236)
        int_461085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 83), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 67), tuple_461079, int_461085)
        
        keyword_461086 = tuple_461079
        kwargs_461087 = {'blocksize': keyword_461086}
        # Getting the type of 'bsr_matrix' (line 236)
        bsr_matrix_461068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 13), 'bsr_matrix', False)
        # Calling bsr_matrix(args, kwargs) (line 236)
        bsr_matrix_call_result_461088 = invoke(stypy.reporting.localization.Localization(__file__, 236, 13), bsr_matrix_461068, *[ones_call_result_461078], **kwargs_461087)
        
        # Assigning a type to the variable 'm2' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'm2', bsr_matrix_call_result_461088)
        
        # Call to dot(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'm2' (line 237)
        m2_461091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 14), 'm2', False)
        # Processing the call keyword arguments (line 237)
        kwargs_461092 = {}
        # Getting the type of 'm' (line 237)
        m_461089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'm', False)
        # Obtaining the member 'dot' of a type (line 237)
        dot_461090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), m_461089, 'dot')
        # Calling dot(args, kwargs) (line 237)
        dot_call_result_461093 = invoke(stypy.reporting.localization.Localization(__file__, 237, 8), dot_461090, *[m2_461091], **kwargs_461092)
        
        # Deleting a member
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 238, 8), module_type_store, 'm2')
        
        # Assigning a Call to a Name (line 241):
        
        # Call to bsr_matrix(...): (line 241)
        # Processing the call arguments (line 241)
        
        # Call to ones(...): (line 241)
        # Processing the call arguments (line 241)
        
        # Obtaining an instance of the builtin type 'tuple' (line 241)
        tuple_461097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 241)
        # Adding element type (line 241)
        int_461098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 33), tuple_461097, int_461098)
        # Adding element type (line 241)
        # Getting the type of 'n' (line 241)
        n_461099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 36), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 33), tuple_461097, n_461099)
        
        # Processing the call keyword arguments (line 241)
        # Getting the type of 'np' (line 241)
        np_461100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 46), 'np', False)
        # Obtaining the member 'int8' of a type (line 241)
        int8_461101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 46), np_461100, 'int8')
        keyword_461102 = int8_461101
        kwargs_461103 = {'dtype': keyword_461102}
        # Getting the type of 'np' (line 241)
        np_461095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 24), 'np', False)
        # Obtaining the member 'ones' of a type (line 241)
        ones_461096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 24), np_461095, 'ones')
        # Calling ones(args, kwargs) (line 241)
        ones_call_result_461104 = invoke(stypy.reporting.localization.Localization(__file__, 241, 24), ones_461096, *[tuple_461097], **kwargs_461103)
        
        # Processing the call keyword arguments (line 241)
        
        # Obtaining an instance of the builtin type 'tuple' (line 241)
        tuple_461105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 67), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 241)
        # Adding element type (line 241)
        int_461106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 67), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 67), tuple_461105, int_461106)
        # Adding element type (line 241)
        
        # Obtaining the type of the subscript
        int_461107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 82), 'int')
        # Getting the type of 'm' (line 241)
        m_461108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 70), 'm', False)
        # Obtaining the member 'blocksize' of a type (line 241)
        blocksize_461109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 70), m_461108, 'blocksize')
        # Obtaining the member '__getitem__' of a type (line 241)
        getitem___461110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 70), blocksize_461109, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 241)
        subscript_call_result_461111 = invoke(stypy.reporting.localization.Localization(__file__, 241, 70), getitem___461110, int_461107)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 67), tuple_461105, subscript_call_result_461111)
        
        keyword_461112 = tuple_461105
        kwargs_461113 = {'blocksize': keyword_461112}
        # Getting the type of 'bsr_matrix' (line 241)
        bsr_matrix_461094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 13), 'bsr_matrix', False)
        # Calling bsr_matrix(args, kwargs) (line 241)
        bsr_matrix_call_result_461114 = invoke(stypy.reporting.localization.Localization(__file__, 241, 13), bsr_matrix_461094, *[ones_call_result_461104], **kwargs_461113)
        
        # Assigning a type to the variable 'm2' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'm2', bsr_matrix_call_result_461114)
        
        # Call to dot(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'm' (line 242)
        m_461117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 15), 'm', False)
        # Processing the call keyword arguments (line 242)
        kwargs_461118 = {}
        # Getting the type of 'm2' (line 242)
        m2_461115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'm2', False)
        # Obtaining the member 'dot' of a type (line 242)
        dot_461116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 8), m2_461115, 'dot')
        # Calling dot(args, kwargs) (line 242)
        dot_call_result_461119 = invoke(stypy.reporting.localization.Localization(__file__, 242, 8), dot_461116, *[m_461117], **kwargs_461118)
        
        
        # ################# End of '_check_bsr_matmat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_bsr_matmat' in the type store
        # Getting the type of 'stypy_return_type' (line 231)
        stypy_return_type_461120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_461120)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_bsr_matmat'
        return stypy_return_type_461120


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 65, 0, False)
        # Assigning a type to the variable 'self' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInt32Overflow.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestInt32Overflow' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'TestInt32Overflow', TestInt32Overflow)

# Assigning a Num to a Name (line 75):
int_461121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 8), 'int')
# Getting the type of 'TestInt32Overflow'
TestInt32Overflow_461122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestInt32Overflow')
# Setting the type of the member 'n' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestInt32Overflow_461122, 'n', int_461121)

# Assigning a List to a Name (line 145):

# Obtaining an instance of the builtin type 'list' (line 145)
list_461123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 145)
# Adding element type (line 145)

# Call to param(...): (line 145)
# Processing the call arguments (line 145)
str_461126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 29), 'str', 'matmat')
# Processing the call keyword arguments (line 145)
# Getting the type of 'pytest' (line 145)
pytest_461127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 45), 'pytest', False)
# Obtaining the member 'mark' of a type (line 145)
mark_461128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 45), pytest_461127, 'mark')
# Obtaining the member 'xslow' of a type (line 145)
xslow_461129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 45), mark_461128, 'xslow')
keyword_461130 = xslow_461129
kwargs_461131 = {'marks': keyword_461130}
# Getting the type of 'pytest' (line 145)
pytest_461124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'pytest', False)
# Obtaining the member 'param' of a type (line 145)
param_461125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 16), pytest_461124, 'param')
# Calling param(args, kwargs) (line 145)
param_call_result_461132 = invoke(stypy.reporting.localization.Localization(__file__, 145, 16), param_461125, *[str_461126], **kwargs_461131)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 15), list_461123, param_call_result_461132)
# Adding element type (line 145)

# Call to param(...): (line 146)
# Processing the call arguments (line 146)
str_461135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 29), 'str', 'matvecs')
# Processing the call keyword arguments (line 146)
# Getting the type of 'pytest' (line 146)
pytest_461136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 46), 'pytest', False)
# Obtaining the member 'mark' of a type (line 146)
mark_461137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 46), pytest_461136, 'mark')
# Obtaining the member 'xslow' of a type (line 146)
xslow_461138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 46), mark_461137, 'xslow')
keyword_461139 = xslow_461138
kwargs_461140 = {'marks': keyword_461139}
# Getting the type of 'pytest' (line 146)
pytest_461133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'pytest', False)
# Obtaining the member 'param' of a type (line 146)
param_461134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 16), pytest_461133, 'param')
# Calling param(args, kwargs) (line 146)
param_call_result_461141 = invoke(stypy.reporting.localization.Localization(__file__, 146, 16), param_461134, *[str_461135], **kwargs_461140)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 15), list_461123, param_call_result_461141)
# Adding element type (line 145)
str_461142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 16), 'str', 'matvec')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 15), list_461123, str_461142)
# Adding element type (line 145)
str_461143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 16), 'str', 'diagonal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 15), list_461123, str_461143)
# Adding element type (line 145)
str_461144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 16), 'str', 'sort_indices')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 15), list_461123, str_461144)
# Adding element type (line 145)

# Call to param(...): (line 150)
# Processing the call arguments (line 150)
str_461147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 29), 'str', 'transpose')
# Processing the call keyword arguments (line 150)
# Getting the type of 'pytest' (line 150)
pytest_461148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 48), 'pytest', False)
# Obtaining the member 'mark' of a type (line 150)
mark_461149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 48), pytest_461148, 'mark')
# Obtaining the member 'xslow' of a type (line 150)
xslow_461150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 48), mark_461149, 'xslow')
keyword_461151 = xslow_461150
kwargs_461152 = {'marks': keyword_461151}
# Getting the type of 'pytest' (line 150)
pytest_461145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'pytest', False)
# Obtaining the member 'param' of a type (line 150)
param_461146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 16), pytest_461145, 'param')
# Calling param(args, kwargs) (line 150)
param_call_result_461153 = invoke(stypy.reporting.localization.Localization(__file__, 150, 16), param_461146, *[str_461147], **kwargs_461152)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 15), list_461123, param_call_result_461153)

# Getting the type of 'TestInt32Overflow'
TestInt32Overflow_461154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestInt32Overflow')
# Setting the type of the member '_bsr_ops' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestInt32Overflow_461154, '_bsr_ops', list_461123)

@norecursion
def test_csr_matmat_int64_overflow(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_csr_matmat_int64_overflow'
    module_type_store = module_type_store.open_function_context('test_csr_matmat_int64_overflow', 245, 0, False)
    
    # Passed parameters checking function
    test_csr_matmat_int64_overflow.stypy_localization = localization
    test_csr_matmat_int64_overflow.stypy_type_of_self = None
    test_csr_matmat_int64_overflow.stypy_type_store = module_type_store
    test_csr_matmat_int64_overflow.stypy_function_name = 'test_csr_matmat_int64_overflow'
    test_csr_matmat_int64_overflow.stypy_param_names_list = []
    test_csr_matmat_int64_overflow.stypy_varargs_param_name = None
    test_csr_matmat_int64_overflow.stypy_kwargs_param_name = None
    test_csr_matmat_int64_overflow.stypy_call_defaults = defaults
    test_csr_matmat_int64_overflow.stypy_call_varargs = varargs
    test_csr_matmat_int64_overflow.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_csr_matmat_int64_overflow', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_csr_matmat_int64_overflow', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_csr_matmat_int64_overflow(...)' code ##################

    
    # Assigning a Num to a Name (line 247):
    long_461155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 8), 'long')
    # Assigning a type to the variable 'n' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'n', long_461155)
    # Evaluating assert statement condition
    
    # Getting the type of 'n' (line 248)
    n_461156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 11), 'n')
    int_461157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 14), 'int')
    # Applying the binary operator '**' (line 248)
    result_pow_461158 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 11), '**', n_461156, int_461157)
    
    
    # Call to iinfo(...): (line 248)
    # Processing the call arguments (line 248)
    # Getting the type of 'np' (line 248)
    np_461161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 27), 'np', False)
    # Obtaining the member 'int64' of a type (line 248)
    int64_461162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 27), np_461161, 'int64')
    # Processing the call keyword arguments (line 248)
    kwargs_461163 = {}
    # Getting the type of 'np' (line 248)
    np_461159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 18), 'np', False)
    # Obtaining the member 'iinfo' of a type (line 248)
    iinfo_461160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 18), np_461159, 'iinfo')
    # Calling iinfo(args, kwargs) (line 248)
    iinfo_call_result_461164 = invoke(stypy.reporting.localization.Localization(__file__, 248, 18), iinfo_461160, *[int64_461162], **kwargs_461163)
    
    # Obtaining the member 'max' of a type (line 248)
    max_461165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 18), iinfo_call_result_461164, 'max')
    # Applying the binary operator '>' (line 248)
    result_gt_461166 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 11), '>', result_pow_461158, max_461165)
    
    
    # Call to check_free_memory(...): (line 251)
    # Processing the call arguments (line 251)
    # Getting the type of 'n' (line 251)
    n_461168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 22), 'n', False)
    int_461169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 27), 'int')
    int_461170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 29), 'int')
    # Applying the binary operator '*' (line 251)
    result_mul_461171 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 27), '*', int_461169, int_461170)
    
    int_461172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 33), 'int')
    # Applying the binary operator '+' (line 251)
    result_add_461173 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 27), '+', result_mul_461171, int_461172)
    
    # Applying the binary operator '*' (line 251)
    result_mul_461174 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 22), '*', n_461168, result_add_461173)
    
    int_461175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 38), 'int')
    # Applying the binary operator '*' (line 251)
    result_mul_461176 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 36), '*', result_mul_461174, int_461175)
    
    float_461177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 42), 'float')
    # Applying the binary operator 'div' (line 251)
    result_div_461178 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 40), 'div', result_mul_461176, float_461177)
    
    # Processing the call keyword arguments (line 251)
    kwargs_461179 = {}
    # Getting the type of 'check_free_memory' (line 251)
    check_free_memory_461167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'check_free_memory', False)
    # Calling check_free_memory(args, kwargs) (line 251)
    check_free_memory_call_result_461180 = invoke(stypy.reporting.localization.Localization(__file__, 251, 4), check_free_memory_461167, *[result_div_461178], **kwargs_461179)
    
    
    # Assigning a Call to a Name (line 254):
    
    # Call to ones(...): (line 254)
    # Processing the call arguments (line 254)
    
    # Obtaining an instance of the builtin type 'tuple' (line 254)
    tuple_461183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 254)
    # Adding element type (line 254)
    # Getting the type of 'n' (line 254)
    n_461184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 20), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 20), tuple_461183, n_461184)
    
    # Processing the call keyword arguments (line 254)
    # Getting the type of 'np' (line 254)
    np_461185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 31), 'np', False)
    # Obtaining the member 'int8' of a type (line 254)
    int8_461186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 31), np_461185, 'int8')
    keyword_461187 = int8_461186
    kwargs_461188 = {'dtype': keyword_461187}
    # Getting the type of 'np' (line 254)
    np_461181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 11), 'np', False)
    # Obtaining the member 'ones' of a type (line 254)
    ones_461182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 11), np_461181, 'ones')
    # Calling ones(args, kwargs) (line 254)
    ones_call_result_461189 = invoke(stypy.reporting.localization.Localization(__file__, 254, 11), ones_461182, *[tuple_461183], **kwargs_461188)
    
    # Assigning a type to the variable 'data' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'data', ones_call_result_461189)
    
    # Assigning a Call to a Name (line 255):
    
    # Call to arange(...): (line 255)
    # Processing the call arguments (line 255)
    # Getting the type of 'n' (line 255)
    n_461192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 23), 'n', False)
    int_461193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 25), 'int')
    # Applying the binary operator '+' (line 255)
    result_add_461194 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 23), '+', n_461192, int_461193)
    
    # Processing the call keyword arguments (line 255)
    # Getting the type of 'np' (line 255)
    np_461195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 34), 'np', False)
    # Obtaining the member 'int64' of a type (line 255)
    int64_461196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 34), np_461195, 'int64')
    keyword_461197 = int64_461196
    kwargs_461198 = {'dtype': keyword_461197}
    # Getting the type of 'np' (line 255)
    np_461190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 13), 'np', False)
    # Obtaining the member 'arange' of a type (line 255)
    arange_461191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 13), np_461190, 'arange')
    # Calling arange(args, kwargs) (line 255)
    arange_call_result_461199 = invoke(stypy.reporting.localization.Localization(__file__, 255, 13), arange_461191, *[result_add_461194], **kwargs_461198)
    
    # Assigning a type to the variable 'indptr' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'indptr', arange_call_result_461199)
    
    # Assigning a Call to a Name (line 256):
    
    # Call to zeros(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'n' (line 256)
    n_461202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 23), 'n', False)
    # Processing the call keyword arguments (line 256)
    # Getting the type of 'np' (line 256)
    np_461203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 32), 'np', False)
    # Obtaining the member 'int64' of a type (line 256)
    int64_461204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 32), np_461203, 'int64')
    keyword_461205 = int64_461204
    kwargs_461206 = {'dtype': keyword_461205}
    # Getting the type of 'np' (line 256)
    np_461200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 14), 'np', False)
    # Obtaining the member 'zeros' of a type (line 256)
    zeros_461201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 14), np_461200, 'zeros')
    # Calling zeros(args, kwargs) (line 256)
    zeros_call_result_461207 = invoke(stypy.reporting.localization.Localization(__file__, 256, 14), zeros_461201, *[n_461202], **kwargs_461206)
    
    # Assigning a type to the variable 'indices' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'indices', zeros_call_result_461207)
    
    # Assigning a Call to a Name (line 257):
    
    # Call to csr_matrix(...): (line 257)
    # Processing the call arguments (line 257)
    
    # Obtaining an instance of the builtin type 'tuple' (line 257)
    tuple_461209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 257)
    # Adding element type (line 257)
    # Getting the type of 'data' (line 257)
    data_461210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 20), 'data', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 20), tuple_461209, data_461210)
    # Adding element type (line 257)
    # Getting the type of 'indices' (line 257)
    indices_461211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 26), 'indices', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 20), tuple_461209, indices_461211)
    # Adding element type (line 257)
    # Getting the type of 'indptr' (line 257)
    indptr_461212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 35), 'indptr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 20), tuple_461209, indptr_461212)
    
    # Processing the call keyword arguments (line 257)
    kwargs_461213 = {}
    # Getting the type of 'csr_matrix' (line 257)
    csr_matrix_461208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'csr_matrix', False)
    # Calling csr_matrix(args, kwargs) (line 257)
    csr_matrix_call_result_461214 = invoke(stypy.reporting.localization.Localization(__file__, 257, 8), csr_matrix_461208, *[tuple_461209], **kwargs_461213)
    
    # Assigning a type to the variable 'a' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'a', csr_matrix_call_result_461214)
    
    # Assigning a Attribute to a Name (line 258):
    # Getting the type of 'a' (line 258)
    a_461215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'a')
    # Obtaining the member 'T' of a type (line 258)
    T_461216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 8), a_461215, 'T')
    # Assigning a type to the variable 'b' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'b', T_461216)
    
    # Call to assert_raises(...): (line 260)
    # Processing the call arguments (line 260)
    # Getting the type of 'RuntimeError' (line 260)
    RuntimeError_461218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 18), 'RuntimeError', False)
    # Getting the type of 'a' (line 260)
    a_461219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 32), 'a', False)
    # Obtaining the member 'dot' of a type (line 260)
    dot_461220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 32), a_461219, 'dot')
    # Getting the type of 'b' (line 260)
    b_461221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 39), 'b', False)
    # Processing the call keyword arguments (line 260)
    kwargs_461222 = {}
    # Getting the type of 'assert_raises' (line 260)
    assert_raises_461217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 260)
    assert_raises_call_result_461223 = invoke(stypy.reporting.localization.Localization(__file__, 260, 4), assert_raises_461217, *[RuntimeError_461218, dot_461220, b_461221], **kwargs_461222)
    
    
    # ################# End of 'test_csr_matmat_int64_overflow(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_csr_matmat_int64_overflow' in the type store
    # Getting the type of 'stypy_return_type' (line 245)
    stypy_return_type_461224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_461224)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_csr_matmat_int64_overflow'
    return stypy_return_type_461224

# Assigning a type to the variable 'test_csr_matmat_int64_overflow' (line 245)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 0), 'test_csr_matmat_int64_overflow', test_csr_matmat_int64_overflow)

@norecursion
def test_upcast(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_upcast'
    module_type_store = module_type_store.open_function_context('test_upcast', 263, 0, False)
    
    # Passed parameters checking function
    test_upcast.stypy_localization = localization
    test_upcast.stypy_type_of_self = None
    test_upcast.stypy_type_store = module_type_store
    test_upcast.stypy_function_name = 'test_upcast'
    test_upcast.stypy_param_names_list = []
    test_upcast.stypy_varargs_param_name = None
    test_upcast.stypy_kwargs_param_name = None
    test_upcast.stypy_call_defaults = defaults
    test_upcast.stypy_call_varargs = varargs
    test_upcast.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_upcast', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_upcast', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_upcast(...)' code ##################

    
    # Assigning a Call to a Name (line 264):
    
    # Call to csr_matrix(...): (line 264)
    # Processing the call arguments (line 264)
    
    # Obtaining an instance of the builtin type 'list' (line 264)
    list_461226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 264)
    # Adding element type (line 264)
    
    # Obtaining an instance of the builtin type 'list' (line 264)
    list_461227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 264)
    # Adding element type (line 264)
    # Getting the type of 'np' (line 264)
    np_461228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 22), 'np', False)
    # Obtaining the member 'pi' of a type (line 264)
    pi_461229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 22), np_461228, 'pi')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 21), list_461227, pi_461229)
    # Adding element type (line 264)
    # Getting the type of 'np' (line 264)
    np_461230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 29), 'np', False)
    # Obtaining the member 'pi' of a type (line 264)
    pi_461231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 29), np_461230, 'pi')
    complex_461232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 35), 'complex')
    # Applying the binary operator '*' (line 264)
    result_mul_461233 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 29), '*', pi_461231, complex_461232)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 21), list_461227, result_mul_461233)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 20), list_461226, list_461227)
    # Adding element type (line 264)
    
    # Obtaining an instance of the builtin type 'list' (line 264)
    list_461234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 264)
    # Adding element type (line 264)
    int_461235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 40), list_461234, int_461235)
    # Adding element type (line 264)
    int_461236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 40), list_461234, int_461236)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 20), list_461226, list_461234)
    
    # Processing the call keyword arguments (line 264)
    # Getting the type of 'complex' (line 264)
    complex_461237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 55), 'complex', False)
    keyword_461238 = complex_461237
    kwargs_461239 = {'dtype': keyword_461238}
    # Getting the type of 'csr_matrix' (line 264)
    csr_matrix_461225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 9), 'csr_matrix', False)
    # Calling csr_matrix(args, kwargs) (line 264)
    csr_matrix_call_result_461240 = invoke(stypy.reporting.localization.Localization(__file__, 264, 9), csr_matrix_461225, *[list_461226], **kwargs_461239)
    
    # Assigning a type to the variable 'a0' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'a0', csr_matrix_call_result_461240)
    
    # Assigning a Call to a Name (line 265):
    
    # Call to array(...): (line 265)
    # Processing the call arguments (line 265)
    
    # Obtaining an instance of the builtin type 'list' (line 265)
    list_461243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 265)
    # Adding element type (line 265)
    int_461244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 19), 'int')
    complex_461245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 23), 'complex')
    # Applying the binary operator '+' (line 265)
    result_add_461246 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 19), '+', int_461244, complex_461245)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 18), list_461243, result_add_461246)
    # Adding element type (line 265)
    int_461247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 27), 'int')
    int_461248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 30), 'int')
    # Applying the binary operator '**' (line 265)
    result_pow_461249 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 27), '**', int_461247, int_461248)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 18), list_461243, result_pow_461249)
    
    # Processing the call keyword arguments (line 265)
    # Getting the type of 'complex' (line 265)
    complex_461250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 41), 'complex', False)
    keyword_461251 = complex_461250
    kwargs_461252 = {'dtype': keyword_461251}
    # Getting the type of 'np' (line 265)
    np_461241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 265)
    array_461242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 9), np_461241, 'array')
    # Calling array(args, kwargs) (line 265)
    array_call_result_461253 = invoke(stypy.reporting.localization.Localization(__file__, 265, 9), array_461242, *[list_461243], **kwargs_461252)
    
    # Assigning a type to the variable 'b0' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'b0', array_call_result_461253)
    
    # Getting the type of 'supported_dtypes' (line 267)
    supported_dtypes_461254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 19), 'supported_dtypes')
    # Testing the type of a for loop iterable (line 267)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 267, 4), supported_dtypes_461254)
    # Getting the type of the for loop variable (line 267)
    for_loop_var_461255 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 267, 4), supported_dtypes_461254)
    # Assigning a type to the variable 'a_dtype' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'a_dtype', for_loop_var_461255)
    # SSA begins for a for statement (line 267)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'supported_dtypes' (line 268)
    supported_dtypes_461256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 23), 'supported_dtypes')
    # Testing the type of a for loop iterable (line 268)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 268, 8), supported_dtypes_461256)
    # Getting the type of the for loop variable (line 268)
    for_loop_var_461257 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 268, 8), supported_dtypes_461256)
    # Assigning a type to the variable 'b_dtype' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'b_dtype', for_loop_var_461257)
    # SSA begins for a for statement (line 268)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 269):
    str_461258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 18), 'str', '(%r, %r)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 269)
    tuple_461259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 269)
    # Adding element type (line 269)
    # Getting the type of 'a_dtype' (line 269)
    a_dtype_461260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 32), 'a_dtype')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 32), tuple_461259, a_dtype_461260)
    # Adding element type (line 269)
    # Getting the type of 'b_dtype' (line 269)
    b_dtype_461261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 41), 'b_dtype')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 32), tuple_461259, b_dtype_461261)
    
    # Applying the binary operator '%' (line 269)
    result_mod_461262 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 18), '%', str_461258, tuple_461259)
    
    # Assigning a type to the variable 'msg' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'msg', result_mod_461262)
    
    
    # Call to issubdtype(...): (line 271)
    # Processing the call arguments (line 271)
    # Getting the type of 'a_dtype' (line 271)
    a_dtype_461265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 29), 'a_dtype', False)
    # Getting the type of 'np' (line 271)
    np_461266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 38), 'np', False)
    # Obtaining the member 'complexfloating' of a type (line 271)
    complexfloating_461267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 38), np_461266, 'complexfloating')
    # Processing the call keyword arguments (line 271)
    kwargs_461268 = {}
    # Getting the type of 'np' (line 271)
    np_461263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 15), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 271)
    issubdtype_461264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 15), np_461263, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 271)
    issubdtype_call_result_461269 = invoke(stypy.reporting.localization.Localization(__file__, 271, 15), issubdtype_461264, *[a_dtype_461265, complexfloating_461267], **kwargs_461268)
    
    # Testing the type of an if condition (line 271)
    if_condition_461270 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 271, 12), issubdtype_call_result_461269)
    # Assigning a type to the variable 'if_condition_461270' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'if_condition_461270', if_condition_461270)
    # SSA begins for if statement (line 271)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 272):
    
    # Call to astype(...): (line 272)
    # Processing the call arguments (line 272)
    # Getting the type of 'a_dtype' (line 272)
    a_dtype_461276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 37), 'a_dtype', False)
    # Processing the call keyword arguments (line 272)
    kwargs_461277 = {}
    
    # Call to copy(...): (line 272)
    # Processing the call keyword arguments (line 272)
    kwargs_461273 = {}
    # Getting the type of 'a0' (line 272)
    a0_461271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 20), 'a0', False)
    # Obtaining the member 'copy' of a type (line 272)
    copy_461272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 20), a0_461271, 'copy')
    # Calling copy(args, kwargs) (line 272)
    copy_call_result_461274 = invoke(stypy.reporting.localization.Localization(__file__, 272, 20), copy_461272, *[], **kwargs_461273)
    
    # Obtaining the member 'astype' of a type (line 272)
    astype_461275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 20), copy_call_result_461274, 'astype')
    # Calling astype(args, kwargs) (line 272)
    astype_call_result_461278 = invoke(stypy.reporting.localization.Localization(__file__, 272, 20), astype_461275, *[a_dtype_461276], **kwargs_461277)
    
    # Assigning a type to the variable 'a' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 16), 'a', astype_call_result_461278)
    # SSA branch for the else part of an if statement (line 271)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 274):
    
    # Call to astype(...): (line 274)
    # Processing the call arguments (line 274)
    # Getting the type of 'a_dtype' (line 274)
    a_dtype_461285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 42), 'a_dtype', False)
    # Processing the call keyword arguments (line 274)
    kwargs_461286 = {}
    
    # Call to copy(...): (line 274)
    # Processing the call keyword arguments (line 274)
    kwargs_461282 = {}
    # Getting the type of 'a0' (line 274)
    a0_461279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 20), 'a0', False)
    # Obtaining the member 'real' of a type (line 274)
    real_461280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 20), a0_461279, 'real')
    # Obtaining the member 'copy' of a type (line 274)
    copy_461281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 20), real_461280, 'copy')
    # Calling copy(args, kwargs) (line 274)
    copy_call_result_461283 = invoke(stypy.reporting.localization.Localization(__file__, 274, 20), copy_461281, *[], **kwargs_461282)
    
    # Obtaining the member 'astype' of a type (line 274)
    astype_461284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 20), copy_call_result_461283, 'astype')
    # Calling astype(args, kwargs) (line 274)
    astype_call_result_461287 = invoke(stypy.reporting.localization.Localization(__file__, 274, 20), astype_461284, *[a_dtype_461285], **kwargs_461286)
    
    # Assigning a type to the variable 'a' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'a', astype_call_result_461287)
    # SSA join for if statement (line 271)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to issubdtype(...): (line 276)
    # Processing the call arguments (line 276)
    # Getting the type of 'b_dtype' (line 276)
    b_dtype_461290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 29), 'b_dtype', False)
    # Getting the type of 'np' (line 276)
    np_461291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 38), 'np', False)
    # Obtaining the member 'complexfloating' of a type (line 276)
    complexfloating_461292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 38), np_461291, 'complexfloating')
    # Processing the call keyword arguments (line 276)
    kwargs_461293 = {}
    # Getting the type of 'np' (line 276)
    np_461288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 15), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 276)
    issubdtype_461289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 15), np_461288, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 276)
    issubdtype_call_result_461294 = invoke(stypy.reporting.localization.Localization(__file__, 276, 15), issubdtype_461289, *[b_dtype_461290, complexfloating_461292], **kwargs_461293)
    
    # Testing the type of an if condition (line 276)
    if_condition_461295 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 276, 12), issubdtype_call_result_461294)
    # Assigning a type to the variable 'if_condition_461295' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'if_condition_461295', if_condition_461295)
    # SSA begins for if statement (line 276)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 277):
    
    # Call to astype(...): (line 277)
    # Processing the call arguments (line 277)
    # Getting the type of 'b_dtype' (line 277)
    b_dtype_461301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 37), 'b_dtype', False)
    # Processing the call keyword arguments (line 277)
    kwargs_461302 = {}
    
    # Call to copy(...): (line 277)
    # Processing the call keyword arguments (line 277)
    kwargs_461298 = {}
    # Getting the type of 'b0' (line 277)
    b0_461296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 20), 'b0', False)
    # Obtaining the member 'copy' of a type (line 277)
    copy_461297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 20), b0_461296, 'copy')
    # Calling copy(args, kwargs) (line 277)
    copy_call_result_461299 = invoke(stypy.reporting.localization.Localization(__file__, 277, 20), copy_461297, *[], **kwargs_461298)
    
    # Obtaining the member 'astype' of a type (line 277)
    astype_461300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 20), copy_call_result_461299, 'astype')
    # Calling astype(args, kwargs) (line 277)
    astype_call_result_461303 = invoke(stypy.reporting.localization.Localization(__file__, 277, 20), astype_461300, *[b_dtype_461301], **kwargs_461302)
    
    # Assigning a type to the variable 'b' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 16), 'b', astype_call_result_461303)
    # SSA branch for the else part of an if statement (line 276)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 279):
    
    # Call to astype(...): (line 279)
    # Processing the call arguments (line 279)
    # Getting the type of 'b_dtype' (line 279)
    b_dtype_461310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 42), 'b_dtype', False)
    # Processing the call keyword arguments (line 279)
    kwargs_461311 = {}
    
    # Call to copy(...): (line 279)
    # Processing the call keyword arguments (line 279)
    kwargs_461307 = {}
    # Getting the type of 'b0' (line 279)
    b0_461304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 20), 'b0', False)
    # Obtaining the member 'real' of a type (line 279)
    real_461305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 20), b0_461304, 'real')
    # Obtaining the member 'copy' of a type (line 279)
    copy_461306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 20), real_461305, 'copy')
    # Calling copy(args, kwargs) (line 279)
    copy_call_result_461308 = invoke(stypy.reporting.localization.Localization(__file__, 279, 20), copy_461306, *[], **kwargs_461307)
    
    # Obtaining the member 'astype' of a type (line 279)
    astype_461309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 20), copy_call_result_461308, 'astype')
    # Calling astype(args, kwargs) (line 279)
    astype_call_result_461312 = invoke(stypy.reporting.localization.Localization(__file__, 279, 20), astype_461309, *[b_dtype_461310], **kwargs_461311)
    
    # Assigning a type to the variable 'b' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 16), 'b', astype_call_result_461312)
    # SSA join for if statement (line 276)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'a_dtype' (line 281)
    a_dtype_461313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 20), 'a_dtype')
    # Getting the type of 'np' (line 281)
    np_461314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 31), 'np')
    # Obtaining the member 'bool_' of a type (line 281)
    bool__461315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 31), np_461314, 'bool_')
    # Applying the binary operator '==' (line 281)
    result_eq_461316 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 20), '==', a_dtype_461313, bool__461315)
    
    
    # Getting the type of 'b_dtype' (line 281)
    b_dtype_461317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 44), 'b_dtype')
    # Getting the type of 'np' (line 281)
    np_461318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 55), 'np')
    # Obtaining the member 'bool_' of a type (line 281)
    bool__461319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 55), np_461318, 'bool_')
    # Applying the binary operator '==' (line 281)
    result_eq_461320 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 44), '==', b_dtype_461317, bool__461319)
    
    # Applying the binary operator 'and' (line 281)
    result_and_keyword_461321 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 20), 'and', result_eq_461316, result_eq_461320)
    
    # Applying the 'not' unary operator (line 281)
    result_not__461322 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 15), 'not', result_and_keyword_461321)
    
    # Testing the type of an if condition (line 281)
    if_condition_461323 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 281, 12), result_not__461322)
    # Assigning a type to the variable 'if_condition_461323' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'if_condition_461323', if_condition_461323)
    # SSA begins for if statement (line 281)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 282):
    
    # Call to zeros(...): (line 282)
    # Processing the call arguments (line 282)
    
    # Obtaining an instance of the builtin type 'tuple' (line 282)
    tuple_461326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 282)
    # Adding element type (line 282)
    int_461327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 30), tuple_461326, int_461327)
    
    # Processing the call keyword arguments (line 282)
    # Getting the type of 'np' (line 282)
    np_461328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 41), 'np', False)
    # Obtaining the member 'bool_' of a type (line 282)
    bool__461329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 41), np_461328, 'bool_')
    keyword_461330 = bool__461329
    kwargs_461331 = {'dtype': keyword_461330}
    # Getting the type of 'np' (line 282)
    np_461324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 20), 'np', False)
    # Obtaining the member 'zeros' of a type (line 282)
    zeros_461325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 20), np_461324, 'zeros')
    # Calling zeros(args, kwargs) (line 282)
    zeros_call_result_461332 = invoke(stypy.reporting.localization.Localization(__file__, 282, 20), zeros_461325, *[tuple_461326], **kwargs_461331)
    
    # Assigning a type to the variable 'c' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 16), 'c', zeros_call_result_461332)
    
    # Call to assert_raises(...): (line 283)
    # Processing the call arguments (line 283)
    # Getting the type of 'ValueError' (line 283)
    ValueError_461334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 30), 'ValueError', False)
    # Getting the type of '_sparsetools' (line 283)
    _sparsetools_461335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 42), '_sparsetools', False)
    # Obtaining the member 'csr_matvec' of a type (line 283)
    csr_matvec_461336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 42), _sparsetools_461335, 'csr_matvec')
    int_461337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 30), 'int')
    int_461338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 33), 'int')
    # Getting the type of 'a' (line 284)
    a_461339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 36), 'a', False)
    # Obtaining the member 'indptr' of a type (line 284)
    indptr_461340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 36), a_461339, 'indptr')
    # Getting the type of 'a' (line 284)
    a_461341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 46), 'a', False)
    # Obtaining the member 'indices' of a type (line 284)
    indices_461342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 46), a_461341, 'indices')
    # Getting the type of 'a' (line 284)
    a_461343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 57), 'a', False)
    # Obtaining the member 'data' of a type (line 284)
    data_461344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 57), a_461343, 'data')
    # Getting the type of 'b' (line 284)
    b_461345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 65), 'b', False)
    # Getting the type of 'c' (line 284)
    c_461346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 68), 'c', False)
    # Processing the call keyword arguments (line 283)
    kwargs_461347 = {}
    # Getting the type of 'assert_raises' (line 283)
    assert_raises_461333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 283)
    assert_raises_call_result_461348 = invoke(stypy.reporting.localization.Localization(__file__, 283, 16), assert_raises_461333, *[ValueError_461334, csr_matvec_461336, int_461337, int_461338, indptr_461340, indices_461342, data_461344, b_461345, c_461346], **kwargs_461347)
    
    # SSA join for if statement (line 281)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    # Call to issubdtype(...): (line 286)
    # Processing the call arguments (line 286)
    # Getting the type of 'a_dtype' (line 286)
    a_dtype_461351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 31), 'a_dtype', False)
    # Getting the type of 'np' (line 286)
    np_461352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 40), 'np', False)
    # Obtaining the member 'complexfloating' of a type (line 286)
    complexfloating_461353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 40), np_461352, 'complexfloating')
    # Processing the call keyword arguments (line 286)
    kwargs_461354 = {}
    # Getting the type of 'np' (line 286)
    np_461349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 17), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 286)
    issubdtype_461350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 17), np_461349, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 286)
    issubdtype_call_result_461355 = invoke(stypy.reporting.localization.Localization(__file__, 286, 17), issubdtype_461350, *[a_dtype_461351, complexfloating_461353], **kwargs_461354)
    
    
    
    # Call to issubdtype(...): (line 287)
    # Processing the call arguments (line 287)
    # Getting the type of 'b_dtype' (line 287)
    b_dtype_461358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 35), 'b_dtype', False)
    # Getting the type of 'np' (line 287)
    np_461359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 44), 'np', False)
    # Obtaining the member 'complexfloating' of a type (line 287)
    complexfloating_461360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 44), np_461359, 'complexfloating')
    # Processing the call keyword arguments (line 287)
    kwargs_461361 = {}
    # Getting the type of 'np' (line 287)
    np_461356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 21), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 287)
    issubdtype_461357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 21), np_461356, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 287)
    issubdtype_call_result_461362 = invoke(stypy.reporting.localization.Localization(__file__, 287, 21), issubdtype_461357, *[b_dtype_461358, complexfloating_461360], **kwargs_461361)
    
    # Applying the 'not' unary operator (line 287)
    result_not__461363 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 17), 'not', issubdtype_call_result_461362)
    
    # Applying the binary operator 'and' (line 286)
    result_and_keyword_461364 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 17), 'and', issubdtype_call_result_461355, result_not__461363)
    
    
    # Evaluating a boolean operation
    
    
    # Call to issubdtype(...): (line 288)
    # Processing the call arguments (line 288)
    # Getting the type of 'a_dtype' (line 288)
    a_dtype_461367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 35), 'a_dtype', False)
    # Getting the type of 'np' (line 288)
    np_461368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 44), 'np', False)
    # Obtaining the member 'complexfloating' of a type (line 288)
    complexfloating_461369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 44), np_461368, 'complexfloating')
    # Processing the call keyword arguments (line 288)
    kwargs_461370 = {}
    # Getting the type of 'np' (line 288)
    np_461365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 21), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 288)
    issubdtype_461366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 21), np_461365, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 288)
    issubdtype_call_result_461371 = invoke(stypy.reporting.localization.Localization(__file__, 288, 21), issubdtype_461366, *[a_dtype_461367, complexfloating_461369], **kwargs_461370)
    
    # Applying the 'not' unary operator (line 288)
    result_not__461372 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 17), 'not', issubdtype_call_result_461371)
    
    
    # Call to issubdtype(...): (line 289)
    # Processing the call arguments (line 289)
    # Getting the type of 'b_dtype' (line 289)
    b_dtype_461375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 31), 'b_dtype', False)
    # Getting the type of 'np' (line 289)
    np_461376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 40), 'np', False)
    # Obtaining the member 'complexfloating' of a type (line 289)
    complexfloating_461377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 40), np_461376, 'complexfloating')
    # Processing the call keyword arguments (line 289)
    kwargs_461378 = {}
    # Getting the type of 'np' (line 289)
    np_461373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 17), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 289)
    issubdtype_461374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 17), np_461373, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 289)
    issubdtype_call_result_461379 = invoke(stypy.reporting.localization.Localization(__file__, 289, 17), issubdtype_461374, *[b_dtype_461375, complexfloating_461377], **kwargs_461378)
    
    # Applying the binary operator 'and' (line 288)
    result_and_keyword_461380 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 17), 'and', result_not__461372, issubdtype_call_result_461379)
    
    # Applying the binary operator 'or' (line 286)
    result_or_keyword_461381 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 16), 'or', result_and_keyword_461364, result_and_keyword_461380)
    
    # Testing the type of an if condition (line 286)
    if_condition_461382 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 286, 12), result_or_keyword_461381)
    # Assigning a type to the variable 'if_condition_461382' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'if_condition_461382', if_condition_461382)
    # SSA begins for if statement (line 286)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 290):
    
    # Call to zeros(...): (line 290)
    # Processing the call arguments (line 290)
    
    # Obtaining an instance of the builtin type 'tuple' (line 290)
    tuple_461385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 290)
    # Adding element type (line 290)
    int_461386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 30), tuple_461385, int_461386)
    
    # Processing the call keyword arguments (line 290)
    # Getting the type of 'np' (line 290)
    np_461387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 41), 'np', False)
    # Obtaining the member 'float64' of a type (line 290)
    float64_461388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 41), np_461387, 'float64')
    keyword_461389 = float64_461388
    kwargs_461390 = {'dtype': keyword_461389}
    # Getting the type of 'np' (line 290)
    np_461383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 20), 'np', False)
    # Obtaining the member 'zeros' of a type (line 290)
    zeros_461384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 20), np_461383, 'zeros')
    # Calling zeros(args, kwargs) (line 290)
    zeros_call_result_461391 = invoke(stypy.reporting.localization.Localization(__file__, 290, 20), zeros_461384, *[tuple_461385], **kwargs_461390)
    
    # Assigning a type to the variable 'c' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 16), 'c', zeros_call_result_461391)
    
    # Call to assert_raises(...): (line 291)
    # Processing the call arguments (line 291)
    # Getting the type of 'ValueError' (line 291)
    ValueError_461393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 30), 'ValueError', False)
    # Getting the type of '_sparsetools' (line 291)
    _sparsetools_461394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 42), '_sparsetools', False)
    # Obtaining the member 'csr_matvec' of a type (line 291)
    csr_matvec_461395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 42), _sparsetools_461394, 'csr_matvec')
    int_461396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 30), 'int')
    int_461397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 33), 'int')
    # Getting the type of 'a' (line 292)
    a_461398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 36), 'a', False)
    # Obtaining the member 'indptr' of a type (line 292)
    indptr_461399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 36), a_461398, 'indptr')
    # Getting the type of 'a' (line 292)
    a_461400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 46), 'a', False)
    # Obtaining the member 'indices' of a type (line 292)
    indices_461401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 46), a_461400, 'indices')
    # Getting the type of 'a' (line 292)
    a_461402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 57), 'a', False)
    # Obtaining the member 'data' of a type (line 292)
    data_461403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 57), a_461402, 'data')
    # Getting the type of 'b' (line 292)
    b_461404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 65), 'b', False)
    # Getting the type of 'c' (line 292)
    c_461405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 68), 'c', False)
    # Processing the call keyword arguments (line 291)
    kwargs_461406 = {}
    # Getting the type of 'assert_raises' (line 291)
    assert_raises_461392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 16), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 291)
    assert_raises_call_result_461407 = invoke(stypy.reporting.localization.Localization(__file__, 291, 16), assert_raises_461392, *[ValueError_461393, csr_matvec_461395, int_461396, int_461397, indptr_461399, indices_461401, data_461403, b_461404, c_461405], **kwargs_461406)
    
    # SSA join for if statement (line 286)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 294):
    
    # Call to zeros(...): (line 294)
    # Processing the call arguments (line 294)
    
    # Obtaining an instance of the builtin type 'tuple' (line 294)
    tuple_461410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 294)
    # Adding element type (line 294)
    int_461411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 26), tuple_461410, int_461411)
    
    # Processing the call keyword arguments (line 294)
    
    # Call to result_type(...): (line 294)
    # Processing the call arguments (line 294)
    # Getting the type of 'a_dtype' (line 294)
    a_dtype_461414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 52), 'a_dtype', False)
    # Getting the type of 'b_dtype' (line 294)
    b_dtype_461415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 61), 'b_dtype', False)
    # Processing the call keyword arguments (line 294)
    kwargs_461416 = {}
    # Getting the type of 'np' (line 294)
    np_461412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 37), 'np', False)
    # Obtaining the member 'result_type' of a type (line 294)
    result_type_461413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 37), np_461412, 'result_type')
    # Calling result_type(args, kwargs) (line 294)
    result_type_call_result_461417 = invoke(stypy.reporting.localization.Localization(__file__, 294, 37), result_type_461413, *[a_dtype_461414, b_dtype_461415], **kwargs_461416)
    
    keyword_461418 = result_type_call_result_461417
    kwargs_461419 = {'dtype': keyword_461418}
    # Getting the type of 'np' (line 294)
    np_461408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 16), 'np', False)
    # Obtaining the member 'zeros' of a type (line 294)
    zeros_461409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 16), np_461408, 'zeros')
    # Calling zeros(args, kwargs) (line 294)
    zeros_call_result_461420 = invoke(stypy.reporting.localization.Localization(__file__, 294, 16), zeros_461409, *[tuple_461410], **kwargs_461419)
    
    # Assigning a type to the variable 'c' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'c', zeros_call_result_461420)
    
    # Call to csr_matvec(...): (line 295)
    # Processing the call arguments (line 295)
    int_461423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 36), 'int')
    int_461424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 39), 'int')
    # Getting the type of 'a' (line 295)
    a_461425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 42), 'a', False)
    # Obtaining the member 'indptr' of a type (line 295)
    indptr_461426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 42), a_461425, 'indptr')
    # Getting the type of 'a' (line 295)
    a_461427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 52), 'a', False)
    # Obtaining the member 'indices' of a type (line 295)
    indices_461428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 52), a_461427, 'indices')
    # Getting the type of 'a' (line 295)
    a_461429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 63), 'a', False)
    # Obtaining the member 'data' of a type (line 295)
    data_461430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 63), a_461429, 'data')
    # Getting the type of 'b' (line 295)
    b_461431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 71), 'b', False)
    # Getting the type of 'c' (line 295)
    c_461432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 74), 'c', False)
    # Processing the call keyword arguments (line 295)
    kwargs_461433 = {}
    # Getting the type of '_sparsetools' (line 295)
    _sparsetools_461421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), '_sparsetools', False)
    # Obtaining the member 'csr_matvec' of a type (line 295)
    csr_matvec_461422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 12), _sparsetools_461421, 'csr_matvec')
    # Calling csr_matvec(args, kwargs) (line 295)
    csr_matvec_call_result_461434 = invoke(stypy.reporting.localization.Localization(__file__, 295, 12), csr_matvec_461422, *[int_461423, int_461424, indptr_461426, indices_461428, data_461430, b_461431, c_461432], **kwargs_461433)
    
    
    # Call to assert_allclose(...): (line 296)
    # Processing the call arguments (line 296)
    # Getting the type of 'c' (line 296)
    c_461436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 28), 'c', False)
    
    # Call to dot(...): (line 296)
    # Processing the call arguments (line 296)
    
    # Call to toarray(...): (line 296)
    # Processing the call keyword arguments (line 296)
    kwargs_461441 = {}
    # Getting the type of 'a' (line 296)
    a_461439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 38), 'a', False)
    # Obtaining the member 'toarray' of a type (line 296)
    toarray_461440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 38), a_461439, 'toarray')
    # Calling toarray(args, kwargs) (line 296)
    toarray_call_result_461442 = invoke(stypy.reporting.localization.Localization(__file__, 296, 38), toarray_461440, *[], **kwargs_461441)
    
    # Getting the type of 'b' (line 296)
    b_461443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 51), 'b', False)
    # Processing the call keyword arguments (line 296)
    kwargs_461444 = {}
    # Getting the type of 'np' (line 296)
    np_461437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 31), 'np', False)
    # Obtaining the member 'dot' of a type (line 296)
    dot_461438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 31), np_461437, 'dot')
    # Calling dot(args, kwargs) (line 296)
    dot_call_result_461445 = invoke(stypy.reporting.localization.Localization(__file__, 296, 31), dot_461438, *[toarray_call_result_461442, b_461443], **kwargs_461444)
    
    # Processing the call keyword arguments (line 296)
    # Getting the type of 'msg' (line 296)
    msg_461446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 63), 'msg', False)
    keyword_461447 = msg_461446
    kwargs_461448 = {'err_msg': keyword_461447}
    # Getting the type of 'assert_allclose' (line 296)
    assert_allclose_461435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 296)
    assert_allclose_call_result_461449 = invoke(stypy.reporting.localization.Localization(__file__, 296, 12), assert_allclose_461435, *[c_461436, dot_call_result_461445], **kwargs_461448)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_upcast(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_upcast' in the type store
    # Getting the type of 'stypy_return_type' (line 263)
    stypy_return_type_461450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_461450)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_upcast'
    return stypy_return_type_461450

# Assigning a type to the variable 'test_upcast' (line 263)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'test_upcast', test_upcast)

@norecursion
def test_endianness(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_endianness'
    module_type_store = module_type_store.open_function_context('test_endianness', 299, 0, False)
    
    # Passed parameters checking function
    test_endianness.stypy_localization = localization
    test_endianness.stypy_type_of_self = None
    test_endianness.stypy_type_store = module_type_store
    test_endianness.stypy_function_name = 'test_endianness'
    test_endianness.stypy_param_names_list = []
    test_endianness.stypy_varargs_param_name = None
    test_endianness.stypy_kwargs_param_name = None
    test_endianness.stypy_call_defaults = defaults
    test_endianness.stypy_call_varargs = varargs
    test_endianness.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_endianness', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_endianness', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_endianness(...)' code ##################

    
    # Assigning a Call to a Name (line 300):
    
    # Call to ones(...): (line 300)
    # Processing the call arguments (line 300)
    
    # Obtaining an instance of the builtin type 'tuple' (line 300)
    tuple_461453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 300)
    # Adding element type (line 300)
    int_461454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 17), tuple_461453, int_461454)
    # Adding element type (line 300)
    int_461455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 17), tuple_461453, int_461455)
    
    # Processing the call keyword arguments (line 300)
    kwargs_461456 = {}
    # Getting the type of 'np' (line 300)
    np_461451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'np', False)
    # Obtaining the member 'ones' of a type (line 300)
    ones_461452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 8), np_461451, 'ones')
    # Calling ones(args, kwargs) (line 300)
    ones_call_result_461457 = invoke(stypy.reporting.localization.Localization(__file__, 300, 8), ones_461452, *[tuple_461453], **kwargs_461456)
    
    # Assigning a type to the variable 'd' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'd', ones_call_result_461457)
    
    # Assigning a List to a Name (line 301):
    
    # Obtaining an instance of the builtin type 'list' (line 301)
    list_461458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 301)
    # Adding element type (line 301)
    int_461459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 14), list_461458, int_461459)
    # Adding element type (line 301)
    int_461460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 14), list_461458, int_461460)
    # Adding element type (line 301)
    int_461461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 14), list_461458, int_461461)
    
    # Assigning a type to the variable 'offsets' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'offsets', list_461458)
    
    # Assigning a Call to a Name (line 303):
    
    # Call to dia_matrix(...): (line 303)
    # Processing the call arguments (line 303)
    
    # Obtaining an instance of the builtin type 'tuple' (line 303)
    tuple_461463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 303)
    # Adding element type (line 303)
    
    # Call to astype(...): (line 303)
    # Processing the call arguments (line 303)
    str_461466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 29), 'str', '<f8')
    # Processing the call keyword arguments (line 303)
    kwargs_461467 = {}
    # Getting the type of 'd' (line 303)
    d_461464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 20), 'd', False)
    # Obtaining the member 'astype' of a type (line 303)
    astype_461465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 20), d_461464, 'astype')
    # Calling astype(args, kwargs) (line 303)
    astype_call_result_461468 = invoke(stypy.reporting.localization.Localization(__file__, 303, 20), astype_461465, *[str_461466], **kwargs_461467)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 20), tuple_461463, astype_call_result_461468)
    # Adding element type (line 303)
    # Getting the type of 'offsets' (line 303)
    offsets_461469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 37), 'offsets', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 20), tuple_461463, offsets_461469)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 303)
    tuple_461470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 303)
    # Adding element type (line 303)
    int_461471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 48), tuple_461470, int_461471)
    # Adding element type (line 303)
    int_461472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 48), tuple_461470, int_461472)
    
    # Processing the call keyword arguments (line 303)
    kwargs_461473 = {}
    # Getting the type of 'dia_matrix' (line 303)
    dia_matrix_461462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'dia_matrix', False)
    # Calling dia_matrix(args, kwargs) (line 303)
    dia_matrix_call_result_461474 = invoke(stypy.reporting.localization.Localization(__file__, 303, 8), dia_matrix_461462, *[tuple_461463, tuple_461470], **kwargs_461473)
    
    # Assigning a type to the variable 'a' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'a', dia_matrix_call_result_461474)
    
    # Assigning a Call to a Name (line 304):
    
    # Call to dia_matrix(...): (line 304)
    # Processing the call arguments (line 304)
    
    # Obtaining an instance of the builtin type 'tuple' (line 304)
    tuple_461476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 304)
    # Adding element type (line 304)
    
    # Call to astype(...): (line 304)
    # Processing the call arguments (line 304)
    str_461479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 29), 'str', '>f8')
    # Processing the call keyword arguments (line 304)
    kwargs_461480 = {}
    # Getting the type of 'd' (line 304)
    d_461477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 20), 'd', False)
    # Obtaining the member 'astype' of a type (line 304)
    astype_461478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 20), d_461477, 'astype')
    # Calling astype(args, kwargs) (line 304)
    astype_call_result_461481 = invoke(stypy.reporting.localization.Localization(__file__, 304, 20), astype_461478, *[str_461479], **kwargs_461480)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 20), tuple_461476, astype_call_result_461481)
    # Adding element type (line 304)
    # Getting the type of 'offsets' (line 304)
    offsets_461482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 37), 'offsets', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 20), tuple_461476, offsets_461482)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 304)
    tuple_461483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 304)
    # Adding element type (line 304)
    int_461484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 48), tuple_461483, int_461484)
    # Adding element type (line 304)
    int_461485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 48), tuple_461483, int_461485)
    
    # Processing the call keyword arguments (line 304)
    kwargs_461486 = {}
    # Getting the type of 'dia_matrix' (line 304)
    dia_matrix_461475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'dia_matrix', False)
    # Calling dia_matrix(args, kwargs) (line 304)
    dia_matrix_call_result_461487 = invoke(stypy.reporting.localization.Localization(__file__, 304, 8), dia_matrix_461475, *[tuple_461476, tuple_461483], **kwargs_461486)
    
    # Assigning a type to the variable 'b' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'b', dia_matrix_call_result_461487)
    
    # Assigning a Call to a Name (line 305):
    
    # Call to arange(...): (line 305)
    # Processing the call arguments (line 305)
    int_461490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 18), 'int')
    # Processing the call keyword arguments (line 305)
    kwargs_461491 = {}
    # Getting the type of 'np' (line 305)
    np_461488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 305)
    arange_461489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 8), np_461488, 'arange')
    # Calling arange(args, kwargs) (line 305)
    arange_call_result_461492 = invoke(stypy.reporting.localization.Localization(__file__, 305, 8), arange_461489, *[int_461490], **kwargs_461491)
    
    # Assigning a type to the variable 'v' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'v', arange_call_result_461492)
    
    # Call to assert_allclose(...): (line 307)
    # Processing the call arguments (line 307)
    
    # Call to dot(...): (line 307)
    # Processing the call arguments (line 307)
    # Getting the type of 'v' (line 307)
    v_461496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 26), 'v', False)
    # Processing the call keyword arguments (line 307)
    kwargs_461497 = {}
    # Getting the type of 'a' (line 307)
    a_461494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 20), 'a', False)
    # Obtaining the member 'dot' of a type (line 307)
    dot_461495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 20), a_461494, 'dot')
    # Calling dot(args, kwargs) (line 307)
    dot_call_result_461498 = invoke(stypy.reporting.localization.Localization(__file__, 307, 20), dot_461495, *[v_461496], **kwargs_461497)
    
    
    # Obtaining an instance of the builtin type 'list' (line 307)
    list_461499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 307)
    # Adding element type (line 307)
    int_461500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 30), list_461499, int_461500)
    # Adding element type (line 307)
    int_461501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 30), list_461499, int_461501)
    # Adding element type (line 307)
    int_461502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 30), list_461499, int_461502)
    # Adding element type (line 307)
    int_461503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 30), list_461499, int_461503)
    
    # Processing the call keyword arguments (line 307)
    kwargs_461504 = {}
    # Getting the type of 'assert_allclose' (line 307)
    assert_allclose_461493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 307)
    assert_allclose_call_result_461505 = invoke(stypy.reporting.localization.Localization(__file__, 307, 4), assert_allclose_461493, *[dot_call_result_461498, list_461499], **kwargs_461504)
    
    
    # Call to assert_allclose(...): (line 308)
    # Processing the call arguments (line 308)
    
    # Call to dot(...): (line 308)
    # Processing the call arguments (line 308)
    # Getting the type of 'v' (line 308)
    v_461509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 26), 'v', False)
    # Processing the call keyword arguments (line 308)
    kwargs_461510 = {}
    # Getting the type of 'b' (line 308)
    b_461507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 20), 'b', False)
    # Obtaining the member 'dot' of a type (line 308)
    dot_461508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 20), b_461507, 'dot')
    # Calling dot(args, kwargs) (line 308)
    dot_call_result_461511 = invoke(stypy.reporting.localization.Localization(__file__, 308, 20), dot_461508, *[v_461509], **kwargs_461510)
    
    
    # Obtaining an instance of the builtin type 'list' (line 308)
    list_461512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 308)
    # Adding element type (line 308)
    int_461513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 30), list_461512, int_461513)
    # Adding element type (line 308)
    int_461514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 30), list_461512, int_461514)
    # Adding element type (line 308)
    int_461515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 30), list_461512, int_461515)
    # Adding element type (line 308)
    int_461516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 30), list_461512, int_461516)
    
    # Processing the call keyword arguments (line 308)
    kwargs_461517 = {}
    # Getting the type of 'assert_allclose' (line 308)
    assert_allclose_461506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 308)
    assert_allclose_call_result_461518 = invoke(stypy.reporting.localization.Localization(__file__, 308, 4), assert_allclose_461506, *[dot_call_result_461511, list_461512], **kwargs_461517)
    
    
    # ################# End of 'test_endianness(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_endianness' in the type store
    # Getting the type of 'stypy_return_type' (line 299)
    stypy_return_type_461519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_461519)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_endianness'
    return stypy_return_type_461519

# Assigning a type to the variable 'test_endianness' (line 299)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 0), 'test_endianness', test_endianness)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
