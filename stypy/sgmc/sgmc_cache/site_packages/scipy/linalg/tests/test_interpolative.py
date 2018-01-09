
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #******************************************************************************
2: #   Copyright (C) 2013 Kenneth L. Ho
3: #   Redistribution and use in source and binary forms, with or without
4: #   modification, are permitted provided that the following conditions are met:
5: #
6: #   Redistributions of source code must retain the above copyright notice, this
7: #   list of conditions and the following disclaimer. Redistributions in binary
8: #   form must reproduce the above copyright notice, this list of conditions and
9: #   the following disclaimer in the documentation and/or other materials
10: #   provided with the distribution.
11: #
12: #   None of the names of the copyright holders may be used to endorse or
13: #   promote products derived from this software without specific prior written
14: #   permission.
15: #
16: #   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
17: #   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
18: #   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
19: #   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
20: #   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
21: #   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
22: #   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
23: #   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
24: #   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
25: #   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
26: #   POSSIBILITY OF SUCH DAMAGE.
27: #******************************************************************************
28: 
29: import scipy.linalg.interpolative as pymatrixid
30: import numpy as np
31: from scipy.linalg import hilbert, svdvals, norm
32: from scipy.sparse.linalg import aslinearoperator
33: import time
34: 
35: from numpy.testing import assert_, assert_allclose
36: from pytest import raises as assert_raises
37: 
38: 
39: def _debug_print(s):
40:     if 0:
41:         print(s)
42: 
43: 
44: class TestInterpolativeDecomposition(object):
45:     def test_id(self):
46:         for dtype in [np.float64, np.complex128]:
47:             self.check_id(dtype)
48: 
49:     def check_id(self, dtype):
50:         # Test ID routines on a Hilbert matrix.
51: 
52:         # set parameters
53:         n = 300
54:         eps = 1e-12
55: 
56:         # construct Hilbert matrix
57:         A = hilbert(n).astype(dtype)
58:         if np.issubdtype(dtype, np.complexfloating):
59:             A = A * (1 + 1j)
60:         L = aslinearoperator(A)
61: 
62:         # find rank
63:         S = np.linalg.svd(A, compute_uv=False)
64:         try:
65:             rank = np.nonzero(S < eps)[0][0]
66:         except:
67:             rank = n
68: 
69:         # print input summary
70:         _debug_print("Hilbert matrix dimension:        %8i" % n)
71:         _debug_print("Working precision:               %8.2e" % eps)
72:         _debug_print("Rank to working precision:       %8i" % rank)
73: 
74:         # set print format
75:         fmt = "%8.2e (s) / %5s"
76: 
77:         # test real ID routines
78:         _debug_print("-----------------------------------------")
79:         _debug_print("Real ID routines")
80:         _debug_print("-----------------------------------------")
81: 
82:         # fixed precision
83:         _debug_print("Calling iddp_id / idzp_id  ...",)
84:         t0 = time.clock()
85:         k, idx, proj = pymatrixid.interp_decomp(A, eps, rand=False)
86:         t = time.clock() - t0
87:         B = pymatrixid.reconstruct_matrix_from_id(A[:, idx[:k]], idx, proj)
88:         _debug_print(fmt % (t, np.allclose(A, B, eps)))
89:         assert_(np.allclose(A, B, eps))
90: 
91:         _debug_print("Calling iddp_aid / idzp_aid ...",)
92:         t0 = time.clock()
93:         k, idx, proj = pymatrixid.interp_decomp(A, eps)
94:         t = time.clock() - t0
95:         B = pymatrixid.reconstruct_matrix_from_id(A[:, idx[:k]], idx, proj)
96:         _debug_print(fmt % (t, np.allclose(A, B, eps)))
97:         assert_(np.allclose(A, B, eps))
98: 
99:         _debug_print("Calling iddp_rid / idzp_rid ...",)
100:         t0 = time.clock()
101:         k, idx, proj = pymatrixid.interp_decomp(L, eps)
102:         t = time.clock() - t0
103:         B = pymatrixid.reconstruct_matrix_from_id(A[:, idx[:k]], idx, proj)
104:         _debug_print(fmt % (t, np.allclose(A, B, eps)))
105:         assert_(np.allclose(A, B, eps))
106: 
107:         # fixed rank
108:         k = rank
109: 
110:         _debug_print("Calling iddr_id / idzr_id  ...",)
111:         t0 = time.clock()
112:         idx, proj = pymatrixid.interp_decomp(A, k, rand=False)
113:         t = time.clock() - t0
114:         B = pymatrixid.reconstruct_matrix_from_id(A[:, idx[:k]], idx, proj)
115:         _debug_print(fmt % (t, np.allclose(A, B, eps)))
116:         assert_(np.allclose(A, B, eps))
117: 
118:         _debug_print("Calling iddr_aid / idzr_aid ...",)
119:         t0 = time.clock()
120:         idx, proj = pymatrixid.interp_decomp(A, k)
121:         t = time.clock() - t0
122:         B = pymatrixid.reconstruct_matrix_from_id(A[:, idx[:k]], idx, proj)
123:         _debug_print(fmt % (t, np.allclose(A, B, eps)))
124:         assert_(np.allclose(A, B, eps))
125: 
126:         _debug_print("Calling iddr_rid / idzr_rid ...",)
127:         t0 = time.clock()
128:         idx, proj = pymatrixid.interp_decomp(L, k)
129:         t = time.clock() - t0
130:         B = pymatrixid.reconstruct_matrix_from_id(A[:, idx[:k]], idx, proj)
131:         _debug_print(fmt % (t, np.allclose(A, B, eps)))
132:         assert_(np.allclose(A, B, eps))
133: 
134:         # check skeleton and interpolation matrices
135:         idx, proj = pymatrixid.interp_decomp(A, k, rand=False)
136:         P = pymatrixid.reconstruct_interp_matrix(idx, proj)
137:         B = pymatrixid.reconstruct_skel_matrix(A, k, idx)
138:         assert_(np.allclose(B, A[:,idx[:k]], eps))
139:         assert_(np.allclose(B.dot(P), A, eps))
140: 
141:         # test SVD routines
142:         _debug_print("-----------------------------------------")
143:         _debug_print("SVD routines")
144:         _debug_print("-----------------------------------------")
145: 
146:         # fixed precision
147:         _debug_print("Calling iddp_svd / idzp_svd ...",)
148:         t0 = time.clock()
149:         U, S, V = pymatrixid.svd(A, eps, rand=False)
150:         t = time.clock() - t0
151:         B = np.dot(U, np.dot(np.diag(S), V.T.conj()))
152:         _debug_print(fmt % (t, np.allclose(A, B, eps)))
153:         assert_(np.allclose(A, B, eps))
154: 
155:         _debug_print("Calling iddp_asvd / idzp_asvd...",)
156:         t0 = time.clock()
157:         U, S, V = pymatrixid.svd(A, eps)
158:         t = time.clock() - t0
159:         B = np.dot(U, np.dot(np.diag(S), V.T.conj()))
160:         _debug_print(fmt % (t, np.allclose(A, B, eps)))
161:         assert_(np.allclose(A, B, eps))
162: 
163:         _debug_print("Calling iddp_rsvd / idzp_rsvd...",)
164:         t0 = time.clock()
165:         U, S, V = pymatrixid.svd(L, eps)
166:         t = time.clock() - t0
167:         B = np.dot(U, np.dot(np.diag(S), V.T.conj()))
168:         _debug_print(fmt % (t, np.allclose(A, B, eps)))
169:         assert_(np.allclose(A, B, eps))
170: 
171:         # fixed rank
172:         k = rank
173: 
174:         _debug_print("Calling iddr_svd / idzr_svd ...",)
175:         t0 = time.clock()
176:         U, S, V = pymatrixid.svd(A, k, rand=False)
177:         t = time.clock() - t0
178:         B = np.dot(U, np.dot(np.diag(S), V.T.conj()))
179:         _debug_print(fmt % (t, np.allclose(A, B, eps)))
180:         assert_(np.allclose(A, B, eps))
181: 
182:         _debug_print("Calling iddr_asvd / idzr_asvd ...",)
183:         t0 = time.clock()
184:         U, S, V = pymatrixid.svd(A, k)
185:         t = time.clock() - t0
186:         B = np.dot(U, np.dot(np.diag(S), V.T.conj()))
187:         _debug_print(fmt % (t, np.allclose(A, B, eps)))
188:         assert_(np.allclose(A, B, eps))
189: 
190:         _debug_print("Calling iddr_rsvd / idzr_rsvd ...",)
191:         t0 = time.clock()
192:         U, S, V = pymatrixid.svd(L, k)
193:         t = time.clock() - t0
194:         B = np.dot(U, np.dot(np.diag(S), V.T.conj()))
195:         _debug_print(fmt % (t, np.allclose(A, B, eps)))
196:         assert_(np.allclose(A, B, eps))
197: 
198:         # ID to SVD
199:         idx, proj = pymatrixid.interp_decomp(A, k, rand=False)
200:         Up, Sp, Vp = pymatrixid.id_to_svd(A[:, idx[:k]], idx, proj)
201:         B = U.dot(np.diag(S).dot(V.T.conj()))
202:         assert_(np.allclose(A, B, eps))
203: 
204:         # Norm estimates
205:         s = svdvals(A)
206:         norm_2_est = pymatrixid.estimate_spectral_norm(A)
207:         assert_(np.allclose(norm_2_est, s[0], 1e-6))
208: 
209:         B = A.copy()
210:         B[:,0] *= 1.2
211:         s = svdvals(A - B)
212:         norm_2_est = pymatrixid.estimate_spectral_norm_diff(A, B)
213:         assert_(np.allclose(norm_2_est, s[0], 1e-6))
214: 
215:         # Rank estimates
216:         B = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=dtype)
217:         for M in [A, B]:
218:             ML = aslinearoperator(M)
219: 
220:             rank_tol = 1e-9
221:             rank_np = np.linalg.matrix_rank(M, norm(M, 2)*rank_tol)
222:             rank_est = pymatrixid.estimate_rank(M, rank_tol)
223:             rank_est_2 = pymatrixid.estimate_rank(ML, rank_tol)
224: 
225:             assert_(rank_est >= rank_np)
226:             assert_(rank_est <= rank_np + 10)
227: 
228:             assert_(rank_est_2 >= rank_np - 4)
229:             assert_(rank_est_2 <= rank_np + 4)
230: 
231:     def test_rand(self):
232:         pymatrixid.seed('default')
233:         assert_(np.allclose(pymatrixid.rand(2), [0.8932059, 0.64500803], 1e-4))
234: 
235:         pymatrixid.seed(1234)
236:         x1 = pymatrixid.rand(2)
237:         assert_(np.allclose(x1, [0.7513823, 0.06861718], 1e-4))
238: 
239:         np.random.seed(1234)
240:         pymatrixid.seed()
241:         x2 = pymatrixid.rand(2)
242: 
243:         np.random.seed(1234)
244:         pymatrixid.seed(np.random.rand(55))
245:         x3 = pymatrixid.rand(2)
246: 
247:         assert_allclose(x1, x2)
248:         assert_allclose(x1, x3)
249: 
250:     def test_badcall(self):
251:         A = hilbert(5).astype(np.float32)
252:         assert_raises(ValueError, pymatrixid.interp_decomp, A, 1e-6, rand=False)
253: 
254:     def test_rank_too_large(self):
255:         # svd(array, k) should not segfault
256:         a = np.ones((4, 3))
257:         with assert_raises(ValueError):
258:             pymatrixid.svd(a, 4)
259: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'import scipy.linalg.interpolative' statement (line 29)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_93472 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'scipy.linalg.interpolative')

if (type(import_93472) is not StypyTypeError):

    if (import_93472 != 'pyd_module'):
        __import__(import_93472)
        sys_modules_93473 = sys.modules[import_93472]
        import_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'pymatrixid', sys_modules_93473.module_type_store, module_type_store)
    else:
        import scipy.linalg.interpolative as pymatrixid

        import_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'pymatrixid', scipy.linalg.interpolative, module_type_store)

else:
    # Assigning a type to the variable 'scipy.linalg.interpolative' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'scipy.linalg.interpolative', import_93472)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'import numpy' statement (line 30)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_93474 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy')

if (type(import_93474) is not StypyTypeError):

    if (import_93474 != 'pyd_module'):
        __import__(import_93474)
        sys_modules_93475 = sys.modules[import_93474]
        import_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'np', sys_modules_93475.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy', import_93474)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 31, 0))

# 'from scipy.linalg import hilbert, svdvals, norm' statement (line 31)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_93476 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'scipy.linalg')

if (type(import_93476) is not StypyTypeError):

    if (import_93476 != 'pyd_module'):
        __import__(import_93476)
        sys_modules_93477 = sys.modules[import_93476]
        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'scipy.linalg', sys_modules_93477.module_type_store, module_type_store, ['hilbert', 'svdvals', 'norm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 31, 0), __file__, sys_modules_93477, sys_modules_93477.module_type_store, module_type_store)
    else:
        from scipy.linalg import hilbert, svdvals, norm

        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'scipy.linalg', None, module_type_store, ['hilbert', 'svdvals', 'norm'], [hilbert, svdvals, norm])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'scipy.linalg', import_93476)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 0))

# 'from scipy.sparse.linalg import aslinearoperator' statement (line 32)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_93478 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'scipy.sparse.linalg')

if (type(import_93478) is not StypyTypeError):

    if (import_93478 != 'pyd_module'):
        __import__(import_93478)
        sys_modules_93479 = sys.modules[import_93478]
        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'scipy.sparse.linalg', sys_modules_93479.module_type_store, module_type_store, ['aslinearoperator'])
        nest_module(stypy.reporting.localization.Localization(__file__, 32, 0), __file__, sys_modules_93479, sys_modules_93479.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import aslinearoperator

        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'scipy.sparse.linalg', None, module_type_store, ['aslinearoperator'], [aslinearoperator])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'scipy.sparse.linalg', import_93478)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 0))

# 'import time' statement (line 33)
import time

import_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'time', time, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 0))

# 'from numpy.testing import assert_, assert_allclose' statement (line 35)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_93480 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'numpy.testing')

if (type(import_93480) is not StypyTypeError):

    if (import_93480 != 'pyd_module'):
        __import__(import_93480)
        sys_modules_93481 = sys.modules[import_93480]
        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'numpy.testing', sys_modules_93481.module_type_store, module_type_store, ['assert_', 'assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 35, 0), __file__, sys_modules_93481, sys_modules_93481.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_allclose'], [assert_, assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'numpy.testing', import_93480)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 36, 0))

# 'from pytest import assert_raises' statement (line 36)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_93482 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'pytest')

if (type(import_93482) is not StypyTypeError):

    if (import_93482 != 'pyd_module'):
        __import__(import_93482)
        sys_modules_93483 = sys.modules[import_93482]
        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'pytest', sys_modules_93483.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 36, 0), __file__, sys_modules_93483, sys_modules_93483.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'pytest', import_93482)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')


@norecursion
def _debug_print(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_debug_print'
    module_type_store = module_type_store.open_function_context('_debug_print', 39, 0, False)
    
    # Passed parameters checking function
    _debug_print.stypy_localization = localization
    _debug_print.stypy_type_of_self = None
    _debug_print.stypy_type_store = module_type_store
    _debug_print.stypy_function_name = '_debug_print'
    _debug_print.stypy_param_names_list = ['s']
    _debug_print.stypy_varargs_param_name = None
    _debug_print.stypy_kwargs_param_name = None
    _debug_print.stypy_call_defaults = defaults
    _debug_print.stypy_call_varargs = varargs
    _debug_print.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_debug_print', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_debug_print', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_debug_print(...)' code ##################

    
    int_93484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 7), 'int')
    # Testing the type of an if condition (line 40)
    if_condition_93485 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 4), int_93484)
    # Assigning a type to the variable 'if_condition_93485' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'if_condition_93485', if_condition_93485)
    # SSA begins for if statement (line 40)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 's' (line 41)
    s_93486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 14), 's')
    # SSA join for if statement (line 40)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_debug_print(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_debug_print' in the type store
    # Getting the type of 'stypy_return_type' (line 39)
    stypy_return_type_93487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_93487)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_debug_print'
    return stypy_return_type_93487

# Assigning a type to the variable '_debug_print' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), '_debug_print', _debug_print)
# Declaration of the 'TestInterpolativeDecomposition' class

class TestInterpolativeDecomposition(object, ):

    @norecursion
    def test_id(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_id'
        module_type_store = module_type_store.open_function_context('test_id', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestInterpolativeDecomposition.test_id.__dict__.__setitem__('stypy_localization', localization)
        TestInterpolativeDecomposition.test_id.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestInterpolativeDecomposition.test_id.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestInterpolativeDecomposition.test_id.__dict__.__setitem__('stypy_function_name', 'TestInterpolativeDecomposition.test_id')
        TestInterpolativeDecomposition.test_id.__dict__.__setitem__('stypy_param_names_list', [])
        TestInterpolativeDecomposition.test_id.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestInterpolativeDecomposition.test_id.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestInterpolativeDecomposition.test_id.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestInterpolativeDecomposition.test_id.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestInterpolativeDecomposition.test_id.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestInterpolativeDecomposition.test_id.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInterpolativeDecomposition.test_id', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_id', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_id(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 46)
        list_93488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 46)
        # Adding element type (line 46)
        # Getting the type of 'np' (line 46)
        np_93489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'np')
        # Obtaining the member 'float64' of a type (line 46)
        float64_93490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 22), np_93489, 'float64')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 21), list_93488, float64_93490)
        # Adding element type (line 46)
        # Getting the type of 'np' (line 46)
        np_93491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 34), 'np')
        # Obtaining the member 'complex128' of a type (line 46)
        complex128_93492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 34), np_93491, 'complex128')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 21), list_93488, complex128_93492)
        
        # Testing the type of a for loop iterable (line 46)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 46, 8), list_93488)
        # Getting the type of the for loop variable (line 46)
        for_loop_var_93493 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 46, 8), list_93488)
        # Assigning a type to the variable 'dtype' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'dtype', for_loop_var_93493)
        # SSA begins for a for statement (line 46)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to check_id(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'dtype' (line 47)
        dtype_93496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 26), 'dtype', False)
        # Processing the call keyword arguments (line 47)
        kwargs_93497 = {}
        # Getting the type of 'self' (line 47)
        self_93494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'self', False)
        # Obtaining the member 'check_id' of a type (line 47)
        check_id_93495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), self_93494, 'check_id')
        # Calling check_id(args, kwargs) (line 47)
        check_id_call_result_93498 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), check_id_93495, *[dtype_93496], **kwargs_93497)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_id(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_id' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_93499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_93499)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_id'
        return stypy_return_type_93499


    @norecursion
    def check_id(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_id'
        module_type_store = module_type_store.open_function_context('check_id', 49, 4, False)
        # Assigning a type to the variable 'self' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestInterpolativeDecomposition.check_id.__dict__.__setitem__('stypy_localization', localization)
        TestInterpolativeDecomposition.check_id.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestInterpolativeDecomposition.check_id.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestInterpolativeDecomposition.check_id.__dict__.__setitem__('stypy_function_name', 'TestInterpolativeDecomposition.check_id')
        TestInterpolativeDecomposition.check_id.__dict__.__setitem__('stypy_param_names_list', ['dtype'])
        TestInterpolativeDecomposition.check_id.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestInterpolativeDecomposition.check_id.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestInterpolativeDecomposition.check_id.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestInterpolativeDecomposition.check_id.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestInterpolativeDecomposition.check_id.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestInterpolativeDecomposition.check_id.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInterpolativeDecomposition.check_id', ['dtype'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_id', localization, ['dtype'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_id(...)' code ##################

        
        # Assigning a Num to a Name (line 53):
        
        # Assigning a Num to a Name (line 53):
        int_93500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 12), 'int')
        # Assigning a type to the variable 'n' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'n', int_93500)
        
        # Assigning a Num to a Name (line 54):
        
        # Assigning a Num to a Name (line 54):
        float_93501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 14), 'float')
        # Assigning a type to the variable 'eps' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'eps', float_93501)
        
        # Assigning a Call to a Name (line 57):
        
        # Assigning a Call to a Name (line 57):
        
        # Call to astype(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'dtype' (line 57)
        dtype_93507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 30), 'dtype', False)
        # Processing the call keyword arguments (line 57)
        kwargs_93508 = {}
        
        # Call to hilbert(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'n' (line 57)
        n_93503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 20), 'n', False)
        # Processing the call keyword arguments (line 57)
        kwargs_93504 = {}
        # Getting the type of 'hilbert' (line 57)
        hilbert_93502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'hilbert', False)
        # Calling hilbert(args, kwargs) (line 57)
        hilbert_call_result_93505 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), hilbert_93502, *[n_93503], **kwargs_93504)
        
        # Obtaining the member 'astype' of a type (line 57)
        astype_93506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), hilbert_call_result_93505, 'astype')
        # Calling astype(args, kwargs) (line 57)
        astype_call_result_93509 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), astype_93506, *[dtype_93507], **kwargs_93508)
        
        # Assigning a type to the variable 'A' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'A', astype_call_result_93509)
        
        
        # Call to issubdtype(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'dtype' (line 58)
        dtype_93512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 25), 'dtype', False)
        # Getting the type of 'np' (line 58)
        np_93513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 32), 'np', False)
        # Obtaining the member 'complexfloating' of a type (line 58)
        complexfloating_93514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 32), np_93513, 'complexfloating')
        # Processing the call keyword arguments (line 58)
        kwargs_93515 = {}
        # Getting the type of 'np' (line 58)
        np_93510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 11), 'np', False)
        # Obtaining the member 'issubdtype' of a type (line 58)
        issubdtype_93511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 11), np_93510, 'issubdtype')
        # Calling issubdtype(args, kwargs) (line 58)
        issubdtype_call_result_93516 = invoke(stypy.reporting.localization.Localization(__file__, 58, 11), issubdtype_93511, *[dtype_93512, complexfloating_93514], **kwargs_93515)
        
        # Testing the type of an if condition (line 58)
        if_condition_93517 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 8), issubdtype_call_result_93516)
        # Assigning a type to the variable 'if_condition_93517' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'if_condition_93517', if_condition_93517)
        # SSA begins for if statement (line 58)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 59):
        
        # Assigning a BinOp to a Name (line 59):
        # Getting the type of 'A' (line 59)
        A_93518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'A')
        int_93519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 21), 'int')
        complex_93520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 25), 'complex')
        # Applying the binary operator '+' (line 59)
        result_add_93521 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 21), '+', int_93519, complex_93520)
        
        # Applying the binary operator '*' (line 59)
        result_mul_93522 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 16), '*', A_93518, result_add_93521)
        
        # Assigning a type to the variable 'A' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'A', result_mul_93522)
        # SSA join for if statement (line 58)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 60):
        
        # Assigning a Call to a Name (line 60):
        
        # Call to aslinearoperator(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'A' (line 60)
        A_93524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 29), 'A', False)
        # Processing the call keyword arguments (line 60)
        kwargs_93525 = {}
        # Getting the type of 'aslinearoperator' (line 60)
        aslinearoperator_93523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'aslinearoperator', False)
        # Calling aslinearoperator(args, kwargs) (line 60)
        aslinearoperator_call_result_93526 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), aslinearoperator_93523, *[A_93524], **kwargs_93525)
        
        # Assigning a type to the variable 'L' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'L', aslinearoperator_call_result_93526)
        
        # Assigning a Call to a Name (line 63):
        
        # Assigning a Call to a Name (line 63):
        
        # Call to svd(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'A' (line 63)
        A_93530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 26), 'A', False)
        # Processing the call keyword arguments (line 63)
        # Getting the type of 'False' (line 63)
        False_93531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 40), 'False', False)
        keyword_93532 = False_93531
        kwargs_93533 = {'compute_uv': keyword_93532}
        # Getting the type of 'np' (line 63)
        np_93527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'np', False)
        # Obtaining the member 'linalg' of a type (line 63)
        linalg_93528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 12), np_93527, 'linalg')
        # Obtaining the member 'svd' of a type (line 63)
        svd_93529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 12), linalg_93528, 'svd')
        # Calling svd(args, kwargs) (line 63)
        svd_call_result_93534 = invoke(stypy.reporting.localization.Localization(__file__, 63, 12), svd_93529, *[A_93530], **kwargs_93533)
        
        # Assigning a type to the variable 'S' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'S', svd_call_result_93534)
        
        
        # SSA begins for try-except statement (line 64)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Name (line 65):
        
        # Assigning a Subscript to a Name (line 65):
        
        # Obtaining the type of the subscript
        int_93535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 42), 'int')
        
        # Obtaining the type of the subscript
        int_93536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 39), 'int')
        
        # Call to nonzero(...): (line 65)
        # Processing the call arguments (line 65)
        
        # Getting the type of 'S' (line 65)
        S_93539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 30), 'S', False)
        # Getting the type of 'eps' (line 65)
        eps_93540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 34), 'eps', False)
        # Applying the binary operator '<' (line 65)
        result_lt_93541 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 30), '<', S_93539, eps_93540)
        
        # Processing the call keyword arguments (line 65)
        kwargs_93542 = {}
        # Getting the type of 'np' (line 65)
        np_93537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'np', False)
        # Obtaining the member 'nonzero' of a type (line 65)
        nonzero_93538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 19), np_93537, 'nonzero')
        # Calling nonzero(args, kwargs) (line 65)
        nonzero_call_result_93543 = invoke(stypy.reporting.localization.Localization(__file__, 65, 19), nonzero_93538, *[result_lt_93541], **kwargs_93542)
        
        # Obtaining the member '__getitem__' of a type (line 65)
        getitem___93544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 19), nonzero_call_result_93543, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 65)
        subscript_call_result_93545 = invoke(stypy.reporting.localization.Localization(__file__, 65, 19), getitem___93544, int_93536)
        
        # Obtaining the member '__getitem__' of a type (line 65)
        getitem___93546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 19), subscript_call_result_93545, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 65)
        subscript_call_result_93547 = invoke(stypy.reporting.localization.Localization(__file__, 65, 19), getitem___93546, int_93535)
        
        # Assigning a type to the variable 'rank' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'rank', subscript_call_result_93547)
        # SSA branch for the except part of a try statement (line 64)
        # SSA branch for the except '<any exception>' branch of a try statement (line 64)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Name to a Name (line 67):
        
        # Assigning a Name to a Name (line 67):
        # Getting the type of 'n' (line 67)
        n_93548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 19), 'n')
        # Assigning a type to the variable 'rank' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'rank', n_93548)
        # SSA join for try-except statement (line 64)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _debug_print(...): (line 70)
        # Processing the call arguments (line 70)
        str_93550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 21), 'str', 'Hilbert matrix dimension:        %8i')
        # Getting the type of 'n' (line 70)
        n_93551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 62), 'n', False)
        # Applying the binary operator '%' (line 70)
        result_mod_93552 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 21), '%', str_93550, n_93551)
        
        # Processing the call keyword arguments (line 70)
        kwargs_93553 = {}
        # Getting the type of '_debug_print' (line 70)
        _debug_print_93549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 70)
        _debug_print_call_result_93554 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), _debug_print_93549, *[result_mod_93552], **kwargs_93553)
        
        
        # Call to _debug_print(...): (line 71)
        # Processing the call arguments (line 71)
        str_93556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 21), 'str', 'Working precision:               %8.2e')
        # Getting the type of 'eps' (line 71)
        eps_93557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 64), 'eps', False)
        # Applying the binary operator '%' (line 71)
        result_mod_93558 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 21), '%', str_93556, eps_93557)
        
        # Processing the call keyword arguments (line 71)
        kwargs_93559 = {}
        # Getting the type of '_debug_print' (line 71)
        _debug_print_93555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 71)
        _debug_print_call_result_93560 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), _debug_print_93555, *[result_mod_93558], **kwargs_93559)
        
        
        # Call to _debug_print(...): (line 72)
        # Processing the call arguments (line 72)
        str_93562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 21), 'str', 'Rank to working precision:       %8i')
        # Getting the type of 'rank' (line 72)
        rank_93563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 62), 'rank', False)
        # Applying the binary operator '%' (line 72)
        result_mod_93564 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 21), '%', str_93562, rank_93563)
        
        # Processing the call keyword arguments (line 72)
        kwargs_93565 = {}
        # Getting the type of '_debug_print' (line 72)
        _debug_print_93561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 72)
        _debug_print_call_result_93566 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), _debug_print_93561, *[result_mod_93564], **kwargs_93565)
        
        
        # Assigning a Str to a Name (line 75):
        
        # Assigning a Str to a Name (line 75):
        str_93567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 14), 'str', '%8.2e (s) / %5s')
        # Assigning a type to the variable 'fmt' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'fmt', str_93567)
        
        # Call to _debug_print(...): (line 78)
        # Processing the call arguments (line 78)
        str_93569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 21), 'str', '-----------------------------------------')
        # Processing the call keyword arguments (line 78)
        kwargs_93570 = {}
        # Getting the type of '_debug_print' (line 78)
        _debug_print_93568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 78)
        _debug_print_call_result_93571 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), _debug_print_93568, *[str_93569], **kwargs_93570)
        
        
        # Call to _debug_print(...): (line 79)
        # Processing the call arguments (line 79)
        str_93573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 21), 'str', 'Real ID routines')
        # Processing the call keyword arguments (line 79)
        kwargs_93574 = {}
        # Getting the type of '_debug_print' (line 79)
        _debug_print_93572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 79)
        _debug_print_call_result_93575 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), _debug_print_93572, *[str_93573], **kwargs_93574)
        
        
        # Call to _debug_print(...): (line 80)
        # Processing the call arguments (line 80)
        str_93577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 21), 'str', '-----------------------------------------')
        # Processing the call keyword arguments (line 80)
        kwargs_93578 = {}
        # Getting the type of '_debug_print' (line 80)
        _debug_print_93576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 80)
        _debug_print_call_result_93579 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), _debug_print_93576, *[str_93577], **kwargs_93578)
        
        
        # Call to _debug_print(...): (line 83)
        # Processing the call arguments (line 83)
        str_93581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 21), 'str', 'Calling iddp_id / idzp_id  ...')
        # Processing the call keyword arguments (line 83)
        kwargs_93582 = {}
        # Getting the type of '_debug_print' (line 83)
        _debug_print_93580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 83)
        _debug_print_call_result_93583 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), _debug_print_93580, *[str_93581], **kwargs_93582)
        
        
        # Assigning a Call to a Name (line 84):
        
        # Assigning a Call to a Name (line 84):
        
        # Call to clock(...): (line 84)
        # Processing the call keyword arguments (line 84)
        kwargs_93586 = {}
        # Getting the type of 'time' (line 84)
        time_93584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 13), 'time', False)
        # Obtaining the member 'clock' of a type (line 84)
        clock_93585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 13), time_93584, 'clock')
        # Calling clock(args, kwargs) (line 84)
        clock_call_result_93587 = invoke(stypy.reporting.localization.Localization(__file__, 84, 13), clock_93585, *[], **kwargs_93586)
        
        # Assigning a type to the variable 't0' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 't0', clock_call_result_93587)
        
        # Assigning a Call to a Tuple (line 85):
        
        # Assigning a Subscript to a Name (line 85):
        
        # Obtaining the type of the subscript
        int_93588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'int')
        
        # Call to interp_decomp(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'A' (line 85)
        A_93591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 48), 'A', False)
        # Getting the type of 'eps' (line 85)
        eps_93592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 51), 'eps', False)
        # Processing the call keyword arguments (line 85)
        # Getting the type of 'False' (line 85)
        False_93593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 61), 'False', False)
        keyword_93594 = False_93593
        kwargs_93595 = {'rand': keyword_93594}
        # Getting the type of 'pymatrixid' (line 85)
        pymatrixid_93589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 23), 'pymatrixid', False)
        # Obtaining the member 'interp_decomp' of a type (line 85)
        interp_decomp_93590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 23), pymatrixid_93589, 'interp_decomp')
        # Calling interp_decomp(args, kwargs) (line 85)
        interp_decomp_call_result_93596 = invoke(stypy.reporting.localization.Localization(__file__, 85, 23), interp_decomp_93590, *[A_93591, eps_93592], **kwargs_93595)
        
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___93597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), interp_decomp_call_result_93596, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_93598 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), getitem___93597, int_93588)
        
        # Assigning a type to the variable 'tuple_var_assignment_93432' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'tuple_var_assignment_93432', subscript_call_result_93598)
        
        # Assigning a Subscript to a Name (line 85):
        
        # Obtaining the type of the subscript
        int_93599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'int')
        
        # Call to interp_decomp(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'A' (line 85)
        A_93602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 48), 'A', False)
        # Getting the type of 'eps' (line 85)
        eps_93603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 51), 'eps', False)
        # Processing the call keyword arguments (line 85)
        # Getting the type of 'False' (line 85)
        False_93604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 61), 'False', False)
        keyword_93605 = False_93604
        kwargs_93606 = {'rand': keyword_93605}
        # Getting the type of 'pymatrixid' (line 85)
        pymatrixid_93600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 23), 'pymatrixid', False)
        # Obtaining the member 'interp_decomp' of a type (line 85)
        interp_decomp_93601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 23), pymatrixid_93600, 'interp_decomp')
        # Calling interp_decomp(args, kwargs) (line 85)
        interp_decomp_call_result_93607 = invoke(stypy.reporting.localization.Localization(__file__, 85, 23), interp_decomp_93601, *[A_93602, eps_93603], **kwargs_93606)
        
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___93608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), interp_decomp_call_result_93607, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_93609 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), getitem___93608, int_93599)
        
        # Assigning a type to the variable 'tuple_var_assignment_93433' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'tuple_var_assignment_93433', subscript_call_result_93609)
        
        # Assigning a Subscript to a Name (line 85):
        
        # Obtaining the type of the subscript
        int_93610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'int')
        
        # Call to interp_decomp(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'A' (line 85)
        A_93613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 48), 'A', False)
        # Getting the type of 'eps' (line 85)
        eps_93614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 51), 'eps', False)
        # Processing the call keyword arguments (line 85)
        # Getting the type of 'False' (line 85)
        False_93615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 61), 'False', False)
        keyword_93616 = False_93615
        kwargs_93617 = {'rand': keyword_93616}
        # Getting the type of 'pymatrixid' (line 85)
        pymatrixid_93611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 23), 'pymatrixid', False)
        # Obtaining the member 'interp_decomp' of a type (line 85)
        interp_decomp_93612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 23), pymatrixid_93611, 'interp_decomp')
        # Calling interp_decomp(args, kwargs) (line 85)
        interp_decomp_call_result_93618 = invoke(stypy.reporting.localization.Localization(__file__, 85, 23), interp_decomp_93612, *[A_93613, eps_93614], **kwargs_93617)
        
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___93619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), interp_decomp_call_result_93618, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_93620 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), getitem___93619, int_93610)
        
        # Assigning a type to the variable 'tuple_var_assignment_93434' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'tuple_var_assignment_93434', subscript_call_result_93620)
        
        # Assigning a Name to a Name (line 85):
        # Getting the type of 'tuple_var_assignment_93432' (line 85)
        tuple_var_assignment_93432_93621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'tuple_var_assignment_93432')
        # Assigning a type to the variable 'k' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'k', tuple_var_assignment_93432_93621)
        
        # Assigning a Name to a Name (line 85):
        # Getting the type of 'tuple_var_assignment_93433' (line 85)
        tuple_var_assignment_93433_93622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'tuple_var_assignment_93433')
        # Assigning a type to the variable 'idx' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 11), 'idx', tuple_var_assignment_93433_93622)
        
        # Assigning a Name to a Name (line 85):
        # Getting the type of 'tuple_var_assignment_93434' (line 85)
        tuple_var_assignment_93434_93623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'tuple_var_assignment_93434')
        # Assigning a type to the variable 'proj' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'proj', tuple_var_assignment_93434_93623)
        
        # Assigning a BinOp to a Name (line 86):
        
        # Assigning a BinOp to a Name (line 86):
        
        # Call to clock(...): (line 86)
        # Processing the call keyword arguments (line 86)
        kwargs_93626 = {}
        # Getting the type of 'time' (line 86)
        time_93624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'time', False)
        # Obtaining the member 'clock' of a type (line 86)
        clock_93625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), time_93624, 'clock')
        # Calling clock(args, kwargs) (line 86)
        clock_call_result_93627 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), clock_93625, *[], **kwargs_93626)
        
        # Getting the type of 't0' (line 86)
        t0_93628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 27), 't0')
        # Applying the binary operator '-' (line 86)
        result_sub_93629 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 12), '-', clock_call_result_93627, t0_93628)
        
        # Assigning a type to the variable 't' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 't', result_sub_93629)
        
        # Assigning a Call to a Name (line 87):
        
        # Assigning a Call to a Name (line 87):
        
        # Call to reconstruct_matrix_from_id(...): (line 87)
        # Processing the call arguments (line 87)
        
        # Obtaining the type of the subscript
        slice_93632 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 87, 50), None, None, None)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 87)
        k_93633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 60), 'k', False)
        slice_93634 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 87, 55), None, k_93633, None)
        # Getting the type of 'idx' (line 87)
        idx_93635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 55), 'idx', False)
        # Obtaining the member '__getitem__' of a type (line 87)
        getitem___93636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 55), idx_93635, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 87)
        subscript_call_result_93637 = invoke(stypy.reporting.localization.Localization(__file__, 87, 55), getitem___93636, slice_93634)
        
        # Getting the type of 'A' (line 87)
        A_93638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 50), 'A', False)
        # Obtaining the member '__getitem__' of a type (line 87)
        getitem___93639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 50), A_93638, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 87)
        subscript_call_result_93640 = invoke(stypy.reporting.localization.Localization(__file__, 87, 50), getitem___93639, (slice_93632, subscript_call_result_93637))
        
        # Getting the type of 'idx' (line 87)
        idx_93641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 65), 'idx', False)
        # Getting the type of 'proj' (line 87)
        proj_93642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 70), 'proj', False)
        # Processing the call keyword arguments (line 87)
        kwargs_93643 = {}
        # Getting the type of 'pymatrixid' (line 87)
        pymatrixid_93630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'pymatrixid', False)
        # Obtaining the member 'reconstruct_matrix_from_id' of a type (line 87)
        reconstruct_matrix_from_id_93631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 12), pymatrixid_93630, 'reconstruct_matrix_from_id')
        # Calling reconstruct_matrix_from_id(args, kwargs) (line 87)
        reconstruct_matrix_from_id_call_result_93644 = invoke(stypy.reporting.localization.Localization(__file__, 87, 12), reconstruct_matrix_from_id_93631, *[subscript_call_result_93640, idx_93641, proj_93642], **kwargs_93643)
        
        # Assigning a type to the variable 'B' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'B', reconstruct_matrix_from_id_call_result_93644)
        
        # Call to _debug_print(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'fmt' (line 88)
        fmt_93646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 21), 'fmt', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 88)
        tuple_93647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 88)
        # Adding element type (line 88)
        # Getting the type of 't' (line 88)
        t_93648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 28), 't', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 28), tuple_93647, t_93648)
        # Adding element type (line 88)
        
        # Call to allclose(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'A' (line 88)
        A_93651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 43), 'A', False)
        # Getting the type of 'B' (line 88)
        B_93652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 46), 'B', False)
        # Getting the type of 'eps' (line 88)
        eps_93653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 49), 'eps', False)
        # Processing the call keyword arguments (line 88)
        kwargs_93654 = {}
        # Getting the type of 'np' (line 88)
        np_93649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 31), 'np', False)
        # Obtaining the member 'allclose' of a type (line 88)
        allclose_93650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 31), np_93649, 'allclose')
        # Calling allclose(args, kwargs) (line 88)
        allclose_call_result_93655 = invoke(stypy.reporting.localization.Localization(__file__, 88, 31), allclose_93650, *[A_93651, B_93652, eps_93653], **kwargs_93654)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 28), tuple_93647, allclose_call_result_93655)
        
        # Applying the binary operator '%' (line 88)
        result_mod_93656 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 21), '%', fmt_93646, tuple_93647)
        
        # Processing the call keyword arguments (line 88)
        kwargs_93657 = {}
        # Getting the type of '_debug_print' (line 88)
        _debug_print_93645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 88)
        _debug_print_call_result_93658 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), _debug_print_93645, *[result_mod_93656], **kwargs_93657)
        
        
        # Call to assert_(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Call to allclose(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'A' (line 89)
        A_93662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 28), 'A', False)
        # Getting the type of 'B' (line 89)
        B_93663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 31), 'B', False)
        # Getting the type of 'eps' (line 89)
        eps_93664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 34), 'eps', False)
        # Processing the call keyword arguments (line 89)
        kwargs_93665 = {}
        # Getting the type of 'np' (line 89)
        np_93660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'np', False)
        # Obtaining the member 'allclose' of a type (line 89)
        allclose_93661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 16), np_93660, 'allclose')
        # Calling allclose(args, kwargs) (line 89)
        allclose_call_result_93666 = invoke(stypy.reporting.localization.Localization(__file__, 89, 16), allclose_93661, *[A_93662, B_93663, eps_93664], **kwargs_93665)
        
        # Processing the call keyword arguments (line 89)
        kwargs_93667 = {}
        # Getting the type of 'assert_' (line 89)
        assert__93659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 89)
        assert__call_result_93668 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), assert__93659, *[allclose_call_result_93666], **kwargs_93667)
        
        
        # Call to _debug_print(...): (line 91)
        # Processing the call arguments (line 91)
        str_93670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 21), 'str', 'Calling iddp_aid / idzp_aid ...')
        # Processing the call keyword arguments (line 91)
        kwargs_93671 = {}
        # Getting the type of '_debug_print' (line 91)
        _debug_print_93669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 91)
        _debug_print_call_result_93672 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), _debug_print_93669, *[str_93670], **kwargs_93671)
        
        
        # Assigning a Call to a Name (line 92):
        
        # Assigning a Call to a Name (line 92):
        
        # Call to clock(...): (line 92)
        # Processing the call keyword arguments (line 92)
        kwargs_93675 = {}
        # Getting the type of 'time' (line 92)
        time_93673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 13), 'time', False)
        # Obtaining the member 'clock' of a type (line 92)
        clock_93674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 13), time_93673, 'clock')
        # Calling clock(args, kwargs) (line 92)
        clock_call_result_93676 = invoke(stypy.reporting.localization.Localization(__file__, 92, 13), clock_93674, *[], **kwargs_93675)
        
        # Assigning a type to the variable 't0' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 't0', clock_call_result_93676)
        
        # Assigning a Call to a Tuple (line 93):
        
        # Assigning a Subscript to a Name (line 93):
        
        # Obtaining the type of the subscript
        int_93677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 8), 'int')
        
        # Call to interp_decomp(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'A' (line 93)
        A_93680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 48), 'A', False)
        # Getting the type of 'eps' (line 93)
        eps_93681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 51), 'eps', False)
        # Processing the call keyword arguments (line 93)
        kwargs_93682 = {}
        # Getting the type of 'pymatrixid' (line 93)
        pymatrixid_93678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 23), 'pymatrixid', False)
        # Obtaining the member 'interp_decomp' of a type (line 93)
        interp_decomp_93679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 23), pymatrixid_93678, 'interp_decomp')
        # Calling interp_decomp(args, kwargs) (line 93)
        interp_decomp_call_result_93683 = invoke(stypy.reporting.localization.Localization(__file__, 93, 23), interp_decomp_93679, *[A_93680, eps_93681], **kwargs_93682)
        
        # Obtaining the member '__getitem__' of a type (line 93)
        getitem___93684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), interp_decomp_call_result_93683, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 93)
        subscript_call_result_93685 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), getitem___93684, int_93677)
        
        # Assigning a type to the variable 'tuple_var_assignment_93435' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'tuple_var_assignment_93435', subscript_call_result_93685)
        
        # Assigning a Subscript to a Name (line 93):
        
        # Obtaining the type of the subscript
        int_93686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 8), 'int')
        
        # Call to interp_decomp(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'A' (line 93)
        A_93689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 48), 'A', False)
        # Getting the type of 'eps' (line 93)
        eps_93690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 51), 'eps', False)
        # Processing the call keyword arguments (line 93)
        kwargs_93691 = {}
        # Getting the type of 'pymatrixid' (line 93)
        pymatrixid_93687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 23), 'pymatrixid', False)
        # Obtaining the member 'interp_decomp' of a type (line 93)
        interp_decomp_93688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 23), pymatrixid_93687, 'interp_decomp')
        # Calling interp_decomp(args, kwargs) (line 93)
        interp_decomp_call_result_93692 = invoke(stypy.reporting.localization.Localization(__file__, 93, 23), interp_decomp_93688, *[A_93689, eps_93690], **kwargs_93691)
        
        # Obtaining the member '__getitem__' of a type (line 93)
        getitem___93693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), interp_decomp_call_result_93692, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 93)
        subscript_call_result_93694 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), getitem___93693, int_93686)
        
        # Assigning a type to the variable 'tuple_var_assignment_93436' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'tuple_var_assignment_93436', subscript_call_result_93694)
        
        # Assigning a Subscript to a Name (line 93):
        
        # Obtaining the type of the subscript
        int_93695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 8), 'int')
        
        # Call to interp_decomp(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'A' (line 93)
        A_93698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 48), 'A', False)
        # Getting the type of 'eps' (line 93)
        eps_93699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 51), 'eps', False)
        # Processing the call keyword arguments (line 93)
        kwargs_93700 = {}
        # Getting the type of 'pymatrixid' (line 93)
        pymatrixid_93696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 23), 'pymatrixid', False)
        # Obtaining the member 'interp_decomp' of a type (line 93)
        interp_decomp_93697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 23), pymatrixid_93696, 'interp_decomp')
        # Calling interp_decomp(args, kwargs) (line 93)
        interp_decomp_call_result_93701 = invoke(stypy.reporting.localization.Localization(__file__, 93, 23), interp_decomp_93697, *[A_93698, eps_93699], **kwargs_93700)
        
        # Obtaining the member '__getitem__' of a type (line 93)
        getitem___93702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), interp_decomp_call_result_93701, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 93)
        subscript_call_result_93703 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), getitem___93702, int_93695)
        
        # Assigning a type to the variable 'tuple_var_assignment_93437' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'tuple_var_assignment_93437', subscript_call_result_93703)
        
        # Assigning a Name to a Name (line 93):
        # Getting the type of 'tuple_var_assignment_93435' (line 93)
        tuple_var_assignment_93435_93704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'tuple_var_assignment_93435')
        # Assigning a type to the variable 'k' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'k', tuple_var_assignment_93435_93704)
        
        # Assigning a Name to a Name (line 93):
        # Getting the type of 'tuple_var_assignment_93436' (line 93)
        tuple_var_assignment_93436_93705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'tuple_var_assignment_93436')
        # Assigning a type to the variable 'idx' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 11), 'idx', tuple_var_assignment_93436_93705)
        
        # Assigning a Name to a Name (line 93):
        # Getting the type of 'tuple_var_assignment_93437' (line 93)
        tuple_var_assignment_93437_93706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'tuple_var_assignment_93437')
        # Assigning a type to the variable 'proj' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'proj', tuple_var_assignment_93437_93706)
        
        # Assigning a BinOp to a Name (line 94):
        
        # Assigning a BinOp to a Name (line 94):
        
        # Call to clock(...): (line 94)
        # Processing the call keyword arguments (line 94)
        kwargs_93709 = {}
        # Getting the type of 'time' (line 94)
        time_93707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'time', False)
        # Obtaining the member 'clock' of a type (line 94)
        clock_93708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), time_93707, 'clock')
        # Calling clock(args, kwargs) (line 94)
        clock_call_result_93710 = invoke(stypy.reporting.localization.Localization(__file__, 94, 12), clock_93708, *[], **kwargs_93709)
        
        # Getting the type of 't0' (line 94)
        t0_93711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 27), 't0')
        # Applying the binary operator '-' (line 94)
        result_sub_93712 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 12), '-', clock_call_result_93710, t0_93711)
        
        # Assigning a type to the variable 't' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 't', result_sub_93712)
        
        # Assigning a Call to a Name (line 95):
        
        # Assigning a Call to a Name (line 95):
        
        # Call to reconstruct_matrix_from_id(...): (line 95)
        # Processing the call arguments (line 95)
        
        # Obtaining the type of the subscript
        slice_93715 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 95, 50), None, None, None)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 95)
        k_93716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 60), 'k', False)
        slice_93717 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 95, 55), None, k_93716, None)
        # Getting the type of 'idx' (line 95)
        idx_93718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 55), 'idx', False)
        # Obtaining the member '__getitem__' of a type (line 95)
        getitem___93719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 55), idx_93718, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 95)
        subscript_call_result_93720 = invoke(stypy.reporting.localization.Localization(__file__, 95, 55), getitem___93719, slice_93717)
        
        # Getting the type of 'A' (line 95)
        A_93721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 50), 'A', False)
        # Obtaining the member '__getitem__' of a type (line 95)
        getitem___93722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 50), A_93721, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 95)
        subscript_call_result_93723 = invoke(stypy.reporting.localization.Localization(__file__, 95, 50), getitem___93722, (slice_93715, subscript_call_result_93720))
        
        # Getting the type of 'idx' (line 95)
        idx_93724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 65), 'idx', False)
        # Getting the type of 'proj' (line 95)
        proj_93725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 70), 'proj', False)
        # Processing the call keyword arguments (line 95)
        kwargs_93726 = {}
        # Getting the type of 'pymatrixid' (line 95)
        pymatrixid_93713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'pymatrixid', False)
        # Obtaining the member 'reconstruct_matrix_from_id' of a type (line 95)
        reconstruct_matrix_from_id_93714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 12), pymatrixid_93713, 'reconstruct_matrix_from_id')
        # Calling reconstruct_matrix_from_id(args, kwargs) (line 95)
        reconstruct_matrix_from_id_call_result_93727 = invoke(stypy.reporting.localization.Localization(__file__, 95, 12), reconstruct_matrix_from_id_93714, *[subscript_call_result_93723, idx_93724, proj_93725], **kwargs_93726)
        
        # Assigning a type to the variable 'B' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'B', reconstruct_matrix_from_id_call_result_93727)
        
        # Call to _debug_print(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'fmt' (line 96)
        fmt_93729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 21), 'fmt', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 96)
        tuple_93730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 96)
        # Adding element type (line 96)
        # Getting the type of 't' (line 96)
        t_93731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 28), 't', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 28), tuple_93730, t_93731)
        # Adding element type (line 96)
        
        # Call to allclose(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'A' (line 96)
        A_93734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 43), 'A', False)
        # Getting the type of 'B' (line 96)
        B_93735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 46), 'B', False)
        # Getting the type of 'eps' (line 96)
        eps_93736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 49), 'eps', False)
        # Processing the call keyword arguments (line 96)
        kwargs_93737 = {}
        # Getting the type of 'np' (line 96)
        np_93732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 31), 'np', False)
        # Obtaining the member 'allclose' of a type (line 96)
        allclose_93733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 31), np_93732, 'allclose')
        # Calling allclose(args, kwargs) (line 96)
        allclose_call_result_93738 = invoke(stypy.reporting.localization.Localization(__file__, 96, 31), allclose_93733, *[A_93734, B_93735, eps_93736], **kwargs_93737)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 28), tuple_93730, allclose_call_result_93738)
        
        # Applying the binary operator '%' (line 96)
        result_mod_93739 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 21), '%', fmt_93729, tuple_93730)
        
        # Processing the call keyword arguments (line 96)
        kwargs_93740 = {}
        # Getting the type of '_debug_print' (line 96)
        _debug_print_93728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 96)
        _debug_print_call_result_93741 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), _debug_print_93728, *[result_mod_93739], **kwargs_93740)
        
        
        # Call to assert_(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Call to allclose(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'A' (line 97)
        A_93745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 28), 'A', False)
        # Getting the type of 'B' (line 97)
        B_93746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 31), 'B', False)
        # Getting the type of 'eps' (line 97)
        eps_93747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 34), 'eps', False)
        # Processing the call keyword arguments (line 97)
        kwargs_93748 = {}
        # Getting the type of 'np' (line 97)
        np_93743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 16), 'np', False)
        # Obtaining the member 'allclose' of a type (line 97)
        allclose_93744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 16), np_93743, 'allclose')
        # Calling allclose(args, kwargs) (line 97)
        allclose_call_result_93749 = invoke(stypy.reporting.localization.Localization(__file__, 97, 16), allclose_93744, *[A_93745, B_93746, eps_93747], **kwargs_93748)
        
        # Processing the call keyword arguments (line 97)
        kwargs_93750 = {}
        # Getting the type of 'assert_' (line 97)
        assert__93742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 97)
        assert__call_result_93751 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), assert__93742, *[allclose_call_result_93749], **kwargs_93750)
        
        
        # Call to _debug_print(...): (line 99)
        # Processing the call arguments (line 99)
        str_93753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 21), 'str', 'Calling iddp_rid / idzp_rid ...')
        # Processing the call keyword arguments (line 99)
        kwargs_93754 = {}
        # Getting the type of '_debug_print' (line 99)
        _debug_print_93752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 99)
        _debug_print_call_result_93755 = invoke(stypy.reporting.localization.Localization(__file__, 99, 8), _debug_print_93752, *[str_93753], **kwargs_93754)
        
        
        # Assigning a Call to a Name (line 100):
        
        # Assigning a Call to a Name (line 100):
        
        # Call to clock(...): (line 100)
        # Processing the call keyword arguments (line 100)
        kwargs_93758 = {}
        # Getting the type of 'time' (line 100)
        time_93756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 13), 'time', False)
        # Obtaining the member 'clock' of a type (line 100)
        clock_93757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 13), time_93756, 'clock')
        # Calling clock(args, kwargs) (line 100)
        clock_call_result_93759 = invoke(stypy.reporting.localization.Localization(__file__, 100, 13), clock_93757, *[], **kwargs_93758)
        
        # Assigning a type to the variable 't0' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 't0', clock_call_result_93759)
        
        # Assigning a Call to a Tuple (line 101):
        
        # Assigning a Subscript to a Name (line 101):
        
        # Obtaining the type of the subscript
        int_93760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 8), 'int')
        
        # Call to interp_decomp(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'L' (line 101)
        L_93763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 48), 'L', False)
        # Getting the type of 'eps' (line 101)
        eps_93764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 51), 'eps', False)
        # Processing the call keyword arguments (line 101)
        kwargs_93765 = {}
        # Getting the type of 'pymatrixid' (line 101)
        pymatrixid_93761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 23), 'pymatrixid', False)
        # Obtaining the member 'interp_decomp' of a type (line 101)
        interp_decomp_93762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 23), pymatrixid_93761, 'interp_decomp')
        # Calling interp_decomp(args, kwargs) (line 101)
        interp_decomp_call_result_93766 = invoke(stypy.reporting.localization.Localization(__file__, 101, 23), interp_decomp_93762, *[L_93763, eps_93764], **kwargs_93765)
        
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___93767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), interp_decomp_call_result_93766, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_93768 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), getitem___93767, int_93760)
        
        # Assigning a type to the variable 'tuple_var_assignment_93438' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_93438', subscript_call_result_93768)
        
        # Assigning a Subscript to a Name (line 101):
        
        # Obtaining the type of the subscript
        int_93769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 8), 'int')
        
        # Call to interp_decomp(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'L' (line 101)
        L_93772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 48), 'L', False)
        # Getting the type of 'eps' (line 101)
        eps_93773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 51), 'eps', False)
        # Processing the call keyword arguments (line 101)
        kwargs_93774 = {}
        # Getting the type of 'pymatrixid' (line 101)
        pymatrixid_93770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 23), 'pymatrixid', False)
        # Obtaining the member 'interp_decomp' of a type (line 101)
        interp_decomp_93771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 23), pymatrixid_93770, 'interp_decomp')
        # Calling interp_decomp(args, kwargs) (line 101)
        interp_decomp_call_result_93775 = invoke(stypy.reporting.localization.Localization(__file__, 101, 23), interp_decomp_93771, *[L_93772, eps_93773], **kwargs_93774)
        
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___93776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), interp_decomp_call_result_93775, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_93777 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), getitem___93776, int_93769)
        
        # Assigning a type to the variable 'tuple_var_assignment_93439' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_93439', subscript_call_result_93777)
        
        # Assigning a Subscript to a Name (line 101):
        
        # Obtaining the type of the subscript
        int_93778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 8), 'int')
        
        # Call to interp_decomp(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'L' (line 101)
        L_93781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 48), 'L', False)
        # Getting the type of 'eps' (line 101)
        eps_93782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 51), 'eps', False)
        # Processing the call keyword arguments (line 101)
        kwargs_93783 = {}
        # Getting the type of 'pymatrixid' (line 101)
        pymatrixid_93779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 23), 'pymatrixid', False)
        # Obtaining the member 'interp_decomp' of a type (line 101)
        interp_decomp_93780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 23), pymatrixid_93779, 'interp_decomp')
        # Calling interp_decomp(args, kwargs) (line 101)
        interp_decomp_call_result_93784 = invoke(stypy.reporting.localization.Localization(__file__, 101, 23), interp_decomp_93780, *[L_93781, eps_93782], **kwargs_93783)
        
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___93785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), interp_decomp_call_result_93784, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_93786 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), getitem___93785, int_93778)
        
        # Assigning a type to the variable 'tuple_var_assignment_93440' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_93440', subscript_call_result_93786)
        
        # Assigning a Name to a Name (line 101):
        # Getting the type of 'tuple_var_assignment_93438' (line 101)
        tuple_var_assignment_93438_93787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_93438')
        # Assigning a type to the variable 'k' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'k', tuple_var_assignment_93438_93787)
        
        # Assigning a Name to a Name (line 101):
        # Getting the type of 'tuple_var_assignment_93439' (line 101)
        tuple_var_assignment_93439_93788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_93439')
        # Assigning a type to the variable 'idx' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'idx', tuple_var_assignment_93439_93788)
        
        # Assigning a Name to a Name (line 101):
        # Getting the type of 'tuple_var_assignment_93440' (line 101)
        tuple_var_assignment_93440_93789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_93440')
        # Assigning a type to the variable 'proj' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'proj', tuple_var_assignment_93440_93789)
        
        # Assigning a BinOp to a Name (line 102):
        
        # Assigning a BinOp to a Name (line 102):
        
        # Call to clock(...): (line 102)
        # Processing the call keyword arguments (line 102)
        kwargs_93792 = {}
        # Getting the type of 'time' (line 102)
        time_93790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'time', False)
        # Obtaining the member 'clock' of a type (line 102)
        clock_93791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), time_93790, 'clock')
        # Calling clock(args, kwargs) (line 102)
        clock_call_result_93793 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), clock_93791, *[], **kwargs_93792)
        
        # Getting the type of 't0' (line 102)
        t0_93794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 27), 't0')
        # Applying the binary operator '-' (line 102)
        result_sub_93795 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 12), '-', clock_call_result_93793, t0_93794)
        
        # Assigning a type to the variable 't' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 't', result_sub_93795)
        
        # Assigning a Call to a Name (line 103):
        
        # Assigning a Call to a Name (line 103):
        
        # Call to reconstruct_matrix_from_id(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Obtaining the type of the subscript
        slice_93798 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 103, 50), None, None, None)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 103)
        k_93799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 60), 'k', False)
        slice_93800 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 103, 55), None, k_93799, None)
        # Getting the type of 'idx' (line 103)
        idx_93801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 55), 'idx', False)
        # Obtaining the member '__getitem__' of a type (line 103)
        getitem___93802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 55), idx_93801, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 103)
        subscript_call_result_93803 = invoke(stypy.reporting.localization.Localization(__file__, 103, 55), getitem___93802, slice_93800)
        
        # Getting the type of 'A' (line 103)
        A_93804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 50), 'A', False)
        # Obtaining the member '__getitem__' of a type (line 103)
        getitem___93805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 50), A_93804, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 103)
        subscript_call_result_93806 = invoke(stypy.reporting.localization.Localization(__file__, 103, 50), getitem___93805, (slice_93798, subscript_call_result_93803))
        
        # Getting the type of 'idx' (line 103)
        idx_93807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 65), 'idx', False)
        # Getting the type of 'proj' (line 103)
        proj_93808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 70), 'proj', False)
        # Processing the call keyword arguments (line 103)
        kwargs_93809 = {}
        # Getting the type of 'pymatrixid' (line 103)
        pymatrixid_93796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'pymatrixid', False)
        # Obtaining the member 'reconstruct_matrix_from_id' of a type (line 103)
        reconstruct_matrix_from_id_93797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), pymatrixid_93796, 'reconstruct_matrix_from_id')
        # Calling reconstruct_matrix_from_id(args, kwargs) (line 103)
        reconstruct_matrix_from_id_call_result_93810 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), reconstruct_matrix_from_id_93797, *[subscript_call_result_93806, idx_93807, proj_93808], **kwargs_93809)
        
        # Assigning a type to the variable 'B' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'B', reconstruct_matrix_from_id_call_result_93810)
        
        # Call to _debug_print(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'fmt' (line 104)
        fmt_93812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 21), 'fmt', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 104)
        tuple_93813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 104)
        # Adding element type (line 104)
        # Getting the type of 't' (line 104)
        t_93814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 28), 't', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 28), tuple_93813, t_93814)
        # Adding element type (line 104)
        
        # Call to allclose(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'A' (line 104)
        A_93817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 43), 'A', False)
        # Getting the type of 'B' (line 104)
        B_93818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 46), 'B', False)
        # Getting the type of 'eps' (line 104)
        eps_93819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 49), 'eps', False)
        # Processing the call keyword arguments (line 104)
        kwargs_93820 = {}
        # Getting the type of 'np' (line 104)
        np_93815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 31), 'np', False)
        # Obtaining the member 'allclose' of a type (line 104)
        allclose_93816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 31), np_93815, 'allclose')
        # Calling allclose(args, kwargs) (line 104)
        allclose_call_result_93821 = invoke(stypy.reporting.localization.Localization(__file__, 104, 31), allclose_93816, *[A_93817, B_93818, eps_93819], **kwargs_93820)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 28), tuple_93813, allclose_call_result_93821)
        
        # Applying the binary operator '%' (line 104)
        result_mod_93822 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 21), '%', fmt_93812, tuple_93813)
        
        # Processing the call keyword arguments (line 104)
        kwargs_93823 = {}
        # Getting the type of '_debug_print' (line 104)
        _debug_print_93811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 104)
        _debug_print_call_result_93824 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), _debug_print_93811, *[result_mod_93822], **kwargs_93823)
        
        
        # Call to assert_(...): (line 105)
        # Processing the call arguments (line 105)
        
        # Call to allclose(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'A' (line 105)
        A_93828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 28), 'A', False)
        # Getting the type of 'B' (line 105)
        B_93829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 31), 'B', False)
        # Getting the type of 'eps' (line 105)
        eps_93830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 34), 'eps', False)
        # Processing the call keyword arguments (line 105)
        kwargs_93831 = {}
        # Getting the type of 'np' (line 105)
        np_93826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'np', False)
        # Obtaining the member 'allclose' of a type (line 105)
        allclose_93827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 16), np_93826, 'allclose')
        # Calling allclose(args, kwargs) (line 105)
        allclose_call_result_93832 = invoke(stypy.reporting.localization.Localization(__file__, 105, 16), allclose_93827, *[A_93828, B_93829, eps_93830], **kwargs_93831)
        
        # Processing the call keyword arguments (line 105)
        kwargs_93833 = {}
        # Getting the type of 'assert_' (line 105)
        assert__93825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 105)
        assert__call_result_93834 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), assert__93825, *[allclose_call_result_93832], **kwargs_93833)
        
        
        # Assigning a Name to a Name (line 108):
        
        # Assigning a Name to a Name (line 108):
        # Getting the type of 'rank' (line 108)
        rank_93835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'rank')
        # Assigning a type to the variable 'k' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'k', rank_93835)
        
        # Call to _debug_print(...): (line 110)
        # Processing the call arguments (line 110)
        str_93837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 21), 'str', 'Calling iddr_id / idzr_id  ...')
        # Processing the call keyword arguments (line 110)
        kwargs_93838 = {}
        # Getting the type of '_debug_print' (line 110)
        _debug_print_93836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 110)
        _debug_print_call_result_93839 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), _debug_print_93836, *[str_93837], **kwargs_93838)
        
        
        # Assigning a Call to a Name (line 111):
        
        # Assigning a Call to a Name (line 111):
        
        # Call to clock(...): (line 111)
        # Processing the call keyword arguments (line 111)
        kwargs_93842 = {}
        # Getting the type of 'time' (line 111)
        time_93840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 13), 'time', False)
        # Obtaining the member 'clock' of a type (line 111)
        clock_93841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 13), time_93840, 'clock')
        # Calling clock(args, kwargs) (line 111)
        clock_call_result_93843 = invoke(stypy.reporting.localization.Localization(__file__, 111, 13), clock_93841, *[], **kwargs_93842)
        
        # Assigning a type to the variable 't0' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 't0', clock_call_result_93843)
        
        # Assigning a Call to a Tuple (line 112):
        
        # Assigning a Subscript to a Name (line 112):
        
        # Obtaining the type of the subscript
        int_93844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 8), 'int')
        
        # Call to interp_decomp(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'A' (line 112)
        A_93847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 45), 'A', False)
        # Getting the type of 'k' (line 112)
        k_93848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 48), 'k', False)
        # Processing the call keyword arguments (line 112)
        # Getting the type of 'False' (line 112)
        False_93849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 56), 'False', False)
        keyword_93850 = False_93849
        kwargs_93851 = {'rand': keyword_93850}
        # Getting the type of 'pymatrixid' (line 112)
        pymatrixid_93845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'pymatrixid', False)
        # Obtaining the member 'interp_decomp' of a type (line 112)
        interp_decomp_93846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 20), pymatrixid_93845, 'interp_decomp')
        # Calling interp_decomp(args, kwargs) (line 112)
        interp_decomp_call_result_93852 = invoke(stypy.reporting.localization.Localization(__file__, 112, 20), interp_decomp_93846, *[A_93847, k_93848], **kwargs_93851)
        
        # Obtaining the member '__getitem__' of a type (line 112)
        getitem___93853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), interp_decomp_call_result_93852, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 112)
        subscript_call_result_93854 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), getitem___93853, int_93844)
        
        # Assigning a type to the variable 'tuple_var_assignment_93441' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'tuple_var_assignment_93441', subscript_call_result_93854)
        
        # Assigning a Subscript to a Name (line 112):
        
        # Obtaining the type of the subscript
        int_93855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 8), 'int')
        
        # Call to interp_decomp(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'A' (line 112)
        A_93858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 45), 'A', False)
        # Getting the type of 'k' (line 112)
        k_93859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 48), 'k', False)
        # Processing the call keyword arguments (line 112)
        # Getting the type of 'False' (line 112)
        False_93860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 56), 'False', False)
        keyword_93861 = False_93860
        kwargs_93862 = {'rand': keyword_93861}
        # Getting the type of 'pymatrixid' (line 112)
        pymatrixid_93856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'pymatrixid', False)
        # Obtaining the member 'interp_decomp' of a type (line 112)
        interp_decomp_93857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 20), pymatrixid_93856, 'interp_decomp')
        # Calling interp_decomp(args, kwargs) (line 112)
        interp_decomp_call_result_93863 = invoke(stypy.reporting.localization.Localization(__file__, 112, 20), interp_decomp_93857, *[A_93858, k_93859], **kwargs_93862)
        
        # Obtaining the member '__getitem__' of a type (line 112)
        getitem___93864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), interp_decomp_call_result_93863, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 112)
        subscript_call_result_93865 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), getitem___93864, int_93855)
        
        # Assigning a type to the variable 'tuple_var_assignment_93442' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'tuple_var_assignment_93442', subscript_call_result_93865)
        
        # Assigning a Name to a Name (line 112):
        # Getting the type of 'tuple_var_assignment_93441' (line 112)
        tuple_var_assignment_93441_93866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'tuple_var_assignment_93441')
        # Assigning a type to the variable 'idx' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'idx', tuple_var_assignment_93441_93866)
        
        # Assigning a Name to a Name (line 112):
        # Getting the type of 'tuple_var_assignment_93442' (line 112)
        tuple_var_assignment_93442_93867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'tuple_var_assignment_93442')
        # Assigning a type to the variable 'proj' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 13), 'proj', tuple_var_assignment_93442_93867)
        
        # Assigning a BinOp to a Name (line 113):
        
        # Assigning a BinOp to a Name (line 113):
        
        # Call to clock(...): (line 113)
        # Processing the call keyword arguments (line 113)
        kwargs_93870 = {}
        # Getting the type of 'time' (line 113)
        time_93868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'time', False)
        # Obtaining the member 'clock' of a type (line 113)
        clock_93869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 12), time_93868, 'clock')
        # Calling clock(args, kwargs) (line 113)
        clock_call_result_93871 = invoke(stypy.reporting.localization.Localization(__file__, 113, 12), clock_93869, *[], **kwargs_93870)
        
        # Getting the type of 't0' (line 113)
        t0_93872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 27), 't0')
        # Applying the binary operator '-' (line 113)
        result_sub_93873 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 12), '-', clock_call_result_93871, t0_93872)
        
        # Assigning a type to the variable 't' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 't', result_sub_93873)
        
        # Assigning a Call to a Name (line 114):
        
        # Assigning a Call to a Name (line 114):
        
        # Call to reconstruct_matrix_from_id(...): (line 114)
        # Processing the call arguments (line 114)
        
        # Obtaining the type of the subscript
        slice_93876 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 114, 50), None, None, None)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 114)
        k_93877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 60), 'k', False)
        slice_93878 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 114, 55), None, k_93877, None)
        # Getting the type of 'idx' (line 114)
        idx_93879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 55), 'idx', False)
        # Obtaining the member '__getitem__' of a type (line 114)
        getitem___93880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 55), idx_93879, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 114)
        subscript_call_result_93881 = invoke(stypy.reporting.localization.Localization(__file__, 114, 55), getitem___93880, slice_93878)
        
        # Getting the type of 'A' (line 114)
        A_93882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 50), 'A', False)
        # Obtaining the member '__getitem__' of a type (line 114)
        getitem___93883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 50), A_93882, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 114)
        subscript_call_result_93884 = invoke(stypy.reporting.localization.Localization(__file__, 114, 50), getitem___93883, (slice_93876, subscript_call_result_93881))
        
        # Getting the type of 'idx' (line 114)
        idx_93885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 65), 'idx', False)
        # Getting the type of 'proj' (line 114)
        proj_93886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 70), 'proj', False)
        # Processing the call keyword arguments (line 114)
        kwargs_93887 = {}
        # Getting the type of 'pymatrixid' (line 114)
        pymatrixid_93874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'pymatrixid', False)
        # Obtaining the member 'reconstruct_matrix_from_id' of a type (line 114)
        reconstruct_matrix_from_id_93875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), pymatrixid_93874, 'reconstruct_matrix_from_id')
        # Calling reconstruct_matrix_from_id(args, kwargs) (line 114)
        reconstruct_matrix_from_id_call_result_93888 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), reconstruct_matrix_from_id_93875, *[subscript_call_result_93884, idx_93885, proj_93886], **kwargs_93887)
        
        # Assigning a type to the variable 'B' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'B', reconstruct_matrix_from_id_call_result_93888)
        
        # Call to _debug_print(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'fmt' (line 115)
        fmt_93890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 21), 'fmt', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 115)
        tuple_93891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 115)
        # Adding element type (line 115)
        # Getting the type of 't' (line 115)
        t_93892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 28), 't', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 28), tuple_93891, t_93892)
        # Adding element type (line 115)
        
        # Call to allclose(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'A' (line 115)
        A_93895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 43), 'A', False)
        # Getting the type of 'B' (line 115)
        B_93896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 46), 'B', False)
        # Getting the type of 'eps' (line 115)
        eps_93897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 49), 'eps', False)
        # Processing the call keyword arguments (line 115)
        kwargs_93898 = {}
        # Getting the type of 'np' (line 115)
        np_93893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 31), 'np', False)
        # Obtaining the member 'allclose' of a type (line 115)
        allclose_93894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 31), np_93893, 'allclose')
        # Calling allclose(args, kwargs) (line 115)
        allclose_call_result_93899 = invoke(stypy.reporting.localization.Localization(__file__, 115, 31), allclose_93894, *[A_93895, B_93896, eps_93897], **kwargs_93898)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 28), tuple_93891, allclose_call_result_93899)
        
        # Applying the binary operator '%' (line 115)
        result_mod_93900 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 21), '%', fmt_93890, tuple_93891)
        
        # Processing the call keyword arguments (line 115)
        kwargs_93901 = {}
        # Getting the type of '_debug_print' (line 115)
        _debug_print_93889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 115)
        _debug_print_call_result_93902 = invoke(stypy.reporting.localization.Localization(__file__, 115, 8), _debug_print_93889, *[result_mod_93900], **kwargs_93901)
        
        
        # Call to assert_(...): (line 116)
        # Processing the call arguments (line 116)
        
        # Call to allclose(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'A' (line 116)
        A_93906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 28), 'A', False)
        # Getting the type of 'B' (line 116)
        B_93907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 31), 'B', False)
        # Getting the type of 'eps' (line 116)
        eps_93908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 34), 'eps', False)
        # Processing the call keyword arguments (line 116)
        kwargs_93909 = {}
        # Getting the type of 'np' (line 116)
        np_93904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'np', False)
        # Obtaining the member 'allclose' of a type (line 116)
        allclose_93905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 16), np_93904, 'allclose')
        # Calling allclose(args, kwargs) (line 116)
        allclose_call_result_93910 = invoke(stypy.reporting.localization.Localization(__file__, 116, 16), allclose_93905, *[A_93906, B_93907, eps_93908], **kwargs_93909)
        
        # Processing the call keyword arguments (line 116)
        kwargs_93911 = {}
        # Getting the type of 'assert_' (line 116)
        assert__93903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 116)
        assert__call_result_93912 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), assert__93903, *[allclose_call_result_93910], **kwargs_93911)
        
        
        # Call to _debug_print(...): (line 118)
        # Processing the call arguments (line 118)
        str_93914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 21), 'str', 'Calling iddr_aid / idzr_aid ...')
        # Processing the call keyword arguments (line 118)
        kwargs_93915 = {}
        # Getting the type of '_debug_print' (line 118)
        _debug_print_93913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 118)
        _debug_print_call_result_93916 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), _debug_print_93913, *[str_93914], **kwargs_93915)
        
        
        # Assigning a Call to a Name (line 119):
        
        # Assigning a Call to a Name (line 119):
        
        # Call to clock(...): (line 119)
        # Processing the call keyword arguments (line 119)
        kwargs_93919 = {}
        # Getting the type of 'time' (line 119)
        time_93917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 13), 'time', False)
        # Obtaining the member 'clock' of a type (line 119)
        clock_93918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 13), time_93917, 'clock')
        # Calling clock(args, kwargs) (line 119)
        clock_call_result_93920 = invoke(stypy.reporting.localization.Localization(__file__, 119, 13), clock_93918, *[], **kwargs_93919)
        
        # Assigning a type to the variable 't0' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 't0', clock_call_result_93920)
        
        # Assigning a Call to a Tuple (line 120):
        
        # Assigning a Subscript to a Name (line 120):
        
        # Obtaining the type of the subscript
        int_93921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 8), 'int')
        
        # Call to interp_decomp(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'A' (line 120)
        A_93924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 45), 'A', False)
        # Getting the type of 'k' (line 120)
        k_93925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 48), 'k', False)
        # Processing the call keyword arguments (line 120)
        kwargs_93926 = {}
        # Getting the type of 'pymatrixid' (line 120)
        pymatrixid_93922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 20), 'pymatrixid', False)
        # Obtaining the member 'interp_decomp' of a type (line 120)
        interp_decomp_93923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 20), pymatrixid_93922, 'interp_decomp')
        # Calling interp_decomp(args, kwargs) (line 120)
        interp_decomp_call_result_93927 = invoke(stypy.reporting.localization.Localization(__file__, 120, 20), interp_decomp_93923, *[A_93924, k_93925], **kwargs_93926)
        
        # Obtaining the member '__getitem__' of a type (line 120)
        getitem___93928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), interp_decomp_call_result_93927, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 120)
        subscript_call_result_93929 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), getitem___93928, int_93921)
        
        # Assigning a type to the variable 'tuple_var_assignment_93443' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'tuple_var_assignment_93443', subscript_call_result_93929)
        
        # Assigning a Subscript to a Name (line 120):
        
        # Obtaining the type of the subscript
        int_93930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 8), 'int')
        
        # Call to interp_decomp(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'A' (line 120)
        A_93933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 45), 'A', False)
        # Getting the type of 'k' (line 120)
        k_93934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 48), 'k', False)
        # Processing the call keyword arguments (line 120)
        kwargs_93935 = {}
        # Getting the type of 'pymatrixid' (line 120)
        pymatrixid_93931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 20), 'pymatrixid', False)
        # Obtaining the member 'interp_decomp' of a type (line 120)
        interp_decomp_93932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 20), pymatrixid_93931, 'interp_decomp')
        # Calling interp_decomp(args, kwargs) (line 120)
        interp_decomp_call_result_93936 = invoke(stypy.reporting.localization.Localization(__file__, 120, 20), interp_decomp_93932, *[A_93933, k_93934], **kwargs_93935)
        
        # Obtaining the member '__getitem__' of a type (line 120)
        getitem___93937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), interp_decomp_call_result_93936, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 120)
        subscript_call_result_93938 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), getitem___93937, int_93930)
        
        # Assigning a type to the variable 'tuple_var_assignment_93444' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'tuple_var_assignment_93444', subscript_call_result_93938)
        
        # Assigning a Name to a Name (line 120):
        # Getting the type of 'tuple_var_assignment_93443' (line 120)
        tuple_var_assignment_93443_93939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'tuple_var_assignment_93443')
        # Assigning a type to the variable 'idx' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'idx', tuple_var_assignment_93443_93939)
        
        # Assigning a Name to a Name (line 120):
        # Getting the type of 'tuple_var_assignment_93444' (line 120)
        tuple_var_assignment_93444_93940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'tuple_var_assignment_93444')
        # Assigning a type to the variable 'proj' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 13), 'proj', tuple_var_assignment_93444_93940)
        
        # Assigning a BinOp to a Name (line 121):
        
        # Assigning a BinOp to a Name (line 121):
        
        # Call to clock(...): (line 121)
        # Processing the call keyword arguments (line 121)
        kwargs_93943 = {}
        # Getting the type of 'time' (line 121)
        time_93941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'time', False)
        # Obtaining the member 'clock' of a type (line 121)
        clock_93942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), time_93941, 'clock')
        # Calling clock(args, kwargs) (line 121)
        clock_call_result_93944 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), clock_93942, *[], **kwargs_93943)
        
        # Getting the type of 't0' (line 121)
        t0_93945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 27), 't0')
        # Applying the binary operator '-' (line 121)
        result_sub_93946 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 12), '-', clock_call_result_93944, t0_93945)
        
        # Assigning a type to the variable 't' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 't', result_sub_93946)
        
        # Assigning a Call to a Name (line 122):
        
        # Assigning a Call to a Name (line 122):
        
        # Call to reconstruct_matrix_from_id(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Obtaining the type of the subscript
        slice_93949 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 122, 50), None, None, None)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 122)
        k_93950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 60), 'k', False)
        slice_93951 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 122, 55), None, k_93950, None)
        # Getting the type of 'idx' (line 122)
        idx_93952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 55), 'idx', False)
        # Obtaining the member '__getitem__' of a type (line 122)
        getitem___93953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 55), idx_93952, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 122)
        subscript_call_result_93954 = invoke(stypy.reporting.localization.Localization(__file__, 122, 55), getitem___93953, slice_93951)
        
        # Getting the type of 'A' (line 122)
        A_93955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 50), 'A', False)
        # Obtaining the member '__getitem__' of a type (line 122)
        getitem___93956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 50), A_93955, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 122)
        subscript_call_result_93957 = invoke(stypy.reporting.localization.Localization(__file__, 122, 50), getitem___93956, (slice_93949, subscript_call_result_93954))
        
        # Getting the type of 'idx' (line 122)
        idx_93958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 65), 'idx', False)
        # Getting the type of 'proj' (line 122)
        proj_93959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 70), 'proj', False)
        # Processing the call keyword arguments (line 122)
        kwargs_93960 = {}
        # Getting the type of 'pymatrixid' (line 122)
        pymatrixid_93947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'pymatrixid', False)
        # Obtaining the member 'reconstruct_matrix_from_id' of a type (line 122)
        reconstruct_matrix_from_id_93948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 12), pymatrixid_93947, 'reconstruct_matrix_from_id')
        # Calling reconstruct_matrix_from_id(args, kwargs) (line 122)
        reconstruct_matrix_from_id_call_result_93961 = invoke(stypy.reporting.localization.Localization(__file__, 122, 12), reconstruct_matrix_from_id_93948, *[subscript_call_result_93957, idx_93958, proj_93959], **kwargs_93960)
        
        # Assigning a type to the variable 'B' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'B', reconstruct_matrix_from_id_call_result_93961)
        
        # Call to _debug_print(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'fmt' (line 123)
        fmt_93963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 21), 'fmt', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 123)
        tuple_93964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 123)
        # Adding element type (line 123)
        # Getting the type of 't' (line 123)
        t_93965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 28), 't', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 28), tuple_93964, t_93965)
        # Adding element type (line 123)
        
        # Call to allclose(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'A' (line 123)
        A_93968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 43), 'A', False)
        # Getting the type of 'B' (line 123)
        B_93969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 46), 'B', False)
        # Getting the type of 'eps' (line 123)
        eps_93970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 49), 'eps', False)
        # Processing the call keyword arguments (line 123)
        kwargs_93971 = {}
        # Getting the type of 'np' (line 123)
        np_93966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 31), 'np', False)
        # Obtaining the member 'allclose' of a type (line 123)
        allclose_93967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 31), np_93966, 'allclose')
        # Calling allclose(args, kwargs) (line 123)
        allclose_call_result_93972 = invoke(stypy.reporting.localization.Localization(__file__, 123, 31), allclose_93967, *[A_93968, B_93969, eps_93970], **kwargs_93971)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 28), tuple_93964, allclose_call_result_93972)
        
        # Applying the binary operator '%' (line 123)
        result_mod_93973 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 21), '%', fmt_93963, tuple_93964)
        
        # Processing the call keyword arguments (line 123)
        kwargs_93974 = {}
        # Getting the type of '_debug_print' (line 123)
        _debug_print_93962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 123)
        _debug_print_call_result_93975 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), _debug_print_93962, *[result_mod_93973], **kwargs_93974)
        
        
        # Call to assert_(...): (line 124)
        # Processing the call arguments (line 124)
        
        # Call to allclose(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'A' (line 124)
        A_93979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 28), 'A', False)
        # Getting the type of 'B' (line 124)
        B_93980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 31), 'B', False)
        # Getting the type of 'eps' (line 124)
        eps_93981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 34), 'eps', False)
        # Processing the call keyword arguments (line 124)
        kwargs_93982 = {}
        # Getting the type of 'np' (line 124)
        np_93977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'np', False)
        # Obtaining the member 'allclose' of a type (line 124)
        allclose_93978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 16), np_93977, 'allclose')
        # Calling allclose(args, kwargs) (line 124)
        allclose_call_result_93983 = invoke(stypy.reporting.localization.Localization(__file__, 124, 16), allclose_93978, *[A_93979, B_93980, eps_93981], **kwargs_93982)
        
        # Processing the call keyword arguments (line 124)
        kwargs_93984 = {}
        # Getting the type of 'assert_' (line 124)
        assert__93976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 124)
        assert__call_result_93985 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), assert__93976, *[allclose_call_result_93983], **kwargs_93984)
        
        
        # Call to _debug_print(...): (line 126)
        # Processing the call arguments (line 126)
        str_93987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 21), 'str', 'Calling iddr_rid / idzr_rid ...')
        # Processing the call keyword arguments (line 126)
        kwargs_93988 = {}
        # Getting the type of '_debug_print' (line 126)
        _debug_print_93986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 126)
        _debug_print_call_result_93989 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), _debug_print_93986, *[str_93987], **kwargs_93988)
        
        
        # Assigning a Call to a Name (line 127):
        
        # Assigning a Call to a Name (line 127):
        
        # Call to clock(...): (line 127)
        # Processing the call keyword arguments (line 127)
        kwargs_93992 = {}
        # Getting the type of 'time' (line 127)
        time_93990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 13), 'time', False)
        # Obtaining the member 'clock' of a type (line 127)
        clock_93991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 13), time_93990, 'clock')
        # Calling clock(args, kwargs) (line 127)
        clock_call_result_93993 = invoke(stypy.reporting.localization.Localization(__file__, 127, 13), clock_93991, *[], **kwargs_93992)
        
        # Assigning a type to the variable 't0' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 't0', clock_call_result_93993)
        
        # Assigning a Call to a Tuple (line 128):
        
        # Assigning a Subscript to a Name (line 128):
        
        # Obtaining the type of the subscript
        int_93994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 8), 'int')
        
        # Call to interp_decomp(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'L' (line 128)
        L_93997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 45), 'L', False)
        # Getting the type of 'k' (line 128)
        k_93998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 48), 'k', False)
        # Processing the call keyword arguments (line 128)
        kwargs_93999 = {}
        # Getting the type of 'pymatrixid' (line 128)
        pymatrixid_93995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 20), 'pymatrixid', False)
        # Obtaining the member 'interp_decomp' of a type (line 128)
        interp_decomp_93996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 20), pymatrixid_93995, 'interp_decomp')
        # Calling interp_decomp(args, kwargs) (line 128)
        interp_decomp_call_result_94000 = invoke(stypy.reporting.localization.Localization(__file__, 128, 20), interp_decomp_93996, *[L_93997, k_93998], **kwargs_93999)
        
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___94001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), interp_decomp_call_result_94000, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_94002 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), getitem___94001, int_93994)
        
        # Assigning a type to the variable 'tuple_var_assignment_93445' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_93445', subscript_call_result_94002)
        
        # Assigning a Subscript to a Name (line 128):
        
        # Obtaining the type of the subscript
        int_94003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 8), 'int')
        
        # Call to interp_decomp(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'L' (line 128)
        L_94006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 45), 'L', False)
        # Getting the type of 'k' (line 128)
        k_94007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 48), 'k', False)
        # Processing the call keyword arguments (line 128)
        kwargs_94008 = {}
        # Getting the type of 'pymatrixid' (line 128)
        pymatrixid_94004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 20), 'pymatrixid', False)
        # Obtaining the member 'interp_decomp' of a type (line 128)
        interp_decomp_94005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 20), pymatrixid_94004, 'interp_decomp')
        # Calling interp_decomp(args, kwargs) (line 128)
        interp_decomp_call_result_94009 = invoke(stypy.reporting.localization.Localization(__file__, 128, 20), interp_decomp_94005, *[L_94006, k_94007], **kwargs_94008)
        
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___94010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), interp_decomp_call_result_94009, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_94011 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), getitem___94010, int_94003)
        
        # Assigning a type to the variable 'tuple_var_assignment_93446' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_93446', subscript_call_result_94011)
        
        # Assigning a Name to a Name (line 128):
        # Getting the type of 'tuple_var_assignment_93445' (line 128)
        tuple_var_assignment_93445_94012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_93445')
        # Assigning a type to the variable 'idx' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'idx', tuple_var_assignment_93445_94012)
        
        # Assigning a Name to a Name (line 128):
        # Getting the type of 'tuple_var_assignment_93446' (line 128)
        tuple_var_assignment_93446_94013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_93446')
        # Assigning a type to the variable 'proj' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 13), 'proj', tuple_var_assignment_93446_94013)
        
        # Assigning a BinOp to a Name (line 129):
        
        # Assigning a BinOp to a Name (line 129):
        
        # Call to clock(...): (line 129)
        # Processing the call keyword arguments (line 129)
        kwargs_94016 = {}
        # Getting the type of 'time' (line 129)
        time_94014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'time', False)
        # Obtaining the member 'clock' of a type (line 129)
        clock_94015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 12), time_94014, 'clock')
        # Calling clock(args, kwargs) (line 129)
        clock_call_result_94017 = invoke(stypy.reporting.localization.Localization(__file__, 129, 12), clock_94015, *[], **kwargs_94016)
        
        # Getting the type of 't0' (line 129)
        t0_94018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 27), 't0')
        # Applying the binary operator '-' (line 129)
        result_sub_94019 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 12), '-', clock_call_result_94017, t0_94018)
        
        # Assigning a type to the variable 't' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 't', result_sub_94019)
        
        # Assigning a Call to a Name (line 130):
        
        # Assigning a Call to a Name (line 130):
        
        # Call to reconstruct_matrix_from_id(...): (line 130)
        # Processing the call arguments (line 130)
        
        # Obtaining the type of the subscript
        slice_94022 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 130, 50), None, None, None)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 130)
        k_94023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 60), 'k', False)
        slice_94024 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 130, 55), None, k_94023, None)
        # Getting the type of 'idx' (line 130)
        idx_94025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 55), 'idx', False)
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___94026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 55), idx_94025, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
        subscript_call_result_94027 = invoke(stypy.reporting.localization.Localization(__file__, 130, 55), getitem___94026, slice_94024)
        
        # Getting the type of 'A' (line 130)
        A_94028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 50), 'A', False)
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___94029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 50), A_94028, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
        subscript_call_result_94030 = invoke(stypy.reporting.localization.Localization(__file__, 130, 50), getitem___94029, (slice_94022, subscript_call_result_94027))
        
        # Getting the type of 'idx' (line 130)
        idx_94031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 65), 'idx', False)
        # Getting the type of 'proj' (line 130)
        proj_94032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 70), 'proj', False)
        # Processing the call keyword arguments (line 130)
        kwargs_94033 = {}
        # Getting the type of 'pymatrixid' (line 130)
        pymatrixid_94020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'pymatrixid', False)
        # Obtaining the member 'reconstruct_matrix_from_id' of a type (line 130)
        reconstruct_matrix_from_id_94021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 12), pymatrixid_94020, 'reconstruct_matrix_from_id')
        # Calling reconstruct_matrix_from_id(args, kwargs) (line 130)
        reconstruct_matrix_from_id_call_result_94034 = invoke(stypy.reporting.localization.Localization(__file__, 130, 12), reconstruct_matrix_from_id_94021, *[subscript_call_result_94030, idx_94031, proj_94032], **kwargs_94033)
        
        # Assigning a type to the variable 'B' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'B', reconstruct_matrix_from_id_call_result_94034)
        
        # Call to _debug_print(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'fmt' (line 131)
        fmt_94036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 21), 'fmt', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 131)
        tuple_94037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 131)
        # Adding element type (line 131)
        # Getting the type of 't' (line 131)
        t_94038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 28), 't', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 28), tuple_94037, t_94038)
        # Adding element type (line 131)
        
        # Call to allclose(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'A' (line 131)
        A_94041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 43), 'A', False)
        # Getting the type of 'B' (line 131)
        B_94042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 46), 'B', False)
        # Getting the type of 'eps' (line 131)
        eps_94043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 49), 'eps', False)
        # Processing the call keyword arguments (line 131)
        kwargs_94044 = {}
        # Getting the type of 'np' (line 131)
        np_94039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 31), 'np', False)
        # Obtaining the member 'allclose' of a type (line 131)
        allclose_94040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 31), np_94039, 'allclose')
        # Calling allclose(args, kwargs) (line 131)
        allclose_call_result_94045 = invoke(stypy.reporting.localization.Localization(__file__, 131, 31), allclose_94040, *[A_94041, B_94042, eps_94043], **kwargs_94044)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 28), tuple_94037, allclose_call_result_94045)
        
        # Applying the binary operator '%' (line 131)
        result_mod_94046 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 21), '%', fmt_94036, tuple_94037)
        
        # Processing the call keyword arguments (line 131)
        kwargs_94047 = {}
        # Getting the type of '_debug_print' (line 131)
        _debug_print_94035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 131)
        _debug_print_call_result_94048 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), _debug_print_94035, *[result_mod_94046], **kwargs_94047)
        
        
        # Call to assert_(...): (line 132)
        # Processing the call arguments (line 132)
        
        # Call to allclose(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'A' (line 132)
        A_94052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 28), 'A', False)
        # Getting the type of 'B' (line 132)
        B_94053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 31), 'B', False)
        # Getting the type of 'eps' (line 132)
        eps_94054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 34), 'eps', False)
        # Processing the call keyword arguments (line 132)
        kwargs_94055 = {}
        # Getting the type of 'np' (line 132)
        np_94050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'np', False)
        # Obtaining the member 'allclose' of a type (line 132)
        allclose_94051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 16), np_94050, 'allclose')
        # Calling allclose(args, kwargs) (line 132)
        allclose_call_result_94056 = invoke(stypy.reporting.localization.Localization(__file__, 132, 16), allclose_94051, *[A_94052, B_94053, eps_94054], **kwargs_94055)
        
        # Processing the call keyword arguments (line 132)
        kwargs_94057 = {}
        # Getting the type of 'assert_' (line 132)
        assert__94049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 132)
        assert__call_result_94058 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), assert__94049, *[allclose_call_result_94056], **kwargs_94057)
        
        
        # Assigning a Call to a Tuple (line 135):
        
        # Assigning a Subscript to a Name (line 135):
        
        # Obtaining the type of the subscript
        int_94059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 8), 'int')
        
        # Call to interp_decomp(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'A' (line 135)
        A_94062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 45), 'A', False)
        # Getting the type of 'k' (line 135)
        k_94063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 48), 'k', False)
        # Processing the call keyword arguments (line 135)
        # Getting the type of 'False' (line 135)
        False_94064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 56), 'False', False)
        keyword_94065 = False_94064
        kwargs_94066 = {'rand': keyword_94065}
        # Getting the type of 'pymatrixid' (line 135)
        pymatrixid_94060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 20), 'pymatrixid', False)
        # Obtaining the member 'interp_decomp' of a type (line 135)
        interp_decomp_94061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 20), pymatrixid_94060, 'interp_decomp')
        # Calling interp_decomp(args, kwargs) (line 135)
        interp_decomp_call_result_94067 = invoke(stypy.reporting.localization.Localization(__file__, 135, 20), interp_decomp_94061, *[A_94062, k_94063], **kwargs_94066)
        
        # Obtaining the member '__getitem__' of a type (line 135)
        getitem___94068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), interp_decomp_call_result_94067, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 135)
        subscript_call_result_94069 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), getitem___94068, int_94059)
        
        # Assigning a type to the variable 'tuple_var_assignment_93447' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple_var_assignment_93447', subscript_call_result_94069)
        
        # Assigning a Subscript to a Name (line 135):
        
        # Obtaining the type of the subscript
        int_94070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 8), 'int')
        
        # Call to interp_decomp(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'A' (line 135)
        A_94073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 45), 'A', False)
        # Getting the type of 'k' (line 135)
        k_94074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 48), 'k', False)
        # Processing the call keyword arguments (line 135)
        # Getting the type of 'False' (line 135)
        False_94075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 56), 'False', False)
        keyword_94076 = False_94075
        kwargs_94077 = {'rand': keyword_94076}
        # Getting the type of 'pymatrixid' (line 135)
        pymatrixid_94071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 20), 'pymatrixid', False)
        # Obtaining the member 'interp_decomp' of a type (line 135)
        interp_decomp_94072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 20), pymatrixid_94071, 'interp_decomp')
        # Calling interp_decomp(args, kwargs) (line 135)
        interp_decomp_call_result_94078 = invoke(stypy.reporting.localization.Localization(__file__, 135, 20), interp_decomp_94072, *[A_94073, k_94074], **kwargs_94077)
        
        # Obtaining the member '__getitem__' of a type (line 135)
        getitem___94079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), interp_decomp_call_result_94078, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 135)
        subscript_call_result_94080 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), getitem___94079, int_94070)
        
        # Assigning a type to the variable 'tuple_var_assignment_93448' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple_var_assignment_93448', subscript_call_result_94080)
        
        # Assigning a Name to a Name (line 135):
        # Getting the type of 'tuple_var_assignment_93447' (line 135)
        tuple_var_assignment_93447_94081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple_var_assignment_93447')
        # Assigning a type to the variable 'idx' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'idx', tuple_var_assignment_93447_94081)
        
        # Assigning a Name to a Name (line 135):
        # Getting the type of 'tuple_var_assignment_93448' (line 135)
        tuple_var_assignment_93448_94082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple_var_assignment_93448')
        # Assigning a type to the variable 'proj' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 13), 'proj', tuple_var_assignment_93448_94082)
        
        # Assigning a Call to a Name (line 136):
        
        # Assigning a Call to a Name (line 136):
        
        # Call to reconstruct_interp_matrix(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'idx' (line 136)
        idx_94085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 49), 'idx', False)
        # Getting the type of 'proj' (line 136)
        proj_94086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 54), 'proj', False)
        # Processing the call keyword arguments (line 136)
        kwargs_94087 = {}
        # Getting the type of 'pymatrixid' (line 136)
        pymatrixid_94083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'pymatrixid', False)
        # Obtaining the member 'reconstruct_interp_matrix' of a type (line 136)
        reconstruct_interp_matrix_94084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 12), pymatrixid_94083, 'reconstruct_interp_matrix')
        # Calling reconstruct_interp_matrix(args, kwargs) (line 136)
        reconstruct_interp_matrix_call_result_94088 = invoke(stypy.reporting.localization.Localization(__file__, 136, 12), reconstruct_interp_matrix_94084, *[idx_94085, proj_94086], **kwargs_94087)
        
        # Assigning a type to the variable 'P' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'P', reconstruct_interp_matrix_call_result_94088)
        
        # Assigning a Call to a Name (line 137):
        
        # Assigning a Call to a Name (line 137):
        
        # Call to reconstruct_skel_matrix(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'A' (line 137)
        A_94091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 47), 'A', False)
        # Getting the type of 'k' (line 137)
        k_94092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 50), 'k', False)
        # Getting the type of 'idx' (line 137)
        idx_94093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 53), 'idx', False)
        # Processing the call keyword arguments (line 137)
        kwargs_94094 = {}
        # Getting the type of 'pymatrixid' (line 137)
        pymatrixid_94089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'pymatrixid', False)
        # Obtaining the member 'reconstruct_skel_matrix' of a type (line 137)
        reconstruct_skel_matrix_94090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 12), pymatrixid_94089, 'reconstruct_skel_matrix')
        # Calling reconstruct_skel_matrix(args, kwargs) (line 137)
        reconstruct_skel_matrix_call_result_94095 = invoke(stypy.reporting.localization.Localization(__file__, 137, 12), reconstruct_skel_matrix_94090, *[A_94091, k_94092, idx_94093], **kwargs_94094)
        
        # Assigning a type to the variable 'B' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'B', reconstruct_skel_matrix_call_result_94095)
        
        # Call to assert_(...): (line 138)
        # Processing the call arguments (line 138)
        
        # Call to allclose(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'B' (line 138)
        B_94099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 28), 'B', False)
        
        # Obtaining the type of the subscript
        slice_94100 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 138, 31), None, None, None)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 138)
        k_94101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 40), 'k', False)
        slice_94102 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 138, 35), None, k_94101, None)
        # Getting the type of 'idx' (line 138)
        idx_94103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 35), 'idx', False)
        # Obtaining the member '__getitem__' of a type (line 138)
        getitem___94104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 35), idx_94103, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 138)
        subscript_call_result_94105 = invoke(stypy.reporting.localization.Localization(__file__, 138, 35), getitem___94104, slice_94102)
        
        # Getting the type of 'A' (line 138)
        A_94106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 31), 'A', False)
        # Obtaining the member '__getitem__' of a type (line 138)
        getitem___94107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 31), A_94106, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 138)
        subscript_call_result_94108 = invoke(stypy.reporting.localization.Localization(__file__, 138, 31), getitem___94107, (slice_94100, subscript_call_result_94105))
        
        # Getting the type of 'eps' (line 138)
        eps_94109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 45), 'eps', False)
        # Processing the call keyword arguments (line 138)
        kwargs_94110 = {}
        # Getting the type of 'np' (line 138)
        np_94097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'np', False)
        # Obtaining the member 'allclose' of a type (line 138)
        allclose_94098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 16), np_94097, 'allclose')
        # Calling allclose(args, kwargs) (line 138)
        allclose_call_result_94111 = invoke(stypy.reporting.localization.Localization(__file__, 138, 16), allclose_94098, *[B_94099, subscript_call_result_94108, eps_94109], **kwargs_94110)
        
        # Processing the call keyword arguments (line 138)
        kwargs_94112 = {}
        # Getting the type of 'assert_' (line 138)
        assert__94096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 138)
        assert__call_result_94113 = invoke(stypy.reporting.localization.Localization(__file__, 138, 8), assert__94096, *[allclose_call_result_94111], **kwargs_94112)
        
        
        # Call to assert_(...): (line 139)
        # Processing the call arguments (line 139)
        
        # Call to allclose(...): (line 139)
        # Processing the call arguments (line 139)
        
        # Call to dot(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'P' (line 139)
        P_94119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 34), 'P', False)
        # Processing the call keyword arguments (line 139)
        kwargs_94120 = {}
        # Getting the type of 'B' (line 139)
        B_94117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 28), 'B', False)
        # Obtaining the member 'dot' of a type (line 139)
        dot_94118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 28), B_94117, 'dot')
        # Calling dot(args, kwargs) (line 139)
        dot_call_result_94121 = invoke(stypy.reporting.localization.Localization(__file__, 139, 28), dot_94118, *[P_94119], **kwargs_94120)
        
        # Getting the type of 'A' (line 139)
        A_94122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 38), 'A', False)
        # Getting the type of 'eps' (line 139)
        eps_94123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 41), 'eps', False)
        # Processing the call keyword arguments (line 139)
        kwargs_94124 = {}
        # Getting the type of 'np' (line 139)
        np_94115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'np', False)
        # Obtaining the member 'allclose' of a type (line 139)
        allclose_94116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 16), np_94115, 'allclose')
        # Calling allclose(args, kwargs) (line 139)
        allclose_call_result_94125 = invoke(stypy.reporting.localization.Localization(__file__, 139, 16), allclose_94116, *[dot_call_result_94121, A_94122, eps_94123], **kwargs_94124)
        
        # Processing the call keyword arguments (line 139)
        kwargs_94126 = {}
        # Getting the type of 'assert_' (line 139)
        assert__94114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 139)
        assert__call_result_94127 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), assert__94114, *[allclose_call_result_94125], **kwargs_94126)
        
        
        # Call to _debug_print(...): (line 142)
        # Processing the call arguments (line 142)
        str_94129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 21), 'str', '-----------------------------------------')
        # Processing the call keyword arguments (line 142)
        kwargs_94130 = {}
        # Getting the type of '_debug_print' (line 142)
        _debug_print_94128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 142)
        _debug_print_call_result_94131 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), _debug_print_94128, *[str_94129], **kwargs_94130)
        
        
        # Call to _debug_print(...): (line 143)
        # Processing the call arguments (line 143)
        str_94133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 21), 'str', 'SVD routines')
        # Processing the call keyword arguments (line 143)
        kwargs_94134 = {}
        # Getting the type of '_debug_print' (line 143)
        _debug_print_94132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 143)
        _debug_print_call_result_94135 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), _debug_print_94132, *[str_94133], **kwargs_94134)
        
        
        # Call to _debug_print(...): (line 144)
        # Processing the call arguments (line 144)
        str_94137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 21), 'str', '-----------------------------------------')
        # Processing the call keyword arguments (line 144)
        kwargs_94138 = {}
        # Getting the type of '_debug_print' (line 144)
        _debug_print_94136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 144)
        _debug_print_call_result_94139 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), _debug_print_94136, *[str_94137], **kwargs_94138)
        
        
        # Call to _debug_print(...): (line 147)
        # Processing the call arguments (line 147)
        str_94141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 21), 'str', 'Calling iddp_svd / idzp_svd ...')
        # Processing the call keyword arguments (line 147)
        kwargs_94142 = {}
        # Getting the type of '_debug_print' (line 147)
        _debug_print_94140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 147)
        _debug_print_call_result_94143 = invoke(stypy.reporting.localization.Localization(__file__, 147, 8), _debug_print_94140, *[str_94141], **kwargs_94142)
        
        
        # Assigning a Call to a Name (line 148):
        
        # Assigning a Call to a Name (line 148):
        
        # Call to clock(...): (line 148)
        # Processing the call keyword arguments (line 148)
        kwargs_94146 = {}
        # Getting the type of 'time' (line 148)
        time_94144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 13), 'time', False)
        # Obtaining the member 'clock' of a type (line 148)
        clock_94145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 13), time_94144, 'clock')
        # Calling clock(args, kwargs) (line 148)
        clock_call_result_94147 = invoke(stypy.reporting.localization.Localization(__file__, 148, 13), clock_94145, *[], **kwargs_94146)
        
        # Assigning a type to the variable 't0' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 't0', clock_call_result_94147)
        
        # Assigning a Call to a Tuple (line 149):
        
        # Assigning a Subscript to a Name (line 149):
        
        # Obtaining the type of the subscript
        int_94148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 8), 'int')
        
        # Call to svd(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'A' (line 149)
        A_94151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 33), 'A', False)
        # Getting the type of 'eps' (line 149)
        eps_94152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 36), 'eps', False)
        # Processing the call keyword arguments (line 149)
        # Getting the type of 'False' (line 149)
        False_94153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 46), 'False', False)
        keyword_94154 = False_94153
        kwargs_94155 = {'rand': keyword_94154}
        # Getting the type of 'pymatrixid' (line 149)
        pymatrixid_94149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 18), 'pymatrixid', False)
        # Obtaining the member 'svd' of a type (line 149)
        svd_94150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 18), pymatrixid_94149, 'svd')
        # Calling svd(args, kwargs) (line 149)
        svd_call_result_94156 = invoke(stypy.reporting.localization.Localization(__file__, 149, 18), svd_94150, *[A_94151, eps_94152], **kwargs_94155)
        
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___94157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), svd_call_result_94156, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_94158 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), getitem___94157, int_94148)
        
        # Assigning a type to the variable 'tuple_var_assignment_93449' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'tuple_var_assignment_93449', subscript_call_result_94158)
        
        # Assigning a Subscript to a Name (line 149):
        
        # Obtaining the type of the subscript
        int_94159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 8), 'int')
        
        # Call to svd(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'A' (line 149)
        A_94162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 33), 'A', False)
        # Getting the type of 'eps' (line 149)
        eps_94163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 36), 'eps', False)
        # Processing the call keyword arguments (line 149)
        # Getting the type of 'False' (line 149)
        False_94164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 46), 'False', False)
        keyword_94165 = False_94164
        kwargs_94166 = {'rand': keyword_94165}
        # Getting the type of 'pymatrixid' (line 149)
        pymatrixid_94160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 18), 'pymatrixid', False)
        # Obtaining the member 'svd' of a type (line 149)
        svd_94161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 18), pymatrixid_94160, 'svd')
        # Calling svd(args, kwargs) (line 149)
        svd_call_result_94167 = invoke(stypy.reporting.localization.Localization(__file__, 149, 18), svd_94161, *[A_94162, eps_94163], **kwargs_94166)
        
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___94168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), svd_call_result_94167, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_94169 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), getitem___94168, int_94159)
        
        # Assigning a type to the variable 'tuple_var_assignment_93450' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'tuple_var_assignment_93450', subscript_call_result_94169)
        
        # Assigning a Subscript to a Name (line 149):
        
        # Obtaining the type of the subscript
        int_94170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 8), 'int')
        
        # Call to svd(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'A' (line 149)
        A_94173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 33), 'A', False)
        # Getting the type of 'eps' (line 149)
        eps_94174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 36), 'eps', False)
        # Processing the call keyword arguments (line 149)
        # Getting the type of 'False' (line 149)
        False_94175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 46), 'False', False)
        keyword_94176 = False_94175
        kwargs_94177 = {'rand': keyword_94176}
        # Getting the type of 'pymatrixid' (line 149)
        pymatrixid_94171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 18), 'pymatrixid', False)
        # Obtaining the member 'svd' of a type (line 149)
        svd_94172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 18), pymatrixid_94171, 'svd')
        # Calling svd(args, kwargs) (line 149)
        svd_call_result_94178 = invoke(stypy.reporting.localization.Localization(__file__, 149, 18), svd_94172, *[A_94173, eps_94174], **kwargs_94177)
        
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___94179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), svd_call_result_94178, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_94180 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), getitem___94179, int_94170)
        
        # Assigning a type to the variable 'tuple_var_assignment_93451' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'tuple_var_assignment_93451', subscript_call_result_94180)
        
        # Assigning a Name to a Name (line 149):
        # Getting the type of 'tuple_var_assignment_93449' (line 149)
        tuple_var_assignment_93449_94181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'tuple_var_assignment_93449')
        # Assigning a type to the variable 'U' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'U', tuple_var_assignment_93449_94181)
        
        # Assigning a Name to a Name (line 149):
        # Getting the type of 'tuple_var_assignment_93450' (line 149)
        tuple_var_assignment_93450_94182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'tuple_var_assignment_93450')
        # Assigning a type to the variable 'S' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 11), 'S', tuple_var_assignment_93450_94182)
        
        # Assigning a Name to a Name (line 149):
        # Getting the type of 'tuple_var_assignment_93451' (line 149)
        tuple_var_assignment_93451_94183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'tuple_var_assignment_93451')
        # Assigning a type to the variable 'V' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 14), 'V', tuple_var_assignment_93451_94183)
        
        # Assigning a BinOp to a Name (line 150):
        
        # Assigning a BinOp to a Name (line 150):
        
        # Call to clock(...): (line 150)
        # Processing the call keyword arguments (line 150)
        kwargs_94186 = {}
        # Getting the type of 'time' (line 150)
        time_94184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'time', False)
        # Obtaining the member 'clock' of a type (line 150)
        clock_94185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 12), time_94184, 'clock')
        # Calling clock(args, kwargs) (line 150)
        clock_call_result_94187 = invoke(stypy.reporting.localization.Localization(__file__, 150, 12), clock_94185, *[], **kwargs_94186)
        
        # Getting the type of 't0' (line 150)
        t0_94188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 27), 't0')
        # Applying the binary operator '-' (line 150)
        result_sub_94189 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 12), '-', clock_call_result_94187, t0_94188)
        
        # Assigning a type to the variable 't' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 't', result_sub_94189)
        
        # Assigning a Call to a Name (line 151):
        
        # Assigning a Call to a Name (line 151):
        
        # Call to dot(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'U' (line 151)
        U_94192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 19), 'U', False)
        
        # Call to dot(...): (line 151)
        # Processing the call arguments (line 151)
        
        # Call to diag(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'S' (line 151)
        S_94197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 37), 'S', False)
        # Processing the call keyword arguments (line 151)
        kwargs_94198 = {}
        # Getting the type of 'np' (line 151)
        np_94195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 29), 'np', False)
        # Obtaining the member 'diag' of a type (line 151)
        diag_94196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 29), np_94195, 'diag')
        # Calling diag(args, kwargs) (line 151)
        diag_call_result_94199 = invoke(stypy.reporting.localization.Localization(__file__, 151, 29), diag_94196, *[S_94197], **kwargs_94198)
        
        
        # Call to conj(...): (line 151)
        # Processing the call keyword arguments (line 151)
        kwargs_94203 = {}
        # Getting the type of 'V' (line 151)
        V_94200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 41), 'V', False)
        # Obtaining the member 'T' of a type (line 151)
        T_94201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 41), V_94200, 'T')
        # Obtaining the member 'conj' of a type (line 151)
        conj_94202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 41), T_94201, 'conj')
        # Calling conj(args, kwargs) (line 151)
        conj_call_result_94204 = invoke(stypy.reporting.localization.Localization(__file__, 151, 41), conj_94202, *[], **kwargs_94203)
        
        # Processing the call keyword arguments (line 151)
        kwargs_94205 = {}
        # Getting the type of 'np' (line 151)
        np_94193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 22), 'np', False)
        # Obtaining the member 'dot' of a type (line 151)
        dot_94194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 22), np_94193, 'dot')
        # Calling dot(args, kwargs) (line 151)
        dot_call_result_94206 = invoke(stypy.reporting.localization.Localization(__file__, 151, 22), dot_94194, *[diag_call_result_94199, conj_call_result_94204], **kwargs_94205)
        
        # Processing the call keyword arguments (line 151)
        kwargs_94207 = {}
        # Getting the type of 'np' (line 151)
        np_94190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'np', False)
        # Obtaining the member 'dot' of a type (line 151)
        dot_94191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 12), np_94190, 'dot')
        # Calling dot(args, kwargs) (line 151)
        dot_call_result_94208 = invoke(stypy.reporting.localization.Localization(__file__, 151, 12), dot_94191, *[U_94192, dot_call_result_94206], **kwargs_94207)
        
        # Assigning a type to the variable 'B' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'B', dot_call_result_94208)
        
        # Call to _debug_print(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'fmt' (line 152)
        fmt_94210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 21), 'fmt', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 152)
        tuple_94211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 152)
        # Adding element type (line 152)
        # Getting the type of 't' (line 152)
        t_94212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 28), 't', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 28), tuple_94211, t_94212)
        # Adding element type (line 152)
        
        # Call to allclose(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'A' (line 152)
        A_94215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 43), 'A', False)
        # Getting the type of 'B' (line 152)
        B_94216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 46), 'B', False)
        # Getting the type of 'eps' (line 152)
        eps_94217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 49), 'eps', False)
        # Processing the call keyword arguments (line 152)
        kwargs_94218 = {}
        # Getting the type of 'np' (line 152)
        np_94213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 31), 'np', False)
        # Obtaining the member 'allclose' of a type (line 152)
        allclose_94214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 31), np_94213, 'allclose')
        # Calling allclose(args, kwargs) (line 152)
        allclose_call_result_94219 = invoke(stypy.reporting.localization.Localization(__file__, 152, 31), allclose_94214, *[A_94215, B_94216, eps_94217], **kwargs_94218)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 28), tuple_94211, allclose_call_result_94219)
        
        # Applying the binary operator '%' (line 152)
        result_mod_94220 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 21), '%', fmt_94210, tuple_94211)
        
        # Processing the call keyword arguments (line 152)
        kwargs_94221 = {}
        # Getting the type of '_debug_print' (line 152)
        _debug_print_94209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 152)
        _debug_print_call_result_94222 = invoke(stypy.reporting.localization.Localization(__file__, 152, 8), _debug_print_94209, *[result_mod_94220], **kwargs_94221)
        
        
        # Call to assert_(...): (line 153)
        # Processing the call arguments (line 153)
        
        # Call to allclose(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'A' (line 153)
        A_94226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 28), 'A', False)
        # Getting the type of 'B' (line 153)
        B_94227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 31), 'B', False)
        # Getting the type of 'eps' (line 153)
        eps_94228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 34), 'eps', False)
        # Processing the call keyword arguments (line 153)
        kwargs_94229 = {}
        # Getting the type of 'np' (line 153)
        np_94224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'np', False)
        # Obtaining the member 'allclose' of a type (line 153)
        allclose_94225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 16), np_94224, 'allclose')
        # Calling allclose(args, kwargs) (line 153)
        allclose_call_result_94230 = invoke(stypy.reporting.localization.Localization(__file__, 153, 16), allclose_94225, *[A_94226, B_94227, eps_94228], **kwargs_94229)
        
        # Processing the call keyword arguments (line 153)
        kwargs_94231 = {}
        # Getting the type of 'assert_' (line 153)
        assert__94223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 153)
        assert__call_result_94232 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), assert__94223, *[allclose_call_result_94230], **kwargs_94231)
        
        
        # Call to _debug_print(...): (line 155)
        # Processing the call arguments (line 155)
        str_94234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 21), 'str', 'Calling iddp_asvd / idzp_asvd...')
        # Processing the call keyword arguments (line 155)
        kwargs_94235 = {}
        # Getting the type of '_debug_print' (line 155)
        _debug_print_94233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 155)
        _debug_print_call_result_94236 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), _debug_print_94233, *[str_94234], **kwargs_94235)
        
        
        # Assigning a Call to a Name (line 156):
        
        # Assigning a Call to a Name (line 156):
        
        # Call to clock(...): (line 156)
        # Processing the call keyword arguments (line 156)
        kwargs_94239 = {}
        # Getting the type of 'time' (line 156)
        time_94237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 13), 'time', False)
        # Obtaining the member 'clock' of a type (line 156)
        clock_94238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 13), time_94237, 'clock')
        # Calling clock(args, kwargs) (line 156)
        clock_call_result_94240 = invoke(stypy.reporting.localization.Localization(__file__, 156, 13), clock_94238, *[], **kwargs_94239)
        
        # Assigning a type to the variable 't0' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 't0', clock_call_result_94240)
        
        # Assigning a Call to a Tuple (line 157):
        
        # Assigning a Subscript to a Name (line 157):
        
        # Obtaining the type of the subscript
        int_94241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 8), 'int')
        
        # Call to svd(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'A' (line 157)
        A_94244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 33), 'A', False)
        # Getting the type of 'eps' (line 157)
        eps_94245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 36), 'eps', False)
        # Processing the call keyword arguments (line 157)
        kwargs_94246 = {}
        # Getting the type of 'pymatrixid' (line 157)
        pymatrixid_94242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 18), 'pymatrixid', False)
        # Obtaining the member 'svd' of a type (line 157)
        svd_94243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 18), pymatrixid_94242, 'svd')
        # Calling svd(args, kwargs) (line 157)
        svd_call_result_94247 = invoke(stypy.reporting.localization.Localization(__file__, 157, 18), svd_94243, *[A_94244, eps_94245], **kwargs_94246)
        
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___94248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), svd_call_result_94247, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_94249 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), getitem___94248, int_94241)
        
        # Assigning a type to the variable 'tuple_var_assignment_93452' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'tuple_var_assignment_93452', subscript_call_result_94249)
        
        # Assigning a Subscript to a Name (line 157):
        
        # Obtaining the type of the subscript
        int_94250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 8), 'int')
        
        # Call to svd(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'A' (line 157)
        A_94253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 33), 'A', False)
        # Getting the type of 'eps' (line 157)
        eps_94254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 36), 'eps', False)
        # Processing the call keyword arguments (line 157)
        kwargs_94255 = {}
        # Getting the type of 'pymatrixid' (line 157)
        pymatrixid_94251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 18), 'pymatrixid', False)
        # Obtaining the member 'svd' of a type (line 157)
        svd_94252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 18), pymatrixid_94251, 'svd')
        # Calling svd(args, kwargs) (line 157)
        svd_call_result_94256 = invoke(stypy.reporting.localization.Localization(__file__, 157, 18), svd_94252, *[A_94253, eps_94254], **kwargs_94255)
        
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___94257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), svd_call_result_94256, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_94258 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), getitem___94257, int_94250)
        
        # Assigning a type to the variable 'tuple_var_assignment_93453' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'tuple_var_assignment_93453', subscript_call_result_94258)
        
        # Assigning a Subscript to a Name (line 157):
        
        # Obtaining the type of the subscript
        int_94259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 8), 'int')
        
        # Call to svd(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'A' (line 157)
        A_94262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 33), 'A', False)
        # Getting the type of 'eps' (line 157)
        eps_94263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 36), 'eps', False)
        # Processing the call keyword arguments (line 157)
        kwargs_94264 = {}
        # Getting the type of 'pymatrixid' (line 157)
        pymatrixid_94260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 18), 'pymatrixid', False)
        # Obtaining the member 'svd' of a type (line 157)
        svd_94261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 18), pymatrixid_94260, 'svd')
        # Calling svd(args, kwargs) (line 157)
        svd_call_result_94265 = invoke(stypy.reporting.localization.Localization(__file__, 157, 18), svd_94261, *[A_94262, eps_94263], **kwargs_94264)
        
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___94266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), svd_call_result_94265, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_94267 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), getitem___94266, int_94259)
        
        # Assigning a type to the variable 'tuple_var_assignment_93454' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'tuple_var_assignment_93454', subscript_call_result_94267)
        
        # Assigning a Name to a Name (line 157):
        # Getting the type of 'tuple_var_assignment_93452' (line 157)
        tuple_var_assignment_93452_94268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'tuple_var_assignment_93452')
        # Assigning a type to the variable 'U' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'U', tuple_var_assignment_93452_94268)
        
        # Assigning a Name to a Name (line 157):
        # Getting the type of 'tuple_var_assignment_93453' (line 157)
        tuple_var_assignment_93453_94269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'tuple_var_assignment_93453')
        # Assigning a type to the variable 'S' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 11), 'S', tuple_var_assignment_93453_94269)
        
        # Assigning a Name to a Name (line 157):
        # Getting the type of 'tuple_var_assignment_93454' (line 157)
        tuple_var_assignment_93454_94270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'tuple_var_assignment_93454')
        # Assigning a type to the variable 'V' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 14), 'V', tuple_var_assignment_93454_94270)
        
        # Assigning a BinOp to a Name (line 158):
        
        # Assigning a BinOp to a Name (line 158):
        
        # Call to clock(...): (line 158)
        # Processing the call keyword arguments (line 158)
        kwargs_94273 = {}
        # Getting the type of 'time' (line 158)
        time_94271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'time', False)
        # Obtaining the member 'clock' of a type (line 158)
        clock_94272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 12), time_94271, 'clock')
        # Calling clock(args, kwargs) (line 158)
        clock_call_result_94274 = invoke(stypy.reporting.localization.Localization(__file__, 158, 12), clock_94272, *[], **kwargs_94273)
        
        # Getting the type of 't0' (line 158)
        t0_94275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 27), 't0')
        # Applying the binary operator '-' (line 158)
        result_sub_94276 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 12), '-', clock_call_result_94274, t0_94275)
        
        # Assigning a type to the variable 't' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 't', result_sub_94276)
        
        # Assigning a Call to a Name (line 159):
        
        # Assigning a Call to a Name (line 159):
        
        # Call to dot(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'U' (line 159)
        U_94279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 19), 'U', False)
        
        # Call to dot(...): (line 159)
        # Processing the call arguments (line 159)
        
        # Call to diag(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'S' (line 159)
        S_94284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 37), 'S', False)
        # Processing the call keyword arguments (line 159)
        kwargs_94285 = {}
        # Getting the type of 'np' (line 159)
        np_94282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 29), 'np', False)
        # Obtaining the member 'diag' of a type (line 159)
        diag_94283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 29), np_94282, 'diag')
        # Calling diag(args, kwargs) (line 159)
        diag_call_result_94286 = invoke(stypy.reporting.localization.Localization(__file__, 159, 29), diag_94283, *[S_94284], **kwargs_94285)
        
        
        # Call to conj(...): (line 159)
        # Processing the call keyword arguments (line 159)
        kwargs_94290 = {}
        # Getting the type of 'V' (line 159)
        V_94287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 41), 'V', False)
        # Obtaining the member 'T' of a type (line 159)
        T_94288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 41), V_94287, 'T')
        # Obtaining the member 'conj' of a type (line 159)
        conj_94289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 41), T_94288, 'conj')
        # Calling conj(args, kwargs) (line 159)
        conj_call_result_94291 = invoke(stypy.reporting.localization.Localization(__file__, 159, 41), conj_94289, *[], **kwargs_94290)
        
        # Processing the call keyword arguments (line 159)
        kwargs_94292 = {}
        # Getting the type of 'np' (line 159)
        np_94280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 22), 'np', False)
        # Obtaining the member 'dot' of a type (line 159)
        dot_94281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 22), np_94280, 'dot')
        # Calling dot(args, kwargs) (line 159)
        dot_call_result_94293 = invoke(stypy.reporting.localization.Localization(__file__, 159, 22), dot_94281, *[diag_call_result_94286, conj_call_result_94291], **kwargs_94292)
        
        # Processing the call keyword arguments (line 159)
        kwargs_94294 = {}
        # Getting the type of 'np' (line 159)
        np_94277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'np', False)
        # Obtaining the member 'dot' of a type (line 159)
        dot_94278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 12), np_94277, 'dot')
        # Calling dot(args, kwargs) (line 159)
        dot_call_result_94295 = invoke(stypy.reporting.localization.Localization(__file__, 159, 12), dot_94278, *[U_94279, dot_call_result_94293], **kwargs_94294)
        
        # Assigning a type to the variable 'B' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'B', dot_call_result_94295)
        
        # Call to _debug_print(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'fmt' (line 160)
        fmt_94297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 21), 'fmt', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 160)
        tuple_94298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 160)
        # Adding element type (line 160)
        # Getting the type of 't' (line 160)
        t_94299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 28), 't', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 28), tuple_94298, t_94299)
        # Adding element type (line 160)
        
        # Call to allclose(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'A' (line 160)
        A_94302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 43), 'A', False)
        # Getting the type of 'B' (line 160)
        B_94303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 46), 'B', False)
        # Getting the type of 'eps' (line 160)
        eps_94304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 49), 'eps', False)
        # Processing the call keyword arguments (line 160)
        kwargs_94305 = {}
        # Getting the type of 'np' (line 160)
        np_94300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 31), 'np', False)
        # Obtaining the member 'allclose' of a type (line 160)
        allclose_94301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 31), np_94300, 'allclose')
        # Calling allclose(args, kwargs) (line 160)
        allclose_call_result_94306 = invoke(stypy.reporting.localization.Localization(__file__, 160, 31), allclose_94301, *[A_94302, B_94303, eps_94304], **kwargs_94305)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 28), tuple_94298, allclose_call_result_94306)
        
        # Applying the binary operator '%' (line 160)
        result_mod_94307 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 21), '%', fmt_94297, tuple_94298)
        
        # Processing the call keyword arguments (line 160)
        kwargs_94308 = {}
        # Getting the type of '_debug_print' (line 160)
        _debug_print_94296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 160)
        _debug_print_call_result_94309 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), _debug_print_94296, *[result_mod_94307], **kwargs_94308)
        
        
        # Call to assert_(...): (line 161)
        # Processing the call arguments (line 161)
        
        # Call to allclose(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'A' (line 161)
        A_94313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'A', False)
        # Getting the type of 'B' (line 161)
        B_94314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 31), 'B', False)
        # Getting the type of 'eps' (line 161)
        eps_94315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 34), 'eps', False)
        # Processing the call keyword arguments (line 161)
        kwargs_94316 = {}
        # Getting the type of 'np' (line 161)
        np_94311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'np', False)
        # Obtaining the member 'allclose' of a type (line 161)
        allclose_94312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 16), np_94311, 'allclose')
        # Calling allclose(args, kwargs) (line 161)
        allclose_call_result_94317 = invoke(stypy.reporting.localization.Localization(__file__, 161, 16), allclose_94312, *[A_94313, B_94314, eps_94315], **kwargs_94316)
        
        # Processing the call keyword arguments (line 161)
        kwargs_94318 = {}
        # Getting the type of 'assert_' (line 161)
        assert__94310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 161)
        assert__call_result_94319 = invoke(stypy.reporting.localization.Localization(__file__, 161, 8), assert__94310, *[allclose_call_result_94317], **kwargs_94318)
        
        
        # Call to _debug_print(...): (line 163)
        # Processing the call arguments (line 163)
        str_94321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 21), 'str', 'Calling iddp_rsvd / idzp_rsvd...')
        # Processing the call keyword arguments (line 163)
        kwargs_94322 = {}
        # Getting the type of '_debug_print' (line 163)
        _debug_print_94320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 163)
        _debug_print_call_result_94323 = invoke(stypy.reporting.localization.Localization(__file__, 163, 8), _debug_print_94320, *[str_94321], **kwargs_94322)
        
        
        # Assigning a Call to a Name (line 164):
        
        # Assigning a Call to a Name (line 164):
        
        # Call to clock(...): (line 164)
        # Processing the call keyword arguments (line 164)
        kwargs_94326 = {}
        # Getting the type of 'time' (line 164)
        time_94324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 13), 'time', False)
        # Obtaining the member 'clock' of a type (line 164)
        clock_94325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 13), time_94324, 'clock')
        # Calling clock(args, kwargs) (line 164)
        clock_call_result_94327 = invoke(stypy.reporting.localization.Localization(__file__, 164, 13), clock_94325, *[], **kwargs_94326)
        
        # Assigning a type to the variable 't0' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 't0', clock_call_result_94327)
        
        # Assigning a Call to a Tuple (line 165):
        
        # Assigning a Subscript to a Name (line 165):
        
        # Obtaining the type of the subscript
        int_94328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 8), 'int')
        
        # Call to svd(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'L' (line 165)
        L_94331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 33), 'L', False)
        # Getting the type of 'eps' (line 165)
        eps_94332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 36), 'eps', False)
        # Processing the call keyword arguments (line 165)
        kwargs_94333 = {}
        # Getting the type of 'pymatrixid' (line 165)
        pymatrixid_94329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 18), 'pymatrixid', False)
        # Obtaining the member 'svd' of a type (line 165)
        svd_94330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 18), pymatrixid_94329, 'svd')
        # Calling svd(args, kwargs) (line 165)
        svd_call_result_94334 = invoke(stypy.reporting.localization.Localization(__file__, 165, 18), svd_94330, *[L_94331, eps_94332], **kwargs_94333)
        
        # Obtaining the member '__getitem__' of a type (line 165)
        getitem___94335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), svd_call_result_94334, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 165)
        subscript_call_result_94336 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), getitem___94335, int_94328)
        
        # Assigning a type to the variable 'tuple_var_assignment_93455' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'tuple_var_assignment_93455', subscript_call_result_94336)
        
        # Assigning a Subscript to a Name (line 165):
        
        # Obtaining the type of the subscript
        int_94337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 8), 'int')
        
        # Call to svd(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'L' (line 165)
        L_94340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 33), 'L', False)
        # Getting the type of 'eps' (line 165)
        eps_94341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 36), 'eps', False)
        # Processing the call keyword arguments (line 165)
        kwargs_94342 = {}
        # Getting the type of 'pymatrixid' (line 165)
        pymatrixid_94338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 18), 'pymatrixid', False)
        # Obtaining the member 'svd' of a type (line 165)
        svd_94339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 18), pymatrixid_94338, 'svd')
        # Calling svd(args, kwargs) (line 165)
        svd_call_result_94343 = invoke(stypy.reporting.localization.Localization(__file__, 165, 18), svd_94339, *[L_94340, eps_94341], **kwargs_94342)
        
        # Obtaining the member '__getitem__' of a type (line 165)
        getitem___94344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), svd_call_result_94343, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 165)
        subscript_call_result_94345 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), getitem___94344, int_94337)
        
        # Assigning a type to the variable 'tuple_var_assignment_93456' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'tuple_var_assignment_93456', subscript_call_result_94345)
        
        # Assigning a Subscript to a Name (line 165):
        
        # Obtaining the type of the subscript
        int_94346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 8), 'int')
        
        # Call to svd(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'L' (line 165)
        L_94349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 33), 'L', False)
        # Getting the type of 'eps' (line 165)
        eps_94350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 36), 'eps', False)
        # Processing the call keyword arguments (line 165)
        kwargs_94351 = {}
        # Getting the type of 'pymatrixid' (line 165)
        pymatrixid_94347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 18), 'pymatrixid', False)
        # Obtaining the member 'svd' of a type (line 165)
        svd_94348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 18), pymatrixid_94347, 'svd')
        # Calling svd(args, kwargs) (line 165)
        svd_call_result_94352 = invoke(stypy.reporting.localization.Localization(__file__, 165, 18), svd_94348, *[L_94349, eps_94350], **kwargs_94351)
        
        # Obtaining the member '__getitem__' of a type (line 165)
        getitem___94353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), svd_call_result_94352, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 165)
        subscript_call_result_94354 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), getitem___94353, int_94346)
        
        # Assigning a type to the variable 'tuple_var_assignment_93457' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'tuple_var_assignment_93457', subscript_call_result_94354)
        
        # Assigning a Name to a Name (line 165):
        # Getting the type of 'tuple_var_assignment_93455' (line 165)
        tuple_var_assignment_93455_94355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'tuple_var_assignment_93455')
        # Assigning a type to the variable 'U' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'U', tuple_var_assignment_93455_94355)
        
        # Assigning a Name to a Name (line 165):
        # Getting the type of 'tuple_var_assignment_93456' (line 165)
        tuple_var_assignment_93456_94356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'tuple_var_assignment_93456')
        # Assigning a type to the variable 'S' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 11), 'S', tuple_var_assignment_93456_94356)
        
        # Assigning a Name to a Name (line 165):
        # Getting the type of 'tuple_var_assignment_93457' (line 165)
        tuple_var_assignment_93457_94357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'tuple_var_assignment_93457')
        # Assigning a type to the variable 'V' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 14), 'V', tuple_var_assignment_93457_94357)
        
        # Assigning a BinOp to a Name (line 166):
        
        # Assigning a BinOp to a Name (line 166):
        
        # Call to clock(...): (line 166)
        # Processing the call keyword arguments (line 166)
        kwargs_94360 = {}
        # Getting the type of 'time' (line 166)
        time_94358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'time', False)
        # Obtaining the member 'clock' of a type (line 166)
        clock_94359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), time_94358, 'clock')
        # Calling clock(args, kwargs) (line 166)
        clock_call_result_94361 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), clock_94359, *[], **kwargs_94360)
        
        # Getting the type of 't0' (line 166)
        t0_94362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 27), 't0')
        # Applying the binary operator '-' (line 166)
        result_sub_94363 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 12), '-', clock_call_result_94361, t0_94362)
        
        # Assigning a type to the variable 't' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 't', result_sub_94363)
        
        # Assigning a Call to a Name (line 167):
        
        # Assigning a Call to a Name (line 167):
        
        # Call to dot(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'U' (line 167)
        U_94366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 19), 'U', False)
        
        # Call to dot(...): (line 167)
        # Processing the call arguments (line 167)
        
        # Call to diag(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'S' (line 167)
        S_94371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 37), 'S', False)
        # Processing the call keyword arguments (line 167)
        kwargs_94372 = {}
        # Getting the type of 'np' (line 167)
        np_94369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 29), 'np', False)
        # Obtaining the member 'diag' of a type (line 167)
        diag_94370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 29), np_94369, 'diag')
        # Calling diag(args, kwargs) (line 167)
        diag_call_result_94373 = invoke(stypy.reporting.localization.Localization(__file__, 167, 29), diag_94370, *[S_94371], **kwargs_94372)
        
        
        # Call to conj(...): (line 167)
        # Processing the call keyword arguments (line 167)
        kwargs_94377 = {}
        # Getting the type of 'V' (line 167)
        V_94374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 41), 'V', False)
        # Obtaining the member 'T' of a type (line 167)
        T_94375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 41), V_94374, 'T')
        # Obtaining the member 'conj' of a type (line 167)
        conj_94376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 41), T_94375, 'conj')
        # Calling conj(args, kwargs) (line 167)
        conj_call_result_94378 = invoke(stypy.reporting.localization.Localization(__file__, 167, 41), conj_94376, *[], **kwargs_94377)
        
        # Processing the call keyword arguments (line 167)
        kwargs_94379 = {}
        # Getting the type of 'np' (line 167)
        np_94367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 22), 'np', False)
        # Obtaining the member 'dot' of a type (line 167)
        dot_94368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 22), np_94367, 'dot')
        # Calling dot(args, kwargs) (line 167)
        dot_call_result_94380 = invoke(stypy.reporting.localization.Localization(__file__, 167, 22), dot_94368, *[diag_call_result_94373, conj_call_result_94378], **kwargs_94379)
        
        # Processing the call keyword arguments (line 167)
        kwargs_94381 = {}
        # Getting the type of 'np' (line 167)
        np_94364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'np', False)
        # Obtaining the member 'dot' of a type (line 167)
        dot_94365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 12), np_94364, 'dot')
        # Calling dot(args, kwargs) (line 167)
        dot_call_result_94382 = invoke(stypy.reporting.localization.Localization(__file__, 167, 12), dot_94365, *[U_94366, dot_call_result_94380], **kwargs_94381)
        
        # Assigning a type to the variable 'B' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'B', dot_call_result_94382)
        
        # Call to _debug_print(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'fmt' (line 168)
        fmt_94384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 21), 'fmt', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 168)
        tuple_94385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 168)
        # Adding element type (line 168)
        # Getting the type of 't' (line 168)
        t_94386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 28), 't', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 28), tuple_94385, t_94386)
        # Adding element type (line 168)
        
        # Call to allclose(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'A' (line 168)
        A_94389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 43), 'A', False)
        # Getting the type of 'B' (line 168)
        B_94390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 46), 'B', False)
        # Getting the type of 'eps' (line 168)
        eps_94391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 49), 'eps', False)
        # Processing the call keyword arguments (line 168)
        kwargs_94392 = {}
        # Getting the type of 'np' (line 168)
        np_94387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 31), 'np', False)
        # Obtaining the member 'allclose' of a type (line 168)
        allclose_94388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 31), np_94387, 'allclose')
        # Calling allclose(args, kwargs) (line 168)
        allclose_call_result_94393 = invoke(stypy.reporting.localization.Localization(__file__, 168, 31), allclose_94388, *[A_94389, B_94390, eps_94391], **kwargs_94392)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 28), tuple_94385, allclose_call_result_94393)
        
        # Applying the binary operator '%' (line 168)
        result_mod_94394 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 21), '%', fmt_94384, tuple_94385)
        
        # Processing the call keyword arguments (line 168)
        kwargs_94395 = {}
        # Getting the type of '_debug_print' (line 168)
        _debug_print_94383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 168)
        _debug_print_call_result_94396 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), _debug_print_94383, *[result_mod_94394], **kwargs_94395)
        
        
        # Call to assert_(...): (line 169)
        # Processing the call arguments (line 169)
        
        # Call to allclose(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'A' (line 169)
        A_94400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 28), 'A', False)
        # Getting the type of 'B' (line 169)
        B_94401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 31), 'B', False)
        # Getting the type of 'eps' (line 169)
        eps_94402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 34), 'eps', False)
        # Processing the call keyword arguments (line 169)
        kwargs_94403 = {}
        # Getting the type of 'np' (line 169)
        np_94398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'np', False)
        # Obtaining the member 'allclose' of a type (line 169)
        allclose_94399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 16), np_94398, 'allclose')
        # Calling allclose(args, kwargs) (line 169)
        allclose_call_result_94404 = invoke(stypy.reporting.localization.Localization(__file__, 169, 16), allclose_94399, *[A_94400, B_94401, eps_94402], **kwargs_94403)
        
        # Processing the call keyword arguments (line 169)
        kwargs_94405 = {}
        # Getting the type of 'assert_' (line 169)
        assert__94397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 169)
        assert__call_result_94406 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), assert__94397, *[allclose_call_result_94404], **kwargs_94405)
        
        
        # Assigning a Name to a Name (line 172):
        
        # Assigning a Name to a Name (line 172):
        # Getting the type of 'rank' (line 172)
        rank_94407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'rank')
        # Assigning a type to the variable 'k' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'k', rank_94407)
        
        # Call to _debug_print(...): (line 174)
        # Processing the call arguments (line 174)
        str_94409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 21), 'str', 'Calling iddr_svd / idzr_svd ...')
        # Processing the call keyword arguments (line 174)
        kwargs_94410 = {}
        # Getting the type of '_debug_print' (line 174)
        _debug_print_94408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 174)
        _debug_print_call_result_94411 = invoke(stypy.reporting.localization.Localization(__file__, 174, 8), _debug_print_94408, *[str_94409], **kwargs_94410)
        
        
        # Assigning a Call to a Name (line 175):
        
        # Assigning a Call to a Name (line 175):
        
        # Call to clock(...): (line 175)
        # Processing the call keyword arguments (line 175)
        kwargs_94414 = {}
        # Getting the type of 'time' (line 175)
        time_94412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 13), 'time', False)
        # Obtaining the member 'clock' of a type (line 175)
        clock_94413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 13), time_94412, 'clock')
        # Calling clock(args, kwargs) (line 175)
        clock_call_result_94415 = invoke(stypy.reporting.localization.Localization(__file__, 175, 13), clock_94413, *[], **kwargs_94414)
        
        # Assigning a type to the variable 't0' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 't0', clock_call_result_94415)
        
        # Assigning a Call to a Tuple (line 176):
        
        # Assigning a Subscript to a Name (line 176):
        
        # Obtaining the type of the subscript
        int_94416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 8), 'int')
        
        # Call to svd(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'A' (line 176)
        A_94419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 33), 'A', False)
        # Getting the type of 'k' (line 176)
        k_94420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 36), 'k', False)
        # Processing the call keyword arguments (line 176)
        # Getting the type of 'False' (line 176)
        False_94421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 44), 'False', False)
        keyword_94422 = False_94421
        kwargs_94423 = {'rand': keyword_94422}
        # Getting the type of 'pymatrixid' (line 176)
        pymatrixid_94417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 18), 'pymatrixid', False)
        # Obtaining the member 'svd' of a type (line 176)
        svd_94418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 18), pymatrixid_94417, 'svd')
        # Calling svd(args, kwargs) (line 176)
        svd_call_result_94424 = invoke(stypy.reporting.localization.Localization(__file__, 176, 18), svd_94418, *[A_94419, k_94420], **kwargs_94423)
        
        # Obtaining the member '__getitem__' of a type (line 176)
        getitem___94425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), svd_call_result_94424, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 176)
        subscript_call_result_94426 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), getitem___94425, int_94416)
        
        # Assigning a type to the variable 'tuple_var_assignment_93458' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'tuple_var_assignment_93458', subscript_call_result_94426)
        
        # Assigning a Subscript to a Name (line 176):
        
        # Obtaining the type of the subscript
        int_94427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 8), 'int')
        
        # Call to svd(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'A' (line 176)
        A_94430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 33), 'A', False)
        # Getting the type of 'k' (line 176)
        k_94431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 36), 'k', False)
        # Processing the call keyword arguments (line 176)
        # Getting the type of 'False' (line 176)
        False_94432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 44), 'False', False)
        keyword_94433 = False_94432
        kwargs_94434 = {'rand': keyword_94433}
        # Getting the type of 'pymatrixid' (line 176)
        pymatrixid_94428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 18), 'pymatrixid', False)
        # Obtaining the member 'svd' of a type (line 176)
        svd_94429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 18), pymatrixid_94428, 'svd')
        # Calling svd(args, kwargs) (line 176)
        svd_call_result_94435 = invoke(stypy.reporting.localization.Localization(__file__, 176, 18), svd_94429, *[A_94430, k_94431], **kwargs_94434)
        
        # Obtaining the member '__getitem__' of a type (line 176)
        getitem___94436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), svd_call_result_94435, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 176)
        subscript_call_result_94437 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), getitem___94436, int_94427)
        
        # Assigning a type to the variable 'tuple_var_assignment_93459' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'tuple_var_assignment_93459', subscript_call_result_94437)
        
        # Assigning a Subscript to a Name (line 176):
        
        # Obtaining the type of the subscript
        int_94438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 8), 'int')
        
        # Call to svd(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'A' (line 176)
        A_94441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 33), 'A', False)
        # Getting the type of 'k' (line 176)
        k_94442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 36), 'k', False)
        # Processing the call keyword arguments (line 176)
        # Getting the type of 'False' (line 176)
        False_94443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 44), 'False', False)
        keyword_94444 = False_94443
        kwargs_94445 = {'rand': keyword_94444}
        # Getting the type of 'pymatrixid' (line 176)
        pymatrixid_94439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 18), 'pymatrixid', False)
        # Obtaining the member 'svd' of a type (line 176)
        svd_94440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 18), pymatrixid_94439, 'svd')
        # Calling svd(args, kwargs) (line 176)
        svd_call_result_94446 = invoke(stypy.reporting.localization.Localization(__file__, 176, 18), svd_94440, *[A_94441, k_94442], **kwargs_94445)
        
        # Obtaining the member '__getitem__' of a type (line 176)
        getitem___94447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), svd_call_result_94446, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 176)
        subscript_call_result_94448 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), getitem___94447, int_94438)
        
        # Assigning a type to the variable 'tuple_var_assignment_93460' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'tuple_var_assignment_93460', subscript_call_result_94448)
        
        # Assigning a Name to a Name (line 176):
        # Getting the type of 'tuple_var_assignment_93458' (line 176)
        tuple_var_assignment_93458_94449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'tuple_var_assignment_93458')
        # Assigning a type to the variable 'U' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'U', tuple_var_assignment_93458_94449)
        
        # Assigning a Name to a Name (line 176):
        # Getting the type of 'tuple_var_assignment_93459' (line 176)
        tuple_var_assignment_93459_94450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'tuple_var_assignment_93459')
        # Assigning a type to the variable 'S' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 11), 'S', tuple_var_assignment_93459_94450)
        
        # Assigning a Name to a Name (line 176):
        # Getting the type of 'tuple_var_assignment_93460' (line 176)
        tuple_var_assignment_93460_94451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'tuple_var_assignment_93460')
        # Assigning a type to the variable 'V' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 14), 'V', tuple_var_assignment_93460_94451)
        
        # Assigning a BinOp to a Name (line 177):
        
        # Assigning a BinOp to a Name (line 177):
        
        # Call to clock(...): (line 177)
        # Processing the call keyword arguments (line 177)
        kwargs_94454 = {}
        # Getting the type of 'time' (line 177)
        time_94452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'time', False)
        # Obtaining the member 'clock' of a type (line 177)
        clock_94453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 12), time_94452, 'clock')
        # Calling clock(args, kwargs) (line 177)
        clock_call_result_94455 = invoke(stypy.reporting.localization.Localization(__file__, 177, 12), clock_94453, *[], **kwargs_94454)
        
        # Getting the type of 't0' (line 177)
        t0_94456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 27), 't0')
        # Applying the binary operator '-' (line 177)
        result_sub_94457 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 12), '-', clock_call_result_94455, t0_94456)
        
        # Assigning a type to the variable 't' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 't', result_sub_94457)
        
        # Assigning a Call to a Name (line 178):
        
        # Assigning a Call to a Name (line 178):
        
        # Call to dot(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'U' (line 178)
        U_94460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 19), 'U', False)
        
        # Call to dot(...): (line 178)
        # Processing the call arguments (line 178)
        
        # Call to diag(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'S' (line 178)
        S_94465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 37), 'S', False)
        # Processing the call keyword arguments (line 178)
        kwargs_94466 = {}
        # Getting the type of 'np' (line 178)
        np_94463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 29), 'np', False)
        # Obtaining the member 'diag' of a type (line 178)
        diag_94464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 29), np_94463, 'diag')
        # Calling diag(args, kwargs) (line 178)
        diag_call_result_94467 = invoke(stypy.reporting.localization.Localization(__file__, 178, 29), diag_94464, *[S_94465], **kwargs_94466)
        
        
        # Call to conj(...): (line 178)
        # Processing the call keyword arguments (line 178)
        kwargs_94471 = {}
        # Getting the type of 'V' (line 178)
        V_94468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 41), 'V', False)
        # Obtaining the member 'T' of a type (line 178)
        T_94469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 41), V_94468, 'T')
        # Obtaining the member 'conj' of a type (line 178)
        conj_94470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 41), T_94469, 'conj')
        # Calling conj(args, kwargs) (line 178)
        conj_call_result_94472 = invoke(stypy.reporting.localization.Localization(__file__, 178, 41), conj_94470, *[], **kwargs_94471)
        
        # Processing the call keyword arguments (line 178)
        kwargs_94473 = {}
        # Getting the type of 'np' (line 178)
        np_94461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 22), 'np', False)
        # Obtaining the member 'dot' of a type (line 178)
        dot_94462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 22), np_94461, 'dot')
        # Calling dot(args, kwargs) (line 178)
        dot_call_result_94474 = invoke(stypy.reporting.localization.Localization(__file__, 178, 22), dot_94462, *[diag_call_result_94467, conj_call_result_94472], **kwargs_94473)
        
        # Processing the call keyword arguments (line 178)
        kwargs_94475 = {}
        # Getting the type of 'np' (line 178)
        np_94458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'np', False)
        # Obtaining the member 'dot' of a type (line 178)
        dot_94459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 12), np_94458, 'dot')
        # Calling dot(args, kwargs) (line 178)
        dot_call_result_94476 = invoke(stypy.reporting.localization.Localization(__file__, 178, 12), dot_94459, *[U_94460, dot_call_result_94474], **kwargs_94475)
        
        # Assigning a type to the variable 'B' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'B', dot_call_result_94476)
        
        # Call to _debug_print(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'fmt' (line 179)
        fmt_94478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 21), 'fmt', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 179)
        tuple_94479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 179)
        # Adding element type (line 179)
        # Getting the type of 't' (line 179)
        t_94480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 28), 't', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 28), tuple_94479, t_94480)
        # Adding element type (line 179)
        
        # Call to allclose(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'A' (line 179)
        A_94483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 43), 'A', False)
        # Getting the type of 'B' (line 179)
        B_94484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 46), 'B', False)
        # Getting the type of 'eps' (line 179)
        eps_94485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 49), 'eps', False)
        # Processing the call keyword arguments (line 179)
        kwargs_94486 = {}
        # Getting the type of 'np' (line 179)
        np_94481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 31), 'np', False)
        # Obtaining the member 'allclose' of a type (line 179)
        allclose_94482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 31), np_94481, 'allclose')
        # Calling allclose(args, kwargs) (line 179)
        allclose_call_result_94487 = invoke(stypy.reporting.localization.Localization(__file__, 179, 31), allclose_94482, *[A_94483, B_94484, eps_94485], **kwargs_94486)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 28), tuple_94479, allclose_call_result_94487)
        
        # Applying the binary operator '%' (line 179)
        result_mod_94488 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 21), '%', fmt_94478, tuple_94479)
        
        # Processing the call keyword arguments (line 179)
        kwargs_94489 = {}
        # Getting the type of '_debug_print' (line 179)
        _debug_print_94477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 179)
        _debug_print_call_result_94490 = invoke(stypy.reporting.localization.Localization(__file__, 179, 8), _debug_print_94477, *[result_mod_94488], **kwargs_94489)
        
        
        # Call to assert_(...): (line 180)
        # Processing the call arguments (line 180)
        
        # Call to allclose(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'A' (line 180)
        A_94494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 28), 'A', False)
        # Getting the type of 'B' (line 180)
        B_94495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 31), 'B', False)
        # Getting the type of 'eps' (line 180)
        eps_94496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 34), 'eps', False)
        # Processing the call keyword arguments (line 180)
        kwargs_94497 = {}
        # Getting the type of 'np' (line 180)
        np_94492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 16), 'np', False)
        # Obtaining the member 'allclose' of a type (line 180)
        allclose_94493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 16), np_94492, 'allclose')
        # Calling allclose(args, kwargs) (line 180)
        allclose_call_result_94498 = invoke(stypy.reporting.localization.Localization(__file__, 180, 16), allclose_94493, *[A_94494, B_94495, eps_94496], **kwargs_94497)
        
        # Processing the call keyword arguments (line 180)
        kwargs_94499 = {}
        # Getting the type of 'assert_' (line 180)
        assert__94491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 180)
        assert__call_result_94500 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), assert__94491, *[allclose_call_result_94498], **kwargs_94499)
        
        
        # Call to _debug_print(...): (line 182)
        # Processing the call arguments (line 182)
        str_94502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 21), 'str', 'Calling iddr_asvd / idzr_asvd ...')
        # Processing the call keyword arguments (line 182)
        kwargs_94503 = {}
        # Getting the type of '_debug_print' (line 182)
        _debug_print_94501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 182)
        _debug_print_call_result_94504 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), _debug_print_94501, *[str_94502], **kwargs_94503)
        
        
        # Assigning a Call to a Name (line 183):
        
        # Assigning a Call to a Name (line 183):
        
        # Call to clock(...): (line 183)
        # Processing the call keyword arguments (line 183)
        kwargs_94507 = {}
        # Getting the type of 'time' (line 183)
        time_94505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 13), 'time', False)
        # Obtaining the member 'clock' of a type (line 183)
        clock_94506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 13), time_94505, 'clock')
        # Calling clock(args, kwargs) (line 183)
        clock_call_result_94508 = invoke(stypy.reporting.localization.Localization(__file__, 183, 13), clock_94506, *[], **kwargs_94507)
        
        # Assigning a type to the variable 't0' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 't0', clock_call_result_94508)
        
        # Assigning a Call to a Tuple (line 184):
        
        # Assigning a Subscript to a Name (line 184):
        
        # Obtaining the type of the subscript
        int_94509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 8), 'int')
        
        # Call to svd(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'A' (line 184)
        A_94512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 33), 'A', False)
        # Getting the type of 'k' (line 184)
        k_94513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 36), 'k', False)
        # Processing the call keyword arguments (line 184)
        kwargs_94514 = {}
        # Getting the type of 'pymatrixid' (line 184)
        pymatrixid_94510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 18), 'pymatrixid', False)
        # Obtaining the member 'svd' of a type (line 184)
        svd_94511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 18), pymatrixid_94510, 'svd')
        # Calling svd(args, kwargs) (line 184)
        svd_call_result_94515 = invoke(stypy.reporting.localization.Localization(__file__, 184, 18), svd_94511, *[A_94512, k_94513], **kwargs_94514)
        
        # Obtaining the member '__getitem__' of a type (line 184)
        getitem___94516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), svd_call_result_94515, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 184)
        subscript_call_result_94517 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), getitem___94516, int_94509)
        
        # Assigning a type to the variable 'tuple_var_assignment_93461' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'tuple_var_assignment_93461', subscript_call_result_94517)
        
        # Assigning a Subscript to a Name (line 184):
        
        # Obtaining the type of the subscript
        int_94518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 8), 'int')
        
        # Call to svd(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'A' (line 184)
        A_94521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 33), 'A', False)
        # Getting the type of 'k' (line 184)
        k_94522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 36), 'k', False)
        # Processing the call keyword arguments (line 184)
        kwargs_94523 = {}
        # Getting the type of 'pymatrixid' (line 184)
        pymatrixid_94519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 18), 'pymatrixid', False)
        # Obtaining the member 'svd' of a type (line 184)
        svd_94520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 18), pymatrixid_94519, 'svd')
        # Calling svd(args, kwargs) (line 184)
        svd_call_result_94524 = invoke(stypy.reporting.localization.Localization(__file__, 184, 18), svd_94520, *[A_94521, k_94522], **kwargs_94523)
        
        # Obtaining the member '__getitem__' of a type (line 184)
        getitem___94525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), svd_call_result_94524, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 184)
        subscript_call_result_94526 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), getitem___94525, int_94518)
        
        # Assigning a type to the variable 'tuple_var_assignment_93462' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'tuple_var_assignment_93462', subscript_call_result_94526)
        
        # Assigning a Subscript to a Name (line 184):
        
        # Obtaining the type of the subscript
        int_94527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 8), 'int')
        
        # Call to svd(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'A' (line 184)
        A_94530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 33), 'A', False)
        # Getting the type of 'k' (line 184)
        k_94531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 36), 'k', False)
        # Processing the call keyword arguments (line 184)
        kwargs_94532 = {}
        # Getting the type of 'pymatrixid' (line 184)
        pymatrixid_94528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 18), 'pymatrixid', False)
        # Obtaining the member 'svd' of a type (line 184)
        svd_94529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 18), pymatrixid_94528, 'svd')
        # Calling svd(args, kwargs) (line 184)
        svd_call_result_94533 = invoke(stypy.reporting.localization.Localization(__file__, 184, 18), svd_94529, *[A_94530, k_94531], **kwargs_94532)
        
        # Obtaining the member '__getitem__' of a type (line 184)
        getitem___94534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), svd_call_result_94533, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 184)
        subscript_call_result_94535 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), getitem___94534, int_94527)
        
        # Assigning a type to the variable 'tuple_var_assignment_93463' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'tuple_var_assignment_93463', subscript_call_result_94535)
        
        # Assigning a Name to a Name (line 184):
        # Getting the type of 'tuple_var_assignment_93461' (line 184)
        tuple_var_assignment_93461_94536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'tuple_var_assignment_93461')
        # Assigning a type to the variable 'U' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'U', tuple_var_assignment_93461_94536)
        
        # Assigning a Name to a Name (line 184):
        # Getting the type of 'tuple_var_assignment_93462' (line 184)
        tuple_var_assignment_93462_94537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'tuple_var_assignment_93462')
        # Assigning a type to the variable 'S' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 11), 'S', tuple_var_assignment_93462_94537)
        
        # Assigning a Name to a Name (line 184):
        # Getting the type of 'tuple_var_assignment_93463' (line 184)
        tuple_var_assignment_93463_94538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'tuple_var_assignment_93463')
        # Assigning a type to the variable 'V' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 14), 'V', tuple_var_assignment_93463_94538)
        
        # Assigning a BinOp to a Name (line 185):
        
        # Assigning a BinOp to a Name (line 185):
        
        # Call to clock(...): (line 185)
        # Processing the call keyword arguments (line 185)
        kwargs_94541 = {}
        # Getting the type of 'time' (line 185)
        time_94539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'time', False)
        # Obtaining the member 'clock' of a type (line 185)
        clock_94540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 12), time_94539, 'clock')
        # Calling clock(args, kwargs) (line 185)
        clock_call_result_94542 = invoke(stypy.reporting.localization.Localization(__file__, 185, 12), clock_94540, *[], **kwargs_94541)
        
        # Getting the type of 't0' (line 185)
        t0_94543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 27), 't0')
        # Applying the binary operator '-' (line 185)
        result_sub_94544 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 12), '-', clock_call_result_94542, t0_94543)
        
        # Assigning a type to the variable 't' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 't', result_sub_94544)
        
        # Assigning a Call to a Name (line 186):
        
        # Assigning a Call to a Name (line 186):
        
        # Call to dot(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'U' (line 186)
        U_94547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 19), 'U', False)
        
        # Call to dot(...): (line 186)
        # Processing the call arguments (line 186)
        
        # Call to diag(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'S' (line 186)
        S_94552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 37), 'S', False)
        # Processing the call keyword arguments (line 186)
        kwargs_94553 = {}
        # Getting the type of 'np' (line 186)
        np_94550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 29), 'np', False)
        # Obtaining the member 'diag' of a type (line 186)
        diag_94551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 29), np_94550, 'diag')
        # Calling diag(args, kwargs) (line 186)
        diag_call_result_94554 = invoke(stypy.reporting.localization.Localization(__file__, 186, 29), diag_94551, *[S_94552], **kwargs_94553)
        
        
        # Call to conj(...): (line 186)
        # Processing the call keyword arguments (line 186)
        kwargs_94558 = {}
        # Getting the type of 'V' (line 186)
        V_94555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 41), 'V', False)
        # Obtaining the member 'T' of a type (line 186)
        T_94556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 41), V_94555, 'T')
        # Obtaining the member 'conj' of a type (line 186)
        conj_94557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 41), T_94556, 'conj')
        # Calling conj(args, kwargs) (line 186)
        conj_call_result_94559 = invoke(stypy.reporting.localization.Localization(__file__, 186, 41), conj_94557, *[], **kwargs_94558)
        
        # Processing the call keyword arguments (line 186)
        kwargs_94560 = {}
        # Getting the type of 'np' (line 186)
        np_94548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 22), 'np', False)
        # Obtaining the member 'dot' of a type (line 186)
        dot_94549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 22), np_94548, 'dot')
        # Calling dot(args, kwargs) (line 186)
        dot_call_result_94561 = invoke(stypy.reporting.localization.Localization(__file__, 186, 22), dot_94549, *[diag_call_result_94554, conj_call_result_94559], **kwargs_94560)
        
        # Processing the call keyword arguments (line 186)
        kwargs_94562 = {}
        # Getting the type of 'np' (line 186)
        np_94545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'np', False)
        # Obtaining the member 'dot' of a type (line 186)
        dot_94546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 12), np_94545, 'dot')
        # Calling dot(args, kwargs) (line 186)
        dot_call_result_94563 = invoke(stypy.reporting.localization.Localization(__file__, 186, 12), dot_94546, *[U_94547, dot_call_result_94561], **kwargs_94562)
        
        # Assigning a type to the variable 'B' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'B', dot_call_result_94563)
        
        # Call to _debug_print(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'fmt' (line 187)
        fmt_94565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 21), 'fmt', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 187)
        tuple_94566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 187)
        # Adding element type (line 187)
        # Getting the type of 't' (line 187)
        t_94567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 28), 't', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 28), tuple_94566, t_94567)
        # Adding element type (line 187)
        
        # Call to allclose(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'A' (line 187)
        A_94570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 43), 'A', False)
        # Getting the type of 'B' (line 187)
        B_94571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 46), 'B', False)
        # Getting the type of 'eps' (line 187)
        eps_94572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 49), 'eps', False)
        # Processing the call keyword arguments (line 187)
        kwargs_94573 = {}
        # Getting the type of 'np' (line 187)
        np_94568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 31), 'np', False)
        # Obtaining the member 'allclose' of a type (line 187)
        allclose_94569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 31), np_94568, 'allclose')
        # Calling allclose(args, kwargs) (line 187)
        allclose_call_result_94574 = invoke(stypy.reporting.localization.Localization(__file__, 187, 31), allclose_94569, *[A_94570, B_94571, eps_94572], **kwargs_94573)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 28), tuple_94566, allclose_call_result_94574)
        
        # Applying the binary operator '%' (line 187)
        result_mod_94575 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 21), '%', fmt_94565, tuple_94566)
        
        # Processing the call keyword arguments (line 187)
        kwargs_94576 = {}
        # Getting the type of '_debug_print' (line 187)
        _debug_print_94564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 187)
        _debug_print_call_result_94577 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), _debug_print_94564, *[result_mod_94575], **kwargs_94576)
        
        
        # Call to assert_(...): (line 188)
        # Processing the call arguments (line 188)
        
        # Call to allclose(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'A' (line 188)
        A_94581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 28), 'A', False)
        # Getting the type of 'B' (line 188)
        B_94582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 31), 'B', False)
        # Getting the type of 'eps' (line 188)
        eps_94583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 34), 'eps', False)
        # Processing the call keyword arguments (line 188)
        kwargs_94584 = {}
        # Getting the type of 'np' (line 188)
        np_94579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 16), 'np', False)
        # Obtaining the member 'allclose' of a type (line 188)
        allclose_94580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 16), np_94579, 'allclose')
        # Calling allclose(args, kwargs) (line 188)
        allclose_call_result_94585 = invoke(stypy.reporting.localization.Localization(__file__, 188, 16), allclose_94580, *[A_94581, B_94582, eps_94583], **kwargs_94584)
        
        # Processing the call keyword arguments (line 188)
        kwargs_94586 = {}
        # Getting the type of 'assert_' (line 188)
        assert__94578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 188)
        assert__call_result_94587 = invoke(stypy.reporting.localization.Localization(__file__, 188, 8), assert__94578, *[allclose_call_result_94585], **kwargs_94586)
        
        
        # Call to _debug_print(...): (line 190)
        # Processing the call arguments (line 190)
        str_94589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 21), 'str', 'Calling iddr_rsvd / idzr_rsvd ...')
        # Processing the call keyword arguments (line 190)
        kwargs_94590 = {}
        # Getting the type of '_debug_print' (line 190)
        _debug_print_94588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 190)
        _debug_print_call_result_94591 = invoke(stypy.reporting.localization.Localization(__file__, 190, 8), _debug_print_94588, *[str_94589], **kwargs_94590)
        
        
        # Assigning a Call to a Name (line 191):
        
        # Assigning a Call to a Name (line 191):
        
        # Call to clock(...): (line 191)
        # Processing the call keyword arguments (line 191)
        kwargs_94594 = {}
        # Getting the type of 'time' (line 191)
        time_94592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 13), 'time', False)
        # Obtaining the member 'clock' of a type (line 191)
        clock_94593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 13), time_94592, 'clock')
        # Calling clock(args, kwargs) (line 191)
        clock_call_result_94595 = invoke(stypy.reporting.localization.Localization(__file__, 191, 13), clock_94593, *[], **kwargs_94594)
        
        # Assigning a type to the variable 't0' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 't0', clock_call_result_94595)
        
        # Assigning a Call to a Tuple (line 192):
        
        # Assigning a Subscript to a Name (line 192):
        
        # Obtaining the type of the subscript
        int_94596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 8), 'int')
        
        # Call to svd(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'L' (line 192)
        L_94599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 33), 'L', False)
        # Getting the type of 'k' (line 192)
        k_94600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 36), 'k', False)
        # Processing the call keyword arguments (line 192)
        kwargs_94601 = {}
        # Getting the type of 'pymatrixid' (line 192)
        pymatrixid_94597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 18), 'pymatrixid', False)
        # Obtaining the member 'svd' of a type (line 192)
        svd_94598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 18), pymatrixid_94597, 'svd')
        # Calling svd(args, kwargs) (line 192)
        svd_call_result_94602 = invoke(stypy.reporting.localization.Localization(__file__, 192, 18), svd_94598, *[L_94599, k_94600], **kwargs_94601)
        
        # Obtaining the member '__getitem__' of a type (line 192)
        getitem___94603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), svd_call_result_94602, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 192)
        subscript_call_result_94604 = invoke(stypy.reporting.localization.Localization(__file__, 192, 8), getitem___94603, int_94596)
        
        # Assigning a type to the variable 'tuple_var_assignment_93464' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'tuple_var_assignment_93464', subscript_call_result_94604)
        
        # Assigning a Subscript to a Name (line 192):
        
        # Obtaining the type of the subscript
        int_94605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 8), 'int')
        
        # Call to svd(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'L' (line 192)
        L_94608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 33), 'L', False)
        # Getting the type of 'k' (line 192)
        k_94609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 36), 'k', False)
        # Processing the call keyword arguments (line 192)
        kwargs_94610 = {}
        # Getting the type of 'pymatrixid' (line 192)
        pymatrixid_94606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 18), 'pymatrixid', False)
        # Obtaining the member 'svd' of a type (line 192)
        svd_94607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 18), pymatrixid_94606, 'svd')
        # Calling svd(args, kwargs) (line 192)
        svd_call_result_94611 = invoke(stypy.reporting.localization.Localization(__file__, 192, 18), svd_94607, *[L_94608, k_94609], **kwargs_94610)
        
        # Obtaining the member '__getitem__' of a type (line 192)
        getitem___94612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), svd_call_result_94611, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 192)
        subscript_call_result_94613 = invoke(stypy.reporting.localization.Localization(__file__, 192, 8), getitem___94612, int_94605)
        
        # Assigning a type to the variable 'tuple_var_assignment_93465' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'tuple_var_assignment_93465', subscript_call_result_94613)
        
        # Assigning a Subscript to a Name (line 192):
        
        # Obtaining the type of the subscript
        int_94614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 8), 'int')
        
        # Call to svd(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'L' (line 192)
        L_94617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 33), 'L', False)
        # Getting the type of 'k' (line 192)
        k_94618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 36), 'k', False)
        # Processing the call keyword arguments (line 192)
        kwargs_94619 = {}
        # Getting the type of 'pymatrixid' (line 192)
        pymatrixid_94615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 18), 'pymatrixid', False)
        # Obtaining the member 'svd' of a type (line 192)
        svd_94616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 18), pymatrixid_94615, 'svd')
        # Calling svd(args, kwargs) (line 192)
        svd_call_result_94620 = invoke(stypy.reporting.localization.Localization(__file__, 192, 18), svd_94616, *[L_94617, k_94618], **kwargs_94619)
        
        # Obtaining the member '__getitem__' of a type (line 192)
        getitem___94621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), svd_call_result_94620, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 192)
        subscript_call_result_94622 = invoke(stypy.reporting.localization.Localization(__file__, 192, 8), getitem___94621, int_94614)
        
        # Assigning a type to the variable 'tuple_var_assignment_93466' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'tuple_var_assignment_93466', subscript_call_result_94622)
        
        # Assigning a Name to a Name (line 192):
        # Getting the type of 'tuple_var_assignment_93464' (line 192)
        tuple_var_assignment_93464_94623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'tuple_var_assignment_93464')
        # Assigning a type to the variable 'U' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'U', tuple_var_assignment_93464_94623)
        
        # Assigning a Name to a Name (line 192):
        # Getting the type of 'tuple_var_assignment_93465' (line 192)
        tuple_var_assignment_93465_94624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'tuple_var_assignment_93465')
        # Assigning a type to the variable 'S' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 11), 'S', tuple_var_assignment_93465_94624)
        
        # Assigning a Name to a Name (line 192):
        # Getting the type of 'tuple_var_assignment_93466' (line 192)
        tuple_var_assignment_93466_94625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'tuple_var_assignment_93466')
        # Assigning a type to the variable 'V' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 14), 'V', tuple_var_assignment_93466_94625)
        
        # Assigning a BinOp to a Name (line 193):
        
        # Assigning a BinOp to a Name (line 193):
        
        # Call to clock(...): (line 193)
        # Processing the call keyword arguments (line 193)
        kwargs_94628 = {}
        # Getting the type of 'time' (line 193)
        time_94626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'time', False)
        # Obtaining the member 'clock' of a type (line 193)
        clock_94627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), time_94626, 'clock')
        # Calling clock(args, kwargs) (line 193)
        clock_call_result_94629 = invoke(stypy.reporting.localization.Localization(__file__, 193, 12), clock_94627, *[], **kwargs_94628)
        
        # Getting the type of 't0' (line 193)
        t0_94630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 27), 't0')
        # Applying the binary operator '-' (line 193)
        result_sub_94631 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 12), '-', clock_call_result_94629, t0_94630)
        
        # Assigning a type to the variable 't' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 't', result_sub_94631)
        
        # Assigning a Call to a Name (line 194):
        
        # Assigning a Call to a Name (line 194):
        
        # Call to dot(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'U' (line 194)
        U_94634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 19), 'U', False)
        
        # Call to dot(...): (line 194)
        # Processing the call arguments (line 194)
        
        # Call to diag(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'S' (line 194)
        S_94639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 37), 'S', False)
        # Processing the call keyword arguments (line 194)
        kwargs_94640 = {}
        # Getting the type of 'np' (line 194)
        np_94637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 29), 'np', False)
        # Obtaining the member 'diag' of a type (line 194)
        diag_94638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 29), np_94637, 'diag')
        # Calling diag(args, kwargs) (line 194)
        diag_call_result_94641 = invoke(stypy.reporting.localization.Localization(__file__, 194, 29), diag_94638, *[S_94639], **kwargs_94640)
        
        
        # Call to conj(...): (line 194)
        # Processing the call keyword arguments (line 194)
        kwargs_94645 = {}
        # Getting the type of 'V' (line 194)
        V_94642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 41), 'V', False)
        # Obtaining the member 'T' of a type (line 194)
        T_94643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 41), V_94642, 'T')
        # Obtaining the member 'conj' of a type (line 194)
        conj_94644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 41), T_94643, 'conj')
        # Calling conj(args, kwargs) (line 194)
        conj_call_result_94646 = invoke(stypy.reporting.localization.Localization(__file__, 194, 41), conj_94644, *[], **kwargs_94645)
        
        # Processing the call keyword arguments (line 194)
        kwargs_94647 = {}
        # Getting the type of 'np' (line 194)
        np_94635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 22), 'np', False)
        # Obtaining the member 'dot' of a type (line 194)
        dot_94636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 22), np_94635, 'dot')
        # Calling dot(args, kwargs) (line 194)
        dot_call_result_94648 = invoke(stypy.reporting.localization.Localization(__file__, 194, 22), dot_94636, *[diag_call_result_94641, conj_call_result_94646], **kwargs_94647)
        
        # Processing the call keyword arguments (line 194)
        kwargs_94649 = {}
        # Getting the type of 'np' (line 194)
        np_94632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'np', False)
        # Obtaining the member 'dot' of a type (line 194)
        dot_94633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 12), np_94632, 'dot')
        # Calling dot(args, kwargs) (line 194)
        dot_call_result_94650 = invoke(stypy.reporting.localization.Localization(__file__, 194, 12), dot_94633, *[U_94634, dot_call_result_94648], **kwargs_94649)
        
        # Assigning a type to the variable 'B' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'B', dot_call_result_94650)
        
        # Call to _debug_print(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'fmt' (line 195)
        fmt_94652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 21), 'fmt', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 195)
        tuple_94653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 195)
        # Adding element type (line 195)
        # Getting the type of 't' (line 195)
        t_94654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 28), 't', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 28), tuple_94653, t_94654)
        # Adding element type (line 195)
        
        # Call to allclose(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'A' (line 195)
        A_94657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 43), 'A', False)
        # Getting the type of 'B' (line 195)
        B_94658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 46), 'B', False)
        # Getting the type of 'eps' (line 195)
        eps_94659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 49), 'eps', False)
        # Processing the call keyword arguments (line 195)
        kwargs_94660 = {}
        # Getting the type of 'np' (line 195)
        np_94655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 31), 'np', False)
        # Obtaining the member 'allclose' of a type (line 195)
        allclose_94656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 31), np_94655, 'allclose')
        # Calling allclose(args, kwargs) (line 195)
        allclose_call_result_94661 = invoke(stypy.reporting.localization.Localization(__file__, 195, 31), allclose_94656, *[A_94657, B_94658, eps_94659], **kwargs_94660)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 28), tuple_94653, allclose_call_result_94661)
        
        # Applying the binary operator '%' (line 195)
        result_mod_94662 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 21), '%', fmt_94652, tuple_94653)
        
        # Processing the call keyword arguments (line 195)
        kwargs_94663 = {}
        # Getting the type of '_debug_print' (line 195)
        _debug_print_94651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), '_debug_print', False)
        # Calling _debug_print(args, kwargs) (line 195)
        _debug_print_call_result_94664 = invoke(stypy.reporting.localization.Localization(__file__, 195, 8), _debug_print_94651, *[result_mod_94662], **kwargs_94663)
        
        
        # Call to assert_(...): (line 196)
        # Processing the call arguments (line 196)
        
        # Call to allclose(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'A' (line 196)
        A_94668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 28), 'A', False)
        # Getting the type of 'B' (line 196)
        B_94669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 31), 'B', False)
        # Getting the type of 'eps' (line 196)
        eps_94670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 34), 'eps', False)
        # Processing the call keyword arguments (line 196)
        kwargs_94671 = {}
        # Getting the type of 'np' (line 196)
        np_94666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), 'np', False)
        # Obtaining the member 'allclose' of a type (line 196)
        allclose_94667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 16), np_94666, 'allclose')
        # Calling allclose(args, kwargs) (line 196)
        allclose_call_result_94672 = invoke(stypy.reporting.localization.Localization(__file__, 196, 16), allclose_94667, *[A_94668, B_94669, eps_94670], **kwargs_94671)
        
        # Processing the call keyword arguments (line 196)
        kwargs_94673 = {}
        # Getting the type of 'assert_' (line 196)
        assert__94665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 196)
        assert__call_result_94674 = invoke(stypy.reporting.localization.Localization(__file__, 196, 8), assert__94665, *[allclose_call_result_94672], **kwargs_94673)
        
        
        # Assigning a Call to a Tuple (line 199):
        
        # Assigning a Subscript to a Name (line 199):
        
        # Obtaining the type of the subscript
        int_94675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 8), 'int')
        
        # Call to interp_decomp(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'A' (line 199)
        A_94678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 45), 'A', False)
        # Getting the type of 'k' (line 199)
        k_94679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 48), 'k', False)
        # Processing the call keyword arguments (line 199)
        # Getting the type of 'False' (line 199)
        False_94680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 56), 'False', False)
        keyword_94681 = False_94680
        kwargs_94682 = {'rand': keyword_94681}
        # Getting the type of 'pymatrixid' (line 199)
        pymatrixid_94676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'pymatrixid', False)
        # Obtaining the member 'interp_decomp' of a type (line 199)
        interp_decomp_94677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 20), pymatrixid_94676, 'interp_decomp')
        # Calling interp_decomp(args, kwargs) (line 199)
        interp_decomp_call_result_94683 = invoke(stypy.reporting.localization.Localization(__file__, 199, 20), interp_decomp_94677, *[A_94678, k_94679], **kwargs_94682)
        
        # Obtaining the member '__getitem__' of a type (line 199)
        getitem___94684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 8), interp_decomp_call_result_94683, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 199)
        subscript_call_result_94685 = invoke(stypy.reporting.localization.Localization(__file__, 199, 8), getitem___94684, int_94675)
        
        # Assigning a type to the variable 'tuple_var_assignment_93467' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'tuple_var_assignment_93467', subscript_call_result_94685)
        
        # Assigning a Subscript to a Name (line 199):
        
        # Obtaining the type of the subscript
        int_94686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 8), 'int')
        
        # Call to interp_decomp(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'A' (line 199)
        A_94689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 45), 'A', False)
        # Getting the type of 'k' (line 199)
        k_94690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 48), 'k', False)
        # Processing the call keyword arguments (line 199)
        # Getting the type of 'False' (line 199)
        False_94691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 56), 'False', False)
        keyword_94692 = False_94691
        kwargs_94693 = {'rand': keyword_94692}
        # Getting the type of 'pymatrixid' (line 199)
        pymatrixid_94687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'pymatrixid', False)
        # Obtaining the member 'interp_decomp' of a type (line 199)
        interp_decomp_94688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 20), pymatrixid_94687, 'interp_decomp')
        # Calling interp_decomp(args, kwargs) (line 199)
        interp_decomp_call_result_94694 = invoke(stypy.reporting.localization.Localization(__file__, 199, 20), interp_decomp_94688, *[A_94689, k_94690], **kwargs_94693)
        
        # Obtaining the member '__getitem__' of a type (line 199)
        getitem___94695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 8), interp_decomp_call_result_94694, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 199)
        subscript_call_result_94696 = invoke(stypy.reporting.localization.Localization(__file__, 199, 8), getitem___94695, int_94686)
        
        # Assigning a type to the variable 'tuple_var_assignment_93468' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'tuple_var_assignment_93468', subscript_call_result_94696)
        
        # Assigning a Name to a Name (line 199):
        # Getting the type of 'tuple_var_assignment_93467' (line 199)
        tuple_var_assignment_93467_94697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'tuple_var_assignment_93467')
        # Assigning a type to the variable 'idx' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'idx', tuple_var_assignment_93467_94697)
        
        # Assigning a Name to a Name (line 199):
        # Getting the type of 'tuple_var_assignment_93468' (line 199)
        tuple_var_assignment_93468_94698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'tuple_var_assignment_93468')
        # Assigning a type to the variable 'proj' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 13), 'proj', tuple_var_assignment_93468_94698)
        
        # Assigning a Call to a Tuple (line 200):
        
        # Assigning a Subscript to a Name (line 200):
        
        # Obtaining the type of the subscript
        int_94699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 8), 'int')
        
        # Call to id_to_svd(...): (line 200)
        # Processing the call arguments (line 200)
        
        # Obtaining the type of the subscript
        slice_94702 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 200, 42), None, None, None)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 200)
        k_94703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 52), 'k', False)
        slice_94704 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 200, 47), None, k_94703, None)
        # Getting the type of 'idx' (line 200)
        idx_94705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 47), 'idx', False)
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___94706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 47), idx_94705, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_94707 = invoke(stypy.reporting.localization.Localization(__file__, 200, 47), getitem___94706, slice_94704)
        
        # Getting the type of 'A' (line 200)
        A_94708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 42), 'A', False)
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___94709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 42), A_94708, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_94710 = invoke(stypy.reporting.localization.Localization(__file__, 200, 42), getitem___94709, (slice_94702, subscript_call_result_94707))
        
        # Getting the type of 'idx' (line 200)
        idx_94711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 57), 'idx', False)
        # Getting the type of 'proj' (line 200)
        proj_94712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 62), 'proj', False)
        # Processing the call keyword arguments (line 200)
        kwargs_94713 = {}
        # Getting the type of 'pymatrixid' (line 200)
        pymatrixid_94700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 21), 'pymatrixid', False)
        # Obtaining the member 'id_to_svd' of a type (line 200)
        id_to_svd_94701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 21), pymatrixid_94700, 'id_to_svd')
        # Calling id_to_svd(args, kwargs) (line 200)
        id_to_svd_call_result_94714 = invoke(stypy.reporting.localization.Localization(__file__, 200, 21), id_to_svd_94701, *[subscript_call_result_94710, idx_94711, proj_94712], **kwargs_94713)
        
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___94715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), id_to_svd_call_result_94714, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_94716 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), getitem___94715, int_94699)
        
        # Assigning a type to the variable 'tuple_var_assignment_93469' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_93469', subscript_call_result_94716)
        
        # Assigning a Subscript to a Name (line 200):
        
        # Obtaining the type of the subscript
        int_94717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 8), 'int')
        
        # Call to id_to_svd(...): (line 200)
        # Processing the call arguments (line 200)
        
        # Obtaining the type of the subscript
        slice_94720 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 200, 42), None, None, None)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 200)
        k_94721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 52), 'k', False)
        slice_94722 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 200, 47), None, k_94721, None)
        # Getting the type of 'idx' (line 200)
        idx_94723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 47), 'idx', False)
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___94724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 47), idx_94723, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_94725 = invoke(stypy.reporting.localization.Localization(__file__, 200, 47), getitem___94724, slice_94722)
        
        # Getting the type of 'A' (line 200)
        A_94726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 42), 'A', False)
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___94727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 42), A_94726, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_94728 = invoke(stypy.reporting.localization.Localization(__file__, 200, 42), getitem___94727, (slice_94720, subscript_call_result_94725))
        
        # Getting the type of 'idx' (line 200)
        idx_94729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 57), 'idx', False)
        # Getting the type of 'proj' (line 200)
        proj_94730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 62), 'proj', False)
        # Processing the call keyword arguments (line 200)
        kwargs_94731 = {}
        # Getting the type of 'pymatrixid' (line 200)
        pymatrixid_94718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 21), 'pymatrixid', False)
        # Obtaining the member 'id_to_svd' of a type (line 200)
        id_to_svd_94719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 21), pymatrixid_94718, 'id_to_svd')
        # Calling id_to_svd(args, kwargs) (line 200)
        id_to_svd_call_result_94732 = invoke(stypy.reporting.localization.Localization(__file__, 200, 21), id_to_svd_94719, *[subscript_call_result_94728, idx_94729, proj_94730], **kwargs_94731)
        
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___94733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), id_to_svd_call_result_94732, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_94734 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), getitem___94733, int_94717)
        
        # Assigning a type to the variable 'tuple_var_assignment_93470' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_93470', subscript_call_result_94734)
        
        # Assigning a Subscript to a Name (line 200):
        
        # Obtaining the type of the subscript
        int_94735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 8), 'int')
        
        # Call to id_to_svd(...): (line 200)
        # Processing the call arguments (line 200)
        
        # Obtaining the type of the subscript
        slice_94738 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 200, 42), None, None, None)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 200)
        k_94739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 52), 'k', False)
        slice_94740 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 200, 47), None, k_94739, None)
        # Getting the type of 'idx' (line 200)
        idx_94741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 47), 'idx', False)
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___94742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 47), idx_94741, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_94743 = invoke(stypy.reporting.localization.Localization(__file__, 200, 47), getitem___94742, slice_94740)
        
        # Getting the type of 'A' (line 200)
        A_94744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 42), 'A', False)
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___94745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 42), A_94744, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_94746 = invoke(stypy.reporting.localization.Localization(__file__, 200, 42), getitem___94745, (slice_94738, subscript_call_result_94743))
        
        # Getting the type of 'idx' (line 200)
        idx_94747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 57), 'idx', False)
        # Getting the type of 'proj' (line 200)
        proj_94748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 62), 'proj', False)
        # Processing the call keyword arguments (line 200)
        kwargs_94749 = {}
        # Getting the type of 'pymatrixid' (line 200)
        pymatrixid_94736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 21), 'pymatrixid', False)
        # Obtaining the member 'id_to_svd' of a type (line 200)
        id_to_svd_94737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 21), pymatrixid_94736, 'id_to_svd')
        # Calling id_to_svd(args, kwargs) (line 200)
        id_to_svd_call_result_94750 = invoke(stypy.reporting.localization.Localization(__file__, 200, 21), id_to_svd_94737, *[subscript_call_result_94746, idx_94747, proj_94748], **kwargs_94749)
        
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___94751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), id_to_svd_call_result_94750, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_94752 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), getitem___94751, int_94735)
        
        # Assigning a type to the variable 'tuple_var_assignment_93471' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_93471', subscript_call_result_94752)
        
        # Assigning a Name to a Name (line 200):
        # Getting the type of 'tuple_var_assignment_93469' (line 200)
        tuple_var_assignment_93469_94753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_93469')
        # Assigning a type to the variable 'Up' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'Up', tuple_var_assignment_93469_94753)
        
        # Assigning a Name to a Name (line 200):
        # Getting the type of 'tuple_var_assignment_93470' (line 200)
        tuple_var_assignment_93470_94754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_93470')
        # Assigning a type to the variable 'Sp' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'Sp', tuple_var_assignment_93470_94754)
        
        # Assigning a Name to a Name (line 200):
        # Getting the type of 'tuple_var_assignment_93471' (line 200)
        tuple_var_assignment_93471_94755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_93471')
        # Assigning a type to the variable 'Vp' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 16), 'Vp', tuple_var_assignment_93471_94755)
        
        # Assigning a Call to a Name (line 201):
        
        # Assigning a Call to a Name (line 201):
        
        # Call to dot(...): (line 201)
        # Processing the call arguments (line 201)
        
        # Call to dot(...): (line 201)
        # Processing the call arguments (line 201)
        
        # Call to conj(...): (line 201)
        # Processing the call keyword arguments (line 201)
        kwargs_94767 = {}
        # Getting the type of 'V' (line 201)
        V_94764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 33), 'V', False)
        # Obtaining the member 'T' of a type (line 201)
        T_94765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 33), V_94764, 'T')
        # Obtaining the member 'conj' of a type (line 201)
        conj_94766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 33), T_94765, 'conj')
        # Calling conj(args, kwargs) (line 201)
        conj_call_result_94768 = invoke(stypy.reporting.localization.Localization(__file__, 201, 33), conj_94766, *[], **kwargs_94767)
        
        # Processing the call keyword arguments (line 201)
        kwargs_94769 = {}
        
        # Call to diag(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'S' (line 201)
        S_94760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 26), 'S', False)
        # Processing the call keyword arguments (line 201)
        kwargs_94761 = {}
        # Getting the type of 'np' (line 201)
        np_94758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 18), 'np', False)
        # Obtaining the member 'diag' of a type (line 201)
        diag_94759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 18), np_94758, 'diag')
        # Calling diag(args, kwargs) (line 201)
        diag_call_result_94762 = invoke(stypy.reporting.localization.Localization(__file__, 201, 18), diag_94759, *[S_94760], **kwargs_94761)
        
        # Obtaining the member 'dot' of a type (line 201)
        dot_94763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 18), diag_call_result_94762, 'dot')
        # Calling dot(args, kwargs) (line 201)
        dot_call_result_94770 = invoke(stypy.reporting.localization.Localization(__file__, 201, 18), dot_94763, *[conj_call_result_94768], **kwargs_94769)
        
        # Processing the call keyword arguments (line 201)
        kwargs_94771 = {}
        # Getting the type of 'U' (line 201)
        U_94756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'U', False)
        # Obtaining the member 'dot' of a type (line 201)
        dot_94757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 12), U_94756, 'dot')
        # Calling dot(args, kwargs) (line 201)
        dot_call_result_94772 = invoke(stypy.reporting.localization.Localization(__file__, 201, 12), dot_94757, *[dot_call_result_94770], **kwargs_94771)
        
        # Assigning a type to the variable 'B' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'B', dot_call_result_94772)
        
        # Call to assert_(...): (line 202)
        # Processing the call arguments (line 202)
        
        # Call to allclose(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'A' (line 202)
        A_94776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 28), 'A', False)
        # Getting the type of 'B' (line 202)
        B_94777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 31), 'B', False)
        # Getting the type of 'eps' (line 202)
        eps_94778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 34), 'eps', False)
        # Processing the call keyword arguments (line 202)
        kwargs_94779 = {}
        # Getting the type of 'np' (line 202)
        np_94774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'np', False)
        # Obtaining the member 'allclose' of a type (line 202)
        allclose_94775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 16), np_94774, 'allclose')
        # Calling allclose(args, kwargs) (line 202)
        allclose_call_result_94780 = invoke(stypy.reporting.localization.Localization(__file__, 202, 16), allclose_94775, *[A_94776, B_94777, eps_94778], **kwargs_94779)
        
        # Processing the call keyword arguments (line 202)
        kwargs_94781 = {}
        # Getting the type of 'assert_' (line 202)
        assert__94773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 202)
        assert__call_result_94782 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), assert__94773, *[allclose_call_result_94780], **kwargs_94781)
        
        
        # Assigning a Call to a Name (line 205):
        
        # Assigning a Call to a Name (line 205):
        
        # Call to svdvals(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'A' (line 205)
        A_94784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'A', False)
        # Processing the call keyword arguments (line 205)
        kwargs_94785 = {}
        # Getting the type of 'svdvals' (line 205)
        svdvals_94783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'svdvals', False)
        # Calling svdvals(args, kwargs) (line 205)
        svdvals_call_result_94786 = invoke(stypy.reporting.localization.Localization(__file__, 205, 12), svdvals_94783, *[A_94784], **kwargs_94785)
        
        # Assigning a type to the variable 's' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 's', svdvals_call_result_94786)
        
        # Assigning a Call to a Name (line 206):
        
        # Assigning a Call to a Name (line 206):
        
        # Call to estimate_spectral_norm(...): (line 206)
        # Processing the call arguments (line 206)
        # Getting the type of 'A' (line 206)
        A_94789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 55), 'A', False)
        # Processing the call keyword arguments (line 206)
        kwargs_94790 = {}
        # Getting the type of 'pymatrixid' (line 206)
        pymatrixid_94787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 21), 'pymatrixid', False)
        # Obtaining the member 'estimate_spectral_norm' of a type (line 206)
        estimate_spectral_norm_94788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 21), pymatrixid_94787, 'estimate_spectral_norm')
        # Calling estimate_spectral_norm(args, kwargs) (line 206)
        estimate_spectral_norm_call_result_94791 = invoke(stypy.reporting.localization.Localization(__file__, 206, 21), estimate_spectral_norm_94788, *[A_94789], **kwargs_94790)
        
        # Assigning a type to the variable 'norm_2_est' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'norm_2_est', estimate_spectral_norm_call_result_94791)
        
        # Call to assert_(...): (line 207)
        # Processing the call arguments (line 207)
        
        # Call to allclose(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'norm_2_est' (line 207)
        norm_2_est_94795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 28), 'norm_2_est', False)
        
        # Obtaining the type of the subscript
        int_94796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 42), 'int')
        # Getting the type of 's' (line 207)
        s_94797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 40), 's', False)
        # Obtaining the member '__getitem__' of a type (line 207)
        getitem___94798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 40), s_94797, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 207)
        subscript_call_result_94799 = invoke(stypy.reporting.localization.Localization(__file__, 207, 40), getitem___94798, int_94796)
        
        float_94800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 46), 'float')
        # Processing the call keyword arguments (line 207)
        kwargs_94801 = {}
        # Getting the type of 'np' (line 207)
        np_94793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'np', False)
        # Obtaining the member 'allclose' of a type (line 207)
        allclose_94794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 16), np_94793, 'allclose')
        # Calling allclose(args, kwargs) (line 207)
        allclose_call_result_94802 = invoke(stypy.reporting.localization.Localization(__file__, 207, 16), allclose_94794, *[norm_2_est_94795, subscript_call_result_94799, float_94800], **kwargs_94801)
        
        # Processing the call keyword arguments (line 207)
        kwargs_94803 = {}
        # Getting the type of 'assert_' (line 207)
        assert__94792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 207)
        assert__call_result_94804 = invoke(stypy.reporting.localization.Localization(__file__, 207, 8), assert__94792, *[allclose_call_result_94802], **kwargs_94803)
        
        
        # Assigning a Call to a Name (line 209):
        
        # Assigning a Call to a Name (line 209):
        
        # Call to copy(...): (line 209)
        # Processing the call keyword arguments (line 209)
        kwargs_94807 = {}
        # Getting the type of 'A' (line 209)
        A_94805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'A', False)
        # Obtaining the member 'copy' of a type (line 209)
        copy_94806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 12), A_94805, 'copy')
        # Calling copy(args, kwargs) (line 209)
        copy_call_result_94808 = invoke(stypy.reporting.localization.Localization(__file__, 209, 12), copy_94806, *[], **kwargs_94807)
        
        # Assigning a type to the variable 'B' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'B', copy_call_result_94808)
        
        # Getting the type of 'B' (line 210)
        B_94809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'B')
        
        # Obtaining the type of the subscript
        slice_94810 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 210, 8), None, None, None)
        int_94811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 12), 'int')
        # Getting the type of 'B' (line 210)
        B_94812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'B')
        # Obtaining the member '__getitem__' of a type (line 210)
        getitem___94813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 8), B_94812, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 210)
        subscript_call_result_94814 = invoke(stypy.reporting.localization.Localization(__file__, 210, 8), getitem___94813, (slice_94810, int_94811))
        
        float_94815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 18), 'float')
        # Applying the binary operator '*=' (line 210)
        result_imul_94816 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 8), '*=', subscript_call_result_94814, float_94815)
        # Getting the type of 'B' (line 210)
        B_94817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'B')
        slice_94818 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 210, 8), None, None, None)
        int_94819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 12), 'int')
        # Storing an element on a container (line 210)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 8), B_94817, ((slice_94818, int_94819), result_imul_94816))
        
        
        # Assigning a Call to a Name (line 211):
        
        # Assigning a Call to a Name (line 211):
        
        # Call to svdvals(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'A' (line 211)
        A_94821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 20), 'A', False)
        # Getting the type of 'B' (line 211)
        B_94822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 24), 'B', False)
        # Applying the binary operator '-' (line 211)
        result_sub_94823 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 20), '-', A_94821, B_94822)
        
        # Processing the call keyword arguments (line 211)
        kwargs_94824 = {}
        # Getting the type of 'svdvals' (line 211)
        svdvals_94820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'svdvals', False)
        # Calling svdvals(args, kwargs) (line 211)
        svdvals_call_result_94825 = invoke(stypy.reporting.localization.Localization(__file__, 211, 12), svdvals_94820, *[result_sub_94823], **kwargs_94824)
        
        # Assigning a type to the variable 's' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 's', svdvals_call_result_94825)
        
        # Assigning a Call to a Name (line 212):
        
        # Assigning a Call to a Name (line 212):
        
        # Call to estimate_spectral_norm_diff(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'A' (line 212)
        A_94828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 60), 'A', False)
        # Getting the type of 'B' (line 212)
        B_94829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 63), 'B', False)
        # Processing the call keyword arguments (line 212)
        kwargs_94830 = {}
        # Getting the type of 'pymatrixid' (line 212)
        pymatrixid_94826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 21), 'pymatrixid', False)
        # Obtaining the member 'estimate_spectral_norm_diff' of a type (line 212)
        estimate_spectral_norm_diff_94827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 21), pymatrixid_94826, 'estimate_spectral_norm_diff')
        # Calling estimate_spectral_norm_diff(args, kwargs) (line 212)
        estimate_spectral_norm_diff_call_result_94831 = invoke(stypy.reporting.localization.Localization(__file__, 212, 21), estimate_spectral_norm_diff_94827, *[A_94828, B_94829], **kwargs_94830)
        
        # Assigning a type to the variable 'norm_2_est' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'norm_2_est', estimate_spectral_norm_diff_call_result_94831)
        
        # Call to assert_(...): (line 213)
        # Processing the call arguments (line 213)
        
        # Call to allclose(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'norm_2_est' (line 213)
        norm_2_est_94835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 28), 'norm_2_est', False)
        
        # Obtaining the type of the subscript
        int_94836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 42), 'int')
        # Getting the type of 's' (line 213)
        s_94837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 40), 's', False)
        # Obtaining the member '__getitem__' of a type (line 213)
        getitem___94838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 40), s_94837, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 213)
        subscript_call_result_94839 = invoke(stypy.reporting.localization.Localization(__file__, 213, 40), getitem___94838, int_94836)
        
        float_94840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 46), 'float')
        # Processing the call keyword arguments (line 213)
        kwargs_94841 = {}
        # Getting the type of 'np' (line 213)
        np_94833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'np', False)
        # Obtaining the member 'allclose' of a type (line 213)
        allclose_94834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 16), np_94833, 'allclose')
        # Calling allclose(args, kwargs) (line 213)
        allclose_call_result_94842 = invoke(stypy.reporting.localization.Localization(__file__, 213, 16), allclose_94834, *[norm_2_est_94835, subscript_call_result_94839, float_94840], **kwargs_94841)
        
        # Processing the call keyword arguments (line 213)
        kwargs_94843 = {}
        # Getting the type of 'assert_' (line 213)
        assert__94832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 213)
        assert__call_result_94844 = invoke(stypy.reporting.localization.Localization(__file__, 213, 8), assert__94832, *[allclose_call_result_94842], **kwargs_94843)
        
        
        # Assigning a Call to a Name (line 216):
        
        # Assigning a Call to a Name (line 216):
        
        # Call to array(...): (line 216)
        # Processing the call arguments (line 216)
        
        # Obtaining an instance of the builtin type 'list' (line 216)
        list_94847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 216)
        # Adding element type (line 216)
        
        # Obtaining an instance of the builtin type 'list' (line 216)
        list_94848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 216)
        # Adding element type (line 216)
        int_94849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 22), list_94848, int_94849)
        # Adding element type (line 216)
        int_94850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 22), list_94848, int_94850)
        # Adding element type (line 216)
        int_94851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 22), list_94848, int_94851)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 21), list_94847, list_94848)
        # Adding element type (line 216)
        
        # Obtaining an instance of the builtin type 'list' (line 216)
        list_94852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 216)
        # Adding element type (line 216)
        int_94853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 33), list_94852, int_94853)
        # Adding element type (line 216)
        int_94854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 33), list_94852, int_94854)
        # Adding element type (line 216)
        int_94855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 33), list_94852, int_94855)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 21), list_94847, list_94852)
        # Adding element type (line 216)
        
        # Obtaining an instance of the builtin type 'list' (line 216)
        list_94856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 216)
        # Adding element type (line 216)
        int_94857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 44), list_94856, int_94857)
        # Adding element type (line 216)
        int_94858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 44), list_94856, int_94858)
        # Adding element type (line 216)
        int_94859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 44), list_94856, int_94859)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 21), list_94847, list_94856)
        
        # Processing the call keyword arguments (line 216)
        # Getting the type of 'dtype' (line 216)
        dtype_94860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 62), 'dtype', False)
        keyword_94861 = dtype_94860
        kwargs_94862 = {'dtype': keyword_94861}
        # Getting the type of 'np' (line 216)
        np_94845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 216)
        array_94846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), np_94845, 'array')
        # Calling array(args, kwargs) (line 216)
        array_call_result_94863 = invoke(stypy.reporting.localization.Localization(__file__, 216, 12), array_94846, *[list_94847], **kwargs_94862)
        
        # Assigning a type to the variable 'B' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'B', array_call_result_94863)
        
        
        # Obtaining an instance of the builtin type 'list' (line 217)
        list_94864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 217)
        # Adding element type (line 217)
        # Getting the type of 'A' (line 217)
        A_94865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 18), 'A')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 17), list_94864, A_94865)
        # Adding element type (line 217)
        # Getting the type of 'B' (line 217)
        B_94866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 21), 'B')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 17), list_94864, B_94866)
        
        # Testing the type of a for loop iterable (line 217)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 217, 8), list_94864)
        # Getting the type of the for loop variable (line 217)
        for_loop_var_94867 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 217, 8), list_94864)
        # Assigning a type to the variable 'M' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'M', for_loop_var_94867)
        # SSA begins for a for statement (line 217)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 218):
        
        # Assigning a Call to a Name (line 218):
        
        # Call to aslinearoperator(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'M' (line 218)
        M_94869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 34), 'M', False)
        # Processing the call keyword arguments (line 218)
        kwargs_94870 = {}
        # Getting the type of 'aslinearoperator' (line 218)
        aslinearoperator_94868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 17), 'aslinearoperator', False)
        # Calling aslinearoperator(args, kwargs) (line 218)
        aslinearoperator_call_result_94871 = invoke(stypy.reporting.localization.Localization(__file__, 218, 17), aslinearoperator_94868, *[M_94869], **kwargs_94870)
        
        # Assigning a type to the variable 'ML' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'ML', aslinearoperator_call_result_94871)
        
        # Assigning a Num to a Name (line 220):
        
        # Assigning a Num to a Name (line 220):
        float_94872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 23), 'float')
        # Assigning a type to the variable 'rank_tol' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'rank_tol', float_94872)
        
        # Assigning a Call to a Name (line 221):
        
        # Assigning a Call to a Name (line 221):
        
        # Call to matrix_rank(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'M' (line 221)
        M_94876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 44), 'M', False)
        
        # Call to norm(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'M' (line 221)
        M_94878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 52), 'M', False)
        int_94879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 55), 'int')
        # Processing the call keyword arguments (line 221)
        kwargs_94880 = {}
        # Getting the type of 'norm' (line 221)
        norm_94877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 47), 'norm', False)
        # Calling norm(args, kwargs) (line 221)
        norm_call_result_94881 = invoke(stypy.reporting.localization.Localization(__file__, 221, 47), norm_94877, *[M_94878, int_94879], **kwargs_94880)
        
        # Getting the type of 'rank_tol' (line 221)
        rank_tol_94882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 58), 'rank_tol', False)
        # Applying the binary operator '*' (line 221)
        result_mul_94883 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 47), '*', norm_call_result_94881, rank_tol_94882)
        
        # Processing the call keyword arguments (line 221)
        kwargs_94884 = {}
        # Getting the type of 'np' (line 221)
        np_94873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 22), 'np', False)
        # Obtaining the member 'linalg' of a type (line 221)
        linalg_94874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 22), np_94873, 'linalg')
        # Obtaining the member 'matrix_rank' of a type (line 221)
        matrix_rank_94875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 22), linalg_94874, 'matrix_rank')
        # Calling matrix_rank(args, kwargs) (line 221)
        matrix_rank_call_result_94885 = invoke(stypy.reporting.localization.Localization(__file__, 221, 22), matrix_rank_94875, *[M_94876, result_mul_94883], **kwargs_94884)
        
        # Assigning a type to the variable 'rank_np' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'rank_np', matrix_rank_call_result_94885)
        
        # Assigning a Call to a Name (line 222):
        
        # Assigning a Call to a Name (line 222):
        
        # Call to estimate_rank(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'M' (line 222)
        M_94888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 48), 'M', False)
        # Getting the type of 'rank_tol' (line 222)
        rank_tol_94889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 51), 'rank_tol', False)
        # Processing the call keyword arguments (line 222)
        kwargs_94890 = {}
        # Getting the type of 'pymatrixid' (line 222)
        pymatrixid_94886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 23), 'pymatrixid', False)
        # Obtaining the member 'estimate_rank' of a type (line 222)
        estimate_rank_94887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 23), pymatrixid_94886, 'estimate_rank')
        # Calling estimate_rank(args, kwargs) (line 222)
        estimate_rank_call_result_94891 = invoke(stypy.reporting.localization.Localization(__file__, 222, 23), estimate_rank_94887, *[M_94888, rank_tol_94889], **kwargs_94890)
        
        # Assigning a type to the variable 'rank_est' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'rank_est', estimate_rank_call_result_94891)
        
        # Assigning a Call to a Name (line 223):
        
        # Assigning a Call to a Name (line 223):
        
        # Call to estimate_rank(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'ML' (line 223)
        ML_94894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 50), 'ML', False)
        # Getting the type of 'rank_tol' (line 223)
        rank_tol_94895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 54), 'rank_tol', False)
        # Processing the call keyword arguments (line 223)
        kwargs_94896 = {}
        # Getting the type of 'pymatrixid' (line 223)
        pymatrixid_94892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 25), 'pymatrixid', False)
        # Obtaining the member 'estimate_rank' of a type (line 223)
        estimate_rank_94893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 25), pymatrixid_94892, 'estimate_rank')
        # Calling estimate_rank(args, kwargs) (line 223)
        estimate_rank_call_result_94897 = invoke(stypy.reporting.localization.Localization(__file__, 223, 25), estimate_rank_94893, *[ML_94894, rank_tol_94895], **kwargs_94896)
        
        # Assigning a type to the variable 'rank_est_2' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'rank_est_2', estimate_rank_call_result_94897)
        
        # Call to assert_(...): (line 225)
        # Processing the call arguments (line 225)
        
        # Getting the type of 'rank_est' (line 225)
        rank_est_94899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 20), 'rank_est', False)
        # Getting the type of 'rank_np' (line 225)
        rank_np_94900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 32), 'rank_np', False)
        # Applying the binary operator '>=' (line 225)
        result_ge_94901 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 20), '>=', rank_est_94899, rank_np_94900)
        
        # Processing the call keyword arguments (line 225)
        kwargs_94902 = {}
        # Getting the type of 'assert_' (line 225)
        assert__94898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 225)
        assert__call_result_94903 = invoke(stypy.reporting.localization.Localization(__file__, 225, 12), assert__94898, *[result_ge_94901], **kwargs_94902)
        
        
        # Call to assert_(...): (line 226)
        # Processing the call arguments (line 226)
        
        # Getting the type of 'rank_est' (line 226)
        rank_est_94905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 20), 'rank_est', False)
        # Getting the type of 'rank_np' (line 226)
        rank_np_94906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 32), 'rank_np', False)
        int_94907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 42), 'int')
        # Applying the binary operator '+' (line 226)
        result_add_94908 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 32), '+', rank_np_94906, int_94907)
        
        # Applying the binary operator '<=' (line 226)
        result_le_94909 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 20), '<=', rank_est_94905, result_add_94908)
        
        # Processing the call keyword arguments (line 226)
        kwargs_94910 = {}
        # Getting the type of 'assert_' (line 226)
        assert__94904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 226)
        assert__call_result_94911 = invoke(stypy.reporting.localization.Localization(__file__, 226, 12), assert__94904, *[result_le_94909], **kwargs_94910)
        
        
        # Call to assert_(...): (line 228)
        # Processing the call arguments (line 228)
        
        # Getting the type of 'rank_est_2' (line 228)
        rank_est_2_94913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 20), 'rank_est_2', False)
        # Getting the type of 'rank_np' (line 228)
        rank_np_94914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 34), 'rank_np', False)
        int_94915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 44), 'int')
        # Applying the binary operator '-' (line 228)
        result_sub_94916 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 34), '-', rank_np_94914, int_94915)
        
        # Applying the binary operator '>=' (line 228)
        result_ge_94917 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 20), '>=', rank_est_2_94913, result_sub_94916)
        
        # Processing the call keyword arguments (line 228)
        kwargs_94918 = {}
        # Getting the type of 'assert_' (line 228)
        assert__94912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 228)
        assert__call_result_94919 = invoke(stypy.reporting.localization.Localization(__file__, 228, 12), assert__94912, *[result_ge_94917], **kwargs_94918)
        
        
        # Call to assert_(...): (line 229)
        # Processing the call arguments (line 229)
        
        # Getting the type of 'rank_est_2' (line 229)
        rank_est_2_94921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 20), 'rank_est_2', False)
        # Getting the type of 'rank_np' (line 229)
        rank_np_94922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 34), 'rank_np', False)
        int_94923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 44), 'int')
        # Applying the binary operator '+' (line 229)
        result_add_94924 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 34), '+', rank_np_94922, int_94923)
        
        # Applying the binary operator '<=' (line 229)
        result_le_94925 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 20), '<=', rank_est_2_94921, result_add_94924)
        
        # Processing the call keyword arguments (line 229)
        kwargs_94926 = {}
        # Getting the type of 'assert_' (line 229)
        assert__94920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 229)
        assert__call_result_94927 = invoke(stypy.reporting.localization.Localization(__file__, 229, 12), assert__94920, *[result_le_94925], **kwargs_94926)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'check_id(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_id' in the type store
        # Getting the type of 'stypy_return_type' (line 49)
        stypy_return_type_94928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_94928)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_id'
        return stypy_return_type_94928


    @norecursion
    def test_rand(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_rand'
        module_type_store = module_type_store.open_function_context('test_rand', 231, 4, False)
        # Assigning a type to the variable 'self' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestInterpolativeDecomposition.test_rand.__dict__.__setitem__('stypy_localization', localization)
        TestInterpolativeDecomposition.test_rand.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestInterpolativeDecomposition.test_rand.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestInterpolativeDecomposition.test_rand.__dict__.__setitem__('stypy_function_name', 'TestInterpolativeDecomposition.test_rand')
        TestInterpolativeDecomposition.test_rand.__dict__.__setitem__('stypy_param_names_list', [])
        TestInterpolativeDecomposition.test_rand.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestInterpolativeDecomposition.test_rand.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestInterpolativeDecomposition.test_rand.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestInterpolativeDecomposition.test_rand.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestInterpolativeDecomposition.test_rand.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestInterpolativeDecomposition.test_rand.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInterpolativeDecomposition.test_rand', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_rand', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_rand(...)' code ##################

        
        # Call to seed(...): (line 232)
        # Processing the call arguments (line 232)
        str_94931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 24), 'str', 'default')
        # Processing the call keyword arguments (line 232)
        kwargs_94932 = {}
        # Getting the type of 'pymatrixid' (line 232)
        pymatrixid_94929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'pymatrixid', False)
        # Obtaining the member 'seed' of a type (line 232)
        seed_94930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 8), pymatrixid_94929, 'seed')
        # Calling seed(args, kwargs) (line 232)
        seed_call_result_94933 = invoke(stypy.reporting.localization.Localization(__file__, 232, 8), seed_94930, *[str_94931], **kwargs_94932)
        
        
        # Call to assert_(...): (line 233)
        # Processing the call arguments (line 233)
        
        # Call to allclose(...): (line 233)
        # Processing the call arguments (line 233)
        
        # Call to rand(...): (line 233)
        # Processing the call arguments (line 233)
        int_94939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 44), 'int')
        # Processing the call keyword arguments (line 233)
        kwargs_94940 = {}
        # Getting the type of 'pymatrixid' (line 233)
        pymatrixid_94937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 28), 'pymatrixid', False)
        # Obtaining the member 'rand' of a type (line 233)
        rand_94938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 28), pymatrixid_94937, 'rand')
        # Calling rand(args, kwargs) (line 233)
        rand_call_result_94941 = invoke(stypy.reporting.localization.Localization(__file__, 233, 28), rand_94938, *[int_94939], **kwargs_94940)
        
        
        # Obtaining an instance of the builtin type 'list' (line 233)
        list_94942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 233)
        # Adding element type (line 233)
        float_94943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 48), list_94942, float_94943)
        # Adding element type (line 233)
        float_94944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 48), list_94942, float_94944)
        
        float_94945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 73), 'float')
        # Processing the call keyword arguments (line 233)
        kwargs_94946 = {}
        # Getting the type of 'np' (line 233)
        np_94935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 16), 'np', False)
        # Obtaining the member 'allclose' of a type (line 233)
        allclose_94936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 16), np_94935, 'allclose')
        # Calling allclose(args, kwargs) (line 233)
        allclose_call_result_94947 = invoke(stypy.reporting.localization.Localization(__file__, 233, 16), allclose_94936, *[rand_call_result_94941, list_94942, float_94945], **kwargs_94946)
        
        # Processing the call keyword arguments (line 233)
        kwargs_94948 = {}
        # Getting the type of 'assert_' (line 233)
        assert__94934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 233)
        assert__call_result_94949 = invoke(stypy.reporting.localization.Localization(__file__, 233, 8), assert__94934, *[allclose_call_result_94947], **kwargs_94948)
        
        
        # Call to seed(...): (line 235)
        # Processing the call arguments (line 235)
        int_94952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 24), 'int')
        # Processing the call keyword arguments (line 235)
        kwargs_94953 = {}
        # Getting the type of 'pymatrixid' (line 235)
        pymatrixid_94950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'pymatrixid', False)
        # Obtaining the member 'seed' of a type (line 235)
        seed_94951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), pymatrixid_94950, 'seed')
        # Calling seed(args, kwargs) (line 235)
        seed_call_result_94954 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), seed_94951, *[int_94952], **kwargs_94953)
        
        
        # Assigning a Call to a Name (line 236):
        
        # Assigning a Call to a Name (line 236):
        
        # Call to rand(...): (line 236)
        # Processing the call arguments (line 236)
        int_94957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 29), 'int')
        # Processing the call keyword arguments (line 236)
        kwargs_94958 = {}
        # Getting the type of 'pymatrixid' (line 236)
        pymatrixid_94955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 13), 'pymatrixid', False)
        # Obtaining the member 'rand' of a type (line 236)
        rand_94956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 13), pymatrixid_94955, 'rand')
        # Calling rand(args, kwargs) (line 236)
        rand_call_result_94959 = invoke(stypy.reporting.localization.Localization(__file__, 236, 13), rand_94956, *[int_94957], **kwargs_94958)
        
        # Assigning a type to the variable 'x1' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'x1', rand_call_result_94959)
        
        # Call to assert_(...): (line 237)
        # Processing the call arguments (line 237)
        
        # Call to allclose(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'x1' (line 237)
        x1_94963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 28), 'x1', False)
        
        # Obtaining an instance of the builtin type 'list' (line 237)
        list_94964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 237)
        # Adding element type (line 237)
        float_94965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 32), list_94964, float_94965)
        # Adding element type (line 237)
        float_94966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 32), list_94964, float_94966)
        
        float_94967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 57), 'float')
        # Processing the call keyword arguments (line 237)
        kwargs_94968 = {}
        # Getting the type of 'np' (line 237)
        np_94961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 16), 'np', False)
        # Obtaining the member 'allclose' of a type (line 237)
        allclose_94962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 16), np_94961, 'allclose')
        # Calling allclose(args, kwargs) (line 237)
        allclose_call_result_94969 = invoke(stypy.reporting.localization.Localization(__file__, 237, 16), allclose_94962, *[x1_94963, list_94964, float_94967], **kwargs_94968)
        
        # Processing the call keyword arguments (line 237)
        kwargs_94970 = {}
        # Getting the type of 'assert_' (line 237)
        assert__94960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 237)
        assert__call_result_94971 = invoke(stypy.reporting.localization.Localization(__file__, 237, 8), assert__94960, *[allclose_call_result_94969], **kwargs_94970)
        
        
        # Call to seed(...): (line 239)
        # Processing the call arguments (line 239)
        int_94975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 23), 'int')
        # Processing the call keyword arguments (line 239)
        kwargs_94976 = {}
        # Getting the type of 'np' (line 239)
        np_94972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 239)
        random_94973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), np_94972, 'random')
        # Obtaining the member 'seed' of a type (line 239)
        seed_94974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), random_94973, 'seed')
        # Calling seed(args, kwargs) (line 239)
        seed_call_result_94977 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), seed_94974, *[int_94975], **kwargs_94976)
        
        
        # Call to seed(...): (line 240)
        # Processing the call keyword arguments (line 240)
        kwargs_94980 = {}
        # Getting the type of 'pymatrixid' (line 240)
        pymatrixid_94978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'pymatrixid', False)
        # Obtaining the member 'seed' of a type (line 240)
        seed_94979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), pymatrixid_94978, 'seed')
        # Calling seed(args, kwargs) (line 240)
        seed_call_result_94981 = invoke(stypy.reporting.localization.Localization(__file__, 240, 8), seed_94979, *[], **kwargs_94980)
        
        
        # Assigning a Call to a Name (line 241):
        
        # Assigning a Call to a Name (line 241):
        
        # Call to rand(...): (line 241)
        # Processing the call arguments (line 241)
        int_94984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 29), 'int')
        # Processing the call keyword arguments (line 241)
        kwargs_94985 = {}
        # Getting the type of 'pymatrixid' (line 241)
        pymatrixid_94982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 13), 'pymatrixid', False)
        # Obtaining the member 'rand' of a type (line 241)
        rand_94983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 13), pymatrixid_94982, 'rand')
        # Calling rand(args, kwargs) (line 241)
        rand_call_result_94986 = invoke(stypy.reporting.localization.Localization(__file__, 241, 13), rand_94983, *[int_94984], **kwargs_94985)
        
        # Assigning a type to the variable 'x2' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'x2', rand_call_result_94986)
        
        # Call to seed(...): (line 243)
        # Processing the call arguments (line 243)
        int_94990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 23), 'int')
        # Processing the call keyword arguments (line 243)
        kwargs_94991 = {}
        # Getting the type of 'np' (line 243)
        np_94987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 243)
        random_94988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), np_94987, 'random')
        # Obtaining the member 'seed' of a type (line 243)
        seed_94989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), random_94988, 'seed')
        # Calling seed(args, kwargs) (line 243)
        seed_call_result_94992 = invoke(stypy.reporting.localization.Localization(__file__, 243, 8), seed_94989, *[int_94990], **kwargs_94991)
        
        
        # Call to seed(...): (line 244)
        # Processing the call arguments (line 244)
        
        # Call to rand(...): (line 244)
        # Processing the call arguments (line 244)
        int_94998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 39), 'int')
        # Processing the call keyword arguments (line 244)
        kwargs_94999 = {}
        # Getting the type of 'np' (line 244)
        np_94995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'np', False)
        # Obtaining the member 'random' of a type (line 244)
        random_94996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 24), np_94995, 'random')
        # Obtaining the member 'rand' of a type (line 244)
        rand_94997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 24), random_94996, 'rand')
        # Calling rand(args, kwargs) (line 244)
        rand_call_result_95000 = invoke(stypy.reporting.localization.Localization(__file__, 244, 24), rand_94997, *[int_94998], **kwargs_94999)
        
        # Processing the call keyword arguments (line 244)
        kwargs_95001 = {}
        # Getting the type of 'pymatrixid' (line 244)
        pymatrixid_94993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'pymatrixid', False)
        # Obtaining the member 'seed' of a type (line 244)
        seed_94994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), pymatrixid_94993, 'seed')
        # Calling seed(args, kwargs) (line 244)
        seed_call_result_95002 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), seed_94994, *[rand_call_result_95000], **kwargs_95001)
        
        
        # Assigning a Call to a Name (line 245):
        
        # Assigning a Call to a Name (line 245):
        
        # Call to rand(...): (line 245)
        # Processing the call arguments (line 245)
        int_95005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 29), 'int')
        # Processing the call keyword arguments (line 245)
        kwargs_95006 = {}
        # Getting the type of 'pymatrixid' (line 245)
        pymatrixid_95003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 13), 'pymatrixid', False)
        # Obtaining the member 'rand' of a type (line 245)
        rand_95004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 13), pymatrixid_95003, 'rand')
        # Calling rand(args, kwargs) (line 245)
        rand_call_result_95007 = invoke(stypy.reporting.localization.Localization(__file__, 245, 13), rand_95004, *[int_95005], **kwargs_95006)
        
        # Assigning a type to the variable 'x3' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'x3', rand_call_result_95007)
        
        # Call to assert_allclose(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'x1' (line 247)
        x1_95009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 24), 'x1', False)
        # Getting the type of 'x2' (line 247)
        x2_95010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 28), 'x2', False)
        # Processing the call keyword arguments (line 247)
        kwargs_95011 = {}
        # Getting the type of 'assert_allclose' (line 247)
        assert_allclose_95008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 247)
        assert_allclose_call_result_95012 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), assert_allclose_95008, *[x1_95009, x2_95010], **kwargs_95011)
        
        
        # Call to assert_allclose(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'x1' (line 248)
        x1_95014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 24), 'x1', False)
        # Getting the type of 'x3' (line 248)
        x3_95015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 28), 'x3', False)
        # Processing the call keyword arguments (line 248)
        kwargs_95016 = {}
        # Getting the type of 'assert_allclose' (line 248)
        assert_allclose_95013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 248)
        assert_allclose_call_result_95017 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), assert_allclose_95013, *[x1_95014, x3_95015], **kwargs_95016)
        
        
        # ################# End of 'test_rand(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_rand' in the type store
        # Getting the type of 'stypy_return_type' (line 231)
        stypy_return_type_95018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_95018)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_rand'
        return stypy_return_type_95018


    @norecursion
    def test_badcall(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_badcall'
        module_type_store = module_type_store.open_function_context('test_badcall', 250, 4, False)
        # Assigning a type to the variable 'self' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestInterpolativeDecomposition.test_badcall.__dict__.__setitem__('stypy_localization', localization)
        TestInterpolativeDecomposition.test_badcall.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestInterpolativeDecomposition.test_badcall.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestInterpolativeDecomposition.test_badcall.__dict__.__setitem__('stypy_function_name', 'TestInterpolativeDecomposition.test_badcall')
        TestInterpolativeDecomposition.test_badcall.__dict__.__setitem__('stypy_param_names_list', [])
        TestInterpolativeDecomposition.test_badcall.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestInterpolativeDecomposition.test_badcall.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestInterpolativeDecomposition.test_badcall.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestInterpolativeDecomposition.test_badcall.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestInterpolativeDecomposition.test_badcall.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestInterpolativeDecomposition.test_badcall.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInterpolativeDecomposition.test_badcall', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_badcall', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_badcall(...)' code ##################

        
        # Assigning a Call to a Name (line 251):
        
        # Assigning a Call to a Name (line 251):
        
        # Call to astype(...): (line 251)
        # Processing the call arguments (line 251)
        # Getting the type of 'np' (line 251)
        np_95024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 30), 'np', False)
        # Obtaining the member 'float32' of a type (line 251)
        float32_95025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 30), np_95024, 'float32')
        # Processing the call keyword arguments (line 251)
        kwargs_95026 = {}
        
        # Call to hilbert(...): (line 251)
        # Processing the call arguments (line 251)
        int_95020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 20), 'int')
        # Processing the call keyword arguments (line 251)
        kwargs_95021 = {}
        # Getting the type of 'hilbert' (line 251)
        hilbert_95019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'hilbert', False)
        # Calling hilbert(args, kwargs) (line 251)
        hilbert_call_result_95022 = invoke(stypy.reporting.localization.Localization(__file__, 251, 12), hilbert_95019, *[int_95020], **kwargs_95021)
        
        # Obtaining the member 'astype' of a type (line 251)
        astype_95023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 12), hilbert_call_result_95022, 'astype')
        # Calling astype(args, kwargs) (line 251)
        astype_call_result_95027 = invoke(stypy.reporting.localization.Localization(__file__, 251, 12), astype_95023, *[float32_95025], **kwargs_95026)
        
        # Assigning a type to the variable 'A' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'A', astype_call_result_95027)
        
        # Call to assert_raises(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'ValueError' (line 252)
        ValueError_95029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 22), 'ValueError', False)
        # Getting the type of 'pymatrixid' (line 252)
        pymatrixid_95030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 34), 'pymatrixid', False)
        # Obtaining the member 'interp_decomp' of a type (line 252)
        interp_decomp_95031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 34), pymatrixid_95030, 'interp_decomp')
        # Getting the type of 'A' (line 252)
        A_95032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 60), 'A', False)
        float_95033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 63), 'float')
        # Processing the call keyword arguments (line 252)
        # Getting the type of 'False' (line 252)
        False_95034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 74), 'False', False)
        keyword_95035 = False_95034
        kwargs_95036 = {'rand': keyword_95035}
        # Getting the type of 'assert_raises' (line 252)
        assert_raises_95028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 252)
        assert_raises_call_result_95037 = invoke(stypy.reporting.localization.Localization(__file__, 252, 8), assert_raises_95028, *[ValueError_95029, interp_decomp_95031, A_95032, float_95033], **kwargs_95036)
        
        
        # ################# End of 'test_badcall(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_badcall' in the type store
        # Getting the type of 'stypy_return_type' (line 250)
        stypy_return_type_95038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_95038)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_badcall'
        return stypy_return_type_95038


    @norecursion
    def test_rank_too_large(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_rank_too_large'
        module_type_store = module_type_store.open_function_context('test_rank_too_large', 254, 4, False)
        # Assigning a type to the variable 'self' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestInterpolativeDecomposition.test_rank_too_large.__dict__.__setitem__('stypy_localization', localization)
        TestInterpolativeDecomposition.test_rank_too_large.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestInterpolativeDecomposition.test_rank_too_large.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestInterpolativeDecomposition.test_rank_too_large.__dict__.__setitem__('stypy_function_name', 'TestInterpolativeDecomposition.test_rank_too_large')
        TestInterpolativeDecomposition.test_rank_too_large.__dict__.__setitem__('stypy_param_names_list', [])
        TestInterpolativeDecomposition.test_rank_too_large.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestInterpolativeDecomposition.test_rank_too_large.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestInterpolativeDecomposition.test_rank_too_large.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestInterpolativeDecomposition.test_rank_too_large.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestInterpolativeDecomposition.test_rank_too_large.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestInterpolativeDecomposition.test_rank_too_large.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInterpolativeDecomposition.test_rank_too_large', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_rank_too_large', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_rank_too_large(...)' code ##################

        
        # Assigning a Call to a Name (line 256):
        
        # Assigning a Call to a Name (line 256):
        
        # Call to ones(...): (line 256)
        # Processing the call arguments (line 256)
        
        # Obtaining an instance of the builtin type 'tuple' (line 256)
        tuple_95041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 256)
        # Adding element type (line 256)
        int_95042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 21), tuple_95041, int_95042)
        # Adding element type (line 256)
        int_95043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 21), tuple_95041, int_95043)
        
        # Processing the call keyword arguments (line 256)
        kwargs_95044 = {}
        # Getting the type of 'np' (line 256)
        np_95039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'np', False)
        # Obtaining the member 'ones' of a type (line 256)
        ones_95040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 12), np_95039, 'ones')
        # Calling ones(args, kwargs) (line 256)
        ones_call_result_95045 = invoke(stypy.reporting.localization.Localization(__file__, 256, 12), ones_95040, *[tuple_95041], **kwargs_95044)
        
        # Assigning a type to the variable 'a' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'a', ones_call_result_95045)
        
        # Call to assert_raises(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'ValueError' (line 257)
        ValueError_95047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 27), 'ValueError', False)
        # Processing the call keyword arguments (line 257)
        kwargs_95048 = {}
        # Getting the type of 'assert_raises' (line 257)
        assert_raises_95046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 13), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 257)
        assert_raises_call_result_95049 = invoke(stypy.reporting.localization.Localization(__file__, 257, 13), assert_raises_95046, *[ValueError_95047], **kwargs_95048)
        
        with_95050 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 257, 13), assert_raises_call_result_95049, 'with parameter', '__enter__', '__exit__')

        if with_95050:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 257)
            enter___95051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 13), assert_raises_call_result_95049, '__enter__')
            with_enter_95052 = invoke(stypy.reporting.localization.Localization(__file__, 257, 13), enter___95051)
            
            # Call to svd(...): (line 258)
            # Processing the call arguments (line 258)
            # Getting the type of 'a' (line 258)
            a_95055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 27), 'a', False)
            int_95056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 30), 'int')
            # Processing the call keyword arguments (line 258)
            kwargs_95057 = {}
            # Getting the type of 'pymatrixid' (line 258)
            pymatrixid_95053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'pymatrixid', False)
            # Obtaining the member 'svd' of a type (line 258)
            svd_95054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 12), pymatrixid_95053, 'svd')
            # Calling svd(args, kwargs) (line 258)
            svd_call_result_95058 = invoke(stypy.reporting.localization.Localization(__file__, 258, 12), svd_95054, *[a_95055, int_95056], **kwargs_95057)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 257)
            exit___95059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 13), assert_raises_call_result_95049, '__exit__')
            with_exit_95060 = invoke(stypy.reporting.localization.Localization(__file__, 257, 13), exit___95059, None, None, None)

        
        # ################# End of 'test_rank_too_large(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_rank_too_large' in the type store
        # Getting the type of 'stypy_return_type' (line 254)
        stypy_return_type_95061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_95061)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_rank_too_large'
        return stypy_return_type_95061


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 44, 0, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInterpolativeDecomposition.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestInterpolativeDecomposition' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'TestInterpolativeDecomposition', TestInterpolativeDecomposition)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
