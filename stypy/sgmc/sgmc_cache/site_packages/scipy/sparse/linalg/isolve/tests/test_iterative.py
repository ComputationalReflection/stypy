
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Test functions for the sparse.linalg.isolve module
2: '''
3: 
4: from __future__ import division, print_function, absolute_import
5: 
6: import numpy as np
7: 
8: from numpy.testing import (assert_equal, assert_array_equal,
9:      assert_, assert_allclose)
10: from pytest import raises as assert_raises
11: from scipy._lib._numpy_compat import suppress_warnings
12: 
13: from numpy import zeros, arange, array, abs, max, ones, eye, iscomplexobj
14: from scipy.linalg import norm
15: from scipy.sparse import spdiags, csr_matrix, SparseEfficiencyWarning
16: 
17: from scipy.sparse.linalg import LinearOperator, aslinearoperator
18: from scipy.sparse.linalg.isolve import cg, cgs, bicg, bicgstab, gmres, qmr, minres, lgmres, gcrotmk
19: 
20: # TODO check that method preserve shape and type
21: # TODO test both preconditioner methods
22: 
23: 
24: class Case(object):
25:     def __init__(self, name, A, skip=None):
26:         self.name = name
27:         self.A = A
28:         if skip is None:
29:             self.skip = []
30:         else:
31:             self.skip = skip
32: 
33:     def __repr__(self):
34:         return "<%s>" % self.name
35: 
36: 
37: class IterativeParams(object):
38:     def __init__(self):
39:         # list of tuples (solver, symmetric, positive_definite )
40:         solvers = [cg, cgs, bicg, bicgstab, gmres, qmr, minres, lgmres, gcrotmk]
41:         sym_solvers = [minres, cg]
42:         posdef_solvers = [cg]
43:         real_solvers = [minres]
44: 
45:         self.solvers = solvers
46: 
47:         # list of tuples (A, symmetric, positive_definite )
48:         self.cases = []
49: 
50:         # Symmetric and Positive Definite
51:         N = 40
52:         data = ones((3,N))
53:         data[0,:] = 2
54:         data[1,:] = -1
55:         data[2,:] = -1
56:         Poisson1D = spdiags(data, [0,-1,1], N, N, format='csr')
57:         self.Poisson1D = Case("poisson1d", Poisson1D)
58:         self.cases.append(Case("poisson1d", Poisson1D))
59:         # note: minres fails for single precision
60:         self.cases.append(Case("poisson1d", Poisson1D.astype('f'),
61:                                skip=[minres]))
62: 
63:         # Symmetric and Negative Definite
64:         self.cases.append(Case("neg-poisson1d", -Poisson1D,
65:                                skip=posdef_solvers))
66:         # note: minres fails for single precision
67:         self.cases.append(Case("neg-poisson1d", (-Poisson1D).astype('f'),
68:                                skip=posdef_solvers + [minres]))
69: 
70:         # Symmetric and Indefinite
71:         data = array([[6, -5, 2, 7, -1, 10, 4, -3, -8, 9]],dtype='d')
72:         RandDiag = spdiags(data, [0], 10, 10, format='csr')
73:         self.cases.append(Case("rand-diag", RandDiag, skip=posdef_solvers))
74:         self.cases.append(Case("rand-diag", RandDiag.astype('f'),
75:                                skip=posdef_solvers))
76: 
77:         # Random real-valued
78:         np.random.seed(1234)
79:         data = np.random.rand(4, 4)
80:         self.cases.append(Case("rand", data, skip=posdef_solvers+sym_solvers))
81:         self.cases.append(Case("rand", data.astype('f'),
82:                                skip=posdef_solvers+sym_solvers))
83: 
84:         # Random symmetric real-valued
85:         np.random.seed(1234)
86:         data = np.random.rand(4, 4)
87:         data = data + data.T
88:         self.cases.append(Case("rand-sym", data, skip=posdef_solvers))
89:         self.cases.append(Case("rand-sym", data.astype('f'),
90:                                skip=posdef_solvers))
91: 
92:         # Random pos-def symmetric real
93:         np.random.seed(1234)
94:         data = np.random.rand(9, 9)
95:         data = np.dot(data.conj(), data.T)
96:         self.cases.append(Case("rand-sym-pd", data))
97:         # note: minres fails for single precision
98:         self.cases.append(Case("rand-sym-pd", data.astype('f'),
99:                                skip=[minres]))
100: 
101:         # Random complex-valued
102:         np.random.seed(1234)
103:         data = np.random.rand(4, 4) + 1j*np.random.rand(4, 4)
104:         self.cases.append(Case("rand-cmplx", data,
105:                                skip=posdef_solvers+sym_solvers+real_solvers))
106:         self.cases.append(Case("rand-cmplx", data.astype('F'),
107:                                skip=posdef_solvers+sym_solvers+real_solvers))
108: 
109:         # Random hermitian complex-valued
110:         np.random.seed(1234)
111:         data = np.random.rand(4, 4) + 1j*np.random.rand(4, 4)
112:         data = data + data.T.conj()
113:         self.cases.append(Case("rand-cmplx-herm", data,
114:                                skip=posdef_solvers+real_solvers))
115:         self.cases.append(Case("rand-cmplx-herm", data.astype('F'),
116:                                skip=posdef_solvers+real_solvers))
117: 
118:         # Random pos-def hermitian complex-valued
119:         np.random.seed(1234)
120:         data = np.random.rand(9, 9) + 1j*np.random.rand(9, 9)
121:         data = np.dot(data.conj(), data.T)
122:         self.cases.append(Case("rand-cmplx-sym-pd", data, skip=real_solvers))
123:         self.cases.append(Case("rand-cmplx-sym-pd", data.astype('F'),
124:                                skip=real_solvers))
125: 
126:         # Non-symmetric and Positive Definite
127:         #
128:         # cgs, qmr, and bicg fail to converge on this one
129:         #   -- algorithmic limitation apparently
130:         data = ones((2,10))
131:         data[0,:] = 2
132:         data[1,:] = -1
133:         A = spdiags(data, [0,-1], 10, 10, format='csr')
134:         self.cases.append(Case("nonsymposdef", A,
135:                                skip=sym_solvers+[cgs, qmr, bicg]))
136:         self.cases.append(Case("nonsymposdef", A.astype('F'),
137:                                skip=sym_solvers+[cgs, qmr, bicg]))
138: 
139: 
140: params = IterativeParams()
141: 
142: 
143: def check_maxiter(solver, case):
144:     A = case.A
145:     tol = 1e-12
146: 
147:     b = arange(A.shape[0], dtype=float)
148:     x0 = 0*b
149: 
150:     residuals = []
151: 
152:     def callback(x):
153:         residuals.append(norm(b - case.A*x))
154: 
155:     x, info = solver(A, b, x0=x0, tol=tol, maxiter=1, callback=callback)
156: 
157:     assert_equal(len(residuals), 1)
158:     assert_equal(info, 1)
159: 
160: 
161: def test_maxiter():
162:     case = params.Poisson1D
163:     for solver in params.solvers:
164:         if solver in case.skip:
165:             continue
166:         check_maxiter(solver, case)
167: 
168: 
169: def assert_normclose(a, b, tol=1e-8):
170:     residual = norm(a - b)
171:     tolerance = tol*norm(b)
172:     msg = "residual (%g) not smaller than tolerance %g" % (residual, tolerance)
173:     assert_(residual < tolerance, msg=msg)
174: 
175: 
176: def check_convergence(solver, case):
177:     A = case.A
178: 
179:     if A.dtype.char in "dD":
180:         tol = 1e-8
181:     else:
182:         tol = 1e-2
183: 
184:     b = arange(A.shape[0], dtype=A.dtype)
185:     x0 = 0*b
186: 
187:     x, info = solver(A, b, x0=x0, tol=tol)
188: 
189:     assert_array_equal(x0, 0*b)  # ensure that x0 is not overwritten
190:     assert_equal(info,0)
191:     assert_normclose(A.dot(x), b, tol=tol)
192: 
193: 
194: def test_convergence():
195:     for solver in params.solvers:
196:         for case in params.cases:
197:             if solver in case.skip:
198:                 continue
199:             check_convergence(solver, case)
200: 
201: 
202: def check_precond_dummy(solver, case):
203:     tol = 1e-8
204: 
205:     def identity(b,which=None):
206:         '''trivial preconditioner'''
207:         return b
208: 
209:     A = case.A
210: 
211:     M,N = A.shape
212:     D = spdiags([1.0/A.diagonal()], [0], M, N)
213: 
214:     b = arange(A.shape[0], dtype=float)
215:     x0 = 0*b
216: 
217:     precond = LinearOperator(A.shape, identity, rmatvec=identity)
218: 
219:     if solver is qmr:
220:         x, info = solver(A, b, M1=precond, M2=precond, x0=x0, tol=tol)
221:     else:
222:         x, info = solver(A, b, M=precond, x0=x0, tol=tol)
223:     assert_equal(info,0)
224:     assert_normclose(A.dot(x), b, tol)
225: 
226:     A = aslinearoperator(A)
227:     A.psolve = identity
228:     A.rpsolve = identity
229: 
230:     x, info = solver(A, b, x0=x0, tol=tol)
231:     assert_equal(info,0)
232:     assert_normclose(A*x, b, tol=tol)
233: 
234: 
235: def test_precond_dummy():
236:     case = params.Poisson1D
237:     for solver in params.solvers:
238:         if solver in case.skip:
239:             continue
240:         check_precond_dummy(solver, case)
241: 
242: 
243: def check_precond_inverse(solver, case):
244:     tol = 1e-8
245: 
246:     def inverse(b,which=None):
247:         '''inverse preconditioner'''
248:         A = case.A
249:         if not isinstance(A, np.ndarray):
250:             A = A.todense()
251:         return np.linalg.solve(A, b)
252: 
253:     def rinverse(b,which=None):
254:         '''inverse preconditioner'''
255:         A = case.A
256:         if not isinstance(A, np.ndarray):
257:             A = A.todense()
258:         return np.linalg.solve(A.T, b)
259: 
260:     matvec_count = [0]
261: 
262:     def matvec(b):
263:         matvec_count[0] += 1
264:         return case.A.dot(b)
265: 
266:     def rmatvec(b):
267:         matvec_count[0] += 1
268:         return case.A.T.dot(b)
269: 
270:     b = arange(case.A.shape[0], dtype=float)
271:     x0 = 0*b
272: 
273:     A = LinearOperator(case.A.shape, matvec, rmatvec=rmatvec)
274:     precond = LinearOperator(case.A.shape, inverse, rmatvec=rinverse)
275: 
276:     # Solve with preconditioner
277:     matvec_count = [0]
278:     x, info = solver(A, b, M=precond, x0=x0, tol=tol)
279: 
280:     assert_equal(info, 0)
281:     assert_normclose(case.A.dot(x), b, tol)
282: 
283:     # Solution should be nearly instant
284:     assert_(matvec_count[0] <= 3, repr(matvec_count))
285: 
286: 
287: def test_precond_inverse():
288:     case = params.Poisson1D
289:     for solver in params.solvers:
290:         if solver in case.skip:
291:             continue
292:         if solver is qmr:
293:             continue
294:         check_precond_inverse(solver, case)
295: 
296: 
297: def test_gmres_basic():
298:     A = np.vander(np.arange(10) + 1)[:, ::-1]
299:     b = np.zeros(10)
300:     b[0] = 1
301:     x = np.linalg.solve(A, b)
302: 
303:     x_gm, err = gmres(A, b, restart=5, maxiter=1)
304: 
305:     assert_allclose(x_gm[0], 0.359, rtol=1e-2)
306: 
307: 
308: def test_reentrancy():
309:     non_reentrant = [cg, cgs, bicg, bicgstab, gmres, qmr]
310:     reentrant = [lgmres, minres, gcrotmk]
311:     for solver in reentrant + non_reentrant:
312:         _check_reentrancy(solver, solver in reentrant)
313: 
314: 
315: def _check_reentrancy(solver, is_reentrant):
316:     def matvec(x):
317:         A = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]])
318:         y, info = solver(A, x)
319:         assert_equal(info, 0)
320:         return y
321:     b = np.array([1, 1./2, 1./3])
322:     op = LinearOperator((3, 3), matvec=matvec, rmatvec=matvec,
323:                         dtype=b.dtype)
324: 
325:     if not is_reentrant:
326:         assert_raises(RuntimeError, solver, op, b)
327:     else:
328:         y, info = solver(op, b)
329:         assert_equal(info, 0)
330:         assert_allclose(y, [1, 1, 1])
331: 
332: 
333: #------------------------------------------------------------------------------
334: 
335: class TestQMR(object):
336:     def test_leftright_precond(self):
337:         '''Check that QMR works with left and right preconditioners'''
338: 
339:         from scipy.sparse.linalg.dsolve import splu
340:         from scipy.sparse.linalg.interface import LinearOperator
341: 
342:         n = 100
343: 
344:         dat = ones(n)
345:         A = spdiags([-2*dat, 4*dat, -dat], [-1,0,1],n,n)
346:         b = arange(n,dtype='d')
347: 
348:         L = spdiags([-dat/2, dat], [-1,0], n, n)
349:         U = spdiags([4*dat, -dat], [0,1], n, n)
350: 
351:         with suppress_warnings() as sup:
352:             sup.filter(SparseEfficiencyWarning, "splu requires CSC matrix format")
353:             L_solver = splu(L)
354:             U_solver = splu(U)
355: 
356:         def L_solve(b):
357:             return L_solver.solve(b)
358: 
359:         def U_solve(b):
360:             return U_solver.solve(b)
361: 
362:         def LT_solve(b):
363:             return L_solver.solve(b,'T')
364: 
365:         def UT_solve(b):
366:             return U_solver.solve(b,'T')
367: 
368:         M1 = LinearOperator((n,n), matvec=L_solve, rmatvec=LT_solve)
369:         M2 = LinearOperator((n,n), matvec=U_solve, rmatvec=UT_solve)
370: 
371:         x,info = qmr(A, b, tol=1e-8, maxiter=15, M1=M1, M2=M2)
372: 
373:         assert_equal(info,0)
374:         assert_normclose(A*x, b, tol=1e-8)
375: 
376: 
377: class TestGMRES(object):
378:     def test_callback(self):
379: 
380:         def store_residual(r, rvec):
381:             rvec[rvec.nonzero()[0].max()+1] = r
382: 
383:         # Define, A,b
384:         A = csr_matrix(array([[-2,1,0,0,0,0],[1,-2,1,0,0,0],[0,1,-2,1,0,0],[0,0,1,-2,1,0],[0,0,0,1,-2,1],[0,0,0,0,1,-2]]))
385:         b = ones((A.shape[0],))
386:         maxiter = 1
387:         rvec = zeros(maxiter+1)
388:         rvec[0] = 1.0
389:         callback = lambda r:store_residual(r, rvec)
390:         x,flag = gmres(A, b, x0=zeros(A.shape[0]), tol=1e-16, maxiter=maxiter, callback=callback)
391:         diff = max(abs((rvec - array([1.0, 0.81649658092772603]))))
392:         assert_(diff < 1e-5)
393: 
394:     def test_abi(self):
395:         # Check we don't segfault on gmres with complex argument
396:         A = eye(2)
397:         b = ones(2)
398:         r_x, r_info = gmres(A, b)
399:         r_x = r_x.astype(complex)
400: 
401:         x, info = gmres(A.astype(complex), b.astype(complex))
402: 
403:         assert_(iscomplexobj(x))
404:         assert_allclose(r_x, x)
405:         assert_(r_info == info)
406: 
407: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_417701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', ' Test functions for the sparse.linalg.isolve module\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_417702 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_417702) is not StypyTypeError):

    if (import_417702 != 'pyd_module'):
        __import__(import_417702)
        sys_modules_417703 = sys.modules[import_417702]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_417703.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_417702)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from numpy.testing import assert_equal, assert_array_equal, assert_, assert_allclose' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_417704 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing')

if (type(import_417704) is not StypyTypeError):

    if (import_417704 != 'pyd_module'):
        __import__(import_417704)
        sys_modules_417705 = sys.modules[import_417704]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', sys_modules_417705.module_type_store, module_type_store, ['assert_equal', 'assert_array_equal', 'assert_', 'assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_417705, sys_modules_417705.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_array_equal, assert_, assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_array_equal', 'assert_', 'assert_allclose'], [assert_equal, assert_array_equal, assert_, assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', import_417704)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from pytest import assert_raises' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_417706 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest')

if (type(import_417706) is not StypyTypeError):

    if (import_417706 != 'pyd_module'):
        __import__(import_417706)
        sys_modules_417707 = sys.modules[import_417706]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest', sys_modules_417707.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_417707, sys_modules_417707.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest', import_417706)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_417708 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._numpy_compat')

if (type(import_417708) is not StypyTypeError):

    if (import_417708 != 'pyd_module'):
        __import__(import_417708)
        sys_modules_417709 = sys.modules[import_417708]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._numpy_compat', sys_modules_417709.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_417709, sys_modules_417709.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._numpy_compat', import_417708)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from numpy import zeros, arange, array, abs, max, ones, eye, iscomplexobj' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_417710 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy')

if (type(import_417710) is not StypyTypeError):

    if (import_417710 != 'pyd_module'):
        __import__(import_417710)
        sys_modules_417711 = sys.modules[import_417710]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy', sys_modules_417711.module_type_store, module_type_store, ['zeros', 'arange', 'array', 'abs', 'max', 'ones', 'eye', 'iscomplexobj'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_417711, sys_modules_417711.module_type_store, module_type_store)
    else:
        from numpy import zeros, arange, array, abs, max, ones, eye, iscomplexobj

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy', None, module_type_store, ['zeros', 'arange', 'array', 'abs', 'max', 'ones', 'eye', 'iscomplexobj'], [zeros, arange, array, abs, max, ones, eye, iscomplexobj])

else:
    # Assigning a type to the variable 'numpy' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy', import_417710)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy.linalg import norm' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_417712 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.linalg')

if (type(import_417712) is not StypyTypeError):

    if (import_417712 != 'pyd_module'):
        __import__(import_417712)
        sys_modules_417713 = sys.modules[import_417712]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.linalg', sys_modules_417713.module_type_store, module_type_store, ['norm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_417713, sys_modules_417713.module_type_store, module_type_store)
    else:
        from scipy.linalg import norm

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.linalg', None, module_type_store, ['norm'], [norm])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.linalg', import_417712)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from scipy.sparse import spdiags, csr_matrix, SparseEfficiencyWarning' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_417714 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse')

if (type(import_417714) is not StypyTypeError):

    if (import_417714 != 'pyd_module'):
        __import__(import_417714)
        sys_modules_417715 = sys.modules[import_417714]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse', sys_modules_417715.module_type_store, module_type_store, ['spdiags', 'csr_matrix', 'SparseEfficiencyWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_417715, sys_modules_417715.module_type_store, module_type_store)
    else:
        from scipy.sparse import spdiags, csr_matrix, SparseEfficiencyWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse', None, module_type_store, ['spdiags', 'csr_matrix', 'SparseEfficiencyWarning'], [spdiags, csr_matrix, SparseEfficiencyWarning])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse', import_417714)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from scipy.sparse.linalg import LinearOperator, aslinearoperator' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_417716 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse.linalg')

if (type(import_417716) is not StypyTypeError):

    if (import_417716 != 'pyd_module'):
        __import__(import_417716)
        sys_modules_417717 = sys.modules[import_417716]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse.linalg', sys_modules_417717.module_type_store, module_type_store, ['LinearOperator', 'aslinearoperator'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_417717, sys_modules_417717.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import LinearOperator, aslinearoperator

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse.linalg', None, module_type_store, ['LinearOperator', 'aslinearoperator'], [LinearOperator, aslinearoperator])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse.linalg', import_417716)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from scipy.sparse.linalg.isolve import cg, cgs, bicg, bicgstab, gmres, qmr, minres, lgmres, gcrotmk' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_417718 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.sparse.linalg.isolve')

if (type(import_417718) is not StypyTypeError):

    if (import_417718 != 'pyd_module'):
        __import__(import_417718)
        sys_modules_417719 = sys.modules[import_417718]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.sparse.linalg.isolve', sys_modules_417719.module_type_store, module_type_store, ['cg', 'cgs', 'bicg', 'bicgstab', 'gmres', 'qmr', 'minres', 'lgmres', 'gcrotmk'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_417719, sys_modules_417719.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.isolve import cg, cgs, bicg, bicgstab, gmres, qmr, minres, lgmres, gcrotmk

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.sparse.linalg.isolve', None, module_type_store, ['cg', 'cgs', 'bicg', 'bicgstab', 'gmres', 'qmr', 'minres', 'lgmres', 'gcrotmk'], [cg, cgs, bicg, bicgstab, gmres, qmr, minres, lgmres, gcrotmk])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.isolve' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.sparse.linalg.isolve', import_417718)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

# Declaration of the 'Case' class

class Case(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 25)
        None_417720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 37), 'None')
        defaults = [None_417720]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 25, 4, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Case.__init__', ['name', 'A', 'skip'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['name', 'A', 'skip'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 26):
        
        # Assigning a Name to a Attribute (line 26):
        # Getting the type of 'name' (line 26)
        name_417721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 20), 'name')
        # Getting the type of 'self' (line 26)
        self_417722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self')
        # Setting the type of the member 'name' of a type (line 26)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_417722, 'name', name_417721)
        
        # Assigning a Name to a Attribute (line 27):
        
        # Assigning a Name to a Attribute (line 27):
        # Getting the type of 'A' (line 27)
        A_417723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 17), 'A')
        # Getting the type of 'self' (line 27)
        self_417724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'self')
        # Setting the type of the member 'A' of a type (line 27)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), self_417724, 'A', A_417723)
        
        # Type idiom detected: calculating its left and rigth part (line 28)
        # Getting the type of 'skip' (line 28)
        skip_417725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 11), 'skip')
        # Getting the type of 'None' (line 28)
        None_417726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 19), 'None')
        
        (may_be_417727, more_types_in_union_417728) = may_be_none(skip_417725, None_417726)

        if may_be_417727:

            if more_types_in_union_417728:
                # Runtime conditional SSA (line 28)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a List to a Attribute (line 29):
            
            # Assigning a List to a Attribute (line 29):
            
            # Obtaining an instance of the builtin type 'list' (line 29)
            list_417729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 24), 'list')
            # Adding type elements to the builtin type 'list' instance (line 29)
            
            # Getting the type of 'self' (line 29)
            self_417730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'self')
            # Setting the type of the member 'skip' of a type (line 29)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 12), self_417730, 'skip', list_417729)

            if more_types_in_union_417728:
                # Runtime conditional SSA for else branch (line 28)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_417727) or more_types_in_union_417728):
            
            # Assigning a Name to a Attribute (line 31):
            
            # Assigning a Name to a Attribute (line 31):
            # Getting the type of 'skip' (line 31)
            skip_417731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 24), 'skip')
            # Getting the type of 'self' (line 31)
            self_417732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'self')
            # Setting the type of the member 'skip' of a type (line 31)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), self_417732, 'skip', skip_417731)

            if (may_be_417727 and more_types_in_union_417728):
                # SSA join for if statement (line 28)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Case.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        Case.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Case.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Case.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'Case.stypy__repr__')
        Case.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        Case.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Case.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Case.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Case.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Case.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Case.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Case.stypy__repr__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__repr__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__repr__(...)' code ##################

        str_417733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 15), 'str', '<%s>')
        # Getting the type of 'self' (line 34)
        self_417734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 24), 'self')
        # Obtaining the member 'name' of a type (line 34)
        name_417735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 24), self_417734, 'name')
        # Applying the binary operator '%' (line 34)
        result_mod_417736 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 15), '%', str_417733, name_417735)
        
        # Assigning a type to the variable 'stypy_return_type' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'stypy_return_type', result_mod_417736)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_417737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_417737)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_417737


# Assigning a type to the variable 'Case' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'Case', Case)
# Declaration of the 'IterativeParams' class

class IterativeParams(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IterativeParams.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Name (line 40):
        
        # Assigning a List to a Name (line 40):
        
        # Obtaining an instance of the builtin type 'list' (line 40)
        list_417738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 40)
        # Adding element type (line 40)
        # Getting the type of 'cg' (line 40)
        cg_417739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'cg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 18), list_417738, cg_417739)
        # Adding element type (line 40)
        # Getting the type of 'cgs' (line 40)
        cgs_417740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 23), 'cgs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 18), list_417738, cgs_417740)
        # Adding element type (line 40)
        # Getting the type of 'bicg' (line 40)
        bicg_417741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 28), 'bicg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 18), list_417738, bicg_417741)
        # Adding element type (line 40)
        # Getting the type of 'bicgstab' (line 40)
        bicgstab_417742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 34), 'bicgstab')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 18), list_417738, bicgstab_417742)
        # Adding element type (line 40)
        # Getting the type of 'gmres' (line 40)
        gmres_417743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 44), 'gmres')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 18), list_417738, gmres_417743)
        # Adding element type (line 40)
        # Getting the type of 'qmr' (line 40)
        qmr_417744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 51), 'qmr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 18), list_417738, qmr_417744)
        # Adding element type (line 40)
        # Getting the type of 'minres' (line 40)
        minres_417745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 56), 'minres')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 18), list_417738, minres_417745)
        # Adding element type (line 40)
        # Getting the type of 'lgmres' (line 40)
        lgmres_417746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 64), 'lgmres')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 18), list_417738, lgmres_417746)
        # Adding element type (line 40)
        # Getting the type of 'gcrotmk' (line 40)
        gcrotmk_417747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 72), 'gcrotmk')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 18), list_417738, gcrotmk_417747)
        
        # Assigning a type to the variable 'solvers' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'solvers', list_417738)
        
        # Assigning a List to a Name (line 41):
        
        # Assigning a List to a Name (line 41):
        
        # Obtaining an instance of the builtin type 'list' (line 41)
        list_417748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 41)
        # Adding element type (line 41)
        # Getting the type of 'minres' (line 41)
        minres_417749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 23), 'minres')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 22), list_417748, minres_417749)
        # Adding element type (line 41)
        # Getting the type of 'cg' (line 41)
        cg_417750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 31), 'cg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 22), list_417748, cg_417750)
        
        # Assigning a type to the variable 'sym_solvers' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'sym_solvers', list_417748)
        
        # Assigning a List to a Name (line 42):
        
        # Assigning a List to a Name (line 42):
        
        # Obtaining an instance of the builtin type 'list' (line 42)
        list_417751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 42)
        # Adding element type (line 42)
        # Getting the type of 'cg' (line 42)
        cg_417752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 26), 'cg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 25), list_417751, cg_417752)
        
        # Assigning a type to the variable 'posdef_solvers' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'posdef_solvers', list_417751)
        
        # Assigning a List to a Name (line 43):
        
        # Assigning a List to a Name (line 43):
        
        # Obtaining an instance of the builtin type 'list' (line 43)
        list_417753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 43)
        # Adding element type (line 43)
        # Getting the type of 'minres' (line 43)
        minres_417754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 24), 'minres')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 23), list_417753, minres_417754)
        
        # Assigning a type to the variable 'real_solvers' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'real_solvers', list_417753)
        
        # Assigning a Name to a Attribute (line 45):
        
        # Assigning a Name to a Attribute (line 45):
        # Getting the type of 'solvers' (line 45)
        solvers_417755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 23), 'solvers')
        # Getting the type of 'self' (line 45)
        self_417756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'self')
        # Setting the type of the member 'solvers' of a type (line 45)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), self_417756, 'solvers', solvers_417755)
        
        # Assigning a List to a Attribute (line 48):
        
        # Assigning a List to a Attribute (line 48):
        
        # Obtaining an instance of the builtin type 'list' (line 48)
        list_417757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 48)
        
        # Getting the type of 'self' (line 48)
        self_417758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'self')
        # Setting the type of the member 'cases' of a type (line 48)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), self_417758, 'cases', list_417757)
        
        # Assigning a Num to a Name (line 51):
        
        # Assigning a Num to a Name (line 51):
        int_417759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 12), 'int')
        # Assigning a type to the variable 'N' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'N', int_417759)
        
        # Assigning a Call to a Name (line 52):
        
        # Assigning a Call to a Name (line 52):
        
        # Call to ones(...): (line 52)
        # Processing the call arguments (line 52)
        
        # Obtaining an instance of the builtin type 'tuple' (line 52)
        tuple_417761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 52)
        # Adding element type (line 52)
        int_417762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 21), tuple_417761, int_417762)
        # Adding element type (line 52)
        # Getting the type of 'N' (line 52)
        N_417763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 23), 'N', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 21), tuple_417761, N_417763)
        
        # Processing the call keyword arguments (line 52)
        kwargs_417764 = {}
        # Getting the type of 'ones' (line 52)
        ones_417760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 15), 'ones', False)
        # Calling ones(args, kwargs) (line 52)
        ones_call_result_417765 = invoke(stypy.reporting.localization.Localization(__file__, 52, 15), ones_417760, *[tuple_417761], **kwargs_417764)
        
        # Assigning a type to the variable 'data' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'data', ones_call_result_417765)
        
        # Assigning a Num to a Subscript (line 53):
        
        # Assigning a Num to a Subscript (line 53):
        int_417766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 20), 'int')
        # Getting the type of 'data' (line 53)
        data_417767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'data')
        int_417768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 13), 'int')
        slice_417769 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 53, 8), None, None, None)
        # Storing an element on a container (line 53)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 8), data_417767, ((int_417768, slice_417769), int_417766))
        
        # Assigning a Num to a Subscript (line 54):
        
        # Assigning a Num to a Subscript (line 54):
        int_417770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 20), 'int')
        # Getting the type of 'data' (line 54)
        data_417771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'data')
        int_417772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 13), 'int')
        slice_417773 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 54, 8), None, None, None)
        # Storing an element on a container (line 54)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 8), data_417771, ((int_417772, slice_417773), int_417770))
        
        # Assigning a Num to a Subscript (line 55):
        
        # Assigning a Num to a Subscript (line 55):
        int_417774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 20), 'int')
        # Getting the type of 'data' (line 55)
        data_417775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'data')
        int_417776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 13), 'int')
        slice_417777 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 55, 8), None, None, None)
        # Storing an element on a container (line 55)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 8), data_417775, ((int_417776, slice_417777), int_417774))
        
        # Assigning a Call to a Name (line 56):
        
        # Assigning a Call to a Name (line 56):
        
        # Call to spdiags(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'data' (line 56)
        data_417779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 28), 'data', False)
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_417780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        # Adding element type (line 56)
        int_417781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 34), list_417780, int_417781)
        # Adding element type (line 56)
        int_417782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 34), list_417780, int_417782)
        # Adding element type (line 56)
        int_417783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 34), list_417780, int_417783)
        
        # Getting the type of 'N' (line 56)
        N_417784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 44), 'N', False)
        # Getting the type of 'N' (line 56)
        N_417785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 47), 'N', False)
        # Processing the call keyword arguments (line 56)
        str_417786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 57), 'str', 'csr')
        keyword_417787 = str_417786
        kwargs_417788 = {'format': keyword_417787}
        # Getting the type of 'spdiags' (line 56)
        spdiags_417778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 20), 'spdiags', False)
        # Calling spdiags(args, kwargs) (line 56)
        spdiags_call_result_417789 = invoke(stypy.reporting.localization.Localization(__file__, 56, 20), spdiags_417778, *[data_417779, list_417780, N_417784, N_417785], **kwargs_417788)
        
        # Assigning a type to the variable 'Poisson1D' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'Poisson1D', spdiags_call_result_417789)
        
        # Assigning a Call to a Attribute (line 57):
        
        # Assigning a Call to a Attribute (line 57):
        
        # Call to Case(...): (line 57)
        # Processing the call arguments (line 57)
        str_417791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 30), 'str', 'poisson1d')
        # Getting the type of 'Poisson1D' (line 57)
        Poisson1D_417792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 43), 'Poisson1D', False)
        # Processing the call keyword arguments (line 57)
        kwargs_417793 = {}
        # Getting the type of 'Case' (line 57)
        Case_417790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'Case', False)
        # Calling Case(args, kwargs) (line 57)
        Case_call_result_417794 = invoke(stypy.reporting.localization.Localization(__file__, 57, 25), Case_417790, *[str_417791, Poisson1D_417792], **kwargs_417793)
        
        # Getting the type of 'self' (line 57)
        self_417795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self')
        # Setting the type of the member 'Poisson1D' of a type (line 57)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_417795, 'Poisson1D', Case_call_result_417794)
        
        # Call to append(...): (line 58)
        # Processing the call arguments (line 58)
        
        # Call to Case(...): (line 58)
        # Processing the call arguments (line 58)
        str_417800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 31), 'str', 'poisson1d')
        # Getting the type of 'Poisson1D' (line 58)
        Poisson1D_417801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 44), 'Poisson1D', False)
        # Processing the call keyword arguments (line 58)
        kwargs_417802 = {}
        # Getting the type of 'Case' (line 58)
        Case_417799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 26), 'Case', False)
        # Calling Case(args, kwargs) (line 58)
        Case_call_result_417803 = invoke(stypy.reporting.localization.Localization(__file__, 58, 26), Case_417799, *[str_417800, Poisson1D_417801], **kwargs_417802)
        
        # Processing the call keyword arguments (line 58)
        kwargs_417804 = {}
        # Getting the type of 'self' (line 58)
        self_417796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'self', False)
        # Obtaining the member 'cases' of a type (line 58)
        cases_417797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), self_417796, 'cases')
        # Obtaining the member 'append' of a type (line 58)
        append_417798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), cases_417797, 'append')
        # Calling append(args, kwargs) (line 58)
        append_call_result_417805 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), append_417798, *[Case_call_result_417803], **kwargs_417804)
        
        
        # Call to append(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Call to Case(...): (line 60)
        # Processing the call arguments (line 60)
        str_417810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 31), 'str', 'poisson1d')
        
        # Call to astype(...): (line 60)
        # Processing the call arguments (line 60)
        str_417813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 61), 'str', 'f')
        # Processing the call keyword arguments (line 60)
        kwargs_417814 = {}
        # Getting the type of 'Poisson1D' (line 60)
        Poisson1D_417811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 44), 'Poisson1D', False)
        # Obtaining the member 'astype' of a type (line 60)
        astype_417812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 44), Poisson1D_417811, 'astype')
        # Calling astype(args, kwargs) (line 60)
        astype_call_result_417815 = invoke(stypy.reporting.localization.Localization(__file__, 60, 44), astype_417812, *[str_417813], **kwargs_417814)
        
        # Processing the call keyword arguments (line 60)
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_417816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        # Getting the type of 'minres' (line 61)
        minres_417817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 37), 'minres', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 36), list_417816, minres_417817)
        
        keyword_417818 = list_417816
        kwargs_417819 = {'skip': keyword_417818}
        # Getting the type of 'Case' (line 60)
        Case_417809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 26), 'Case', False)
        # Calling Case(args, kwargs) (line 60)
        Case_call_result_417820 = invoke(stypy.reporting.localization.Localization(__file__, 60, 26), Case_417809, *[str_417810, astype_call_result_417815], **kwargs_417819)
        
        # Processing the call keyword arguments (line 60)
        kwargs_417821 = {}
        # Getting the type of 'self' (line 60)
        self_417806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'self', False)
        # Obtaining the member 'cases' of a type (line 60)
        cases_417807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), self_417806, 'cases')
        # Obtaining the member 'append' of a type (line 60)
        append_417808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), cases_417807, 'append')
        # Calling append(args, kwargs) (line 60)
        append_call_result_417822 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), append_417808, *[Case_call_result_417820], **kwargs_417821)
        
        
        # Call to append(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Call to Case(...): (line 64)
        # Processing the call arguments (line 64)
        str_417827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 31), 'str', 'neg-poisson1d')
        
        # Getting the type of 'Poisson1D' (line 64)
        Poisson1D_417828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 49), 'Poisson1D', False)
        # Applying the 'usub' unary operator (line 64)
        result___neg___417829 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 48), 'usub', Poisson1D_417828)
        
        # Processing the call keyword arguments (line 64)
        # Getting the type of 'posdef_solvers' (line 65)
        posdef_solvers_417830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 36), 'posdef_solvers', False)
        keyword_417831 = posdef_solvers_417830
        kwargs_417832 = {'skip': keyword_417831}
        # Getting the type of 'Case' (line 64)
        Case_417826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 26), 'Case', False)
        # Calling Case(args, kwargs) (line 64)
        Case_call_result_417833 = invoke(stypy.reporting.localization.Localization(__file__, 64, 26), Case_417826, *[str_417827, result___neg___417829], **kwargs_417832)
        
        # Processing the call keyword arguments (line 64)
        kwargs_417834 = {}
        # Getting the type of 'self' (line 64)
        self_417823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self', False)
        # Obtaining the member 'cases' of a type (line 64)
        cases_417824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_417823, 'cases')
        # Obtaining the member 'append' of a type (line 64)
        append_417825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), cases_417824, 'append')
        # Calling append(args, kwargs) (line 64)
        append_call_result_417835 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), append_417825, *[Case_call_result_417833], **kwargs_417834)
        
        
        # Call to append(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Call to Case(...): (line 67)
        # Processing the call arguments (line 67)
        str_417840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 31), 'str', 'neg-poisson1d')
        
        # Call to astype(...): (line 67)
        # Processing the call arguments (line 67)
        str_417844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 68), 'str', 'f')
        # Processing the call keyword arguments (line 67)
        kwargs_417845 = {}
        
        # Getting the type of 'Poisson1D' (line 67)
        Poisson1D_417841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 50), 'Poisson1D', False)
        # Applying the 'usub' unary operator (line 67)
        result___neg___417842 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 49), 'usub', Poisson1D_417841)
        
        # Obtaining the member 'astype' of a type (line 67)
        astype_417843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 49), result___neg___417842, 'astype')
        # Calling astype(args, kwargs) (line 67)
        astype_call_result_417846 = invoke(stypy.reporting.localization.Localization(__file__, 67, 49), astype_417843, *[str_417844], **kwargs_417845)
        
        # Processing the call keyword arguments (line 67)
        # Getting the type of 'posdef_solvers' (line 68)
        posdef_solvers_417847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 36), 'posdef_solvers', False)
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_417848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        # Adding element type (line 68)
        # Getting the type of 'minres' (line 68)
        minres_417849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 54), 'minres', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 53), list_417848, minres_417849)
        
        # Applying the binary operator '+' (line 68)
        result_add_417850 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 36), '+', posdef_solvers_417847, list_417848)
        
        keyword_417851 = result_add_417850
        kwargs_417852 = {'skip': keyword_417851}
        # Getting the type of 'Case' (line 67)
        Case_417839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 26), 'Case', False)
        # Calling Case(args, kwargs) (line 67)
        Case_call_result_417853 = invoke(stypy.reporting.localization.Localization(__file__, 67, 26), Case_417839, *[str_417840, astype_call_result_417846], **kwargs_417852)
        
        # Processing the call keyword arguments (line 67)
        kwargs_417854 = {}
        # Getting the type of 'self' (line 67)
        self_417836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self', False)
        # Obtaining the member 'cases' of a type (line 67)
        cases_417837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_417836, 'cases')
        # Obtaining the member 'append' of a type (line 67)
        append_417838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), cases_417837, 'append')
        # Calling append(args, kwargs) (line 67)
        append_call_result_417855 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), append_417838, *[Case_call_result_417853], **kwargs_417854)
        
        
        # Assigning a Call to a Name (line 71):
        
        # Assigning a Call to a Name (line 71):
        
        # Call to array(...): (line 71)
        # Processing the call arguments (line 71)
        
        # Obtaining an instance of the builtin type 'list' (line 71)
        list_417857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 71)
        # Adding element type (line 71)
        
        # Obtaining an instance of the builtin type 'list' (line 71)
        list_417858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 71)
        # Adding element type (line 71)
        int_417859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 22), list_417858, int_417859)
        # Adding element type (line 71)
        int_417860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 22), list_417858, int_417860)
        # Adding element type (line 71)
        int_417861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 22), list_417858, int_417861)
        # Adding element type (line 71)
        int_417862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 22), list_417858, int_417862)
        # Adding element type (line 71)
        int_417863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 22), list_417858, int_417863)
        # Adding element type (line 71)
        int_417864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 22), list_417858, int_417864)
        # Adding element type (line 71)
        int_417865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 22), list_417858, int_417865)
        # Adding element type (line 71)
        int_417866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 22), list_417858, int_417866)
        # Adding element type (line 71)
        int_417867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 22), list_417858, int_417867)
        # Adding element type (line 71)
        int_417868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 22), list_417858, int_417868)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 21), list_417857, list_417858)
        
        # Processing the call keyword arguments (line 71)
        str_417869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 65), 'str', 'd')
        keyword_417870 = str_417869
        kwargs_417871 = {'dtype': keyword_417870}
        # Getting the type of 'array' (line 71)
        array_417856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 15), 'array', False)
        # Calling array(args, kwargs) (line 71)
        array_call_result_417872 = invoke(stypy.reporting.localization.Localization(__file__, 71, 15), array_417856, *[list_417857], **kwargs_417871)
        
        # Assigning a type to the variable 'data' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'data', array_call_result_417872)
        
        # Assigning a Call to a Name (line 72):
        
        # Assigning a Call to a Name (line 72):
        
        # Call to spdiags(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'data' (line 72)
        data_417874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 27), 'data', False)
        
        # Obtaining an instance of the builtin type 'list' (line 72)
        list_417875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 72)
        # Adding element type (line 72)
        int_417876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 33), list_417875, int_417876)
        
        int_417877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 38), 'int')
        int_417878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 42), 'int')
        # Processing the call keyword arguments (line 72)
        str_417879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 53), 'str', 'csr')
        keyword_417880 = str_417879
        kwargs_417881 = {'format': keyword_417880}
        # Getting the type of 'spdiags' (line 72)
        spdiags_417873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), 'spdiags', False)
        # Calling spdiags(args, kwargs) (line 72)
        spdiags_call_result_417882 = invoke(stypy.reporting.localization.Localization(__file__, 72, 19), spdiags_417873, *[data_417874, list_417875, int_417877, int_417878], **kwargs_417881)
        
        # Assigning a type to the variable 'RandDiag' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'RandDiag', spdiags_call_result_417882)
        
        # Call to append(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Call to Case(...): (line 73)
        # Processing the call arguments (line 73)
        str_417887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 31), 'str', 'rand-diag')
        # Getting the type of 'RandDiag' (line 73)
        RandDiag_417888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 44), 'RandDiag', False)
        # Processing the call keyword arguments (line 73)
        # Getting the type of 'posdef_solvers' (line 73)
        posdef_solvers_417889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 59), 'posdef_solvers', False)
        keyword_417890 = posdef_solvers_417889
        kwargs_417891 = {'skip': keyword_417890}
        # Getting the type of 'Case' (line 73)
        Case_417886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 26), 'Case', False)
        # Calling Case(args, kwargs) (line 73)
        Case_call_result_417892 = invoke(stypy.reporting.localization.Localization(__file__, 73, 26), Case_417886, *[str_417887, RandDiag_417888], **kwargs_417891)
        
        # Processing the call keyword arguments (line 73)
        kwargs_417893 = {}
        # Getting the type of 'self' (line 73)
        self_417883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self', False)
        # Obtaining the member 'cases' of a type (line 73)
        cases_417884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_417883, 'cases')
        # Obtaining the member 'append' of a type (line 73)
        append_417885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), cases_417884, 'append')
        # Calling append(args, kwargs) (line 73)
        append_call_result_417894 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), append_417885, *[Case_call_result_417892], **kwargs_417893)
        
        
        # Call to append(...): (line 74)
        # Processing the call arguments (line 74)
        
        # Call to Case(...): (line 74)
        # Processing the call arguments (line 74)
        str_417899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 31), 'str', 'rand-diag')
        
        # Call to astype(...): (line 74)
        # Processing the call arguments (line 74)
        str_417902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 60), 'str', 'f')
        # Processing the call keyword arguments (line 74)
        kwargs_417903 = {}
        # Getting the type of 'RandDiag' (line 74)
        RandDiag_417900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 44), 'RandDiag', False)
        # Obtaining the member 'astype' of a type (line 74)
        astype_417901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 44), RandDiag_417900, 'astype')
        # Calling astype(args, kwargs) (line 74)
        astype_call_result_417904 = invoke(stypy.reporting.localization.Localization(__file__, 74, 44), astype_417901, *[str_417902], **kwargs_417903)
        
        # Processing the call keyword arguments (line 74)
        # Getting the type of 'posdef_solvers' (line 75)
        posdef_solvers_417905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 36), 'posdef_solvers', False)
        keyword_417906 = posdef_solvers_417905
        kwargs_417907 = {'skip': keyword_417906}
        # Getting the type of 'Case' (line 74)
        Case_417898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 26), 'Case', False)
        # Calling Case(args, kwargs) (line 74)
        Case_call_result_417908 = invoke(stypy.reporting.localization.Localization(__file__, 74, 26), Case_417898, *[str_417899, astype_call_result_417904], **kwargs_417907)
        
        # Processing the call keyword arguments (line 74)
        kwargs_417909 = {}
        # Getting the type of 'self' (line 74)
        self_417895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'self', False)
        # Obtaining the member 'cases' of a type (line 74)
        cases_417896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), self_417895, 'cases')
        # Obtaining the member 'append' of a type (line 74)
        append_417897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), cases_417896, 'append')
        # Calling append(args, kwargs) (line 74)
        append_call_result_417910 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), append_417897, *[Case_call_result_417908], **kwargs_417909)
        
        
        # Call to seed(...): (line 78)
        # Processing the call arguments (line 78)
        int_417914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 23), 'int')
        # Processing the call keyword arguments (line 78)
        kwargs_417915 = {}
        # Getting the type of 'np' (line 78)
        np_417911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 78)
        random_417912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), np_417911, 'random')
        # Obtaining the member 'seed' of a type (line 78)
        seed_417913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), random_417912, 'seed')
        # Calling seed(args, kwargs) (line 78)
        seed_call_result_417916 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), seed_417913, *[int_417914], **kwargs_417915)
        
        
        # Assigning a Call to a Name (line 79):
        
        # Assigning a Call to a Name (line 79):
        
        # Call to rand(...): (line 79)
        # Processing the call arguments (line 79)
        int_417920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 30), 'int')
        int_417921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 33), 'int')
        # Processing the call keyword arguments (line 79)
        kwargs_417922 = {}
        # Getting the type of 'np' (line 79)
        np_417917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 15), 'np', False)
        # Obtaining the member 'random' of a type (line 79)
        random_417918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 15), np_417917, 'random')
        # Obtaining the member 'rand' of a type (line 79)
        rand_417919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 15), random_417918, 'rand')
        # Calling rand(args, kwargs) (line 79)
        rand_call_result_417923 = invoke(stypy.reporting.localization.Localization(__file__, 79, 15), rand_417919, *[int_417920, int_417921], **kwargs_417922)
        
        # Assigning a type to the variable 'data' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'data', rand_call_result_417923)
        
        # Call to append(...): (line 80)
        # Processing the call arguments (line 80)
        
        # Call to Case(...): (line 80)
        # Processing the call arguments (line 80)
        str_417928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 31), 'str', 'rand')
        # Getting the type of 'data' (line 80)
        data_417929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 39), 'data', False)
        # Processing the call keyword arguments (line 80)
        # Getting the type of 'posdef_solvers' (line 80)
        posdef_solvers_417930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 50), 'posdef_solvers', False)
        # Getting the type of 'sym_solvers' (line 80)
        sym_solvers_417931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 65), 'sym_solvers', False)
        # Applying the binary operator '+' (line 80)
        result_add_417932 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 50), '+', posdef_solvers_417930, sym_solvers_417931)
        
        keyword_417933 = result_add_417932
        kwargs_417934 = {'skip': keyword_417933}
        # Getting the type of 'Case' (line 80)
        Case_417927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 26), 'Case', False)
        # Calling Case(args, kwargs) (line 80)
        Case_call_result_417935 = invoke(stypy.reporting.localization.Localization(__file__, 80, 26), Case_417927, *[str_417928, data_417929], **kwargs_417934)
        
        # Processing the call keyword arguments (line 80)
        kwargs_417936 = {}
        # Getting the type of 'self' (line 80)
        self_417924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'self', False)
        # Obtaining the member 'cases' of a type (line 80)
        cases_417925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), self_417924, 'cases')
        # Obtaining the member 'append' of a type (line 80)
        append_417926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), cases_417925, 'append')
        # Calling append(args, kwargs) (line 80)
        append_call_result_417937 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), append_417926, *[Case_call_result_417935], **kwargs_417936)
        
        
        # Call to append(...): (line 81)
        # Processing the call arguments (line 81)
        
        # Call to Case(...): (line 81)
        # Processing the call arguments (line 81)
        str_417942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 31), 'str', 'rand')
        
        # Call to astype(...): (line 81)
        # Processing the call arguments (line 81)
        str_417945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 51), 'str', 'f')
        # Processing the call keyword arguments (line 81)
        kwargs_417946 = {}
        # Getting the type of 'data' (line 81)
        data_417943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 39), 'data', False)
        # Obtaining the member 'astype' of a type (line 81)
        astype_417944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 39), data_417943, 'astype')
        # Calling astype(args, kwargs) (line 81)
        astype_call_result_417947 = invoke(stypy.reporting.localization.Localization(__file__, 81, 39), astype_417944, *[str_417945], **kwargs_417946)
        
        # Processing the call keyword arguments (line 81)
        # Getting the type of 'posdef_solvers' (line 82)
        posdef_solvers_417948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 36), 'posdef_solvers', False)
        # Getting the type of 'sym_solvers' (line 82)
        sym_solvers_417949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 51), 'sym_solvers', False)
        # Applying the binary operator '+' (line 82)
        result_add_417950 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 36), '+', posdef_solvers_417948, sym_solvers_417949)
        
        keyword_417951 = result_add_417950
        kwargs_417952 = {'skip': keyword_417951}
        # Getting the type of 'Case' (line 81)
        Case_417941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 26), 'Case', False)
        # Calling Case(args, kwargs) (line 81)
        Case_call_result_417953 = invoke(stypy.reporting.localization.Localization(__file__, 81, 26), Case_417941, *[str_417942, astype_call_result_417947], **kwargs_417952)
        
        # Processing the call keyword arguments (line 81)
        kwargs_417954 = {}
        # Getting the type of 'self' (line 81)
        self_417938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'self', False)
        # Obtaining the member 'cases' of a type (line 81)
        cases_417939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), self_417938, 'cases')
        # Obtaining the member 'append' of a type (line 81)
        append_417940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), cases_417939, 'append')
        # Calling append(args, kwargs) (line 81)
        append_call_result_417955 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), append_417940, *[Case_call_result_417953], **kwargs_417954)
        
        
        # Call to seed(...): (line 85)
        # Processing the call arguments (line 85)
        int_417959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 23), 'int')
        # Processing the call keyword arguments (line 85)
        kwargs_417960 = {}
        # Getting the type of 'np' (line 85)
        np_417956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 85)
        random_417957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), np_417956, 'random')
        # Obtaining the member 'seed' of a type (line 85)
        seed_417958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), random_417957, 'seed')
        # Calling seed(args, kwargs) (line 85)
        seed_call_result_417961 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), seed_417958, *[int_417959], **kwargs_417960)
        
        
        # Assigning a Call to a Name (line 86):
        
        # Assigning a Call to a Name (line 86):
        
        # Call to rand(...): (line 86)
        # Processing the call arguments (line 86)
        int_417965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 30), 'int')
        int_417966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 33), 'int')
        # Processing the call keyword arguments (line 86)
        kwargs_417967 = {}
        # Getting the type of 'np' (line 86)
        np_417962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'np', False)
        # Obtaining the member 'random' of a type (line 86)
        random_417963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 15), np_417962, 'random')
        # Obtaining the member 'rand' of a type (line 86)
        rand_417964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 15), random_417963, 'rand')
        # Calling rand(args, kwargs) (line 86)
        rand_call_result_417968 = invoke(stypy.reporting.localization.Localization(__file__, 86, 15), rand_417964, *[int_417965, int_417966], **kwargs_417967)
        
        # Assigning a type to the variable 'data' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'data', rand_call_result_417968)
        
        # Assigning a BinOp to a Name (line 87):
        
        # Assigning a BinOp to a Name (line 87):
        # Getting the type of 'data' (line 87)
        data_417969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'data')
        # Getting the type of 'data' (line 87)
        data_417970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 22), 'data')
        # Obtaining the member 'T' of a type (line 87)
        T_417971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 22), data_417970, 'T')
        # Applying the binary operator '+' (line 87)
        result_add_417972 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 15), '+', data_417969, T_417971)
        
        # Assigning a type to the variable 'data' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'data', result_add_417972)
        
        # Call to append(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Call to Case(...): (line 88)
        # Processing the call arguments (line 88)
        str_417977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 31), 'str', 'rand-sym')
        # Getting the type of 'data' (line 88)
        data_417978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 43), 'data', False)
        # Processing the call keyword arguments (line 88)
        # Getting the type of 'posdef_solvers' (line 88)
        posdef_solvers_417979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 54), 'posdef_solvers', False)
        keyword_417980 = posdef_solvers_417979
        kwargs_417981 = {'skip': keyword_417980}
        # Getting the type of 'Case' (line 88)
        Case_417976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 26), 'Case', False)
        # Calling Case(args, kwargs) (line 88)
        Case_call_result_417982 = invoke(stypy.reporting.localization.Localization(__file__, 88, 26), Case_417976, *[str_417977, data_417978], **kwargs_417981)
        
        # Processing the call keyword arguments (line 88)
        kwargs_417983 = {}
        # Getting the type of 'self' (line 88)
        self_417973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'self', False)
        # Obtaining the member 'cases' of a type (line 88)
        cases_417974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), self_417973, 'cases')
        # Obtaining the member 'append' of a type (line 88)
        append_417975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), cases_417974, 'append')
        # Calling append(args, kwargs) (line 88)
        append_call_result_417984 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), append_417975, *[Case_call_result_417982], **kwargs_417983)
        
        
        # Call to append(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Call to Case(...): (line 89)
        # Processing the call arguments (line 89)
        str_417989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 31), 'str', 'rand-sym')
        
        # Call to astype(...): (line 89)
        # Processing the call arguments (line 89)
        str_417992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 55), 'str', 'f')
        # Processing the call keyword arguments (line 89)
        kwargs_417993 = {}
        # Getting the type of 'data' (line 89)
        data_417990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 43), 'data', False)
        # Obtaining the member 'astype' of a type (line 89)
        astype_417991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 43), data_417990, 'astype')
        # Calling astype(args, kwargs) (line 89)
        astype_call_result_417994 = invoke(stypy.reporting.localization.Localization(__file__, 89, 43), astype_417991, *[str_417992], **kwargs_417993)
        
        # Processing the call keyword arguments (line 89)
        # Getting the type of 'posdef_solvers' (line 90)
        posdef_solvers_417995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 36), 'posdef_solvers', False)
        keyword_417996 = posdef_solvers_417995
        kwargs_417997 = {'skip': keyword_417996}
        # Getting the type of 'Case' (line 89)
        Case_417988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'Case', False)
        # Calling Case(args, kwargs) (line 89)
        Case_call_result_417998 = invoke(stypy.reporting.localization.Localization(__file__, 89, 26), Case_417988, *[str_417989, astype_call_result_417994], **kwargs_417997)
        
        # Processing the call keyword arguments (line 89)
        kwargs_417999 = {}
        # Getting the type of 'self' (line 89)
        self_417985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'self', False)
        # Obtaining the member 'cases' of a type (line 89)
        cases_417986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), self_417985, 'cases')
        # Obtaining the member 'append' of a type (line 89)
        append_417987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), cases_417986, 'append')
        # Calling append(args, kwargs) (line 89)
        append_call_result_418000 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), append_417987, *[Case_call_result_417998], **kwargs_417999)
        
        
        # Call to seed(...): (line 93)
        # Processing the call arguments (line 93)
        int_418004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 23), 'int')
        # Processing the call keyword arguments (line 93)
        kwargs_418005 = {}
        # Getting the type of 'np' (line 93)
        np_418001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 93)
        random_418002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), np_418001, 'random')
        # Obtaining the member 'seed' of a type (line 93)
        seed_418003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), random_418002, 'seed')
        # Calling seed(args, kwargs) (line 93)
        seed_call_result_418006 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), seed_418003, *[int_418004], **kwargs_418005)
        
        
        # Assigning a Call to a Name (line 94):
        
        # Assigning a Call to a Name (line 94):
        
        # Call to rand(...): (line 94)
        # Processing the call arguments (line 94)
        int_418010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 30), 'int')
        int_418011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 33), 'int')
        # Processing the call keyword arguments (line 94)
        kwargs_418012 = {}
        # Getting the type of 'np' (line 94)
        np_418007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), 'np', False)
        # Obtaining the member 'random' of a type (line 94)
        random_418008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 15), np_418007, 'random')
        # Obtaining the member 'rand' of a type (line 94)
        rand_418009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 15), random_418008, 'rand')
        # Calling rand(args, kwargs) (line 94)
        rand_call_result_418013 = invoke(stypy.reporting.localization.Localization(__file__, 94, 15), rand_418009, *[int_418010, int_418011], **kwargs_418012)
        
        # Assigning a type to the variable 'data' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'data', rand_call_result_418013)
        
        # Assigning a Call to a Name (line 95):
        
        # Assigning a Call to a Name (line 95):
        
        # Call to dot(...): (line 95)
        # Processing the call arguments (line 95)
        
        # Call to conj(...): (line 95)
        # Processing the call keyword arguments (line 95)
        kwargs_418018 = {}
        # Getting the type of 'data' (line 95)
        data_418016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 22), 'data', False)
        # Obtaining the member 'conj' of a type (line 95)
        conj_418017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 22), data_418016, 'conj')
        # Calling conj(args, kwargs) (line 95)
        conj_call_result_418019 = invoke(stypy.reporting.localization.Localization(__file__, 95, 22), conj_418017, *[], **kwargs_418018)
        
        # Getting the type of 'data' (line 95)
        data_418020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 35), 'data', False)
        # Obtaining the member 'T' of a type (line 95)
        T_418021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 35), data_418020, 'T')
        # Processing the call keyword arguments (line 95)
        kwargs_418022 = {}
        # Getting the type of 'np' (line 95)
        np_418014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'np', False)
        # Obtaining the member 'dot' of a type (line 95)
        dot_418015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 15), np_418014, 'dot')
        # Calling dot(args, kwargs) (line 95)
        dot_call_result_418023 = invoke(stypy.reporting.localization.Localization(__file__, 95, 15), dot_418015, *[conj_call_result_418019, T_418021], **kwargs_418022)
        
        # Assigning a type to the variable 'data' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'data', dot_call_result_418023)
        
        # Call to append(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Call to Case(...): (line 96)
        # Processing the call arguments (line 96)
        str_418028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 31), 'str', 'rand-sym-pd')
        # Getting the type of 'data' (line 96)
        data_418029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 46), 'data', False)
        # Processing the call keyword arguments (line 96)
        kwargs_418030 = {}
        # Getting the type of 'Case' (line 96)
        Case_418027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'Case', False)
        # Calling Case(args, kwargs) (line 96)
        Case_call_result_418031 = invoke(stypy.reporting.localization.Localization(__file__, 96, 26), Case_418027, *[str_418028, data_418029], **kwargs_418030)
        
        # Processing the call keyword arguments (line 96)
        kwargs_418032 = {}
        # Getting the type of 'self' (line 96)
        self_418024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'self', False)
        # Obtaining the member 'cases' of a type (line 96)
        cases_418025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), self_418024, 'cases')
        # Obtaining the member 'append' of a type (line 96)
        append_418026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), cases_418025, 'append')
        # Calling append(args, kwargs) (line 96)
        append_call_result_418033 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), append_418026, *[Case_call_result_418031], **kwargs_418032)
        
        
        # Call to append(...): (line 98)
        # Processing the call arguments (line 98)
        
        # Call to Case(...): (line 98)
        # Processing the call arguments (line 98)
        str_418038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 31), 'str', 'rand-sym-pd')
        
        # Call to astype(...): (line 98)
        # Processing the call arguments (line 98)
        str_418041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 58), 'str', 'f')
        # Processing the call keyword arguments (line 98)
        kwargs_418042 = {}
        # Getting the type of 'data' (line 98)
        data_418039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 46), 'data', False)
        # Obtaining the member 'astype' of a type (line 98)
        astype_418040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 46), data_418039, 'astype')
        # Calling astype(args, kwargs) (line 98)
        astype_call_result_418043 = invoke(stypy.reporting.localization.Localization(__file__, 98, 46), astype_418040, *[str_418041], **kwargs_418042)
        
        # Processing the call keyword arguments (line 98)
        
        # Obtaining an instance of the builtin type 'list' (line 99)
        list_418044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 99)
        # Adding element type (line 99)
        # Getting the type of 'minres' (line 99)
        minres_418045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 37), 'minres', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 36), list_418044, minres_418045)
        
        keyword_418046 = list_418044
        kwargs_418047 = {'skip': keyword_418046}
        # Getting the type of 'Case' (line 98)
        Case_418037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 26), 'Case', False)
        # Calling Case(args, kwargs) (line 98)
        Case_call_result_418048 = invoke(stypy.reporting.localization.Localization(__file__, 98, 26), Case_418037, *[str_418038, astype_call_result_418043], **kwargs_418047)
        
        # Processing the call keyword arguments (line 98)
        kwargs_418049 = {}
        # Getting the type of 'self' (line 98)
        self_418034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'self', False)
        # Obtaining the member 'cases' of a type (line 98)
        cases_418035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), self_418034, 'cases')
        # Obtaining the member 'append' of a type (line 98)
        append_418036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), cases_418035, 'append')
        # Calling append(args, kwargs) (line 98)
        append_call_result_418050 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), append_418036, *[Case_call_result_418048], **kwargs_418049)
        
        
        # Call to seed(...): (line 102)
        # Processing the call arguments (line 102)
        int_418054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 23), 'int')
        # Processing the call keyword arguments (line 102)
        kwargs_418055 = {}
        # Getting the type of 'np' (line 102)
        np_418051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 102)
        random_418052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), np_418051, 'random')
        # Obtaining the member 'seed' of a type (line 102)
        seed_418053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), random_418052, 'seed')
        # Calling seed(args, kwargs) (line 102)
        seed_call_result_418056 = invoke(stypy.reporting.localization.Localization(__file__, 102, 8), seed_418053, *[int_418054], **kwargs_418055)
        
        
        # Assigning a BinOp to a Name (line 103):
        
        # Assigning a BinOp to a Name (line 103):
        
        # Call to rand(...): (line 103)
        # Processing the call arguments (line 103)
        int_418060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 30), 'int')
        int_418061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 33), 'int')
        # Processing the call keyword arguments (line 103)
        kwargs_418062 = {}
        # Getting the type of 'np' (line 103)
        np_418057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'np', False)
        # Obtaining the member 'random' of a type (line 103)
        random_418058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 15), np_418057, 'random')
        # Obtaining the member 'rand' of a type (line 103)
        rand_418059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 15), random_418058, 'rand')
        # Calling rand(args, kwargs) (line 103)
        rand_call_result_418063 = invoke(stypy.reporting.localization.Localization(__file__, 103, 15), rand_418059, *[int_418060, int_418061], **kwargs_418062)
        
        complex_418064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 38), 'complex')
        
        # Call to rand(...): (line 103)
        # Processing the call arguments (line 103)
        int_418068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 56), 'int')
        int_418069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 59), 'int')
        # Processing the call keyword arguments (line 103)
        kwargs_418070 = {}
        # Getting the type of 'np' (line 103)
        np_418065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 41), 'np', False)
        # Obtaining the member 'random' of a type (line 103)
        random_418066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 41), np_418065, 'random')
        # Obtaining the member 'rand' of a type (line 103)
        rand_418067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 41), random_418066, 'rand')
        # Calling rand(args, kwargs) (line 103)
        rand_call_result_418071 = invoke(stypy.reporting.localization.Localization(__file__, 103, 41), rand_418067, *[int_418068, int_418069], **kwargs_418070)
        
        # Applying the binary operator '*' (line 103)
        result_mul_418072 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 38), '*', complex_418064, rand_call_result_418071)
        
        # Applying the binary operator '+' (line 103)
        result_add_418073 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 15), '+', rand_call_result_418063, result_mul_418072)
        
        # Assigning a type to the variable 'data' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'data', result_add_418073)
        
        # Call to append(...): (line 104)
        # Processing the call arguments (line 104)
        
        # Call to Case(...): (line 104)
        # Processing the call arguments (line 104)
        str_418078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 31), 'str', 'rand-cmplx')
        # Getting the type of 'data' (line 104)
        data_418079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 45), 'data', False)
        # Processing the call keyword arguments (line 104)
        # Getting the type of 'posdef_solvers' (line 105)
        posdef_solvers_418080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 36), 'posdef_solvers', False)
        # Getting the type of 'sym_solvers' (line 105)
        sym_solvers_418081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 51), 'sym_solvers', False)
        # Applying the binary operator '+' (line 105)
        result_add_418082 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 36), '+', posdef_solvers_418080, sym_solvers_418081)
        
        # Getting the type of 'real_solvers' (line 105)
        real_solvers_418083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 63), 'real_solvers', False)
        # Applying the binary operator '+' (line 105)
        result_add_418084 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 62), '+', result_add_418082, real_solvers_418083)
        
        keyword_418085 = result_add_418084
        kwargs_418086 = {'skip': keyword_418085}
        # Getting the type of 'Case' (line 104)
        Case_418077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 26), 'Case', False)
        # Calling Case(args, kwargs) (line 104)
        Case_call_result_418087 = invoke(stypy.reporting.localization.Localization(__file__, 104, 26), Case_418077, *[str_418078, data_418079], **kwargs_418086)
        
        # Processing the call keyword arguments (line 104)
        kwargs_418088 = {}
        # Getting the type of 'self' (line 104)
        self_418074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'self', False)
        # Obtaining the member 'cases' of a type (line 104)
        cases_418075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), self_418074, 'cases')
        # Obtaining the member 'append' of a type (line 104)
        append_418076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), cases_418075, 'append')
        # Calling append(args, kwargs) (line 104)
        append_call_result_418089 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), append_418076, *[Case_call_result_418087], **kwargs_418088)
        
        
        # Call to append(...): (line 106)
        # Processing the call arguments (line 106)
        
        # Call to Case(...): (line 106)
        # Processing the call arguments (line 106)
        str_418094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 31), 'str', 'rand-cmplx')
        
        # Call to astype(...): (line 106)
        # Processing the call arguments (line 106)
        str_418097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 57), 'str', 'F')
        # Processing the call keyword arguments (line 106)
        kwargs_418098 = {}
        # Getting the type of 'data' (line 106)
        data_418095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 45), 'data', False)
        # Obtaining the member 'astype' of a type (line 106)
        astype_418096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 45), data_418095, 'astype')
        # Calling astype(args, kwargs) (line 106)
        astype_call_result_418099 = invoke(stypy.reporting.localization.Localization(__file__, 106, 45), astype_418096, *[str_418097], **kwargs_418098)
        
        # Processing the call keyword arguments (line 106)
        # Getting the type of 'posdef_solvers' (line 107)
        posdef_solvers_418100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 36), 'posdef_solvers', False)
        # Getting the type of 'sym_solvers' (line 107)
        sym_solvers_418101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 51), 'sym_solvers', False)
        # Applying the binary operator '+' (line 107)
        result_add_418102 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 36), '+', posdef_solvers_418100, sym_solvers_418101)
        
        # Getting the type of 'real_solvers' (line 107)
        real_solvers_418103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 63), 'real_solvers', False)
        # Applying the binary operator '+' (line 107)
        result_add_418104 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 62), '+', result_add_418102, real_solvers_418103)
        
        keyword_418105 = result_add_418104
        kwargs_418106 = {'skip': keyword_418105}
        # Getting the type of 'Case' (line 106)
        Case_418093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 26), 'Case', False)
        # Calling Case(args, kwargs) (line 106)
        Case_call_result_418107 = invoke(stypy.reporting.localization.Localization(__file__, 106, 26), Case_418093, *[str_418094, astype_call_result_418099], **kwargs_418106)
        
        # Processing the call keyword arguments (line 106)
        kwargs_418108 = {}
        # Getting the type of 'self' (line 106)
        self_418090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'self', False)
        # Obtaining the member 'cases' of a type (line 106)
        cases_418091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), self_418090, 'cases')
        # Obtaining the member 'append' of a type (line 106)
        append_418092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), cases_418091, 'append')
        # Calling append(args, kwargs) (line 106)
        append_call_result_418109 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), append_418092, *[Case_call_result_418107], **kwargs_418108)
        
        
        # Call to seed(...): (line 110)
        # Processing the call arguments (line 110)
        int_418113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 23), 'int')
        # Processing the call keyword arguments (line 110)
        kwargs_418114 = {}
        # Getting the type of 'np' (line 110)
        np_418110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 110)
        random_418111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), np_418110, 'random')
        # Obtaining the member 'seed' of a type (line 110)
        seed_418112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), random_418111, 'seed')
        # Calling seed(args, kwargs) (line 110)
        seed_call_result_418115 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), seed_418112, *[int_418113], **kwargs_418114)
        
        
        # Assigning a BinOp to a Name (line 111):
        
        # Assigning a BinOp to a Name (line 111):
        
        # Call to rand(...): (line 111)
        # Processing the call arguments (line 111)
        int_418119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 30), 'int')
        int_418120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 33), 'int')
        # Processing the call keyword arguments (line 111)
        kwargs_418121 = {}
        # Getting the type of 'np' (line 111)
        np_418116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'np', False)
        # Obtaining the member 'random' of a type (line 111)
        random_418117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 15), np_418116, 'random')
        # Obtaining the member 'rand' of a type (line 111)
        rand_418118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 15), random_418117, 'rand')
        # Calling rand(args, kwargs) (line 111)
        rand_call_result_418122 = invoke(stypy.reporting.localization.Localization(__file__, 111, 15), rand_418118, *[int_418119, int_418120], **kwargs_418121)
        
        complex_418123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 38), 'complex')
        
        # Call to rand(...): (line 111)
        # Processing the call arguments (line 111)
        int_418127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 56), 'int')
        int_418128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 59), 'int')
        # Processing the call keyword arguments (line 111)
        kwargs_418129 = {}
        # Getting the type of 'np' (line 111)
        np_418124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 41), 'np', False)
        # Obtaining the member 'random' of a type (line 111)
        random_418125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 41), np_418124, 'random')
        # Obtaining the member 'rand' of a type (line 111)
        rand_418126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 41), random_418125, 'rand')
        # Calling rand(args, kwargs) (line 111)
        rand_call_result_418130 = invoke(stypy.reporting.localization.Localization(__file__, 111, 41), rand_418126, *[int_418127, int_418128], **kwargs_418129)
        
        # Applying the binary operator '*' (line 111)
        result_mul_418131 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 38), '*', complex_418123, rand_call_result_418130)
        
        # Applying the binary operator '+' (line 111)
        result_add_418132 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 15), '+', rand_call_result_418122, result_mul_418131)
        
        # Assigning a type to the variable 'data' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'data', result_add_418132)
        
        # Assigning a BinOp to a Name (line 112):
        
        # Assigning a BinOp to a Name (line 112):
        # Getting the type of 'data' (line 112)
        data_418133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 15), 'data')
        
        # Call to conj(...): (line 112)
        # Processing the call keyword arguments (line 112)
        kwargs_418137 = {}
        # Getting the type of 'data' (line 112)
        data_418134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 22), 'data', False)
        # Obtaining the member 'T' of a type (line 112)
        T_418135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 22), data_418134, 'T')
        # Obtaining the member 'conj' of a type (line 112)
        conj_418136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 22), T_418135, 'conj')
        # Calling conj(args, kwargs) (line 112)
        conj_call_result_418138 = invoke(stypy.reporting.localization.Localization(__file__, 112, 22), conj_418136, *[], **kwargs_418137)
        
        # Applying the binary operator '+' (line 112)
        result_add_418139 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 15), '+', data_418133, conj_call_result_418138)
        
        # Assigning a type to the variable 'data' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'data', result_add_418139)
        
        # Call to append(...): (line 113)
        # Processing the call arguments (line 113)
        
        # Call to Case(...): (line 113)
        # Processing the call arguments (line 113)
        str_418144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 31), 'str', 'rand-cmplx-herm')
        # Getting the type of 'data' (line 113)
        data_418145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 50), 'data', False)
        # Processing the call keyword arguments (line 113)
        # Getting the type of 'posdef_solvers' (line 114)
        posdef_solvers_418146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 36), 'posdef_solvers', False)
        # Getting the type of 'real_solvers' (line 114)
        real_solvers_418147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 51), 'real_solvers', False)
        # Applying the binary operator '+' (line 114)
        result_add_418148 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 36), '+', posdef_solvers_418146, real_solvers_418147)
        
        keyword_418149 = result_add_418148
        kwargs_418150 = {'skip': keyword_418149}
        # Getting the type of 'Case' (line 113)
        Case_418143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 26), 'Case', False)
        # Calling Case(args, kwargs) (line 113)
        Case_call_result_418151 = invoke(stypy.reporting.localization.Localization(__file__, 113, 26), Case_418143, *[str_418144, data_418145], **kwargs_418150)
        
        # Processing the call keyword arguments (line 113)
        kwargs_418152 = {}
        # Getting the type of 'self' (line 113)
        self_418140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'self', False)
        # Obtaining the member 'cases' of a type (line 113)
        cases_418141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), self_418140, 'cases')
        # Obtaining the member 'append' of a type (line 113)
        append_418142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), cases_418141, 'append')
        # Calling append(args, kwargs) (line 113)
        append_call_result_418153 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), append_418142, *[Case_call_result_418151], **kwargs_418152)
        
        
        # Call to append(...): (line 115)
        # Processing the call arguments (line 115)
        
        # Call to Case(...): (line 115)
        # Processing the call arguments (line 115)
        str_418158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 31), 'str', 'rand-cmplx-herm')
        
        # Call to astype(...): (line 115)
        # Processing the call arguments (line 115)
        str_418161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 62), 'str', 'F')
        # Processing the call keyword arguments (line 115)
        kwargs_418162 = {}
        # Getting the type of 'data' (line 115)
        data_418159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 50), 'data', False)
        # Obtaining the member 'astype' of a type (line 115)
        astype_418160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 50), data_418159, 'astype')
        # Calling astype(args, kwargs) (line 115)
        astype_call_result_418163 = invoke(stypy.reporting.localization.Localization(__file__, 115, 50), astype_418160, *[str_418161], **kwargs_418162)
        
        # Processing the call keyword arguments (line 115)
        # Getting the type of 'posdef_solvers' (line 116)
        posdef_solvers_418164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 36), 'posdef_solvers', False)
        # Getting the type of 'real_solvers' (line 116)
        real_solvers_418165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 51), 'real_solvers', False)
        # Applying the binary operator '+' (line 116)
        result_add_418166 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 36), '+', posdef_solvers_418164, real_solvers_418165)
        
        keyword_418167 = result_add_418166
        kwargs_418168 = {'skip': keyword_418167}
        # Getting the type of 'Case' (line 115)
        Case_418157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 26), 'Case', False)
        # Calling Case(args, kwargs) (line 115)
        Case_call_result_418169 = invoke(stypy.reporting.localization.Localization(__file__, 115, 26), Case_418157, *[str_418158, astype_call_result_418163], **kwargs_418168)
        
        # Processing the call keyword arguments (line 115)
        kwargs_418170 = {}
        # Getting the type of 'self' (line 115)
        self_418154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'self', False)
        # Obtaining the member 'cases' of a type (line 115)
        cases_418155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 8), self_418154, 'cases')
        # Obtaining the member 'append' of a type (line 115)
        append_418156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 8), cases_418155, 'append')
        # Calling append(args, kwargs) (line 115)
        append_call_result_418171 = invoke(stypy.reporting.localization.Localization(__file__, 115, 8), append_418156, *[Case_call_result_418169], **kwargs_418170)
        
        
        # Call to seed(...): (line 119)
        # Processing the call arguments (line 119)
        int_418175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 23), 'int')
        # Processing the call keyword arguments (line 119)
        kwargs_418176 = {}
        # Getting the type of 'np' (line 119)
        np_418172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 119)
        random_418173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), np_418172, 'random')
        # Obtaining the member 'seed' of a type (line 119)
        seed_418174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), random_418173, 'seed')
        # Calling seed(args, kwargs) (line 119)
        seed_call_result_418177 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), seed_418174, *[int_418175], **kwargs_418176)
        
        
        # Assigning a BinOp to a Name (line 120):
        
        # Assigning a BinOp to a Name (line 120):
        
        # Call to rand(...): (line 120)
        # Processing the call arguments (line 120)
        int_418181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 30), 'int')
        int_418182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 33), 'int')
        # Processing the call keyword arguments (line 120)
        kwargs_418183 = {}
        # Getting the type of 'np' (line 120)
        np_418178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'np', False)
        # Obtaining the member 'random' of a type (line 120)
        random_418179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 15), np_418178, 'random')
        # Obtaining the member 'rand' of a type (line 120)
        rand_418180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 15), random_418179, 'rand')
        # Calling rand(args, kwargs) (line 120)
        rand_call_result_418184 = invoke(stypy.reporting.localization.Localization(__file__, 120, 15), rand_418180, *[int_418181, int_418182], **kwargs_418183)
        
        complex_418185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 38), 'complex')
        
        # Call to rand(...): (line 120)
        # Processing the call arguments (line 120)
        int_418189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 56), 'int')
        int_418190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 59), 'int')
        # Processing the call keyword arguments (line 120)
        kwargs_418191 = {}
        # Getting the type of 'np' (line 120)
        np_418186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 41), 'np', False)
        # Obtaining the member 'random' of a type (line 120)
        random_418187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 41), np_418186, 'random')
        # Obtaining the member 'rand' of a type (line 120)
        rand_418188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 41), random_418187, 'rand')
        # Calling rand(args, kwargs) (line 120)
        rand_call_result_418192 = invoke(stypy.reporting.localization.Localization(__file__, 120, 41), rand_418188, *[int_418189, int_418190], **kwargs_418191)
        
        # Applying the binary operator '*' (line 120)
        result_mul_418193 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 38), '*', complex_418185, rand_call_result_418192)
        
        # Applying the binary operator '+' (line 120)
        result_add_418194 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 15), '+', rand_call_result_418184, result_mul_418193)
        
        # Assigning a type to the variable 'data' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'data', result_add_418194)
        
        # Assigning a Call to a Name (line 121):
        
        # Assigning a Call to a Name (line 121):
        
        # Call to dot(...): (line 121)
        # Processing the call arguments (line 121)
        
        # Call to conj(...): (line 121)
        # Processing the call keyword arguments (line 121)
        kwargs_418199 = {}
        # Getting the type of 'data' (line 121)
        data_418197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 22), 'data', False)
        # Obtaining the member 'conj' of a type (line 121)
        conj_418198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 22), data_418197, 'conj')
        # Calling conj(args, kwargs) (line 121)
        conj_call_result_418200 = invoke(stypy.reporting.localization.Localization(__file__, 121, 22), conj_418198, *[], **kwargs_418199)
        
        # Getting the type of 'data' (line 121)
        data_418201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 35), 'data', False)
        # Obtaining the member 'T' of a type (line 121)
        T_418202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 35), data_418201, 'T')
        # Processing the call keyword arguments (line 121)
        kwargs_418203 = {}
        # Getting the type of 'np' (line 121)
        np_418195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'np', False)
        # Obtaining the member 'dot' of a type (line 121)
        dot_418196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 15), np_418195, 'dot')
        # Calling dot(args, kwargs) (line 121)
        dot_call_result_418204 = invoke(stypy.reporting.localization.Localization(__file__, 121, 15), dot_418196, *[conj_call_result_418200, T_418202], **kwargs_418203)
        
        # Assigning a type to the variable 'data' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'data', dot_call_result_418204)
        
        # Call to append(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Call to Case(...): (line 122)
        # Processing the call arguments (line 122)
        str_418209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 31), 'str', 'rand-cmplx-sym-pd')
        # Getting the type of 'data' (line 122)
        data_418210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 52), 'data', False)
        # Processing the call keyword arguments (line 122)
        # Getting the type of 'real_solvers' (line 122)
        real_solvers_418211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 63), 'real_solvers', False)
        keyword_418212 = real_solvers_418211
        kwargs_418213 = {'skip': keyword_418212}
        # Getting the type of 'Case' (line 122)
        Case_418208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 26), 'Case', False)
        # Calling Case(args, kwargs) (line 122)
        Case_call_result_418214 = invoke(stypy.reporting.localization.Localization(__file__, 122, 26), Case_418208, *[str_418209, data_418210], **kwargs_418213)
        
        # Processing the call keyword arguments (line 122)
        kwargs_418215 = {}
        # Getting the type of 'self' (line 122)
        self_418205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'self', False)
        # Obtaining the member 'cases' of a type (line 122)
        cases_418206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), self_418205, 'cases')
        # Obtaining the member 'append' of a type (line 122)
        append_418207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), cases_418206, 'append')
        # Calling append(args, kwargs) (line 122)
        append_call_result_418216 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), append_418207, *[Case_call_result_418214], **kwargs_418215)
        
        
        # Call to append(...): (line 123)
        # Processing the call arguments (line 123)
        
        # Call to Case(...): (line 123)
        # Processing the call arguments (line 123)
        str_418221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 31), 'str', 'rand-cmplx-sym-pd')
        
        # Call to astype(...): (line 123)
        # Processing the call arguments (line 123)
        str_418224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 64), 'str', 'F')
        # Processing the call keyword arguments (line 123)
        kwargs_418225 = {}
        # Getting the type of 'data' (line 123)
        data_418222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 52), 'data', False)
        # Obtaining the member 'astype' of a type (line 123)
        astype_418223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 52), data_418222, 'astype')
        # Calling astype(args, kwargs) (line 123)
        astype_call_result_418226 = invoke(stypy.reporting.localization.Localization(__file__, 123, 52), astype_418223, *[str_418224], **kwargs_418225)
        
        # Processing the call keyword arguments (line 123)
        # Getting the type of 'real_solvers' (line 124)
        real_solvers_418227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 36), 'real_solvers', False)
        keyword_418228 = real_solvers_418227
        kwargs_418229 = {'skip': keyword_418228}
        # Getting the type of 'Case' (line 123)
        Case_418220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 26), 'Case', False)
        # Calling Case(args, kwargs) (line 123)
        Case_call_result_418230 = invoke(stypy.reporting.localization.Localization(__file__, 123, 26), Case_418220, *[str_418221, astype_call_result_418226], **kwargs_418229)
        
        # Processing the call keyword arguments (line 123)
        kwargs_418231 = {}
        # Getting the type of 'self' (line 123)
        self_418217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'self', False)
        # Obtaining the member 'cases' of a type (line 123)
        cases_418218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), self_418217, 'cases')
        # Obtaining the member 'append' of a type (line 123)
        append_418219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), cases_418218, 'append')
        # Calling append(args, kwargs) (line 123)
        append_call_result_418232 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), append_418219, *[Case_call_result_418230], **kwargs_418231)
        
        
        # Assigning a Call to a Name (line 130):
        
        # Assigning a Call to a Name (line 130):
        
        # Call to ones(...): (line 130)
        # Processing the call arguments (line 130)
        
        # Obtaining an instance of the builtin type 'tuple' (line 130)
        tuple_418234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 130)
        # Adding element type (line 130)
        int_418235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 21), tuple_418234, int_418235)
        # Adding element type (line 130)
        int_418236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 21), tuple_418234, int_418236)
        
        # Processing the call keyword arguments (line 130)
        kwargs_418237 = {}
        # Getting the type of 'ones' (line 130)
        ones_418233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 15), 'ones', False)
        # Calling ones(args, kwargs) (line 130)
        ones_call_result_418238 = invoke(stypy.reporting.localization.Localization(__file__, 130, 15), ones_418233, *[tuple_418234], **kwargs_418237)
        
        # Assigning a type to the variable 'data' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'data', ones_call_result_418238)
        
        # Assigning a Num to a Subscript (line 131):
        
        # Assigning a Num to a Subscript (line 131):
        int_418239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 20), 'int')
        # Getting the type of 'data' (line 131)
        data_418240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'data')
        int_418241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 13), 'int')
        slice_418242 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 131, 8), None, None, None)
        # Storing an element on a container (line 131)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 8), data_418240, ((int_418241, slice_418242), int_418239))
        
        # Assigning a Num to a Subscript (line 132):
        
        # Assigning a Num to a Subscript (line 132):
        int_418243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 20), 'int')
        # Getting the type of 'data' (line 132)
        data_418244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'data')
        int_418245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 13), 'int')
        slice_418246 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 132, 8), None, None, None)
        # Storing an element on a container (line 132)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 8), data_418244, ((int_418245, slice_418246), int_418243))
        
        # Assigning a Call to a Name (line 133):
        
        # Assigning a Call to a Name (line 133):
        
        # Call to spdiags(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'data' (line 133)
        data_418248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 20), 'data', False)
        
        # Obtaining an instance of the builtin type 'list' (line 133)
        list_418249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 133)
        # Adding element type (line 133)
        int_418250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 26), list_418249, int_418250)
        # Adding element type (line 133)
        int_418251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 26), list_418249, int_418251)
        
        int_418252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 34), 'int')
        int_418253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 38), 'int')
        # Processing the call keyword arguments (line 133)
        str_418254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 49), 'str', 'csr')
        keyword_418255 = str_418254
        kwargs_418256 = {'format': keyword_418255}
        # Getting the type of 'spdiags' (line 133)
        spdiags_418247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'spdiags', False)
        # Calling spdiags(args, kwargs) (line 133)
        spdiags_call_result_418257 = invoke(stypy.reporting.localization.Localization(__file__, 133, 12), spdiags_418247, *[data_418248, list_418249, int_418252, int_418253], **kwargs_418256)
        
        # Assigning a type to the variable 'A' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'A', spdiags_call_result_418257)
        
        # Call to append(...): (line 134)
        # Processing the call arguments (line 134)
        
        # Call to Case(...): (line 134)
        # Processing the call arguments (line 134)
        str_418262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 31), 'str', 'nonsymposdef')
        # Getting the type of 'A' (line 134)
        A_418263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 47), 'A', False)
        # Processing the call keyword arguments (line 134)
        # Getting the type of 'sym_solvers' (line 135)
        sym_solvers_418264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 36), 'sym_solvers', False)
        
        # Obtaining an instance of the builtin type 'list' (line 135)
        list_418265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 135)
        # Adding element type (line 135)
        # Getting the type of 'cgs' (line 135)
        cgs_418266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 49), 'cgs', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 48), list_418265, cgs_418266)
        # Adding element type (line 135)
        # Getting the type of 'qmr' (line 135)
        qmr_418267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 54), 'qmr', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 48), list_418265, qmr_418267)
        # Adding element type (line 135)
        # Getting the type of 'bicg' (line 135)
        bicg_418268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 59), 'bicg', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 48), list_418265, bicg_418268)
        
        # Applying the binary operator '+' (line 135)
        result_add_418269 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 36), '+', sym_solvers_418264, list_418265)
        
        keyword_418270 = result_add_418269
        kwargs_418271 = {'skip': keyword_418270}
        # Getting the type of 'Case' (line 134)
        Case_418261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 26), 'Case', False)
        # Calling Case(args, kwargs) (line 134)
        Case_call_result_418272 = invoke(stypy.reporting.localization.Localization(__file__, 134, 26), Case_418261, *[str_418262, A_418263], **kwargs_418271)
        
        # Processing the call keyword arguments (line 134)
        kwargs_418273 = {}
        # Getting the type of 'self' (line 134)
        self_418258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'self', False)
        # Obtaining the member 'cases' of a type (line 134)
        cases_418259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), self_418258, 'cases')
        # Obtaining the member 'append' of a type (line 134)
        append_418260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), cases_418259, 'append')
        # Calling append(args, kwargs) (line 134)
        append_call_result_418274 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), append_418260, *[Case_call_result_418272], **kwargs_418273)
        
        
        # Call to append(...): (line 136)
        # Processing the call arguments (line 136)
        
        # Call to Case(...): (line 136)
        # Processing the call arguments (line 136)
        str_418279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 31), 'str', 'nonsymposdef')
        
        # Call to astype(...): (line 136)
        # Processing the call arguments (line 136)
        str_418282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 56), 'str', 'F')
        # Processing the call keyword arguments (line 136)
        kwargs_418283 = {}
        # Getting the type of 'A' (line 136)
        A_418280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 47), 'A', False)
        # Obtaining the member 'astype' of a type (line 136)
        astype_418281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 47), A_418280, 'astype')
        # Calling astype(args, kwargs) (line 136)
        astype_call_result_418284 = invoke(stypy.reporting.localization.Localization(__file__, 136, 47), astype_418281, *[str_418282], **kwargs_418283)
        
        # Processing the call keyword arguments (line 136)
        # Getting the type of 'sym_solvers' (line 137)
        sym_solvers_418285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 36), 'sym_solvers', False)
        
        # Obtaining an instance of the builtin type 'list' (line 137)
        list_418286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 137)
        # Adding element type (line 137)
        # Getting the type of 'cgs' (line 137)
        cgs_418287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 49), 'cgs', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 48), list_418286, cgs_418287)
        # Adding element type (line 137)
        # Getting the type of 'qmr' (line 137)
        qmr_418288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 54), 'qmr', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 48), list_418286, qmr_418288)
        # Adding element type (line 137)
        # Getting the type of 'bicg' (line 137)
        bicg_418289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 59), 'bicg', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 48), list_418286, bicg_418289)
        
        # Applying the binary operator '+' (line 137)
        result_add_418290 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 36), '+', sym_solvers_418285, list_418286)
        
        keyword_418291 = result_add_418290
        kwargs_418292 = {'skip': keyword_418291}
        # Getting the type of 'Case' (line 136)
        Case_418278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 26), 'Case', False)
        # Calling Case(args, kwargs) (line 136)
        Case_call_result_418293 = invoke(stypy.reporting.localization.Localization(__file__, 136, 26), Case_418278, *[str_418279, astype_call_result_418284], **kwargs_418292)
        
        # Processing the call keyword arguments (line 136)
        kwargs_418294 = {}
        # Getting the type of 'self' (line 136)
        self_418275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'self', False)
        # Obtaining the member 'cases' of a type (line 136)
        cases_418276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), self_418275, 'cases')
        # Obtaining the member 'append' of a type (line 136)
        append_418277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), cases_418276, 'append')
        # Calling append(args, kwargs) (line 136)
        append_call_result_418295 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), append_418277, *[Case_call_result_418293], **kwargs_418294)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'IterativeParams' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'IterativeParams', IterativeParams)

# Assigning a Call to a Name (line 140):

# Assigning a Call to a Name (line 140):

# Call to IterativeParams(...): (line 140)
# Processing the call keyword arguments (line 140)
kwargs_418297 = {}
# Getting the type of 'IterativeParams' (line 140)
IterativeParams_418296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 9), 'IterativeParams', False)
# Calling IterativeParams(args, kwargs) (line 140)
IterativeParams_call_result_418298 = invoke(stypy.reporting.localization.Localization(__file__, 140, 9), IterativeParams_418296, *[], **kwargs_418297)

# Assigning a type to the variable 'params' (line 140)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 0), 'params', IterativeParams_call_result_418298)

@norecursion
def check_maxiter(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_maxiter'
    module_type_store = module_type_store.open_function_context('check_maxiter', 143, 0, False)
    
    # Passed parameters checking function
    check_maxiter.stypy_localization = localization
    check_maxiter.stypy_type_of_self = None
    check_maxiter.stypy_type_store = module_type_store
    check_maxiter.stypy_function_name = 'check_maxiter'
    check_maxiter.stypy_param_names_list = ['solver', 'case']
    check_maxiter.stypy_varargs_param_name = None
    check_maxiter.stypy_kwargs_param_name = None
    check_maxiter.stypy_call_defaults = defaults
    check_maxiter.stypy_call_varargs = varargs
    check_maxiter.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_maxiter', ['solver', 'case'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_maxiter', localization, ['solver', 'case'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_maxiter(...)' code ##################

    
    # Assigning a Attribute to a Name (line 144):
    
    # Assigning a Attribute to a Name (line 144):
    # Getting the type of 'case' (line 144)
    case_418299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'case')
    # Obtaining the member 'A' of a type (line 144)
    A_418300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), case_418299, 'A')
    # Assigning a type to the variable 'A' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'A', A_418300)
    
    # Assigning a Num to a Name (line 145):
    
    # Assigning a Num to a Name (line 145):
    float_418301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 10), 'float')
    # Assigning a type to the variable 'tol' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'tol', float_418301)
    
    # Assigning a Call to a Name (line 147):
    
    # Assigning a Call to a Name (line 147):
    
    # Call to arange(...): (line 147)
    # Processing the call arguments (line 147)
    
    # Obtaining the type of the subscript
    int_418303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 23), 'int')
    # Getting the type of 'A' (line 147)
    A_418304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 15), 'A', False)
    # Obtaining the member 'shape' of a type (line 147)
    shape_418305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 15), A_418304, 'shape')
    # Obtaining the member '__getitem__' of a type (line 147)
    getitem___418306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 15), shape_418305, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 147)
    subscript_call_result_418307 = invoke(stypy.reporting.localization.Localization(__file__, 147, 15), getitem___418306, int_418303)
    
    # Processing the call keyword arguments (line 147)
    # Getting the type of 'float' (line 147)
    float_418308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 33), 'float', False)
    keyword_418309 = float_418308
    kwargs_418310 = {'dtype': keyword_418309}
    # Getting the type of 'arange' (line 147)
    arange_418302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'arange', False)
    # Calling arange(args, kwargs) (line 147)
    arange_call_result_418311 = invoke(stypy.reporting.localization.Localization(__file__, 147, 8), arange_418302, *[subscript_call_result_418307], **kwargs_418310)
    
    # Assigning a type to the variable 'b' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'b', arange_call_result_418311)
    
    # Assigning a BinOp to a Name (line 148):
    
    # Assigning a BinOp to a Name (line 148):
    int_418312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 9), 'int')
    # Getting the type of 'b' (line 148)
    b_418313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 11), 'b')
    # Applying the binary operator '*' (line 148)
    result_mul_418314 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 9), '*', int_418312, b_418313)
    
    # Assigning a type to the variable 'x0' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'x0', result_mul_418314)
    
    # Assigning a List to a Name (line 150):
    
    # Assigning a List to a Name (line 150):
    
    # Obtaining an instance of the builtin type 'list' (line 150)
    list_418315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 150)
    
    # Assigning a type to the variable 'residuals' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'residuals', list_418315)

    @norecursion
    def callback(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'callback'
        module_type_store = module_type_store.open_function_context('callback', 152, 4, False)
        
        # Passed parameters checking function
        callback.stypy_localization = localization
        callback.stypy_type_of_self = None
        callback.stypy_type_store = module_type_store
        callback.stypy_function_name = 'callback'
        callback.stypy_param_names_list = ['x']
        callback.stypy_varargs_param_name = None
        callback.stypy_kwargs_param_name = None
        callback.stypy_call_defaults = defaults
        callback.stypy_call_varargs = varargs
        callback.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'callback', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'callback', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'callback(...)' code ##################

        
        # Call to append(...): (line 153)
        # Processing the call arguments (line 153)
        
        # Call to norm(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'b' (line 153)
        b_418319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 30), 'b', False)
        # Getting the type of 'case' (line 153)
        case_418320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 34), 'case', False)
        # Obtaining the member 'A' of a type (line 153)
        A_418321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 34), case_418320, 'A')
        # Getting the type of 'x' (line 153)
        x_418322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 41), 'x', False)
        # Applying the binary operator '*' (line 153)
        result_mul_418323 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 34), '*', A_418321, x_418322)
        
        # Applying the binary operator '-' (line 153)
        result_sub_418324 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 30), '-', b_418319, result_mul_418323)
        
        # Processing the call keyword arguments (line 153)
        kwargs_418325 = {}
        # Getting the type of 'norm' (line 153)
        norm_418318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 25), 'norm', False)
        # Calling norm(args, kwargs) (line 153)
        norm_call_result_418326 = invoke(stypy.reporting.localization.Localization(__file__, 153, 25), norm_418318, *[result_sub_418324], **kwargs_418325)
        
        # Processing the call keyword arguments (line 153)
        kwargs_418327 = {}
        # Getting the type of 'residuals' (line 153)
        residuals_418316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'residuals', False)
        # Obtaining the member 'append' of a type (line 153)
        append_418317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), residuals_418316, 'append')
        # Calling append(args, kwargs) (line 153)
        append_call_result_418328 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), append_418317, *[norm_call_result_418326], **kwargs_418327)
        
        
        # ################# End of 'callback(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'callback' in the type store
        # Getting the type of 'stypy_return_type' (line 152)
        stypy_return_type_418329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_418329)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'callback'
        return stypy_return_type_418329

    # Assigning a type to the variable 'callback' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'callback', callback)
    
    # Assigning a Call to a Tuple (line 155):
    
    # Assigning a Subscript to a Name (line 155):
    
    # Obtaining the type of the subscript
    int_418330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 4), 'int')
    
    # Call to solver(...): (line 155)
    # Processing the call arguments (line 155)
    # Getting the type of 'A' (line 155)
    A_418332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 21), 'A', False)
    # Getting the type of 'b' (line 155)
    b_418333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 24), 'b', False)
    # Processing the call keyword arguments (line 155)
    # Getting the type of 'x0' (line 155)
    x0_418334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 30), 'x0', False)
    keyword_418335 = x0_418334
    # Getting the type of 'tol' (line 155)
    tol_418336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 38), 'tol', False)
    keyword_418337 = tol_418336
    int_418338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 51), 'int')
    keyword_418339 = int_418338
    # Getting the type of 'callback' (line 155)
    callback_418340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 63), 'callback', False)
    keyword_418341 = callback_418340
    kwargs_418342 = {'callback': keyword_418341, 'x0': keyword_418335, 'tol': keyword_418337, 'maxiter': keyword_418339}
    # Getting the type of 'solver' (line 155)
    solver_418331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 14), 'solver', False)
    # Calling solver(args, kwargs) (line 155)
    solver_call_result_418343 = invoke(stypy.reporting.localization.Localization(__file__, 155, 14), solver_418331, *[A_418332, b_418333], **kwargs_418342)
    
    # Obtaining the member '__getitem__' of a type (line 155)
    getitem___418344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 4), solver_call_result_418343, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 155)
    subscript_call_result_418345 = invoke(stypy.reporting.localization.Localization(__file__, 155, 4), getitem___418344, int_418330)
    
    # Assigning a type to the variable 'tuple_var_assignment_417673' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'tuple_var_assignment_417673', subscript_call_result_418345)
    
    # Assigning a Subscript to a Name (line 155):
    
    # Obtaining the type of the subscript
    int_418346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 4), 'int')
    
    # Call to solver(...): (line 155)
    # Processing the call arguments (line 155)
    # Getting the type of 'A' (line 155)
    A_418348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 21), 'A', False)
    # Getting the type of 'b' (line 155)
    b_418349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 24), 'b', False)
    # Processing the call keyword arguments (line 155)
    # Getting the type of 'x0' (line 155)
    x0_418350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 30), 'x0', False)
    keyword_418351 = x0_418350
    # Getting the type of 'tol' (line 155)
    tol_418352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 38), 'tol', False)
    keyword_418353 = tol_418352
    int_418354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 51), 'int')
    keyword_418355 = int_418354
    # Getting the type of 'callback' (line 155)
    callback_418356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 63), 'callback', False)
    keyword_418357 = callback_418356
    kwargs_418358 = {'callback': keyword_418357, 'x0': keyword_418351, 'tol': keyword_418353, 'maxiter': keyword_418355}
    # Getting the type of 'solver' (line 155)
    solver_418347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 14), 'solver', False)
    # Calling solver(args, kwargs) (line 155)
    solver_call_result_418359 = invoke(stypy.reporting.localization.Localization(__file__, 155, 14), solver_418347, *[A_418348, b_418349], **kwargs_418358)
    
    # Obtaining the member '__getitem__' of a type (line 155)
    getitem___418360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 4), solver_call_result_418359, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 155)
    subscript_call_result_418361 = invoke(stypy.reporting.localization.Localization(__file__, 155, 4), getitem___418360, int_418346)
    
    # Assigning a type to the variable 'tuple_var_assignment_417674' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'tuple_var_assignment_417674', subscript_call_result_418361)
    
    # Assigning a Name to a Name (line 155):
    # Getting the type of 'tuple_var_assignment_417673' (line 155)
    tuple_var_assignment_417673_418362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'tuple_var_assignment_417673')
    # Assigning a type to the variable 'x' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'x', tuple_var_assignment_417673_418362)
    
    # Assigning a Name to a Name (line 155):
    # Getting the type of 'tuple_var_assignment_417674' (line 155)
    tuple_var_assignment_417674_418363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'tuple_var_assignment_417674')
    # Assigning a type to the variable 'info' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 7), 'info', tuple_var_assignment_417674_418363)
    
    # Call to assert_equal(...): (line 157)
    # Processing the call arguments (line 157)
    
    # Call to len(...): (line 157)
    # Processing the call arguments (line 157)
    # Getting the type of 'residuals' (line 157)
    residuals_418366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 21), 'residuals', False)
    # Processing the call keyword arguments (line 157)
    kwargs_418367 = {}
    # Getting the type of 'len' (line 157)
    len_418365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 17), 'len', False)
    # Calling len(args, kwargs) (line 157)
    len_call_result_418368 = invoke(stypy.reporting.localization.Localization(__file__, 157, 17), len_418365, *[residuals_418366], **kwargs_418367)
    
    int_418369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 33), 'int')
    # Processing the call keyword arguments (line 157)
    kwargs_418370 = {}
    # Getting the type of 'assert_equal' (line 157)
    assert_equal_418364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 157)
    assert_equal_call_result_418371 = invoke(stypy.reporting.localization.Localization(__file__, 157, 4), assert_equal_418364, *[len_call_result_418368, int_418369], **kwargs_418370)
    
    
    # Call to assert_equal(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'info' (line 158)
    info_418373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 17), 'info', False)
    int_418374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 23), 'int')
    # Processing the call keyword arguments (line 158)
    kwargs_418375 = {}
    # Getting the type of 'assert_equal' (line 158)
    assert_equal_418372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 158)
    assert_equal_call_result_418376 = invoke(stypy.reporting.localization.Localization(__file__, 158, 4), assert_equal_418372, *[info_418373, int_418374], **kwargs_418375)
    
    
    # ################# End of 'check_maxiter(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_maxiter' in the type store
    # Getting the type of 'stypy_return_type' (line 143)
    stypy_return_type_418377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_418377)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_maxiter'
    return stypy_return_type_418377

# Assigning a type to the variable 'check_maxiter' (line 143)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 0), 'check_maxiter', check_maxiter)

@norecursion
def test_maxiter(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_maxiter'
    module_type_store = module_type_store.open_function_context('test_maxiter', 161, 0, False)
    
    # Passed parameters checking function
    test_maxiter.stypy_localization = localization
    test_maxiter.stypy_type_of_self = None
    test_maxiter.stypy_type_store = module_type_store
    test_maxiter.stypy_function_name = 'test_maxiter'
    test_maxiter.stypy_param_names_list = []
    test_maxiter.stypy_varargs_param_name = None
    test_maxiter.stypy_kwargs_param_name = None
    test_maxiter.stypy_call_defaults = defaults
    test_maxiter.stypy_call_varargs = varargs
    test_maxiter.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_maxiter', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_maxiter', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_maxiter(...)' code ##################

    
    # Assigning a Attribute to a Name (line 162):
    
    # Assigning a Attribute to a Name (line 162):
    # Getting the type of 'params' (line 162)
    params_418378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 11), 'params')
    # Obtaining the member 'Poisson1D' of a type (line 162)
    Poisson1D_418379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 11), params_418378, 'Poisson1D')
    # Assigning a type to the variable 'case' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'case', Poisson1D_418379)
    
    # Getting the type of 'params' (line 163)
    params_418380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 18), 'params')
    # Obtaining the member 'solvers' of a type (line 163)
    solvers_418381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 18), params_418380, 'solvers')
    # Testing the type of a for loop iterable (line 163)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 163, 4), solvers_418381)
    # Getting the type of the for loop variable (line 163)
    for_loop_var_418382 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 163, 4), solvers_418381)
    # Assigning a type to the variable 'solver' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'solver', for_loop_var_418382)
    # SSA begins for a for statement (line 163)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'solver' (line 164)
    solver_418383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 11), 'solver')
    # Getting the type of 'case' (line 164)
    case_418384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 21), 'case')
    # Obtaining the member 'skip' of a type (line 164)
    skip_418385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 21), case_418384, 'skip')
    # Applying the binary operator 'in' (line 164)
    result_contains_418386 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 11), 'in', solver_418383, skip_418385)
    
    # Testing the type of an if condition (line 164)
    if_condition_418387 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 8), result_contains_418386)
    # Assigning a type to the variable 'if_condition_418387' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'if_condition_418387', if_condition_418387)
    # SSA begins for if statement (line 164)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 164)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to check_maxiter(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'solver' (line 166)
    solver_418389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 22), 'solver', False)
    # Getting the type of 'case' (line 166)
    case_418390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 30), 'case', False)
    # Processing the call keyword arguments (line 166)
    kwargs_418391 = {}
    # Getting the type of 'check_maxiter' (line 166)
    check_maxiter_418388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'check_maxiter', False)
    # Calling check_maxiter(args, kwargs) (line 166)
    check_maxiter_call_result_418392 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), check_maxiter_418388, *[solver_418389, case_418390], **kwargs_418391)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_maxiter(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_maxiter' in the type store
    # Getting the type of 'stypy_return_type' (line 161)
    stypy_return_type_418393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_418393)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_maxiter'
    return stypy_return_type_418393

# Assigning a type to the variable 'test_maxiter' (line 161)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'test_maxiter', test_maxiter)

@norecursion
def assert_normclose(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_418394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 31), 'float')
    defaults = [float_418394]
    # Create a new context for function 'assert_normclose'
    module_type_store = module_type_store.open_function_context('assert_normclose', 169, 0, False)
    
    # Passed parameters checking function
    assert_normclose.stypy_localization = localization
    assert_normclose.stypy_type_of_self = None
    assert_normclose.stypy_type_store = module_type_store
    assert_normclose.stypy_function_name = 'assert_normclose'
    assert_normclose.stypy_param_names_list = ['a', 'b', 'tol']
    assert_normclose.stypy_varargs_param_name = None
    assert_normclose.stypy_kwargs_param_name = None
    assert_normclose.stypy_call_defaults = defaults
    assert_normclose.stypy_call_varargs = varargs
    assert_normclose.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_normclose', ['a', 'b', 'tol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_normclose', localization, ['a', 'b', 'tol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_normclose(...)' code ##################

    
    # Assigning a Call to a Name (line 170):
    
    # Assigning a Call to a Name (line 170):
    
    # Call to norm(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'a' (line 170)
    a_418396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 20), 'a', False)
    # Getting the type of 'b' (line 170)
    b_418397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 24), 'b', False)
    # Applying the binary operator '-' (line 170)
    result_sub_418398 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 20), '-', a_418396, b_418397)
    
    # Processing the call keyword arguments (line 170)
    kwargs_418399 = {}
    # Getting the type of 'norm' (line 170)
    norm_418395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 15), 'norm', False)
    # Calling norm(args, kwargs) (line 170)
    norm_call_result_418400 = invoke(stypy.reporting.localization.Localization(__file__, 170, 15), norm_418395, *[result_sub_418398], **kwargs_418399)
    
    # Assigning a type to the variable 'residual' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'residual', norm_call_result_418400)
    
    # Assigning a BinOp to a Name (line 171):
    
    # Assigning a BinOp to a Name (line 171):
    # Getting the type of 'tol' (line 171)
    tol_418401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'tol')
    
    # Call to norm(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'b' (line 171)
    b_418403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 25), 'b', False)
    # Processing the call keyword arguments (line 171)
    kwargs_418404 = {}
    # Getting the type of 'norm' (line 171)
    norm_418402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 20), 'norm', False)
    # Calling norm(args, kwargs) (line 171)
    norm_call_result_418405 = invoke(stypy.reporting.localization.Localization(__file__, 171, 20), norm_418402, *[b_418403], **kwargs_418404)
    
    # Applying the binary operator '*' (line 171)
    result_mul_418406 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 16), '*', tol_418401, norm_call_result_418405)
    
    # Assigning a type to the variable 'tolerance' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'tolerance', result_mul_418406)
    
    # Assigning a BinOp to a Name (line 172):
    
    # Assigning a BinOp to a Name (line 172):
    str_418407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 10), 'str', 'residual (%g) not smaller than tolerance %g')
    
    # Obtaining an instance of the builtin type 'tuple' (line 172)
    tuple_418408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 59), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 172)
    # Adding element type (line 172)
    # Getting the type of 'residual' (line 172)
    residual_418409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 59), 'residual')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 59), tuple_418408, residual_418409)
    # Adding element type (line 172)
    # Getting the type of 'tolerance' (line 172)
    tolerance_418410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 69), 'tolerance')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 59), tuple_418408, tolerance_418410)
    
    # Applying the binary operator '%' (line 172)
    result_mod_418411 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 10), '%', str_418407, tuple_418408)
    
    # Assigning a type to the variable 'msg' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'msg', result_mod_418411)
    
    # Call to assert_(...): (line 173)
    # Processing the call arguments (line 173)
    
    # Getting the type of 'residual' (line 173)
    residual_418413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'residual', False)
    # Getting the type of 'tolerance' (line 173)
    tolerance_418414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 23), 'tolerance', False)
    # Applying the binary operator '<' (line 173)
    result_lt_418415 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 12), '<', residual_418413, tolerance_418414)
    
    # Processing the call keyword arguments (line 173)
    # Getting the type of 'msg' (line 173)
    msg_418416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 38), 'msg', False)
    keyword_418417 = msg_418416
    kwargs_418418 = {'msg': keyword_418417}
    # Getting the type of 'assert_' (line 173)
    assert__418412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 173)
    assert__call_result_418419 = invoke(stypy.reporting.localization.Localization(__file__, 173, 4), assert__418412, *[result_lt_418415], **kwargs_418418)
    
    
    # ################# End of 'assert_normclose(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_normclose' in the type store
    # Getting the type of 'stypy_return_type' (line 169)
    stypy_return_type_418420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_418420)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_normclose'
    return stypy_return_type_418420

# Assigning a type to the variable 'assert_normclose' (line 169)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 0), 'assert_normclose', assert_normclose)

@norecursion
def check_convergence(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_convergence'
    module_type_store = module_type_store.open_function_context('check_convergence', 176, 0, False)
    
    # Passed parameters checking function
    check_convergence.stypy_localization = localization
    check_convergence.stypy_type_of_self = None
    check_convergence.stypy_type_store = module_type_store
    check_convergence.stypy_function_name = 'check_convergence'
    check_convergence.stypy_param_names_list = ['solver', 'case']
    check_convergence.stypy_varargs_param_name = None
    check_convergence.stypy_kwargs_param_name = None
    check_convergence.stypy_call_defaults = defaults
    check_convergence.stypy_call_varargs = varargs
    check_convergence.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_convergence', ['solver', 'case'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_convergence', localization, ['solver', 'case'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_convergence(...)' code ##################

    
    # Assigning a Attribute to a Name (line 177):
    
    # Assigning a Attribute to a Name (line 177):
    # Getting the type of 'case' (line 177)
    case_418421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'case')
    # Obtaining the member 'A' of a type (line 177)
    A_418422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), case_418421, 'A')
    # Assigning a type to the variable 'A' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'A', A_418422)
    
    
    # Getting the type of 'A' (line 179)
    A_418423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 7), 'A')
    # Obtaining the member 'dtype' of a type (line 179)
    dtype_418424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 7), A_418423, 'dtype')
    # Obtaining the member 'char' of a type (line 179)
    char_418425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 7), dtype_418424, 'char')
    str_418426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 23), 'str', 'dD')
    # Applying the binary operator 'in' (line 179)
    result_contains_418427 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 7), 'in', char_418425, str_418426)
    
    # Testing the type of an if condition (line 179)
    if_condition_418428 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 179, 4), result_contains_418427)
    # Assigning a type to the variable 'if_condition_418428' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'if_condition_418428', if_condition_418428)
    # SSA begins for if statement (line 179)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 180):
    
    # Assigning a Num to a Name (line 180):
    float_418429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 14), 'float')
    # Assigning a type to the variable 'tol' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'tol', float_418429)
    # SSA branch for the else part of an if statement (line 179)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 182):
    
    # Assigning a Num to a Name (line 182):
    float_418430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 14), 'float')
    # Assigning a type to the variable 'tol' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'tol', float_418430)
    # SSA join for if statement (line 179)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 184):
    
    # Assigning a Call to a Name (line 184):
    
    # Call to arange(...): (line 184)
    # Processing the call arguments (line 184)
    
    # Obtaining the type of the subscript
    int_418432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 23), 'int')
    # Getting the type of 'A' (line 184)
    A_418433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 15), 'A', False)
    # Obtaining the member 'shape' of a type (line 184)
    shape_418434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 15), A_418433, 'shape')
    # Obtaining the member '__getitem__' of a type (line 184)
    getitem___418435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 15), shape_418434, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 184)
    subscript_call_result_418436 = invoke(stypy.reporting.localization.Localization(__file__, 184, 15), getitem___418435, int_418432)
    
    # Processing the call keyword arguments (line 184)
    # Getting the type of 'A' (line 184)
    A_418437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 33), 'A', False)
    # Obtaining the member 'dtype' of a type (line 184)
    dtype_418438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 33), A_418437, 'dtype')
    keyword_418439 = dtype_418438
    kwargs_418440 = {'dtype': keyword_418439}
    # Getting the type of 'arange' (line 184)
    arange_418431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'arange', False)
    # Calling arange(args, kwargs) (line 184)
    arange_call_result_418441 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), arange_418431, *[subscript_call_result_418436], **kwargs_418440)
    
    # Assigning a type to the variable 'b' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'b', arange_call_result_418441)
    
    # Assigning a BinOp to a Name (line 185):
    
    # Assigning a BinOp to a Name (line 185):
    int_418442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 9), 'int')
    # Getting the type of 'b' (line 185)
    b_418443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 11), 'b')
    # Applying the binary operator '*' (line 185)
    result_mul_418444 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 9), '*', int_418442, b_418443)
    
    # Assigning a type to the variable 'x0' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'x0', result_mul_418444)
    
    # Assigning a Call to a Tuple (line 187):
    
    # Assigning a Subscript to a Name (line 187):
    
    # Obtaining the type of the subscript
    int_418445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 4), 'int')
    
    # Call to solver(...): (line 187)
    # Processing the call arguments (line 187)
    # Getting the type of 'A' (line 187)
    A_418447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 21), 'A', False)
    # Getting the type of 'b' (line 187)
    b_418448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 24), 'b', False)
    # Processing the call keyword arguments (line 187)
    # Getting the type of 'x0' (line 187)
    x0_418449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 30), 'x0', False)
    keyword_418450 = x0_418449
    # Getting the type of 'tol' (line 187)
    tol_418451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 38), 'tol', False)
    keyword_418452 = tol_418451
    kwargs_418453 = {'x0': keyword_418450, 'tol': keyword_418452}
    # Getting the type of 'solver' (line 187)
    solver_418446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 14), 'solver', False)
    # Calling solver(args, kwargs) (line 187)
    solver_call_result_418454 = invoke(stypy.reporting.localization.Localization(__file__, 187, 14), solver_418446, *[A_418447, b_418448], **kwargs_418453)
    
    # Obtaining the member '__getitem__' of a type (line 187)
    getitem___418455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 4), solver_call_result_418454, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 187)
    subscript_call_result_418456 = invoke(stypy.reporting.localization.Localization(__file__, 187, 4), getitem___418455, int_418445)
    
    # Assigning a type to the variable 'tuple_var_assignment_417675' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'tuple_var_assignment_417675', subscript_call_result_418456)
    
    # Assigning a Subscript to a Name (line 187):
    
    # Obtaining the type of the subscript
    int_418457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 4), 'int')
    
    # Call to solver(...): (line 187)
    # Processing the call arguments (line 187)
    # Getting the type of 'A' (line 187)
    A_418459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 21), 'A', False)
    # Getting the type of 'b' (line 187)
    b_418460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 24), 'b', False)
    # Processing the call keyword arguments (line 187)
    # Getting the type of 'x0' (line 187)
    x0_418461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 30), 'x0', False)
    keyword_418462 = x0_418461
    # Getting the type of 'tol' (line 187)
    tol_418463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 38), 'tol', False)
    keyword_418464 = tol_418463
    kwargs_418465 = {'x0': keyword_418462, 'tol': keyword_418464}
    # Getting the type of 'solver' (line 187)
    solver_418458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 14), 'solver', False)
    # Calling solver(args, kwargs) (line 187)
    solver_call_result_418466 = invoke(stypy.reporting.localization.Localization(__file__, 187, 14), solver_418458, *[A_418459, b_418460], **kwargs_418465)
    
    # Obtaining the member '__getitem__' of a type (line 187)
    getitem___418467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 4), solver_call_result_418466, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 187)
    subscript_call_result_418468 = invoke(stypy.reporting.localization.Localization(__file__, 187, 4), getitem___418467, int_418457)
    
    # Assigning a type to the variable 'tuple_var_assignment_417676' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'tuple_var_assignment_417676', subscript_call_result_418468)
    
    # Assigning a Name to a Name (line 187):
    # Getting the type of 'tuple_var_assignment_417675' (line 187)
    tuple_var_assignment_417675_418469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'tuple_var_assignment_417675')
    # Assigning a type to the variable 'x' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'x', tuple_var_assignment_417675_418469)
    
    # Assigning a Name to a Name (line 187):
    # Getting the type of 'tuple_var_assignment_417676' (line 187)
    tuple_var_assignment_417676_418470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'tuple_var_assignment_417676')
    # Assigning a type to the variable 'info' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 7), 'info', tuple_var_assignment_417676_418470)
    
    # Call to assert_array_equal(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of 'x0' (line 189)
    x0_418472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 23), 'x0', False)
    int_418473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 27), 'int')
    # Getting the type of 'b' (line 189)
    b_418474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 29), 'b', False)
    # Applying the binary operator '*' (line 189)
    result_mul_418475 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 27), '*', int_418473, b_418474)
    
    # Processing the call keyword arguments (line 189)
    kwargs_418476 = {}
    # Getting the type of 'assert_array_equal' (line 189)
    assert_array_equal_418471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 189)
    assert_array_equal_call_result_418477 = invoke(stypy.reporting.localization.Localization(__file__, 189, 4), assert_array_equal_418471, *[x0_418472, result_mul_418475], **kwargs_418476)
    
    
    # Call to assert_equal(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'info' (line 190)
    info_418479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 17), 'info', False)
    int_418480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 22), 'int')
    # Processing the call keyword arguments (line 190)
    kwargs_418481 = {}
    # Getting the type of 'assert_equal' (line 190)
    assert_equal_418478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 190)
    assert_equal_call_result_418482 = invoke(stypy.reporting.localization.Localization(__file__, 190, 4), assert_equal_418478, *[info_418479, int_418480], **kwargs_418481)
    
    
    # Call to assert_normclose(...): (line 191)
    # Processing the call arguments (line 191)
    
    # Call to dot(...): (line 191)
    # Processing the call arguments (line 191)
    # Getting the type of 'x' (line 191)
    x_418486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 27), 'x', False)
    # Processing the call keyword arguments (line 191)
    kwargs_418487 = {}
    # Getting the type of 'A' (line 191)
    A_418484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 21), 'A', False)
    # Obtaining the member 'dot' of a type (line 191)
    dot_418485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 21), A_418484, 'dot')
    # Calling dot(args, kwargs) (line 191)
    dot_call_result_418488 = invoke(stypy.reporting.localization.Localization(__file__, 191, 21), dot_418485, *[x_418486], **kwargs_418487)
    
    # Getting the type of 'b' (line 191)
    b_418489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 31), 'b', False)
    # Processing the call keyword arguments (line 191)
    # Getting the type of 'tol' (line 191)
    tol_418490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 38), 'tol', False)
    keyword_418491 = tol_418490
    kwargs_418492 = {'tol': keyword_418491}
    # Getting the type of 'assert_normclose' (line 191)
    assert_normclose_418483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'assert_normclose', False)
    # Calling assert_normclose(args, kwargs) (line 191)
    assert_normclose_call_result_418493 = invoke(stypy.reporting.localization.Localization(__file__, 191, 4), assert_normclose_418483, *[dot_call_result_418488, b_418489], **kwargs_418492)
    
    
    # ################# End of 'check_convergence(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_convergence' in the type store
    # Getting the type of 'stypy_return_type' (line 176)
    stypy_return_type_418494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_418494)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_convergence'
    return stypy_return_type_418494

# Assigning a type to the variable 'check_convergence' (line 176)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'check_convergence', check_convergence)

@norecursion
def test_convergence(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_convergence'
    module_type_store = module_type_store.open_function_context('test_convergence', 194, 0, False)
    
    # Passed parameters checking function
    test_convergence.stypy_localization = localization
    test_convergence.stypy_type_of_self = None
    test_convergence.stypy_type_store = module_type_store
    test_convergence.stypy_function_name = 'test_convergence'
    test_convergence.stypy_param_names_list = []
    test_convergence.stypy_varargs_param_name = None
    test_convergence.stypy_kwargs_param_name = None
    test_convergence.stypy_call_defaults = defaults
    test_convergence.stypy_call_varargs = varargs
    test_convergence.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_convergence', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_convergence', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_convergence(...)' code ##################

    
    # Getting the type of 'params' (line 195)
    params_418495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 18), 'params')
    # Obtaining the member 'solvers' of a type (line 195)
    solvers_418496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 18), params_418495, 'solvers')
    # Testing the type of a for loop iterable (line 195)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 195, 4), solvers_418496)
    # Getting the type of the for loop variable (line 195)
    for_loop_var_418497 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 195, 4), solvers_418496)
    # Assigning a type to the variable 'solver' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'solver', for_loop_var_418497)
    # SSA begins for a for statement (line 195)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'params' (line 196)
    params_418498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 20), 'params')
    # Obtaining the member 'cases' of a type (line 196)
    cases_418499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 20), params_418498, 'cases')
    # Testing the type of a for loop iterable (line 196)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 196, 8), cases_418499)
    # Getting the type of the for loop variable (line 196)
    for_loop_var_418500 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 196, 8), cases_418499)
    # Assigning a type to the variable 'case' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'case', for_loop_var_418500)
    # SSA begins for a for statement (line 196)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'solver' (line 197)
    solver_418501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 15), 'solver')
    # Getting the type of 'case' (line 197)
    case_418502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 25), 'case')
    # Obtaining the member 'skip' of a type (line 197)
    skip_418503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 25), case_418502, 'skip')
    # Applying the binary operator 'in' (line 197)
    result_contains_418504 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 15), 'in', solver_418501, skip_418503)
    
    # Testing the type of an if condition (line 197)
    if_condition_418505 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 12), result_contains_418504)
    # Assigning a type to the variable 'if_condition_418505' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'if_condition_418505', if_condition_418505)
    # SSA begins for if statement (line 197)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 197)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to check_convergence(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'solver' (line 199)
    solver_418507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 30), 'solver', False)
    # Getting the type of 'case' (line 199)
    case_418508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 38), 'case', False)
    # Processing the call keyword arguments (line 199)
    kwargs_418509 = {}
    # Getting the type of 'check_convergence' (line 199)
    check_convergence_418506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'check_convergence', False)
    # Calling check_convergence(args, kwargs) (line 199)
    check_convergence_call_result_418510 = invoke(stypy.reporting.localization.Localization(__file__, 199, 12), check_convergence_418506, *[solver_418507, case_418508], **kwargs_418509)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_convergence(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_convergence' in the type store
    # Getting the type of 'stypy_return_type' (line 194)
    stypy_return_type_418511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_418511)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_convergence'
    return stypy_return_type_418511

# Assigning a type to the variable 'test_convergence' (line 194)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 0), 'test_convergence', test_convergence)

@norecursion
def check_precond_dummy(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_precond_dummy'
    module_type_store = module_type_store.open_function_context('check_precond_dummy', 202, 0, False)
    
    # Passed parameters checking function
    check_precond_dummy.stypy_localization = localization
    check_precond_dummy.stypy_type_of_self = None
    check_precond_dummy.stypy_type_store = module_type_store
    check_precond_dummy.stypy_function_name = 'check_precond_dummy'
    check_precond_dummy.stypy_param_names_list = ['solver', 'case']
    check_precond_dummy.stypy_varargs_param_name = None
    check_precond_dummy.stypy_kwargs_param_name = None
    check_precond_dummy.stypy_call_defaults = defaults
    check_precond_dummy.stypy_call_varargs = varargs
    check_precond_dummy.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_precond_dummy', ['solver', 'case'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_precond_dummy', localization, ['solver', 'case'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_precond_dummy(...)' code ##################

    
    # Assigning a Num to a Name (line 203):
    
    # Assigning a Num to a Name (line 203):
    float_418512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 10), 'float')
    # Assigning a type to the variable 'tol' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'tol', float_418512)

    @norecursion
    def identity(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 205)
        None_418513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 25), 'None')
        defaults = [None_418513]
        # Create a new context for function 'identity'
        module_type_store = module_type_store.open_function_context('identity', 205, 4, False)
        
        # Passed parameters checking function
        identity.stypy_localization = localization
        identity.stypy_type_of_self = None
        identity.stypy_type_store = module_type_store
        identity.stypy_function_name = 'identity'
        identity.stypy_param_names_list = ['b', 'which']
        identity.stypy_varargs_param_name = None
        identity.stypy_kwargs_param_name = None
        identity.stypy_call_defaults = defaults
        identity.stypy_call_varargs = varargs
        identity.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'identity', ['b', 'which'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'identity', localization, ['b', 'which'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'identity(...)' code ##################

        str_418514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 8), 'str', 'trivial preconditioner')
        # Getting the type of 'b' (line 207)
        b_418515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 15), 'b')
        # Assigning a type to the variable 'stypy_return_type' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'stypy_return_type', b_418515)
        
        # ################# End of 'identity(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'identity' in the type store
        # Getting the type of 'stypy_return_type' (line 205)
        stypy_return_type_418516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_418516)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'identity'
        return stypy_return_type_418516

    # Assigning a type to the variable 'identity' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'identity', identity)
    
    # Assigning a Attribute to a Name (line 209):
    
    # Assigning a Attribute to a Name (line 209):
    # Getting the type of 'case' (line 209)
    case_418517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'case')
    # Obtaining the member 'A' of a type (line 209)
    A_418518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 8), case_418517, 'A')
    # Assigning a type to the variable 'A' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'A', A_418518)
    
    # Assigning a Attribute to a Tuple (line 211):
    
    # Assigning a Subscript to a Name (line 211):
    
    # Obtaining the type of the subscript
    int_418519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 4), 'int')
    # Getting the type of 'A' (line 211)
    A_418520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 10), 'A')
    # Obtaining the member 'shape' of a type (line 211)
    shape_418521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 10), A_418520, 'shape')
    # Obtaining the member '__getitem__' of a type (line 211)
    getitem___418522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 4), shape_418521, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 211)
    subscript_call_result_418523 = invoke(stypy.reporting.localization.Localization(__file__, 211, 4), getitem___418522, int_418519)
    
    # Assigning a type to the variable 'tuple_var_assignment_417677' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'tuple_var_assignment_417677', subscript_call_result_418523)
    
    # Assigning a Subscript to a Name (line 211):
    
    # Obtaining the type of the subscript
    int_418524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 4), 'int')
    # Getting the type of 'A' (line 211)
    A_418525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 10), 'A')
    # Obtaining the member 'shape' of a type (line 211)
    shape_418526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 10), A_418525, 'shape')
    # Obtaining the member '__getitem__' of a type (line 211)
    getitem___418527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 4), shape_418526, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 211)
    subscript_call_result_418528 = invoke(stypy.reporting.localization.Localization(__file__, 211, 4), getitem___418527, int_418524)
    
    # Assigning a type to the variable 'tuple_var_assignment_417678' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'tuple_var_assignment_417678', subscript_call_result_418528)
    
    # Assigning a Name to a Name (line 211):
    # Getting the type of 'tuple_var_assignment_417677' (line 211)
    tuple_var_assignment_417677_418529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'tuple_var_assignment_417677')
    # Assigning a type to the variable 'M' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'M', tuple_var_assignment_417677_418529)
    
    # Assigning a Name to a Name (line 211):
    # Getting the type of 'tuple_var_assignment_417678' (line 211)
    tuple_var_assignment_417678_418530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'tuple_var_assignment_417678')
    # Assigning a type to the variable 'N' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 6), 'N', tuple_var_assignment_417678_418530)
    
    # Assigning a Call to a Name (line 212):
    
    # Assigning a Call to a Name (line 212):
    
    # Call to spdiags(...): (line 212)
    # Processing the call arguments (line 212)
    
    # Obtaining an instance of the builtin type 'list' (line 212)
    list_418532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 212)
    # Adding element type (line 212)
    float_418533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 17), 'float')
    
    # Call to diagonal(...): (line 212)
    # Processing the call keyword arguments (line 212)
    kwargs_418536 = {}
    # Getting the type of 'A' (line 212)
    A_418534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 21), 'A', False)
    # Obtaining the member 'diagonal' of a type (line 212)
    diagonal_418535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 21), A_418534, 'diagonal')
    # Calling diagonal(args, kwargs) (line 212)
    diagonal_call_result_418537 = invoke(stypy.reporting.localization.Localization(__file__, 212, 21), diagonal_418535, *[], **kwargs_418536)
    
    # Applying the binary operator 'div' (line 212)
    result_div_418538 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 17), 'div', float_418533, diagonal_call_result_418537)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 16), list_418532, result_div_418538)
    
    
    # Obtaining an instance of the builtin type 'list' (line 212)
    list_418539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 212)
    # Adding element type (line 212)
    int_418540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 36), list_418539, int_418540)
    
    # Getting the type of 'M' (line 212)
    M_418541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 41), 'M', False)
    # Getting the type of 'N' (line 212)
    N_418542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 44), 'N', False)
    # Processing the call keyword arguments (line 212)
    kwargs_418543 = {}
    # Getting the type of 'spdiags' (line 212)
    spdiags_418531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'spdiags', False)
    # Calling spdiags(args, kwargs) (line 212)
    spdiags_call_result_418544 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), spdiags_418531, *[list_418532, list_418539, M_418541, N_418542], **kwargs_418543)
    
    # Assigning a type to the variable 'D' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'D', spdiags_call_result_418544)
    
    # Assigning a Call to a Name (line 214):
    
    # Assigning a Call to a Name (line 214):
    
    # Call to arange(...): (line 214)
    # Processing the call arguments (line 214)
    
    # Obtaining the type of the subscript
    int_418546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 23), 'int')
    # Getting the type of 'A' (line 214)
    A_418547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 15), 'A', False)
    # Obtaining the member 'shape' of a type (line 214)
    shape_418548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 15), A_418547, 'shape')
    # Obtaining the member '__getitem__' of a type (line 214)
    getitem___418549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 15), shape_418548, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 214)
    subscript_call_result_418550 = invoke(stypy.reporting.localization.Localization(__file__, 214, 15), getitem___418549, int_418546)
    
    # Processing the call keyword arguments (line 214)
    # Getting the type of 'float' (line 214)
    float_418551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 33), 'float', False)
    keyword_418552 = float_418551
    kwargs_418553 = {'dtype': keyword_418552}
    # Getting the type of 'arange' (line 214)
    arange_418545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'arange', False)
    # Calling arange(args, kwargs) (line 214)
    arange_call_result_418554 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), arange_418545, *[subscript_call_result_418550], **kwargs_418553)
    
    # Assigning a type to the variable 'b' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'b', arange_call_result_418554)
    
    # Assigning a BinOp to a Name (line 215):
    
    # Assigning a BinOp to a Name (line 215):
    int_418555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 9), 'int')
    # Getting the type of 'b' (line 215)
    b_418556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 11), 'b')
    # Applying the binary operator '*' (line 215)
    result_mul_418557 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 9), '*', int_418555, b_418556)
    
    # Assigning a type to the variable 'x0' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'x0', result_mul_418557)
    
    # Assigning a Call to a Name (line 217):
    
    # Assigning a Call to a Name (line 217):
    
    # Call to LinearOperator(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'A' (line 217)
    A_418559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 29), 'A', False)
    # Obtaining the member 'shape' of a type (line 217)
    shape_418560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 29), A_418559, 'shape')
    # Getting the type of 'identity' (line 217)
    identity_418561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 38), 'identity', False)
    # Processing the call keyword arguments (line 217)
    # Getting the type of 'identity' (line 217)
    identity_418562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 56), 'identity', False)
    keyword_418563 = identity_418562
    kwargs_418564 = {'rmatvec': keyword_418563}
    # Getting the type of 'LinearOperator' (line 217)
    LinearOperator_418558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 14), 'LinearOperator', False)
    # Calling LinearOperator(args, kwargs) (line 217)
    LinearOperator_call_result_418565 = invoke(stypy.reporting.localization.Localization(__file__, 217, 14), LinearOperator_418558, *[shape_418560, identity_418561], **kwargs_418564)
    
    # Assigning a type to the variable 'precond' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'precond', LinearOperator_call_result_418565)
    
    
    # Getting the type of 'solver' (line 219)
    solver_418566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 7), 'solver')
    # Getting the type of 'qmr' (line 219)
    qmr_418567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 17), 'qmr')
    # Applying the binary operator 'is' (line 219)
    result_is__418568 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 7), 'is', solver_418566, qmr_418567)
    
    # Testing the type of an if condition (line 219)
    if_condition_418569 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 219, 4), result_is__418568)
    # Assigning a type to the variable 'if_condition_418569' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'if_condition_418569', if_condition_418569)
    # SSA begins for if statement (line 219)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 220):
    
    # Assigning a Subscript to a Name (line 220):
    
    # Obtaining the type of the subscript
    int_418570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 8), 'int')
    
    # Call to solver(...): (line 220)
    # Processing the call arguments (line 220)
    # Getting the type of 'A' (line 220)
    A_418572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 25), 'A', False)
    # Getting the type of 'b' (line 220)
    b_418573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 28), 'b', False)
    # Processing the call keyword arguments (line 220)
    # Getting the type of 'precond' (line 220)
    precond_418574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 34), 'precond', False)
    keyword_418575 = precond_418574
    # Getting the type of 'precond' (line 220)
    precond_418576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 46), 'precond', False)
    keyword_418577 = precond_418576
    # Getting the type of 'x0' (line 220)
    x0_418578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 58), 'x0', False)
    keyword_418579 = x0_418578
    # Getting the type of 'tol' (line 220)
    tol_418580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 66), 'tol', False)
    keyword_418581 = tol_418580
    kwargs_418582 = {'x0': keyword_418579, 'M1': keyword_418575, 'tol': keyword_418581, 'M2': keyword_418577}
    # Getting the type of 'solver' (line 220)
    solver_418571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 18), 'solver', False)
    # Calling solver(args, kwargs) (line 220)
    solver_call_result_418583 = invoke(stypy.reporting.localization.Localization(__file__, 220, 18), solver_418571, *[A_418572, b_418573], **kwargs_418582)
    
    # Obtaining the member '__getitem__' of a type (line 220)
    getitem___418584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 8), solver_call_result_418583, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 220)
    subscript_call_result_418585 = invoke(stypy.reporting.localization.Localization(__file__, 220, 8), getitem___418584, int_418570)
    
    # Assigning a type to the variable 'tuple_var_assignment_417679' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'tuple_var_assignment_417679', subscript_call_result_418585)
    
    # Assigning a Subscript to a Name (line 220):
    
    # Obtaining the type of the subscript
    int_418586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 8), 'int')
    
    # Call to solver(...): (line 220)
    # Processing the call arguments (line 220)
    # Getting the type of 'A' (line 220)
    A_418588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 25), 'A', False)
    # Getting the type of 'b' (line 220)
    b_418589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 28), 'b', False)
    # Processing the call keyword arguments (line 220)
    # Getting the type of 'precond' (line 220)
    precond_418590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 34), 'precond', False)
    keyword_418591 = precond_418590
    # Getting the type of 'precond' (line 220)
    precond_418592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 46), 'precond', False)
    keyword_418593 = precond_418592
    # Getting the type of 'x0' (line 220)
    x0_418594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 58), 'x0', False)
    keyword_418595 = x0_418594
    # Getting the type of 'tol' (line 220)
    tol_418596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 66), 'tol', False)
    keyword_418597 = tol_418596
    kwargs_418598 = {'x0': keyword_418595, 'M1': keyword_418591, 'tol': keyword_418597, 'M2': keyword_418593}
    # Getting the type of 'solver' (line 220)
    solver_418587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 18), 'solver', False)
    # Calling solver(args, kwargs) (line 220)
    solver_call_result_418599 = invoke(stypy.reporting.localization.Localization(__file__, 220, 18), solver_418587, *[A_418588, b_418589], **kwargs_418598)
    
    # Obtaining the member '__getitem__' of a type (line 220)
    getitem___418600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 8), solver_call_result_418599, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 220)
    subscript_call_result_418601 = invoke(stypy.reporting.localization.Localization(__file__, 220, 8), getitem___418600, int_418586)
    
    # Assigning a type to the variable 'tuple_var_assignment_417680' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'tuple_var_assignment_417680', subscript_call_result_418601)
    
    # Assigning a Name to a Name (line 220):
    # Getting the type of 'tuple_var_assignment_417679' (line 220)
    tuple_var_assignment_417679_418602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'tuple_var_assignment_417679')
    # Assigning a type to the variable 'x' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'x', tuple_var_assignment_417679_418602)
    
    # Assigning a Name to a Name (line 220):
    # Getting the type of 'tuple_var_assignment_417680' (line 220)
    tuple_var_assignment_417680_418603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'tuple_var_assignment_417680')
    # Assigning a type to the variable 'info' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 11), 'info', tuple_var_assignment_417680_418603)
    # SSA branch for the else part of an if statement (line 219)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 222):
    
    # Assigning a Subscript to a Name (line 222):
    
    # Obtaining the type of the subscript
    int_418604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 8), 'int')
    
    # Call to solver(...): (line 222)
    # Processing the call arguments (line 222)
    # Getting the type of 'A' (line 222)
    A_418606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 25), 'A', False)
    # Getting the type of 'b' (line 222)
    b_418607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 28), 'b', False)
    # Processing the call keyword arguments (line 222)
    # Getting the type of 'precond' (line 222)
    precond_418608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 33), 'precond', False)
    keyword_418609 = precond_418608
    # Getting the type of 'x0' (line 222)
    x0_418610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 45), 'x0', False)
    keyword_418611 = x0_418610
    # Getting the type of 'tol' (line 222)
    tol_418612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 53), 'tol', False)
    keyword_418613 = tol_418612
    kwargs_418614 = {'x0': keyword_418611, 'M': keyword_418609, 'tol': keyword_418613}
    # Getting the type of 'solver' (line 222)
    solver_418605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 18), 'solver', False)
    # Calling solver(args, kwargs) (line 222)
    solver_call_result_418615 = invoke(stypy.reporting.localization.Localization(__file__, 222, 18), solver_418605, *[A_418606, b_418607], **kwargs_418614)
    
    # Obtaining the member '__getitem__' of a type (line 222)
    getitem___418616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), solver_call_result_418615, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 222)
    subscript_call_result_418617 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), getitem___418616, int_418604)
    
    # Assigning a type to the variable 'tuple_var_assignment_417681' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'tuple_var_assignment_417681', subscript_call_result_418617)
    
    # Assigning a Subscript to a Name (line 222):
    
    # Obtaining the type of the subscript
    int_418618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 8), 'int')
    
    # Call to solver(...): (line 222)
    # Processing the call arguments (line 222)
    # Getting the type of 'A' (line 222)
    A_418620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 25), 'A', False)
    # Getting the type of 'b' (line 222)
    b_418621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 28), 'b', False)
    # Processing the call keyword arguments (line 222)
    # Getting the type of 'precond' (line 222)
    precond_418622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 33), 'precond', False)
    keyword_418623 = precond_418622
    # Getting the type of 'x0' (line 222)
    x0_418624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 45), 'x0', False)
    keyword_418625 = x0_418624
    # Getting the type of 'tol' (line 222)
    tol_418626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 53), 'tol', False)
    keyword_418627 = tol_418626
    kwargs_418628 = {'x0': keyword_418625, 'M': keyword_418623, 'tol': keyword_418627}
    # Getting the type of 'solver' (line 222)
    solver_418619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 18), 'solver', False)
    # Calling solver(args, kwargs) (line 222)
    solver_call_result_418629 = invoke(stypy.reporting.localization.Localization(__file__, 222, 18), solver_418619, *[A_418620, b_418621], **kwargs_418628)
    
    # Obtaining the member '__getitem__' of a type (line 222)
    getitem___418630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), solver_call_result_418629, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 222)
    subscript_call_result_418631 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), getitem___418630, int_418618)
    
    # Assigning a type to the variable 'tuple_var_assignment_417682' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'tuple_var_assignment_417682', subscript_call_result_418631)
    
    # Assigning a Name to a Name (line 222):
    # Getting the type of 'tuple_var_assignment_417681' (line 222)
    tuple_var_assignment_417681_418632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'tuple_var_assignment_417681')
    # Assigning a type to the variable 'x' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'x', tuple_var_assignment_417681_418632)
    
    # Assigning a Name to a Name (line 222):
    # Getting the type of 'tuple_var_assignment_417682' (line 222)
    tuple_var_assignment_417682_418633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'tuple_var_assignment_417682')
    # Assigning a type to the variable 'info' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 11), 'info', tuple_var_assignment_417682_418633)
    # SSA join for if statement (line 219)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to assert_equal(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'info' (line 223)
    info_418635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 17), 'info', False)
    int_418636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 22), 'int')
    # Processing the call keyword arguments (line 223)
    kwargs_418637 = {}
    # Getting the type of 'assert_equal' (line 223)
    assert_equal_418634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 223)
    assert_equal_call_result_418638 = invoke(stypy.reporting.localization.Localization(__file__, 223, 4), assert_equal_418634, *[info_418635, int_418636], **kwargs_418637)
    
    
    # Call to assert_normclose(...): (line 224)
    # Processing the call arguments (line 224)
    
    # Call to dot(...): (line 224)
    # Processing the call arguments (line 224)
    # Getting the type of 'x' (line 224)
    x_418642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 27), 'x', False)
    # Processing the call keyword arguments (line 224)
    kwargs_418643 = {}
    # Getting the type of 'A' (line 224)
    A_418640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 21), 'A', False)
    # Obtaining the member 'dot' of a type (line 224)
    dot_418641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 21), A_418640, 'dot')
    # Calling dot(args, kwargs) (line 224)
    dot_call_result_418644 = invoke(stypy.reporting.localization.Localization(__file__, 224, 21), dot_418641, *[x_418642], **kwargs_418643)
    
    # Getting the type of 'b' (line 224)
    b_418645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 31), 'b', False)
    # Getting the type of 'tol' (line 224)
    tol_418646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 34), 'tol', False)
    # Processing the call keyword arguments (line 224)
    kwargs_418647 = {}
    # Getting the type of 'assert_normclose' (line 224)
    assert_normclose_418639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'assert_normclose', False)
    # Calling assert_normclose(args, kwargs) (line 224)
    assert_normclose_call_result_418648 = invoke(stypy.reporting.localization.Localization(__file__, 224, 4), assert_normclose_418639, *[dot_call_result_418644, b_418645, tol_418646], **kwargs_418647)
    
    
    # Assigning a Call to a Name (line 226):
    
    # Assigning a Call to a Name (line 226):
    
    # Call to aslinearoperator(...): (line 226)
    # Processing the call arguments (line 226)
    # Getting the type of 'A' (line 226)
    A_418650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 25), 'A', False)
    # Processing the call keyword arguments (line 226)
    kwargs_418651 = {}
    # Getting the type of 'aslinearoperator' (line 226)
    aslinearoperator_418649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'aslinearoperator', False)
    # Calling aslinearoperator(args, kwargs) (line 226)
    aslinearoperator_call_result_418652 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), aslinearoperator_418649, *[A_418650], **kwargs_418651)
    
    # Assigning a type to the variable 'A' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'A', aslinearoperator_call_result_418652)
    
    # Assigning a Name to a Attribute (line 227):
    
    # Assigning a Name to a Attribute (line 227):
    # Getting the type of 'identity' (line 227)
    identity_418653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 15), 'identity')
    # Getting the type of 'A' (line 227)
    A_418654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'A')
    # Setting the type of the member 'psolve' of a type (line 227)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 4), A_418654, 'psolve', identity_418653)
    
    # Assigning a Name to a Attribute (line 228):
    
    # Assigning a Name to a Attribute (line 228):
    # Getting the type of 'identity' (line 228)
    identity_418655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'identity')
    # Getting the type of 'A' (line 228)
    A_418656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'A')
    # Setting the type of the member 'rpsolve' of a type (line 228)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 4), A_418656, 'rpsolve', identity_418655)
    
    # Assigning a Call to a Tuple (line 230):
    
    # Assigning a Subscript to a Name (line 230):
    
    # Obtaining the type of the subscript
    int_418657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 4), 'int')
    
    # Call to solver(...): (line 230)
    # Processing the call arguments (line 230)
    # Getting the type of 'A' (line 230)
    A_418659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 21), 'A', False)
    # Getting the type of 'b' (line 230)
    b_418660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 24), 'b', False)
    # Processing the call keyword arguments (line 230)
    # Getting the type of 'x0' (line 230)
    x0_418661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 30), 'x0', False)
    keyword_418662 = x0_418661
    # Getting the type of 'tol' (line 230)
    tol_418663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 38), 'tol', False)
    keyword_418664 = tol_418663
    kwargs_418665 = {'x0': keyword_418662, 'tol': keyword_418664}
    # Getting the type of 'solver' (line 230)
    solver_418658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 14), 'solver', False)
    # Calling solver(args, kwargs) (line 230)
    solver_call_result_418666 = invoke(stypy.reporting.localization.Localization(__file__, 230, 14), solver_418658, *[A_418659, b_418660], **kwargs_418665)
    
    # Obtaining the member '__getitem__' of a type (line 230)
    getitem___418667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 4), solver_call_result_418666, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 230)
    subscript_call_result_418668 = invoke(stypy.reporting.localization.Localization(__file__, 230, 4), getitem___418667, int_418657)
    
    # Assigning a type to the variable 'tuple_var_assignment_417683' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'tuple_var_assignment_417683', subscript_call_result_418668)
    
    # Assigning a Subscript to a Name (line 230):
    
    # Obtaining the type of the subscript
    int_418669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 4), 'int')
    
    # Call to solver(...): (line 230)
    # Processing the call arguments (line 230)
    # Getting the type of 'A' (line 230)
    A_418671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 21), 'A', False)
    # Getting the type of 'b' (line 230)
    b_418672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 24), 'b', False)
    # Processing the call keyword arguments (line 230)
    # Getting the type of 'x0' (line 230)
    x0_418673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 30), 'x0', False)
    keyword_418674 = x0_418673
    # Getting the type of 'tol' (line 230)
    tol_418675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 38), 'tol', False)
    keyword_418676 = tol_418675
    kwargs_418677 = {'x0': keyword_418674, 'tol': keyword_418676}
    # Getting the type of 'solver' (line 230)
    solver_418670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 14), 'solver', False)
    # Calling solver(args, kwargs) (line 230)
    solver_call_result_418678 = invoke(stypy.reporting.localization.Localization(__file__, 230, 14), solver_418670, *[A_418671, b_418672], **kwargs_418677)
    
    # Obtaining the member '__getitem__' of a type (line 230)
    getitem___418679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 4), solver_call_result_418678, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 230)
    subscript_call_result_418680 = invoke(stypy.reporting.localization.Localization(__file__, 230, 4), getitem___418679, int_418669)
    
    # Assigning a type to the variable 'tuple_var_assignment_417684' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'tuple_var_assignment_417684', subscript_call_result_418680)
    
    # Assigning a Name to a Name (line 230):
    # Getting the type of 'tuple_var_assignment_417683' (line 230)
    tuple_var_assignment_417683_418681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'tuple_var_assignment_417683')
    # Assigning a type to the variable 'x' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'x', tuple_var_assignment_417683_418681)
    
    # Assigning a Name to a Name (line 230):
    # Getting the type of 'tuple_var_assignment_417684' (line 230)
    tuple_var_assignment_417684_418682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'tuple_var_assignment_417684')
    # Assigning a type to the variable 'info' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 7), 'info', tuple_var_assignment_417684_418682)
    
    # Call to assert_equal(...): (line 231)
    # Processing the call arguments (line 231)
    # Getting the type of 'info' (line 231)
    info_418684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 17), 'info', False)
    int_418685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 22), 'int')
    # Processing the call keyword arguments (line 231)
    kwargs_418686 = {}
    # Getting the type of 'assert_equal' (line 231)
    assert_equal_418683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 231)
    assert_equal_call_result_418687 = invoke(stypy.reporting.localization.Localization(__file__, 231, 4), assert_equal_418683, *[info_418684, int_418685], **kwargs_418686)
    
    
    # Call to assert_normclose(...): (line 232)
    # Processing the call arguments (line 232)
    # Getting the type of 'A' (line 232)
    A_418689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 21), 'A', False)
    # Getting the type of 'x' (line 232)
    x_418690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 23), 'x', False)
    # Applying the binary operator '*' (line 232)
    result_mul_418691 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 21), '*', A_418689, x_418690)
    
    # Getting the type of 'b' (line 232)
    b_418692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 26), 'b', False)
    # Processing the call keyword arguments (line 232)
    # Getting the type of 'tol' (line 232)
    tol_418693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 33), 'tol', False)
    keyword_418694 = tol_418693
    kwargs_418695 = {'tol': keyword_418694}
    # Getting the type of 'assert_normclose' (line 232)
    assert_normclose_418688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'assert_normclose', False)
    # Calling assert_normclose(args, kwargs) (line 232)
    assert_normclose_call_result_418696 = invoke(stypy.reporting.localization.Localization(__file__, 232, 4), assert_normclose_418688, *[result_mul_418691, b_418692], **kwargs_418695)
    
    
    # ################# End of 'check_precond_dummy(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_precond_dummy' in the type store
    # Getting the type of 'stypy_return_type' (line 202)
    stypy_return_type_418697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_418697)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_precond_dummy'
    return stypy_return_type_418697

# Assigning a type to the variable 'check_precond_dummy' (line 202)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 0), 'check_precond_dummy', check_precond_dummy)

@norecursion
def test_precond_dummy(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_precond_dummy'
    module_type_store = module_type_store.open_function_context('test_precond_dummy', 235, 0, False)
    
    # Passed parameters checking function
    test_precond_dummy.stypy_localization = localization
    test_precond_dummy.stypy_type_of_self = None
    test_precond_dummy.stypy_type_store = module_type_store
    test_precond_dummy.stypy_function_name = 'test_precond_dummy'
    test_precond_dummy.stypy_param_names_list = []
    test_precond_dummy.stypy_varargs_param_name = None
    test_precond_dummy.stypy_kwargs_param_name = None
    test_precond_dummy.stypy_call_defaults = defaults
    test_precond_dummy.stypy_call_varargs = varargs
    test_precond_dummy.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_precond_dummy', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_precond_dummy', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_precond_dummy(...)' code ##################

    
    # Assigning a Attribute to a Name (line 236):
    
    # Assigning a Attribute to a Name (line 236):
    # Getting the type of 'params' (line 236)
    params_418698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 11), 'params')
    # Obtaining the member 'Poisson1D' of a type (line 236)
    Poisson1D_418699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 11), params_418698, 'Poisson1D')
    # Assigning a type to the variable 'case' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'case', Poisson1D_418699)
    
    # Getting the type of 'params' (line 237)
    params_418700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 18), 'params')
    # Obtaining the member 'solvers' of a type (line 237)
    solvers_418701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 18), params_418700, 'solvers')
    # Testing the type of a for loop iterable (line 237)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 237, 4), solvers_418701)
    # Getting the type of the for loop variable (line 237)
    for_loop_var_418702 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 237, 4), solvers_418701)
    # Assigning a type to the variable 'solver' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'solver', for_loop_var_418702)
    # SSA begins for a for statement (line 237)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'solver' (line 238)
    solver_418703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 11), 'solver')
    # Getting the type of 'case' (line 238)
    case_418704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 21), 'case')
    # Obtaining the member 'skip' of a type (line 238)
    skip_418705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 21), case_418704, 'skip')
    # Applying the binary operator 'in' (line 238)
    result_contains_418706 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 11), 'in', solver_418703, skip_418705)
    
    # Testing the type of an if condition (line 238)
    if_condition_418707 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 238, 8), result_contains_418706)
    # Assigning a type to the variable 'if_condition_418707' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'if_condition_418707', if_condition_418707)
    # SSA begins for if statement (line 238)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 238)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to check_precond_dummy(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'solver' (line 240)
    solver_418709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 28), 'solver', False)
    # Getting the type of 'case' (line 240)
    case_418710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 36), 'case', False)
    # Processing the call keyword arguments (line 240)
    kwargs_418711 = {}
    # Getting the type of 'check_precond_dummy' (line 240)
    check_precond_dummy_418708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'check_precond_dummy', False)
    # Calling check_precond_dummy(args, kwargs) (line 240)
    check_precond_dummy_call_result_418712 = invoke(stypy.reporting.localization.Localization(__file__, 240, 8), check_precond_dummy_418708, *[solver_418709, case_418710], **kwargs_418711)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_precond_dummy(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_precond_dummy' in the type store
    # Getting the type of 'stypy_return_type' (line 235)
    stypy_return_type_418713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_418713)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_precond_dummy'
    return stypy_return_type_418713

# Assigning a type to the variable 'test_precond_dummy' (line 235)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 0), 'test_precond_dummy', test_precond_dummy)

@norecursion
def check_precond_inverse(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_precond_inverse'
    module_type_store = module_type_store.open_function_context('check_precond_inverse', 243, 0, False)
    
    # Passed parameters checking function
    check_precond_inverse.stypy_localization = localization
    check_precond_inverse.stypy_type_of_self = None
    check_precond_inverse.stypy_type_store = module_type_store
    check_precond_inverse.stypy_function_name = 'check_precond_inverse'
    check_precond_inverse.stypy_param_names_list = ['solver', 'case']
    check_precond_inverse.stypy_varargs_param_name = None
    check_precond_inverse.stypy_kwargs_param_name = None
    check_precond_inverse.stypy_call_defaults = defaults
    check_precond_inverse.stypy_call_varargs = varargs
    check_precond_inverse.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_precond_inverse', ['solver', 'case'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_precond_inverse', localization, ['solver', 'case'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_precond_inverse(...)' code ##################

    
    # Assigning a Num to a Name (line 244):
    
    # Assigning a Num to a Name (line 244):
    float_418714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 10), 'float')
    # Assigning a type to the variable 'tol' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'tol', float_418714)

    @norecursion
    def inverse(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 246)
        None_418715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 24), 'None')
        defaults = [None_418715]
        # Create a new context for function 'inverse'
        module_type_store = module_type_store.open_function_context('inverse', 246, 4, False)
        
        # Passed parameters checking function
        inverse.stypy_localization = localization
        inverse.stypy_type_of_self = None
        inverse.stypy_type_store = module_type_store
        inverse.stypy_function_name = 'inverse'
        inverse.stypy_param_names_list = ['b', 'which']
        inverse.stypy_varargs_param_name = None
        inverse.stypy_kwargs_param_name = None
        inverse.stypy_call_defaults = defaults
        inverse.stypy_call_varargs = varargs
        inverse.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'inverse', ['b', 'which'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'inverse', localization, ['b', 'which'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'inverse(...)' code ##################

        str_418716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 8), 'str', 'inverse preconditioner')
        
        # Assigning a Attribute to a Name (line 248):
        
        # Assigning a Attribute to a Name (line 248):
        # Getting the type of 'case' (line 248)
        case_418717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'case')
        # Obtaining the member 'A' of a type (line 248)
        A_418718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 12), case_418717, 'A')
        # Assigning a type to the variable 'A' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'A', A_418718)
        
        
        
        # Call to isinstance(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 'A' (line 249)
        A_418720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 26), 'A', False)
        # Getting the type of 'np' (line 249)
        np_418721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 29), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 249)
        ndarray_418722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 29), np_418721, 'ndarray')
        # Processing the call keyword arguments (line 249)
        kwargs_418723 = {}
        # Getting the type of 'isinstance' (line 249)
        isinstance_418719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 249)
        isinstance_call_result_418724 = invoke(stypy.reporting.localization.Localization(__file__, 249, 15), isinstance_418719, *[A_418720, ndarray_418722], **kwargs_418723)
        
        # Applying the 'not' unary operator (line 249)
        result_not__418725 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 11), 'not', isinstance_call_result_418724)
        
        # Testing the type of an if condition (line 249)
        if_condition_418726 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 8), result_not__418725)
        # Assigning a type to the variable 'if_condition_418726' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'if_condition_418726', if_condition_418726)
        # SSA begins for if statement (line 249)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 250):
        
        # Assigning a Call to a Name (line 250):
        
        # Call to todense(...): (line 250)
        # Processing the call keyword arguments (line 250)
        kwargs_418729 = {}
        # Getting the type of 'A' (line 250)
        A_418727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'A', False)
        # Obtaining the member 'todense' of a type (line 250)
        todense_418728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 16), A_418727, 'todense')
        # Calling todense(args, kwargs) (line 250)
        todense_call_result_418730 = invoke(stypy.reporting.localization.Localization(__file__, 250, 16), todense_418728, *[], **kwargs_418729)
        
        # Assigning a type to the variable 'A' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'A', todense_call_result_418730)
        # SSA join for if statement (line 249)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to solve(...): (line 251)
        # Processing the call arguments (line 251)
        # Getting the type of 'A' (line 251)
        A_418734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 31), 'A', False)
        # Getting the type of 'b' (line 251)
        b_418735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 34), 'b', False)
        # Processing the call keyword arguments (line 251)
        kwargs_418736 = {}
        # Getting the type of 'np' (line 251)
        np_418731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 15), 'np', False)
        # Obtaining the member 'linalg' of a type (line 251)
        linalg_418732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 15), np_418731, 'linalg')
        # Obtaining the member 'solve' of a type (line 251)
        solve_418733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 15), linalg_418732, 'solve')
        # Calling solve(args, kwargs) (line 251)
        solve_call_result_418737 = invoke(stypy.reporting.localization.Localization(__file__, 251, 15), solve_418733, *[A_418734, b_418735], **kwargs_418736)
        
        # Assigning a type to the variable 'stypy_return_type' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'stypy_return_type', solve_call_result_418737)
        
        # ################# End of 'inverse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'inverse' in the type store
        # Getting the type of 'stypy_return_type' (line 246)
        stypy_return_type_418738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_418738)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'inverse'
        return stypy_return_type_418738

    # Assigning a type to the variable 'inverse' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'inverse', inverse)

    @norecursion
    def rinverse(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 253)
        None_418739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 25), 'None')
        defaults = [None_418739]
        # Create a new context for function 'rinverse'
        module_type_store = module_type_store.open_function_context('rinverse', 253, 4, False)
        
        # Passed parameters checking function
        rinverse.stypy_localization = localization
        rinverse.stypy_type_of_self = None
        rinverse.stypy_type_store = module_type_store
        rinverse.stypy_function_name = 'rinverse'
        rinverse.stypy_param_names_list = ['b', 'which']
        rinverse.stypy_varargs_param_name = None
        rinverse.stypy_kwargs_param_name = None
        rinverse.stypy_call_defaults = defaults
        rinverse.stypy_call_varargs = varargs
        rinverse.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'rinverse', ['b', 'which'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'rinverse', localization, ['b', 'which'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'rinverse(...)' code ##################

        str_418740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 8), 'str', 'inverse preconditioner')
        
        # Assigning a Attribute to a Name (line 255):
        
        # Assigning a Attribute to a Name (line 255):
        # Getting the type of 'case' (line 255)
        case_418741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'case')
        # Obtaining the member 'A' of a type (line 255)
        A_418742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 12), case_418741, 'A')
        # Assigning a type to the variable 'A' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'A', A_418742)
        
        
        
        # Call to isinstance(...): (line 256)
        # Processing the call arguments (line 256)
        # Getting the type of 'A' (line 256)
        A_418744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 26), 'A', False)
        # Getting the type of 'np' (line 256)
        np_418745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 29), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 256)
        ndarray_418746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 29), np_418745, 'ndarray')
        # Processing the call keyword arguments (line 256)
        kwargs_418747 = {}
        # Getting the type of 'isinstance' (line 256)
        isinstance_418743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 256)
        isinstance_call_result_418748 = invoke(stypy.reporting.localization.Localization(__file__, 256, 15), isinstance_418743, *[A_418744, ndarray_418746], **kwargs_418747)
        
        # Applying the 'not' unary operator (line 256)
        result_not__418749 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 11), 'not', isinstance_call_result_418748)
        
        # Testing the type of an if condition (line 256)
        if_condition_418750 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 256, 8), result_not__418749)
        # Assigning a type to the variable 'if_condition_418750' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'if_condition_418750', if_condition_418750)
        # SSA begins for if statement (line 256)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 257):
        
        # Assigning a Call to a Name (line 257):
        
        # Call to todense(...): (line 257)
        # Processing the call keyword arguments (line 257)
        kwargs_418753 = {}
        # Getting the type of 'A' (line 257)
        A_418751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'A', False)
        # Obtaining the member 'todense' of a type (line 257)
        todense_418752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 16), A_418751, 'todense')
        # Calling todense(args, kwargs) (line 257)
        todense_call_result_418754 = invoke(stypy.reporting.localization.Localization(__file__, 257, 16), todense_418752, *[], **kwargs_418753)
        
        # Assigning a type to the variable 'A' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'A', todense_call_result_418754)
        # SSA join for if statement (line 256)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to solve(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 'A' (line 258)
        A_418758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 31), 'A', False)
        # Obtaining the member 'T' of a type (line 258)
        T_418759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 31), A_418758, 'T')
        # Getting the type of 'b' (line 258)
        b_418760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 36), 'b', False)
        # Processing the call keyword arguments (line 258)
        kwargs_418761 = {}
        # Getting the type of 'np' (line 258)
        np_418755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 15), 'np', False)
        # Obtaining the member 'linalg' of a type (line 258)
        linalg_418756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 15), np_418755, 'linalg')
        # Obtaining the member 'solve' of a type (line 258)
        solve_418757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 15), linalg_418756, 'solve')
        # Calling solve(args, kwargs) (line 258)
        solve_call_result_418762 = invoke(stypy.reporting.localization.Localization(__file__, 258, 15), solve_418757, *[T_418759, b_418760], **kwargs_418761)
        
        # Assigning a type to the variable 'stypy_return_type' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'stypy_return_type', solve_call_result_418762)
        
        # ################# End of 'rinverse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'rinverse' in the type store
        # Getting the type of 'stypy_return_type' (line 253)
        stypy_return_type_418763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_418763)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'rinverse'
        return stypy_return_type_418763

    # Assigning a type to the variable 'rinverse' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'rinverse', rinverse)
    
    # Assigning a List to a Name (line 260):
    
    # Assigning a List to a Name (line 260):
    
    # Obtaining an instance of the builtin type 'list' (line 260)
    list_418764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 260)
    # Adding element type (line 260)
    int_418765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 19), list_418764, int_418765)
    
    # Assigning a type to the variable 'matvec_count' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'matvec_count', list_418764)

    @norecursion
    def matvec(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'matvec'
        module_type_store = module_type_store.open_function_context('matvec', 262, 4, False)
        
        # Passed parameters checking function
        matvec.stypy_localization = localization
        matvec.stypy_type_of_self = None
        matvec.stypy_type_store = module_type_store
        matvec.stypy_function_name = 'matvec'
        matvec.stypy_param_names_list = ['b']
        matvec.stypy_varargs_param_name = None
        matvec.stypy_kwargs_param_name = None
        matvec.stypy_call_defaults = defaults
        matvec.stypy_call_varargs = varargs
        matvec.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'matvec', ['b'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'matvec', localization, ['b'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'matvec(...)' code ##################

        
        # Getting the type of 'matvec_count' (line 263)
        matvec_count_418766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'matvec_count')
        
        # Obtaining the type of the subscript
        int_418767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 21), 'int')
        # Getting the type of 'matvec_count' (line 263)
        matvec_count_418768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'matvec_count')
        # Obtaining the member '__getitem__' of a type (line 263)
        getitem___418769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 8), matvec_count_418768, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 263)
        subscript_call_result_418770 = invoke(stypy.reporting.localization.Localization(__file__, 263, 8), getitem___418769, int_418767)
        
        int_418771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 27), 'int')
        # Applying the binary operator '+=' (line 263)
        result_iadd_418772 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 8), '+=', subscript_call_result_418770, int_418771)
        # Getting the type of 'matvec_count' (line 263)
        matvec_count_418773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'matvec_count')
        int_418774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 21), 'int')
        # Storing an element on a container (line 263)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 8), matvec_count_418773, (int_418774, result_iadd_418772))
        
        
        # Call to dot(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 'b' (line 264)
        b_418778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 26), 'b', False)
        # Processing the call keyword arguments (line 264)
        kwargs_418779 = {}
        # Getting the type of 'case' (line 264)
        case_418775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 15), 'case', False)
        # Obtaining the member 'A' of a type (line 264)
        A_418776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 15), case_418775, 'A')
        # Obtaining the member 'dot' of a type (line 264)
        dot_418777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 15), A_418776, 'dot')
        # Calling dot(args, kwargs) (line 264)
        dot_call_result_418780 = invoke(stypy.reporting.localization.Localization(__file__, 264, 15), dot_418777, *[b_418778], **kwargs_418779)
        
        # Assigning a type to the variable 'stypy_return_type' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'stypy_return_type', dot_call_result_418780)
        
        # ################# End of 'matvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'matvec' in the type store
        # Getting the type of 'stypy_return_type' (line 262)
        stypy_return_type_418781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_418781)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'matvec'
        return stypy_return_type_418781

    # Assigning a type to the variable 'matvec' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'matvec', matvec)

    @norecursion
    def rmatvec(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'rmatvec'
        module_type_store = module_type_store.open_function_context('rmatvec', 266, 4, False)
        
        # Passed parameters checking function
        rmatvec.stypy_localization = localization
        rmatvec.stypy_type_of_self = None
        rmatvec.stypy_type_store = module_type_store
        rmatvec.stypy_function_name = 'rmatvec'
        rmatvec.stypy_param_names_list = ['b']
        rmatvec.stypy_varargs_param_name = None
        rmatvec.stypy_kwargs_param_name = None
        rmatvec.stypy_call_defaults = defaults
        rmatvec.stypy_call_varargs = varargs
        rmatvec.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'rmatvec', ['b'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'rmatvec', localization, ['b'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'rmatvec(...)' code ##################

        
        # Getting the type of 'matvec_count' (line 267)
        matvec_count_418782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'matvec_count')
        
        # Obtaining the type of the subscript
        int_418783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 21), 'int')
        # Getting the type of 'matvec_count' (line 267)
        matvec_count_418784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'matvec_count')
        # Obtaining the member '__getitem__' of a type (line 267)
        getitem___418785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 8), matvec_count_418784, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 267)
        subscript_call_result_418786 = invoke(stypy.reporting.localization.Localization(__file__, 267, 8), getitem___418785, int_418783)
        
        int_418787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 27), 'int')
        # Applying the binary operator '+=' (line 267)
        result_iadd_418788 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 8), '+=', subscript_call_result_418786, int_418787)
        # Getting the type of 'matvec_count' (line 267)
        matvec_count_418789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'matvec_count')
        int_418790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 21), 'int')
        # Storing an element on a container (line 267)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 8), matvec_count_418789, (int_418790, result_iadd_418788))
        
        
        # Call to dot(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'b' (line 268)
        b_418795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 28), 'b', False)
        # Processing the call keyword arguments (line 268)
        kwargs_418796 = {}
        # Getting the type of 'case' (line 268)
        case_418791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 15), 'case', False)
        # Obtaining the member 'A' of a type (line 268)
        A_418792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 15), case_418791, 'A')
        # Obtaining the member 'T' of a type (line 268)
        T_418793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 15), A_418792, 'T')
        # Obtaining the member 'dot' of a type (line 268)
        dot_418794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 15), T_418793, 'dot')
        # Calling dot(args, kwargs) (line 268)
        dot_call_result_418797 = invoke(stypy.reporting.localization.Localization(__file__, 268, 15), dot_418794, *[b_418795], **kwargs_418796)
        
        # Assigning a type to the variable 'stypy_return_type' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'stypy_return_type', dot_call_result_418797)
        
        # ################# End of 'rmatvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'rmatvec' in the type store
        # Getting the type of 'stypy_return_type' (line 266)
        stypy_return_type_418798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_418798)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'rmatvec'
        return stypy_return_type_418798

    # Assigning a type to the variable 'rmatvec' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'rmatvec', rmatvec)
    
    # Assigning a Call to a Name (line 270):
    
    # Assigning a Call to a Name (line 270):
    
    # Call to arange(...): (line 270)
    # Processing the call arguments (line 270)
    
    # Obtaining the type of the subscript
    int_418800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 28), 'int')
    # Getting the type of 'case' (line 270)
    case_418801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 15), 'case', False)
    # Obtaining the member 'A' of a type (line 270)
    A_418802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 15), case_418801, 'A')
    # Obtaining the member 'shape' of a type (line 270)
    shape_418803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 15), A_418802, 'shape')
    # Obtaining the member '__getitem__' of a type (line 270)
    getitem___418804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 15), shape_418803, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 270)
    subscript_call_result_418805 = invoke(stypy.reporting.localization.Localization(__file__, 270, 15), getitem___418804, int_418800)
    
    # Processing the call keyword arguments (line 270)
    # Getting the type of 'float' (line 270)
    float_418806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 38), 'float', False)
    keyword_418807 = float_418806
    kwargs_418808 = {'dtype': keyword_418807}
    # Getting the type of 'arange' (line 270)
    arange_418799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'arange', False)
    # Calling arange(args, kwargs) (line 270)
    arange_call_result_418809 = invoke(stypy.reporting.localization.Localization(__file__, 270, 8), arange_418799, *[subscript_call_result_418805], **kwargs_418808)
    
    # Assigning a type to the variable 'b' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'b', arange_call_result_418809)
    
    # Assigning a BinOp to a Name (line 271):
    
    # Assigning a BinOp to a Name (line 271):
    int_418810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 9), 'int')
    # Getting the type of 'b' (line 271)
    b_418811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 11), 'b')
    # Applying the binary operator '*' (line 271)
    result_mul_418812 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 9), '*', int_418810, b_418811)
    
    # Assigning a type to the variable 'x0' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'x0', result_mul_418812)
    
    # Assigning a Call to a Name (line 273):
    
    # Assigning a Call to a Name (line 273):
    
    # Call to LinearOperator(...): (line 273)
    # Processing the call arguments (line 273)
    # Getting the type of 'case' (line 273)
    case_418814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 23), 'case', False)
    # Obtaining the member 'A' of a type (line 273)
    A_418815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 23), case_418814, 'A')
    # Obtaining the member 'shape' of a type (line 273)
    shape_418816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 23), A_418815, 'shape')
    # Getting the type of 'matvec' (line 273)
    matvec_418817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 37), 'matvec', False)
    # Processing the call keyword arguments (line 273)
    # Getting the type of 'rmatvec' (line 273)
    rmatvec_418818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 53), 'rmatvec', False)
    keyword_418819 = rmatvec_418818
    kwargs_418820 = {'rmatvec': keyword_418819}
    # Getting the type of 'LinearOperator' (line 273)
    LinearOperator_418813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'LinearOperator', False)
    # Calling LinearOperator(args, kwargs) (line 273)
    LinearOperator_call_result_418821 = invoke(stypy.reporting.localization.Localization(__file__, 273, 8), LinearOperator_418813, *[shape_418816, matvec_418817], **kwargs_418820)
    
    # Assigning a type to the variable 'A' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'A', LinearOperator_call_result_418821)
    
    # Assigning a Call to a Name (line 274):
    
    # Assigning a Call to a Name (line 274):
    
    # Call to LinearOperator(...): (line 274)
    # Processing the call arguments (line 274)
    # Getting the type of 'case' (line 274)
    case_418823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 29), 'case', False)
    # Obtaining the member 'A' of a type (line 274)
    A_418824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 29), case_418823, 'A')
    # Obtaining the member 'shape' of a type (line 274)
    shape_418825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 29), A_418824, 'shape')
    # Getting the type of 'inverse' (line 274)
    inverse_418826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 43), 'inverse', False)
    # Processing the call keyword arguments (line 274)
    # Getting the type of 'rinverse' (line 274)
    rinverse_418827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 60), 'rinverse', False)
    keyword_418828 = rinverse_418827
    kwargs_418829 = {'rmatvec': keyword_418828}
    # Getting the type of 'LinearOperator' (line 274)
    LinearOperator_418822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 14), 'LinearOperator', False)
    # Calling LinearOperator(args, kwargs) (line 274)
    LinearOperator_call_result_418830 = invoke(stypy.reporting.localization.Localization(__file__, 274, 14), LinearOperator_418822, *[shape_418825, inverse_418826], **kwargs_418829)
    
    # Assigning a type to the variable 'precond' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'precond', LinearOperator_call_result_418830)
    
    # Assigning a List to a Name (line 277):
    
    # Assigning a List to a Name (line 277):
    
    # Obtaining an instance of the builtin type 'list' (line 277)
    list_418831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 277)
    # Adding element type (line 277)
    int_418832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 19), list_418831, int_418832)
    
    # Assigning a type to the variable 'matvec_count' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'matvec_count', list_418831)
    
    # Assigning a Call to a Tuple (line 278):
    
    # Assigning a Subscript to a Name (line 278):
    
    # Obtaining the type of the subscript
    int_418833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 4), 'int')
    
    # Call to solver(...): (line 278)
    # Processing the call arguments (line 278)
    # Getting the type of 'A' (line 278)
    A_418835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 21), 'A', False)
    # Getting the type of 'b' (line 278)
    b_418836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 24), 'b', False)
    # Processing the call keyword arguments (line 278)
    # Getting the type of 'precond' (line 278)
    precond_418837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 29), 'precond', False)
    keyword_418838 = precond_418837
    # Getting the type of 'x0' (line 278)
    x0_418839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 41), 'x0', False)
    keyword_418840 = x0_418839
    # Getting the type of 'tol' (line 278)
    tol_418841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 49), 'tol', False)
    keyword_418842 = tol_418841
    kwargs_418843 = {'x0': keyword_418840, 'M': keyword_418838, 'tol': keyword_418842}
    # Getting the type of 'solver' (line 278)
    solver_418834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 14), 'solver', False)
    # Calling solver(args, kwargs) (line 278)
    solver_call_result_418844 = invoke(stypy.reporting.localization.Localization(__file__, 278, 14), solver_418834, *[A_418835, b_418836], **kwargs_418843)
    
    # Obtaining the member '__getitem__' of a type (line 278)
    getitem___418845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 4), solver_call_result_418844, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 278)
    subscript_call_result_418846 = invoke(stypy.reporting.localization.Localization(__file__, 278, 4), getitem___418845, int_418833)
    
    # Assigning a type to the variable 'tuple_var_assignment_417685' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'tuple_var_assignment_417685', subscript_call_result_418846)
    
    # Assigning a Subscript to a Name (line 278):
    
    # Obtaining the type of the subscript
    int_418847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 4), 'int')
    
    # Call to solver(...): (line 278)
    # Processing the call arguments (line 278)
    # Getting the type of 'A' (line 278)
    A_418849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 21), 'A', False)
    # Getting the type of 'b' (line 278)
    b_418850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 24), 'b', False)
    # Processing the call keyword arguments (line 278)
    # Getting the type of 'precond' (line 278)
    precond_418851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 29), 'precond', False)
    keyword_418852 = precond_418851
    # Getting the type of 'x0' (line 278)
    x0_418853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 41), 'x0', False)
    keyword_418854 = x0_418853
    # Getting the type of 'tol' (line 278)
    tol_418855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 49), 'tol', False)
    keyword_418856 = tol_418855
    kwargs_418857 = {'x0': keyword_418854, 'M': keyword_418852, 'tol': keyword_418856}
    # Getting the type of 'solver' (line 278)
    solver_418848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 14), 'solver', False)
    # Calling solver(args, kwargs) (line 278)
    solver_call_result_418858 = invoke(stypy.reporting.localization.Localization(__file__, 278, 14), solver_418848, *[A_418849, b_418850], **kwargs_418857)
    
    # Obtaining the member '__getitem__' of a type (line 278)
    getitem___418859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 4), solver_call_result_418858, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 278)
    subscript_call_result_418860 = invoke(stypy.reporting.localization.Localization(__file__, 278, 4), getitem___418859, int_418847)
    
    # Assigning a type to the variable 'tuple_var_assignment_417686' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'tuple_var_assignment_417686', subscript_call_result_418860)
    
    # Assigning a Name to a Name (line 278):
    # Getting the type of 'tuple_var_assignment_417685' (line 278)
    tuple_var_assignment_417685_418861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'tuple_var_assignment_417685')
    # Assigning a type to the variable 'x' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'x', tuple_var_assignment_417685_418861)
    
    # Assigning a Name to a Name (line 278):
    # Getting the type of 'tuple_var_assignment_417686' (line 278)
    tuple_var_assignment_417686_418862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'tuple_var_assignment_417686')
    # Assigning a type to the variable 'info' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 7), 'info', tuple_var_assignment_417686_418862)
    
    # Call to assert_equal(...): (line 280)
    # Processing the call arguments (line 280)
    # Getting the type of 'info' (line 280)
    info_418864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 17), 'info', False)
    int_418865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 23), 'int')
    # Processing the call keyword arguments (line 280)
    kwargs_418866 = {}
    # Getting the type of 'assert_equal' (line 280)
    assert_equal_418863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 280)
    assert_equal_call_result_418867 = invoke(stypy.reporting.localization.Localization(__file__, 280, 4), assert_equal_418863, *[info_418864, int_418865], **kwargs_418866)
    
    
    # Call to assert_normclose(...): (line 281)
    # Processing the call arguments (line 281)
    
    # Call to dot(...): (line 281)
    # Processing the call arguments (line 281)
    # Getting the type of 'x' (line 281)
    x_418872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 32), 'x', False)
    # Processing the call keyword arguments (line 281)
    kwargs_418873 = {}
    # Getting the type of 'case' (line 281)
    case_418869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 21), 'case', False)
    # Obtaining the member 'A' of a type (line 281)
    A_418870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 21), case_418869, 'A')
    # Obtaining the member 'dot' of a type (line 281)
    dot_418871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 21), A_418870, 'dot')
    # Calling dot(args, kwargs) (line 281)
    dot_call_result_418874 = invoke(stypy.reporting.localization.Localization(__file__, 281, 21), dot_418871, *[x_418872], **kwargs_418873)
    
    # Getting the type of 'b' (line 281)
    b_418875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 36), 'b', False)
    # Getting the type of 'tol' (line 281)
    tol_418876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 39), 'tol', False)
    # Processing the call keyword arguments (line 281)
    kwargs_418877 = {}
    # Getting the type of 'assert_normclose' (line 281)
    assert_normclose_418868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'assert_normclose', False)
    # Calling assert_normclose(args, kwargs) (line 281)
    assert_normclose_call_result_418878 = invoke(stypy.reporting.localization.Localization(__file__, 281, 4), assert_normclose_418868, *[dot_call_result_418874, b_418875, tol_418876], **kwargs_418877)
    
    
    # Call to assert_(...): (line 284)
    # Processing the call arguments (line 284)
    
    
    # Obtaining the type of the subscript
    int_418880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 25), 'int')
    # Getting the type of 'matvec_count' (line 284)
    matvec_count_418881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'matvec_count', False)
    # Obtaining the member '__getitem__' of a type (line 284)
    getitem___418882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 12), matvec_count_418881, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 284)
    subscript_call_result_418883 = invoke(stypy.reporting.localization.Localization(__file__, 284, 12), getitem___418882, int_418880)
    
    int_418884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 31), 'int')
    # Applying the binary operator '<=' (line 284)
    result_le_418885 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 12), '<=', subscript_call_result_418883, int_418884)
    
    
    # Call to repr(...): (line 284)
    # Processing the call arguments (line 284)
    # Getting the type of 'matvec_count' (line 284)
    matvec_count_418887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 39), 'matvec_count', False)
    # Processing the call keyword arguments (line 284)
    kwargs_418888 = {}
    # Getting the type of 'repr' (line 284)
    repr_418886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 34), 'repr', False)
    # Calling repr(args, kwargs) (line 284)
    repr_call_result_418889 = invoke(stypy.reporting.localization.Localization(__file__, 284, 34), repr_418886, *[matvec_count_418887], **kwargs_418888)
    
    # Processing the call keyword arguments (line 284)
    kwargs_418890 = {}
    # Getting the type of 'assert_' (line 284)
    assert__418879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 284)
    assert__call_result_418891 = invoke(stypy.reporting.localization.Localization(__file__, 284, 4), assert__418879, *[result_le_418885, repr_call_result_418889], **kwargs_418890)
    
    
    # ################# End of 'check_precond_inverse(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_precond_inverse' in the type store
    # Getting the type of 'stypy_return_type' (line 243)
    stypy_return_type_418892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_418892)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_precond_inverse'
    return stypy_return_type_418892

# Assigning a type to the variable 'check_precond_inverse' (line 243)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 0), 'check_precond_inverse', check_precond_inverse)

@norecursion
def test_precond_inverse(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_precond_inverse'
    module_type_store = module_type_store.open_function_context('test_precond_inverse', 287, 0, False)
    
    # Passed parameters checking function
    test_precond_inverse.stypy_localization = localization
    test_precond_inverse.stypy_type_of_self = None
    test_precond_inverse.stypy_type_store = module_type_store
    test_precond_inverse.stypy_function_name = 'test_precond_inverse'
    test_precond_inverse.stypy_param_names_list = []
    test_precond_inverse.stypy_varargs_param_name = None
    test_precond_inverse.stypy_kwargs_param_name = None
    test_precond_inverse.stypy_call_defaults = defaults
    test_precond_inverse.stypy_call_varargs = varargs
    test_precond_inverse.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_precond_inverse', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_precond_inverse', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_precond_inverse(...)' code ##################

    
    # Assigning a Attribute to a Name (line 288):
    
    # Assigning a Attribute to a Name (line 288):
    # Getting the type of 'params' (line 288)
    params_418893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 11), 'params')
    # Obtaining the member 'Poisson1D' of a type (line 288)
    Poisson1D_418894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 11), params_418893, 'Poisson1D')
    # Assigning a type to the variable 'case' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'case', Poisson1D_418894)
    
    # Getting the type of 'params' (line 289)
    params_418895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 18), 'params')
    # Obtaining the member 'solvers' of a type (line 289)
    solvers_418896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 18), params_418895, 'solvers')
    # Testing the type of a for loop iterable (line 289)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 289, 4), solvers_418896)
    # Getting the type of the for loop variable (line 289)
    for_loop_var_418897 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 289, 4), solvers_418896)
    # Assigning a type to the variable 'solver' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'solver', for_loop_var_418897)
    # SSA begins for a for statement (line 289)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'solver' (line 290)
    solver_418898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 11), 'solver')
    # Getting the type of 'case' (line 290)
    case_418899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 21), 'case')
    # Obtaining the member 'skip' of a type (line 290)
    skip_418900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 21), case_418899, 'skip')
    # Applying the binary operator 'in' (line 290)
    result_contains_418901 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 11), 'in', solver_418898, skip_418900)
    
    # Testing the type of an if condition (line 290)
    if_condition_418902 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 290, 8), result_contains_418901)
    # Assigning a type to the variable 'if_condition_418902' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'if_condition_418902', if_condition_418902)
    # SSA begins for if statement (line 290)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 290)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'solver' (line 292)
    solver_418903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 11), 'solver')
    # Getting the type of 'qmr' (line 292)
    qmr_418904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 21), 'qmr')
    # Applying the binary operator 'is' (line 292)
    result_is__418905 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 11), 'is', solver_418903, qmr_418904)
    
    # Testing the type of an if condition (line 292)
    if_condition_418906 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 292, 8), result_is__418905)
    # Assigning a type to the variable 'if_condition_418906' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'if_condition_418906', if_condition_418906)
    # SSA begins for if statement (line 292)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 292)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to check_precond_inverse(...): (line 294)
    # Processing the call arguments (line 294)
    # Getting the type of 'solver' (line 294)
    solver_418908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 30), 'solver', False)
    # Getting the type of 'case' (line 294)
    case_418909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 38), 'case', False)
    # Processing the call keyword arguments (line 294)
    kwargs_418910 = {}
    # Getting the type of 'check_precond_inverse' (line 294)
    check_precond_inverse_418907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'check_precond_inverse', False)
    # Calling check_precond_inverse(args, kwargs) (line 294)
    check_precond_inverse_call_result_418911 = invoke(stypy.reporting.localization.Localization(__file__, 294, 8), check_precond_inverse_418907, *[solver_418908, case_418909], **kwargs_418910)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_precond_inverse(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_precond_inverse' in the type store
    # Getting the type of 'stypy_return_type' (line 287)
    stypy_return_type_418912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_418912)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_precond_inverse'
    return stypy_return_type_418912

# Assigning a type to the variable 'test_precond_inverse' (line 287)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 0), 'test_precond_inverse', test_precond_inverse)

@norecursion
def test_gmres_basic(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_gmres_basic'
    module_type_store = module_type_store.open_function_context('test_gmres_basic', 297, 0, False)
    
    # Passed parameters checking function
    test_gmres_basic.stypy_localization = localization
    test_gmres_basic.stypy_type_of_self = None
    test_gmres_basic.stypy_type_store = module_type_store
    test_gmres_basic.stypy_function_name = 'test_gmres_basic'
    test_gmres_basic.stypy_param_names_list = []
    test_gmres_basic.stypy_varargs_param_name = None
    test_gmres_basic.stypy_kwargs_param_name = None
    test_gmres_basic.stypy_call_defaults = defaults
    test_gmres_basic.stypy_call_varargs = varargs
    test_gmres_basic.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_gmres_basic', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_gmres_basic', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_gmres_basic(...)' code ##################

    
    # Assigning a Subscript to a Name (line 298):
    
    # Assigning a Subscript to a Name (line 298):
    
    # Obtaining the type of the subscript
    slice_418913 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 298, 8), None, None, None)
    int_418914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 42), 'int')
    slice_418915 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 298, 8), None, None, int_418914)
    
    # Call to vander(...): (line 298)
    # Processing the call arguments (line 298)
    
    # Call to arange(...): (line 298)
    # Processing the call arguments (line 298)
    int_418920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 28), 'int')
    # Processing the call keyword arguments (line 298)
    kwargs_418921 = {}
    # Getting the type of 'np' (line 298)
    np_418918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 18), 'np', False)
    # Obtaining the member 'arange' of a type (line 298)
    arange_418919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 18), np_418918, 'arange')
    # Calling arange(args, kwargs) (line 298)
    arange_call_result_418922 = invoke(stypy.reporting.localization.Localization(__file__, 298, 18), arange_418919, *[int_418920], **kwargs_418921)
    
    int_418923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 34), 'int')
    # Applying the binary operator '+' (line 298)
    result_add_418924 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 18), '+', arange_call_result_418922, int_418923)
    
    # Processing the call keyword arguments (line 298)
    kwargs_418925 = {}
    # Getting the type of 'np' (line 298)
    np_418916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'np', False)
    # Obtaining the member 'vander' of a type (line 298)
    vander_418917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 8), np_418916, 'vander')
    # Calling vander(args, kwargs) (line 298)
    vander_call_result_418926 = invoke(stypy.reporting.localization.Localization(__file__, 298, 8), vander_418917, *[result_add_418924], **kwargs_418925)
    
    # Obtaining the member '__getitem__' of a type (line 298)
    getitem___418927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 8), vander_call_result_418926, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 298)
    subscript_call_result_418928 = invoke(stypy.reporting.localization.Localization(__file__, 298, 8), getitem___418927, (slice_418913, slice_418915))
    
    # Assigning a type to the variable 'A' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'A', subscript_call_result_418928)
    
    # Assigning a Call to a Name (line 299):
    
    # Assigning a Call to a Name (line 299):
    
    # Call to zeros(...): (line 299)
    # Processing the call arguments (line 299)
    int_418931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 17), 'int')
    # Processing the call keyword arguments (line 299)
    kwargs_418932 = {}
    # Getting the type of 'np' (line 299)
    np_418929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 299)
    zeros_418930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 8), np_418929, 'zeros')
    # Calling zeros(args, kwargs) (line 299)
    zeros_call_result_418933 = invoke(stypy.reporting.localization.Localization(__file__, 299, 8), zeros_418930, *[int_418931], **kwargs_418932)
    
    # Assigning a type to the variable 'b' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'b', zeros_call_result_418933)
    
    # Assigning a Num to a Subscript (line 300):
    
    # Assigning a Num to a Subscript (line 300):
    int_418934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 11), 'int')
    # Getting the type of 'b' (line 300)
    b_418935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'b')
    int_418936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 6), 'int')
    # Storing an element on a container (line 300)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 4), b_418935, (int_418936, int_418934))
    
    # Assigning a Call to a Name (line 301):
    
    # Assigning a Call to a Name (line 301):
    
    # Call to solve(...): (line 301)
    # Processing the call arguments (line 301)
    # Getting the type of 'A' (line 301)
    A_418940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 24), 'A', False)
    # Getting the type of 'b' (line 301)
    b_418941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 27), 'b', False)
    # Processing the call keyword arguments (line 301)
    kwargs_418942 = {}
    # Getting the type of 'np' (line 301)
    np_418937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'np', False)
    # Obtaining the member 'linalg' of a type (line 301)
    linalg_418938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 8), np_418937, 'linalg')
    # Obtaining the member 'solve' of a type (line 301)
    solve_418939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 8), linalg_418938, 'solve')
    # Calling solve(args, kwargs) (line 301)
    solve_call_result_418943 = invoke(stypy.reporting.localization.Localization(__file__, 301, 8), solve_418939, *[A_418940, b_418941], **kwargs_418942)
    
    # Assigning a type to the variable 'x' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'x', solve_call_result_418943)
    
    # Assigning a Call to a Tuple (line 303):
    
    # Assigning a Subscript to a Name (line 303):
    
    # Obtaining the type of the subscript
    int_418944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 4), 'int')
    
    # Call to gmres(...): (line 303)
    # Processing the call arguments (line 303)
    # Getting the type of 'A' (line 303)
    A_418946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 22), 'A', False)
    # Getting the type of 'b' (line 303)
    b_418947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 25), 'b', False)
    # Processing the call keyword arguments (line 303)
    int_418948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 36), 'int')
    keyword_418949 = int_418948
    int_418950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 47), 'int')
    keyword_418951 = int_418950
    kwargs_418952 = {'restart': keyword_418949, 'maxiter': keyword_418951}
    # Getting the type of 'gmres' (line 303)
    gmres_418945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'gmres', False)
    # Calling gmres(args, kwargs) (line 303)
    gmres_call_result_418953 = invoke(stypy.reporting.localization.Localization(__file__, 303, 16), gmres_418945, *[A_418946, b_418947], **kwargs_418952)
    
    # Obtaining the member '__getitem__' of a type (line 303)
    getitem___418954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 4), gmres_call_result_418953, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 303)
    subscript_call_result_418955 = invoke(stypy.reporting.localization.Localization(__file__, 303, 4), getitem___418954, int_418944)
    
    # Assigning a type to the variable 'tuple_var_assignment_417687' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'tuple_var_assignment_417687', subscript_call_result_418955)
    
    # Assigning a Subscript to a Name (line 303):
    
    # Obtaining the type of the subscript
    int_418956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 4), 'int')
    
    # Call to gmres(...): (line 303)
    # Processing the call arguments (line 303)
    # Getting the type of 'A' (line 303)
    A_418958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 22), 'A', False)
    # Getting the type of 'b' (line 303)
    b_418959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 25), 'b', False)
    # Processing the call keyword arguments (line 303)
    int_418960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 36), 'int')
    keyword_418961 = int_418960
    int_418962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 47), 'int')
    keyword_418963 = int_418962
    kwargs_418964 = {'restart': keyword_418961, 'maxiter': keyword_418963}
    # Getting the type of 'gmres' (line 303)
    gmres_418957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'gmres', False)
    # Calling gmres(args, kwargs) (line 303)
    gmres_call_result_418965 = invoke(stypy.reporting.localization.Localization(__file__, 303, 16), gmres_418957, *[A_418958, b_418959], **kwargs_418964)
    
    # Obtaining the member '__getitem__' of a type (line 303)
    getitem___418966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 4), gmres_call_result_418965, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 303)
    subscript_call_result_418967 = invoke(stypy.reporting.localization.Localization(__file__, 303, 4), getitem___418966, int_418956)
    
    # Assigning a type to the variable 'tuple_var_assignment_417688' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'tuple_var_assignment_417688', subscript_call_result_418967)
    
    # Assigning a Name to a Name (line 303):
    # Getting the type of 'tuple_var_assignment_417687' (line 303)
    tuple_var_assignment_417687_418968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'tuple_var_assignment_417687')
    # Assigning a type to the variable 'x_gm' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'x_gm', tuple_var_assignment_417687_418968)
    
    # Assigning a Name to a Name (line 303):
    # Getting the type of 'tuple_var_assignment_417688' (line 303)
    tuple_var_assignment_417688_418969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'tuple_var_assignment_417688')
    # Assigning a type to the variable 'err' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 10), 'err', tuple_var_assignment_417688_418969)
    
    # Call to assert_allclose(...): (line 305)
    # Processing the call arguments (line 305)
    
    # Obtaining the type of the subscript
    int_418971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 25), 'int')
    # Getting the type of 'x_gm' (line 305)
    x_gm_418972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 20), 'x_gm', False)
    # Obtaining the member '__getitem__' of a type (line 305)
    getitem___418973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 20), x_gm_418972, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 305)
    subscript_call_result_418974 = invoke(stypy.reporting.localization.Localization(__file__, 305, 20), getitem___418973, int_418971)
    
    float_418975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 29), 'float')
    # Processing the call keyword arguments (line 305)
    float_418976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 41), 'float')
    keyword_418977 = float_418976
    kwargs_418978 = {'rtol': keyword_418977}
    # Getting the type of 'assert_allclose' (line 305)
    assert_allclose_418970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 305)
    assert_allclose_call_result_418979 = invoke(stypy.reporting.localization.Localization(__file__, 305, 4), assert_allclose_418970, *[subscript_call_result_418974, float_418975], **kwargs_418978)
    
    
    # ################# End of 'test_gmres_basic(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_gmres_basic' in the type store
    # Getting the type of 'stypy_return_type' (line 297)
    stypy_return_type_418980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_418980)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_gmres_basic'
    return stypy_return_type_418980

# Assigning a type to the variable 'test_gmres_basic' (line 297)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 0), 'test_gmres_basic', test_gmres_basic)

@norecursion
def test_reentrancy(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_reentrancy'
    module_type_store = module_type_store.open_function_context('test_reentrancy', 308, 0, False)
    
    # Passed parameters checking function
    test_reentrancy.stypy_localization = localization
    test_reentrancy.stypy_type_of_self = None
    test_reentrancy.stypy_type_store = module_type_store
    test_reentrancy.stypy_function_name = 'test_reentrancy'
    test_reentrancy.stypy_param_names_list = []
    test_reentrancy.stypy_varargs_param_name = None
    test_reentrancy.stypy_kwargs_param_name = None
    test_reentrancy.stypy_call_defaults = defaults
    test_reentrancy.stypy_call_varargs = varargs
    test_reentrancy.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_reentrancy', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_reentrancy', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_reentrancy(...)' code ##################

    
    # Assigning a List to a Name (line 309):
    
    # Assigning a List to a Name (line 309):
    
    # Obtaining an instance of the builtin type 'list' (line 309)
    list_418981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 309)
    # Adding element type (line 309)
    # Getting the type of 'cg' (line 309)
    cg_418982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 21), 'cg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 20), list_418981, cg_418982)
    # Adding element type (line 309)
    # Getting the type of 'cgs' (line 309)
    cgs_418983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 25), 'cgs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 20), list_418981, cgs_418983)
    # Adding element type (line 309)
    # Getting the type of 'bicg' (line 309)
    bicg_418984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 30), 'bicg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 20), list_418981, bicg_418984)
    # Adding element type (line 309)
    # Getting the type of 'bicgstab' (line 309)
    bicgstab_418985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 36), 'bicgstab')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 20), list_418981, bicgstab_418985)
    # Adding element type (line 309)
    # Getting the type of 'gmres' (line 309)
    gmres_418986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 46), 'gmres')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 20), list_418981, gmres_418986)
    # Adding element type (line 309)
    # Getting the type of 'qmr' (line 309)
    qmr_418987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 53), 'qmr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 20), list_418981, qmr_418987)
    
    # Assigning a type to the variable 'non_reentrant' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'non_reentrant', list_418981)
    
    # Assigning a List to a Name (line 310):
    
    # Assigning a List to a Name (line 310):
    
    # Obtaining an instance of the builtin type 'list' (line 310)
    list_418988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 310)
    # Adding element type (line 310)
    # Getting the type of 'lgmres' (line 310)
    lgmres_418989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 17), 'lgmres')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 16), list_418988, lgmres_418989)
    # Adding element type (line 310)
    # Getting the type of 'minres' (line 310)
    minres_418990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 25), 'minres')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 16), list_418988, minres_418990)
    # Adding element type (line 310)
    # Getting the type of 'gcrotmk' (line 310)
    gcrotmk_418991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 33), 'gcrotmk')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 16), list_418988, gcrotmk_418991)
    
    # Assigning a type to the variable 'reentrant' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'reentrant', list_418988)
    
    # Getting the type of 'reentrant' (line 311)
    reentrant_418992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 18), 'reentrant')
    # Getting the type of 'non_reentrant' (line 311)
    non_reentrant_418993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 30), 'non_reentrant')
    # Applying the binary operator '+' (line 311)
    result_add_418994 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 18), '+', reentrant_418992, non_reentrant_418993)
    
    # Testing the type of a for loop iterable (line 311)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 311, 4), result_add_418994)
    # Getting the type of the for loop variable (line 311)
    for_loop_var_418995 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 311, 4), result_add_418994)
    # Assigning a type to the variable 'solver' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'solver', for_loop_var_418995)
    # SSA begins for a for statement (line 311)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to _check_reentrancy(...): (line 312)
    # Processing the call arguments (line 312)
    # Getting the type of 'solver' (line 312)
    solver_418997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 26), 'solver', False)
    
    # Getting the type of 'solver' (line 312)
    solver_418998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 34), 'solver', False)
    # Getting the type of 'reentrant' (line 312)
    reentrant_418999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 44), 'reentrant', False)
    # Applying the binary operator 'in' (line 312)
    result_contains_419000 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 34), 'in', solver_418998, reentrant_418999)
    
    # Processing the call keyword arguments (line 312)
    kwargs_419001 = {}
    # Getting the type of '_check_reentrancy' (line 312)
    _check_reentrancy_418996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), '_check_reentrancy', False)
    # Calling _check_reentrancy(args, kwargs) (line 312)
    _check_reentrancy_call_result_419002 = invoke(stypy.reporting.localization.Localization(__file__, 312, 8), _check_reentrancy_418996, *[solver_418997, result_contains_419000], **kwargs_419001)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_reentrancy(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_reentrancy' in the type store
    # Getting the type of 'stypy_return_type' (line 308)
    stypy_return_type_419003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_419003)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_reentrancy'
    return stypy_return_type_419003

# Assigning a type to the variable 'test_reentrancy' (line 308)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 0), 'test_reentrancy', test_reentrancy)

@norecursion
def _check_reentrancy(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_check_reentrancy'
    module_type_store = module_type_store.open_function_context('_check_reentrancy', 315, 0, False)
    
    # Passed parameters checking function
    _check_reentrancy.stypy_localization = localization
    _check_reentrancy.stypy_type_of_self = None
    _check_reentrancy.stypy_type_store = module_type_store
    _check_reentrancy.stypy_function_name = '_check_reentrancy'
    _check_reentrancy.stypy_param_names_list = ['solver', 'is_reentrant']
    _check_reentrancy.stypy_varargs_param_name = None
    _check_reentrancy.stypy_kwargs_param_name = None
    _check_reentrancy.stypy_call_defaults = defaults
    _check_reentrancy.stypy_call_varargs = varargs
    _check_reentrancy.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_check_reentrancy', ['solver', 'is_reentrant'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_check_reentrancy', localization, ['solver', 'is_reentrant'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_check_reentrancy(...)' code ##################


    @norecursion
    def matvec(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'matvec'
        module_type_store = module_type_store.open_function_context('matvec', 316, 4, False)
        
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

        
        # Assigning a Call to a Name (line 317):
        
        # Assigning a Call to a Name (line 317):
        
        # Call to array(...): (line 317)
        # Processing the call arguments (line 317)
        
        # Obtaining an instance of the builtin type 'list' (line 317)
        list_419006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 317)
        # Adding element type (line 317)
        
        # Obtaining an instance of the builtin type 'list' (line 317)
        list_419007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 317)
        # Adding element type (line 317)
        float_419008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 22), list_419007, float_419008)
        # Adding element type (line 317)
        int_419009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 22), list_419007, int_419009)
        # Adding element type (line 317)
        int_419010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 22), list_419007, int_419010)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 21), list_419006, list_419007)
        # Adding element type (line 317)
        
        # Obtaining an instance of the builtin type 'list' (line 317)
        list_419011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 317)
        # Adding element type (line 317)
        int_419012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 35), list_419011, int_419012)
        # Adding element type (line 317)
        float_419013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 35), list_419011, float_419013)
        # Adding element type (line 317)
        int_419014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 35), list_419011, int_419014)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 21), list_419006, list_419011)
        # Adding element type (line 317)
        
        # Obtaining an instance of the builtin type 'list' (line 317)
        list_419015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 317)
        # Adding element type (line 317)
        int_419016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 48), list_419015, int_419016)
        # Adding element type (line 317)
        int_419017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 48), list_419015, int_419017)
        # Adding element type (line 317)
        float_419018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 48), list_419015, float_419018)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 21), list_419006, list_419015)
        
        # Processing the call keyword arguments (line 317)
        kwargs_419019 = {}
        # Getting the type of 'np' (line 317)
        np_419004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 317)
        array_419005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 12), np_419004, 'array')
        # Calling array(args, kwargs) (line 317)
        array_call_result_419020 = invoke(stypy.reporting.localization.Localization(__file__, 317, 12), array_419005, *[list_419006], **kwargs_419019)
        
        # Assigning a type to the variable 'A' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'A', array_call_result_419020)
        
        # Assigning a Call to a Tuple (line 318):
        
        # Assigning a Subscript to a Name (line 318):
        
        # Obtaining the type of the subscript
        int_419021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 8), 'int')
        
        # Call to solver(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'A' (line 318)
        A_419023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 25), 'A', False)
        # Getting the type of 'x' (line 318)
        x_419024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 28), 'x', False)
        # Processing the call keyword arguments (line 318)
        kwargs_419025 = {}
        # Getting the type of 'solver' (line 318)
        solver_419022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 18), 'solver', False)
        # Calling solver(args, kwargs) (line 318)
        solver_call_result_419026 = invoke(stypy.reporting.localization.Localization(__file__, 318, 18), solver_419022, *[A_419023, x_419024], **kwargs_419025)
        
        # Obtaining the member '__getitem__' of a type (line 318)
        getitem___419027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 8), solver_call_result_419026, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 318)
        subscript_call_result_419028 = invoke(stypy.reporting.localization.Localization(__file__, 318, 8), getitem___419027, int_419021)
        
        # Assigning a type to the variable 'tuple_var_assignment_417689' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'tuple_var_assignment_417689', subscript_call_result_419028)
        
        # Assigning a Subscript to a Name (line 318):
        
        # Obtaining the type of the subscript
        int_419029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 8), 'int')
        
        # Call to solver(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'A' (line 318)
        A_419031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 25), 'A', False)
        # Getting the type of 'x' (line 318)
        x_419032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 28), 'x', False)
        # Processing the call keyword arguments (line 318)
        kwargs_419033 = {}
        # Getting the type of 'solver' (line 318)
        solver_419030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 18), 'solver', False)
        # Calling solver(args, kwargs) (line 318)
        solver_call_result_419034 = invoke(stypy.reporting.localization.Localization(__file__, 318, 18), solver_419030, *[A_419031, x_419032], **kwargs_419033)
        
        # Obtaining the member '__getitem__' of a type (line 318)
        getitem___419035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 8), solver_call_result_419034, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 318)
        subscript_call_result_419036 = invoke(stypy.reporting.localization.Localization(__file__, 318, 8), getitem___419035, int_419029)
        
        # Assigning a type to the variable 'tuple_var_assignment_417690' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'tuple_var_assignment_417690', subscript_call_result_419036)
        
        # Assigning a Name to a Name (line 318):
        # Getting the type of 'tuple_var_assignment_417689' (line 318)
        tuple_var_assignment_417689_419037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'tuple_var_assignment_417689')
        # Assigning a type to the variable 'y' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'y', tuple_var_assignment_417689_419037)
        
        # Assigning a Name to a Name (line 318):
        # Getting the type of 'tuple_var_assignment_417690' (line 318)
        tuple_var_assignment_417690_419038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'tuple_var_assignment_417690')
        # Assigning a type to the variable 'info' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 11), 'info', tuple_var_assignment_417690_419038)
        
        # Call to assert_equal(...): (line 319)
        # Processing the call arguments (line 319)
        # Getting the type of 'info' (line 319)
        info_419040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 21), 'info', False)
        int_419041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 27), 'int')
        # Processing the call keyword arguments (line 319)
        kwargs_419042 = {}
        # Getting the type of 'assert_equal' (line 319)
        assert_equal_419039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 319)
        assert_equal_call_result_419043 = invoke(stypy.reporting.localization.Localization(__file__, 319, 8), assert_equal_419039, *[info_419040, int_419041], **kwargs_419042)
        
        # Getting the type of 'y' (line 320)
        y_419044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 15), 'y')
        # Assigning a type to the variable 'stypy_return_type' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'stypy_return_type', y_419044)
        
        # ################# End of 'matvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'matvec' in the type store
        # Getting the type of 'stypy_return_type' (line 316)
        stypy_return_type_419045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_419045)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'matvec'
        return stypy_return_type_419045

    # Assigning a type to the variable 'matvec' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'matvec', matvec)
    
    # Assigning a Call to a Name (line 321):
    
    # Assigning a Call to a Name (line 321):
    
    # Call to array(...): (line 321)
    # Processing the call arguments (line 321)
    
    # Obtaining an instance of the builtin type 'list' (line 321)
    list_419048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 321)
    # Adding element type (line 321)
    int_419049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 17), list_419048, int_419049)
    # Adding element type (line 321)
    float_419050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 21), 'float')
    int_419051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 24), 'int')
    # Applying the binary operator 'div' (line 321)
    result_div_419052 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 21), 'div', float_419050, int_419051)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 17), list_419048, result_div_419052)
    # Adding element type (line 321)
    float_419053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 27), 'float')
    int_419054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 30), 'int')
    # Applying the binary operator 'div' (line 321)
    result_div_419055 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 27), 'div', float_419053, int_419054)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 17), list_419048, result_div_419055)
    
    # Processing the call keyword arguments (line 321)
    kwargs_419056 = {}
    # Getting the type of 'np' (line 321)
    np_419046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 321)
    array_419047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 8), np_419046, 'array')
    # Calling array(args, kwargs) (line 321)
    array_call_result_419057 = invoke(stypy.reporting.localization.Localization(__file__, 321, 8), array_419047, *[list_419048], **kwargs_419056)
    
    # Assigning a type to the variable 'b' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'b', array_call_result_419057)
    
    # Assigning a Call to a Name (line 322):
    
    # Assigning a Call to a Name (line 322):
    
    # Call to LinearOperator(...): (line 322)
    # Processing the call arguments (line 322)
    
    # Obtaining an instance of the builtin type 'tuple' (line 322)
    tuple_419059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 322)
    # Adding element type (line 322)
    int_419060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 25), tuple_419059, int_419060)
    # Adding element type (line 322)
    int_419061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 25), tuple_419059, int_419061)
    
    # Processing the call keyword arguments (line 322)
    # Getting the type of 'matvec' (line 322)
    matvec_419062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 39), 'matvec', False)
    keyword_419063 = matvec_419062
    # Getting the type of 'matvec' (line 322)
    matvec_419064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 55), 'matvec', False)
    keyword_419065 = matvec_419064
    # Getting the type of 'b' (line 323)
    b_419066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 30), 'b', False)
    # Obtaining the member 'dtype' of a type (line 323)
    dtype_419067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 30), b_419066, 'dtype')
    keyword_419068 = dtype_419067
    kwargs_419069 = {'dtype': keyword_419068, 'rmatvec': keyword_419065, 'matvec': keyword_419063}
    # Getting the type of 'LinearOperator' (line 322)
    LinearOperator_419058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 9), 'LinearOperator', False)
    # Calling LinearOperator(args, kwargs) (line 322)
    LinearOperator_call_result_419070 = invoke(stypy.reporting.localization.Localization(__file__, 322, 9), LinearOperator_419058, *[tuple_419059], **kwargs_419069)
    
    # Assigning a type to the variable 'op' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'op', LinearOperator_call_result_419070)
    
    
    # Getting the type of 'is_reentrant' (line 325)
    is_reentrant_419071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 11), 'is_reentrant')
    # Applying the 'not' unary operator (line 325)
    result_not__419072 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 7), 'not', is_reentrant_419071)
    
    # Testing the type of an if condition (line 325)
    if_condition_419073 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 325, 4), result_not__419072)
    # Assigning a type to the variable 'if_condition_419073' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'if_condition_419073', if_condition_419073)
    # SSA begins for if statement (line 325)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to assert_raises(...): (line 326)
    # Processing the call arguments (line 326)
    # Getting the type of 'RuntimeError' (line 326)
    RuntimeError_419075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 22), 'RuntimeError', False)
    # Getting the type of 'solver' (line 326)
    solver_419076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 36), 'solver', False)
    # Getting the type of 'op' (line 326)
    op_419077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 44), 'op', False)
    # Getting the type of 'b' (line 326)
    b_419078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 48), 'b', False)
    # Processing the call keyword arguments (line 326)
    kwargs_419079 = {}
    # Getting the type of 'assert_raises' (line 326)
    assert_raises_419074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 326)
    assert_raises_call_result_419080 = invoke(stypy.reporting.localization.Localization(__file__, 326, 8), assert_raises_419074, *[RuntimeError_419075, solver_419076, op_419077, b_419078], **kwargs_419079)
    
    # SSA branch for the else part of an if statement (line 325)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 328):
    
    # Assigning a Subscript to a Name (line 328):
    
    # Obtaining the type of the subscript
    int_419081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 8), 'int')
    
    # Call to solver(...): (line 328)
    # Processing the call arguments (line 328)
    # Getting the type of 'op' (line 328)
    op_419083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 25), 'op', False)
    # Getting the type of 'b' (line 328)
    b_419084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 29), 'b', False)
    # Processing the call keyword arguments (line 328)
    kwargs_419085 = {}
    # Getting the type of 'solver' (line 328)
    solver_419082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 18), 'solver', False)
    # Calling solver(args, kwargs) (line 328)
    solver_call_result_419086 = invoke(stypy.reporting.localization.Localization(__file__, 328, 18), solver_419082, *[op_419083, b_419084], **kwargs_419085)
    
    # Obtaining the member '__getitem__' of a type (line 328)
    getitem___419087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 8), solver_call_result_419086, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 328)
    subscript_call_result_419088 = invoke(stypy.reporting.localization.Localization(__file__, 328, 8), getitem___419087, int_419081)
    
    # Assigning a type to the variable 'tuple_var_assignment_417691' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'tuple_var_assignment_417691', subscript_call_result_419088)
    
    # Assigning a Subscript to a Name (line 328):
    
    # Obtaining the type of the subscript
    int_419089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 8), 'int')
    
    # Call to solver(...): (line 328)
    # Processing the call arguments (line 328)
    # Getting the type of 'op' (line 328)
    op_419091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 25), 'op', False)
    # Getting the type of 'b' (line 328)
    b_419092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 29), 'b', False)
    # Processing the call keyword arguments (line 328)
    kwargs_419093 = {}
    # Getting the type of 'solver' (line 328)
    solver_419090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 18), 'solver', False)
    # Calling solver(args, kwargs) (line 328)
    solver_call_result_419094 = invoke(stypy.reporting.localization.Localization(__file__, 328, 18), solver_419090, *[op_419091, b_419092], **kwargs_419093)
    
    # Obtaining the member '__getitem__' of a type (line 328)
    getitem___419095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 8), solver_call_result_419094, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 328)
    subscript_call_result_419096 = invoke(stypy.reporting.localization.Localization(__file__, 328, 8), getitem___419095, int_419089)
    
    # Assigning a type to the variable 'tuple_var_assignment_417692' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'tuple_var_assignment_417692', subscript_call_result_419096)
    
    # Assigning a Name to a Name (line 328):
    # Getting the type of 'tuple_var_assignment_417691' (line 328)
    tuple_var_assignment_417691_419097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'tuple_var_assignment_417691')
    # Assigning a type to the variable 'y' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'y', tuple_var_assignment_417691_419097)
    
    # Assigning a Name to a Name (line 328):
    # Getting the type of 'tuple_var_assignment_417692' (line 328)
    tuple_var_assignment_417692_419098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'tuple_var_assignment_417692')
    # Assigning a type to the variable 'info' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 11), 'info', tuple_var_assignment_417692_419098)
    
    # Call to assert_equal(...): (line 329)
    # Processing the call arguments (line 329)
    # Getting the type of 'info' (line 329)
    info_419100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 21), 'info', False)
    int_419101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 27), 'int')
    # Processing the call keyword arguments (line 329)
    kwargs_419102 = {}
    # Getting the type of 'assert_equal' (line 329)
    assert_equal_419099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 329)
    assert_equal_call_result_419103 = invoke(stypy.reporting.localization.Localization(__file__, 329, 8), assert_equal_419099, *[info_419100, int_419101], **kwargs_419102)
    
    
    # Call to assert_allclose(...): (line 330)
    # Processing the call arguments (line 330)
    # Getting the type of 'y' (line 330)
    y_419105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 24), 'y', False)
    
    # Obtaining an instance of the builtin type 'list' (line 330)
    list_419106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 330)
    # Adding element type (line 330)
    int_419107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 27), list_419106, int_419107)
    # Adding element type (line 330)
    int_419108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 27), list_419106, int_419108)
    # Adding element type (line 330)
    int_419109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 27), list_419106, int_419109)
    
    # Processing the call keyword arguments (line 330)
    kwargs_419110 = {}
    # Getting the type of 'assert_allclose' (line 330)
    assert_allclose_419104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 330)
    assert_allclose_call_result_419111 = invoke(stypy.reporting.localization.Localization(__file__, 330, 8), assert_allclose_419104, *[y_419105, list_419106], **kwargs_419110)
    
    # SSA join for if statement (line 325)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_check_reentrancy(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_check_reentrancy' in the type store
    # Getting the type of 'stypy_return_type' (line 315)
    stypy_return_type_419112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_419112)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_check_reentrancy'
    return stypy_return_type_419112

# Assigning a type to the variable '_check_reentrancy' (line 315)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 0), '_check_reentrancy', _check_reentrancy)
# Declaration of the 'TestQMR' class

class TestQMR(object, ):

    @norecursion
    def test_leftright_precond(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_leftright_precond'
        module_type_store = module_type_store.open_function_context('test_leftright_precond', 336, 4, False)
        # Assigning a type to the variable 'self' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQMR.test_leftright_precond.__dict__.__setitem__('stypy_localization', localization)
        TestQMR.test_leftright_precond.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQMR.test_leftright_precond.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQMR.test_leftright_precond.__dict__.__setitem__('stypy_function_name', 'TestQMR.test_leftright_precond')
        TestQMR.test_leftright_precond.__dict__.__setitem__('stypy_param_names_list', [])
        TestQMR.test_leftright_precond.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQMR.test_leftright_precond.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQMR.test_leftright_precond.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQMR.test_leftright_precond.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQMR.test_leftright_precond.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQMR.test_leftright_precond.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQMR.test_leftright_precond', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_leftright_precond', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_leftright_precond(...)' code ##################

        str_419113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 8), 'str', 'Check that QMR works with left and right preconditioners')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 339, 8))
        
        # 'from scipy.sparse.linalg.dsolve import splu' statement (line 339)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
        import_419114 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 339, 8), 'scipy.sparse.linalg.dsolve')

        if (type(import_419114) is not StypyTypeError):

            if (import_419114 != 'pyd_module'):
                __import__(import_419114)
                sys_modules_419115 = sys.modules[import_419114]
                import_from_module(stypy.reporting.localization.Localization(__file__, 339, 8), 'scipy.sparse.linalg.dsolve', sys_modules_419115.module_type_store, module_type_store, ['splu'])
                nest_module(stypy.reporting.localization.Localization(__file__, 339, 8), __file__, sys_modules_419115, sys_modules_419115.module_type_store, module_type_store)
            else:
                from scipy.sparse.linalg.dsolve import splu

                import_from_module(stypy.reporting.localization.Localization(__file__, 339, 8), 'scipy.sparse.linalg.dsolve', None, module_type_store, ['splu'], [splu])

        else:
            # Assigning a type to the variable 'scipy.sparse.linalg.dsolve' (line 339)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'scipy.sparse.linalg.dsolve', import_419114)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 340, 8))
        
        # 'from scipy.sparse.linalg.interface import LinearOperator' statement (line 340)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
        import_419116 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 340, 8), 'scipy.sparse.linalg.interface')

        if (type(import_419116) is not StypyTypeError):

            if (import_419116 != 'pyd_module'):
                __import__(import_419116)
                sys_modules_419117 = sys.modules[import_419116]
                import_from_module(stypy.reporting.localization.Localization(__file__, 340, 8), 'scipy.sparse.linalg.interface', sys_modules_419117.module_type_store, module_type_store, ['LinearOperator'])
                nest_module(stypy.reporting.localization.Localization(__file__, 340, 8), __file__, sys_modules_419117, sys_modules_419117.module_type_store, module_type_store)
            else:
                from scipy.sparse.linalg.interface import LinearOperator

                import_from_module(stypy.reporting.localization.Localization(__file__, 340, 8), 'scipy.sparse.linalg.interface', None, module_type_store, ['LinearOperator'], [LinearOperator])

        else:
            # Assigning a type to the variable 'scipy.sparse.linalg.interface' (line 340)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'scipy.sparse.linalg.interface', import_419116)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
        
        
        # Assigning a Num to a Name (line 342):
        
        # Assigning a Num to a Name (line 342):
        int_419118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 12), 'int')
        # Assigning a type to the variable 'n' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'n', int_419118)
        
        # Assigning a Call to a Name (line 344):
        
        # Assigning a Call to a Name (line 344):
        
        # Call to ones(...): (line 344)
        # Processing the call arguments (line 344)
        # Getting the type of 'n' (line 344)
        n_419120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 19), 'n', False)
        # Processing the call keyword arguments (line 344)
        kwargs_419121 = {}
        # Getting the type of 'ones' (line 344)
        ones_419119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 14), 'ones', False)
        # Calling ones(args, kwargs) (line 344)
        ones_call_result_419122 = invoke(stypy.reporting.localization.Localization(__file__, 344, 14), ones_419119, *[n_419120], **kwargs_419121)
        
        # Assigning a type to the variable 'dat' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'dat', ones_call_result_419122)
        
        # Assigning a Call to a Name (line 345):
        
        # Assigning a Call to a Name (line 345):
        
        # Call to spdiags(...): (line 345)
        # Processing the call arguments (line 345)
        
        # Obtaining an instance of the builtin type 'list' (line 345)
        list_419124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 345)
        # Adding element type (line 345)
        int_419125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 21), 'int')
        # Getting the type of 'dat' (line 345)
        dat_419126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 24), 'dat', False)
        # Applying the binary operator '*' (line 345)
        result_mul_419127 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 21), '*', int_419125, dat_419126)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 20), list_419124, result_mul_419127)
        # Adding element type (line 345)
        int_419128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 29), 'int')
        # Getting the type of 'dat' (line 345)
        dat_419129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 31), 'dat', False)
        # Applying the binary operator '*' (line 345)
        result_mul_419130 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 29), '*', int_419128, dat_419129)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 20), list_419124, result_mul_419130)
        # Adding element type (line 345)
        
        # Getting the type of 'dat' (line 345)
        dat_419131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 37), 'dat', False)
        # Applying the 'usub' unary operator (line 345)
        result___neg___419132 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 36), 'usub', dat_419131)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 20), list_419124, result___neg___419132)
        
        
        # Obtaining an instance of the builtin type 'list' (line 345)
        list_419133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 345)
        # Adding element type (line 345)
        int_419134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 43), list_419133, int_419134)
        # Adding element type (line 345)
        int_419135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 43), list_419133, int_419135)
        # Adding element type (line 345)
        int_419136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 43), list_419133, int_419136)
        
        # Getting the type of 'n' (line 345)
        n_419137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 52), 'n', False)
        # Getting the type of 'n' (line 345)
        n_419138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 54), 'n', False)
        # Processing the call keyword arguments (line 345)
        kwargs_419139 = {}
        # Getting the type of 'spdiags' (line 345)
        spdiags_419123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'spdiags', False)
        # Calling spdiags(args, kwargs) (line 345)
        spdiags_call_result_419140 = invoke(stypy.reporting.localization.Localization(__file__, 345, 12), spdiags_419123, *[list_419124, list_419133, n_419137, n_419138], **kwargs_419139)
        
        # Assigning a type to the variable 'A' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'A', spdiags_call_result_419140)
        
        # Assigning a Call to a Name (line 346):
        
        # Assigning a Call to a Name (line 346):
        
        # Call to arange(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'n' (line 346)
        n_419142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 19), 'n', False)
        # Processing the call keyword arguments (line 346)
        str_419143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 27), 'str', 'd')
        keyword_419144 = str_419143
        kwargs_419145 = {'dtype': keyword_419144}
        # Getting the type of 'arange' (line 346)
        arange_419141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 346)
        arange_call_result_419146 = invoke(stypy.reporting.localization.Localization(__file__, 346, 12), arange_419141, *[n_419142], **kwargs_419145)
        
        # Assigning a type to the variable 'b' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'b', arange_call_result_419146)
        
        # Assigning a Call to a Name (line 348):
        
        # Assigning a Call to a Name (line 348):
        
        # Call to spdiags(...): (line 348)
        # Processing the call arguments (line 348)
        
        # Obtaining an instance of the builtin type 'list' (line 348)
        list_419148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 348)
        # Adding element type (line 348)
        
        # Getting the type of 'dat' (line 348)
        dat_419149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 22), 'dat', False)
        # Applying the 'usub' unary operator (line 348)
        result___neg___419150 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 21), 'usub', dat_419149)
        
        int_419151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 26), 'int')
        # Applying the binary operator 'div' (line 348)
        result_div_419152 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 21), 'div', result___neg___419150, int_419151)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 20), list_419148, result_div_419152)
        # Adding element type (line 348)
        # Getting the type of 'dat' (line 348)
        dat_419153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 29), 'dat', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 20), list_419148, dat_419153)
        
        
        # Obtaining an instance of the builtin type 'list' (line 348)
        list_419154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 348)
        # Adding element type (line 348)
        int_419155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 35), list_419154, int_419155)
        # Adding element type (line 348)
        int_419156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 35), list_419154, int_419156)
        
        # Getting the type of 'n' (line 348)
        n_419157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 43), 'n', False)
        # Getting the type of 'n' (line 348)
        n_419158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 46), 'n', False)
        # Processing the call keyword arguments (line 348)
        kwargs_419159 = {}
        # Getting the type of 'spdiags' (line 348)
        spdiags_419147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'spdiags', False)
        # Calling spdiags(args, kwargs) (line 348)
        spdiags_call_result_419160 = invoke(stypy.reporting.localization.Localization(__file__, 348, 12), spdiags_419147, *[list_419148, list_419154, n_419157, n_419158], **kwargs_419159)
        
        # Assigning a type to the variable 'L' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'L', spdiags_call_result_419160)
        
        # Assigning a Call to a Name (line 349):
        
        # Assigning a Call to a Name (line 349):
        
        # Call to spdiags(...): (line 349)
        # Processing the call arguments (line 349)
        
        # Obtaining an instance of the builtin type 'list' (line 349)
        list_419162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 349)
        # Adding element type (line 349)
        int_419163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 21), 'int')
        # Getting the type of 'dat' (line 349)
        dat_419164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 23), 'dat', False)
        # Applying the binary operator '*' (line 349)
        result_mul_419165 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 21), '*', int_419163, dat_419164)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 20), list_419162, result_mul_419165)
        # Adding element type (line 349)
        
        # Getting the type of 'dat' (line 349)
        dat_419166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 29), 'dat', False)
        # Applying the 'usub' unary operator (line 349)
        result___neg___419167 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 28), 'usub', dat_419166)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 20), list_419162, result___neg___419167)
        
        
        # Obtaining an instance of the builtin type 'list' (line 349)
        list_419168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 349)
        # Adding element type (line 349)
        int_419169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 35), list_419168, int_419169)
        # Adding element type (line 349)
        int_419170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 35), list_419168, int_419170)
        
        # Getting the type of 'n' (line 349)
        n_419171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 42), 'n', False)
        # Getting the type of 'n' (line 349)
        n_419172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 45), 'n', False)
        # Processing the call keyword arguments (line 349)
        kwargs_419173 = {}
        # Getting the type of 'spdiags' (line 349)
        spdiags_419161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'spdiags', False)
        # Calling spdiags(args, kwargs) (line 349)
        spdiags_call_result_419174 = invoke(stypy.reporting.localization.Localization(__file__, 349, 12), spdiags_419161, *[list_419162, list_419168, n_419171, n_419172], **kwargs_419173)
        
        # Assigning a type to the variable 'U' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'U', spdiags_call_result_419174)
        
        # Call to suppress_warnings(...): (line 351)
        # Processing the call keyword arguments (line 351)
        kwargs_419176 = {}
        # Getting the type of 'suppress_warnings' (line 351)
        suppress_warnings_419175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 351)
        suppress_warnings_call_result_419177 = invoke(stypy.reporting.localization.Localization(__file__, 351, 13), suppress_warnings_419175, *[], **kwargs_419176)
        
        with_419178 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 351, 13), suppress_warnings_call_result_419177, 'with parameter', '__enter__', '__exit__')

        if with_419178:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 351)
            enter___419179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 13), suppress_warnings_call_result_419177, '__enter__')
            with_enter_419180 = invoke(stypy.reporting.localization.Localization(__file__, 351, 13), enter___419179)
            # Assigning a type to the variable 'sup' (line 351)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 13), 'sup', with_enter_419180)
            
            # Call to filter(...): (line 352)
            # Processing the call arguments (line 352)
            # Getting the type of 'SparseEfficiencyWarning' (line 352)
            SparseEfficiencyWarning_419183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 23), 'SparseEfficiencyWarning', False)
            str_419184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 48), 'str', 'splu requires CSC matrix format')
            # Processing the call keyword arguments (line 352)
            kwargs_419185 = {}
            # Getting the type of 'sup' (line 352)
            sup_419181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 352)
            filter_419182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 12), sup_419181, 'filter')
            # Calling filter(args, kwargs) (line 352)
            filter_call_result_419186 = invoke(stypy.reporting.localization.Localization(__file__, 352, 12), filter_419182, *[SparseEfficiencyWarning_419183, str_419184], **kwargs_419185)
            
            
            # Assigning a Call to a Name (line 353):
            
            # Assigning a Call to a Name (line 353):
            
            # Call to splu(...): (line 353)
            # Processing the call arguments (line 353)
            # Getting the type of 'L' (line 353)
            L_419188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 28), 'L', False)
            # Processing the call keyword arguments (line 353)
            kwargs_419189 = {}
            # Getting the type of 'splu' (line 353)
            splu_419187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 23), 'splu', False)
            # Calling splu(args, kwargs) (line 353)
            splu_call_result_419190 = invoke(stypy.reporting.localization.Localization(__file__, 353, 23), splu_419187, *[L_419188], **kwargs_419189)
            
            # Assigning a type to the variable 'L_solver' (line 353)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'L_solver', splu_call_result_419190)
            
            # Assigning a Call to a Name (line 354):
            
            # Assigning a Call to a Name (line 354):
            
            # Call to splu(...): (line 354)
            # Processing the call arguments (line 354)
            # Getting the type of 'U' (line 354)
            U_419192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 28), 'U', False)
            # Processing the call keyword arguments (line 354)
            kwargs_419193 = {}
            # Getting the type of 'splu' (line 354)
            splu_419191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 23), 'splu', False)
            # Calling splu(args, kwargs) (line 354)
            splu_call_result_419194 = invoke(stypy.reporting.localization.Localization(__file__, 354, 23), splu_419191, *[U_419192], **kwargs_419193)
            
            # Assigning a type to the variable 'U_solver' (line 354)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), 'U_solver', splu_call_result_419194)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 351)
            exit___419195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 13), suppress_warnings_call_result_419177, '__exit__')
            with_exit_419196 = invoke(stypy.reporting.localization.Localization(__file__, 351, 13), exit___419195, None, None, None)


        @norecursion
        def L_solve(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'L_solve'
            module_type_store = module_type_store.open_function_context('L_solve', 356, 8, False)
            
            # Passed parameters checking function
            L_solve.stypy_localization = localization
            L_solve.stypy_type_of_self = None
            L_solve.stypy_type_store = module_type_store
            L_solve.stypy_function_name = 'L_solve'
            L_solve.stypy_param_names_list = ['b']
            L_solve.stypy_varargs_param_name = None
            L_solve.stypy_kwargs_param_name = None
            L_solve.stypy_call_defaults = defaults
            L_solve.stypy_call_varargs = varargs
            L_solve.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'L_solve', ['b'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'L_solve', localization, ['b'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'L_solve(...)' code ##################

            
            # Call to solve(...): (line 357)
            # Processing the call arguments (line 357)
            # Getting the type of 'b' (line 357)
            b_419199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 34), 'b', False)
            # Processing the call keyword arguments (line 357)
            kwargs_419200 = {}
            # Getting the type of 'L_solver' (line 357)
            L_solver_419197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 19), 'L_solver', False)
            # Obtaining the member 'solve' of a type (line 357)
            solve_419198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 19), L_solver_419197, 'solve')
            # Calling solve(args, kwargs) (line 357)
            solve_call_result_419201 = invoke(stypy.reporting.localization.Localization(__file__, 357, 19), solve_419198, *[b_419199], **kwargs_419200)
            
            # Assigning a type to the variable 'stypy_return_type' (line 357)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'stypy_return_type', solve_call_result_419201)
            
            # ################# End of 'L_solve(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'L_solve' in the type store
            # Getting the type of 'stypy_return_type' (line 356)
            stypy_return_type_419202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_419202)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'L_solve'
            return stypy_return_type_419202

        # Assigning a type to the variable 'L_solve' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'L_solve', L_solve)

        @norecursion
        def U_solve(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'U_solve'
            module_type_store = module_type_store.open_function_context('U_solve', 359, 8, False)
            
            # Passed parameters checking function
            U_solve.stypy_localization = localization
            U_solve.stypy_type_of_self = None
            U_solve.stypy_type_store = module_type_store
            U_solve.stypy_function_name = 'U_solve'
            U_solve.stypy_param_names_list = ['b']
            U_solve.stypy_varargs_param_name = None
            U_solve.stypy_kwargs_param_name = None
            U_solve.stypy_call_defaults = defaults
            U_solve.stypy_call_varargs = varargs
            U_solve.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'U_solve', ['b'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'U_solve', localization, ['b'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'U_solve(...)' code ##################

            
            # Call to solve(...): (line 360)
            # Processing the call arguments (line 360)
            # Getting the type of 'b' (line 360)
            b_419205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 34), 'b', False)
            # Processing the call keyword arguments (line 360)
            kwargs_419206 = {}
            # Getting the type of 'U_solver' (line 360)
            U_solver_419203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 19), 'U_solver', False)
            # Obtaining the member 'solve' of a type (line 360)
            solve_419204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 19), U_solver_419203, 'solve')
            # Calling solve(args, kwargs) (line 360)
            solve_call_result_419207 = invoke(stypy.reporting.localization.Localization(__file__, 360, 19), solve_419204, *[b_419205], **kwargs_419206)
            
            # Assigning a type to the variable 'stypy_return_type' (line 360)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'stypy_return_type', solve_call_result_419207)
            
            # ################# End of 'U_solve(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'U_solve' in the type store
            # Getting the type of 'stypy_return_type' (line 359)
            stypy_return_type_419208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_419208)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'U_solve'
            return stypy_return_type_419208

        # Assigning a type to the variable 'U_solve' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'U_solve', U_solve)

        @norecursion
        def LT_solve(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'LT_solve'
            module_type_store = module_type_store.open_function_context('LT_solve', 362, 8, False)
            
            # Passed parameters checking function
            LT_solve.stypy_localization = localization
            LT_solve.stypy_type_of_self = None
            LT_solve.stypy_type_store = module_type_store
            LT_solve.stypy_function_name = 'LT_solve'
            LT_solve.stypy_param_names_list = ['b']
            LT_solve.stypy_varargs_param_name = None
            LT_solve.stypy_kwargs_param_name = None
            LT_solve.stypy_call_defaults = defaults
            LT_solve.stypy_call_varargs = varargs
            LT_solve.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'LT_solve', ['b'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'LT_solve', localization, ['b'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'LT_solve(...)' code ##################

            
            # Call to solve(...): (line 363)
            # Processing the call arguments (line 363)
            # Getting the type of 'b' (line 363)
            b_419211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 34), 'b', False)
            str_419212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 36), 'str', 'T')
            # Processing the call keyword arguments (line 363)
            kwargs_419213 = {}
            # Getting the type of 'L_solver' (line 363)
            L_solver_419209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 19), 'L_solver', False)
            # Obtaining the member 'solve' of a type (line 363)
            solve_419210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 19), L_solver_419209, 'solve')
            # Calling solve(args, kwargs) (line 363)
            solve_call_result_419214 = invoke(stypy.reporting.localization.Localization(__file__, 363, 19), solve_419210, *[b_419211, str_419212], **kwargs_419213)
            
            # Assigning a type to the variable 'stypy_return_type' (line 363)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'stypy_return_type', solve_call_result_419214)
            
            # ################# End of 'LT_solve(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'LT_solve' in the type store
            # Getting the type of 'stypy_return_type' (line 362)
            stypy_return_type_419215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_419215)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'LT_solve'
            return stypy_return_type_419215

        # Assigning a type to the variable 'LT_solve' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'LT_solve', LT_solve)

        @norecursion
        def UT_solve(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'UT_solve'
            module_type_store = module_type_store.open_function_context('UT_solve', 365, 8, False)
            
            # Passed parameters checking function
            UT_solve.stypy_localization = localization
            UT_solve.stypy_type_of_self = None
            UT_solve.stypy_type_store = module_type_store
            UT_solve.stypy_function_name = 'UT_solve'
            UT_solve.stypy_param_names_list = ['b']
            UT_solve.stypy_varargs_param_name = None
            UT_solve.stypy_kwargs_param_name = None
            UT_solve.stypy_call_defaults = defaults
            UT_solve.stypy_call_varargs = varargs
            UT_solve.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'UT_solve', ['b'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'UT_solve', localization, ['b'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'UT_solve(...)' code ##################

            
            # Call to solve(...): (line 366)
            # Processing the call arguments (line 366)
            # Getting the type of 'b' (line 366)
            b_419218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 34), 'b', False)
            str_419219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 36), 'str', 'T')
            # Processing the call keyword arguments (line 366)
            kwargs_419220 = {}
            # Getting the type of 'U_solver' (line 366)
            U_solver_419216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 19), 'U_solver', False)
            # Obtaining the member 'solve' of a type (line 366)
            solve_419217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 19), U_solver_419216, 'solve')
            # Calling solve(args, kwargs) (line 366)
            solve_call_result_419221 = invoke(stypy.reporting.localization.Localization(__file__, 366, 19), solve_419217, *[b_419218, str_419219], **kwargs_419220)
            
            # Assigning a type to the variable 'stypy_return_type' (line 366)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'stypy_return_type', solve_call_result_419221)
            
            # ################# End of 'UT_solve(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'UT_solve' in the type store
            # Getting the type of 'stypy_return_type' (line 365)
            stypy_return_type_419222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_419222)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'UT_solve'
            return stypy_return_type_419222

        # Assigning a type to the variable 'UT_solve' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'UT_solve', UT_solve)
        
        # Assigning a Call to a Name (line 368):
        
        # Assigning a Call to a Name (line 368):
        
        # Call to LinearOperator(...): (line 368)
        # Processing the call arguments (line 368)
        
        # Obtaining an instance of the builtin type 'tuple' (line 368)
        tuple_419224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 368)
        # Adding element type (line 368)
        # Getting the type of 'n' (line 368)
        n_419225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 29), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 29), tuple_419224, n_419225)
        # Adding element type (line 368)
        # Getting the type of 'n' (line 368)
        n_419226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 31), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 29), tuple_419224, n_419226)
        
        # Processing the call keyword arguments (line 368)
        # Getting the type of 'L_solve' (line 368)
        L_solve_419227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 42), 'L_solve', False)
        keyword_419228 = L_solve_419227
        # Getting the type of 'LT_solve' (line 368)
        LT_solve_419229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 59), 'LT_solve', False)
        keyword_419230 = LT_solve_419229
        kwargs_419231 = {'rmatvec': keyword_419230, 'matvec': keyword_419228}
        # Getting the type of 'LinearOperator' (line 368)
        LinearOperator_419223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 13), 'LinearOperator', False)
        # Calling LinearOperator(args, kwargs) (line 368)
        LinearOperator_call_result_419232 = invoke(stypy.reporting.localization.Localization(__file__, 368, 13), LinearOperator_419223, *[tuple_419224], **kwargs_419231)
        
        # Assigning a type to the variable 'M1' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'M1', LinearOperator_call_result_419232)
        
        # Assigning a Call to a Name (line 369):
        
        # Assigning a Call to a Name (line 369):
        
        # Call to LinearOperator(...): (line 369)
        # Processing the call arguments (line 369)
        
        # Obtaining an instance of the builtin type 'tuple' (line 369)
        tuple_419234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 369)
        # Adding element type (line 369)
        # Getting the type of 'n' (line 369)
        n_419235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 29), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 29), tuple_419234, n_419235)
        # Adding element type (line 369)
        # Getting the type of 'n' (line 369)
        n_419236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 31), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 29), tuple_419234, n_419236)
        
        # Processing the call keyword arguments (line 369)
        # Getting the type of 'U_solve' (line 369)
        U_solve_419237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 42), 'U_solve', False)
        keyword_419238 = U_solve_419237
        # Getting the type of 'UT_solve' (line 369)
        UT_solve_419239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 59), 'UT_solve', False)
        keyword_419240 = UT_solve_419239
        kwargs_419241 = {'rmatvec': keyword_419240, 'matvec': keyword_419238}
        # Getting the type of 'LinearOperator' (line 369)
        LinearOperator_419233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 13), 'LinearOperator', False)
        # Calling LinearOperator(args, kwargs) (line 369)
        LinearOperator_call_result_419242 = invoke(stypy.reporting.localization.Localization(__file__, 369, 13), LinearOperator_419233, *[tuple_419234], **kwargs_419241)
        
        # Assigning a type to the variable 'M2' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'M2', LinearOperator_call_result_419242)
        
        # Assigning a Call to a Tuple (line 371):
        
        # Assigning a Subscript to a Name (line 371):
        
        # Obtaining the type of the subscript
        int_419243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 8), 'int')
        
        # Call to qmr(...): (line 371)
        # Processing the call arguments (line 371)
        # Getting the type of 'A' (line 371)
        A_419245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 21), 'A', False)
        # Getting the type of 'b' (line 371)
        b_419246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 24), 'b', False)
        # Processing the call keyword arguments (line 371)
        float_419247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 31), 'float')
        keyword_419248 = float_419247
        int_419249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 45), 'int')
        keyword_419250 = int_419249
        # Getting the type of 'M1' (line 371)
        M1_419251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 52), 'M1', False)
        keyword_419252 = M1_419251
        # Getting the type of 'M2' (line 371)
        M2_419253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 59), 'M2', False)
        keyword_419254 = M2_419253
        kwargs_419255 = {'M2': keyword_419254, 'M1': keyword_419252, 'tol': keyword_419248, 'maxiter': keyword_419250}
        # Getting the type of 'qmr' (line 371)
        qmr_419244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 17), 'qmr', False)
        # Calling qmr(args, kwargs) (line 371)
        qmr_call_result_419256 = invoke(stypy.reporting.localization.Localization(__file__, 371, 17), qmr_419244, *[A_419245, b_419246], **kwargs_419255)
        
        # Obtaining the member '__getitem__' of a type (line 371)
        getitem___419257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 8), qmr_call_result_419256, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 371)
        subscript_call_result_419258 = invoke(stypy.reporting.localization.Localization(__file__, 371, 8), getitem___419257, int_419243)
        
        # Assigning a type to the variable 'tuple_var_assignment_417693' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'tuple_var_assignment_417693', subscript_call_result_419258)
        
        # Assigning a Subscript to a Name (line 371):
        
        # Obtaining the type of the subscript
        int_419259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 8), 'int')
        
        # Call to qmr(...): (line 371)
        # Processing the call arguments (line 371)
        # Getting the type of 'A' (line 371)
        A_419261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 21), 'A', False)
        # Getting the type of 'b' (line 371)
        b_419262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 24), 'b', False)
        # Processing the call keyword arguments (line 371)
        float_419263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 31), 'float')
        keyword_419264 = float_419263
        int_419265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 45), 'int')
        keyword_419266 = int_419265
        # Getting the type of 'M1' (line 371)
        M1_419267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 52), 'M1', False)
        keyword_419268 = M1_419267
        # Getting the type of 'M2' (line 371)
        M2_419269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 59), 'M2', False)
        keyword_419270 = M2_419269
        kwargs_419271 = {'M2': keyword_419270, 'M1': keyword_419268, 'tol': keyword_419264, 'maxiter': keyword_419266}
        # Getting the type of 'qmr' (line 371)
        qmr_419260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 17), 'qmr', False)
        # Calling qmr(args, kwargs) (line 371)
        qmr_call_result_419272 = invoke(stypy.reporting.localization.Localization(__file__, 371, 17), qmr_419260, *[A_419261, b_419262], **kwargs_419271)
        
        # Obtaining the member '__getitem__' of a type (line 371)
        getitem___419273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 8), qmr_call_result_419272, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 371)
        subscript_call_result_419274 = invoke(stypy.reporting.localization.Localization(__file__, 371, 8), getitem___419273, int_419259)
        
        # Assigning a type to the variable 'tuple_var_assignment_417694' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'tuple_var_assignment_417694', subscript_call_result_419274)
        
        # Assigning a Name to a Name (line 371):
        # Getting the type of 'tuple_var_assignment_417693' (line 371)
        tuple_var_assignment_417693_419275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'tuple_var_assignment_417693')
        # Assigning a type to the variable 'x' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'x', tuple_var_assignment_417693_419275)
        
        # Assigning a Name to a Name (line 371):
        # Getting the type of 'tuple_var_assignment_417694' (line 371)
        tuple_var_assignment_417694_419276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'tuple_var_assignment_417694')
        # Assigning a type to the variable 'info' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 10), 'info', tuple_var_assignment_417694_419276)
        
        # Call to assert_equal(...): (line 373)
        # Processing the call arguments (line 373)
        # Getting the type of 'info' (line 373)
        info_419278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 21), 'info', False)
        int_419279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 26), 'int')
        # Processing the call keyword arguments (line 373)
        kwargs_419280 = {}
        # Getting the type of 'assert_equal' (line 373)
        assert_equal_419277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 373)
        assert_equal_call_result_419281 = invoke(stypy.reporting.localization.Localization(__file__, 373, 8), assert_equal_419277, *[info_419278, int_419279], **kwargs_419280)
        
        
        # Call to assert_normclose(...): (line 374)
        # Processing the call arguments (line 374)
        # Getting the type of 'A' (line 374)
        A_419283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 25), 'A', False)
        # Getting the type of 'x' (line 374)
        x_419284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 27), 'x', False)
        # Applying the binary operator '*' (line 374)
        result_mul_419285 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 25), '*', A_419283, x_419284)
        
        # Getting the type of 'b' (line 374)
        b_419286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 30), 'b', False)
        # Processing the call keyword arguments (line 374)
        float_419287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 37), 'float')
        keyword_419288 = float_419287
        kwargs_419289 = {'tol': keyword_419288}
        # Getting the type of 'assert_normclose' (line 374)
        assert_normclose_419282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'assert_normclose', False)
        # Calling assert_normclose(args, kwargs) (line 374)
        assert_normclose_call_result_419290 = invoke(stypy.reporting.localization.Localization(__file__, 374, 8), assert_normclose_419282, *[result_mul_419285, b_419286], **kwargs_419289)
        
        
        # ################# End of 'test_leftright_precond(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_leftright_precond' in the type store
        # Getting the type of 'stypy_return_type' (line 336)
        stypy_return_type_419291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_419291)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_leftright_precond'
        return stypy_return_type_419291


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 335, 0, False)
        # Assigning a type to the variable 'self' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQMR.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestQMR' (line 335)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 0), 'TestQMR', TestQMR)
# Declaration of the 'TestGMRES' class

class TestGMRES(object, ):

    @norecursion
    def test_callback(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_callback'
        module_type_store = module_type_store.open_function_context('test_callback', 378, 4, False)
        # Assigning a type to the variable 'self' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGMRES.test_callback.__dict__.__setitem__('stypy_localization', localization)
        TestGMRES.test_callback.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGMRES.test_callback.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGMRES.test_callback.__dict__.__setitem__('stypy_function_name', 'TestGMRES.test_callback')
        TestGMRES.test_callback.__dict__.__setitem__('stypy_param_names_list', [])
        TestGMRES.test_callback.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGMRES.test_callback.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGMRES.test_callback.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGMRES.test_callback.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGMRES.test_callback.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGMRES.test_callback.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGMRES.test_callback', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_callback', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_callback(...)' code ##################


        @norecursion
        def store_residual(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'store_residual'
            module_type_store = module_type_store.open_function_context('store_residual', 380, 8, False)
            
            # Passed parameters checking function
            store_residual.stypy_localization = localization
            store_residual.stypy_type_of_self = None
            store_residual.stypy_type_store = module_type_store
            store_residual.stypy_function_name = 'store_residual'
            store_residual.stypy_param_names_list = ['r', 'rvec']
            store_residual.stypy_varargs_param_name = None
            store_residual.stypy_kwargs_param_name = None
            store_residual.stypy_call_defaults = defaults
            store_residual.stypy_call_varargs = varargs
            store_residual.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'store_residual', ['r', 'rvec'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'store_residual', localization, ['r', 'rvec'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'store_residual(...)' code ##################

            
            # Assigning a Name to a Subscript (line 381):
            
            # Assigning a Name to a Subscript (line 381):
            # Getting the type of 'r' (line 381)
            r_419292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 46), 'r')
            # Getting the type of 'rvec' (line 381)
            rvec_419293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'rvec')
            
            # Call to max(...): (line 381)
            # Processing the call keyword arguments (line 381)
            kwargs_419302 = {}
            
            # Obtaining the type of the subscript
            int_419294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 32), 'int')
            
            # Call to nonzero(...): (line 381)
            # Processing the call keyword arguments (line 381)
            kwargs_419297 = {}
            # Getting the type of 'rvec' (line 381)
            rvec_419295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 17), 'rvec', False)
            # Obtaining the member 'nonzero' of a type (line 381)
            nonzero_419296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 17), rvec_419295, 'nonzero')
            # Calling nonzero(args, kwargs) (line 381)
            nonzero_call_result_419298 = invoke(stypy.reporting.localization.Localization(__file__, 381, 17), nonzero_419296, *[], **kwargs_419297)
            
            # Obtaining the member '__getitem__' of a type (line 381)
            getitem___419299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 17), nonzero_call_result_419298, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 381)
            subscript_call_result_419300 = invoke(stypy.reporting.localization.Localization(__file__, 381, 17), getitem___419299, int_419294)
            
            # Obtaining the member 'max' of a type (line 381)
            max_419301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 17), subscript_call_result_419300, 'max')
            # Calling max(args, kwargs) (line 381)
            max_call_result_419303 = invoke(stypy.reporting.localization.Localization(__file__, 381, 17), max_419301, *[], **kwargs_419302)
            
            int_419304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 41), 'int')
            # Applying the binary operator '+' (line 381)
            result_add_419305 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 17), '+', max_call_result_419303, int_419304)
            
            # Storing an element on a container (line 381)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 12), rvec_419293, (result_add_419305, r_419292))
            
            # ################# End of 'store_residual(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'store_residual' in the type store
            # Getting the type of 'stypy_return_type' (line 380)
            stypy_return_type_419306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_419306)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'store_residual'
            return stypy_return_type_419306

        # Assigning a type to the variable 'store_residual' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'store_residual', store_residual)
        
        # Assigning a Call to a Name (line 384):
        
        # Assigning a Call to a Name (line 384):
        
        # Call to csr_matrix(...): (line 384)
        # Processing the call arguments (line 384)
        
        # Call to array(...): (line 384)
        # Processing the call arguments (line 384)
        
        # Obtaining an instance of the builtin type 'list' (line 384)
        list_419309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 384)
        # Adding element type (line 384)
        
        # Obtaining an instance of the builtin type 'list' (line 384)
        list_419310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 384)
        # Adding element type (line 384)
        int_419311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 30), list_419310, int_419311)
        # Adding element type (line 384)
        int_419312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 30), list_419310, int_419312)
        # Adding element type (line 384)
        int_419313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 30), list_419310, int_419313)
        # Adding element type (line 384)
        int_419314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 30), list_419310, int_419314)
        # Adding element type (line 384)
        int_419315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 30), list_419310, int_419315)
        # Adding element type (line 384)
        int_419316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 30), list_419310, int_419316)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 29), list_419309, list_419310)
        # Adding element type (line 384)
        
        # Obtaining an instance of the builtin type 'list' (line 384)
        list_419317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 384)
        # Adding element type (line 384)
        int_419318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 45), list_419317, int_419318)
        # Adding element type (line 384)
        int_419319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 45), list_419317, int_419319)
        # Adding element type (line 384)
        int_419320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 45), list_419317, int_419320)
        # Adding element type (line 384)
        int_419321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 45), list_419317, int_419321)
        # Adding element type (line 384)
        int_419322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 45), list_419317, int_419322)
        # Adding element type (line 384)
        int_419323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 45), list_419317, int_419323)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 29), list_419309, list_419317)
        # Adding element type (line 384)
        
        # Obtaining an instance of the builtin type 'list' (line 384)
        list_419324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 60), 'list')
        # Adding type elements to the builtin type 'list' instance (line 384)
        # Adding element type (line 384)
        int_419325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 60), list_419324, int_419325)
        # Adding element type (line 384)
        int_419326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 60), list_419324, int_419326)
        # Adding element type (line 384)
        int_419327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 60), list_419324, int_419327)
        # Adding element type (line 384)
        int_419328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 60), list_419324, int_419328)
        # Adding element type (line 384)
        int_419329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 70), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 60), list_419324, int_419329)
        # Adding element type (line 384)
        int_419330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 72), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 60), list_419324, int_419330)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 29), list_419309, list_419324)
        # Adding element type (line 384)
        
        # Obtaining an instance of the builtin type 'list' (line 384)
        list_419331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 75), 'list')
        # Adding type elements to the builtin type 'list' instance (line 384)
        # Adding element type (line 384)
        int_419332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 76), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 75), list_419331, int_419332)
        # Adding element type (line 384)
        int_419333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 78), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 75), list_419331, int_419333)
        # Adding element type (line 384)
        int_419334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 80), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 75), list_419331, int_419334)
        # Adding element type (line 384)
        int_419335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 82), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 75), list_419331, int_419335)
        # Adding element type (line 384)
        int_419336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 85), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 75), list_419331, int_419336)
        # Adding element type (line 384)
        int_419337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 87), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 75), list_419331, int_419337)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 29), list_419309, list_419331)
        # Adding element type (line 384)
        
        # Obtaining an instance of the builtin type 'list' (line 384)
        list_419338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 90), 'list')
        # Adding type elements to the builtin type 'list' instance (line 384)
        # Adding element type (line 384)
        int_419339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 91), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 90), list_419338, int_419339)
        # Adding element type (line 384)
        int_419340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 93), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 90), list_419338, int_419340)
        # Adding element type (line 384)
        int_419341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 95), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 90), list_419338, int_419341)
        # Adding element type (line 384)
        int_419342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 97), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 90), list_419338, int_419342)
        # Adding element type (line 384)
        int_419343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 99), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 90), list_419338, int_419343)
        # Adding element type (line 384)
        int_419344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 102), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 90), list_419338, int_419344)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 29), list_419309, list_419338)
        # Adding element type (line 384)
        
        # Obtaining an instance of the builtin type 'list' (line 384)
        list_419345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 105), 'list')
        # Adding type elements to the builtin type 'list' instance (line 384)
        # Adding element type (line 384)
        int_419346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 106), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 105), list_419345, int_419346)
        # Adding element type (line 384)
        int_419347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 108), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 105), list_419345, int_419347)
        # Adding element type (line 384)
        int_419348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 110), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 105), list_419345, int_419348)
        # Adding element type (line 384)
        int_419349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 112), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 105), list_419345, int_419349)
        # Adding element type (line 384)
        int_419350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 114), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 105), list_419345, int_419350)
        # Adding element type (line 384)
        int_419351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 116), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 105), list_419345, int_419351)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 29), list_419309, list_419345)
        
        # Processing the call keyword arguments (line 384)
        kwargs_419352 = {}
        # Getting the type of 'array' (line 384)
        array_419308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 23), 'array', False)
        # Calling array(args, kwargs) (line 384)
        array_call_result_419353 = invoke(stypy.reporting.localization.Localization(__file__, 384, 23), array_419308, *[list_419309], **kwargs_419352)
        
        # Processing the call keyword arguments (line 384)
        kwargs_419354 = {}
        # Getting the type of 'csr_matrix' (line 384)
        csr_matrix_419307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 384)
        csr_matrix_call_result_419355 = invoke(stypy.reporting.localization.Localization(__file__, 384, 12), csr_matrix_419307, *[array_call_result_419353], **kwargs_419354)
        
        # Assigning a type to the variable 'A' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'A', csr_matrix_call_result_419355)
        
        # Assigning a Call to a Name (line 385):
        
        # Assigning a Call to a Name (line 385):
        
        # Call to ones(...): (line 385)
        # Processing the call arguments (line 385)
        
        # Obtaining an instance of the builtin type 'tuple' (line 385)
        tuple_419357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 385)
        # Adding element type (line 385)
        
        # Obtaining the type of the subscript
        int_419358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 26), 'int')
        # Getting the type of 'A' (line 385)
        A_419359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 18), 'A', False)
        # Obtaining the member 'shape' of a type (line 385)
        shape_419360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 18), A_419359, 'shape')
        # Obtaining the member '__getitem__' of a type (line 385)
        getitem___419361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 18), shape_419360, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 385)
        subscript_call_result_419362 = invoke(stypy.reporting.localization.Localization(__file__, 385, 18), getitem___419361, int_419358)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 18), tuple_419357, subscript_call_result_419362)
        
        # Processing the call keyword arguments (line 385)
        kwargs_419363 = {}
        # Getting the type of 'ones' (line 385)
        ones_419356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'ones', False)
        # Calling ones(args, kwargs) (line 385)
        ones_call_result_419364 = invoke(stypy.reporting.localization.Localization(__file__, 385, 12), ones_419356, *[tuple_419357], **kwargs_419363)
        
        # Assigning a type to the variable 'b' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'b', ones_call_result_419364)
        
        # Assigning a Num to a Name (line 386):
        
        # Assigning a Num to a Name (line 386):
        int_419365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 18), 'int')
        # Assigning a type to the variable 'maxiter' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'maxiter', int_419365)
        
        # Assigning a Call to a Name (line 387):
        
        # Assigning a Call to a Name (line 387):
        
        # Call to zeros(...): (line 387)
        # Processing the call arguments (line 387)
        # Getting the type of 'maxiter' (line 387)
        maxiter_419367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 21), 'maxiter', False)
        int_419368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 29), 'int')
        # Applying the binary operator '+' (line 387)
        result_add_419369 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 21), '+', maxiter_419367, int_419368)
        
        # Processing the call keyword arguments (line 387)
        kwargs_419370 = {}
        # Getting the type of 'zeros' (line 387)
        zeros_419366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 15), 'zeros', False)
        # Calling zeros(args, kwargs) (line 387)
        zeros_call_result_419371 = invoke(stypy.reporting.localization.Localization(__file__, 387, 15), zeros_419366, *[result_add_419369], **kwargs_419370)
        
        # Assigning a type to the variable 'rvec' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'rvec', zeros_call_result_419371)
        
        # Assigning a Num to a Subscript (line 388):
        
        # Assigning a Num to a Subscript (line 388):
        float_419372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 18), 'float')
        # Getting the type of 'rvec' (line 388)
        rvec_419373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'rvec')
        int_419374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 13), 'int')
        # Storing an element on a container (line 388)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 8), rvec_419373, (int_419374, float_419372))
        
        # Assigning a Lambda to a Name (line 389):
        
        # Assigning a Lambda to a Name (line 389):

        @norecursion
        def _stypy_temp_lambda_225(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_225'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_225', 389, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_225.stypy_localization = localization
            _stypy_temp_lambda_225.stypy_type_of_self = None
            _stypy_temp_lambda_225.stypy_type_store = module_type_store
            _stypy_temp_lambda_225.stypy_function_name = '_stypy_temp_lambda_225'
            _stypy_temp_lambda_225.stypy_param_names_list = ['r']
            _stypy_temp_lambda_225.stypy_varargs_param_name = None
            _stypy_temp_lambda_225.stypy_kwargs_param_name = None
            _stypy_temp_lambda_225.stypy_call_defaults = defaults
            _stypy_temp_lambda_225.stypy_call_varargs = varargs
            _stypy_temp_lambda_225.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_225', ['r'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_225', ['r'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to store_residual(...): (line 389)
            # Processing the call arguments (line 389)
            # Getting the type of 'r' (line 389)
            r_419376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 43), 'r', False)
            # Getting the type of 'rvec' (line 389)
            rvec_419377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 46), 'rvec', False)
            # Processing the call keyword arguments (line 389)
            kwargs_419378 = {}
            # Getting the type of 'store_residual' (line 389)
            store_residual_419375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 28), 'store_residual', False)
            # Calling store_residual(args, kwargs) (line 389)
            store_residual_call_result_419379 = invoke(stypy.reporting.localization.Localization(__file__, 389, 28), store_residual_419375, *[r_419376, rvec_419377], **kwargs_419378)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 389)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 19), 'stypy_return_type', store_residual_call_result_419379)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_225' in the type store
            # Getting the type of 'stypy_return_type' (line 389)
            stypy_return_type_419380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_419380)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_225'
            return stypy_return_type_419380

        # Assigning a type to the variable '_stypy_temp_lambda_225' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 19), '_stypy_temp_lambda_225', _stypy_temp_lambda_225)
        # Getting the type of '_stypy_temp_lambda_225' (line 389)
        _stypy_temp_lambda_225_419381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 19), '_stypy_temp_lambda_225')
        # Assigning a type to the variable 'callback' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'callback', _stypy_temp_lambda_225_419381)
        
        # Assigning a Call to a Tuple (line 390):
        
        # Assigning a Subscript to a Name (line 390):
        
        # Obtaining the type of the subscript
        int_419382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 8), 'int')
        
        # Call to gmres(...): (line 390)
        # Processing the call arguments (line 390)
        # Getting the type of 'A' (line 390)
        A_419384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 23), 'A', False)
        # Getting the type of 'b' (line 390)
        b_419385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 26), 'b', False)
        # Processing the call keyword arguments (line 390)
        
        # Call to zeros(...): (line 390)
        # Processing the call arguments (line 390)
        
        # Obtaining the type of the subscript
        int_419387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 46), 'int')
        # Getting the type of 'A' (line 390)
        A_419388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 38), 'A', False)
        # Obtaining the member 'shape' of a type (line 390)
        shape_419389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 38), A_419388, 'shape')
        # Obtaining the member '__getitem__' of a type (line 390)
        getitem___419390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 38), shape_419389, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 390)
        subscript_call_result_419391 = invoke(stypy.reporting.localization.Localization(__file__, 390, 38), getitem___419390, int_419387)
        
        # Processing the call keyword arguments (line 390)
        kwargs_419392 = {}
        # Getting the type of 'zeros' (line 390)
        zeros_419386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 32), 'zeros', False)
        # Calling zeros(args, kwargs) (line 390)
        zeros_call_result_419393 = invoke(stypy.reporting.localization.Localization(__file__, 390, 32), zeros_419386, *[subscript_call_result_419391], **kwargs_419392)
        
        keyword_419394 = zeros_call_result_419393
        float_419395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 55), 'float')
        keyword_419396 = float_419395
        # Getting the type of 'maxiter' (line 390)
        maxiter_419397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 70), 'maxiter', False)
        keyword_419398 = maxiter_419397
        # Getting the type of 'callback' (line 390)
        callback_419399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 88), 'callback', False)
        keyword_419400 = callback_419399
        kwargs_419401 = {'callback': keyword_419400, 'x0': keyword_419394, 'tol': keyword_419396, 'maxiter': keyword_419398}
        # Getting the type of 'gmres' (line 390)
        gmres_419383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 17), 'gmres', False)
        # Calling gmres(args, kwargs) (line 390)
        gmres_call_result_419402 = invoke(stypy.reporting.localization.Localization(__file__, 390, 17), gmres_419383, *[A_419384, b_419385], **kwargs_419401)
        
        # Obtaining the member '__getitem__' of a type (line 390)
        getitem___419403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 8), gmres_call_result_419402, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 390)
        subscript_call_result_419404 = invoke(stypy.reporting.localization.Localization(__file__, 390, 8), getitem___419403, int_419382)
        
        # Assigning a type to the variable 'tuple_var_assignment_417695' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'tuple_var_assignment_417695', subscript_call_result_419404)
        
        # Assigning a Subscript to a Name (line 390):
        
        # Obtaining the type of the subscript
        int_419405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 8), 'int')
        
        # Call to gmres(...): (line 390)
        # Processing the call arguments (line 390)
        # Getting the type of 'A' (line 390)
        A_419407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 23), 'A', False)
        # Getting the type of 'b' (line 390)
        b_419408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 26), 'b', False)
        # Processing the call keyword arguments (line 390)
        
        # Call to zeros(...): (line 390)
        # Processing the call arguments (line 390)
        
        # Obtaining the type of the subscript
        int_419410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 46), 'int')
        # Getting the type of 'A' (line 390)
        A_419411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 38), 'A', False)
        # Obtaining the member 'shape' of a type (line 390)
        shape_419412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 38), A_419411, 'shape')
        # Obtaining the member '__getitem__' of a type (line 390)
        getitem___419413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 38), shape_419412, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 390)
        subscript_call_result_419414 = invoke(stypy.reporting.localization.Localization(__file__, 390, 38), getitem___419413, int_419410)
        
        # Processing the call keyword arguments (line 390)
        kwargs_419415 = {}
        # Getting the type of 'zeros' (line 390)
        zeros_419409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 32), 'zeros', False)
        # Calling zeros(args, kwargs) (line 390)
        zeros_call_result_419416 = invoke(stypy.reporting.localization.Localization(__file__, 390, 32), zeros_419409, *[subscript_call_result_419414], **kwargs_419415)
        
        keyword_419417 = zeros_call_result_419416
        float_419418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 55), 'float')
        keyword_419419 = float_419418
        # Getting the type of 'maxiter' (line 390)
        maxiter_419420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 70), 'maxiter', False)
        keyword_419421 = maxiter_419420
        # Getting the type of 'callback' (line 390)
        callback_419422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 88), 'callback', False)
        keyword_419423 = callback_419422
        kwargs_419424 = {'callback': keyword_419423, 'x0': keyword_419417, 'tol': keyword_419419, 'maxiter': keyword_419421}
        # Getting the type of 'gmres' (line 390)
        gmres_419406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 17), 'gmres', False)
        # Calling gmres(args, kwargs) (line 390)
        gmres_call_result_419425 = invoke(stypy.reporting.localization.Localization(__file__, 390, 17), gmres_419406, *[A_419407, b_419408], **kwargs_419424)
        
        # Obtaining the member '__getitem__' of a type (line 390)
        getitem___419426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 8), gmres_call_result_419425, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 390)
        subscript_call_result_419427 = invoke(stypy.reporting.localization.Localization(__file__, 390, 8), getitem___419426, int_419405)
        
        # Assigning a type to the variable 'tuple_var_assignment_417696' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'tuple_var_assignment_417696', subscript_call_result_419427)
        
        # Assigning a Name to a Name (line 390):
        # Getting the type of 'tuple_var_assignment_417695' (line 390)
        tuple_var_assignment_417695_419428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'tuple_var_assignment_417695')
        # Assigning a type to the variable 'x' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'x', tuple_var_assignment_417695_419428)
        
        # Assigning a Name to a Name (line 390):
        # Getting the type of 'tuple_var_assignment_417696' (line 390)
        tuple_var_assignment_417696_419429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'tuple_var_assignment_417696')
        # Assigning a type to the variable 'flag' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 10), 'flag', tuple_var_assignment_417696_419429)
        
        # Assigning a Call to a Name (line 391):
        
        # Assigning a Call to a Name (line 391):
        
        # Call to max(...): (line 391)
        # Processing the call arguments (line 391)
        
        # Call to abs(...): (line 391)
        # Processing the call arguments (line 391)
        # Getting the type of 'rvec' (line 391)
        rvec_419432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 24), 'rvec', False)
        
        # Call to array(...): (line 391)
        # Processing the call arguments (line 391)
        
        # Obtaining an instance of the builtin type 'list' (line 391)
        list_419434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 391)
        # Adding element type (line 391)
        float_419435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 37), list_419434, float_419435)
        # Adding element type (line 391)
        float_419436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 37), list_419434, float_419436)
        
        # Processing the call keyword arguments (line 391)
        kwargs_419437 = {}
        # Getting the type of 'array' (line 391)
        array_419433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 31), 'array', False)
        # Calling array(args, kwargs) (line 391)
        array_call_result_419438 = invoke(stypy.reporting.localization.Localization(__file__, 391, 31), array_419433, *[list_419434], **kwargs_419437)
        
        # Applying the binary operator '-' (line 391)
        result_sub_419439 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 24), '-', rvec_419432, array_call_result_419438)
        
        # Processing the call keyword arguments (line 391)
        kwargs_419440 = {}
        # Getting the type of 'abs' (line 391)
        abs_419431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 19), 'abs', False)
        # Calling abs(args, kwargs) (line 391)
        abs_call_result_419441 = invoke(stypy.reporting.localization.Localization(__file__, 391, 19), abs_419431, *[result_sub_419439], **kwargs_419440)
        
        # Processing the call keyword arguments (line 391)
        kwargs_419442 = {}
        # Getting the type of 'max' (line 391)
        max_419430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 15), 'max', False)
        # Calling max(args, kwargs) (line 391)
        max_call_result_419443 = invoke(stypy.reporting.localization.Localization(__file__, 391, 15), max_419430, *[abs_call_result_419441], **kwargs_419442)
        
        # Assigning a type to the variable 'diff' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'diff', max_call_result_419443)
        
        # Call to assert_(...): (line 392)
        # Processing the call arguments (line 392)
        
        # Getting the type of 'diff' (line 392)
        diff_419445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 16), 'diff', False)
        float_419446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 23), 'float')
        # Applying the binary operator '<' (line 392)
        result_lt_419447 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 16), '<', diff_419445, float_419446)
        
        # Processing the call keyword arguments (line 392)
        kwargs_419448 = {}
        # Getting the type of 'assert_' (line 392)
        assert__419444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 392)
        assert__call_result_419449 = invoke(stypy.reporting.localization.Localization(__file__, 392, 8), assert__419444, *[result_lt_419447], **kwargs_419448)
        
        
        # ################# End of 'test_callback(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_callback' in the type store
        # Getting the type of 'stypy_return_type' (line 378)
        stypy_return_type_419450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_419450)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_callback'
        return stypy_return_type_419450


    @norecursion
    def test_abi(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_abi'
        module_type_store = module_type_store.open_function_context('test_abi', 394, 4, False)
        # Assigning a type to the variable 'self' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGMRES.test_abi.__dict__.__setitem__('stypy_localization', localization)
        TestGMRES.test_abi.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGMRES.test_abi.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGMRES.test_abi.__dict__.__setitem__('stypy_function_name', 'TestGMRES.test_abi')
        TestGMRES.test_abi.__dict__.__setitem__('stypy_param_names_list', [])
        TestGMRES.test_abi.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGMRES.test_abi.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGMRES.test_abi.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGMRES.test_abi.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGMRES.test_abi.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGMRES.test_abi.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGMRES.test_abi', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_abi', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_abi(...)' code ##################

        
        # Assigning a Call to a Name (line 396):
        
        # Assigning a Call to a Name (line 396):
        
        # Call to eye(...): (line 396)
        # Processing the call arguments (line 396)
        int_419452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 16), 'int')
        # Processing the call keyword arguments (line 396)
        kwargs_419453 = {}
        # Getting the type of 'eye' (line 396)
        eye_419451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'eye', False)
        # Calling eye(args, kwargs) (line 396)
        eye_call_result_419454 = invoke(stypy.reporting.localization.Localization(__file__, 396, 12), eye_419451, *[int_419452], **kwargs_419453)
        
        # Assigning a type to the variable 'A' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'A', eye_call_result_419454)
        
        # Assigning a Call to a Name (line 397):
        
        # Assigning a Call to a Name (line 397):
        
        # Call to ones(...): (line 397)
        # Processing the call arguments (line 397)
        int_419456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 17), 'int')
        # Processing the call keyword arguments (line 397)
        kwargs_419457 = {}
        # Getting the type of 'ones' (line 397)
        ones_419455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'ones', False)
        # Calling ones(args, kwargs) (line 397)
        ones_call_result_419458 = invoke(stypy.reporting.localization.Localization(__file__, 397, 12), ones_419455, *[int_419456], **kwargs_419457)
        
        # Assigning a type to the variable 'b' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'b', ones_call_result_419458)
        
        # Assigning a Call to a Tuple (line 398):
        
        # Assigning a Subscript to a Name (line 398):
        
        # Obtaining the type of the subscript
        int_419459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 8), 'int')
        
        # Call to gmres(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'A' (line 398)
        A_419461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 28), 'A', False)
        # Getting the type of 'b' (line 398)
        b_419462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 31), 'b', False)
        # Processing the call keyword arguments (line 398)
        kwargs_419463 = {}
        # Getting the type of 'gmres' (line 398)
        gmres_419460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 22), 'gmres', False)
        # Calling gmres(args, kwargs) (line 398)
        gmres_call_result_419464 = invoke(stypy.reporting.localization.Localization(__file__, 398, 22), gmres_419460, *[A_419461, b_419462], **kwargs_419463)
        
        # Obtaining the member '__getitem__' of a type (line 398)
        getitem___419465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 8), gmres_call_result_419464, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 398)
        subscript_call_result_419466 = invoke(stypy.reporting.localization.Localization(__file__, 398, 8), getitem___419465, int_419459)
        
        # Assigning a type to the variable 'tuple_var_assignment_417697' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'tuple_var_assignment_417697', subscript_call_result_419466)
        
        # Assigning a Subscript to a Name (line 398):
        
        # Obtaining the type of the subscript
        int_419467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 8), 'int')
        
        # Call to gmres(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'A' (line 398)
        A_419469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 28), 'A', False)
        # Getting the type of 'b' (line 398)
        b_419470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 31), 'b', False)
        # Processing the call keyword arguments (line 398)
        kwargs_419471 = {}
        # Getting the type of 'gmres' (line 398)
        gmres_419468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 22), 'gmres', False)
        # Calling gmres(args, kwargs) (line 398)
        gmres_call_result_419472 = invoke(stypy.reporting.localization.Localization(__file__, 398, 22), gmres_419468, *[A_419469, b_419470], **kwargs_419471)
        
        # Obtaining the member '__getitem__' of a type (line 398)
        getitem___419473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 8), gmres_call_result_419472, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 398)
        subscript_call_result_419474 = invoke(stypy.reporting.localization.Localization(__file__, 398, 8), getitem___419473, int_419467)
        
        # Assigning a type to the variable 'tuple_var_assignment_417698' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'tuple_var_assignment_417698', subscript_call_result_419474)
        
        # Assigning a Name to a Name (line 398):
        # Getting the type of 'tuple_var_assignment_417697' (line 398)
        tuple_var_assignment_417697_419475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'tuple_var_assignment_417697')
        # Assigning a type to the variable 'r_x' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'r_x', tuple_var_assignment_417697_419475)
        
        # Assigning a Name to a Name (line 398):
        # Getting the type of 'tuple_var_assignment_417698' (line 398)
        tuple_var_assignment_417698_419476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'tuple_var_assignment_417698')
        # Assigning a type to the variable 'r_info' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 13), 'r_info', tuple_var_assignment_417698_419476)
        
        # Assigning a Call to a Name (line 399):
        
        # Assigning a Call to a Name (line 399):
        
        # Call to astype(...): (line 399)
        # Processing the call arguments (line 399)
        # Getting the type of 'complex' (line 399)
        complex_419479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 25), 'complex', False)
        # Processing the call keyword arguments (line 399)
        kwargs_419480 = {}
        # Getting the type of 'r_x' (line 399)
        r_x_419477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 14), 'r_x', False)
        # Obtaining the member 'astype' of a type (line 399)
        astype_419478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 14), r_x_419477, 'astype')
        # Calling astype(args, kwargs) (line 399)
        astype_call_result_419481 = invoke(stypy.reporting.localization.Localization(__file__, 399, 14), astype_419478, *[complex_419479], **kwargs_419480)
        
        # Assigning a type to the variable 'r_x' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'r_x', astype_call_result_419481)
        
        # Assigning a Call to a Tuple (line 401):
        
        # Assigning a Subscript to a Name (line 401):
        
        # Obtaining the type of the subscript
        int_419482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 8), 'int')
        
        # Call to gmres(...): (line 401)
        # Processing the call arguments (line 401)
        
        # Call to astype(...): (line 401)
        # Processing the call arguments (line 401)
        # Getting the type of 'complex' (line 401)
        complex_419486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 33), 'complex', False)
        # Processing the call keyword arguments (line 401)
        kwargs_419487 = {}
        # Getting the type of 'A' (line 401)
        A_419484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 24), 'A', False)
        # Obtaining the member 'astype' of a type (line 401)
        astype_419485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 24), A_419484, 'astype')
        # Calling astype(args, kwargs) (line 401)
        astype_call_result_419488 = invoke(stypy.reporting.localization.Localization(__file__, 401, 24), astype_419485, *[complex_419486], **kwargs_419487)
        
        
        # Call to astype(...): (line 401)
        # Processing the call arguments (line 401)
        # Getting the type of 'complex' (line 401)
        complex_419491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 52), 'complex', False)
        # Processing the call keyword arguments (line 401)
        kwargs_419492 = {}
        # Getting the type of 'b' (line 401)
        b_419489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 43), 'b', False)
        # Obtaining the member 'astype' of a type (line 401)
        astype_419490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 43), b_419489, 'astype')
        # Calling astype(args, kwargs) (line 401)
        astype_call_result_419493 = invoke(stypy.reporting.localization.Localization(__file__, 401, 43), astype_419490, *[complex_419491], **kwargs_419492)
        
        # Processing the call keyword arguments (line 401)
        kwargs_419494 = {}
        # Getting the type of 'gmres' (line 401)
        gmres_419483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 18), 'gmres', False)
        # Calling gmres(args, kwargs) (line 401)
        gmres_call_result_419495 = invoke(stypy.reporting.localization.Localization(__file__, 401, 18), gmres_419483, *[astype_call_result_419488, astype_call_result_419493], **kwargs_419494)
        
        # Obtaining the member '__getitem__' of a type (line 401)
        getitem___419496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 8), gmres_call_result_419495, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 401)
        subscript_call_result_419497 = invoke(stypy.reporting.localization.Localization(__file__, 401, 8), getitem___419496, int_419482)
        
        # Assigning a type to the variable 'tuple_var_assignment_417699' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'tuple_var_assignment_417699', subscript_call_result_419497)
        
        # Assigning a Subscript to a Name (line 401):
        
        # Obtaining the type of the subscript
        int_419498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 8), 'int')
        
        # Call to gmres(...): (line 401)
        # Processing the call arguments (line 401)
        
        # Call to astype(...): (line 401)
        # Processing the call arguments (line 401)
        # Getting the type of 'complex' (line 401)
        complex_419502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 33), 'complex', False)
        # Processing the call keyword arguments (line 401)
        kwargs_419503 = {}
        # Getting the type of 'A' (line 401)
        A_419500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 24), 'A', False)
        # Obtaining the member 'astype' of a type (line 401)
        astype_419501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 24), A_419500, 'astype')
        # Calling astype(args, kwargs) (line 401)
        astype_call_result_419504 = invoke(stypy.reporting.localization.Localization(__file__, 401, 24), astype_419501, *[complex_419502], **kwargs_419503)
        
        
        # Call to astype(...): (line 401)
        # Processing the call arguments (line 401)
        # Getting the type of 'complex' (line 401)
        complex_419507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 52), 'complex', False)
        # Processing the call keyword arguments (line 401)
        kwargs_419508 = {}
        # Getting the type of 'b' (line 401)
        b_419505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 43), 'b', False)
        # Obtaining the member 'astype' of a type (line 401)
        astype_419506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 43), b_419505, 'astype')
        # Calling astype(args, kwargs) (line 401)
        astype_call_result_419509 = invoke(stypy.reporting.localization.Localization(__file__, 401, 43), astype_419506, *[complex_419507], **kwargs_419508)
        
        # Processing the call keyword arguments (line 401)
        kwargs_419510 = {}
        # Getting the type of 'gmres' (line 401)
        gmres_419499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 18), 'gmres', False)
        # Calling gmres(args, kwargs) (line 401)
        gmres_call_result_419511 = invoke(stypy.reporting.localization.Localization(__file__, 401, 18), gmres_419499, *[astype_call_result_419504, astype_call_result_419509], **kwargs_419510)
        
        # Obtaining the member '__getitem__' of a type (line 401)
        getitem___419512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 8), gmres_call_result_419511, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 401)
        subscript_call_result_419513 = invoke(stypy.reporting.localization.Localization(__file__, 401, 8), getitem___419512, int_419498)
        
        # Assigning a type to the variable 'tuple_var_assignment_417700' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'tuple_var_assignment_417700', subscript_call_result_419513)
        
        # Assigning a Name to a Name (line 401):
        # Getting the type of 'tuple_var_assignment_417699' (line 401)
        tuple_var_assignment_417699_419514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'tuple_var_assignment_417699')
        # Assigning a type to the variable 'x' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'x', tuple_var_assignment_417699_419514)
        
        # Assigning a Name to a Name (line 401):
        # Getting the type of 'tuple_var_assignment_417700' (line 401)
        tuple_var_assignment_417700_419515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'tuple_var_assignment_417700')
        # Assigning a type to the variable 'info' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 11), 'info', tuple_var_assignment_417700_419515)
        
        # Call to assert_(...): (line 403)
        # Processing the call arguments (line 403)
        
        # Call to iscomplexobj(...): (line 403)
        # Processing the call arguments (line 403)
        # Getting the type of 'x' (line 403)
        x_419518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 29), 'x', False)
        # Processing the call keyword arguments (line 403)
        kwargs_419519 = {}
        # Getting the type of 'iscomplexobj' (line 403)
        iscomplexobj_419517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 16), 'iscomplexobj', False)
        # Calling iscomplexobj(args, kwargs) (line 403)
        iscomplexobj_call_result_419520 = invoke(stypy.reporting.localization.Localization(__file__, 403, 16), iscomplexobj_419517, *[x_419518], **kwargs_419519)
        
        # Processing the call keyword arguments (line 403)
        kwargs_419521 = {}
        # Getting the type of 'assert_' (line 403)
        assert__419516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 403)
        assert__call_result_419522 = invoke(stypy.reporting.localization.Localization(__file__, 403, 8), assert__419516, *[iscomplexobj_call_result_419520], **kwargs_419521)
        
        
        # Call to assert_allclose(...): (line 404)
        # Processing the call arguments (line 404)
        # Getting the type of 'r_x' (line 404)
        r_x_419524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 24), 'r_x', False)
        # Getting the type of 'x' (line 404)
        x_419525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 29), 'x', False)
        # Processing the call keyword arguments (line 404)
        kwargs_419526 = {}
        # Getting the type of 'assert_allclose' (line 404)
        assert_allclose_419523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 404)
        assert_allclose_call_result_419527 = invoke(stypy.reporting.localization.Localization(__file__, 404, 8), assert_allclose_419523, *[r_x_419524, x_419525], **kwargs_419526)
        
        
        # Call to assert_(...): (line 405)
        # Processing the call arguments (line 405)
        
        # Getting the type of 'r_info' (line 405)
        r_info_419529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'r_info', False)
        # Getting the type of 'info' (line 405)
        info_419530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 26), 'info', False)
        # Applying the binary operator '==' (line 405)
        result_eq_419531 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 16), '==', r_info_419529, info_419530)
        
        # Processing the call keyword arguments (line 405)
        kwargs_419532 = {}
        # Getting the type of 'assert_' (line 405)
        assert__419528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 405)
        assert__call_result_419533 = invoke(stypy.reporting.localization.Localization(__file__, 405, 8), assert__419528, *[result_eq_419531], **kwargs_419532)
        
        
        # ################# End of 'test_abi(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_abi' in the type store
        # Getting the type of 'stypy_return_type' (line 394)
        stypy_return_type_419534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_419534)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_abi'
        return stypy_return_type_419534


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 377, 0, False)
        # Assigning a type to the variable 'self' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGMRES.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestGMRES' (line 377)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 0), 'TestGMRES', TestGMRES)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
