
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Routines for removing redundant (linearly dependent) equations from linear
3: programming equality constraints.
4: '''
5: # Author: Matt Haberland
6: 
7: from __future__ import division, print_function, absolute_import
8: import numpy as np
9: from scipy.linalg import svd
10: import scipy
11: 
12: 
13: def _row_count(A):
14:     '''
15:     Counts the number of nonzeros in each row of input array A.
16:     Nonzeros are defined as any element with absolute value greater than
17:     tol = 1e-13. This value should probably be an input to the function.
18: 
19:     Parameters
20:     ----------
21:     A : 2-D array
22:         An array representing a matrix
23: 
24:     Returns
25:     -------
26:     rowcount : 1-D array
27:         Number of nonzeros in each row of A
28: 
29:     '''
30:     tol = 1e-13
31:     return np.array((abs(A) > tol).sum(axis=1)).flatten()
32: 
33: 
34: def _get_densest(A, eligibleRows):
35:     '''
36:     Returns the index of the densest row of A. Ignores rows that are not
37:     eligible for consideration.
38: 
39:     Parameters
40:     ----------
41:     A : 2-D array
42:         An array representing a matrix
43:     eligibleRows : 1-D logical array
44:         Values indicate whether the corresponding row of A is eligible
45:         to be considered
46: 
47:     Returns
48:     -------
49:     i_densest : int
50:         Index of the densest row in A eligible for consideration
51: 
52:     '''
53:     rowCounts = _row_count(A)
54:     return np.argmax(rowCounts * eligibleRows)
55: 
56: 
57: def _remove_zero_rows(A, b):
58:     '''
59:     Eliminates trivial equations from system of equations defined by Ax = b
60:    and identifies trivial infeasibilities
61: 
62:     Parameters
63:     ----------
64:     A : 2-D array
65:         An array representing the left-hand side of a system of equations
66:     b : 1-D array
67:         An array representing the right-hand side of a system of equations
68: 
69:     Returns
70:     -------
71:     A : 2-D array
72:         An array representing the left-hand side of a system of equations
73:     b : 1-D array
74:         An array representing the right-hand side of a system of equations
75:     status: int
76:         An integer indicating the status of the removal operation
77:         0: No infeasibility identified
78:         2: Trivially infeasible
79:     message : str
80:         A string descriptor of the exit status of the optimization.
81: 
82:     '''
83:     status = 0
84:     message = ""
85:     i_zero = _row_count(A) == 0
86:     A = A[np.logical_not(i_zero), :]
87:     if not(np.allclose(b[i_zero], 0)):
88:         status = 2
89:         message = "There is a zero row in A_eq with a nonzero corresponding " \
90:                   "entry in b_eq. The problem is infeasible."
91:     b = b[np.logical_not(i_zero)]
92:     return A, b, status, message
93: 
94: 
95: def bg_update_dense(plu, perm_r, v, j):
96:     LU, p = plu
97: 
98:     u = scipy.linalg.solve_triangular(LU, v[perm_r], lower=True,
99:                                       unit_diagonal=True)
100:     LU[:j+1, j] = u[:j+1]
101:     l = u[j+1:]
102:     piv = LU[j, j]
103:     LU[j+1:, j] += (l/piv)
104:     return LU, p
105: 
106: 
107: def _remove_redundancy_dense(A, rhs):
108:     '''
109:     Eliminates redundant equations from system of equations defined by Ax = b
110:     and identifies infeasibilities.
111: 
112:     Parameters
113:     ----------
114:     A : 2-D sparse matrix
115:         An matrix representing the left-hand side of a system of equations
116:     rhs : 1-D array
117:         An array representing the right-hand side of a system of equations
118: 
119:     Returns
120:     ----------
121:     A : 2-D sparse matrix
122:         A matrix representing the left-hand side of a system of equations
123:     rhs : 1-D array
124:         An array representing the right-hand side of a system of equations
125:     status: int
126:         An integer indicating the status of the system
127:         0: No infeasibility identified
128:         2: Trivially infeasible
129:     message : str
130:         A string descriptor of the exit status of the optimization.
131: 
132:     References
133:     ----------
134:     .. [2] Andersen, Erling D. "Finding all linearly dependent rows in
135:            large-scale linear programming." Optimization Methods and Software
136:            6.3 (1995): 219-227.
137: 
138:     '''
139:     tolapiv = 1e-8
140:     tolprimal = 1e-8
141:     status = 0
142:     message = ""
143:     inconsistent = ("There is a linear combination of rows of A_eq that "
144:                     "results in zero, suggesting a redundant constraint. "
145:                     "However the same linear combination of b_eq is "
146:                     "nonzero, suggesting that the constraints conflict "
147:                     "and the problem is infeasible.")
148:     A, rhs, status, message = _remove_zero_rows(A, rhs)
149: 
150:     if status != 0:
151:         return A, rhs, status, message
152: 
153:     m, n = A.shape
154: 
155:     v = list(range(m))      # Artificial column indices.
156:     b = list(v)             # Basis column indices.
157:     # This is better as a list than a set because column order of basis matrix
158:     # needs to be consistent.
159:     k = set(range(m, m+n))  # Structural column indices.
160:     d = []                  # Indices of dependent rows
161:     lu = None
162:     perm_r = None
163: 
164:     A_orig = A
165:     A = np.hstack((np.eye(m), A))
166:     e = np.zeros(m)
167: 
168:     # Implements basic algorithm from [2]
169:     # Uses some of the suggested improvements (removing zero rows and
170:     # Bartels-Golub update idea).
171:     # Removing column singletons would be easy, but it is not as important
172:     # because the procedure is performed only on the equality constraint
173:     # matrix from the original problem - not on the canonical form matrix,
174:     # which would have many more column singletons due to slack variables
175:     # from the inequality constraints.
176:     # The thoughts on "crashing" the initial basis sound useful, but the
177:     # description of the procedure seems to assume a lot of familiarity with
178:     # the subject; it is not very explicit. I already went through enough
179:     # trouble getting the basic algorithm working, so I was not interested in
180:     # trying to decipher this, too. (Overall, the paper is fraught with
181:     # mistakes and ambiguities - which is strange, because the rest of
182:     # Andersen's papers are quite good.)
183: 
184:     B = A[:, b]
185:     for i in v:
186: 
187:         e[i] = 1
188:         if i > 0:
189:             e[i-1] = 0
190: 
191:         try:  # fails for i==0 and any time it gets ill-conditioned
192:             j = b[i-1]
193:             lu = bg_update_dense(lu, perm_r, A[:, j], i-1)
194:         except:
195:             lu = scipy.linalg.lu_factor(B)
196:             LU, p = lu
197:             perm_r = list(range(m))
198:             for i1, i2 in enumerate(p):
199:                 perm_r[i1], perm_r[i2] = perm_r[i2], perm_r[i1]
200: 
201:         pi = scipy.linalg.lu_solve(lu, e, trans=1)
202: 
203:         # not efficient, but this is not the time sink...
204:         js = np.array(list(k-set(b)))
205:         batch = 50
206:         dependent = True
207: 
208:         # This is a tiny bit faster than looping over columns indivually,
209:         # like for j in js: if abs(A[:,j].transpose().dot(pi)) > tolapiv:
210:         for j_index in range(0, len(js), batch):
211:             j_indices = js[np.arange(j_index, min(j_index+batch, len(js)))]
212: 
213:             c = abs(A[:, j_indices].transpose().dot(pi))
214:             if (c > tolapiv).any():
215:                 j = js[j_index + np.argmax(c)]  # very independent column
216:                 B[:, i] = A[:, j]
217:                 b[i] = j
218:                 dependent = False
219:                 break
220:         if dependent:
221:             bibar = pi.T.dot(rhs.reshape(-1, 1))
222:             bnorm = np.linalg.norm(rhs)
223:             if abs(bibar)/(1+bnorm) > tolprimal:  # inconsistent
224:                 status = 2
225:                 message = inconsistent
226:                 return A_orig, rhs, status, message
227:             else:  # dependent
228:                 d.append(i)
229: 
230:     keep = set(range(m))
231:     keep = list(keep - set(d))
232:     return A_orig[keep, :], rhs[keep], status, message
233: 
234: 
235: def _remove_redundancy_sparse(A, rhs):
236:     '''
237:     Eliminates redundant equations from system of equations defined by Ax = b
238:     and identifies infeasibilities.
239: 
240:     Parameters
241:     ----------
242:     A : 2-D sparse matrix
243:         An matrix representing the left-hand side of a system of equations
244:     rhs : 1-D array
245:         An array representing the right-hand side of a system of equations
246: 
247:     Returns
248:     -------
249:     A : 2-D sparse matrix
250:         A matrix representing the left-hand side of a system of equations
251:     rhs : 1-D array
252:         An array representing the right-hand side of a system of equations
253:     status: int
254:         An integer indicating the status of the system
255:         0: No infeasibility identified
256:         2: Trivially infeasible
257:     message : str
258:         A string descriptor of the exit status of the optimization.
259: 
260:     References
261:     ----------
262:     .. [2] Andersen, Erling D. "Finding all linearly dependent rows in
263:            large-scale linear programming." Optimization Methods and Software
264:            6.3 (1995): 219-227.
265: 
266:     '''
267: 
268:     tolapiv = 1e-8
269:     tolprimal = 1e-8
270:     status = 0
271:     message = ""
272:     inconsistent = ("There is a linear combination of rows of A_eq that "
273:                     "results in zero, suggesting a redundant constraint. "
274:                     "However the same linear combination of b_eq is "
275:                     "nonzero, suggesting that the constraints conflict "
276:                     "and the problem is infeasible.")
277:     A, rhs, status, message = _remove_zero_rows(A, rhs)
278: 
279:     if status != 0:
280:         return A, rhs, status, message
281: 
282:     m, n = A.shape
283: 
284:     v = list(range(m))      # Artificial column indices.
285:     b = list(v)             # Basis column indices.
286:     # This is better as a list than a set because column order of basis matrix
287:     # needs to be consistent.
288:     k = set(range(m, m+n))  # Structural column indices.
289:     d = []                  # Indices of dependent rows
290: 
291:     A_orig = A
292:     A = scipy.sparse.hstack((scipy.sparse.eye(m), A)).tocsc()
293:     e = np.zeros(m)
294: 
295:     # Implements basic algorithm from [2]
296:     # Uses only one of the suggested improvements (removing zero rows).
297:     # Removing column singletons would be easy, but it is not as important
298:     # because the procedure is performed only on the equality constraint
299:     # matrix from the original problem - not on the canonical form matrix,
300:     # which would have many more column singletons due to slack variables
301:     # from the inequality constraints.
302:     # The thoughts on "crashing" the initial basis sound useful, but the
303:     # description of the procedure seems to assume a lot of familiarity with
304:     # the subject; it is not very explicit. I already went through enough
305:     # trouble getting the basic algorithm working, so I was not interested in
306:     # trying to decipher this, too. (Overall, the paper is fraught with
307:     # mistakes and ambiguities - which is strange, because the rest of
308:     # Andersen's papers are quite good.)
309:     # I tried and tried and tried to improve performance using the
310:     # Bartels-Golub update. It works, but it's only practical if the LU
311:     # factorization can be specialized as described, and that is not possible
312:     # until the Scipy SuperLU interface permits control over column
313:     # permutation - see issue #7700.
314: 
315:     for i in v:
316:         B = A[:, b]
317: 
318:         e[i] = 1
319:         if i > 0:
320:             e[i-1] = 0
321: 
322:         pi = scipy.sparse.linalg.spsolve(B.transpose(), e).reshape(-1, 1)
323: 
324:         js = list(k-set(b))  # not efficient, but this is not the time sink...
325: 
326:         # Due to overhead, it tends to be faster (for problems tested) to
327:         # compute the full matrix-vector product rather than individual
328:         # vector-vector products (with the chance of terminating as soon
329:         # as any are nonzero). For very large matrices, it might be worth
330:         # it to compute, say, 100 or 1000 at a time and stop when a nonzero
331:         # is found.
332:         c = abs(A[:, js].transpose().dot(pi))
333:         if (c > tolapiv).any():  # independent
334:             j = js[np.argmax(c)]  # select very independent column
335:             b[i] = j  # replace artificial column
336:         else:
337:             bibar = pi.T.dot(rhs.reshape(-1, 1))
338:             bnorm = np.linalg.norm(rhs)
339:             if abs(bibar)/(1 + bnorm) > tolprimal:
340:                 status = 2
341:                 message = inconsistent
342:                 return A_orig, rhs, status, message
343:             else:  # dependent
344:                 d.append(i)
345: 
346:     keep = set(range(m))
347:     keep = list(keep - set(d))
348:     return A_orig[keep, :], rhs[keep], status, message
349: 
350: 
351: def _remove_redundancy(A, b):
352:     '''
353:     Eliminates redundant equations from system of equations defined by Ax = b
354:     and identifies infeasibilities.
355: 
356:     Parameters
357:     ----------
358:     A : 2-D array
359:         An array representing the left-hand side of a system of equations
360:     b : 1-D array
361:         An array representing the right-hand side of a system of equations
362: 
363:     Returns
364:     -------
365:     A : 2-D array
366:         An array representing the left-hand side of a system of equations
367:     b : 1-D array
368:         An array representing the right-hand side of a system of equations
369:     status: int
370:         An integer indicating the status of the system
371:         0: No infeasibility identified
372:         2: Trivially infeasible
373:     message : str
374:         A string descriptor of the exit status of the optimization.
375: 
376:     References
377:     ----------
378:     .. [2] Andersen, Erling D. "Finding all linearly dependent rows in
379:            large-scale linear programming." Optimization Methods and Software
380:            6.3 (1995): 219-227.
381: 
382:     '''
383: 
384:     A, b, status, message = _remove_zero_rows(A, b)
385: 
386:     if status != 0:
387:         return A, b, status, message
388: 
389:     U, s, Vh = svd(A)
390:     eps = np.finfo(float).eps
391:     tol = s.max() * max(A.shape) * eps
392: 
393:     m, n = A.shape
394:     s_min = s[-1] if m <= n else 0
395: 
396:     # this algorithm is faster than that of [2] when the nullspace is small
397:     # but it could probably be improvement by randomized algorithms and with
398:     # a sparse implementation.
399:     # it relies on repeated singular value decomposition to find linearly
400:     # dependent rows (as identified by columns of U that correspond with zero
401:     # singular values). Unfortunately, only one row can be removed per
402:     # decomposition (I tried otherwise; doing so can cause problems.)
403:     # It would be nice if we could do truncated SVD like sp.sparse.linalg.svds
404:     # but that function is unreliable at finding singular values near zero.
405:     # Finding max eigenvalue L of A A^T, then largest eigenvalue (and
406:     # associated eigenvector) of -A A^T + L I (I is identity) via power
407:     # iteration would also work in theory, but is only efficient if the
408:     # smallest nonzero eigenvalue of A A^T is close to the largest nonzero
409:     # eigenvalue.
410: 
411:     while abs(s_min) < tol:
412:         v = U[:, -1]  # TODO: return these so user can eliminate from problem?
413:         # rows need to be represented in significant amount
414:         eligibleRows = np.abs(v) > tol * 10e6
415:         if not np.any(eligibleRows) or np.any(np.abs(v.dot(A)) > tol):
416:             status = 4
417:             message = ("Due to numerical issues, redundant equality "
418:                        "constraints could not be removed automatically. "
419:                        "Try providing your constraint matrices as sparse "
420:                        "matrices to activate sparse presolve, try turning "
421:                        "off redundancy removal, or try turning off presolve "
422:                        "altogether.")
423:             break
424:         if np.any(np.abs(v.dot(b)) > tol):
425:             status = 2
426:             message = ("There is a linear combination of rows of A_eq that "
427:                        "results in zero, suggesting a redundant constraint. "
428:                        "However the same linear combination of b_eq is "
429:                        "nonzero, suggesting that the constraints conflict "
430:                        "and the problem is infeasible.")
431:             break
432: 
433:         i_remove = _get_densest(A, eligibleRows)
434:         A = np.delete(A, i_remove, axis=0)
435:         b = np.delete(b, i_remove)
436:         U, s, Vh = svd(A)
437:         m, n = A.shape
438:         s_min = s[-1] if m <= n else 0
439: 
440:     return A, b, status, message
441: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_200133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', '\nRoutines for removing redundant (linearly dependent) equations from linear\nprogramming equality constraints.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import numpy' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_200134 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy')

if (type(import_200134) is not StypyTypeError):

    if (import_200134 != 'pyd_module'):
        __import__(import_200134)
        sys_modules_200135 = sys.modules[import_200134]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', sys_modules_200135.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', import_200134)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.linalg import svd' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_200136 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg')

if (type(import_200136) is not StypyTypeError):

    if (import_200136 != 'pyd_module'):
        __import__(import_200136)
        sys_modules_200137 = sys.modules[import_200136]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg', sys_modules_200137.module_type_store, module_type_store, ['svd'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_200137, sys_modules_200137.module_type_store, module_type_store)
    else:
        from scipy.linalg import svd

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg', None, module_type_store, ['svd'], [svd])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg', import_200136)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import scipy' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_200138 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy')

if (type(import_200138) is not StypyTypeError):

    if (import_200138 != 'pyd_module'):
        __import__(import_200138)
        sys_modules_200139 = sys.modules[import_200138]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy', sys_modules_200139.module_type_store, module_type_store)
    else:
        import scipy

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy', scipy, module_type_store)

else:
    # Assigning a type to the variable 'scipy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy', import_200138)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')


@norecursion
def _row_count(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_row_count'
    module_type_store = module_type_store.open_function_context('_row_count', 13, 0, False)
    
    # Passed parameters checking function
    _row_count.stypy_localization = localization
    _row_count.stypy_type_of_self = None
    _row_count.stypy_type_store = module_type_store
    _row_count.stypy_function_name = '_row_count'
    _row_count.stypy_param_names_list = ['A']
    _row_count.stypy_varargs_param_name = None
    _row_count.stypy_kwargs_param_name = None
    _row_count.stypy_call_defaults = defaults
    _row_count.stypy_call_varargs = varargs
    _row_count.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_row_count', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_row_count', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_row_count(...)' code ##################

    str_200140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, (-1)), 'str', '\n    Counts the number of nonzeros in each row of input array A.\n    Nonzeros are defined as any element with absolute value greater than\n    tol = 1e-13. This value should probably be an input to the function.\n\n    Parameters\n    ----------\n    A : 2-D array\n        An array representing a matrix\n\n    Returns\n    -------\n    rowcount : 1-D array\n        Number of nonzeros in each row of A\n\n    ')
    
    # Assigning a Num to a Name (line 30):
    
    # Assigning a Num to a Name (line 30):
    float_200141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 10), 'float')
    # Assigning a type to the variable 'tol' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'tol', float_200141)
    
    # Call to flatten(...): (line 31)
    # Processing the call keyword arguments (line 31)
    kwargs_200158 = {}
    
    # Call to array(...): (line 31)
    # Processing the call arguments (line 31)
    
    # Call to sum(...): (line 31)
    # Processing the call keyword arguments (line 31)
    int_200151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 44), 'int')
    keyword_200152 = int_200151
    kwargs_200153 = {'axis': keyword_200152}
    
    
    # Call to abs(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'A' (line 31)
    A_200145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 25), 'A', False)
    # Processing the call keyword arguments (line 31)
    kwargs_200146 = {}
    # Getting the type of 'abs' (line 31)
    abs_200144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 21), 'abs', False)
    # Calling abs(args, kwargs) (line 31)
    abs_call_result_200147 = invoke(stypy.reporting.localization.Localization(__file__, 31, 21), abs_200144, *[A_200145], **kwargs_200146)
    
    # Getting the type of 'tol' (line 31)
    tol_200148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 30), 'tol', False)
    # Applying the binary operator '>' (line 31)
    result_gt_200149 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 21), '>', abs_call_result_200147, tol_200148)
    
    # Obtaining the member 'sum' of a type (line 31)
    sum_200150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 21), result_gt_200149, 'sum')
    # Calling sum(args, kwargs) (line 31)
    sum_call_result_200154 = invoke(stypy.reporting.localization.Localization(__file__, 31, 21), sum_200150, *[], **kwargs_200153)
    
    # Processing the call keyword arguments (line 31)
    kwargs_200155 = {}
    # Getting the type of 'np' (line 31)
    np_200142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 31)
    array_200143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 11), np_200142, 'array')
    # Calling array(args, kwargs) (line 31)
    array_call_result_200156 = invoke(stypy.reporting.localization.Localization(__file__, 31, 11), array_200143, *[sum_call_result_200154], **kwargs_200155)
    
    # Obtaining the member 'flatten' of a type (line 31)
    flatten_200157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 11), array_call_result_200156, 'flatten')
    # Calling flatten(args, kwargs) (line 31)
    flatten_call_result_200159 = invoke(stypy.reporting.localization.Localization(__file__, 31, 11), flatten_200157, *[], **kwargs_200158)
    
    # Assigning a type to the variable 'stypy_return_type' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type', flatten_call_result_200159)
    
    # ################# End of '_row_count(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_row_count' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_200160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_200160)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_row_count'
    return stypy_return_type_200160

# Assigning a type to the variable '_row_count' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '_row_count', _row_count)

@norecursion
def _get_densest(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_densest'
    module_type_store = module_type_store.open_function_context('_get_densest', 34, 0, False)
    
    # Passed parameters checking function
    _get_densest.stypy_localization = localization
    _get_densest.stypy_type_of_self = None
    _get_densest.stypy_type_store = module_type_store
    _get_densest.stypy_function_name = '_get_densest'
    _get_densest.stypy_param_names_list = ['A', 'eligibleRows']
    _get_densest.stypy_varargs_param_name = None
    _get_densest.stypy_kwargs_param_name = None
    _get_densest.stypy_call_defaults = defaults
    _get_densest.stypy_call_varargs = varargs
    _get_densest.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_densest', ['A', 'eligibleRows'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_densest', localization, ['A', 'eligibleRows'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_densest(...)' code ##################

    str_200161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, (-1)), 'str', '\n    Returns the index of the densest row of A. Ignores rows that are not\n    eligible for consideration.\n\n    Parameters\n    ----------\n    A : 2-D array\n        An array representing a matrix\n    eligibleRows : 1-D logical array\n        Values indicate whether the corresponding row of A is eligible\n        to be considered\n\n    Returns\n    -------\n    i_densest : int\n        Index of the densest row in A eligible for consideration\n\n    ')
    
    # Assigning a Call to a Name (line 53):
    
    # Assigning a Call to a Name (line 53):
    
    # Call to _row_count(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'A' (line 53)
    A_200163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 27), 'A', False)
    # Processing the call keyword arguments (line 53)
    kwargs_200164 = {}
    # Getting the type of '_row_count' (line 53)
    _row_count_200162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), '_row_count', False)
    # Calling _row_count(args, kwargs) (line 53)
    _row_count_call_result_200165 = invoke(stypy.reporting.localization.Localization(__file__, 53, 16), _row_count_200162, *[A_200163], **kwargs_200164)
    
    # Assigning a type to the variable 'rowCounts' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'rowCounts', _row_count_call_result_200165)
    
    # Call to argmax(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'rowCounts' (line 54)
    rowCounts_200168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 21), 'rowCounts', False)
    # Getting the type of 'eligibleRows' (line 54)
    eligibleRows_200169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 33), 'eligibleRows', False)
    # Applying the binary operator '*' (line 54)
    result_mul_200170 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 21), '*', rowCounts_200168, eligibleRows_200169)
    
    # Processing the call keyword arguments (line 54)
    kwargs_200171 = {}
    # Getting the type of 'np' (line 54)
    np_200166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 11), 'np', False)
    # Obtaining the member 'argmax' of a type (line 54)
    argmax_200167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 11), np_200166, 'argmax')
    # Calling argmax(args, kwargs) (line 54)
    argmax_call_result_200172 = invoke(stypy.reporting.localization.Localization(__file__, 54, 11), argmax_200167, *[result_mul_200170], **kwargs_200171)
    
    # Assigning a type to the variable 'stypy_return_type' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type', argmax_call_result_200172)
    
    # ################# End of '_get_densest(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_densest' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_200173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_200173)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_densest'
    return stypy_return_type_200173

# Assigning a type to the variable '_get_densest' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), '_get_densest', _get_densest)

@norecursion
def _remove_zero_rows(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_remove_zero_rows'
    module_type_store = module_type_store.open_function_context('_remove_zero_rows', 57, 0, False)
    
    # Passed parameters checking function
    _remove_zero_rows.stypy_localization = localization
    _remove_zero_rows.stypy_type_of_self = None
    _remove_zero_rows.stypy_type_store = module_type_store
    _remove_zero_rows.stypy_function_name = '_remove_zero_rows'
    _remove_zero_rows.stypy_param_names_list = ['A', 'b']
    _remove_zero_rows.stypy_varargs_param_name = None
    _remove_zero_rows.stypy_kwargs_param_name = None
    _remove_zero_rows.stypy_call_defaults = defaults
    _remove_zero_rows.stypy_call_varargs = varargs
    _remove_zero_rows.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_remove_zero_rows', ['A', 'b'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_remove_zero_rows', localization, ['A', 'b'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_remove_zero_rows(...)' code ##################

    str_200174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, (-1)), 'str', '\n    Eliminates trivial equations from system of equations defined by Ax = b\n   and identifies trivial infeasibilities\n\n    Parameters\n    ----------\n    A : 2-D array\n        An array representing the left-hand side of a system of equations\n    b : 1-D array\n        An array representing the right-hand side of a system of equations\n\n    Returns\n    -------\n    A : 2-D array\n        An array representing the left-hand side of a system of equations\n    b : 1-D array\n        An array representing the right-hand side of a system of equations\n    status: int\n        An integer indicating the status of the removal operation\n        0: No infeasibility identified\n        2: Trivially infeasible\n    message : str\n        A string descriptor of the exit status of the optimization.\n\n    ')
    
    # Assigning a Num to a Name (line 83):
    
    # Assigning a Num to a Name (line 83):
    int_200175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 13), 'int')
    # Assigning a type to the variable 'status' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'status', int_200175)
    
    # Assigning a Str to a Name (line 84):
    
    # Assigning a Str to a Name (line 84):
    str_200176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 14), 'str', '')
    # Assigning a type to the variable 'message' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'message', str_200176)
    
    # Assigning a Compare to a Name (line 85):
    
    # Assigning a Compare to a Name (line 85):
    
    
    # Call to _row_count(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'A' (line 85)
    A_200178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 24), 'A', False)
    # Processing the call keyword arguments (line 85)
    kwargs_200179 = {}
    # Getting the type of '_row_count' (line 85)
    _row_count_200177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 13), '_row_count', False)
    # Calling _row_count(args, kwargs) (line 85)
    _row_count_call_result_200180 = invoke(stypy.reporting.localization.Localization(__file__, 85, 13), _row_count_200177, *[A_200178], **kwargs_200179)
    
    int_200181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 30), 'int')
    # Applying the binary operator '==' (line 85)
    result_eq_200182 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 13), '==', _row_count_call_result_200180, int_200181)
    
    # Assigning a type to the variable 'i_zero' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'i_zero', result_eq_200182)
    
    # Assigning a Subscript to a Name (line 86):
    
    # Assigning a Subscript to a Name (line 86):
    
    # Obtaining the type of the subscript
    
    # Call to logical_not(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'i_zero' (line 86)
    i_zero_200185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 25), 'i_zero', False)
    # Processing the call keyword arguments (line 86)
    kwargs_200186 = {}
    # Getting the type of 'np' (line 86)
    np_200183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 10), 'np', False)
    # Obtaining the member 'logical_not' of a type (line 86)
    logical_not_200184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 10), np_200183, 'logical_not')
    # Calling logical_not(args, kwargs) (line 86)
    logical_not_call_result_200187 = invoke(stypy.reporting.localization.Localization(__file__, 86, 10), logical_not_200184, *[i_zero_200185], **kwargs_200186)
    
    slice_200188 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 86, 8), None, None, None)
    # Getting the type of 'A' (line 86)
    A_200189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'A')
    # Obtaining the member '__getitem__' of a type (line 86)
    getitem___200190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), A_200189, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 86)
    subscript_call_result_200191 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), getitem___200190, (logical_not_call_result_200187, slice_200188))
    
    # Assigning a type to the variable 'A' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'A', subscript_call_result_200191)
    
    
    
    # Call to allclose(...): (line 87)
    # Processing the call arguments (line 87)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i_zero' (line 87)
    i_zero_200194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 25), 'i_zero', False)
    # Getting the type of 'b' (line 87)
    b_200195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 23), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 87)
    getitem___200196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 23), b_200195, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 87)
    subscript_call_result_200197 = invoke(stypy.reporting.localization.Localization(__file__, 87, 23), getitem___200196, i_zero_200194)
    
    int_200198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 34), 'int')
    # Processing the call keyword arguments (line 87)
    kwargs_200199 = {}
    # Getting the type of 'np' (line 87)
    np_200192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 11), 'np', False)
    # Obtaining the member 'allclose' of a type (line 87)
    allclose_200193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 11), np_200192, 'allclose')
    # Calling allclose(args, kwargs) (line 87)
    allclose_call_result_200200 = invoke(stypy.reporting.localization.Localization(__file__, 87, 11), allclose_200193, *[subscript_call_result_200197, int_200198], **kwargs_200199)
    
    # Applying the 'not' unary operator (line 87)
    result_not__200201 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 7), 'not', allclose_call_result_200200)
    
    # Testing the type of an if condition (line 87)
    if_condition_200202 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 4), result_not__200201)
    # Assigning a type to the variable 'if_condition_200202' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'if_condition_200202', if_condition_200202)
    # SSA begins for if statement (line 87)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 88):
    
    # Assigning a Num to a Name (line 88):
    int_200203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 17), 'int')
    # Assigning a type to the variable 'status' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'status', int_200203)
    
    # Assigning a Str to a Name (line 89):
    
    # Assigning a Str to a Name (line 89):
    str_200204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 18), 'str', 'There is a zero row in A_eq with a nonzero corresponding entry in b_eq. The problem is infeasible.')
    # Assigning a type to the variable 'message' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'message', str_200204)
    # SSA join for if statement (line 87)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 91):
    
    # Assigning a Subscript to a Name (line 91):
    
    # Obtaining the type of the subscript
    
    # Call to logical_not(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'i_zero' (line 91)
    i_zero_200207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 25), 'i_zero', False)
    # Processing the call keyword arguments (line 91)
    kwargs_200208 = {}
    # Getting the type of 'np' (line 91)
    np_200205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 10), 'np', False)
    # Obtaining the member 'logical_not' of a type (line 91)
    logical_not_200206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 10), np_200205, 'logical_not')
    # Calling logical_not(args, kwargs) (line 91)
    logical_not_call_result_200209 = invoke(stypy.reporting.localization.Localization(__file__, 91, 10), logical_not_200206, *[i_zero_200207], **kwargs_200208)
    
    # Getting the type of 'b' (line 91)
    b_200210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'b')
    # Obtaining the member '__getitem__' of a type (line 91)
    getitem___200211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), b_200210, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 91)
    subscript_call_result_200212 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), getitem___200211, logical_not_call_result_200209)
    
    # Assigning a type to the variable 'b' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'b', subscript_call_result_200212)
    
    # Obtaining an instance of the builtin type 'tuple' (line 92)
    tuple_200213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 92)
    # Adding element type (line 92)
    # Getting the type of 'A' (line 92)
    A_200214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'A')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 11), tuple_200213, A_200214)
    # Adding element type (line 92)
    # Getting the type of 'b' (line 92)
    b_200215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 14), 'b')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 11), tuple_200213, b_200215)
    # Adding element type (line 92)
    # Getting the type of 'status' (line 92)
    status_200216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 17), 'status')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 11), tuple_200213, status_200216)
    # Adding element type (line 92)
    # Getting the type of 'message' (line 92)
    message_200217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), 'message')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 11), tuple_200213, message_200217)
    
    # Assigning a type to the variable 'stypy_return_type' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type', tuple_200213)
    
    # ################# End of '_remove_zero_rows(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_remove_zero_rows' in the type store
    # Getting the type of 'stypy_return_type' (line 57)
    stypy_return_type_200218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_200218)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_remove_zero_rows'
    return stypy_return_type_200218

# Assigning a type to the variable '_remove_zero_rows' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), '_remove_zero_rows', _remove_zero_rows)

@norecursion
def bg_update_dense(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'bg_update_dense'
    module_type_store = module_type_store.open_function_context('bg_update_dense', 95, 0, False)
    
    # Passed parameters checking function
    bg_update_dense.stypy_localization = localization
    bg_update_dense.stypy_type_of_self = None
    bg_update_dense.stypy_type_store = module_type_store
    bg_update_dense.stypy_function_name = 'bg_update_dense'
    bg_update_dense.stypy_param_names_list = ['plu', 'perm_r', 'v', 'j']
    bg_update_dense.stypy_varargs_param_name = None
    bg_update_dense.stypy_kwargs_param_name = None
    bg_update_dense.stypy_call_defaults = defaults
    bg_update_dense.stypy_call_varargs = varargs
    bg_update_dense.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'bg_update_dense', ['plu', 'perm_r', 'v', 'j'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'bg_update_dense', localization, ['plu', 'perm_r', 'v', 'j'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'bg_update_dense(...)' code ##################

    
    # Assigning a Name to a Tuple (line 96):
    
    # Assigning a Subscript to a Name (line 96):
    
    # Obtaining the type of the subscript
    int_200219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 4), 'int')
    # Getting the type of 'plu' (line 96)
    plu_200220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'plu')
    # Obtaining the member '__getitem__' of a type (line 96)
    getitem___200221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 4), plu_200220, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 96)
    subscript_call_result_200222 = invoke(stypy.reporting.localization.Localization(__file__, 96, 4), getitem___200221, int_200219)
    
    # Assigning a type to the variable 'tuple_var_assignment_200101' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'tuple_var_assignment_200101', subscript_call_result_200222)
    
    # Assigning a Subscript to a Name (line 96):
    
    # Obtaining the type of the subscript
    int_200223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 4), 'int')
    # Getting the type of 'plu' (line 96)
    plu_200224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'plu')
    # Obtaining the member '__getitem__' of a type (line 96)
    getitem___200225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 4), plu_200224, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 96)
    subscript_call_result_200226 = invoke(stypy.reporting.localization.Localization(__file__, 96, 4), getitem___200225, int_200223)
    
    # Assigning a type to the variable 'tuple_var_assignment_200102' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'tuple_var_assignment_200102', subscript_call_result_200226)
    
    # Assigning a Name to a Name (line 96):
    # Getting the type of 'tuple_var_assignment_200101' (line 96)
    tuple_var_assignment_200101_200227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'tuple_var_assignment_200101')
    # Assigning a type to the variable 'LU' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'LU', tuple_var_assignment_200101_200227)
    
    # Assigning a Name to a Name (line 96):
    # Getting the type of 'tuple_var_assignment_200102' (line 96)
    tuple_var_assignment_200102_200228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'tuple_var_assignment_200102')
    # Assigning a type to the variable 'p' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'p', tuple_var_assignment_200102_200228)
    
    # Assigning a Call to a Name (line 98):
    
    # Assigning a Call to a Name (line 98):
    
    # Call to solve_triangular(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'LU' (line 98)
    LU_200232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 38), 'LU', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'perm_r' (line 98)
    perm_r_200233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 44), 'perm_r', False)
    # Getting the type of 'v' (line 98)
    v_200234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 42), 'v', False)
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___200235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 42), v_200234, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_200236 = invoke(stypy.reporting.localization.Localization(__file__, 98, 42), getitem___200235, perm_r_200233)
    
    # Processing the call keyword arguments (line 98)
    # Getting the type of 'True' (line 98)
    True_200237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 59), 'True', False)
    keyword_200238 = True_200237
    # Getting the type of 'True' (line 99)
    True_200239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 52), 'True', False)
    keyword_200240 = True_200239
    kwargs_200241 = {'unit_diagonal': keyword_200240, 'lower': keyword_200238}
    # Getting the type of 'scipy' (line 98)
    scipy_200229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'scipy', False)
    # Obtaining the member 'linalg' of a type (line 98)
    linalg_200230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), scipy_200229, 'linalg')
    # Obtaining the member 'solve_triangular' of a type (line 98)
    solve_triangular_200231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), linalg_200230, 'solve_triangular')
    # Calling solve_triangular(args, kwargs) (line 98)
    solve_triangular_call_result_200242 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), solve_triangular_200231, *[LU_200232, subscript_call_result_200236], **kwargs_200241)
    
    # Assigning a type to the variable 'u' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'u', solve_triangular_call_result_200242)
    
    # Assigning a Subscript to a Subscript (line 100):
    
    # Assigning a Subscript to a Subscript (line 100):
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 100)
    j_200243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 21), 'j')
    int_200244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 23), 'int')
    # Applying the binary operator '+' (line 100)
    result_add_200245 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 21), '+', j_200243, int_200244)
    
    slice_200246 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 100, 18), None, result_add_200245, None)
    # Getting the type of 'u' (line 100)
    u_200247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 18), 'u')
    # Obtaining the member '__getitem__' of a type (line 100)
    getitem___200248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 18), u_200247, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 100)
    subscript_call_result_200249 = invoke(stypy.reporting.localization.Localization(__file__, 100, 18), getitem___200248, slice_200246)
    
    # Getting the type of 'LU' (line 100)
    LU_200250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'LU')
    # Getting the type of 'j' (line 100)
    j_200251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'j')
    int_200252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 10), 'int')
    # Applying the binary operator '+' (line 100)
    result_add_200253 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 8), '+', j_200251, int_200252)
    
    slice_200254 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 100, 4), None, result_add_200253, None)
    # Getting the type of 'j' (line 100)
    j_200255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 13), 'j')
    # Storing an element on a container (line 100)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 4), LU_200250, ((slice_200254, j_200255), subscript_call_result_200249))
    
    # Assigning a Subscript to a Name (line 101):
    
    # Assigning a Subscript to a Name (line 101):
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 101)
    j_200256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 10), 'j')
    int_200257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 12), 'int')
    # Applying the binary operator '+' (line 101)
    result_add_200258 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 10), '+', j_200256, int_200257)
    
    slice_200259 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 101, 8), result_add_200258, None, None)
    # Getting the type of 'u' (line 101)
    u_200260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'u')
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___200261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), u_200260, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_200262 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), getitem___200261, slice_200259)
    
    # Assigning a type to the variable 'l' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'l', subscript_call_result_200262)
    
    # Assigning a Subscript to a Name (line 102):
    
    # Assigning a Subscript to a Name (line 102):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 102)
    tuple_200263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 102)
    # Adding element type (line 102)
    # Getting the type of 'j' (line 102)
    j_200264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 13), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 13), tuple_200263, j_200264)
    # Adding element type (line 102)
    # Getting the type of 'j' (line 102)
    j_200265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 13), tuple_200263, j_200265)
    
    # Getting the type of 'LU' (line 102)
    LU_200266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 10), 'LU')
    # Obtaining the member '__getitem__' of a type (line 102)
    getitem___200267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 10), LU_200266, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 102)
    subscript_call_result_200268 = invoke(stypy.reporting.localization.Localization(__file__, 102, 10), getitem___200267, tuple_200263)
    
    # Assigning a type to the variable 'piv' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'piv', subscript_call_result_200268)
    
    # Getting the type of 'LU' (line 103)
    LU_200269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'LU')
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 103)
    j_200270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 7), 'j')
    int_200271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 9), 'int')
    # Applying the binary operator '+' (line 103)
    result_add_200272 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 7), '+', j_200270, int_200271)
    
    slice_200273 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 103, 4), result_add_200272, None, None)
    # Getting the type of 'j' (line 103)
    j_200274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 13), 'j')
    # Getting the type of 'LU' (line 103)
    LU_200275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'LU')
    # Obtaining the member '__getitem__' of a type (line 103)
    getitem___200276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 4), LU_200275, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 103)
    subscript_call_result_200277 = invoke(stypy.reporting.localization.Localization(__file__, 103, 4), getitem___200276, (slice_200273, j_200274))
    
    # Getting the type of 'l' (line 103)
    l_200278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'l')
    # Getting the type of 'piv' (line 103)
    piv_200279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 22), 'piv')
    # Applying the binary operator 'div' (line 103)
    result_div_200280 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 20), 'div', l_200278, piv_200279)
    
    # Applying the binary operator '+=' (line 103)
    result_iadd_200281 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 4), '+=', subscript_call_result_200277, result_div_200280)
    # Getting the type of 'LU' (line 103)
    LU_200282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'LU')
    # Getting the type of 'j' (line 103)
    j_200283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 7), 'j')
    int_200284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 9), 'int')
    # Applying the binary operator '+' (line 103)
    result_add_200285 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 7), '+', j_200283, int_200284)
    
    slice_200286 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 103, 4), result_add_200285, None, None)
    # Getting the type of 'j' (line 103)
    j_200287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 13), 'j')
    # Storing an element on a container (line 103)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 4), LU_200282, ((slice_200286, j_200287), result_iadd_200281))
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 104)
    tuple_200288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 104)
    # Adding element type (line 104)
    # Getting the type of 'LU' (line 104)
    LU_200289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 11), 'LU')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 11), tuple_200288, LU_200289)
    # Adding element type (line 104)
    # Getting the type of 'p' (line 104)
    p_200290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'p')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 11), tuple_200288, p_200290)
    
    # Assigning a type to the variable 'stypy_return_type' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'stypy_return_type', tuple_200288)
    
    # ################# End of 'bg_update_dense(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'bg_update_dense' in the type store
    # Getting the type of 'stypy_return_type' (line 95)
    stypy_return_type_200291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_200291)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'bg_update_dense'
    return stypy_return_type_200291

# Assigning a type to the variable 'bg_update_dense' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'bg_update_dense', bg_update_dense)

@norecursion
def _remove_redundancy_dense(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_remove_redundancy_dense'
    module_type_store = module_type_store.open_function_context('_remove_redundancy_dense', 107, 0, False)
    
    # Passed parameters checking function
    _remove_redundancy_dense.stypy_localization = localization
    _remove_redundancy_dense.stypy_type_of_self = None
    _remove_redundancy_dense.stypy_type_store = module_type_store
    _remove_redundancy_dense.stypy_function_name = '_remove_redundancy_dense'
    _remove_redundancy_dense.stypy_param_names_list = ['A', 'rhs']
    _remove_redundancy_dense.stypy_varargs_param_name = None
    _remove_redundancy_dense.stypy_kwargs_param_name = None
    _remove_redundancy_dense.stypy_call_defaults = defaults
    _remove_redundancy_dense.stypy_call_varargs = varargs
    _remove_redundancy_dense.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_remove_redundancy_dense', ['A', 'rhs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_remove_redundancy_dense', localization, ['A', 'rhs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_remove_redundancy_dense(...)' code ##################

    str_200292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, (-1)), 'str', '\n    Eliminates redundant equations from system of equations defined by Ax = b\n    and identifies infeasibilities.\n\n    Parameters\n    ----------\n    A : 2-D sparse matrix\n        An matrix representing the left-hand side of a system of equations\n    rhs : 1-D array\n        An array representing the right-hand side of a system of equations\n\n    Returns\n    ----------\n    A : 2-D sparse matrix\n        A matrix representing the left-hand side of a system of equations\n    rhs : 1-D array\n        An array representing the right-hand side of a system of equations\n    status: int\n        An integer indicating the status of the system\n        0: No infeasibility identified\n        2: Trivially infeasible\n    message : str\n        A string descriptor of the exit status of the optimization.\n\n    References\n    ----------\n    .. [2] Andersen, Erling D. "Finding all linearly dependent rows in\n           large-scale linear programming." Optimization Methods and Software\n           6.3 (1995): 219-227.\n\n    ')
    
    # Assigning a Num to a Name (line 139):
    
    # Assigning a Num to a Name (line 139):
    float_200293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 14), 'float')
    # Assigning a type to the variable 'tolapiv' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'tolapiv', float_200293)
    
    # Assigning a Num to a Name (line 140):
    
    # Assigning a Num to a Name (line 140):
    float_200294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 16), 'float')
    # Assigning a type to the variable 'tolprimal' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'tolprimal', float_200294)
    
    # Assigning a Num to a Name (line 141):
    
    # Assigning a Num to a Name (line 141):
    int_200295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 13), 'int')
    # Assigning a type to the variable 'status' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'status', int_200295)
    
    # Assigning a Str to a Name (line 142):
    
    # Assigning a Str to a Name (line 142):
    str_200296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 14), 'str', '')
    # Assigning a type to the variable 'message' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'message', str_200296)
    
    # Assigning a Str to a Name (line 143):
    
    # Assigning a Str to a Name (line 143):
    str_200297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 20), 'str', 'There is a linear combination of rows of A_eq that results in zero, suggesting a redundant constraint. However the same linear combination of b_eq is nonzero, suggesting that the constraints conflict and the problem is infeasible.')
    # Assigning a type to the variable 'inconsistent' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'inconsistent', str_200297)
    
    # Assigning a Call to a Tuple (line 148):
    
    # Assigning a Subscript to a Name (line 148):
    
    # Obtaining the type of the subscript
    int_200298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 4), 'int')
    
    # Call to _remove_zero_rows(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'A' (line 148)
    A_200300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 48), 'A', False)
    # Getting the type of 'rhs' (line 148)
    rhs_200301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 51), 'rhs', False)
    # Processing the call keyword arguments (line 148)
    kwargs_200302 = {}
    # Getting the type of '_remove_zero_rows' (line 148)
    _remove_zero_rows_200299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 30), '_remove_zero_rows', False)
    # Calling _remove_zero_rows(args, kwargs) (line 148)
    _remove_zero_rows_call_result_200303 = invoke(stypy.reporting.localization.Localization(__file__, 148, 30), _remove_zero_rows_200299, *[A_200300, rhs_200301], **kwargs_200302)
    
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___200304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 4), _remove_zero_rows_call_result_200303, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_200305 = invoke(stypy.reporting.localization.Localization(__file__, 148, 4), getitem___200304, int_200298)
    
    # Assigning a type to the variable 'tuple_var_assignment_200103' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'tuple_var_assignment_200103', subscript_call_result_200305)
    
    # Assigning a Subscript to a Name (line 148):
    
    # Obtaining the type of the subscript
    int_200306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 4), 'int')
    
    # Call to _remove_zero_rows(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'A' (line 148)
    A_200308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 48), 'A', False)
    # Getting the type of 'rhs' (line 148)
    rhs_200309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 51), 'rhs', False)
    # Processing the call keyword arguments (line 148)
    kwargs_200310 = {}
    # Getting the type of '_remove_zero_rows' (line 148)
    _remove_zero_rows_200307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 30), '_remove_zero_rows', False)
    # Calling _remove_zero_rows(args, kwargs) (line 148)
    _remove_zero_rows_call_result_200311 = invoke(stypy.reporting.localization.Localization(__file__, 148, 30), _remove_zero_rows_200307, *[A_200308, rhs_200309], **kwargs_200310)
    
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___200312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 4), _remove_zero_rows_call_result_200311, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_200313 = invoke(stypy.reporting.localization.Localization(__file__, 148, 4), getitem___200312, int_200306)
    
    # Assigning a type to the variable 'tuple_var_assignment_200104' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'tuple_var_assignment_200104', subscript_call_result_200313)
    
    # Assigning a Subscript to a Name (line 148):
    
    # Obtaining the type of the subscript
    int_200314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 4), 'int')
    
    # Call to _remove_zero_rows(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'A' (line 148)
    A_200316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 48), 'A', False)
    # Getting the type of 'rhs' (line 148)
    rhs_200317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 51), 'rhs', False)
    # Processing the call keyword arguments (line 148)
    kwargs_200318 = {}
    # Getting the type of '_remove_zero_rows' (line 148)
    _remove_zero_rows_200315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 30), '_remove_zero_rows', False)
    # Calling _remove_zero_rows(args, kwargs) (line 148)
    _remove_zero_rows_call_result_200319 = invoke(stypy.reporting.localization.Localization(__file__, 148, 30), _remove_zero_rows_200315, *[A_200316, rhs_200317], **kwargs_200318)
    
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___200320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 4), _remove_zero_rows_call_result_200319, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_200321 = invoke(stypy.reporting.localization.Localization(__file__, 148, 4), getitem___200320, int_200314)
    
    # Assigning a type to the variable 'tuple_var_assignment_200105' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'tuple_var_assignment_200105', subscript_call_result_200321)
    
    # Assigning a Subscript to a Name (line 148):
    
    # Obtaining the type of the subscript
    int_200322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 4), 'int')
    
    # Call to _remove_zero_rows(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'A' (line 148)
    A_200324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 48), 'A', False)
    # Getting the type of 'rhs' (line 148)
    rhs_200325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 51), 'rhs', False)
    # Processing the call keyword arguments (line 148)
    kwargs_200326 = {}
    # Getting the type of '_remove_zero_rows' (line 148)
    _remove_zero_rows_200323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 30), '_remove_zero_rows', False)
    # Calling _remove_zero_rows(args, kwargs) (line 148)
    _remove_zero_rows_call_result_200327 = invoke(stypy.reporting.localization.Localization(__file__, 148, 30), _remove_zero_rows_200323, *[A_200324, rhs_200325], **kwargs_200326)
    
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___200328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 4), _remove_zero_rows_call_result_200327, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_200329 = invoke(stypy.reporting.localization.Localization(__file__, 148, 4), getitem___200328, int_200322)
    
    # Assigning a type to the variable 'tuple_var_assignment_200106' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'tuple_var_assignment_200106', subscript_call_result_200329)
    
    # Assigning a Name to a Name (line 148):
    # Getting the type of 'tuple_var_assignment_200103' (line 148)
    tuple_var_assignment_200103_200330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'tuple_var_assignment_200103')
    # Assigning a type to the variable 'A' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'A', tuple_var_assignment_200103_200330)
    
    # Assigning a Name to a Name (line 148):
    # Getting the type of 'tuple_var_assignment_200104' (line 148)
    tuple_var_assignment_200104_200331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'tuple_var_assignment_200104')
    # Assigning a type to the variable 'rhs' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 7), 'rhs', tuple_var_assignment_200104_200331)
    
    # Assigning a Name to a Name (line 148):
    # Getting the type of 'tuple_var_assignment_200105' (line 148)
    tuple_var_assignment_200105_200332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'tuple_var_assignment_200105')
    # Assigning a type to the variable 'status' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'status', tuple_var_assignment_200105_200332)
    
    # Assigning a Name to a Name (line 148):
    # Getting the type of 'tuple_var_assignment_200106' (line 148)
    tuple_var_assignment_200106_200333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'tuple_var_assignment_200106')
    # Assigning a type to the variable 'message' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'message', tuple_var_assignment_200106_200333)
    
    
    # Getting the type of 'status' (line 150)
    status_200334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 7), 'status')
    int_200335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 17), 'int')
    # Applying the binary operator '!=' (line 150)
    result_ne_200336 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 7), '!=', status_200334, int_200335)
    
    # Testing the type of an if condition (line 150)
    if_condition_200337 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 4), result_ne_200336)
    # Assigning a type to the variable 'if_condition_200337' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'if_condition_200337', if_condition_200337)
    # SSA begins for if statement (line 150)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 151)
    tuple_200338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 151)
    # Adding element type (line 151)
    # Getting the type of 'A' (line 151)
    A_200339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 15), 'A')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 15), tuple_200338, A_200339)
    # Adding element type (line 151)
    # Getting the type of 'rhs' (line 151)
    rhs_200340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 18), 'rhs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 15), tuple_200338, rhs_200340)
    # Adding element type (line 151)
    # Getting the type of 'status' (line 151)
    status_200341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 23), 'status')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 15), tuple_200338, status_200341)
    # Adding element type (line 151)
    # Getting the type of 'message' (line 151)
    message_200342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 31), 'message')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 15), tuple_200338, message_200342)
    
    # Assigning a type to the variable 'stypy_return_type' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'stypy_return_type', tuple_200338)
    # SSA join for if statement (line 150)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Tuple (line 153):
    
    # Assigning a Subscript to a Name (line 153):
    
    # Obtaining the type of the subscript
    int_200343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 4), 'int')
    # Getting the type of 'A' (line 153)
    A_200344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 11), 'A')
    # Obtaining the member 'shape' of a type (line 153)
    shape_200345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 11), A_200344, 'shape')
    # Obtaining the member '__getitem__' of a type (line 153)
    getitem___200346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 4), shape_200345, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 153)
    subscript_call_result_200347 = invoke(stypy.reporting.localization.Localization(__file__, 153, 4), getitem___200346, int_200343)
    
    # Assigning a type to the variable 'tuple_var_assignment_200107' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'tuple_var_assignment_200107', subscript_call_result_200347)
    
    # Assigning a Subscript to a Name (line 153):
    
    # Obtaining the type of the subscript
    int_200348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 4), 'int')
    # Getting the type of 'A' (line 153)
    A_200349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 11), 'A')
    # Obtaining the member 'shape' of a type (line 153)
    shape_200350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 11), A_200349, 'shape')
    # Obtaining the member '__getitem__' of a type (line 153)
    getitem___200351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 4), shape_200350, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 153)
    subscript_call_result_200352 = invoke(stypy.reporting.localization.Localization(__file__, 153, 4), getitem___200351, int_200348)
    
    # Assigning a type to the variable 'tuple_var_assignment_200108' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'tuple_var_assignment_200108', subscript_call_result_200352)
    
    # Assigning a Name to a Name (line 153):
    # Getting the type of 'tuple_var_assignment_200107' (line 153)
    tuple_var_assignment_200107_200353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'tuple_var_assignment_200107')
    # Assigning a type to the variable 'm' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'm', tuple_var_assignment_200107_200353)
    
    # Assigning a Name to a Name (line 153):
    # Getting the type of 'tuple_var_assignment_200108' (line 153)
    tuple_var_assignment_200108_200354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'tuple_var_assignment_200108')
    # Assigning a type to the variable 'n' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 7), 'n', tuple_var_assignment_200108_200354)
    
    # Assigning a Call to a Name (line 155):
    
    # Assigning a Call to a Name (line 155):
    
    # Call to list(...): (line 155)
    # Processing the call arguments (line 155)
    
    # Call to range(...): (line 155)
    # Processing the call arguments (line 155)
    # Getting the type of 'm' (line 155)
    m_200357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 19), 'm', False)
    # Processing the call keyword arguments (line 155)
    kwargs_200358 = {}
    # Getting the type of 'range' (line 155)
    range_200356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 13), 'range', False)
    # Calling range(args, kwargs) (line 155)
    range_call_result_200359 = invoke(stypy.reporting.localization.Localization(__file__, 155, 13), range_200356, *[m_200357], **kwargs_200358)
    
    # Processing the call keyword arguments (line 155)
    kwargs_200360 = {}
    # Getting the type of 'list' (line 155)
    list_200355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'list', False)
    # Calling list(args, kwargs) (line 155)
    list_call_result_200361 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), list_200355, *[range_call_result_200359], **kwargs_200360)
    
    # Assigning a type to the variable 'v' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'v', list_call_result_200361)
    
    # Assigning a Call to a Name (line 156):
    
    # Assigning a Call to a Name (line 156):
    
    # Call to list(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'v' (line 156)
    v_200363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 13), 'v', False)
    # Processing the call keyword arguments (line 156)
    kwargs_200364 = {}
    # Getting the type of 'list' (line 156)
    list_200362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'list', False)
    # Calling list(args, kwargs) (line 156)
    list_call_result_200365 = invoke(stypy.reporting.localization.Localization(__file__, 156, 8), list_200362, *[v_200363], **kwargs_200364)
    
    # Assigning a type to the variable 'b' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'b', list_call_result_200365)
    
    # Assigning a Call to a Name (line 159):
    
    # Assigning a Call to a Name (line 159):
    
    # Call to set(...): (line 159)
    # Processing the call arguments (line 159)
    
    # Call to range(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'm' (line 159)
    m_200368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 18), 'm', False)
    # Getting the type of 'm' (line 159)
    m_200369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 21), 'm', False)
    # Getting the type of 'n' (line 159)
    n_200370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), 'n', False)
    # Applying the binary operator '+' (line 159)
    result_add_200371 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 21), '+', m_200369, n_200370)
    
    # Processing the call keyword arguments (line 159)
    kwargs_200372 = {}
    # Getting the type of 'range' (line 159)
    range_200367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'range', False)
    # Calling range(args, kwargs) (line 159)
    range_call_result_200373 = invoke(stypy.reporting.localization.Localization(__file__, 159, 12), range_200367, *[m_200368, result_add_200371], **kwargs_200372)
    
    # Processing the call keyword arguments (line 159)
    kwargs_200374 = {}
    # Getting the type of 'set' (line 159)
    set_200366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'set', False)
    # Calling set(args, kwargs) (line 159)
    set_call_result_200375 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), set_200366, *[range_call_result_200373], **kwargs_200374)
    
    # Assigning a type to the variable 'k' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'k', set_call_result_200375)
    
    # Assigning a List to a Name (line 160):
    
    # Assigning a List to a Name (line 160):
    
    # Obtaining an instance of the builtin type 'list' (line 160)
    list_200376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 160)
    
    # Assigning a type to the variable 'd' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'd', list_200376)
    
    # Assigning a Name to a Name (line 161):
    
    # Assigning a Name to a Name (line 161):
    # Getting the type of 'None' (line 161)
    None_200377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 9), 'None')
    # Assigning a type to the variable 'lu' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'lu', None_200377)
    
    # Assigning a Name to a Name (line 162):
    
    # Assigning a Name to a Name (line 162):
    # Getting the type of 'None' (line 162)
    None_200378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 13), 'None')
    # Assigning a type to the variable 'perm_r' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'perm_r', None_200378)
    
    # Assigning a Name to a Name (line 164):
    
    # Assigning a Name to a Name (line 164):
    # Getting the type of 'A' (line 164)
    A_200379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 13), 'A')
    # Assigning a type to the variable 'A_orig' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'A_orig', A_200379)
    
    # Assigning a Call to a Name (line 165):
    
    # Assigning a Call to a Name (line 165):
    
    # Call to hstack(...): (line 165)
    # Processing the call arguments (line 165)
    
    # Obtaining an instance of the builtin type 'tuple' (line 165)
    tuple_200382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 165)
    # Adding element type (line 165)
    
    # Call to eye(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'm' (line 165)
    m_200385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 26), 'm', False)
    # Processing the call keyword arguments (line 165)
    kwargs_200386 = {}
    # Getting the type of 'np' (line 165)
    np_200383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 19), 'np', False)
    # Obtaining the member 'eye' of a type (line 165)
    eye_200384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 19), np_200383, 'eye')
    # Calling eye(args, kwargs) (line 165)
    eye_call_result_200387 = invoke(stypy.reporting.localization.Localization(__file__, 165, 19), eye_200384, *[m_200385], **kwargs_200386)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 19), tuple_200382, eye_call_result_200387)
    # Adding element type (line 165)
    # Getting the type of 'A' (line 165)
    A_200388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 30), 'A', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 19), tuple_200382, A_200388)
    
    # Processing the call keyword arguments (line 165)
    kwargs_200389 = {}
    # Getting the type of 'np' (line 165)
    np_200380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'np', False)
    # Obtaining the member 'hstack' of a type (line 165)
    hstack_200381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), np_200380, 'hstack')
    # Calling hstack(args, kwargs) (line 165)
    hstack_call_result_200390 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), hstack_200381, *[tuple_200382], **kwargs_200389)
    
    # Assigning a type to the variable 'A' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'A', hstack_call_result_200390)
    
    # Assigning a Call to a Name (line 166):
    
    # Assigning a Call to a Name (line 166):
    
    # Call to zeros(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'm' (line 166)
    m_200393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 17), 'm', False)
    # Processing the call keyword arguments (line 166)
    kwargs_200394 = {}
    # Getting the type of 'np' (line 166)
    np_200391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 166)
    zeros_200392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), np_200391, 'zeros')
    # Calling zeros(args, kwargs) (line 166)
    zeros_call_result_200395 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), zeros_200392, *[m_200393], **kwargs_200394)
    
    # Assigning a type to the variable 'e' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'e', zeros_call_result_200395)
    
    # Assigning a Subscript to a Name (line 184):
    
    # Assigning a Subscript to a Name (line 184):
    
    # Obtaining the type of the subscript
    slice_200396 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 184, 8), None, None, None)
    # Getting the type of 'b' (line 184)
    b_200397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 13), 'b')
    # Getting the type of 'A' (line 184)
    A_200398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'A')
    # Obtaining the member '__getitem__' of a type (line 184)
    getitem___200399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), A_200398, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 184)
    subscript_call_result_200400 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), getitem___200399, (slice_200396, b_200397))
    
    # Assigning a type to the variable 'B' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'B', subscript_call_result_200400)
    
    # Getting the type of 'v' (line 185)
    v_200401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 13), 'v')
    # Testing the type of a for loop iterable (line 185)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 185, 4), v_200401)
    # Getting the type of the for loop variable (line 185)
    for_loop_var_200402 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 185, 4), v_200401)
    # Assigning a type to the variable 'i' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'i', for_loop_var_200402)
    # SSA begins for a for statement (line 185)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Num to a Subscript (line 187):
    
    # Assigning a Num to a Subscript (line 187):
    int_200403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 15), 'int')
    # Getting the type of 'e' (line 187)
    e_200404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'e')
    # Getting the type of 'i' (line 187)
    i_200405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 10), 'i')
    # Storing an element on a container (line 187)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 8), e_200404, (i_200405, int_200403))
    
    
    # Getting the type of 'i' (line 188)
    i_200406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 11), 'i')
    int_200407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 15), 'int')
    # Applying the binary operator '>' (line 188)
    result_gt_200408 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 11), '>', i_200406, int_200407)
    
    # Testing the type of an if condition (line 188)
    if_condition_200409 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 8), result_gt_200408)
    # Assigning a type to the variable 'if_condition_200409' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'if_condition_200409', if_condition_200409)
    # SSA begins for if statement (line 188)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 189):
    
    # Assigning a Num to a Subscript (line 189):
    int_200410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 21), 'int')
    # Getting the type of 'e' (line 189)
    e_200411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'e')
    # Getting the type of 'i' (line 189)
    i_200412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 14), 'i')
    int_200413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 16), 'int')
    # Applying the binary operator '-' (line 189)
    result_sub_200414 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 14), '-', i_200412, int_200413)
    
    # Storing an element on a container (line 189)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 12), e_200411, (result_sub_200414, int_200410))
    # SSA join for if statement (line 188)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 191)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 192):
    
    # Assigning a Subscript to a Name (line 192):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 192)
    i_200415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 18), 'i')
    int_200416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 20), 'int')
    # Applying the binary operator '-' (line 192)
    result_sub_200417 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 18), '-', i_200415, int_200416)
    
    # Getting the type of 'b' (line 192)
    b_200418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 16), 'b')
    # Obtaining the member '__getitem__' of a type (line 192)
    getitem___200419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 16), b_200418, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 192)
    subscript_call_result_200420 = invoke(stypy.reporting.localization.Localization(__file__, 192, 16), getitem___200419, result_sub_200417)
    
    # Assigning a type to the variable 'j' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'j', subscript_call_result_200420)
    
    # Assigning a Call to a Name (line 193):
    
    # Assigning a Call to a Name (line 193):
    
    # Call to bg_update_dense(...): (line 193)
    # Processing the call arguments (line 193)
    # Getting the type of 'lu' (line 193)
    lu_200422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 33), 'lu', False)
    # Getting the type of 'perm_r' (line 193)
    perm_r_200423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 37), 'perm_r', False)
    
    # Obtaining the type of the subscript
    slice_200424 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 193, 45), None, None, None)
    # Getting the type of 'j' (line 193)
    j_200425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 50), 'j', False)
    # Getting the type of 'A' (line 193)
    A_200426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 45), 'A', False)
    # Obtaining the member '__getitem__' of a type (line 193)
    getitem___200427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 45), A_200426, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 193)
    subscript_call_result_200428 = invoke(stypy.reporting.localization.Localization(__file__, 193, 45), getitem___200427, (slice_200424, j_200425))
    
    # Getting the type of 'i' (line 193)
    i_200429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 54), 'i', False)
    int_200430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 56), 'int')
    # Applying the binary operator '-' (line 193)
    result_sub_200431 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 54), '-', i_200429, int_200430)
    
    # Processing the call keyword arguments (line 193)
    kwargs_200432 = {}
    # Getting the type of 'bg_update_dense' (line 193)
    bg_update_dense_200421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 17), 'bg_update_dense', False)
    # Calling bg_update_dense(args, kwargs) (line 193)
    bg_update_dense_call_result_200433 = invoke(stypy.reporting.localization.Localization(__file__, 193, 17), bg_update_dense_200421, *[lu_200422, perm_r_200423, subscript_call_result_200428, result_sub_200431], **kwargs_200432)
    
    # Assigning a type to the variable 'lu' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'lu', bg_update_dense_call_result_200433)
    # SSA branch for the except part of a try statement (line 191)
    # SSA branch for the except '<any exception>' branch of a try statement (line 191)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Name (line 195):
    
    # Assigning a Call to a Name (line 195):
    
    # Call to lu_factor(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of 'B' (line 195)
    B_200437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 40), 'B', False)
    # Processing the call keyword arguments (line 195)
    kwargs_200438 = {}
    # Getting the type of 'scipy' (line 195)
    scipy_200434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 17), 'scipy', False)
    # Obtaining the member 'linalg' of a type (line 195)
    linalg_200435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 17), scipy_200434, 'linalg')
    # Obtaining the member 'lu_factor' of a type (line 195)
    lu_factor_200436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 17), linalg_200435, 'lu_factor')
    # Calling lu_factor(args, kwargs) (line 195)
    lu_factor_call_result_200439 = invoke(stypy.reporting.localization.Localization(__file__, 195, 17), lu_factor_200436, *[B_200437], **kwargs_200438)
    
    # Assigning a type to the variable 'lu' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'lu', lu_factor_call_result_200439)
    
    # Assigning a Name to a Tuple (line 196):
    
    # Assigning a Subscript to a Name (line 196):
    
    # Obtaining the type of the subscript
    int_200440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 12), 'int')
    # Getting the type of 'lu' (line 196)
    lu_200441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 20), 'lu')
    # Obtaining the member '__getitem__' of a type (line 196)
    getitem___200442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 12), lu_200441, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 196)
    subscript_call_result_200443 = invoke(stypy.reporting.localization.Localization(__file__, 196, 12), getitem___200442, int_200440)
    
    # Assigning a type to the variable 'tuple_var_assignment_200109' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'tuple_var_assignment_200109', subscript_call_result_200443)
    
    # Assigning a Subscript to a Name (line 196):
    
    # Obtaining the type of the subscript
    int_200444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 12), 'int')
    # Getting the type of 'lu' (line 196)
    lu_200445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 20), 'lu')
    # Obtaining the member '__getitem__' of a type (line 196)
    getitem___200446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 12), lu_200445, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 196)
    subscript_call_result_200447 = invoke(stypy.reporting.localization.Localization(__file__, 196, 12), getitem___200446, int_200444)
    
    # Assigning a type to the variable 'tuple_var_assignment_200110' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'tuple_var_assignment_200110', subscript_call_result_200447)
    
    # Assigning a Name to a Name (line 196):
    # Getting the type of 'tuple_var_assignment_200109' (line 196)
    tuple_var_assignment_200109_200448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'tuple_var_assignment_200109')
    # Assigning a type to the variable 'LU' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'LU', tuple_var_assignment_200109_200448)
    
    # Assigning a Name to a Name (line 196):
    # Getting the type of 'tuple_var_assignment_200110' (line 196)
    tuple_var_assignment_200110_200449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'tuple_var_assignment_200110')
    # Assigning a type to the variable 'p' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), 'p', tuple_var_assignment_200110_200449)
    
    # Assigning a Call to a Name (line 197):
    
    # Assigning a Call to a Name (line 197):
    
    # Call to list(...): (line 197)
    # Processing the call arguments (line 197)
    
    # Call to range(...): (line 197)
    # Processing the call arguments (line 197)
    # Getting the type of 'm' (line 197)
    m_200452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 32), 'm', False)
    # Processing the call keyword arguments (line 197)
    kwargs_200453 = {}
    # Getting the type of 'range' (line 197)
    range_200451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 26), 'range', False)
    # Calling range(args, kwargs) (line 197)
    range_call_result_200454 = invoke(stypy.reporting.localization.Localization(__file__, 197, 26), range_200451, *[m_200452], **kwargs_200453)
    
    # Processing the call keyword arguments (line 197)
    kwargs_200455 = {}
    # Getting the type of 'list' (line 197)
    list_200450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 21), 'list', False)
    # Calling list(args, kwargs) (line 197)
    list_call_result_200456 = invoke(stypy.reporting.localization.Localization(__file__, 197, 21), list_200450, *[range_call_result_200454], **kwargs_200455)
    
    # Assigning a type to the variable 'perm_r' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'perm_r', list_call_result_200456)
    
    
    # Call to enumerate(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'p' (line 198)
    p_200458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 36), 'p', False)
    # Processing the call keyword arguments (line 198)
    kwargs_200459 = {}
    # Getting the type of 'enumerate' (line 198)
    enumerate_200457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 26), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 198)
    enumerate_call_result_200460 = invoke(stypy.reporting.localization.Localization(__file__, 198, 26), enumerate_200457, *[p_200458], **kwargs_200459)
    
    # Testing the type of a for loop iterable (line 198)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 198, 12), enumerate_call_result_200460)
    # Getting the type of the for loop variable (line 198)
    for_loop_var_200461 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 198, 12), enumerate_call_result_200460)
    # Assigning a type to the variable 'i1' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'i1', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 12), for_loop_var_200461))
    # Assigning a type to the variable 'i2' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'i2', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 12), for_loop_var_200461))
    # SSA begins for a for statement (line 198)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Tuple to a Tuple (line 199):
    
    # Assigning a Subscript to a Name (line 199):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i2' (line 199)
    i2_200462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 48), 'i2')
    # Getting the type of 'perm_r' (line 199)
    perm_r_200463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 41), 'perm_r')
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___200464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 41), perm_r_200463, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_200465 = invoke(stypy.reporting.localization.Localization(__file__, 199, 41), getitem___200464, i2_200462)
    
    # Assigning a type to the variable 'tuple_assignment_200111' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'tuple_assignment_200111', subscript_call_result_200465)
    
    # Assigning a Subscript to a Name (line 199):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i1' (line 199)
    i1_200466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 60), 'i1')
    # Getting the type of 'perm_r' (line 199)
    perm_r_200467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 53), 'perm_r')
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___200468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 53), perm_r_200467, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_200469 = invoke(stypy.reporting.localization.Localization(__file__, 199, 53), getitem___200468, i1_200466)
    
    # Assigning a type to the variable 'tuple_assignment_200112' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'tuple_assignment_200112', subscript_call_result_200469)
    
    # Assigning a Name to a Subscript (line 199):
    # Getting the type of 'tuple_assignment_200111' (line 199)
    tuple_assignment_200111_200470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'tuple_assignment_200111')
    # Getting the type of 'perm_r' (line 199)
    perm_r_200471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'perm_r')
    # Getting the type of 'i1' (line 199)
    i1_200472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 23), 'i1')
    # Storing an element on a container (line 199)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 16), perm_r_200471, (i1_200472, tuple_assignment_200111_200470))
    
    # Assigning a Name to a Subscript (line 199):
    # Getting the type of 'tuple_assignment_200112' (line 199)
    tuple_assignment_200112_200473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'tuple_assignment_200112')
    # Getting the type of 'perm_r' (line 199)
    perm_r_200474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 28), 'perm_r')
    # Getting the type of 'i2' (line 199)
    i2_200475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 35), 'i2')
    # Storing an element on a container (line 199)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 28), perm_r_200474, (i2_200475, tuple_assignment_200112_200473))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 191)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 201):
    
    # Assigning a Call to a Name (line 201):
    
    # Call to lu_solve(...): (line 201)
    # Processing the call arguments (line 201)
    # Getting the type of 'lu' (line 201)
    lu_200479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 35), 'lu', False)
    # Getting the type of 'e' (line 201)
    e_200480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 39), 'e', False)
    # Processing the call keyword arguments (line 201)
    int_200481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 48), 'int')
    keyword_200482 = int_200481
    kwargs_200483 = {'trans': keyword_200482}
    # Getting the type of 'scipy' (line 201)
    scipy_200476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 13), 'scipy', False)
    # Obtaining the member 'linalg' of a type (line 201)
    linalg_200477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 13), scipy_200476, 'linalg')
    # Obtaining the member 'lu_solve' of a type (line 201)
    lu_solve_200478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 13), linalg_200477, 'lu_solve')
    # Calling lu_solve(args, kwargs) (line 201)
    lu_solve_call_result_200484 = invoke(stypy.reporting.localization.Localization(__file__, 201, 13), lu_solve_200478, *[lu_200479, e_200480], **kwargs_200483)
    
    # Assigning a type to the variable 'pi' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'pi', lu_solve_call_result_200484)
    
    # Assigning a Call to a Name (line 204):
    
    # Assigning a Call to a Name (line 204):
    
    # Call to array(...): (line 204)
    # Processing the call arguments (line 204)
    
    # Call to list(...): (line 204)
    # Processing the call arguments (line 204)
    # Getting the type of 'k' (line 204)
    k_200488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 27), 'k', False)
    
    # Call to set(...): (line 204)
    # Processing the call arguments (line 204)
    # Getting the type of 'b' (line 204)
    b_200490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 33), 'b', False)
    # Processing the call keyword arguments (line 204)
    kwargs_200491 = {}
    # Getting the type of 'set' (line 204)
    set_200489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 29), 'set', False)
    # Calling set(args, kwargs) (line 204)
    set_call_result_200492 = invoke(stypy.reporting.localization.Localization(__file__, 204, 29), set_200489, *[b_200490], **kwargs_200491)
    
    # Applying the binary operator '-' (line 204)
    result_sub_200493 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 27), '-', k_200488, set_call_result_200492)
    
    # Processing the call keyword arguments (line 204)
    kwargs_200494 = {}
    # Getting the type of 'list' (line 204)
    list_200487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 22), 'list', False)
    # Calling list(args, kwargs) (line 204)
    list_call_result_200495 = invoke(stypy.reporting.localization.Localization(__file__, 204, 22), list_200487, *[result_sub_200493], **kwargs_200494)
    
    # Processing the call keyword arguments (line 204)
    kwargs_200496 = {}
    # Getting the type of 'np' (line 204)
    np_200485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 13), 'np', False)
    # Obtaining the member 'array' of a type (line 204)
    array_200486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 13), np_200485, 'array')
    # Calling array(args, kwargs) (line 204)
    array_call_result_200497 = invoke(stypy.reporting.localization.Localization(__file__, 204, 13), array_200486, *[list_call_result_200495], **kwargs_200496)
    
    # Assigning a type to the variable 'js' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'js', array_call_result_200497)
    
    # Assigning a Num to a Name (line 205):
    
    # Assigning a Num to a Name (line 205):
    int_200498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 16), 'int')
    # Assigning a type to the variable 'batch' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'batch', int_200498)
    
    # Assigning a Name to a Name (line 206):
    
    # Assigning a Name to a Name (line 206):
    # Getting the type of 'True' (line 206)
    True_200499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 20), 'True')
    # Assigning a type to the variable 'dependent' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'dependent', True_200499)
    
    
    # Call to range(...): (line 210)
    # Processing the call arguments (line 210)
    int_200501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 29), 'int')
    
    # Call to len(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'js' (line 210)
    js_200503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 36), 'js', False)
    # Processing the call keyword arguments (line 210)
    kwargs_200504 = {}
    # Getting the type of 'len' (line 210)
    len_200502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 32), 'len', False)
    # Calling len(args, kwargs) (line 210)
    len_call_result_200505 = invoke(stypy.reporting.localization.Localization(__file__, 210, 32), len_200502, *[js_200503], **kwargs_200504)
    
    # Getting the type of 'batch' (line 210)
    batch_200506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 41), 'batch', False)
    # Processing the call keyword arguments (line 210)
    kwargs_200507 = {}
    # Getting the type of 'range' (line 210)
    range_200500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 23), 'range', False)
    # Calling range(args, kwargs) (line 210)
    range_call_result_200508 = invoke(stypy.reporting.localization.Localization(__file__, 210, 23), range_200500, *[int_200501, len_call_result_200505, batch_200506], **kwargs_200507)
    
    # Testing the type of a for loop iterable (line 210)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 210, 8), range_call_result_200508)
    # Getting the type of the for loop variable (line 210)
    for_loop_var_200509 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 210, 8), range_call_result_200508)
    # Assigning a type to the variable 'j_index' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'j_index', for_loop_var_200509)
    # SSA begins for a for statement (line 210)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 211):
    
    # Assigning a Subscript to a Name (line 211):
    
    # Obtaining the type of the subscript
    
    # Call to arange(...): (line 211)
    # Processing the call arguments (line 211)
    # Getting the type of 'j_index' (line 211)
    j_index_200512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 37), 'j_index', False)
    
    # Call to min(...): (line 211)
    # Processing the call arguments (line 211)
    # Getting the type of 'j_index' (line 211)
    j_index_200514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 50), 'j_index', False)
    # Getting the type of 'batch' (line 211)
    batch_200515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 58), 'batch', False)
    # Applying the binary operator '+' (line 211)
    result_add_200516 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 50), '+', j_index_200514, batch_200515)
    
    
    # Call to len(...): (line 211)
    # Processing the call arguments (line 211)
    # Getting the type of 'js' (line 211)
    js_200518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 69), 'js', False)
    # Processing the call keyword arguments (line 211)
    kwargs_200519 = {}
    # Getting the type of 'len' (line 211)
    len_200517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 65), 'len', False)
    # Calling len(args, kwargs) (line 211)
    len_call_result_200520 = invoke(stypy.reporting.localization.Localization(__file__, 211, 65), len_200517, *[js_200518], **kwargs_200519)
    
    # Processing the call keyword arguments (line 211)
    kwargs_200521 = {}
    # Getting the type of 'min' (line 211)
    min_200513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 46), 'min', False)
    # Calling min(args, kwargs) (line 211)
    min_call_result_200522 = invoke(stypy.reporting.localization.Localization(__file__, 211, 46), min_200513, *[result_add_200516, len_call_result_200520], **kwargs_200521)
    
    # Processing the call keyword arguments (line 211)
    kwargs_200523 = {}
    # Getting the type of 'np' (line 211)
    np_200510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 27), 'np', False)
    # Obtaining the member 'arange' of a type (line 211)
    arange_200511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 27), np_200510, 'arange')
    # Calling arange(args, kwargs) (line 211)
    arange_call_result_200524 = invoke(stypy.reporting.localization.Localization(__file__, 211, 27), arange_200511, *[j_index_200512, min_call_result_200522], **kwargs_200523)
    
    # Getting the type of 'js' (line 211)
    js_200525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 24), 'js')
    # Obtaining the member '__getitem__' of a type (line 211)
    getitem___200526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 24), js_200525, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 211)
    subscript_call_result_200527 = invoke(stypy.reporting.localization.Localization(__file__, 211, 24), getitem___200526, arange_call_result_200524)
    
    # Assigning a type to the variable 'j_indices' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'j_indices', subscript_call_result_200527)
    
    # Assigning a Call to a Name (line 213):
    
    # Assigning a Call to a Name (line 213):
    
    # Call to abs(...): (line 213)
    # Processing the call arguments (line 213)
    
    # Call to dot(...): (line 213)
    # Processing the call arguments (line 213)
    # Getting the type of 'pi' (line 213)
    pi_200538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 52), 'pi', False)
    # Processing the call keyword arguments (line 213)
    kwargs_200539 = {}
    
    # Call to transpose(...): (line 213)
    # Processing the call keyword arguments (line 213)
    kwargs_200535 = {}
    
    # Obtaining the type of the subscript
    slice_200529 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 213, 20), None, None, None)
    # Getting the type of 'j_indices' (line 213)
    j_indices_200530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 25), 'j_indices', False)
    # Getting the type of 'A' (line 213)
    A_200531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 20), 'A', False)
    # Obtaining the member '__getitem__' of a type (line 213)
    getitem___200532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 20), A_200531, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 213)
    subscript_call_result_200533 = invoke(stypy.reporting.localization.Localization(__file__, 213, 20), getitem___200532, (slice_200529, j_indices_200530))
    
    # Obtaining the member 'transpose' of a type (line 213)
    transpose_200534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 20), subscript_call_result_200533, 'transpose')
    # Calling transpose(args, kwargs) (line 213)
    transpose_call_result_200536 = invoke(stypy.reporting.localization.Localization(__file__, 213, 20), transpose_200534, *[], **kwargs_200535)
    
    # Obtaining the member 'dot' of a type (line 213)
    dot_200537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 20), transpose_call_result_200536, 'dot')
    # Calling dot(args, kwargs) (line 213)
    dot_call_result_200540 = invoke(stypy.reporting.localization.Localization(__file__, 213, 20), dot_200537, *[pi_200538], **kwargs_200539)
    
    # Processing the call keyword arguments (line 213)
    kwargs_200541 = {}
    # Getting the type of 'abs' (line 213)
    abs_200528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'abs', False)
    # Calling abs(args, kwargs) (line 213)
    abs_call_result_200542 = invoke(stypy.reporting.localization.Localization(__file__, 213, 16), abs_200528, *[dot_call_result_200540], **kwargs_200541)
    
    # Assigning a type to the variable 'c' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'c', abs_call_result_200542)
    
    
    # Call to any(...): (line 214)
    # Processing the call keyword arguments (line 214)
    kwargs_200547 = {}
    
    # Getting the type of 'c' (line 214)
    c_200543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 'c', False)
    # Getting the type of 'tolapiv' (line 214)
    tolapiv_200544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 20), 'tolapiv', False)
    # Applying the binary operator '>' (line 214)
    result_gt_200545 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 16), '>', c_200543, tolapiv_200544)
    
    # Obtaining the member 'any' of a type (line 214)
    any_200546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 16), result_gt_200545, 'any')
    # Calling any(args, kwargs) (line 214)
    any_call_result_200548 = invoke(stypy.reporting.localization.Localization(__file__, 214, 16), any_200546, *[], **kwargs_200547)
    
    # Testing the type of an if condition (line 214)
    if_condition_200549 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 214, 12), any_call_result_200548)
    # Assigning a type to the variable 'if_condition_200549' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'if_condition_200549', if_condition_200549)
    # SSA begins for if statement (line 214)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 215):
    
    # Assigning a Subscript to a Name (line 215):
    
    # Obtaining the type of the subscript
    # Getting the type of 'j_index' (line 215)
    j_index_200550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 23), 'j_index')
    
    # Call to argmax(...): (line 215)
    # Processing the call arguments (line 215)
    # Getting the type of 'c' (line 215)
    c_200553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 43), 'c', False)
    # Processing the call keyword arguments (line 215)
    kwargs_200554 = {}
    # Getting the type of 'np' (line 215)
    np_200551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 33), 'np', False)
    # Obtaining the member 'argmax' of a type (line 215)
    argmax_200552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 33), np_200551, 'argmax')
    # Calling argmax(args, kwargs) (line 215)
    argmax_call_result_200555 = invoke(stypy.reporting.localization.Localization(__file__, 215, 33), argmax_200552, *[c_200553], **kwargs_200554)
    
    # Applying the binary operator '+' (line 215)
    result_add_200556 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 23), '+', j_index_200550, argmax_call_result_200555)
    
    # Getting the type of 'js' (line 215)
    js_200557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 20), 'js')
    # Obtaining the member '__getitem__' of a type (line 215)
    getitem___200558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 20), js_200557, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 215)
    subscript_call_result_200559 = invoke(stypy.reporting.localization.Localization(__file__, 215, 20), getitem___200558, result_add_200556)
    
    # Assigning a type to the variable 'j' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 16), 'j', subscript_call_result_200559)
    
    # Assigning a Subscript to a Subscript (line 216):
    
    # Assigning a Subscript to a Subscript (line 216):
    
    # Obtaining the type of the subscript
    slice_200560 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 216, 26), None, None, None)
    # Getting the type of 'j' (line 216)
    j_200561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 31), 'j')
    # Getting the type of 'A' (line 216)
    A_200562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 26), 'A')
    # Obtaining the member '__getitem__' of a type (line 216)
    getitem___200563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 26), A_200562, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 216)
    subscript_call_result_200564 = invoke(stypy.reporting.localization.Localization(__file__, 216, 26), getitem___200563, (slice_200560, j_200561))
    
    # Getting the type of 'B' (line 216)
    B_200565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'B')
    slice_200566 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 216, 16), None, None, None)
    # Getting the type of 'i' (line 216)
    i_200567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 21), 'i')
    # Storing an element on a container (line 216)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 16), B_200565, ((slice_200566, i_200567), subscript_call_result_200564))
    
    # Assigning a Name to a Subscript (line 217):
    
    # Assigning a Name to a Subscript (line 217):
    # Getting the type of 'j' (line 217)
    j_200568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 23), 'j')
    # Getting the type of 'b' (line 217)
    b_200569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 16), 'b')
    # Getting the type of 'i' (line 217)
    i_200570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 18), 'i')
    # Storing an element on a container (line 217)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 16), b_200569, (i_200570, j_200568))
    
    # Assigning a Name to a Name (line 218):
    
    # Assigning a Name to a Name (line 218):
    # Getting the type of 'False' (line 218)
    False_200571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 28), 'False')
    # Assigning a type to the variable 'dependent' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'dependent', False_200571)
    # SSA join for if statement (line 214)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'dependent' (line 220)
    dependent_200572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 11), 'dependent')
    # Testing the type of an if condition (line 220)
    if_condition_200573 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 8), dependent_200572)
    # Assigning a type to the variable 'if_condition_200573' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'if_condition_200573', if_condition_200573)
    # SSA begins for if statement (line 220)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 221):
    
    # Assigning a Call to a Name (line 221):
    
    # Call to dot(...): (line 221)
    # Processing the call arguments (line 221)
    
    # Call to reshape(...): (line 221)
    # Processing the call arguments (line 221)
    int_200579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 41), 'int')
    int_200580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 45), 'int')
    # Processing the call keyword arguments (line 221)
    kwargs_200581 = {}
    # Getting the type of 'rhs' (line 221)
    rhs_200577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 29), 'rhs', False)
    # Obtaining the member 'reshape' of a type (line 221)
    reshape_200578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 29), rhs_200577, 'reshape')
    # Calling reshape(args, kwargs) (line 221)
    reshape_call_result_200582 = invoke(stypy.reporting.localization.Localization(__file__, 221, 29), reshape_200578, *[int_200579, int_200580], **kwargs_200581)
    
    # Processing the call keyword arguments (line 221)
    kwargs_200583 = {}
    # Getting the type of 'pi' (line 221)
    pi_200574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 20), 'pi', False)
    # Obtaining the member 'T' of a type (line 221)
    T_200575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 20), pi_200574, 'T')
    # Obtaining the member 'dot' of a type (line 221)
    dot_200576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 20), T_200575, 'dot')
    # Calling dot(args, kwargs) (line 221)
    dot_call_result_200584 = invoke(stypy.reporting.localization.Localization(__file__, 221, 20), dot_200576, *[reshape_call_result_200582], **kwargs_200583)
    
    # Assigning a type to the variable 'bibar' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'bibar', dot_call_result_200584)
    
    # Assigning a Call to a Name (line 222):
    
    # Assigning a Call to a Name (line 222):
    
    # Call to norm(...): (line 222)
    # Processing the call arguments (line 222)
    # Getting the type of 'rhs' (line 222)
    rhs_200588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 35), 'rhs', False)
    # Processing the call keyword arguments (line 222)
    kwargs_200589 = {}
    # Getting the type of 'np' (line 222)
    np_200585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'np', False)
    # Obtaining the member 'linalg' of a type (line 222)
    linalg_200586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 20), np_200585, 'linalg')
    # Obtaining the member 'norm' of a type (line 222)
    norm_200587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 20), linalg_200586, 'norm')
    # Calling norm(args, kwargs) (line 222)
    norm_call_result_200590 = invoke(stypy.reporting.localization.Localization(__file__, 222, 20), norm_200587, *[rhs_200588], **kwargs_200589)
    
    # Assigning a type to the variable 'bnorm' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'bnorm', norm_call_result_200590)
    
    
    
    # Call to abs(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'bibar' (line 223)
    bibar_200592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 19), 'bibar', False)
    # Processing the call keyword arguments (line 223)
    kwargs_200593 = {}
    # Getting the type of 'abs' (line 223)
    abs_200591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 15), 'abs', False)
    # Calling abs(args, kwargs) (line 223)
    abs_call_result_200594 = invoke(stypy.reporting.localization.Localization(__file__, 223, 15), abs_200591, *[bibar_200592], **kwargs_200593)
    
    int_200595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 27), 'int')
    # Getting the type of 'bnorm' (line 223)
    bnorm_200596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 29), 'bnorm')
    # Applying the binary operator '+' (line 223)
    result_add_200597 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 27), '+', int_200595, bnorm_200596)
    
    # Applying the binary operator 'div' (line 223)
    result_div_200598 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 15), 'div', abs_call_result_200594, result_add_200597)
    
    # Getting the type of 'tolprimal' (line 223)
    tolprimal_200599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 38), 'tolprimal')
    # Applying the binary operator '>' (line 223)
    result_gt_200600 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 15), '>', result_div_200598, tolprimal_200599)
    
    # Testing the type of an if condition (line 223)
    if_condition_200601 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 12), result_gt_200600)
    # Assigning a type to the variable 'if_condition_200601' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'if_condition_200601', if_condition_200601)
    # SSA begins for if statement (line 223)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 224):
    
    # Assigning a Num to a Name (line 224):
    int_200602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 25), 'int')
    # Assigning a type to the variable 'status' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'status', int_200602)
    
    # Assigning a Name to a Name (line 225):
    
    # Assigning a Name to a Name (line 225):
    # Getting the type of 'inconsistent' (line 225)
    inconsistent_200603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 26), 'inconsistent')
    # Assigning a type to the variable 'message' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'message', inconsistent_200603)
    
    # Obtaining an instance of the builtin type 'tuple' (line 226)
    tuple_200604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 226)
    # Adding element type (line 226)
    # Getting the type of 'A_orig' (line 226)
    A_orig_200605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 23), 'A_orig')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 23), tuple_200604, A_orig_200605)
    # Adding element type (line 226)
    # Getting the type of 'rhs' (line 226)
    rhs_200606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 31), 'rhs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 23), tuple_200604, rhs_200606)
    # Adding element type (line 226)
    # Getting the type of 'status' (line 226)
    status_200607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 36), 'status')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 23), tuple_200604, status_200607)
    # Adding element type (line 226)
    # Getting the type of 'message' (line 226)
    message_200608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 44), 'message')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 23), tuple_200604, message_200608)
    
    # Assigning a type to the variable 'stypy_return_type' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 16), 'stypy_return_type', tuple_200604)
    # SSA branch for the else part of an if statement (line 223)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'i' (line 228)
    i_200611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 25), 'i', False)
    # Processing the call keyword arguments (line 228)
    kwargs_200612 = {}
    # Getting the type of 'd' (line 228)
    d_200609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'd', False)
    # Obtaining the member 'append' of a type (line 228)
    append_200610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 16), d_200609, 'append')
    # Calling append(args, kwargs) (line 228)
    append_call_result_200613 = invoke(stypy.reporting.localization.Localization(__file__, 228, 16), append_200610, *[i_200611], **kwargs_200612)
    
    # SSA join for if statement (line 223)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 220)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 230):
    
    # Assigning a Call to a Name (line 230):
    
    # Call to set(...): (line 230)
    # Processing the call arguments (line 230)
    
    # Call to range(...): (line 230)
    # Processing the call arguments (line 230)
    # Getting the type of 'm' (line 230)
    m_200616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 21), 'm', False)
    # Processing the call keyword arguments (line 230)
    kwargs_200617 = {}
    # Getting the type of 'range' (line 230)
    range_200615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 15), 'range', False)
    # Calling range(args, kwargs) (line 230)
    range_call_result_200618 = invoke(stypy.reporting.localization.Localization(__file__, 230, 15), range_200615, *[m_200616], **kwargs_200617)
    
    # Processing the call keyword arguments (line 230)
    kwargs_200619 = {}
    # Getting the type of 'set' (line 230)
    set_200614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 11), 'set', False)
    # Calling set(args, kwargs) (line 230)
    set_call_result_200620 = invoke(stypy.reporting.localization.Localization(__file__, 230, 11), set_200614, *[range_call_result_200618], **kwargs_200619)
    
    # Assigning a type to the variable 'keep' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'keep', set_call_result_200620)
    
    # Assigning a Call to a Name (line 231):
    
    # Assigning a Call to a Name (line 231):
    
    # Call to list(...): (line 231)
    # Processing the call arguments (line 231)
    # Getting the type of 'keep' (line 231)
    keep_200622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 16), 'keep', False)
    
    # Call to set(...): (line 231)
    # Processing the call arguments (line 231)
    # Getting the type of 'd' (line 231)
    d_200624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 27), 'd', False)
    # Processing the call keyword arguments (line 231)
    kwargs_200625 = {}
    # Getting the type of 'set' (line 231)
    set_200623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 23), 'set', False)
    # Calling set(args, kwargs) (line 231)
    set_call_result_200626 = invoke(stypy.reporting.localization.Localization(__file__, 231, 23), set_200623, *[d_200624], **kwargs_200625)
    
    # Applying the binary operator '-' (line 231)
    result_sub_200627 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 16), '-', keep_200622, set_call_result_200626)
    
    # Processing the call keyword arguments (line 231)
    kwargs_200628 = {}
    # Getting the type of 'list' (line 231)
    list_200621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 11), 'list', False)
    # Calling list(args, kwargs) (line 231)
    list_call_result_200629 = invoke(stypy.reporting.localization.Localization(__file__, 231, 11), list_200621, *[result_sub_200627], **kwargs_200628)
    
    # Assigning a type to the variable 'keep' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'keep', list_call_result_200629)
    
    # Obtaining an instance of the builtin type 'tuple' (line 232)
    tuple_200630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 232)
    # Adding element type (line 232)
    
    # Obtaining the type of the subscript
    # Getting the type of 'keep' (line 232)
    keep_200631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 18), 'keep')
    slice_200632 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 232, 11), None, None, None)
    # Getting the type of 'A_orig' (line 232)
    A_orig_200633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 11), 'A_orig')
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___200634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 11), A_orig_200633, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_200635 = invoke(stypy.reporting.localization.Localization(__file__, 232, 11), getitem___200634, (keep_200631, slice_200632))
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 11), tuple_200630, subscript_call_result_200635)
    # Adding element type (line 232)
    
    # Obtaining the type of the subscript
    # Getting the type of 'keep' (line 232)
    keep_200636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 32), 'keep')
    # Getting the type of 'rhs' (line 232)
    rhs_200637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 28), 'rhs')
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___200638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 28), rhs_200637, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_200639 = invoke(stypy.reporting.localization.Localization(__file__, 232, 28), getitem___200638, keep_200636)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 11), tuple_200630, subscript_call_result_200639)
    # Adding element type (line 232)
    # Getting the type of 'status' (line 232)
    status_200640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 39), 'status')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 11), tuple_200630, status_200640)
    # Adding element type (line 232)
    # Getting the type of 'message' (line 232)
    message_200641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 47), 'message')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 11), tuple_200630, message_200641)
    
    # Assigning a type to the variable 'stypy_return_type' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'stypy_return_type', tuple_200630)
    
    # ################# End of '_remove_redundancy_dense(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_remove_redundancy_dense' in the type store
    # Getting the type of 'stypy_return_type' (line 107)
    stypy_return_type_200642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_200642)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_remove_redundancy_dense'
    return stypy_return_type_200642

# Assigning a type to the variable '_remove_redundancy_dense' (line 107)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), '_remove_redundancy_dense', _remove_redundancy_dense)

@norecursion
def _remove_redundancy_sparse(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_remove_redundancy_sparse'
    module_type_store = module_type_store.open_function_context('_remove_redundancy_sparse', 235, 0, False)
    
    # Passed parameters checking function
    _remove_redundancy_sparse.stypy_localization = localization
    _remove_redundancy_sparse.stypy_type_of_self = None
    _remove_redundancy_sparse.stypy_type_store = module_type_store
    _remove_redundancy_sparse.stypy_function_name = '_remove_redundancy_sparse'
    _remove_redundancy_sparse.stypy_param_names_list = ['A', 'rhs']
    _remove_redundancy_sparse.stypy_varargs_param_name = None
    _remove_redundancy_sparse.stypy_kwargs_param_name = None
    _remove_redundancy_sparse.stypy_call_defaults = defaults
    _remove_redundancy_sparse.stypy_call_varargs = varargs
    _remove_redundancy_sparse.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_remove_redundancy_sparse', ['A', 'rhs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_remove_redundancy_sparse', localization, ['A', 'rhs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_remove_redundancy_sparse(...)' code ##################

    str_200643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, (-1)), 'str', '\n    Eliminates redundant equations from system of equations defined by Ax = b\n    and identifies infeasibilities.\n\n    Parameters\n    ----------\n    A : 2-D sparse matrix\n        An matrix representing the left-hand side of a system of equations\n    rhs : 1-D array\n        An array representing the right-hand side of a system of equations\n\n    Returns\n    -------\n    A : 2-D sparse matrix\n        A matrix representing the left-hand side of a system of equations\n    rhs : 1-D array\n        An array representing the right-hand side of a system of equations\n    status: int\n        An integer indicating the status of the system\n        0: No infeasibility identified\n        2: Trivially infeasible\n    message : str\n        A string descriptor of the exit status of the optimization.\n\n    References\n    ----------\n    .. [2] Andersen, Erling D. "Finding all linearly dependent rows in\n           large-scale linear programming." Optimization Methods and Software\n           6.3 (1995): 219-227.\n\n    ')
    
    # Assigning a Num to a Name (line 268):
    
    # Assigning a Num to a Name (line 268):
    float_200644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 14), 'float')
    # Assigning a type to the variable 'tolapiv' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'tolapiv', float_200644)
    
    # Assigning a Num to a Name (line 269):
    
    # Assigning a Num to a Name (line 269):
    float_200645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 16), 'float')
    # Assigning a type to the variable 'tolprimal' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'tolprimal', float_200645)
    
    # Assigning a Num to a Name (line 270):
    
    # Assigning a Num to a Name (line 270):
    int_200646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 13), 'int')
    # Assigning a type to the variable 'status' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'status', int_200646)
    
    # Assigning a Str to a Name (line 271):
    
    # Assigning a Str to a Name (line 271):
    str_200647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 14), 'str', '')
    # Assigning a type to the variable 'message' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'message', str_200647)
    
    # Assigning a Str to a Name (line 272):
    
    # Assigning a Str to a Name (line 272):
    str_200648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 20), 'str', 'There is a linear combination of rows of A_eq that results in zero, suggesting a redundant constraint. However the same linear combination of b_eq is nonzero, suggesting that the constraints conflict and the problem is infeasible.')
    # Assigning a type to the variable 'inconsistent' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'inconsistent', str_200648)
    
    # Assigning a Call to a Tuple (line 277):
    
    # Assigning a Subscript to a Name (line 277):
    
    # Obtaining the type of the subscript
    int_200649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 4), 'int')
    
    # Call to _remove_zero_rows(...): (line 277)
    # Processing the call arguments (line 277)
    # Getting the type of 'A' (line 277)
    A_200651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 48), 'A', False)
    # Getting the type of 'rhs' (line 277)
    rhs_200652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 51), 'rhs', False)
    # Processing the call keyword arguments (line 277)
    kwargs_200653 = {}
    # Getting the type of '_remove_zero_rows' (line 277)
    _remove_zero_rows_200650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 30), '_remove_zero_rows', False)
    # Calling _remove_zero_rows(args, kwargs) (line 277)
    _remove_zero_rows_call_result_200654 = invoke(stypy.reporting.localization.Localization(__file__, 277, 30), _remove_zero_rows_200650, *[A_200651, rhs_200652], **kwargs_200653)
    
    # Obtaining the member '__getitem__' of a type (line 277)
    getitem___200655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 4), _remove_zero_rows_call_result_200654, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 277)
    subscript_call_result_200656 = invoke(stypy.reporting.localization.Localization(__file__, 277, 4), getitem___200655, int_200649)
    
    # Assigning a type to the variable 'tuple_var_assignment_200113' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'tuple_var_assignment_200113', subscript_call_result_200656)
    
    # Assigning a Subscript to a Name (line 277):
    
    # Obtaining the type of the subscript
    int_200657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 4), 'int')
    
    # Call to _remove_zero_rows(...): (line 277)
    # Processing the call arguments (line 277)
    # Getting the type of 'A' (line 277)
    A_200659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 48), 'A', False)
    # Getting the type of 'rhs' (line 277)
    rhs_200660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 51), 'rhs', False)
    # Processing the call keyword arguments (line 277)
    kwargs_200661 = {}
    # Getting the type of '_remove_zero_rows' (line 277)
    _remove_zero_rows_200658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 30), '_remove_zero_rows', False)
    # Calling _remove_zero_rows(args, kwargs) (line 277)
    _remove_zero_rows_call_result_200662 = invoke(stypy.reporting.localization.Localization(__file__, 277, 30), _remove_zero_rows_200658, *[A_200659, rhs_200660], **kwargs_200661)
    
    # Obtaining the member '__getitem__' of a type (line 277)
    getitem___200663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 4), _remove_zero_rows_call_result_200662, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 277)
    subscript_call_result_200664 = invoke(stypy.reporting.localization.Localization(__file__, 277, 4), getitem___200663, int_200657)
    
    # Assigning a type to the variable 'tuple_var_assignment_200114' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'tuple_var_assignment_200114', subscript_call_result_200664)
    
    # Assigning a Subscript to a Name (line 277):
    
    # Obtaining the type of the subscript
    int_200665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 4), 'int')
    
    # Call to _remove_zero_rows(...): (line 277)
    # Processing the call arguments (line 277)
    # Getting the type of 'A' (line 277)
    A_200667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 48), 'A', False)
    # Getting the type of 'rhs' (line 277)
    rhs_200668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 51), 'rhs', False)
    # Processing the call keyword arguments (line 277)
    kwargs_200669 = {}
    # Getting the type of '_remove_zero_rows' (line 277)
    _remove_zero_rows_200666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 30), '_remove_zero_rows', False)
    # Calling _remove_zero_rows(args, kwargs) (line 277)
    _remove_zero_rows_call_result_200670 = invoke(stypy.reporting.localization.Localization(__file__, 277, 30), _remove_zero_rows_200666, *[A_200667, rhs_200668], **kwargs_200669)
    
    # Obtaining the member '__getitem__' of a type (line 277)
    getitem___200671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 4), _remove_zero_rows_call_result_200670, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 277)
    subscript_call_result_200672 = invoke(stypy.reporting.localization.Localization(__file__, 277, 4), getitem___200671, int_200665)
    
    # Assigning a type to the variable 'tuple_var_assignment_200115' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'tuple_var_assignment_200115', subscript_call_result_200672)
    
    # Assigning a Subscript to a Name (line 277):
    
    # Obtaining the type of the subscript
    int_200673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 4), 'int')
    
    # Call to _remove_zero_rows(...): (line 277)
    # Processing the call arguments (line 277)
    # Getting the type of 'A' (line 277)
    A_200675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 48), 'A', False)
    # Getting the type of 'rhs' (line 277)
    rhs_200676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 51), 'rhs', False)
    # Processing the call keyword arguments (line 277)
    kwargs_200677 = {}
    # Getting the type of '_remove_zero_rows' (line 277)
    _remove_zero_rows_200674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 30), '_remove_zero_rows', False)
    # Calling _remove_zero_rows(args, kwargs) (line 277)
    _remove_zero_rows_call_result_200678 = invoke(stypy.reporting.localization.Localization(__file__, 277, 30), _remove_zero_rows_200674, *[A_200675, rhs_200676], **kwargs_200677)
    
    # Obtaining the member '__getitem__' of a type (line 277)
    getitem___200679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 4), _remove_zero_rows_call_result_200678, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 277)
    subscript_call_result_200680 = invoke(stypy.reporting.localization.Localization(__file__, 277, 4), getitem___200679, int_200673)
    
    # Assigning a type to the variable 'tuple_var_assignment_200116' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'tuple_var_assignment_200116', subscript_call_result_200680)
    
    # Assigning a Name to a Name (line 277):
    # Getting the type of 'tuple_var_assignment_200113' (line 277)
    tuple_var_assignment_200113_200681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'tuple_var_assignment_200113')
    # Assigning a type to the variable 'A' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'A', tuple_var_assignment_200113_200681)
    
    # Assigning a Name to a Name (line 277):
    # Getting the type of 'tuple_var_assignment_200114' (line 277)
    tuple_var_assignment_200114_200682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'tuple_var_assignment_200114')
    # Assigning a type to the variable 'rhs' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 7), 'rhs', tuple_var_assignment_200114_200682)
    
    # Assigning a Name to a Name (line 277):
    # Getting the type of 'tuple_var_assignment_200115' (line 277)
    tuple_var_assignment_200115_200683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'tuple_var_assignment_200115')
    # Assigning a type to the variable 'status' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'status', tuple_var_assignment_200115_200683)
    
    # Assigning a Name to a Name (line 277):
    # Getting the type of 'tuple_var_assignment_200116' (line 277)
    tuple_var_assignment_200116_200684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'tuple_var_assignment_200116')
    # Assigning a type to the variable 'message' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 20), 'message', tuple_var_assignment_200116_200684)
    
    
    # Getting the type of 'status' (line 279)
    status_200685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 7), 'status')
    int_200686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 17), 'int')
    # Applying the binary operator '!=' (line 279)
    result_ne_200687 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 7), '!=', status_200685, int_200686)
    
    # Testing the type of an if condition (line 279)
    if_condition_200688 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 4), result_ne_200687)
    # Assigning a type to the variable 'if_condition_200688' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'if_condition_200688', if_condition_200688)
    # SSA begins for if statement (line 279)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 280)
    tuple_200689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 280)
    # Adding element type (line 280)
    # Getting the type of 'A' (line 280)
    A_200690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 15), 'A')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 15), tuple_200689, A_200690)
    # Adding element type (line 280)
    # Getting the type of 'rhs' (line 280)
    rhs_200691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 18), 'rhs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 15), tuple_200689, rhs_200691)
    # Adding element type (line 280)
    # Getting the type of 'status' (line 280)
    status_200692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 23), 'status')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 15), tuple_200689, status_200692)
    # Adding element type (line 280)
    # Getting the type of 'message' (line 280)
    message_200693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 31), 'message')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 15), tuple_200689, message_200693)
    
    # Assigning a type to the variable 'stypy_return_type' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'stypy_return_type', tuple_200689)
    # SSA join for if statement (line 279)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Tuple (line 282):
    
    # Assigning a Subscript to a Name (line 282):
    
    # Obtaining the type of the subscript
    int_200694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 4), 'int')
    # Getting the type of 'A' (line 282)
    A_200695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 11), 'A')
    # Obtaining the member 'shape' of a type (line 282)
    shape_200696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 11), A_200695, 'shape')
    # Obtaining the member '__getitem__' of a type (line 282)
    getitem___200697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 4), shape_200696, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 282)
    subscript_call_result_200698 = invoke(stypy.reporting.localization.Localization(__file__, 282, 4), getitem___200697, int_200694)
    
    # Assigning a type to the variable 'tuple_var_assignment_200117' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'tuple_var_assignment_200117', subscript_call_result_200698)
    
    # Assigning a Subscript to a Name (line 282):
    
    # Obtaining the type of the subscript
    int_200699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 4), 'int')
    # Getting the type of 'A' (line 282)
    A_200700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 11), 'A')
    # Obtaining the member 'shape' of a type (line 282)
    shape_200701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 11), A_200700, 'shape')
    # Obtaining the member '__getitem__' of a type (line 282)
    getitem___200702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 4), shape_200701, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 282)
    subscript_call_result_200703 = invoke(stypy.reporting.localization.Localization(__file__, 282, 4), getitem___200702, int_200699)
    
    # Assigning a type to the variable 'tuple_var_assignment_200118' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'tuple_var_assignment_200118', subscript_call_result_200703)
    
    # Assigning a Name to a Name (line 282):
    # Getting the type of 'tuple_var_assignment_200117' (line 282)
    tuple_var_assignment_200117_200704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'tuple_var_assignment_200117')
    # Assigning a type to the variable 'm' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'm', tuple_var_assignment_200117_200704)
    
    # Assigning a Name to a Name (line 282):
    # Getting the type of 'tuple_var_assignment_200118' (line 282)
    tuple_var_assignment_200118_200705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'tuple_var_assignment_200118')
    # Assigning a type to the variable 'n' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 7), 'n', tuple_var_assignment_200118_200705)
    
    # Assigning a Call to a Name (line 284):
    
    # Assigning a Call to a Name (line 284):
    
    # Call to list(...): (line 284)
    # Processing the call arguments (line 284)
    
    # Call to range(...): (line 284)
    # Processing the call arguments (line 284)
    # Getting the type of 'm' (line 284)
    m_200708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 19), 'm', False)
    # Processing the call keyword arguments (line 284)
    kwargs_200709 = {}
    # Getting the type of 'range' (line 284)
    range_200707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 13), 'range', False)
    # Calling range(args, kwargs) (line 284)
    range_call_result_200710 = invoke(stypy.reporting.localization.Localization(__file__, 284, 13), range_200707, *[m_200708], **kwargs_200709)
    
    # Processing the call keyword arguments (line 284)
    kwargs_200711 = {}
    # Getting the type of 'list' (line 284)
    list_200706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'list', False)
    # Calling list(args, kwargs) (line 284)
    list_call_result_200712 = invoke(stypy.reporting.localization.Localization(__file__, 284, 8), list_200706, *[range_call_result_200710], **kwargs_200711)
    
    # Assigning a type to the variable 'v' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'v', list_call_result_200712)
    
    # Assigning a Call to a Name (line 285):
    
    # Assigning a Call to a Name (line 285):
    
    # Call to list(...): (line 285)
    # Processing the call arguments (line 285)
    # Getting the type of 'v' (line 285)
    v_200714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 13), 'v', False)
    # Processing the call keyword arguments (line 285)
    kwargs_200715 = {}
    # Getting the type of 'list' (line 285)
    list_200713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'list', False)
    # Calling list(args, kwargs) (line 285)
    list_call_result_200716 = invoke(stypy.reporting.localization.Localization(__file__, 285, 8), list_200713, *[v_200714], **kwargs_200715)
    
    # Assigning a type to the variable 'b' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'b', list_call_result_200716)
    
    # Assigning a Call to a Name (line 288):
    
    # Assigning a Call to a Name (line 288):
    
    # Call to set(...): (line 288)
    # Processing the call arguments (line 288)
    
    # Call to range(...): (line 288)
    # Processing the call arguments (line 288)
    # Getting the type of 'm' (line 288)
    m_200719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 18), 'm', False)
    # Getting the type of 'm' (line 288)
    m_200720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 21), 'm', False)
    # Getting the type of 'n' (line 288)
    n_200721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 23), 'n', False)
    # Applying the binary operator '+' (line 288)
    result_add_200722 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 21), '+', m_200720, n_200721)
    
    # Processing the call keyword arguments (line 288)
    kwargs_200723 = {}
    # Getting the type of 'range' (line 288)
    range_200718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'range', False)
    # Calling range(args, kwargs) (line 288)
    range_call_result_200724 = invoke(stypy.reporting.localization.Localization(__file__, 288, 12), range_200718, *[m_200719, result_add_200722], **kwargs_200723)
    
    # Processing the call keyword arguments (line 288)
    kwargs_200725 = {}
    # Getting the type of 'set' (line 288)
    set_200717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'set', False)
    # Calling set(args, kwargs) (line 288)
    set_call_result_200726 = invoke(stypy.reporting.localization.Localization(__file__, 288, 8), set_200717, *[range_call_result_200724], **kwargs_200725)
    
    # Assigning a type to the variable 'k' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'k', set_call_result_200726)
    
    # Assigning a List to a Name (line 289):
    
    # Assigning a List to a Name (line 289):
    
    # Obtaining an instance of the builtin type 'list' (line 289)
    list_200727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 289)
    
    # Assigning a type to the variable 'd' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'd', list_200727)
    
    # Assigning a Name to a Name (line 291):
    
    # Assigning a Name to a Name (line 291):
    # Getting the type of 'A' (line 291)
    A_200728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 13), 'A')
    # Assigning a type to the variable 'A_orig' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'A_orig', A_200728)
    
    # Assigning a Call to a Name (line 292):
    
    # Assigning a Call to a Name (line 292):
    
    # Call to tocsc(...): (line 292)
    # Processing the call keyword arguments (line 292)
    kwargs_200743 = {}
    
    # Call to hstack(...): (line 292)
    # Processing the call arguments (line 292)
    
    # Obtaining an instance of the builtin type 'tuple' (line 292)
    tuple_200732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 292)
    # Adding element type (line 292)
    
    # Call to eye(...): (line 292)
    # Processing the call arguments (line 292)
    # Getting the type of 'm' (line 292)
    m_200736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 46), 'm', False)
    # Processing the call keyword arguments (line 292)
    kwargs_200737 = {}
    # Getting the type of 'scipy' (line 292)
    scipy_200733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 29), 'scipy', False)
    # Obtaining the member 'sparse' of a type (line 292)
    sparse_200734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 29), scipy_200733, 'sparse')
    # Obtaining the member 'eye' of a type (line 292)
    eye_200735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 29), sparse_200734, 'eye')
    # Calling eye(args, kwargs) (line 292)
    eye_call_result_200738 = invoke(stypy.reporting.localization.Localization(__file__, 292, 29), eye_200735, *[m_200736], **kwargs_200737)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 29), tuple_200732, eye_call_result_200738)
    # Adding element type (line 292)
    # Getting the type of 'A' (line 292)
    A_200739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 50), 'A', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 29), tuple_200732, A_200739)
    
    # Processing the call keyword arguments (line 292)
    kwargs_200740 = {}
    # Getting the type of 'scipy' (line 292)
    scipy_200729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'scipy', False)
    # Obtaining the member 'sparse' of a type (line 292)
    sparse_200730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), scipy_200729, 'sparse')
    # Obtaining the member 'hstack' of a type (line 292)
    hstack_200731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), sparse_200730, 'hstack')
    # Calling hstack(args, kwargs) (line 292)
    hstack_call_result_200741 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), hstack_200731, *[tuple_200732], **kwargs_200740)
    
    # Obtaining the member 'tocsc' of a type (line 292)
    tocsc_200742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), hstack_call_result_200741, 'tocsc')
    # Calling tocsc(args, kwargs) (line 292)
    tocsc_call_result_200744 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), tocsc_200742, *[], **kwargs_200743)
    
    # Assigning a type to the variable 'A' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'A', tocsc_call_result_200744)
    
    # Assigning a Call to a Name (line 293):
    
    # Assigning a Call to a Name (line 293):
    
    # Call to zeros(...): (line 293)
    # Processing the call arguments (line 293)
    # Getting the type of 'm' (line 293)
    m_200747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 17), 'm', False)
    # Processing the call keyword arguments (line 293)
    kwargs_200748 = {}
    # Getting the type of 'np' (line 293)
    np_200745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 293)
    zeros_200746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 8), np_200745, 'zeros')
    # Calling zeros(args, kwargs) (line 293)
    zeros_call_result_200749 = invoke(stypy.reporting.localization.Localization(__file__, 293, 8), zeros_200746, *[m_200747], **kwargs_200748)
    
    # Assigning a type to the variable 'e' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'e', zeros_call_result_200749)
    
    # Getting the type of 'v' (line 315)
    v_200750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 13), 'v')
    # Testing the type of a for loop iterable (line 315)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 315, 4), v_200750)
    # Getting the type of the for loop variable (line 315)
    for_loop_var_200751 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 315, 4), v_200750)
    # Assigning a type to the variable 'i' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'i', for_loop_var_200751)
    # SSA begins for a for statement (line 315)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 316):
    
    # Assigning a Subscript to a Name (line 316):
    
    # Obtaining the type of the subscript
    slice_200752 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 316, 12), None, None, None)
    # Getting the type of 'b' (line 316)
    b_200753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 17), 'b')
    # Getting the type of 'A' (line 316)
    A_200754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'A')
    # Obtaining the member '__getitem__' of a type (line 316)
    getitem___200755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 12), A_200754, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 316)
    subscript_call_result_200756 = invoke(stypy.reporting.localization.Localization(__file__, 316, 12), getitem___200755, (slice_200752, b_200753))
    
    # Assigning a type to the variable 'B' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'B', subscript_call_result_200756)
    
    # Assigning a Num to a Subscript (line 318):
    
    # Assigning a Num to a Subscript (line 318):
    int_200757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 15), 'int')
    # Getting the type of 'e' (line 318)
    e_200758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'e')
    # Getting the type of 'i' (line 318)
    i_200759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 10), 'i')
    # Storing an element on a container (line 318)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 8), e_200758, (i_200759, int_200757))
    
    
    # Getting the type of 'i' (line 319)
    i_200760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 11), 'i')
    int_200761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 15), 'int')
    # Applying the binary operator '>' (line 319)
    result_gt_200762 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 11), '>', i_200760, int_200761)
    
    # Testing the type of an if condition (line 319)
    if_condition_200763 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 319, 8), result_gt_200762)
    # Assigning a type to the variable 'if_condition_200763' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'if_condition_200763', if_condition_200763)
    # SSA begins for if statement (line 319)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 320):
    
    # Assigning a Num to a Subscript (line 320):
    int_200764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 21), 'int')
    # Getting the type of 'e' (line 320)
    e_200765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'e')
    # Getting the type of 'i' (line 320)
    i_200766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 14), 'i')
    int_200767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 16), 'int')
    # Applying the binary operator '-' (line 320)
    result_sub_200768 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 14), '-', i_200766, int_200767)
    
    # Storing an element on a container (line 320)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 12), e_200765, (result_sub_200768, int_200764))
    # SSA join for if statement (line 319)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 322):
    
    # Assigning a Call to a Name (line 322):
    
    # Call to reshape(...): (line 322)
    # Processing the call arguments (line 322)
    int_200781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 67), 'int')
    int_200782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 71), 'int')
    # Processing the call keyword arguments (line 322)
    kwargs_200783 = {}
    
    # Call to spsolve(...): (line 322)
    # Processing the call arguments (line 322)
    
    # Call to transpose(...): (line 322)
    # Processing the call keyword arguments (line 322)
    kwargs_200775 = {}
    # Getting the type of 'B' (line 322)
    B_200773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 41), 'B', False)
    # Obtaining the member 'transpose' of a type (line 322)
    transpose_200774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 41), B_200773, 'transpose')
    # Calling transpose(args, kwargs) (line 322)
    transpose_call_result_200776 = invoke(stypy.reporting.localization.Localization(__file__, 322, 41), transpose_200774, *[], **kwargs_200775)
    
    # Getting the type of 'e' (line 322)
    e_200777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 56), 'e', False)
    # Processing the call keyword arguments (line 322)
    kwargs_200778 = {}
    # Getting the type of 'scipy' (line 322)
    scipy_200769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 13), 'scipy', False)
    # Obtaining the member 'sparse' of a type (line 322)
    sparse_200770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 13), scipy_200769, 'sparse')
    # Obtaining the member 'linalg' of a type (line 322)
    linalg_200771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 13), sparse_200770, 'linalg')
    # Obtaining the member 'spsolve' of a type (line 322)
    spsolve_200772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 13), linalg_200771, 'spsolve')
    # Calling spsolve(args, kwargs) (line 322)
    spsolve_call_result_200779 = invoke(stypy.reporting.localization.Localization(__file__, 322, 13), spsolve_200772, *[transpose_call_result_200776, e_200777], **kwargs_200778)
    
    # Obtaining the member 'reshape' of a type (line 322)
    reshape_200780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 13), spsolve_call_result_200779, 'reshape')
    # Calling reshape(args, kwargs) (line 322)
    reshape_call_result_200784 = invoke(stypy.reporting.localization.Localization(__file__, 322, 13), reshape_200780, *[int_200781, int_200782], **kwargs_200783)
    
    # Assigning a type to the variable 'pi' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'pi', reshape_call_result_200784)
    
    # Assigning a Call to a Name (line 324):
    
    # Assigning a Call to a Name (line 324):
    
    # Call to list(...): (line 324)
    # Processing the call arguments (line 324)
    # Getting the type of 'k' (line 324)
    k_200786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 18), 'k', False)
    
    # Call to set(...): (line 324)
    # Processing the call arguments (line 324)
    # Getting the type of 'b' (line 324)
    b_200788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 24), 'b', False)
    # Processing the call keyword arguments (line 324)
    kwargs_200789 = {}
    # Getting the type of 'set' (line 324)
    set_200787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 20), 'set', False)
    # Calling set(args, kwargs) (line 324)
    set_call_result_200790 = invoke(stypy.reporting.localization.Localization(__file__, 324, 20), set_200787, *[b_200788], **kwargs_200789)
    
    # Applying the binary operator '-' (line 324)
    result_sub_200791 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 18), '-', k_200786, set_call_result_200790)
    
    # Processing the call keyword arguments (line 324)
    kwargs_200792 = {}
    # Getting the type of 'list' (line 324)
    list_200785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 13), 'list', False)
    # Calling list(args, kwargs) (line 324)
    list_call_result_200793 = invoke(stypy.reporting.localization.Localization(__file__, 324, 13), list_200785, *[result_sub_200791], **kwargs_200792)
    
    # Assigning a type to the variable 'js' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'js', list_call_result_200793)
    
    # Assigning a Call to a Name (line 332):
    
    # Assigning a Call to a Name (line 332):
    
    # Call to abs(...): (line 332)
    # Processing the call arguments (line 332)
    
    # Call to dot(...): (line 332)
    # Processing the call arguments (line 332)
    # Getting the type of 'pi' (line 332)
    pi_200804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 41), 'pi', False)
    # Processing the call keyword arguments (line 332)
    kwargs_200805 = {}
    
    # Call to transpose(...): (line 332)
    # Processing the call keyword arguments (line 332)
    kwargs_200801 = {}
    
    # Obtaining the type of the subscript
    slice_200795 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 332, 16), None, None, None)
    # Getting the type of 'js' (line 332)
    js_200796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 21), 'js', False)
    # Getting the type of 'A' (line 332)
    A_200797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 16), 'A', False)
    # Obtaining the member '__getitem__' of a type (line 332)
    getitem___200798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 16), A_200797, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 332)
    subscript_call_result_200799 = invoke(stypy.reporting.localization.Localization(__file__, 332, 16), getitem___200798, (slice_200795, js_200796))
    
    # Obtaining the member 'transpose' of a type (line 332)
    transpose_200800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 16), subscript_call_result_200799, 'transpose')
    # Calling transpose(args, kwargs) (line 332)
    transpose_call_result_200802 = invoke(stypy.reporting.localization.Localization(__file__, 332, 16), transpose_200800, *[], **kwargs_200801)
    
    # Obtaining the member 'dot' of a type (line 332)
    dot_200803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 16), transpose_call_result_200802, 'dot')
    # Calling dot(args, kwargs) (line 332)
    dot_call_result_200806 = invoke(stypy.reporting.localization.Localization(__file__, 332, 16), dot_200803, *[pi_200804], **kwargs_200805)
    
    # Processing the call keyword arguments (line 332)
    kwargs_200807 = {}
    # Getting the type of 'abs' (line 332)
    abs_200794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'abs', False)
    # Calling abs(args, kwargs) (line 332)
    abs_call_result_200808 = invoke(stypy.reporting.localization.Localization(__file__, 332, 12), abs_200794, *[dot_call_result_200806], **kwargs_200807)
    
    # Assigning a type to the variable 'c' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'c', abs_call_result_200808)
    
    
    # Call to any(...): (line 333)
    # Processing the call keyword arguments (line 333)
    kwargs_200813 = {}
    
    # Getting the type of 'c' (line 333)
    c_200809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'c', False)
    # Getting the type of 'tolapiv' (line 333)
    tolapiv_200810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 16), 'tolapiv', False)
    # Applying the binary operator '>' (line 333)
    result_gt_200811 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 12), '>', c_200809, tolapiv_200810)
    
    # Obtaining the member 'any' of a type (line 333)
    any_200812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 12), result_gt_200811, 'any')
    # Calling any(args, kwargs) (line 333)
    any_call_result_200814 = invoke(stypy.reporting.localization.Localization(__file__, 333, 12), any_200812, *[], **kwargs_200813)
    
    # Testing the type of an if condition (line 333)
    if_condition_200815 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 333, 8), any_call_result_200814)
    # Assigning a type to the variable 'if_condition_200815' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'if_condition_200815', if_condition_200815)
    # SSA begins for if statement (line 333)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 334):
    
    # Assigning a Subscript to a Name (line 334):
    
    # Obtaining the type of the subscript
    
    # Call to argmax(...): (line 334)
    # Processing the call arguments (line 334)
    # Getting the type of 'c' (line 334)
    c_200818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 29), 'c', False)
    # Processing the call keyword arguments (line 334)
    kwargs_200819 = {}
    # Getting the type of 'np' (line 334)
    np_200816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 19), 'np', False)
    # Obtaining the member 'argmax' of a type (line 334)
    argmax_200817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 19), np_200816, 'argmax')
    # Calling argmax(args, kwargs) (line 334)
    argmax_call_result_200820 = invoke(stypy.reporting.localization.Localization(__file__, 334, 19), argmax_200817, *[c_200818], **kwargs_200819)
    
    # Getting the type of 'js' (line 334)
    js_200821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'js')
    # Obtaining the member '__getitem__' of a type (line 334)
    getitem___200822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 16), js_200821, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 334)
    subscript_call_result_200823 = invoke(stypy.reporting.localization.Localization(__file__, 334, 16), getitem___200822, argmax_call_result_200820)
    
    # Assigning a type to the variable 'j' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'j', subscript_call_result_200823)
    
    # Assigning a Name to a Subscript (line 335):
    
    # Assigning a Name to a Subscript (line 335):
    # Getting the type of 'j' (line 335)
    j_200824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 19), 'j')
    # Getting the type of 'b' (line 335)
    b_200825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'b')
    # Getting the type of 'i' (line 335)
    i_200826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 14), 'i')
    # Storing an element on a container (line 335)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 12), b_200825, (i_200826, j_200824))
    # SSA branch for the else part of an if statement (line 333)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 337):
    
    # Assigning a Call to a Name (line 337):
    
    # Call to dot(...): (line 337)
    # Processing the call arguments (line 337)
    
    # Call to reshape(...): (line 337)
    # Processing the call arguments (line 337)
    int_200832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 41), 'int')
    int_200833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 45), 'int')
    # Processing the call keyword arguments (line 337)
    kwargs_200834 = {}
    # Getting the type of 'rhs' (line 337)
    rhs_200830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 29), 'rhs', False)
    # Obtaining the member 'reshape' of a type (line 337)
    reshape_200831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 29), rhs_200830, 'reshape')
    # Calling reshape(args, kwargs) (line 337)
    reshape_call_result_200835 = invoke(stypy.reporting.localization.Localization(__file__, 337, 29), reshape_200831, *[int_200832, int_200833], **kwargs_200834)
    
    # Processing the call keyword arguments (line 337)
    kwargs_200836 = {}
    # Getting the type of 'pi' (line 337)
    pi_200827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 20), 'pi', False)
    # Obtaining the member 'T' of a type (line 337)
    T_200828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 20), pi_200827, 'T')
    # Obtaining the member 'dot' of a type (line 337)
    dot_200829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 20), T_200828, 'dot')
    # Calling dot(args, kwargs) (line 337)
    dot_call_result_200837 = invoke(stypy.reporting.localization.Localization(__file__, 337, 20), dot_200829, *[reshape_call_result_200835], **kwargs_200836)
    
    # Assigning a type to the variable 'bibar' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'bibar', dot_call_result_200837)
    
    # Assigning a Call to a Name (line 338):
    
    # Assigning a Call to a Name (line 338):
    
    # Call to norm(...): (line 338)
    # Processing the call arguments (line 338)
    # Getting the type of 'rhs' (line 338)
    rhs_200841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 35), 'rhs', False)
    # Processing the call keyword arguments (line 338)
    kwargs_200842 = {}
    # Getting the type of 'np' (line 338)
    np_200838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 20), 'np', False)
    # Obtaining the member 'linalg' of a type (line 338)
    linalg_200839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 20), np_200838, 'linalg')
    # Obtaining the member 'norm' of a type (line 338)
    norm_200840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 20), linalg_200839, 'norm')
    # Calling norm(args, kwargs) (line 338)
    norm_call_result_200843 = invoke(stypy.reporting.localization.Localization(__file__, 338, 20), norm_200840, *[rhs_200841], **kwargs_200842)
    
    # Assigning a type to the variable 'bnorm' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'bnorm', norm_call_result_200843)
    
    
    
    # Call to abs(...): (line 339)
    # Processing the call arguments (line 339)
    # Getting the type of 'bibar' (line 339)
    bibar_200845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 19), 'bibar', False)
    # Processing the call keyword arguments (line 339)
    kwargs_200846 = {}
    # Getting the type of 'abs' (line 339)
    abs_200844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 15), 'abs', False)
    # Calling abs(args, kwargs) (line 339)
    abs_call_result_200847 = invoke(stypy.reporting.localization.Localization(__file__, 339, 15), abs_200844, *[bibar_200845], **kwargs_200846)
    
    int_200848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 27), 'int')
    # Getting the type of 'bnorm' (line 339)
    bnorm_200849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 31), 'bnorm')
    # Applying the binary operator '+' (line 339)
    result_add_200850 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 27), '+', int_200848, bnorm_200849)
    
    # Applying the binary operator 'div' (line 339)
    result_div_200851 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 15), 'div', abs_call_result_200847, result_add_200850)
    
    # Getting the type of 'tolprimal' (line 339)
    tolprimal_200852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 40), 'tolprimal')
    # Applying the binary operator '>' (line 339)
    result_gt_200853 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 15), '>', result_div_200851, tolprimal_200852)
    
    # Testing the type of an if condition (line 339)
    if_condition_200854 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 339, 12), result_gt_200853)
    # Assigning a type to the variable 'if_condition_200854' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'if_condition_200854', if_condition_200854)
    # SSA begins for if statement (line 339)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 340):
    
    # Assigning a Num to a Name (line 340):
    int_200855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 25), 'int')
    # Assigning a type to the variable 'status' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 16), 'status', int_200855)
    
    # Assigning a Name to a Name (line 341):
    
    # Assigning a Name to a Name (line 341):
    # Getting the type of 'inconsistent' (line 341)
    inconsistent_200856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 26), 'inconsistent')
    # Assigning a type to the variable 'message' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 16), 'message', inconsistent_200856)
    
    # Obtaining an instance of the builtin type 'tuple' (line 342)
    tuple_200857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 342)
    # Adding element type (line 342)
    # Getting the type of 'A_orig' (line 342)
    A_orig_200858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 23), 'A_orig')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 23), tuple_200857, A_orig_200858)
    # Adding element type (line 342)
    # Getting the type of 'rhs' (line 342)
    rhs_200859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 31), 'rhs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 23), tuple_200857, rhs_200859)
    # Adding element type (line 342)
    # Getting the type of 'status' (line 342)
    status_200860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 36), 'status')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 23), tuple_200857, status_200860)
    # Adding element type (line 342)
    # Getting the type of 'message' (line 342)
    message_200861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 44), 'message')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 23), tuple_200857, message_200861)
    
    # Assigning a type to the variable 'stypy_return_type' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 16), 'stypy_return_type', tuple_200857)
    # SSA branch for the else part of an if statement (line 339)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 344)
    # Processing the call arguments (line 344)
    # Getting the type of 'i' (line 344)
    i_200864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 25), 'i', False)
    # Processing the call keyword arguments (line 344)
    kwargs_200865 = {}
    # Getting the type of 'd' (line 344)
    d_200862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 16), 'd', False)
    # Obtaining the member 'append' of a type (line 344)
    append_200863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 16), d_200862, 'append')
    # Calling append(args, kwargs) (line 344)
    append_call_result_200866 = invoke(stypy.reporting.localization.Localization(__file__, 344, 16), append_200863, *[i_200864], **kwargs_200865)
    
    # SSA join for if statement (line 339)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 333)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 346):
    
    # Assigning a Call to a Name (line 346):
    
    # Call to set(...): (line 346)
    # Processing the call arguments (line 346)
    
    # Call to range(...): (line 346)
    # Processing the call arguments (line 346)
    # Getting the type of 'm' (line 346)
    m_200869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 21), 'm', False)
    # Processing the call keyword arguments (line 346)
    kwargs_200870 = {}
    # Getting the type of 'range' (line 346)
    range_200868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 15), 'range', False)
    # Calling range(args, kwargs) (line 346)
    range_call_result_200871 = invoke(stypy.reporting.localization.Localization(__file__, 346, 15), range_200868, *[m_200869], **kwargs_200870)
    
    # Processing the call keyword arguments (line 346)
    kwargs_200872 = {}
    # Getting the type of 'set' (line 346)
    set_200867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 11), 'set', False)
    # Calling set(args, kwargs) (line 346)
    set_call_result_200873 = invoke(stypy.reporting.localization.Localization(__file__, 346, 11), set_200867, *[range_call_result_200871], **kwargs_200872)
    
    # Assigning a type to the variable 'keep' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'keep', set_call_result_200873)
    
    # Assigning a Call to a Name (line 347):
    
    # Assigning a Call to a Name (line 347):
    
    # Call to list(...): (line 347)
    # Processing the call arguments (line 347)
    # Getting the type of 'keep' (line 347)
    keep_200875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 16), 'keep', False)
    
    # Call to set(...): (line 347)
    # Processing the call arguments (line 347)
    # Getting the type of 'd' (line 347)
    d_200877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 27), 'd', False)
    # Processing the call keyword arguments (line 347)
    kwargs_200878 = {}
    # Getting the type of 'set' (line 347)
    set_200876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 23), 'set', False)
    # Calling set(args, kwargs) (line 347)
    set_call_result_200879 = invoke(stypy.reporting.localization.Localization(__file__, 347, 23), set_200876, *[d_200877], **kwargs_200878)
    
    # Applying the binary operator '-' (line 347)
    result_sub_200880 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 16), '-', keep_200875, set_call_result_200879)
    
    # Processing the call keyword arguments (line 347)
    kwargs_200881 = {}
    # Getting the type of 'list' (line 347)
    list_200874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 11), 'list', False)
    # Calling list(args, kwargs) (line 347)
    list_call_result_200882 = invoke(stypy.reporting.localization.Localization(__file__, 347, 11), list_200874, *[result_sub_200880], **kwargs_200881)
    
    # Assigning a type to the variable 'keep' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'keep', list_call_result_200882)
    
    # Obtaining an instance of the builtin type 'tuple' (line 348)
    tuple_200883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 348)
    # Adding element type (line 348)
    
    # Obtaining the type of the subscript
    # Getting the type of 'keep' (line 348)
    keep_200884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 18), 'keep')
    slice_200885 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 348, 11), None, None, None)
    # Getting the type of 'A_orig' (line 348)
    A_orig_200886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 11), 'A_orig')
    # Obtaining the member '__getitem__' of a type (line 348)
    getitem___200887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 11), A_orig_200886, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 348)
    subscript_call_result_200888 = invoke(stypy.reporting.localization.Localization(__file__, 348, 11), getitem___200887, (keep_200884, slice_200885))
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 11), tuple_200883, subscript_call_result_200888)
    # Adding element type (line 348)
    
    # Obtaining the type of the subscript
    # Getting the type of 'keep' (line 348)
    keep_200889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 32), 'keep')
    # Getting the type of 'rhs' (line 348)
    rhs_200890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 28), 'rhs')
    # Obtaining the member '__getitem__' of a type (line 348)
    getitem___200891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 28), rhs_200890, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 348)
    subscript_call_result_200892 = invoke(stypy.reporting.localization.Localization(__file__, 348, 28), getitem___200891, keep_200889)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 11), tuple_200883, subscript_call_result_200892)
    # Adding element type (line 348)
    # Getting the type of 'status' (line 348)
    status_200893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 39), 'status')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 11), tuple_200883, status_200893)
    # Adding element type (line 348)
    # Getting the type of 'message' (line 348)
    message_200894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 47), 'message')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 11), tuple_200883, message_200894)
    
    # Assigning a type to the variable 'stypy_return_type' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'stypy_return_type', tuple_200883)
    
    # ################# End of '_remove_redundancy_sparse(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_remove_redundancy_sparse' in the type store
    # Getting the type of 'stypy_return_type' (line 235)
    stypy_return_type_200895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_200895)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_remove_redundancy_sparse'
    return stypy_return_type_200895

# Assigning a type to the variable '_remove_redundancy_sparse' (line 235)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 0), '_remove_redundancy_sparse', _remove_redundancy_sparse)

@norecursion
def _remove_redundancy(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_remove_redundancy'
    module_type_store = module_type_store.open_function_context('_remove_redundancy', 351, 0, False)
    
    # Passed parameters checking function
    _remove_redundancy.stypy_localization = localization
    _remove_redundancy.stypy_type_of_self = None
    _remove_redundancy.stypy_type_store = module_type_store
    _remove_redundancy.stypy_function_name = '_remove_redundancy'
    _remove_redundancy.stypy_param_names_list = ['A', 'b']
    _remove_redundancy.stypy_varargs_param_name = None
    _remove_redundancy.stypy_kwargs_param_name = None
    _remove_redundancy.stypy_call_defaults = defaults
    _remove_redundancy.stypy_call_varargs = varargs
    _remove_redundancy.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_remove_redundancy', ['A', 'b'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_remove_redundancy', localization, ['A', 'b'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_remove_redundancy(...)' code ##################

    str_200896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, (-1)), 'str', '\n    Eliminates redundant equations from system of equations defined by Ax = b\n    and identifies infeasibilities.\n\n    Parameters\n    ----------\n    A : 2-D array\n        An array representing the left-hand side of a system of equations\n    b : 1-D array\n        An array representing the right-hand side of a system of equations\n\n    Returns\n    -------\n    A : 2-D array\n        An array representing the left-hand side of a system of equations\n    b : 1-D array\n        An array representing the right-hand side of a system of equations\n    status: int\n        An integer indicating the status of the system\n        0: No infeasibility identified\n        2: Trivially infeasible\n    message : str\n        A string descriptor of the exit status of the optimization.\n\n    References\n    ----------\n    .. [2] Andersen, Erling D. "Finding all linearly dependent rows in\n           large-scale linear programming." Optimization Methods and Software\n           6.3 (1995): 219-227.\n\n    ')
    
    # Assigning a Call to a Tuple (line 384):
    
    # Assigning a Subscript to a Name (line 384):
    
    # Obtaining the type of the subscript
    int_200897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 4), 'int')
    
    # Call to _remove_zero_rows(...): (line 384)
    # Processing the call arguments (line 384)
    # Getting the type of 'A' (line 384)
    A_200899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 46), 'A', False)
    # Getting the type of 'b' (line 384)
    b_200900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 49), 'b', False)
    # Processing the call keyword arguments (line 384)
    kwargs_200901 = {}
    # Getting the type of '_remove_zero_rows' (line 384)
    _remove_zero_rows_200898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 28), '_remove_zero_rows', False)
    # Calling _remove_zero_rows(args, kwargs) (line 384)
    _remove_zero_rows_call_result_200902 = invoke(stypy.reporting.localization.Localization(__file__, 384, 28), _remove_zero_rows_200898, *[A_200899, b_200900], **kwargs_200901)
    
    # Obtaining the member '__getitem__' of a type (line 384)
    getitem___200903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 4), _remove_zero_rows_call_result_200902, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 384)
    subscript_call_result_200904 = invoke(stypy.reporting.localization.Localization(__file__, 384, 4), getitem___200903, int_200897)
    
    # Assigning a type to the variable 'tuple_var_assignment_200119' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'tuple_var_assignment_200119', subscript_call_result_200904)
    
    # Assigning a Subscript to a Name (line 384):
    
    # Obtaining the type of the subscript
    int_200905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 4), 'int')
    
    # Call to _remove_zero_rows(...): (line 384)
    # Processing the call arguments (line 384)
    # Getting the type of 'A' (line 384)
    A_200907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 46), 'A', False)
    # Getting the type of 'b' (line 384)
    b_200908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 49), 'b', False)
    # Processing the call keyword arguments (line 384)
    kwargs_200909 = {}
    # Getting the type of '_remove_zero_rows' (line 384)
    _remove_zero_rows_200906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 28), '_remove_zero_rows', False)
    # Calling _remove_zero_rows(args, kwargs) (line 384)
    _remove_zero_rows_call_result_200910 = invoke(stypy.reporting.localization.Localization(__file__, 384, 28), _remove_zero_rows_200906, *[A_200907, b_200908], **kwargs_200909)
    
    # Obtaining the member '__getitem__' of a type (line 384)
    getitem___200911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 4), _remove_zero_rows_call_result_200910, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 384)
    subscript_call_result_200912 = invoke(stypy.reporting.localization.Localization(__file__, 384, 4), getitem___200911, int_200905)
    
    # Assigning a type to the variable 'tuple_var_assignment_200120' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'tuple_var_assignment_200120', subscript_call_result_200912)
    
    # Assigning a Subscript to a Name (line 384):
    
    # Obtaining the type of the subscript
    int_200913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 4), 'int')
    
    # Call to _remove_zero_rows(...): (line 384)
    # Processing the call arguments (line 384)
    # Getting the type of 'A' (line 384)
    A_200915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 46), 'A', False)
    # Getting the type of 'b' (line 384)
    b_200916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 49), 'b', False)
    # Processing the call keyword arguments (line 384)
    kwargs_200917 = {}
    # Getting the type of '_remove_zero_rows' (line 384)
    _remove_zero_rows_200914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 28), '_remove_zero_rows', False)
    # Calling _remove_zero_rows(args, kwargs) (line 384)
    _remove_zero_rows_call_result_200918 = invoke(stypy.reporting.localization.Localization(__file__, 384, 28), _remove_zero_rows_200914, *[A_200915, b_200916], **kwargs_200917)
    
    # Obtaining the member '__getitem__' of a type (line 384)
    getitem___200919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 4), _remove_zero_rows_call_result_200918, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 384)
    subscript_call_result_200920 = invoke(stypy.reporting.localization.Localization(__file__, 384, 4), getitem___200919, int_200913)
    
    # Assigning a type to the variable 'tuple_var_assignment_200121' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'tuple_var_assignment_200121', subscript_call_result_200920)
    
    # Assigning a Subscript to a Name (line 384):
    
    # Obtaining the type of the subscript
    int_200921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 4), 'int')
    
    # Call to _remove_zero_rows(...): (line 384)
    # Processing the call arguments (line 384)
    # Getting the type of 'A' (line 384)
    A_200923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 46), 'A', False)
    # Getting the type of 'b' (line 384)
    b_200924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 49), 'b', False)
    # Processing the call keyword arguments (line 384)
    kwargs_200925 = {}
    # Getting the type of '_remove_zero_rows' (line 384)
    _remove_zero_rows_200922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 28), '_remove_zero_rows', False)
    # Calling _remove_zero_rows(args, kwargs) (line 384)
    _remove_zero_rows_call_result_200926 = invoke(stypy.reporting.localization.Localization(__file__, 384, 28), _remove_zero_rows_200922, *[A_200923, b_200924], **kwargs_200925)
    
    # Obtaining the member '__getitem__' of a type (line 384)
    getitem___200927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 4), _remove_zero_rows_call_result_200926, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 384)
    subscript_call_result_200928 = invoke(stypy.reporting.localization.Localization(__file__, 384, 4), getitem___200927, int_200921)
    
    # Assigning a type to the variable 'tuple_var_assignment_200122' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'tuple_var_assignment_200122', subscript_call_result_200928)
    
    # Assigning a Name to a Name (line 384):
    # Getting the type of 'tuple_var_assignment_200119' (line 384)
    tuple_var_assignment_200119_200929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'tuple_var_assignment_200119')
    # Assigning a type to the variable 'A' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'A', tuple_var_assignment_200119_200929)
    
    # Assigning a Name to a Name (line 384):
    # Getting the type of 'tuple_var_assignment_200120' (line 384)
    tuple_var_assignment_200120_200930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'tuple_var_assignment_200120')
    # Assigning a type to the variable 'b' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 7), 'b', tuple_var_assignment_200120_200930)
    
    # Assigning a Name to a Name (line 384):
    # Getting the type of 'tuple_var_assignment_200121' (line 384)
    tuple_var_assignment_200121_200931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'tuple_var_assignment_200121')
    # Assigning a type to the variable 'status' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 10), 'status', tuple_var_assignment_200121_200931)
    
    # Assigning a Name to a Name (line 384):
    # Getting the type of 'tuple_var_assignment_200122' (line 384)
    tuple_var_assignment_200122_200932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'tuple_var_assignment_200122')
    # Assigning a type to the variable 'message' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 18), 'message', tuple_var_assignment_200122_200932)
    
    
    # Getting the type of 'status' (line 386)
    status_200933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 7), 'status')
    int_200934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 17), 'int')
    # Applying the binary operator '!=' (line 386)
    result_ne_200935 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 7), '!=', status_200933, int_200934)
    
    # Testing the type of an if condition (line 386)
    if_condition_200936 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 386, 4), result_ne_200935)
    # Assigning a type to the variable 'if_condition_200936' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'if_condition_200936', if_condition_200936)
    # SSA begins for if statement (line 386)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 387)
    tuple_200937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 387)
    # Adding element type (line 387)
    # Getting the type of 'A' (line 387)
    A_200938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 15), 'A')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 15), tuple_200937, A_200938)
    # Adding element type (line 387)
    # Getting the type of 'b' (line 387)
    b_200939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 18), 'b')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 15), tuple_200937, b_200939)
    # Adding element type (line 387)
    # Getting the type of 'status' (line 387)
    status_200940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 21), 'status')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 15), tuple_200937, status_200940)
    # Adding element type (line 387)
    # Getting the type of 'message' (line 387)
    message_200941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 29), 'message')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 15), tuple_200937, message_200941)
    
    # Assigning a type to the variable 'stypy_return_type' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'stypy_return_type', tuple_200937)
    # SSA join for if statement (line 386)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 389):
    
    # Assigning a Subscript to a Name (line 389):
    
    # Obtaining the type of the subscript
    int_200942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 4), 'int')
    
    # Call to svd(...): (line 389)
    # Processing the call arguments (line 389)
    # Getting the type of 'A' (line 389)
    A_200944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 19), 'A', False)
    # Processing the call keyword arguments (line 389)
    kwargs_200945 = {}
    # Getting the type of 'svd' (line 389)
    svd_200943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 15), 'svd', False)
    # Calling svd(args, kwargs) (line 389)
    svd_call_result_200946 = invoke(stypy.reporting.localization.Localization(__file__, 389, 15), svd_200943, *[A_200944], **kwargs_200945)
    
    # Obtaining the member '__getitem__' of a type (line 389)
    getitem___200947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 4), svd_call_result_200946, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 389)
    subscript_call_result_200948 = invoke(stypy.reporting.localization.Localization(__file__, 389, 4), getitem___200947, int_200942)
    
    # Assigning a type to the variable 'tuple_var_assignment_200123' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'tuple_var_assignment_200123', subscript_call_result_200948)
    
    # Assigning a Subscript to a Name (line 389):
    
    # Obtaining the type of the subscript
    int_200949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 4), 'int')
    
    # Call to svd(...): (line 389)
    # Processing the call arguments (line 389)
    # Getting the type of 'A' (line 389)
    A_200951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 19), 'A', False)
    # Processing the call keyword arguments (line 389)
    kwargs_200952 = {}
    # Getting the type of 'svd' (line 389)
    svd_200950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 15), 'svd', False)
    # Calling svd(args, kwargs) (line 389)
    svd_call_result_200953 = invoke(stypy.reporting.localization.Localization(__file__, 389, 15), svd_200950, *[A_200951], **kwargs_200952)
    
    # Obtaining the member '__getitem__' of a type (line 389)
    getitem___200954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 4), svd_call_result_200953, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 389)
    subscript_call_result_200955 = invoke(stypy.reporting.localization.Localization(__file__, 389, 4), getitem___200954, int_200949)
    
    # Assigning a type to the variable 'tuple_var_assignment_200124' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'tuple_var_assignment_200124', subscript_call_result_200955)
    
    # Assigning a Subscript to a Name (line 389):
    
    # Obtaining the type of the subscript
    int_200956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 4), 'int')
    
    # Call to svd(...): (line 389)
    # Processing the call arguments (line 389)
    # Getting the type of 'A' (line 389)
    A_200958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 19), 'A', False)
    # Processing the call keyword arguments (line 389)
    kwargs_200959 = {}
    # Getting the type of 'svd' (line 389)
    svd_200957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 15), 'svd', False)
    # Calling svd(args, kwargs) (line 389)
    svd_call_result_200960 = invoke(stypy.reporting.localization.Localization(__file__, 389, 15), svd_200957, *[A_200958], **kwargs_200959)
    
    # Obtaining the member '__getitem__' of a type (line 389)
    getitem___200961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 4), svd_call_result_200960, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 389)
    subscript_call_result_200962 = invoke(stypy.reporting.localization.Localization(__file__, 389, 4), getitem___200961, int_200956)
    
    # Assigning a type to the variable 'tuple_var_assignment_200125' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'tuple_var_assignment_200125', subscript_call_result_200962)
    
    # Assigning a Name to a Name (line 389):
    # Getting the type of 'tuple_var_assignment_200123' (line 389)
    tuple_var_assignment_200123_200963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'tuple_var_assignment_200123')
    # Assigning a type to the variable 'U' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'U', tuple_var_assignment_200123_200963)
    
    # Assigning a Name to a Name (line 389):
    # Getting the type of 'tuple_var_assignment_200124' (line 389)
    tuple_var_assignment_200124_200964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'tuple_var_assignment_200124')
    # Assigning a type to the variable 's' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 7), 's', tuple_var_assignment_200124_200964)
    
    # Assigning a Name to a Name (line 389):
    # Getting the type of 'tuple_var_assignment_200125' (line 389)
    tuple_var_assignment_200125_200965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'tuple_var_assignment_200125')
    # Assigning a type to the variable 'Vh' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 10), 'Vh', tuple_var_assignment_200125_200965)
    
    # Assigning a Attribute to a Name (line 390):
    
    # Assigning a Attribute to a Name (line 390):
    
    # Call to finfo(...): (line 390)
    # Processing the call arguments (line 390)
    # Getting the type of 'float' (line 390)
    float_200968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 19), 'float', False)
    # Processing the call keyword arguments (line 390)
    kwargs_200969 = {}
    # Getting the type of 'np' (line 390)
    np_200966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 10), 'np', False)
    # Obtaining the member 'finfo' of a type (line 390)
    finfo_200967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 10), np_200966, 'finfo')
    # Calling finfo(args, kwargs) (line 390)
    finfo_call_result_200970 = invoke(stypy.reporting.localization.Localization(__file__, 390, 10), finfo_200967, *[float_200968], **kwargs_200969)
    
    # Obtaining the member 'eps' of a type (line 390)
    eps_200971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 10), finfo_call_result_200970, 'eps')
    # Assigning a type to the variable 'eps' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'eps', eps_200971)
    
    # Assigning a BinOp to a Name (line 391):
    
    # Assigning a BinOp to a Name (line 391):
    
    # Call to max(...): (line 391)
    # Processing the call keyword arguments (line 391)
    kwargs_200974 = {}
    # Getting the type of 's' (line 391)
    s_200972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 10), 's', False)
    # Obtaining the member 'max' of a type (line 391)
    max_200973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 10), s_200972, 'max')
    # Calling max(args, kwargs) (line 391)
    max_call_result_200975 = invoke(stypy.reporting.localization.Localization(__file__, 391, 10), max_200973, *[], **kwargs_200974)
    
    
    # Call to max(...): (line 391)
    # Processing the call arguments (line 391)
    # Getting the type of 'A' (line 391)
    A_200977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 24), 'A', False)
    # Obtaining the member 'shape' of a type (line 391)
    shape_200978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 24), A_200977, 'shape')
    # Processing the call keyword arguments (line 391)
    kwargs_200979 = {}
    # Getting the type of 'max' (line 391)
    max_200976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 20), 'max', False)
    # Calling max(args, kwargs) (line 391)
    max_call_result_200980 = invoke(stypy.reporting.localization.Localization(__file__, 391, 20), max_200976, *[shape_200978], **kwargs_200979)
    
    # Applying the binary operator '*' (line 391)
    result_mul_200981 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 10), '*', max_call_result_200975, max_call_result_200980)
    
    # Getting the type of 'eps' (line 391)
    eps_200982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 35), 'eps')
    # Applying the binary operator '*' (line 391)
    result_mul_200983 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 33), '*', result_mul_200981, eps_200982)
    
    # Assigning a type to the variable 'tol' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'tol', result_mul_200983)
    
    # Assigning a Attribute to a Tuple (line 393):
    
    # Assigning a Subscript to a Name (line 393):
    
    # Obtaining the type of the subscript
    int_200984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 4), 'int')
    # Getting the type of 'A' (line 393)
    A_200985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 11), 'A')
    # Obtaining the member 'shape' of a type (line 393)
    shape_200986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 11), A_200985, 'shape')
    # Obtaining the member '__getitem__' of a type (line 393)
    getitem___200987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 4), shape_200986, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 393)
    subscript_call_result_200988 = invoke(stypy.reporting.localization.Localization(__file__, 393, 4), getitem___200987, int_200984)
    
    # Assigning a type to the variable 'tuple_var_assignment_200126' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'tuple_var_assignment_200126', subscript_call_result_200988)
    
    # Assigning a Subscript to a Name (line 393):
    
    # Obtaining the type of the subscript
    int_200989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 4), 'int')
    # Getting the type of 'A' (line 393)
    A_200990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 11), 'A')
    # Obtaining the member 'shape' of a type (line 393)
    shape_200991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 11), A_200990, 'shape')
    # Obtaining the member '__getitem__' of a type (line 393)
    getitem___200992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 4), shape_200991, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 393)
    subscript_call_result_200993 = invoke(stypy.reporting.localization.Localization(__file__, 393, 4), getitem___200992, int_200989)
    
    # Assigning a type to the variable 'tuple_var_assignment_200127' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'tuple_var_assignment_200127', subscript_call_result_200993)
    
    # Assigning a Name to a Name (line 393):
    # Getting the type of 'tuple_var_assignment_200126' (line 393)
    tuple_var_assignment_200126_200994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'tuple_var_assignment_200126')
    # Assigning a type to the variable 'm' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'm', tuple_var_assignment_200126_200994)
    
    # Assigning a Name to a Name (line 393):
    # Getting the type of 'tuple_var_assignment_200127' (line 393)
    tuple_var_assignment_200127_200995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'tuple_var_assignment_200127')
    # Assigning a type to the variable 'n' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 7), 'n', tuple_var_assignment_200127_200995)
    
    # Assigning a IfExp to a Name (line 394):
    
    # Assigning a IfExp to a Name (line 394):
    
    
    # Getting the type of 'm' (line 394)
    m_200996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 21), 'm')
    # Getting the type of 'n' (line 394)
    n_200997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 26), 'n')
    # Applying the binary operator '<=' (line 394)
    result_le_200998 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 21), '<=', m_200996, n_200997)
    
    # Testing the type of an if expression (line 394)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 394, 12), result_le_200998)
    # SSA begins for if expression (line 394)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Obtaining the type of the subscript
    int_200999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 14), 'int')
    # Getting the type of 's' (line 394)
    s_201000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 's')
    # Obtaining the member '__getitem__' of a type (line 394)
    getitem___201001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 12), s_201000, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 394)
    subscript_call_result_201002 = invoke(stypy.reporting.localization.Localization(__file__, 394, 12), getitem___201001, int_200999)
    
    # SSA branch for the else part of an if expression (line 394)
    module_type_store.open_ssa_branch('if expression else')
    int_201003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 33), 'int')
    # SSA join for if expression (line 394)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_201004 = union_type.UnionType.add(subscript_call_result_201002, int_201003)
    
    # Assigning a type to the variable 's_min' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 's_min', if_exp_201004)
    
    
    
    # Call to abs(...): (line 411)
    # Processing the call arguments (line 411)
    # Getting the type of 's_min' (line 411)
    s_min_201006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 14), 's_min', False)
    # Processing the call keyword arguments (line 411)
    kwargs_201007 = {}
    # Getting the type of 'abs' (line 411)
    abs_201005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 10), 'abs', False)
    # Calling abs(args, kwargs) (line 411)
    abs_call_result_201008 = invoke(stypy.reporting.localization.Localization(__file__, 411, 10), abs_201005, *[s_min_201006], **kwargs_201007)
    
    # Getting the type of 'tol' (line 411)
    tol_201009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 23), 'tol')
    # Applying the binary operator '<' (line 411)
    result_lt_201010 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 10), '<', abs_call_result_201008, tol_201009)
    
    # Testing the type of an if condition (line 411)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 411, 4), result_lt_201010)
    # SSA begins for while statement (line 411)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Subscript to a Name (line 412):
    
    # Assigning a Subscript to a Name (line 412):
    
    # Obtaining the type of the subscript
    slice_201011 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 412, 12), None, None, None)
    int_201012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 17), 'int')
    # Getting the type of 'U' (line 412)
    U_201013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 12), 'U')
    # Obtaining the member '__getitem__' of a type (line 412)
    getitem___201014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 12), U_201013, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 412)
    subscript_call_result_201015 = invoke(stypy.reporting.localization.Localization(__file__, 412, 12), getitem___201014, (slice_201011, int_201012))
    
    # Assigning a type to the variable 'v' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'v', subscript_call_result_201015)
    
    # Assigning a Compare to a Name (line 414):
    
    # Assigning a Compare to a Name (line 414):
    
    
    # Call to abs(...): (line 414)
    # Processing the call arguments (line 414)
    # Getting the type of 'v' (line 414)
    v_201018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 30), 'v', False)
    # Processing the call keyword arguments (line 414)
    kwargs_201019 = {}
    # Getting the type of 'np' (line 414)
    np_201016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 23), 'np', False)
    # Obtaining the member 'abs' of a type (line 414)
    abs_201017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 23), np_201016, 'abs')
    # Calling abs(args, kwargs) (line 414)
    abs_call_result_201020 = invoke(stypy.reporting.localization.Localization(__file__, 414, 23), abs_201017, *[v_201018], **kwargs_201019)
    
    # Getting the type of 'tol' (line 414)
    tol_201021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 35), 'tol')
    float_201022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 41), 'float')
    # Applying the binary operator '*' (line 414)
    result_mul_201023 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 35), '*', tol_201021, float_201022)
    
    # Applying the binary operator '>' (line 414)
    result_gt_201024 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 23), '>', abs_call_result_201020, result_mul_201023)
    
    # Assigning a type to the variable 'eligibleRows' (line 414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'eligibleRows', result_gt_201024)
    
    
    # Evaluating a boolean operation
    
    
    # Call to any(...): (line 415)
    # Processing the call arguments (line 415)
    # Getting the type of 'eligibleRows' (line 415)
    eligibleRows_201027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 22), 'eligibleRows', False)
    # Processing the call keyword arguments (line 415)
    kwargs_201028 = {}
    # Getting the type of 'np' (line 415)
    np_201025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 15), 'np', False)
    # Obtaining the member 'any' of a type (line 415)
    any_201026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 15), np_201025, 'any')
    # Calling any(args, kwargs) (line 415)
    any_call_result_201029 = invoke(stypy.reporting.localization.Localization(__file__, 415, 15), any_201026, *[eligibleRows_201027], **kwargs_201028)
    
    # Applying the 'not' unary operator (line 415)
    result_not__201030 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 11), 'not', any_call_result_201029)
    
    
    # Call to any(...): (line 415)
    # Processing the call arguments (line 415)
    
    
    # Call to abs(...): (line 415)
    # Processing the call arguments (line 415)
    
    # Call to dot(...): (line 415)
    # Processing the call arguments (line 415)
    # Getting the type of 'A' (line 415)
    A_201037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 59), 'A', False)
    # Processing the call keyword arguments (line 415)
    kwargs_201038 = {}
    # Getting the type of 'v' (line 415)
    v_201035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 53), 'v', False)
    # Obtaining the member 'dot' of a type (line 415)
    dot_201036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 53), v_201035, 'dot')
    # Calling dot(args, kwargs) (line 415)
    dot_call_result_201039 = invoke(stypy.reporting.localization.Localization(__file__, 415, 53), dot_201036, *[A_201037], **kwargs_201038)
    
    # Processing the call keyword arguments (line 415)
    kwargs_201040 = {}
    # Getting the type of 'np' (line 415)
    np_201033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 46), 'np', False)
    # Obtaining the member 'abs' of a type (line 415)
    abs_201034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 46), np_201033, 'abs')
    # Calling abs(args, kwargs) (line 415)
    abs_call_result_201041 = invoke(stypy.reporting.localization.Localization(__file__, 415, 46), abs_201034, *[dot_call_result_201039], **kwargs_201040)
    
    # Getting the type of 'tol' (line 415)
    tol_201042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 65), 'tol', False)
    # Applying the binary operator '>' (line 415)
    result_gt_201043 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 46), '>', abs_call_result_201041, tol_201042)
    
    # Processing the call keyword arguments (line 415)
    kwargs_201044 = {}
    # Getting the type of 'np' (line 415)
    np_201031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 39), 'np', False)
    # Obtaining the member 'any' of a type (line 415)
    any_201032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 39), np_201031, 'any')
    # Calling any(args, kwargs) (line 415)
    any_call_result_201045 = invoke(stypy.reporting.localization.Localization(__file__, 415, 39), any_201032, *[result_gt_201043], **kwargs_201044)
    
    # Applying the binary operator 'or' (line 415)
    result_or_keyword_201046 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 11), 'or', result_not__201030, any_call_result_201045)
    
    # Testing the type of an if condition (line 415)
    if_condition_201047 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 415, 8), result_or_keyword_201046)
    # Assigning a type to the variable 'if_condition_201047' (line 415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'if_condition_201047', if_condition_201047)
    # SSA begins for if statement (line 415)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 416):
    
    # Assigning a Num to a Name (line 416):
    int_201048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 21), 'int')
    # Assigning a type to the variable 'status' (line 416)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 12), 'status', int_201048)
    
    # Assigning a Str to a Name (line 417):
    
    # Assigning a Str to a Name (line 417):
    str_201049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 23), 'str', 'Due to numerical issues, redundant equality constraints could not be removed automatically. Try providing your constraint matrices as sparse matrices to activate sparse presolve, try turning off redundancy removal, or try turning off presolve altogether.')
    # Assigning a type to the variable 'message' (line 417)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'message', str_201049)
    # SSA join for if statement (line 415)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to any(...): (line 424)
    # Processing the call arguments (line 424)
    
    
    # Call to abs(...): (line 424)
    # Processing the call arguments (line 424)
    
    # Call to dot(...): (line 424)
    # Processing the call arguments (line 424)
    # Getting the type of 'b' (line 424)
    b_201056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 31), 'b', False)
    # Processing the call keyword arguments (line 424)
    kwargs_201057 = {}
    # Getting the type of 'v' (line 424)
    v_201054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 25), 'v', False)
    # Obtaining the member 'dot' of a type (line 424)
    dot_201055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 25), v_201054, 'dot')
    # Calling dot(args, kwargs) (line 424)
    dot_call_result_201058 = invoke(stypy.reporting.localization.Localization(__file__, 424, 25), dot_201055, *[b_201056], **kwargs_201057)
    
    # Processing the call keyword arguments (line 424)
    kwargs_201059 = {}
    # Getting the type of 'np' (line 424)
    np_201052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 18), 'np', False)
    # Obtaining the member 'abs' of a type (line 424)
    abs_201053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 18), np_201052, 'abs')
    # Calling abs(args, kwargs) (line 424)
    abs_call_result_201060 = invoke(stypy.reporting.localization.Localization(__file__, 424, 18), abs_201053, *[dot_call_result_201058], **kwargs_201059)
    
    # Getting the type of 'tol' (line 424)
    tol_201061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 37), 'tol', False)
    # Applying the binary operator '>' (line 424)
    result_gt_201062 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 18), '>', abs_call_result_201060, tol_201061)
    
    # Processing the call keyword arguments (line 424)
    kwargs_201063 = {}
    # Getting the type of 'np' (line 424)
    np_201050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 11), 'np', False)
    # Obtaining the member 'any' of a type (line 424)
    any_201051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 11), np_201050, 'any')
    # Calling any(args, kwargs) (line 424)
    any_call_result_201064 = invoke(stypy.reporting.localization.Localization(__file__, 424, 11), any_201051, *[result_gt_201062], **kwargs_201063)
    
    # Testing the type of an if condition (line 424)
    if_condition_201065 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 424, 8), any_call_result_201064)
    # Assigning a type to the variable 'if_condition_201065' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'if_condition_201065', if_condition_201065)
    # SSA begins for if statement (line 424)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 425):
    
    # Assigning a Num to a Name (line 425):
    int_201066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 21), 'int')
    # Assigning a type to the variable 'status' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'status', int_201066)
    
    # Assigning a Str to a Name (line 426):
    
    # Assigning a Str to a Name (line 426):
    str_201067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 23), 'str', 'There is a linear combination of rows of A_eq that results in zero, suggesting a redundant constraint. However the same linear combination of b_eq is nonzero, suggesting that the constraints conflict and the problem is infeasible.')
    # Assigning a type to the variable 'message' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'message', str_201067)
    # SSA join for if statement (line 424)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 433):
    
    # Assigning a Call to a Name (line 433):
    
    # Call to _get_densest(...): (line 433)
    # Processing the call arguments (line 433)
    # Getting the type of 'A' (line 433)
    A_201069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 32), 'A', False)
    # Getting the type of 'eligibleRows' (line 433)
    eligibleRows_201070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 35), 'eligibleRows', False)
    # Processing the call keyword arguments (line 433)
    kwargs_201071 = {}
    # Getting the type of '_get_densest' (line 433)
    _get_densest_201068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 19), '_get_densest', False)
    # Calling _get_densest(args, kwargs) (line 433)
    _get_densest_call_result_201072 = invoke(stypy.reporting.localization.Localization(__file__, 433, 19), _get_densest_201068, *[A_201069, eligibleRows_201070], **kwargs_201071)
    
    # Assigning a type to the variable 'i_remove' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'i_remove', _get_densest_call_result_201072)
    
    # Assigning a Call to a Name (line 434):
    
    # Assigning a Call to a Name (line 434):
    
    # Call to delete(...): (line 434)
    # Processing the call arguments (line 434)
    # Getting the type of 'A' (line 434)
    A_201075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 22), 'A', False)
    # Getting the type of 'i_remove' (line 434)
    i_remove_201076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 25), 'i_remove', False)
    # Processing the call keyword arguments (line 434)
    int_201077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 40), 'int')
    keyword_201078 = int_201077
    kwargs_201079 = {'axis': keyword_201078}
    # Getting the type of 'np' (line 434)
    np_201073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'np', False)
    # Obtaining the member 'delete' of a type (line 434)
    delete_201074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 12), np_201073, 'delete')
    # Calling delete(args, kwargs) (line 434)
    delete_call_result_201080 = invoke(stypy.reporting.localization.Localization(__file__, 434, 12), delete_201074, *[A_201075, i_remove_201076], **kwargs_201079)
    
    # Assigning a type to the variable 'A' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'A', delete_call_result_201080)
    
    # Assigning a Call to a Name (line 435):
    
    # Assigning a Call to a Name (line 435):
    
    # Call to delete(...): (line 435)
    # Processing the call arguments (line 435)
    # Getting the type of 'b' (line 435)
    b_201083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 22), 'b', False)
    # Getting the type of 'i_remove' (line 435)
    i_remove_201084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 25), 'i_remove', False)
    # Processing the call keyword arguments (line 435)
    kwargs_201085 = {}
    # Getting the type of 'np' (line 435)
    np_201081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'np', False)
    # Obtaining the member 'delete' of a type (line 435)
    delete_201082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 12), np_201081, 'delete')
    # Calling delete(args, kwargs) (line 435)
    delete_call_result_201086 = invoke(stypy.reporting.localization.Localization(__file__, 435, 12), delete_201082, *[b_201083, i_remove_201084], **kwargs_201085)
    
    # Assigning a type to the variable 'b' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'b', delete_call_result_201086)
    
    # Assigning a Call to a Tuple (line 436):
    
    # Assigning a Subscript to a Name (line 436):
    
    # Obtaining the type of the subscript
    int_201087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 8), 'int')
    
    # Call to svd(...): (line 436)
    # Processing the call arguments (line 436)
    # Getting the type of 'A' (line 436)
    A_201089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 23), 'A', False)
    # Processing the call keyword arguments (line 436)
    kwargs_201090 = {}
    # Getting the type of 'svd' (line 436)
    svd_201088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 19), 'svd', False)
    # Calling svd(args, kwargs) (line 436)
    svd_call_result_201091 = invoke(stypy.reporting.localization.Localization(__file__, 436, 19), svd_201088, *[A_201089], **kwargs_201090)
    
    # Obtaining the member '__getitem__' of a type (line 436)
    getitem___201092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 8), svd_call_result_201091, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 436)
    subscript_call_result_201093 = invoke(stypy.reporting.localization.Localization(__file__, 436, 8), getitem___201092, int_201087)
    
    # Assigning a type to the variable 'tuple_var_assignment_200128' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'tuple_var_assignment_200128', subscript_call_result_201093)
    
    # Assigning a Subscript to a Name (line 436):
    
    # Obtaining the type of the subscript
    int_201094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 8), 'int')
    
    # Call to svd(...): (line 436)
    # Processing the call arguments (line 436)
    # Getting the type of 'A' (line 436)
    A_201096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 23), 'A', False)
    # Processing the call keyword arguments (line 436)
    kwargs_201097 = {}
    # Getting the type of 'svd' (line 436)
    svd_201095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 19), 'svd', False)
    # Calling svd(args, kwargs) (line 436)
    svd_call_result_201098 = invoke(stypy.reporting.localization.Localization(__file__, 436, 19), svd_201095, *[A_201096], **kwargs_201097)
    
    # Obtaining the member '__getitem__' of a type (line 436)
    getitem___201099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 8), svd_call_result_201098, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 436)
    subscript_call_result_201100 = invoke(stypy.reporting.localization.Localization(__file__, 436, 8), getitem___201099, int_201094)
    
    # Assigning a type to the variable 'tuple_var_assignment_200129' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'tuple_var_assignment_200129', subscript_call_result_201100)
    
    # Assigning a Subscript to a Name (line 436):
    
    # Obtaining the type of the subscript
    int_201101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 8), 'int')
    
    # Call to svd(...): (line 436)
    # Processing the call arguments (line 436)
    # Getting the type of 'A' (line 436)
    A_201103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 23), 'A', False)
    # Processing the call keyword arguments (line 436)
    kwargs_201104 = {}
    # Getting the type of 'svd' (line 436)
    svd_201102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 19), 'svd', False)
    # Calling svd(args, kwargs) (line 436)
    svd_call_result_201105 = invoke(stypy.reporting.localization.Localization(__file__, 436, 19), svd_201102, *[A_201103], **kwargs_201104)
    
    # Obtaining the member '__getitem__' of a type (line 436)
    getitem___201106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 8), svd_call_result_201105, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 436)
    subscript_call_result_201107 = invoke(stypy.reporting.localization.Localization(__file__, 436, 8), getitem___201106, int_201101)
    
    # Assigning a type to the variable 'tuple_var_assignment_200130' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'tuple_var_assignment_200130', subscript_call_result_201107)
    
    # Assigning a Name to a Name (line 436):
    # Getting the type of 'tuple_var_assignment_200128' (line 436)
    tuple_var_assignment_200128_201108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'tuple_var_assignment_200128')
    # Assigning a type to the variable 'U' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'U', tuple_var_assignment_200128_201108)
    
    # Assigning a Name to a Name (line 436):
    # Getting the type of 'tuple_var_assignment_200129' (line 436)
    tuple_var_assignment_200129_201109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'tuple_var_assignment_200129')
    # Assigning a type to the variable 's' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 11), 's', tuple_var_assignment_200129_201109)
    
    # Assigning a Name to a Name (line 436):
    # Getting the type of 'tuple_var_assignment_200130' (line 436)
    tuple_var_assignment_200130_201110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'tuple_var_assignment_200130')
    # Assigning a type to the variable 'Vh' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 14), 'Vh', tuple_var_assignment_200130_201110)
    
    # Assigning a Attribute to a Tuple (line 437):
    
    # Assigning a Subscript to a Name (line 437):
    
    # Obtaining the type of the subscript
    int_201111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 8), 'int')
    # Getting the type of 'A' (line 437)
    A_201112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 15), 'A')
    # Obtaining the member 'shape' of a type (line 437)
    shape_201113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 15), A_201112, 'shape')
    # Obtaining the member '__getitem__' of a type (line 437)
    getitem___201114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), shape_201113, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 437)
    subscript_call_result_201115 = invoke(stypy.reporting.localization.Localization(__file__, 437, 8), getitem___201114, int_201111)
    
    # Assigning a type to the variable 'tuple_var_assignment_200131' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_200131', subscript_call_result_201115)
    
    # Assigning a Subscript to a Name (line 437):
    
    # Obtaining the type of the subscript
    int_201116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 8), 'int')
    # Getting the type of 'A' (line 437)
    A_201117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 15), 'A')
    # Obtaining the member 'shape' of a type (line 437)
    shape_201118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 15), A_201117, 'shape')
    # Obtaining the member '__getitem__' of a type (line 437)
    getitem___201119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), shape_201118, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 437)
    subscript_call_result_201120 = invoke(stypy.reporting.localization.Localization(__file__, 437, 8), getitem___201119, int_201116)
    
    # Assigning a type to the variable 'tuple_var_assignment_200132' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_200132', subscript_call_result_201120)
    
    # Assigning a Name to a Name (line 437):
    # Getting the type of 'tuple_var_assignment_200131' (line 437)
    tuple_var_assignment_200131_201121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_200131')
    # Assigning a type to the variable 'm' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'm', tuple_var_assignment_200131_201121)
    
    # Assigning a Name to a Name (line 437):
    # Getting the type of 'tuple_var_assignment_200132' (line 437)
    tuple_var_assignment_200132_201122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_200132')
    # Assigning a type to the variable 'n' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 11), 'n', tuple_var_assignment_200132_201122)
    
    # Assigning a IfExp to a Name (line 438):
    
    # Assigning a IfExp to a Name (line 438):
    
    
    # Getting the type of 'm' (line 438)
    m_201123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 25), 'm')
    # Getting the type of 'n' (line 438)
    n_201124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 30), 'n')
    # Applying the binary operator '<=' (line 438)
    result_le_201125 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 25), '<=', m_201123, n_201124)
    
    # Testing the type of an if expression (line 438)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 438, 16), result_le_201125)
    # SSA begins for if expression (line 438)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Obtaining the type of the subscript
    int_201126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 18), 'int')
    # Getting the type of 's' (line 438)
    s_201127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 16), 's')
    # Obtaining the member '__getitem__' of a type (line 438)
    getitem___201128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 16), s_201127, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 438)
    subscript_call_result_201129 = invoke(stypy.reporting.localization.Localization(__file__, 438, 16), getitem___201128, int_201126)
    
    # SSA branch for the else part of an if expression (line 438)
    module_type_store.open_ssa_branch('if expression else')
    int_201130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 37), 'int')
    # SSA join for if expression (line 438)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_201131 = union_type.UnionType.add(subscript_call_result_201129, int_201130)
    
    # Assigning a type to the variable 's_min' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 's_min', if_exp_201131)
    # SSA join for while statement (line 411)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 440)
    tuple_201132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 440)
    # Adding element type (line 440)
    # Getting the type of 'A' (line 440)
    A_201133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 11), 'A')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 11), tuple_201132, A_201133)
    # Adding element type (line 440)
    # Getting the type of 'b' (line 440)
    b_201134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 14), 'b')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 11), tuple_201132, b_201134)
    # Adding element type (line 440)
    # Getting the type of 'status' (line 440)
    status_201135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 17), 'status')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 11), tuple_201132, status_201135)
    # Adding element type (line 440)
    # Getting the type of 'message' (line 440)
    message_201136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 25), 'message')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 11), tuple_201132, message_201136)
    
    # Assigning a type to the variable 'stypy_return_type' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'stypy_return_type', tuple_201132)
    
    # ################# End of '_remove_redundancy(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_remove_redundancy' in the type store
    # Getting the type of 'stypy_return_type' (line 351)
    stypy_return_type_201137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_201137)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_remove_redundancy'
    return stypy_return_type_201137

# Assigning a type to the variable '_remove_redundancy' (line 351)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 0), '_remove_redundancy', _remove_redundancy)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
