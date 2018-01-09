
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Sparse block 1-norm estimator.
2: '''
3: 
4: from __future__ import division, print_function, absolute_import
5: 
6: import numpy as np
7: from scipy.sparse.linalg import aslinearoperator
8: 
9: 
10: __all__ = ['onenormest']
11: 
12: 
13: def onenormest(A, t=2, itmax=5, compute_v=False, compute_w=False):
14:     '''
15:     Compute a lower bound of the 1-norm of a sparse matrix.
16: 
17:     Parameters
18:     ----------
19:     A : ndarray or other linear operator
20:         A linear operator that can be transposed and that can
21:         produce matrix products.
22:     t : int, optional
23:         A positive parameter controlling the tradeoff between
24:         accuracy versus time and memory usage.
25:         Larger values take longer and use more memory
26:         but give more accurate output.
27:     itmax : int, optional
28:         Use at most this many iterations.
29:     compute_v : bool, optional
30:         Request a norm-maximizing linear operator input vector if True.
31:     compute_w : bool, optional
32:         Request a norm-maximizing linear operator output vector if True.
33: 
34:     Returns
35:     -------
36:     est : float
37:         An underestimate of the 1-norm of the sparse matrix.
38:     v : ndarray, optional
39:         The vector such that ||Av||_1 == est*||v||_1.
40:         It can be thought of as an input to the linear operator
41:         that gives an output with particularly large norm.
42:     w : ndarray, optional
43:         The vector Av which has relatively large 1-norm.
44:         It can be thought of as an output of the linear operator
45:         that is relatively large in norm compared to the input.
46: 
47:     Notes
48:     -----
49:     This is algorithm 2.4 of [1].
50: 
51:     In [2] it is described as follows.
52:     "This algorithm typically requires the evaluation of
53:     about 4t matrix-vector products and almost invariably
54:     produces a norm estimate (which is, in fact, a lower
55:     bound on the norm) correct to within a factor 3."
56: 
57:     .. versionadded:: 0.13.0
58: 
59:     References
60:     ----------
61:     .. [1] Nicholas J. Higham and Francoise Tisseur (2000),
62:            "A Block Algorithm for Matrix 1-Norm Estimation,
63:            with an Application to 1-Norm Pseudospectra."
64:            SIAM J. Matrix Anal. Appl. Vol. 21, No. 4, pp. 1185-1201.
65: 
66:     .. [2] Awad H. Al-Mohy and Nicholas J. Higham (2009),
67:            "A new scaling and squaring algorithm for the matrix exponential."
68:            SIAM J. Matrix Anal. Appl. Vol. 31, No. 3, pp. 970-989.
69: 
70:     Examples
71:     --------
72:     >>> from scipy.sparse import csc_matrix
73:     >>> from scipy.sparse.linalg import onenormest
74:     >>> A = csc_matrix([[1., 0., 0.], [5., 8., 2.], [0., -1., 0.]], dtype=float)
75:     >>> A.todense()
76:     matrix([[ 1.,  0.,  0.],
77:             [ 5.,  8.,  2.],
78:             [ 0., -1.,  0.]])
79:     >>> onenormest(A)
80:     9.0
81:     >>> np.linalg.norm(A.todense(), ord=1)
82:     9.0
83:     '''
84: 
85:     # Check the input.
86:     A = aslinearoperator(A)
87:     if A.shape[0] != A.shape[1]:
88:         raise ValueError('expected the operator to act like a square matrix')
89: 
90:     # If the operator size is small compared to t,
91:     # then it is easier to compute the exact norm.
92:     # Otherwise estimate the norm.
93:     n = A.shape[1]
94:     if t >= n:
95:         A_explicit = np.asarray(aslinearoperator(A).matmat(np.identity(n)))
96:         if A_explicit.shape != (n, n):
97:             raise Exception('internal error: ',
98:                     'unexpected shape ' + str(A_explicit.shape))
99:         col_abs_sums = abs(A_explicit).sum(axis=0)
100:         if col_abs_sums.shape != (n, ):
101:             raise Exception('internal error: ',
102:                     'unexpected shape ' + str(col_abs_sums.shape))
103:         argmax_j = np.argmax(col_abs_sums)
104:         v = elementary_vector(n, argmax_j)
105:         w = A_explicit[:, argmax_j]
106:         est = col_abs_sums[argmax_j]
107:     else:
108:         est, v, w, nmults, nresamples = _onenormest_core(A, A.H, t, itmax)
109: 
110:     # Report the norm estimate along with some certificates of the estimate.
111:     if compute_v or compute_w:
112:         result = (est,)
113:         if compute_v:
114:             result += (v,)
115:         if compute_w:
116:             result += (w,)
117:         return result
118:     else:
119:         return est
120: 
121: 
122: def _blocked_elementwise(func):
123:     '''
124:     Decorator for an elementwise function, to apply it blockwise along
125:     first dimension, to avoid excessive memory usage in temporaries.
126:     '''
127:     block_size = 2**20
128: 
129:     def wrapper(x):
130:         if x.shape[0] < block_size:
131:             return func(x)
132:         else:
133:             y0 = func(x[:block_size])
134:             y = np.zeros((x.shape[0],) + y0.shape[1:], dtype=y0.dtype)
135:             y[:block_size] = y0
136:             del y0
137:             for j in range(block_size, x.shape[0], block_size):
138:                 y[j:j+block_size] = func(x[j:j+block_size])
139:             return y
140:     return wrapper
141: 
142: 
143: @_blocked_elementwise
144: def sign_round_up(X):
145:     '''
146:     This should do the right thing for both real and complex matrices.
147: 
148:     From Higham and Tisseur:
149:     "Everything in this section remains valid for complex matrices
150:     provided that sign(A) is redefined as the matrix (aij / |aij|)
151:     (and sign(0) = 1) transposes are replaced by conjugate transposes."
152: 
153:     '''
154:     Y = X.copy()
155:     Y[Y == 0] = 1
156:     Y /= np.abs(Y)
157:     return Y
158: 
159: 
160: @_blocked_elementwise
161: def _max_abs_axis1(X):
162:     return np.max(np.abs(X), axis=1)
163: 
164: 
165: def _sum_abs_axis0(X):
166:     block_size = 2**20
167:     r = None
168:     for j in range(0, X.shape[0], block_size):
169:         y = np.sum(np.abs(X[j:j+block_size]), axis=0)
170:         if r is None:
171:             r = y
172:         else:
173:             r += y
174:     return r
175: 
176: 
177: def elementary_vector(n, i):
178:     v = np.zeros(n, dtype=float)
179:     v[i] = 1
180:     return v
181: 
182: 
183: def vectors_are_parallel(v, w):
184:     # Columns are considered parallel when they are equal or negative.
185:     # Entries are required to be in {-1, 1},
186:     # which guarantees that the magnitudes of the vectors are identical.
187:     if v.ndim != 1 or v.shape != w.shape:
188:         raise ValueError('expected conformant vectors with entries in {-1,1}')
189:     n = v.shape[0]
190:     return np.dot(v, w) == n
191: 
192: 
193: def every_col_of_X_is_parallel_to_a_col_of_Y(X, Y):
194:     for v in X.T:
195:         if not any(vectors_are_parallel(v, w) for w in Y.T):
196:             return False
197:     return True
198: 
199: 
200: def column_needs_resampling(i, X, Y=None):
201:     # column i of X needs resampling if either
202:     # it is parallel to a previous column of X or
203:     # it is parallel to a column of Y
204:     n, t = X.shape
205:     v = X[:, i]
206:     if any(vectors_are_parallel(v, X[:, j]) for j in range(i)):
207:         return True
208:     if Y is not None:
209:         if any(vectors_are_parallel(v, w) for w in Y.T):
210:             return True
211:     return False
212: 
213: 
214: def resample_column(i, X):
215:     X[:, i] = np.random.randint(0, 2, size=X.shape[0])*2 - 1
216: 
217: 
218: def less_than_or_close(a, b):
219:     return np.allclose(a, b) or (a < b)
220: 
221: 
222: def _algorithm_2_2(A, AT, t):
223:     '''
224:     This is Algorithm 2.2.
225: 
226:     Parameters
227:     ----------
228:     A : ndarray or other linear operator
229:         A linear operator that can produce matrix products.
230:     AT : ndarray or other linear operator
231:         The transpose of A.
232:     t : int, optional
233:         A positive parameter controlling the tradeoff between
234:         accuracy versus time and memory usage.
235: 
236:     Returns
237:     -------
238:     g : sequence
239:         A non-negative decreasing vector
240:         such that g[j] is a lower bound for the 1-norm
241:         of the column of A of jth largest 1-norm.
242:         The first entry of this vector is therefore a lower bound
243:         on the 1-norm of the linear operator A.
244:         This sequence has length t.
245:     ind : sequence
246:         The ith entry of ind is the index of the column A whose 1-norm
247:         is given by g[i].
248:         This sequence of indices has length t, and its entries are
249:         chosen from range(n), possibly with repetition,
250:         where n is the order of the operator A.
251: 
252:     Notes
253:     -----
254:     This algorithm is mainly for testing.
255:     It uses the 'ind' array in a way that is similar to
256:     its usage in algorithm 2.4.  This algorithm 2.2 may be easier to test,
257:     so it gives a chance of uncovering bugs related to indexing
258:     which could have propagated less noticeably to algorithm 2.4.
259: 
260:     '''
261:     A_linear_operator = aslinearoperator(A)
262:     AT_linear_operator = aslinearoperator(AT)
263:     n = A_linear_operator.shape[0]
264: 
265:     # Initialize the X block with columns of unit 1-norm.
266:     X = np.ones((n, t))
267:     if t > 1:
268:         X[:, 1:] = np.random.randint(0, 2, size=(n, t-1))*2 - 1
269:     X /= float(n)
270: 
271:     # Iteratively improve the lower bounds.
272:     # Track extra things, to assert invariants for debugging.
273:     g_prev = None
274:     h_prev = None
275:     k = 1
276:     ind = range(t)
277:     while True:
278:         Y = np.asarray(A_linear_operator.matmat(X))
279:         g = _sum_abs_axis0(Y)
280:         best_j = np.argmax(g)
281:         g.sort()
282:         g = g[::-1]
283:         S = sign_round_up(Y)
284:         Z = np.asarray(AT_linear_operator.matmat(S))
285:         h = _max_abs_axis1(Z)
286: 
287:         # If this algorithm runs for fewer than two iterations,
288:         # then its return values do not have the properties indicated
289:         # in the description of the algorithm.
290:         # In particular, the entries of g are not 1-norms of any
291:         # column of A until the second iteration.
292:         # Therefore we will require the algorithm to run for at least
293:         # two iterations, even though this requirement is not stated
294:         # in the description of the algorithm.
295:         if k >= 2:
296:             if less_than_or_close(max(h), np.dot(Z[:, best_j], X[:, best_j])):
297:                 break
298:         ind = np.argsort(h)[::-1][:t]
299:         h = h[ind]
300:         for j in range(t):
301:             X[:, j] = elementary_vector(n, ind[j])
302: 
303:         # Check invariant (2.2).
304:         if k >= 2:
305:             if not less_than_or_close(g_prev[0], h_prev[0]):
306:                 raise Exception('invariant (2.2) is violated')
307:             if not less_than_or_close(h_prev[0], g[0]):
308:                 raise Exception('invariant (2.2) is violated')
309: 
310:         # Check invariant (2.3).
311:         if k >= 3:
312:             for j in range(t):
313:                 if not less_than_or_close(g[j], g_prev[j]):
314:                     raise Exception('invariant (2.3) is violated')
315: 
316:         # Update for the next iteration.
317:         g_prev = g
318:         h_prev = h
319:         k += 1
320: 
321:     # Return the lower bounds and the corresponding column indices.
322:     return g, ind
323: 
324: 
325: def _onenormest_core(A, AT, t, itmax):
326:     '''
327:     Compute a lower bound of the 1-norm of a sparse matrix.
328: 
329:     Parameters
330:     ----------
331:     A : ndarray or other linear operator
332:         A linear operator that can produce matrix products.
333:     AT : ndarray or other linear operator
334:         The transpose of A.
335:     t : int, optional
336:         A positive parameter controlling the tradeoff between
337:         accuracy versus time and memory usage.
338:     itmax : int, optional
339:         Use at most this many iterations.
340: 
341:     Returns
342:     -------
343:     est : float
344:         An underestimate of the 1-norm of the sparse matrix.
345:     v : ndarray, optional
346:         The vector such that ||Av||_1 == est*||v||_1.
347:         It can be thought of as an input to the linear operator
348:         that gives an output with particularly large norm.
349:     w : ndarray, optional
350:         The vector Av which has relatively large 1-norm.
351:         It can be thought of as an output of the linear operator
352:         that is relatively large in norm compared to the input.
353:     nmults : int, optional
354:         The number of matrix products that were computed.
355:     nresamples : int, optional
356:         The number of times a parallel column was observed,
357:         necessitating a re-randomization of the column.
358: 
359:     Notes
360:     -----
361:     This is algorithm 2.4.
362: 
363:     '''
364:     # This function is a more or less direct translation
365:     # of Algorithm 2.4 from the Higham and Tisseur (2000) paper.
366:     A_linear_operator = aslinearoperator(A)
367:     AT_linear_operator = aslinearoperator(AT)
368:     if itmax < 2:
369:         raise ValueError('at least two iterations are required')
370:     if t < 1:
371:         raise ValueError('at least one column is required')
372:     n = A.shape[0]
373:     if t >= n:
374:         raise ValueError('t should be smaller than the order of A')
375:     # Track the number of big*small matrix multiplications
376:     # and the number of resamplings.
377:     nmults = 0
378:     nresamples = 0
379:     # "We now explain our choice of starting matrix.  We take the first
380:     # column of X to be the vector of 1s [...] This has the advantage that
381:     # for a matrix with nonnegative elements the algorithm converges
382:     # with an exact estimate on the second iteration, and such matrices
383:     # arise in applications [...]"
384:     X = np.ones((n, t), dtype=float)
385:     # "The remaining columns are chosen as rand{-1,1},
386:     # with a check for and correction of parallel columns,
387:     # exactly as for S in the body of the algorithm."
388:     if t > 1:
389:         for i in range(1, t):
390:             # These are technically initial samples, not resamples,
391:             # so the resampling count is not incremented.
392:             resample_column(i, X)
393:         for i in range(t):
394:             while column_needs_resampling(i, X):
395:                 resample_column(i, X)
396:                 nresamples += 1
397:     # "Choose starting matrix X with columns of unit 1-norm."
398:     X /= float(n)
399:     # "indices of used unit vectors e_j"
400:     ind_hist = np.zeros(0, dtype=np.intp)
401:     est_old = 0
402:     S = np.zeros((n, t), dtype=float)
403:     k = 1
404:     ind = None
405:     while True:
406:         Y = np.asarray(A_linear_operator.matmat(X))
407:         nmults += 1
408:         mags = _sum_abs_axis0(Y)
409:         est = np.max(mags)
410:         best_j = np.argmax(mags)
411:         if est > est_old or k == 2:
412:             if k >= 2:
413:                 ind_best = ind[best_j]
414:             w = Y[:, best_j]
415:         # (1)
416:         if k >= 2 and est <= est_old:
417:             est = est_old
418:             break
419:         est_old = est
420:         S_old = S
421:         if k > itmax:
422:             break
423:         S = sign_round_up(Y)
424:         del Y
425:         # (2)
426:         if every_col_of_X_is_parallel_to_a_col_of_Y(S, S_old):
427:             break
428:         if t > 1:
429:             # "Ensure that no column of S is parallel to another column of S
430:             # or to a column of S_old by replacing columns of S by rand{-1,1}."
431:             for i in range(t):
432:                 while column_needs_resampling(i, S, S_old):
433:                     resample_column(i, S)
434:                     nresamples += 1
435:         del S_old
436:         # (3)
437:         Z = np.asarray(AT_linear_operator.matmat(S))
438:         nmults += 1
439:         h = _max_abs_axis1(Z)
440:         del Z
441:         # (4)
442:         if k >= 2 and max(h) == h[ind_best]:
443:             break
444:         # "Sort h so that h_first >= ... >= h_last
445:         # and re-order ind correspondingly."
446:         #
447:         # Later on, we will need at most t+len(ind_hist) largest
448:         # entries, so drop the rest
449:         ind = np.argsort(h)[::-1][:t+len(ind_hist)].copy()
450:         del h
451:         if t > 1:
452:             # (5)
453:             # Break if the most promising t vectors have been visited already.
454:             if np.in1d(ind[:t], ind_hist).all():
455:                 break
456:             # Put the most promising unvisited vectors at the front of the list
457:             # and put the visited vectors at the end of the list.
458:             # Preserve the order of the indices induced by the ordering of h.
459:             seen = np.in1d(ind, ind_hist)
460:             ind = np.concatenate((ind[~seen], ind[seen]))
461:         for j in range(t):
462:             X[:, j] = elementary_vector(n, ind[j])
463: 
464:         new_ind = ind[:t][~np.in1d(ind[:t], ind_hist)]
465:         ind_hist = np.concatenate((ind_hist, new_ind))
466:         k += 1
467:     v = elementary_vector(n, ind_best)
468:     return est, v, w, nmults, nresamples
469: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_390532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', 'Sparse block 1-norm estimator.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_390533 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_390533) is not StypyTypeError):

    if (import_390533 != 'pyd_module'):
        __import__(import_390533)
        sys_modules_390534 = sys.modules[import_390533]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_390534.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_390533)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.sparse.linalg import aslinearoperator' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_390535 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse.linalg')

if (type(import_390535) is not StypyTypeError):

    if (import_390535 != 'pyd_module'):
        __import__(import_390535)
        sys_modules_390536 = sys.modules[import_390535]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse.linalg', sys_modules_390536.module_type_store, module_type_store, ['aslinearoperator'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_390536, sys_modules_390536.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import aslinearoperator

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse.linalg', None, module_type_store, ['aslinearoperator'], [aslinearoperator])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse.linalg', import_390535)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')


# Assigning a List to a Name (line 10):

# Assigning a List to a Name (line 10):
__all__ = ['onenormest']
module_type_store.set_exportable_members(['onenormest'])

# Obtaining an instance of the builtin type 'list' (line 10)
list_390537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)
str_390538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 11), 'str', 'onenormest')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_390537, str_390538)

# Assigning a type to the variable '__all__' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), '__all__', list_390537)

@norecursion
def onenormest(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_390539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 20), 'int')
    int_390540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 29), 'int')
    # Getting the type of 'False' (line 13)
    False_390541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 42), 'False')
    # Getting the type of 'False' (line 13)
    False_390542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 59), 'False')
    defaults = [int_390539, int_390540, False_390541, False_390542]
    # Create a new context for function 'onenormest'
    module_type_store = module_type_store.open_function_context('onenormest', 13, 0, False)
    
    # Passed parameters checking function
    onenormest.stypy_localization = localization
    onenormest.stypy_type_of_self = None
    onenormest.stypy_type_store = module_type_store
    onenormest.stypy_function_name = 'onenormest'
    onenormest.stypy_param_names_list = ['A', 't', 'itmax', 'compute_v', 'compute_w']
    onenormest.stypy_varargs_param_name = None
    onenormest.stypy_kwargs_param_name = None
    onenormest.stypy_call_defaults = defaults
    onenormest.stypy_call_varargs = varargs
    onenormest.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'onenormest', ['A', 't', 'itmax', 'compute_v', 'compute_w'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'onenormest', localization, ['A', 't', 'itmax', 'compute_v', 'compute_w'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'onenormest(...)' code ##################

    str_390543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, (-1)), 'str', '\n    Compute a lower bound of the 1-norm of a sparse matrix.\n\n    Parameters\n    ----------\n    A : ndarray or other linear operator\n        A linear operator that can be transposed and that can\n        produce matrix products.\n    t : int, optional\n        A positive parameter controlling the tradeoff between\n        accuracy versus time and memory usage.\n        Larger values take longer and use more memory\n        but give more accurate output.\n    itmax : int, optional\n        Use at most this many iterations.\n    compute_v : bool, optional\n        Request a norm-maximizing linear operator input vector if True.\n    compute_w : bool, optional\n        Request a norm-maximizing linear operator output vector if True.\n\n    Returns\n    -------\n    est : float\n        An underestimate of the 1-norm of the sparse matrix.\n    v : ndarray, optional\n        The vector such that ||Av||_1 == est*||v||_1.\n        It can be thought of as an input to the linear operator\n        that gives an output with particularly large norm.\n    w : ndarray, optional\n        The vector Av which has relatively large 1-norm.\n        It can be thought of as an output of the linear operator\n        that is relatively large in norm compared to the input.\n\n    Notes\n    -----\n    This is algorithm 2.4 of [1].\n\n    In [2] it is described as follows.\n    "This algorithm typically requires the evaluation of\n    about 4t matrix-vector products and almost invariably\n    produces a norm estimate (which is, in fact, a lower\n    bound on the norm) correct to within a factor 3."\n\n    .. versionadded:: 0.13.0\n\n    References\n    ----------\n    .. [1] Nicholas J. Higham and Francoise Tisseur (2000),\n           "A Block Algorithm for Matrix 1-Norm Estimation,\n           with an Application to 1-Norm Pseudospectra."\n           SIAM J. Matrix Anal. Appl. Vol. 21, No. 4, pp. 1185-1201.\n\n    .. [2] Awad H. Al-Mohy and Nicholas J. Higham (2009),\n           "A new scaling and squaring algorithm for the matrix exponential."\n           SIAM J. Matrix Anal. Appl. Vol. 31, No. 3, pp. 970-989.\n\n    Examples\n    --------\n    >>> from scipy.sparse import csc_matrix\n    >>> from scipy.sparse.linalg import onenormest\n    >>> A = csc_matrix([[1., 0., 0.], [5., 8., 2.], [0., -1., 0.]], dtype=float)\n    >>> A.todense()\n    matrix([[ 1.,  0.,  0.],\n            [ 5.,  8.,  2.],\n            [ 0., -1.,  0.]])\n    >>> onenormest(A)\n    9.0\n    >>> np.linalg.norm(A.todense(), ord=1)\n    9.0\n    ')
    
    # Assigning a Call to a Name (line 86):
    
    # Assigning a Call to a Name (line 86):
    
    # Call to aslinearoperator(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'A' (line 86)
    A_390545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 25), 'A', False)
    # Processing the call keyword arguments (line 86)
    kwargs_390546 = {}
    # Getting the type of 'aslinearoperator' (line 86)
    aslinearoperator_390544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'aslinearoperator', False)
    # Calling aslinearoperator(args, kwargs) (line 86)
    aslinearoperator_call_result_390547 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), aslinearoperator_390544, *[A_390545], **kwargs_390546)
    
    # Assigning a type to the variable 'A' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'A', aslinearoperator_call_result_390547)
    
    
    
    # Obtaining the type of the subscript
    int_390548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 15), 'int')
    # Getting the type of 'A' (line 87)
    A_390549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 7), 'A')
    # Obtaining the member 'shape' of a type (line 87)
    shape_390550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 7), A_390549, 'shape')
    # Obtaining the member '__getitem__' of a type (line 87)
    getitem___390551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 7), shape_390550, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 87)
    subscript_call_result_390552 = invoke(stypy.reporting.localization.Localization(__file__, 87, 7), getitem___390551, int_390548)
    
    
    # Obtaining the type of the subscript
    int_390553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 29), 'int')
    # Getting the type of 'A' (line 87)
    A_390554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 21), 'A')
    # Obtaining the member 'shape' of a type (line 87)
    shape_390555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 21), A_390554, 'shape')
    # Obtaining the member '__getitem__' of a type (line 87)
    getitem___390556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 21), shape_390555, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 87)
    subscript_call_result_390557 = invoke(stypy.reporting.localization.Localization(__file__, 87, 21), getitem___390556, int_390553)
    
    # Applying the binary operator '!=' (line 87)
    result_ne_390558 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 7), '!=', subscript_call_result_390552, subscript_call_result_390557)
    
    # Testing the type of an if condition (line 87)
    if_condition_390559 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 4), result_ne_390558)
    # Assigning a type to the variable 'if_condition_390559' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'if_condition_390559', if_condition_390559)
    # SSA begins for if statement (line 87)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 88)
    # Processing the call arguments (line 88)
    str_390561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 25), 'str', 'expected the operator to act like a square matrix')
    # Processing the call keyword arguments (line 88)
    kwargs_390562 = {}
    # Getting the type of 'ValueError' (line 88)
    ValueError_390560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 88)
    ValueError_call_result_390563 = invoke(stypy.reporting.localization.Localization(__file__, 88, 14), ValueError_390560, *[str_390561], **kwargs_390562)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 88, 8), ValueError_call_result_390563, 'raise parameter', BaseException)
    # SSA join for if statement (line 87)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 93):
    
    # Assigning a Subscript to a Name (line 93):
    
    # Obtaining the type of the subscript
    int_390564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 16), 'int')
    # Getting the type of 'A' (line 93)
    A_390565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'A')
    # Obtaining the member 'shape' of a type (line 93)
    shape_390566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), A_390565, 'shape')
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___390567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), shape_390566, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_390568 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), getitem___390567, int_390564)
    
    # Assigning a type to the variable 'n' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'n', subscript_call_result_390568)
    
    
    # Getting the type of 't' (line 94)
    t_390569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 7), 't')
    # Getting the type of 'n' (line 94)
    n_390570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'n')
    # Applying the binary operator '>=' (line 94)
    result_ge_390571 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 7), '>=', t_390569, n_390570)
    
    # Testing the type of an if condition (line 94)
    if_condition_390572 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 4), result_ge_390571)
    # Assigning a type to the variable 'if_condition_390572' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'if_condition_390572', if_condition_390572)
    # SSA begins for if statement (line 94)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 95):
    
    # Assigning a Call to a Name (line 95):
    
    # Call to asarray(...): (line 95)
    # Processing the call arguments (line 95)
    
    # Call to matmat(...): (line 95)
    # Processing the call arguments (line 95)
    
    # Call to identity(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'n' (line 95)
    n_390582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 71), 'n', False)
    # Processing the call keyword arguments (line 95)
    kwargs_390583 = {}
    # Getting the type of 'np' (line 95)
    np_390580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 59), 'np', False)
    # Obtaining the member 'identity' of a type (line 95)
    identity_390581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 59), np_390580, 'identity')
    # Calling identity(args, kwargs) (line 95)
    identity_call_result_390584 = invoke(stypy.reporting.localization.Localization(__file__, 95, 59), identity_390581, *[n_390582], **kwargs_390583)
    
    # Processing the call keyword arguments (line 95)
    kwargs_390585 = {}
    
    # Call to aslinearoperator(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'A' (line 95)
    A_390576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 49), 'A', False)
    # Processing the call keyword arguments (line 95)
    kwargs_390577 = {}
    # Getting the type of 'aslinearoperator' (line 95)
    aslinearoperator_390575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 32), 'aslinearoperator', False)
    # Calling aslinearoperator(args, kwargs) (line 95)
    aslinearoperator_call_result_390578 = invoke(stypy.reporting.localization.Localization(__file__, 95, 32), aslinearoperator_390575, *[A_390576], **kwargs_390577)
    
    # Obtaining the member 'matmat' of a type (line 95)
    matmat_390579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 32), aslinearoperator_call_result_390578, 'matmat')
    # Calling matmat(args, kwargs) (line 95)
    matmat_call_result_390586 = invoke(stypy.reporting.localization.Localization(__file__, 95, 32), matmat_390579, *[identity_call_result_390584], **kwargs_390585)
    
    # Processing the call keyword arguments (line 95)
    kwargs_390587 = {}
    # Getting the type of 'np' (line 95)
    np_390573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 21), 'np', False)
    # Obtaining the member 'asarray' of a type (line 95)
    asarray_390574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 21), np_390573, 'asarray')
    # Calling asarray(args, kwargs) (line 95)
    asarray_call_result_390588 = invoke(stypy.reporting.localization.Localization(__file__, 95, 21), asarray_390574, *[matmat_call_result_390586], **kwargs_390587)
    
    # Assigning a type to the variable 'A_explicit' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'A_explicit', asarray_call_result_390588)
    
    
    # Getting the type of 'A_explicit' (line 96)
    A_explicit_390589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 11), 'A_explicit')
    # Obtaining the member 'shape' of a type (line 96)
    shape_390590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 11), A_explicit_390589, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 96)
    tuple_390591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 96)
    # Adding element type (line 96)
    # Getting the type of 'n' (line 96)
    n_390592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 32), 'n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 32), tuple_390591, n_390592)
    # Adding element type (line 96)
    # Getting the type of 'n' (line 96)
    n_390593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 35), 'n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 32), tuple_390591, n_390593)
    
    # Applying the binary operator '!=' (line 96)
    result_ne_390594 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 11), '!=', shape_390590, tuple_390591)
    
    # Testing the type of an if condition (line 96)
    if_condition_390595 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 8), result_ne_390594)
    # Assigning a type to the variable 'if_condition_390595' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'if_condition_390595', if_condition_390595)
    # SSA begins for if statement (line 96)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 97)
    # Processing the call arguments (line 97)
    str_390597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 28), 'str', 'internal error: ')
    str_390598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 20), 'str', 'unexpected shape ')
    
    # Call to str(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'A_explicit' (line 98)
    A_explicit_390600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 46), 'A_explicit', False)
    # Obtaining the member 'shape' of a type (line 98)
    shape_390601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 46), A_explicit_390600, 'shape')
    # Processing the call keyword arguments (line 98)
    kwargs_390602 = {}
    # Getting the type of 'str' (line 98)
    str_390599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 42), 'str', False)
    # Calling str(args, kwargs) (line 98)
    str_call_result_390603 = invoke(stypy.reporting.localization.Localization(__file__, 98, 42), str_390599, *[shape_390601], **kwargs_390602)
    
    # Applying the binary operator '+' (line 98)
    result_add_390604 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 20), '+', str_390598, str_call_result_390603)
    
    # Processing the call keyword arguments (line 97)
    kwargs_390605 = {}
    # Getting the type of 'Exception' (line 97)
    Exception_390596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 18), 'Exception', False)
    # Calling Exception(args, kwargs) (line 97)
    Exception_call_result_390606 = invoke(stypy.reporting.localization.Localization(__file__, 97, 18), Exception_390596, *[str_390597, result_add_390604], **kwargs_390605)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 97, 12), Exception_call_result_390606, 'raise parameter', BaseException)
    # SSA join for if statement (line 96)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 99):
    
    # Assigning a Call to a Name (line 99):
    
    # Call to sum(...): (line 99)
    # Processing the call keyword arguments (line 99)
    int_390612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 48), 'int')
    keyword_390613 = int_390612
    kwargs_390614 = {'axis': keyword_390613}
    
    # Call to abs(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'A_explicit' (line 99)
    A_explicit_390608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 27), 'A_explicit', False)
    # Processing the call keyword arguments (line 99)
    kwargs_390609 = {}
    # Getting the type of 'abs' (line 99)
    abs_390607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 23), 'abs', False)
    # Calling abs(args, kwargs) (line 99)
    abs_call_result_390610 = invoke(stypy.reporting.localization.Localization(__file__, 99, 23), abs_390607, *[A_explicit_390608], **kwargs_390609)
    
    # Obtaining the member 'sum' of a type (line 99)
    sum_390611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 23), abs_call_result_390610, 'sum')
    # Calling sum(args, kwargs) (line 99)
    sum_call_result_390615 = invoke(stypy.reporting.localization.Localization(__file__, 99, 23), sum_390611, *[], **kwargs_390614)
    
    # Assigning a type to the variable 'col_abs_sums' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'col_abs_sums', sum_call_result_390615)
    
    
    # Getting the type of 'col_abs_sums' (line 100)
    col_abs_sums_390616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 11), 'col_abs_sums')
    # Obtaining the member 'shape' of a type (line 100)
    shape_390617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 11), col_abs_sums_390616, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 100)
    tuple_390618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 100)
    # Adding element type (line 100)
    # Getting the type of 'n' (line 100)
    n_390619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 34), 'n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 34), tuple_390618, n_390619)
    
    # Applying the binary operator '!=' (line 100)
    result_ne_390620 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 11), '!=', shape_390617, tuple_390618)
    
    # Testing the type of an if condition (line 100)
    if_condition_390621 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 8), result_ne_390620)
    # Assigning a type to the variable 'if_condition_390621' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'if_condition_390621', if_condition_390621)
    # SSA begins for if statement (line 100)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 101)
    # Processing the call arguments (line 101)
    str_390623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 28), 'str', 'internal error: ')
    str_390624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 20), 'str', 'unexpected shape ')
    
    # Call to str(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'col_abs_sums' (line 102)
    col_abs_sums_390626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 46), 'col_abs_sums', False)
    # Obtaining the member 'shape' of a type (line 102)
    shape_390627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 46), col_abs_sums_390626, 'shape')
    # Processing the call keyword arguments (line 102)
    kwargs_390628 = {}
    # Getting the type of 'str' (line 102)
    str_390625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 42), 'str', False)
    # Calling str(args, kwargs) (line 102)
    str_call_result_390629 = invoke(stypy.reporting.localization.Localization(__file__, 102, 42), str_390625, *[shape_390627], **kwargs_390628)
    
    # Applying the binary operator '+' (line 102)
    result_add_390630 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 20), '+', str_390624, str_call_result_390629)
    
    # Processing the call keyword arguments (line 101)
    kwargs_390631 = {}
    # Getting the type of 'Exception' (line 101)
    Exception_390622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 18), 'Exception', False)
    # Calling Exception(args, kwargs) (line 101)
    Exception_call_result_390632 = invoke(stypy.reporting.localization.Localization(__file__, 101, 18), Exception_390622, *[str_390623, result_add_390630], **kwargs_390631)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 101, 12), Exception_call_result_390632, 'raise parameter', BaseException)
    # SSA join for if statement (line 100)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 103):
    
    # Assigning a Call to a Name (line 103):
    
    # Call to argmax(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'col_abs_sums' (line 103)
    col_abs_sums_390635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 29), 'col_abs_sums', False)
    # Processing the call keyword arguments (line 103)
    kwargs_390636 = {}
    # Getting the type of 'np' (line 103)
    np_390633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 19), 'np', False)
    # Obtaining the member 'argmax' of a type (line 103)
    argmax_390634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 19), np_390633, 'argmax')
    # Calling argmax(args, kwargs) (line 103)
    argmax_call_result_390637 = invoke(stypy.reporting.localization.Localization(__file__, 103, 19), argmax_390634, *[col_abs_sums_390635], **kwargs_390636)
    
    # Assigning a type to the variable 'argmax_j' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'argmax_j', argmax_call_result_390637)
    
    # Assigning a Call to a Name (line 104):
    
    # Assigning a Call to a Name (line 104):
    
    # Call to elementary_vector(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'n' (line 104)
    n_390639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 30), 'n', False)
    # Getting the type of 'argmax_j' (line 104)
    argmax_j_390640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 33), 'argmax_j', False)
    # Processing the call keyword arguments (line 104)
    kwargs_390641 = {}
    # Getting the type of 'elementary_vector' (line 104)
    elementary_vector_390638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'elementary_vector', False)
    # Calling elementary_vector(args, kwargs) (line 104)
    elementary_vector_call_result_390642 = invoke(stypy.reporting.localization.Localization(__file__, 104, 12), elementary_vector_390638, *[n_390639, argmax_j_390640], **kwargs_390641)
    
    # Assigning a type to the variable 'v' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'v', elementary_vector_call_result_390642)
    
    # Assigning a Subscript to a Name (line 105):
    
    # Assigning a Subscript to a Name (line 105):
    
    # Obtaining the type of the subscript
    slice_390643 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 105, 12), None, None, None)
    # Getting the type of 'argmax_j' (line 105)
    argmax_j_390644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 26), 'argmax_j')
    # Getting the type of 'A_explicit' (line 105)
    A_explicit_390645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'A_explicit')
    # Obtaining the member '__getitem__' of a type (line 105)
    getitem___390646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), A_explicit_390645, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 105)
    subscript_call_result_390647 = invoke(stypy.reporting.localization.Localization(__file__, 105, 12), getitem___390646, (slice_390643, argmax_j_390644))
    
    # Assigning a type to the variable 'w' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'w', subscript_call_result_390647)
    
    # Assigning a Subscript to a Name (line 106):
    
    # Assigning a Subscript to a Name (line 106):
    
    # Obtaining the type of the subscript
    # Getting the type of 'argmax_j' (line 106)
    argmax_j_390648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 27), 'argmax_j')
    # Getting the type of 'col_abs_sums' (line 106)
    col_abs_sums_390649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 'col_abs_sums')
    # Obtaining the member '__getitem__' of a type (line 106)
    getitem___390650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 14), col_abs_sums_390649, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
    subscript_call_result_390651 = invoke(stypy.reporting.localization.Localization(__file__, 106, 14), getitem___390650, argmax_j_390648)
    
    # Assigning a type to the variable 'est' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'est', subscript_call_result_390651)
    # SSA branch for the else part of an if statement (line 94)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 108):
    
    # Assigning a Subscript to a Name (line 108):
    
    # Obtaining the type of the subscript
    int_390652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 8), 'int')
    
    # Call to _onenormest_core(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'A' (line 108)
    A_390654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 57), 'A', False)
    # Getting the type of 'A' (line 108)
    A_390655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 60), 'A', False)
    # Obtaining the member 'H' of a type (line 108)
    H_390656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 60), A_390655, 'H')
    # Getting the type of 't' (line 108)
    t_390657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 65), 't', False)
    # Getting the type of 'itmax' (line 108)
    itmax_390658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 68), 'itmax', False)
    # Processing the call keyword arguments (line 108)
    kwargs_390659 = {}
    # Getting the type of '_onenormest_core' (line 108)
    _onenormest_core_390653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 40), '_onenormest_core', False)
    # Calling _onenormest_core(args, kwargs) (line 108)
    _onenormest_core_call_result_390660 = invoke(stypy.reporting.localization.Localization(__file__, 108, 40), _onenormest_core_390653, *[A_390654, H_390656, t_390657, itmax_390658], **kwargs_390659)
    
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___390661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), _onenormest_core_call_result_390660, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_390662 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), getitem___390661, int_390652)
    
    # Assigning a type to the variable 'tuple_var_assignment_390525' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_var_assignment_390525', subscript_call_result_390662)
    
    # Assigning a Subscript to a Name (line 108):
    
    # Obtaining the type of the subscript
    int_390663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 8), 'int')
    
    # Call to _onenormest_core(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'A' (line 108)
    A_390665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 57), 'A', False)
    # Getting the type of 'A' (line 108)
    A_390666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 60), 'A', False)
    # Obtaining the member 'H' of a type (line 108)
    H_390667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 60), A_390666, 'H')
    # Getting the type of 't' (line 108)
    t_390668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 65), 't', False)
    # Getting the type of 'itmax' (line 108)
    itmax_390669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 68), 'itmax', False)
    # Processing the call keyword arguments (line 108)
    kwargs_390670 = {}
    # Getting the type of '_onenormest_core' (line 108)
    _onenormest_core_390664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 40), '_onenormest_core', False)
    # Calling _onenormest_core(args, kwargs) (line 108)
    _onenormest_core_call_result_390671 = invoke(stypy.reporting.localization.Localization(__file__, 108, 40), _onenormest_core_390664, *[A_390665, H_390667, t_390668, itmax_390669], **kwargs_390670)
    
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___390672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), _onenormest_core_call_result_390671, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_390673 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), getitem___390672, int_390663)
    
    # Assigning a type to the variable 'tuple_var_assignment_390526' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_var_assignment_390526', subscript_call_result_390673)
    
    # Assigning a Subscript to a Name (line 108):
    
    # Obtaining the type of the subscript
    int_390674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 8), 'int')
    
    # Call to _onenormest_core(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'A' (line 108)
    A_390676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 57), 'A', False)
    # Getting the type of 'A' (line 108)
    A_390677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 60), 'A', False)
    # Obtaining the member 'H' of a type (line 108)
    H_390678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 60), A_390677, 'H')
    # Getting the type of 't' (line 108)
    t_390679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 65), 't', False)
    # Getting the type of 'itmax' (line 108)
    itmax_390680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 68), 'itmax', False)
    # Processing the call keyword arguments (line 108)
    kwargs_390681 = {}
    # Getting the type of '_onenormest_core' (line 108)
    _onenormest_core_390675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 40), '_onenormest_core', False)
    # Calling _onenormest_core(args, kwargs) (line 108)
    _onenormest_core_call_result_390682 = invoke(stypy.reporting.localization.Localization(__file__, 108, 40), _onenormest_core_390675, *[A_390676, H_390678, t_390679, itmax_390680], **kwargs_390681)
    
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___390683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), _onenormest_core_call_result_390682, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_390684 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), getitem___390683, int_390674)
    
    # Assigning a type to the variable 'tuple_var_assignment_390527' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_var_assignment_390527', subscript_call_result_390684)
    
    # Assigning a Subscript to a Name (line 108):
    
    # Obtaining the type of the subscript
    int_390685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 8), 'int')
    
    # Call to _onenormest_core(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'A' (line 108)
    A_390687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 57), 'A', False)
    # Getting the type of 'A' (line 108)
    A_390688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 60), 'A', False)
    # Obtaining the member 'H' of a type (line 108)
    H_390689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 60), A_390688, 'H')
    # Getting the type of 't' (line 108)
    t_390690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 65), 't', False)
    # Getting the type of 'itmax' (line 108)
    itmax_390691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 68), 'itmax', False)
    # Processing the call keyword arguments (line 108)
    kwargs_390692 = {}
    # Getting the type of '_onenormest_core' (line 108)
    _onenormest_core_390686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 40), '_onenormest_core', False)
    # Calling _onenormest_core(args, kwargs) (line 108)
    _onenormest_core_call_result_390693 = invoke(stypy.reporting.localization.Localization(__file__, 108, 40), _onenormest_core_390686, *[A_390687, H_390689, t_390690, itmax_390691], **kwargs_390692)
    
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___390694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), _onenormest_core_call_result_390693, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_390695 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), getitem___390694, int_390685)
    
    # Assigning a type to the variable 'tuple_var_assignment_390528' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_var_assignment_390528', subscript_call_result_390695)
    
    # Assigning a Subscript to a Name (line 108):
    
    # Obtaining the type of the subscript
    int_390696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 8), 'int')
    
    # Call to _onenormest_core(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'A' (line 108)
    A_390698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 57), 'A', False)
    # Getting the type of 'A' (line 108)
    A_390699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 60), 'A', False)
    # Obtaining the member 'H' of a type (line 108)
    H_390700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 60), A_390699, 'H')
    # Getting the type of 't' (line 108)
    t_390701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 65), 't', False)
    # Getting the type of 'itmax' (line 108)
    itmax_390702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 68), 'itmax', False)
    # Processing the call keyword arguments (line 108)
    kwargs_390703 = {}
    # Getting the type of '_onenormest_core' (line 108)
    _onenormest_core_390697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 40), '_onenormest_core', False)
    # Calling _onenormest_core(args, kwargs) (line 108)
    _onenormest_core_call_result_390704 = invoke(stypy.reporting.localization.Localization(__file__, 108, 40), _onenormest_core_390697, *[A_390698, H_390700, t_390701, itmax_390702], **kwargs_390703)
    
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___390705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), _onenormest_core_call_result_390704, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_390706 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), getitem___390705, int_390696)
    
    # Assigning a type to the variable 'tuple_var_assignment_390529' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_var_assignment_390529', subscript_call_result_390706)
    
    # Assigning a Name to a Name (line 108):
    # Getting the type of 'tuple_var_assignment_390525' (line 108)
    tuple_var_assignment_390525_390707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_var_assignment_390525')
    # Assigning a type to the variable 'est' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'est', tuple_var_assignment_390525_390707)
    
    # Assigning a Name to a Name (line 108):
    # Getting the type of 'tuple_var_assignment_390526' (line 108)
    tuple_var_assignment_390526_390708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_var_assignment_390526')
    # Assigning a type to the variable 'v' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'v', tuple_var_assignment_390526_390708)
    
    # Assigning a Name to a Name (line 108):
    # Getting the type of 'tuple_var_assignment_390527' (line 108)
    tuple_var_assignment_390527_390709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_var_assignment_390527')
    # Assigning a type to the variable 'w' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'w', tuple_var_assignment_390527_390709)
    
    # Assigning a Name to a Name (line 108):
    # Getting the type of 'tuple_var_assignment_390528' (line 108)
    tuple_var_assignment_390528_390710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_var_assignment_390528')
    # Assigning a type to the variable 'nmults' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 19), 'nmults', tuple_var_assignment_390528_390710)
    
    # Assigning a Name to a Name (line 108):
    # Getting the type of 'tuple_var_assignment_390529' (line 108)
    tuple_var_assignment_390529_390711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_var_assignment_390529')
    # Assigning a type to the variable 'nresamples' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'nresamples', tuple_var_assignment_390529_390711)
    # SSA join for if statement (line 94)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'compute_v' (line 111)
    compute_v_390712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 7), 'compute_v')
    # Getting the type of 'compute_w' (line 111)
    compute_w_390713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 20), 'compute_w')
    # Applying the binary operator 'or' (line 111)
    result_or_keyword_390714 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 7), 'or', compute_v_390712, compute_w_390713)
    
    # Testing the type of an if condition (line 111)
    if_condition_390715 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 4), result_or_keyword_390714)
    # Assigning a type to the variable 'if_condition_390715' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'if_condition_390715', if_condition_390715)
    # SSA begins for if statement (line 111)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 112):
    
    # Assigning a Tuple to a Name (line 112):
    
    # Obtaining an instance of the builtin type 'tuple' (line 112)
    tuple_390716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 112)
    # Adding element type (line 112)
    # Getting the type of 'est' (line 112)
    est_390717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 18), 'est')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 18), tuple_390716, est_390717)
    
    # Assigning a type to the variable 'result' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'result', tuple_390716)
    
    # Getting the type of 'compute_v' (line 113)
    compute_v_390718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 11), 'compute_v')
    # Testing the type of an if condition (line 113)
    if_condition_390719 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 8), compute_v_390718)
    # Assigning a type to the variable 'if_condition_390719' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'if_condition_390719', if_condition_390719)
    # SSA begins for if statement (line 113)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'result' (line 114)
    result_390720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'result')
    
    # Obtaining an instance of the builtin type 'tuple' (line 114)
    tuple_390721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 114)
    # Adding element type (line 114)
    # Getting the type of 'v' (line 114)
    v_390722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 23), 'v')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 23), tuple_390721, v_390722)
    
    # Applying the binary operator '+=' (line 114)
    result_iadd_390723 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 12), '+=', result_390720, tuple_390721)
    # Assigning a type to the variable 'result' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'result', result_iadd_390723)
    
    # SSA join for if statement (line 113)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'compute_w' (line 115)
    compute_w_390724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'compute_w')
    # Testing the type of an if condition (line 115)
    if_condition_390725 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 8), compute_w_390724)
    # Assigning a type to the variable 'if_condition_390725' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'if_condition_390725', if_condition_390725)
    # SSA begins for if statement (line 115)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'result' (line 116)
    result_390726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'result')
    
    # Obtaining an instance of the builtin type 'tuple' (line 116)
    tuple_390727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 116)
    # Adding element type (line 116)
    # Getting the type of 'w' (line 116)
    w_390728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 23), 'w')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 23), tuple_390727, w_390728)
    
    # Applying the binary operator '+=' (line 116)
    result_iadd_390729 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 12), '+=', result_390726, tuple_390727)
    # Assigning a type to the variable 'result' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'result', result_iadd_390729)
    
    # SSA join for if statement (line 115)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'result' (line 117)
    result_390730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'stypy_return_type', result_390730)
    # SSA branch for the else part of an if statement (line 111)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'est' (line 119)
    est_390731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'est')
    # Assigning a type to the variable 'stypy_return_type' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'stypy_return_type', est_390731)
    # SSA join for if statement (line 111)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'onenormest(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'onenormest' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_390732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_390732)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'onenormest'
    return stypy_return_type_390732

# Assigning a type to the variable 'onenormest' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'onenormest', onenormest)

@norecursion
def _blocked_elementwise(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_blocked_elementwise'
    module_type_store = module_type_store.open_function_context('_blocked_elementwise', 122, 0, False)
    
    # Passed parameters checking function
    _blocked_elementwise.stypy_localization = localization
    _blocked_elementwise.stypy_type_of_self = None
    _blocked_elementwise.stypy_type_store = module_type_store
    _blocked_elementwise.stypy_function_name = '_blocked_elementwise'
    _blocked_elementwise.stypy_param_names_list = ['func']
    _blocked_elementwise.stypy_varargs_param_name = None
    _blocked_elementwise.stypy_kwargs_param_name = None
    _blocked_elementwise.stypy_call_defaults = defaults
    _blocked_elementwise.stypy_call_varargs = varargs
    _blocked_elementwise.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_blocked_elementwise', ['func'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_blocked_elementwise', localization, ['func'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_blocked_elementwise(...)' code ##################

    str_390733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, (-1)), 'str', '\n    Decorator for an elementwise function, to apply it blockwise along\n    first dimension, to avoid excessive memory usage in temporaries.\n    ')
    
    # Assigning a BinOp to a Name (line 127):
    
    # Assigning a BinOp to a Name (line 127):
    int_390734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 17), 'int')
    int_390735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 20), 'int')
    # Applying the binary operator '**' (line 127)
    result_pow_390736 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 17), '**', int_390734, int_390735)
    
    # Assigning a type to the variable 'block_size' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'block_size', result_pow_390736)

    @norecursion
    def wrapper(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'wrapper'
        module_type_store = module_type_store.open_function_context('wrapper', 129, 4, False)
        
        # Passed parameters checking function
        wrapper.stypy_localization = localization
        wrapper.stypy_type_of_self = None
        wrapper.stypy_type_store = module_type_store
        wrapper.stypy_function_name = 'wrapper'
        wrapper.stypy_param_names_list = ['x']
        wrapper.stypy_varargs_param_name = None
        wrapper.stypy_kwargs_param_name = None
        wrapper.stypy_call_defaults = defaults
        wrapper.stypy_call_varargs = varargs
        wrapper.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'wrapper', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'wrapper', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'wrapper(...)' code ##################

        
        
        
        # Obtaining the type of the subscript
        int_390737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 19), 'int')
        # Getting the type of 'x' (line 130)
        x_390738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 11), 'x')
        # Obtaining the member 'shape' of a type (line 130)
        shape_390739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 11), x_390738, 'shape')
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___390740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 11), shape_390739, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
        subscript_call_result_390741 = invoke(stypy.reporting.localization.Localization(__file__, 130, 11), getitem___390740, int_390737)
        
        # Getting the type of 'block_size' (line 130)
        block_size_390742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), 'block_size')
        # Applying the binary operator '<' (line 130)
        result_lt_390743 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 11), '<', subscript_call_result_390741, block_size_390742)
        
        # Testing the type of an if condition (line 130)
        if_condition_390744 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 8), result_lt_390743)
        # Assigning a type to the variable 'if_condition_390744' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'if_condition_390744', if_condition_390744)
        # SSA begins for if statement (line 130)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to func(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'x' (line 131)
        x_390746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 24), 'x', False)
        # Processing the call keyword arguments (line 131)
        kwargs_390747 = {}
        # Getting the type of 'func' (line 131)
        func_390745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 19), 'func', False)
        # Calling func(args, kwargs) (line 131)
        func_call_result_390748 = invoke(stypy.reporting.localization.Localization(__file__, 131, 19), func_390745, *[x_390746], **kwargs_390747)
        
        # Assigning a type to the variable 'stypy_return_type' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'stypy_return_type', func_call_result_390748)
        # SSA branch for the else part of an if statement (line 130)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 133):
        
        # Assigning a Call to a Name (line 133):
        
        # Call to func(...): (line 133)
        # Processing the call arguments (line 133)
        
        # Obtaining the type of the subscript
        # Getting the type of 'block_size' (line 133)
        block_size_390750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 25), 'block_size', False)
        slice_390751 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 133, 22), None, block_size_390750, None)
        # Getting the type of 'x' (line 133)
        x_390752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 22), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 133)
        getitem___390753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 22), x_390752, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 133)
        subscript_call_result_390754 = invoke(stypy.reporting.localization.Localization(__file__, 133, 22), getitem___390753, slice_390751)
        
        # Processing the call keyword arguments (line 133)
        kwargs_390755 = {}
        # Getting the type of 'func' (line 133)
        func_390749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 17), 'func', False)
        # Calling func(args, kwargs) (line 133)
        func_call_result_390756 = invoke(stypy.reporting.localization.Localization(__file__, 133, 17), func_390749, *[subscript_call_result_390754], **kwargs_390755)
        
        # Assigning a type to the variable 'y0' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'y0', func_call_result_390756)
        
        # Assigning a Call to a Name (line 134):
        
        # Assigning a Call to a Name (line 134):
        
        # Call to zeros(...): (line 134)
        # Processing the call arguments (line 134)
        
        # Obtaining an instance of the builtin type 'tuple' (line 134)
        tuple_390759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 134)
        # Adding element type (line 134)
        
        # Obtaining the type of the subscript
        int_390760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 34), 'int')
        # Getting the type of 'x' (line 134)
        x_390761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 26), 'x', False)
        # Obtaining the member 'shape' of a type (line 134)
        shape_390762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 26), x_390761, 'shape')
        # Obtaining the member '__getitem__' of a type (line 134)
        getitem___390763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 26), shape_390762, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 134)
        subscript_call_result_390764 = invoke(stypy.reporting.localization.Localization(__file__, 134, 26), getitem___390763, int_390760)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 26), tuple_390759, subscript_call_result_390764)
        
        
        # Obtaining the type of the subscript
        int_390765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 50), 'int')
        slice_390766 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 134, 41), int_390765, None, None)
        # Getting the type of 'y0' (line 134)
        y0_390767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 41), 'y0', False)
        # Obtaining the member 'shape' of a type (line 134)
        shape_390768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 41), y0_390767, 'shape')
        # Obtaining the member '__getitem__' of a type (line 134)
        getitem___390769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 41), shape_390768, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 134)
        subscript_call_result_390770 = invoke(stypy.reporting.localization.Localization(__file__, 134, 41), getitem___390769, slice_390766)
        
        # Applying the binary operator '+' (line 134)
        result_add_390771 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 25), '+', tuple_390759, subscript_call_result_390770)
        
        # Processing the call keyword arguments (line 134)
        # Getting the type of 'y0' (line 134)
        y0_390772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 61), 'y0', False)
        # Obtaining the member 'dtype' of a type (line 134)
        dtype_390773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 61), y0_390772, 'dtype')
        keyword_390774 = dtype_390773
        kwargs_390775 = {'dtype': keyword_390774}
        # Getting the type of 'np' (line 134)
        np_390757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'np', False)
        # Obtaining the member 'zeros' of a type (line 134)
        zeros_390758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 16), np_390757, 'zeros')
        # Calling zeros(args, kwargs) (line 134)
        zeros_call_result_390776 = invoke(stypy.reporting.localization.Localization(__file__, 134, 16), zeros_390758, *[result_add_390771], **kwargs_390775)
        
        # Assigning a type to the variable 'y' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'y', zeros_call_result_390776)
        
        # Assigning a Name to a Subscript (line 135):
        
        # Assigning a Name to a Subscript (line 135):
        # Getting the type of 'y0' (line 135)
        y0_390777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 29), 'y0')
        # Getting the type of 'y' (line 135)
        y_390778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'y')
        # Getting the type of 'block_size' (line 135)
        block_size_390779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), 'block_size')
        slice_390780 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 135, 12), None, block_size_390779, None)
        # Storing an element on a container (line 135)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 12), y_390778, (slice_390780, y0_390777))
        # Deleting a member
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 136, 12), module_type_store, 'y0')
        
        
        # Call to range(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'block_size' (line 137)
        block_size_390782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 27), 'block_size', False)
        
        # Obtaining the type of the subscript
        int_390783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 47), 'int')
        # Getting the type of 'x' (line 137)
        x_390784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 39), 'x', False)
        # Obtaining the member 'shape' of a type (line 137)
        shape_390785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 39), x_390784, 'shape')
        # Obtaining the member '__getitem__' of a type (line 137)
        getitem___390786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 39), shape_390785, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 137)
        subscript_call_result_390787 = invoke(stypy.reporting.localization.Localization(__file__, 137, 39), getitem___390786, int_390783)
        
        # Getting the type of 'block_size' (line 137)
        block_size_390788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 51), 'block_size', False)
        # Processing the call keyword arguments (line 137)
        kwargs_390789 = {}
        # Getting the type of 'range' (line 137)
        range_390781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 21), 'range', False)
        # Calling range(args, kwargs) (line 137)
        range_call_result_390790 = invoke(stypy.reporting.localization.Localization(__file__, 137, 21), range_390781, *[block_size_390782, subscript_call_result_390787, block_size_390788], **kwargs_390789)
        
        # Testing the type of a for loop iterable (line 137)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 137, 12), range_call_result_390790)
        # Getting the type of the for loop variable (line 137)
        for_loop_var_390791 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 137, 12), range_call_result_390790)
        # Assigning a type to the variable 'j' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'j', for_loop_var_390791)
        # SSA begins for a for statement (line 137)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Subscript (line 138):
        
        # Assigning a Call to a Subscript (line 138):
        
        # Call to func(...): (line 138)
        # Processing the call arguments (line 138)
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 138)
        j_390793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 43), 'j', False)
        # Getting the type of 'j' (line 138)
        j_390794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 45), 'j', False)
        # Getting the type of 'block_size' (line 138)
        block_size_390795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 47), 'block_size', False)
        # Applying the binary operator '+' (line 138)
        result_add_390796 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 45), '+', j_390794, block_size_390795)
        
        slice_390797 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 138, 41), j_390793, result_add_390796, None)
        # Getting the type of 'x' (line 138)
        x_390798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 41), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 138)
        getitem___390799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 41), x_390798, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 138)
        subscript_call_result_390800 = invoke(stypy.reporting.localization.Localization(__file__, 138, 41), getitem___390799, slice_390797)
        
        # Processing the call keyword arguments (line 138)
        kwargs_390801 = {}
        # Getting the type of 'func' (line 138)
        func_390792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 36), 'func', False)
        # Calling func(args, kwargs) (line 138)
        func_call_result_390802 = invoke(stypy.reporting.localization.Localization(__file__, 138, 36), func_390792, *[subscript_call_result_390800], **kwargs_390801)
        
        # Getting the type of 'y' (line 138)
        y_390803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'y')
        # Getting the type of 'j' (line 138)
        j_390804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 18), 'j')
        # Getting the type of 'j' (line 138)
        j_390805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 20), 'j')
        # Getting the type of 'block_size' (line 138)
        block_size_390806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 22), 'block_size')
        # Applying the binary operator '+' (line 138)
        result_add_390807 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 20), '+', j_390805, block_size_390806)
        
        slice_390808 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 138, 16), j_390804, result_add_390807, None)
        # Storing an element on a container (line 138)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 16), y_390803, (slice_390808, func_call_result_390802))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'y' (line 139)
        y_390809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 19), 'y')
        # Assigning a type to the variable 'stypy_return_type' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'stypy_return_type', y_390809)
        # SSA join for if statement (line 130)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'wrapper(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'wrapper' in the type store
        # Getting the type of 'stypy_return_type' (line 129)
        stypy_return_type_390810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_390810)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'wrapper'
        return stypy_return_type_390810

    # Assigning a type to the variable 'wrapper' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'wrapper', wrapper)
    # Getting the type of 'wrapper' (line 140)
    wrapper_390811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 11), 'wrapper')
    # Assigning a type to the variable 'stypy_return_type' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type', wrapper_390811)
    
    # ################# End of '_blocked_elementwise(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_blocked_elementwise' in the type store
    # Getting the type of 'stypy_return_type' (line 122)
    stypy_return_type_390812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_390812)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_blocked_elementwise'
    return stypy_return_type_390812

# Assigning a type to the variable '_blocked_elementwise' (line 122)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), '_blocked_elementwise', _blocked_elementwise)

@norecursion
def sign_round_up(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sign_round_up'
    module_type_store = module_type_store.open_function_context('sign_round_up', 143, 0, False)
    
    # Passed parameters checking function
    sign_round_up.stypy_localization = localization
    sign_round_up.stypy_type_of_self = None
    sign_round_up.stypy_type_store = module_type_store
    sign_round_up.stypy_function_name = 'sign_round_up'
    sign_round_up.stypy_param_names_list = ['X']
    sign_round_up.stypy_varargs_param_name = None
    sign_round_up.stypy_kwargs_param_name = None
    sign_round_up.stypy_call_defaults = defaults
    sign_round_up.stypy_call_varargs = varargs
    sign_round_up.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sign_round_up', ['X'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sign_round_up', localization, ['X'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sign_round_up(...)' code ##################

    str_390813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, (-1)), 'str', '\n    This should do the right thing for both real and complex matrices.\n\n    From Higham and Tisseur:\n    "Everything in this section remains valid for complex matrices\n    provided that sign(A) is redefined as the matrix (aij / |aij|)\n    (and sign(0) = 1) transposes are replaced by conjugate transposes."\n\n    ')
    
    # Assigning a Call to a Name (line 154):
    
    # Assigning a Call to a Name (line 154):
    
    # Call to copy(...): (line 154)
    # Processing the call keyword arguments (line 154)
    kwargs_390816 = {}
    # Getting the type of 'X' (line 154)
    X_390814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'X', False)
    # Obtaining the member 'copy' of a type (line 154)
    copy_390815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), X_390814, 'copy')
    # Calling copy(args, kwargs) (line 154)
    copy_call_result_390817 = invoke(stypy.reporting.localization.Localization(__file__, 154, 8), copy_390815, *[], **kwargs_390816)
    
    # Assigning a type to the variable 'Y' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'Y', copy_call_result_390817)
    
    # Assigning a Num to a Subscript (line 155):
    
    # Assigning a Num to a Subscript (line 155):
    int_390818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 16), 'int')
    # Getting the type of 'Y' (line 155)
    Y_390819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'Y')
    
    # Getting the type of 'Y' (line 155)
    Y_390820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 6), 'Y')
    int_390821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 11), 'int')
    # Applying the binary operator '==' (line 155)
    result_eq_390822 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 6), '==', Y_390820, int_390821)
    
    # Storing an element on a container (line 155)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 4), Y_390819, (result_eq_390822, int_390818))
    
    # Getting the type of 'Y' (line 156)
    Y_390823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'Y')
    
    # Call to abs(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'Y' (line 156)
    Y_390826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'Y', False)
    # Processing the call keyword arguments (line 156)
    kwargs_390827 = {}
    # Getting the type of 'np' (line 156)
    np_390824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 9), 'np', False)
    # Obtaining the member 'abs' of a type (line 156)
    abs_390825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 9), np_390824, 'abs')
    # Calling abs(args, kwargs) (line 156)
    abs_call_result_390828 = invoke(stypy.reporting.localization.Localization(__file__, 156, 9), abs_390825, *[Y_390826], **kwargs_390827)
    
    # Applying the binary operator 'div=' (line 156)
    result_div_390829 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 4), 'div=', Y_390823, abs_call_result_390828)
    # Assigning a type to the variable 'Y' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'Y', result_div_390829)
    
    # Getting the type of 'Y' (line 157)
    Y_390830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 11), 'Y')
    # Assigning a type to the variable 'stypy_return_type' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'stypy_return_type', Y_390830)
    
    # ################# End of 'sign_round_up(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sign_round_up' in the type store
    # Getting the type of 'stypy_return_type' (line 143)
    stypy_return_type_390831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_390831)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sign_round_up'
    return stypy_return_type_390831

# Assigning a type to the variable 'sign_round_up' (line 143)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 0), 'sign_round_up', sign_round_up)

@norecursion
def _max_abs_axis1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_max_abs_axis1'
    module_type_store = module_type_store.open_function_context('_max_abs_axis1', 160, 0, False)
    
    # Passed parameters checking function
    _max_abs_axis1.stypy_localization = localization
    _max_abs_axis1.stypy_type_of_self = None
    _max_abs_axis1.stypy_type_store = module_type_store
    _max_abs_axis1.stypy_function_name = '_max_abs_axis1'
    _max_abs_axis1.stypy_param_names_list = ['X']
    _max_abs_axis1.stypy_varargs_param_name = None
    _max_abs_axis1.stypy_kwargs_param_name = None
    _max_abs_axis1.stypy_call_defaults = defaults
    _max_abs_axis1.stypy_call_varargs = varargs
    _max_abs_axis1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_max_abs_axis1', ['X'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_max_abs_axis1', localization, ['X'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_max_abs_axis1(...)' code ##################

    
    # Call to max(...): (line 162)
    # Processing the call arguments (line 162)
    
    # Call to abs(...): (line 162)
    # Processing the call arguments (line 162)
    # Getting the type of 'X' (line 162)
    X_390836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 25), 'X', False)
    # Processing the call keyword arguments (line 162)
    kwargs_390837 = {}
    # Getting the type of 'np' (line 162)
    np_390834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 18), 'np', False)
    # Obtaining the member 'abs' of a type (line 162)
    abs_390835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 18), np_390834, 'abs')
    # Calling abs(args, kwargs) (line 162)
    abs_call_result_390838 = invoke(stypy.reporting.localization.Localization(__file__, 162, 18), abs_390835, *[X_390836], **kwargs_390837)
    
    # Processing the call keyword arguments (line 162)
    int_390839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 34), 'int')
    keyword_390840 = int_390839
    kwargs_390841 = {'axis': keyword_390840}
    # Getting the type of 'np' (line 162)
    np_390832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 11), 'np', False)
    # Obtaining the member 'max' of a type (line 162)
    max_390833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 11), np_390832, 'max')
    # Calling max(args, kwargs) (line 162)
    max_call_result_390842 = invoke(stypy.reporting.localization.Localization(__file__, 162, 11), max_390833, *[abs_call_result_390838], **kwargs_390841)
    
    # Assigning a type to the variable 'stypy_return_type' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'stypy_return_type', max_call_result_390842)
    
    # ################# End of '_max_abs_axis1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_max_abs_axis1' in the type store
    # Getting the type of 'stypy_return_type' (line 160)
    stypy_return_type_390843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_390843)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_max_abs_axis1'
    return stypy_return_type_390843

# Assigning a type to the variable '_max_abs_axis1' (line 160)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), '_max_abs_axis1', _max_abs_axis1)

@norecursion
def _sum_abs_axis0(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_sum_abs_axis0'
    module_type_store = module_type_store.open_function_context('_sum_abs_axis0', 165, 0, False)
    
    # Passed parameters checking function
    _sum_abs_axis0.stypy_localization = localization
    _sum_abs_axis0.stypy_type_of_self = None
    _sum_abs_axis0.stypy_type_store = module_type_store
    _sum_abs_axis0.stypy_function_name = '_sum_abs_axis0'
    _sum_abs_axis0.stypy_param_names_list = ['X']
    _sum_abs_axis0.stypy_varargs_param_name = None
    _sum_abs_axis0.stypy_kwargs_param_name = None
    _sum_abs_axis0.stypy_call_defaults = defaults
    _sum_abs_axis0.stypy_call_varargs = varargs
    _sum_abs_axis0.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_sum_abs_axis0', ['X'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_sum_abs_axis0', localization, ['X'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_sum_abs_axis0(...)' code ##################

    
    # Assigning a BinOp to a Name (line 166):
    
    # Assigning a BinOp to a Name (line 166):
    int_390844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 17), 'int')
    int_390845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 20), 'int')
    # Applying the binary operator '**' (line 166)
    result_pow_390846 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 17), '**', int_390844, int_390845)
    
    # Assigning a type to the variable 'block_size' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'block_size', result_pow_390846)
    
    # Assigning a Name to a Name (line 167):
    
    # Assigning a Name to a Name (line 167):
    # Getting the type of 'None' (line 167)
    None_390847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'None')
    # Assigning a type to the variable 'r' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'r', None_390847)
    
    
    # Call to range(...): (line 168)
    # Processing the call arguments (line 168)
    int_390849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 19), 'int')
    
    # Obtaining the type of the subscript
    int_390850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 30), 'int')
    # Getting the type of 'X' (line 168)
    X_390851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 22), 'X', False)
    # Obtaining the member 'shape' of a type (line 168)
    shape_390852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 22), X_390851, 'shape')
    # Obtaining the member '__getitem__' of a type (line 168)
    getitem___390853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 22), shape_390852, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 168)
    subscript_call_result_390854 = invoke(stypy.reporting.localization.Localization(__file__, 168, 22), getitem___390853, int_390850)
    
    # Getting the type of 'block_size' (line 168)
    block_size_390855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 34), 'block_size', False)
    # Processing the call keyword arguments (line 168)
    kwargs_390856 = {}
    # Getting the type of 'range' (line 168)
    range_390848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 13), 'range', False)
    # Calling range(args, kwargs) (line 168)
    range_call_result_390857 = invoke(stypy.reporting.localization.Localization(__file__, 168, 13), range_390848, *[int_390849, subscript_call_result_390854, block_size_390855], **kwargs_390856)
    
    # Testing the type of a for loop iterable (line 168)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 168, 4), range_call_result_390857)
    # Getting the type of the for loop variable (line 168)
    for_loop_var_390858 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 168, 4), range_call_result_390857)
    # Assigning a type to the variable 'j' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'j', for_loop_var_390858)
    # SSA begins for a for statement (line 168)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 169):
    
    # Assigning a Call to a Name (line 169):
    
    # Call to sum(...): (line 169)
    # Processing the call arguments (line 169)
    
    # Call to abs(...): (line 169)
    # Processing the call arguments (line 169)
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 169)
    j_390863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 28), 'j', False)
    # Getting the type of 'j' (line 169)
    j_390864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 30), 'j', False)
    # Getting the type of 'block_size' (line 169)
    block_size_390865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 32), 'block_size', False)
    # Applying the binary operator '+' (line 169)
    result_add_390866 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 30), '+', j_390864, block_size_390865)
    
    slice_390867 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 169, 26), j_390863, result_add_390866, None)
    # Getting the type of 'X' (line 169)
    X_390868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 26), 'X', False)
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___390869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 26), X_390868, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_390870 = invoke(stypy.reporting.localization.Localization(__file__, 169, 26), getitem___390869, slice_390867)
    
    # Processing the call keyword arguments (line 169)
    kwargs_390871 = {}
    # Getting the type of 'np' (line 169)
    np_390861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 19), 'np', False)
    # Obtaining the member 'abs' of a type (line 169)
    abs_390862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 19), np_390861, 'abs')
    # Calling abs(args, kwargs) (line 169)
    abs_call_result_390872 = invoke(stypy.reporting.localization.Localization(__file__, 169, 19), abs_390862, *[subscript_call_result_390870], **kwargs_390871)
    
    # Processing the call keyword arguments (line 169)
    int_390873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 51), 'int')
    keyword_390874 = int_390873
    kwargs_390875 = {'axis': keyword_390874}
    # Getting the type of 'np' (line 169)
    np_390859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'np', False)
    # Obtaining the member 'sum' of a type (line 169)
    sum_390860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 12), np_390859, 'sum')
    # Calling sum(args, kwargs) (line 169)
    sum_call_result_390876 = invoke(stypy.reporting.localization.Localization(__file__, 169, 12), sum_390860, *[abs_call_result_390872], **kwargs_390875)
    
    # Assigning a type to the variable 'y' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'y', sum_call_result_390876)
    
    # Type idiom detected: calculating its left and rigth part (line 170)
    # Getting the type of 'r' (line 170)
    r_390877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 11), 'r')
    # Getting the type of 'None' (line 170)
    None_390878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'None')
    
    (may_be_390879, more_types_in_union_390880) = may_be_none(r_390877, None_390878)

    if may_be_390879:

        if more_types_in_union_390880:
            # Runtime conditional SSA (line 170)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 171):
        
        # Assigning a Name to a Name (line 171):
        # Getting the type of 'y' (line 171)
        y_390881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'y')
        # Assigning a type to the variable 'r' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'r', y_390881)

        if more_types_in_union_390880:
            # Runtime conditional SSA for else branch (line 170)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_390879) or more_types_in_union_390880):
        
        # Getting the type of 'r' (line 173)
        r_390882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'r')
        # Getting the type of 'y' (line 173)
        y_390883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 17), 'y')
        # Applying the binary operator '+=' (line 173)
        result_iadd_390884 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 12), '+=', r_390882, y_390883)
        # Assigning a type to the variable 'r' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'r', result_iadd_390884)
        

        if (may_be_390879 and more_types_in_union_390880):
            # SSA join for if statement (line 170)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'r' (line 174)
    r_390885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 11), 'r')
    # Assigning a type to the variable 'stypy_return_type' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type', r_390885)
    
    # ################# End of '_sum_abs_axis0(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_sum_abs_axis0' in the type store
    # Getting the type of 'stypy_return_type' (line 165)
    stypy_return_type_390886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_390886)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_sum_abs_axis0'
    return stypy_return_type_390886

# Assigning a type to the variable '_sum_abs_axis0' (line 165)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 0), '_sum_abs_axis0', _sum_abs_axis0)

@norecursion
def elementary_vector(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'elementary_vector'
    module_type_store = module_type_store.open_function_context('elementary_vector', 177, 0, False)
    
    # Passed parameters checking function
    elementary_vector.stypy_localization = localization
    elementary_vector.stypy_type_of_self = None
    elementary_vector.stypy_type_store = module_type_store
    elementary_vector.stypy_function_name = 'elementary_vector'
    elementary_vector.stypy_param_names_list = ['n', 'i']
    elementary_vector.stypy_varargs_param_name = None
    elementary_vector.stypy_kwargs_param_name = None
    elementary_vector.stypy_call_defaults = defaults
    elementary_vector.stypy_call_varargs = varargs
    elementary_vector.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'elementary_vector', ['n', 'i'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'elementary_vector', localization, ['n', 'i'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'elementary_vector(...)' code ##################

    
    # Assigning a Call to a Name (line 178):
    
    # Assigning a Call to a Name (line 178):
    
    # Call to zeros(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'n' (line 178)
    n_390889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 17), 'n', False)
    # Processing the call keyword arguments (line 178)
    # Getting the type of 'float' (line 178)
    float_390890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 26), 'float', False)
    keyword_390891 = float_390890
    kwargs_390892 = {'dtype': keyword_390891}
    # Getting the type of 'np' (line 178)
    np_390887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 178)
    zeros_390888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), np_390887, 'zeros')
    # Calling zeros(args, kwargs) (line 178)
    zeros_call_result_390893 = invoke(stypy.reporting.localization.Localization(__file__, 178, 8), zeros_390888, *[n_390889], **kwargs_390892)
    
    # Assigning a type to the variable 'v' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'v', zeros_call_result_390893)
    
    # Assigning a Num to a Subscript (line 179):
    
    # Assigning a Num to a Subscript (line 179):
    int_390894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 11), 'int')
    # Getting the type of 'v' (line 179)
    v_390895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'v')
    # Getting the type of 'i' (line 179)
    i_390896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 6), 'i')
    # Storing an element on a container (line 179)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 4), v_390895, (i_390896, int_390894))
    # Getting the type of 'v' (line 180)
    v_390897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 11), 'v')
    # Assigning a type to the variable 'stypy_return_type' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type', v_390897)
    
    # ################# End of 'elementary_vector(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'elementary_vector' in the type store
    # Getting the type of 'stypy_return_type' (line 177)
    stypy_return_type_390898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_390898)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'elementary_vector'
    return stypy_return_type_390898

# Assigning a type to the variable 'elementary_vector' (line 177)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'elementary_vector', elementary_vector)

@norecursion
def vectors_are_parallel(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'vectors_are_parallel'
    module_type_store = module_type_store.open_function_context('vectors_are_parallel', 183, 0, False)
    
    # Passed parameters checking function
    vectors_are_parallel.stypy_localization = localization
    vectors_are_parallel.stypy_type_of_self = None
    vectors_are_parallel.stypy_type_store = module_type_store
    vectors_are_parallel.stypy_function_name = 'vectors_are_parallel'
    vectors_are_parallel.stypy_param_names_list = ['v', 'w']
    vectors_are_parallel.stypy_varargs_param_name = None
    vectors_are_parallel.stypy_kwargs_param_name = None
    vectors_are_parallel.stypy_call_defaults = defaults
    vectors_are_parallel.stypy_call_varargs = varargs
    vectors_are_parallel.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'vectors_are_parallel', ['v', 'w'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'vectors_are_parallel', localization, ['v', 'w'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'vectors_are_parallel(...)' code ##################

    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'v' (line 187)
    v_390899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 7), 'v')
    # Obtaining the member 'ndim' of a type (line 187)
    ndim_390900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 7), v_390899, 'ndim')
    int_390901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 17), 'int')
    # Applying the binary operator '!=' (line 187)
    result_ne_390902 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 7), '!=', ndim_390900, int_390901)
    
    
    # Getting the type of 'v' (line 187)
    v_390903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 22), 'v')
    # Obtaining the member 'shape' of a type (line 187)
    shape_390904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 22), v_390903, 'shape')
    # Getting the type of 'w' (line 187)
    w_390905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 33), 'w')
    # Obtaining the member 'shape' of a type (line 187)
    shape_390906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 33), w_390905, 'shape')
    # Applying the binary operator '!=' (line 187)
    result_ne_390907 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 22), '!=', shape_390904, shape_390906)
    
    # Applying the binary operator 'or' (line 187)
    result_or_keyword_390908 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 7), 'or', result_ne_390902, result_ne_390907)
    
    # Testing the type of an if condition (line 187)
    if_condition_390909 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 187, 4), result_or_keyword_390908)
    # Assigning a type to the variable 'if_condition_390909' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'if_condition_390909', if_condition_390909)
    # SSA begins for if statement (line 187)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 188)
    # Processing the call arguments (line 188)
    str_390911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 25), 'str', 'expected conformant vectors with entries in {-1,1}')
    # Processing the call keyword arguments (line 188)
    kwargs_390912 = {}
    # Getting the type of 'ValueError' (line 188)
    ValueError_390910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 188)
    ValueError_call_result_390913 = invoke(stypy.reporting.localization.Localization(__file__, 188, 14), ValueError_390910, *[str_390911], **kwargs_390912)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 188, 8), ValueError_call_result_390913, 'raise parameter', BaseException)
    # SSA join for if statement (line 187)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 189):
    
    # Assigning a Subscript to a Name (line 189):
    
    # Obtaining the type of the subscript
    int_390914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 16), 'int')
    # Getting the type of 'v' (line 189)
    v_390915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'v')
    # Obtaining the member 'shape' of a type (line 189)
    shape_390916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), v_390915, 'shape')
    # Obtaining the member '__getitem__' of a type (line 189)
    getitem___390917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), shape_390916, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 189)
    subscript_call_result_390918 = invoke(stypy.reporting.localization.Localization(__file__, 189, 8), getitem___390917, int_390914)
    
    # Assigning a type to the variable 'n' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'n', subscript_call_result_390918)
    
    
    # Call to dot(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'v' (line 190)
    v_390921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 18), 'v', False)
    # Getting the type of 'w' (line 190)
    w_390922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 21), 'w', False)
    # Processing the call keyword arguments (line 190)
    kwargs_390923 = {}
    # Getting the type of 'np' (line 190)
    np_390919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 11), 'np', False)
    # Obtaining the member 'dot' of a type (line 190)
    dot_390920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 11), np_390919, 'dot')
    # Calling dot(args, kwargs) (line 190)
    dot_call_result_390924 = invoke(stypy.reporting.localization.Localization(__file__, 190, 11), dot_390920, *[v_390921, w_390922], **kwargs_390923)
    
    # Getting the type of 'n' (line 190)
    n_390925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 27), 'n')
    # Applying the binary operator '==' (line 190)
    result_eq_390926 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 11), '==', dot_call_result_390924, n_390925)
    
    # Assigning a type to the variable 'stypy_return_type' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'stypy_return_type', result_eq_390926)
    
    # ################# End of 'vectors_are_parallel(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'vectors_are_parallel' in the type store
    # Getting the type of 'stypy_return_type' (line 183)
    stypy_return_type_390927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_390927)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'vectors_are_parallel'
    return stypy_return_type_390927

# Assigning a type to the variable 'vectors_are_parallel' (line 183)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'vectors_are_parallel', vectors_are_parallel)

@norecursion
def every_col_of_X_is_parallel_to_a_col_of_Y(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'every_col_of_X_is_parallel_to_a_col_of_Y'
    module_type_store = module_type_store.open_function_context('every_col_of_X_is_parallel_to_a_col_of_Y', 193, 0, False)
    
    # Passed parameters checking function
    every_col_of_X_is_parallel_to_a_col_of_Y.stypy_localization = localization
    every_col_of_X_is_parallel_to_a_col_of_Y.stypy_type_of_self = None
    every_col_of_X_is_parallel_to_a_col_of_Y.stypy_type_store = module_type_store
    every_col_of_X_is_parallel_to_a_col_of_Y.stypy_function_name = 'every_col_of_X_is_parallel_to_a_col_of_Y'
    every_col_of_X_is_parallel_to_a_col_of_Y.stypy_param_names_list = ['X', 'Y']
    every_col_of_X_is_parallel_to_a_col_of_Y.stypy_varargs_param_name = None
    every_col_of_X_is_parallel_to_a_col_of_Y.stypy_kwargs_param_name = None
    every_col_of_X_is_parallel_to_a_col_of_Y.stypy_call_defaults = defaults
    every_col_of_X_is_parallel_to_a_col_of_Y.stypy_call_varargs = varargs
    every_col_of_X_is_parallel_to_a_col_of_Y.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'every_col_of_X_is_parallel_to_a_col_of_Y', ['X', 'Y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'every_col_of_X_is_parallel_to_a_col_of_Y', localization, ['X', 'Y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'every_col_of_X_is_parallel_to_a_col_of_Y(...)' code ##################

    
    # Getting the type of 'X' (line 194)
    X_390928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 13), 'X')
    # Obtaining the member 'T' of a type (line 194)
    T_390929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 13), X_390928, 'T')
    # Testing the type of a for loop iterable (line 194)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 194, 4), T_390929)
    # Getting the type of the for loop variable (line 194)
    for_loop_var_390930 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 194, 4), T_390929)
    # Assigning a type to the variable 'v' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'v', for_loop_var_390930)
    # SSA begins for a for statement (line 194)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Call to any(...): (line 195)
    # Processing the call arguments (line 195)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 195, 19, True)
    # Calculating comprehension expression
    # Getting the type of 'Y' (line 195)
    Y_390937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 55), 'Y', False)
    # Obtaining the member 'T' of a type (line 195)
    T_390938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 55), Y_390937, 'T')
    comprehension_390939 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 19), T_390938)
    # Assigning a type to the variable 'w' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 19), 'w', comprehension_390939)
    
    # Call to vectors_are_parallel(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of 'v' (line 195)
    v_390933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 40), 'v', False)
    # Getting the type of 'w' (line 195)
    w_390934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 43), 'w', False)
    # Processing the call keyword arguments (line 195)
    kwargs_390935 = {}
    # Getting the type of 'vectors_are_parallel' (line 195)
    vectors_are_parallel_390932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 19), 'vectors_are_parallel', False)
    # Calling vectors_are_parallel(args, kwargs) (line 195)
    vectors_are_parallel_call_result_390936 = invoke(stypy.reporting.localization.Localization(__file__, 195, 19), vectors_are_parallel_390932, *[v_390933, w_390934], **kwargs_390935)
    
    list_390940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 19), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 19), list_390940, vectors_are_parallel_call_result_390936)
    # Processing the call keyword arguments (line 195)
    kwargs_390941 = {}
    # Getting the type of 'any' (line 195)
    any_390931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), 'any', False)
    # Calling any(args, kwargs) (line 195)
    any_call_result_390942 = invoke(stypy.reporting.localization.Localization(__file__, 195, 15), any_390931, *[list_390940], **kwargs_390941)
    
    # Applying the 'not' unary operator (line 195)
    result_not__390943 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 11), 'not', any_call_result_390942)
    
    # Testing the type of an if condition (line 195)
    if_condition_390944 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 195, 8), result_not__390943)
    # Assigning a type to the variable 'if_condition_390944' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'if_condition_390944', if_condition_390944)
    # SSA begins for if statement (line 195)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'False' (line 196)
    False_390945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 19), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'stypy_return_type', False_390945)
    # SSA join for if statement (line 195)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'True' (line 197)
    True_390946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'stypy_return_type', True_390946)
    
    # ################# End of 'every_col_of_X_is_parallel_to_a_col_of_Y(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'every_col_of_X_is_parallel_to_a_col_of_Y' in the type store
    # Getting the type of 'stypy_return_type' (line 193)
    stypy_return_type_390947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_390947)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'every_col_of_X_is_parallel_to_a_col_of_Y'
    return stypy_return_type_390947

# Assigning a type to the variable 'every_col_of_X_is_parallel_to_a_col_of_Y' (line 193)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), 'every_col_of_X_is_parallel_to_a_col_of_Y', every_col_of_X_is_parallel_to_a_col_of_Y)

@norecursion
def column_needs_resampling(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 200)
    None_390948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 36), 'None')
    defaults = [None_390948]
    # Create a new context for function 'column_needs_resampling'
    module_type_store = module_type_store.open_function_context('column_needs_resampling', 200, 0, False)
    
    # Passed parameters checking function
    column_needs_resampling.stypy_localization = localization
    column_needs_resampling.stypy_type_of_self = None
    column_needs_resampling.stypy_type_store = module_type_store
    column_needs_resampling.stypy_function_name = 'column_needs_resampling'
    column_needs_resampling.stypy_param_names_list = ['i', 'X', 'Y']
    column_needs_resampling.stypy_varargs_param_name = None
    column_needs_resampling.stypy_kwargs_param_name = None
    column_needs_resampling.stypy_call_defaults = defaults
    column_needs_resampling.stypy_call_varargs = varargs
    column_needs_resampling.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'column_needs_resampling', ['i', 'X', 'Y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'column_needs_resampling', localization, ['i', 'X', 'Y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'column_needs_resampling(...)' code ##################

    
    # Assigning a Attribute to a Tuple (line 204):
    
    # Assigning a Subscript to a Name (line 204):
    
    # Obtaining the type of the subscript
    int_390949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 4), 'int')
    # Getting the type of 'X' (line 204)
    X_390950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 11), 'X')
    # Obtaining the member 'shape' of a type (line 204)
    shape_390951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 11), X_390950, 'shape')
    # Obtaining the member '__getitem__' of a type (line 204)
    getitem___390952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 4), shape_390951, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 204)
    subscript_call_result_390953 = invoke(stypy.reporting.localization.Localization(__file__, 204, 4), getitem___390952, int_390949)
    
    # Assigning a type to the variable 'tuple_var_assignment_390530' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'tuple_var_assignment_390530', subscript_call_result_390953)
    
    # Assigning a Subscript to a Name (line 204):
    
    # Obtaining the type of the subscript
    int_390954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 4), 'int')
    # Getting the type of 'X' (line 204)
    X_390955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 11), 'X')
    # Obtaining the member 'shape' of a type (line 204)
    shape_390956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 11), X_390955, 'shape')
    # Obtaining the member '__getitem__' of a type (line 204)
    getitem___390957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 4), shape_390956, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 204)
    subscript_call_result_390958 = invoke(stypy.reporting.localization.Localization(__file__, 204, 4), getitem___390957, int_390954)
    
    # Assigning a type to the variable 'tuple_var_assignment_390531' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'tuple_var_assignment_390531', subscript_call_result_390958)
    
    # Assigning a Name to a Name (line 204):
    # Getting the type of 'tuple_var_assignment_390530' (line 204)
    tuple_var_assignment_390530_390959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'tuple_var_assignment_390530')
    # Assigning a type to the variable 'n' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'n', tuple_var_assignment_390530_390959)
    
    # Assigning a Name to a Name (line 204):
    # Getting the type of 'tuple_var_assignment_390531' (line 204)
    tuple_var_assignment_390531_390960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'tuple_var_assignment_390531')
    # Assigning a type to the variable 't' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 7), 't', tuple_var_assignment_390531_390960)
    
    # Assigning a Subscript to a Name (line 205):
    
    # Assigning a Subscript to a Name (line 205):
    
    # Obtaining the type of the subscript
    slice_390961 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 205, 8), None, None, None)
    # Getting the type of 'i' (line 205)
    i_390962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 13), 'i')
    # Getting the type of 'X' (line 205)
    X_390963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'X')
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___390964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), X_390963, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_390965 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), getitem___390964, (slice_390961, i_390962))
    
    # Assigning a type to the variable 'v' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'v', subscript_call_result_390965)
    
    
    # Call to any(...): (line 206)
    # Processing the call arguments (line 206)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 206, 11, True)
    # Calculating comprehension expression
    
    # Call to range(...): (line 206)
    # Processing the call arguments (line 206)
    # Getting the type of 'i' (line 206)
    i_390977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 59), 'i', False)
    # Processing the call keyword arguments (line 206)
    kwargs_390978 = {}
    # Getting the type of 'range' (line 206)
    range_390976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 53), 'range', False)
    # Calling range(args, kwargs) (line 206)
    range_call_result_390979 = invoke(stypy.reporting.localization.Localization(__file__, 206, 53), range_390976, *[i_390977], **kwargs_390978)
    
    comprehension_390980 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 11), range_call_result_390979)
    # Assigning a type to the variable 'j' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 11), 'j', comprehension_390980)
    
    # Call to vectors_are_parallel(...): (line 206)
    # Processing the call arguments (line 206)
    # Getting the type of 'v' (line 206)
    v_390968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 32), 'v', False)
    
    # Obtaining the type of the subscript
    slice_390969 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 206, 35), None, None, None)
    # Getting the type of 'j' (line 206)
    j_390970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 40), 'j', False)
    # Getting the type of 'X' (line 206)
    X_390971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 35), 'X', False)
    # Obtaining the member '__getitem__' of a type (line 206)
    getitem___390972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 35), X_390971, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 206)
    subscript_call_result_390973 = invoke(stypy.reporting.localization.Localization(__file__, 206, 35), getitem___390972, (slice_390969, j_390970))
    
    # Processing the call keyword arguments (line 206)
    kwargs_390974 = {}
    # Getting the type of 'vectors_are_parallel' (line 206)
    vectors_are_parallel_390967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 11), 'vectors_are_parallel', False)
    # Calling vectors_are_parallel(args, kwargs) (line 206)
    vectors_are_parallel_call_result_390975 = invoke(stypy.reporting.localization.Localization(__file__, 206, 11), vectors_are_parallel_390967, *[v_390968, subscript_call_result_390973], **kwargs_390974)
    
    list_390981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 11), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 11), list_390981, vectors_are_parallel_call_result_390975)
    # Processing the call keyword arguments (line 206)
    kwargs_390982 = {}
    # Getting the type of 'any' (line 206)
    any_390966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 7), 'any', False)
    # Calling any(args, kwargs) (line 206)
    any_call_result_390983 = invoke(stypy.reporting.localization.Localization(__file__, 206, 7), any_390966, *[list_390981], **kwargs_390982)
    
    # Testing the type of an if condition (line 206)
    if_condition_390984 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 4), any_call_result_390983)
    # Assigning a type to the variable 'if_condition_390984' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'if_condition_390984', if_condition_390984)
    # SSA begins for if statement (line 206)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'True' (line 207)
    True_390985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 15), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'stypy_return_type', True_390985)
    # SSA join for if statement (line 206)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 208)
    # Getting the type of 'Y' (line 208)
    Y_390986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'Y')
    # Getting the type of 'None' (line 208)
    None_390987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 16), 'None')
    
    (may_be_390988, more_types_in_union_390989) = may_not_be_none(Y_390986, None_390987)

    if may_be_390988:

        if more_types_in_union_390989:
            # Runtime conditional SSA (line 208)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Call to any(...): (line 209)
        # Processing the call arguments (line 209)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 209, 15, True)
        # Calculating comprehension expression
        # Getting the type of 'Y' (line 209)
        Y_390996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 51), 'Y', False)
        # Obtaining the member 'T' of a type (line 209)
        T_390997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 51), Y_390996, 'T')
        comprehension_390998 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 15), T_390997)
        # Assigning a type to the variable 'w' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 15), 'w', comprehension_390998)
        
        # Call to vectors_are_parallel(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 'v' (line 209)
        v_390992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 36), 'v', False)
        # Getting the type of 'w' (line 209)
        w_390993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 39), 'w', False)
        # Processing the call keyword arguments (line 209)
        kwargs_390994 = {}
        # Getting the type of 'vectors_are_parallel' (line 209)
        vectors_are_parallel_390991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 15), 'vectors_are_parallel', False)
        # Calling vectors_are_parallel(args, kwargs) (line 209)
        vectors_are_parallel_call_result_390995 = invoke(stypy.reporting.localization.Localization(__file__, 209, 15), vectors_are_parallel_390991, *[v_390992, w_390993], **kwargs_390994)
        
        list_390999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 15), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 15), list_390999, vectors_are_parallel_call_result_390995)
        # Processing the call keyword arguments (line 209)
        kwargs_391000 = {}
        # Getting the type of 'any' (line 209)
        any_390990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 11), 'any', False)
        # Calling any(args, kwargs) (line 209)
        any_call_result_391001 = invoke(stypy.reporting.localization.Localization(__file__, 209, 11), any_390990, *[list_390999], **kwargs_391000)
        
        # Testing the type of an if condition (line 209)
        if_condition_391002 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 209, 8), any_call_result_391001)
        # Assigning a type to the variable 'if_condition_391002' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'if_condition_391002', if_condition_391002)
        # SSA begins for if statement (line 209)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 210)
        True_391003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 19), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'stypy_return_type', True_391003)
        # SSA join for if statement (line 209)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_390989:
            # SSA join for if statement (line 208)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'False' (line 211)
    False_391004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'stypy_return_type', False_391004)
    
    # ################# End of 'column_needs_resampling(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'column_needs_resampling' in the type store
    # Getting the type of 'stypy_return_type' (line 200)
    stypy_return_type_391005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_391005)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'column_needs_resampling'
    return stypy_return_type_391005

# Assigning a type to the variable 'column_needs_resampling' (line 200)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 0), 'column_needs_resampling', column_needs_resampling)

@norecursion
def resample_column(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'resample_column'
    module_type_store = module_type_store.open_function_context('resample_column', 214, 0, False)
    
    # Passed parameters checking function
    resample_column.stypy_localization = localization
    resample_column.stypy_type_of_self = None
    resample_column.stypy_type_store = module_type_store
    resample_column.stypy_function_name = 'resample_column'
    resample_column.stypy_param_names_list = ['i', 'X']
    resample_column.stypy_varargs_param_name = None
    resample_column.stypy_kwargs_param_name = None
    resample_column.stypy_call_defaults = defaults
    resample_column.stypy_call_varargs = varargs
    resample_column.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'resample_column', ['i', 'X'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'resample_column', localization, ['i', 'X'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'resample_column(...)' code ##################

    
    # Assigning a BinOp to a Subscript (line 215):
    
    # Assigning a BinOp to a Subscript (line 215):
    
    # Call to randint(...): (line 215)
    # Processing the call arguments (line 215)
    int_391009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 32), 'int')
    int_391010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 35), 'int')
    # Processing the call keyword arguments (line 215)
    
    # Obtaining the type of the subscript
    int_391011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 51), 'int')
    # Getting the type of 'X' (line 215)
    X_391012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 43), 'X', False)
    # Obtaining the member 'shape' of a type (line 215)
    shape_391013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 43), X_391012, 'shape')
    # Obtaining the member '__getitem__' of a type (line 215)
    getitem___391014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 43), shape_391013, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 215)
    subscript_call_result_391015 = invoke(stypy.reporting.localization.Localization(__file__, 215, 43), getitem___391014, int_391011)
    
    keyword_391016 = subscript_call_result_391015
    kwargs_391017 = {'size': keyword_391016}
    # Getting the type of 'np' (line 215)
    np_391006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 14), 'np', False)
    # Obtaining the member 'random' of a type (line 215)
    random_391007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 14), np_391006, 'random')
    # Obtaining the member 'randint' of a type (line 215)
    randint_391008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 14), random_391007, 'randint')
    # Calling randint(args, kwargs) (line 215)
    randint_call_result_391018 = invoke(stypy.reporting.localization.Localization(__file__, 215, 14), randint_391008, *[int_391009, int_391010], **kwargs_391017)
    
    int_391019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 55), 'int')
    # Applying the binary operator '*' (line 215)
    result_mul_391020 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 14), '*', randint_call_result_391018, int_391019)
    
    int_391021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 59), 'int')
    # Applying the binary operator '-' (line 215)
    result_sub_391022 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 14), '-', result_mul_391020, int_391021)
    
    # Getting the type of 'X' (line 215)
    X_391023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'X')
    slice_391024 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 215, 4), None, None, None)
    # Getting the type of 'i' (line 215)
    i_391025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 9), 'i')
    # Storing an element on a container (line 215)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 4), X_391023, ((slice_391024, i_391025), result_sub_391022))
    
    # ################# End of 'resample_column(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'resample_column' in the type store
    # Getting the type of 'stypy_return_type' (line 214)
    stypy_return_type_391026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_391026)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'resample_column'
    return stypy_return_type_391026

# Assigning a type to the variable 'resample_column' (line 214)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 0), 'resample_column', resample_column)

@norecursion
def less_than_or_close(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'less_than_or_close'
    module_type_store = module_type_store.open_function_context('less_than_or_close', 218, 0, False)
    
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
    
    # Call to allclose(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'a' (line 219)
    a_391029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 23), 'a', False)
    # Getting the type of 'b' (line 219)
    b_391030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 26), 'b', False)
    # Processing the call keyword arguments (line 219)
    kwargs_391031 = {}
    # Getting the type of 'np' (line 219)
    np_391027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 11), 'np', False)
    # Obtaining the member 'allclose' of a type (line 219)
    allclose_391028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 11), np_391027, 'allclose')
    # Calling allclose(args, kwargs) (line 219)
    allclose_call_result_391032 = invoke(stypy.reporting.localization.Localization(__file__, 219, 11), allclose_391028, *[a_391029, b_391030], **kwargs_391031)
    
    
    # Getting the type of 'a' (line 219)
    a_391033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 33), 'a')
    # Getting the type of 'b' (line 219)
    b_391034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 37), 'b')
    # Applying the binary operator '<' (line 219)
    result_lt_391035 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 33), '<', a_391033, b_391034)
    
    # Applying the binary operator 'or' (line 219)
    result_or_keyword_391036 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 11), 'or', allclose_call_result_391032, result_lt_391035)
    
    # Assigning a type to the variable 'stypy_return_type' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'stypy_return_type', result_or_keyword_391036)
    
    # ################# End of 'less_than_or_close(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'less_than_or_close' in the type store
    # Getting the type of 'stypy_return_type' (line 218)
    stypy_return_type_391037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_391037)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'less_than_or_close'
    return stypy_return_type_391037

# Assigning a type to the variable 'less_than_or_close' (line 218)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), 'less_than_or_close', less_than_or_close)

@norecursion
def _algorithm_2_2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_algorithm_2_2'
    module_type_store = module_type_store.open_function_context('_algorithm_2_2', 222, 0, False)
    
    # Passed parameters checking function
    _algorithm_2_2.stypy_localization = localization
    _algorithm_2_2.stypy_type_of_self = None
    _algorithm_2_2.stypy_type_store = module_type_store
    _algorithm_2_2.stypy_function_name = '_algorithm_2_2'
    _algorithm_2_2.stypy_param_names_list = ['A', 'AT', 't']
    _algorithm_2_2.stypy_varargs_param_name = None
    _algorithm_2_2.stypy_kwargs_param_name = None
    _algorithm_2_2.stypy_call_defaults = defaults
    _algorithm_2_2.stypy_call_varargs = varargs
    _algorithm_2_2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_algorithm_2_2', ['A', 'AT', 't'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_algorithm_2_2', localization, ['A', 'AT', 't'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_algorithm_2_2(...)' code ##################

    str_391038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, (-1)), 'str', "\n    This is Algorithm 2.2.\n\n    Parameters\n    ----------\n    A : ndarray or other linear operator\n        A linear operator that can produce matrix products.\n    AT : ndarray or other linear operator\n        The transpose of A.\n    t : int, optional\n        A positive parameter controlling the tradeoff between\n        accuracy versus time and memory usage.\n\n    Returns\n    -------\n    g : sequence\n        A non-negative decreasing vector\n        such that g[j] is a lower bound for the 1-norm\n        of the column of A of jth largest 1-norm.\n        The first entry of this vector is therefore a lower bound\n        on the 1-norm of the linear operator A.\n        This sequence has length t.\n    ind : sequence\n        The ith entry of ind is the index of the column A whose 1-norm\n        is given by g[i].\n        This sequence of indices has length t, and its entries are\n        chosen from range(n), possibly with repetition,\n        where n is the order of the operator A.\n\n    Notes\n    -----\n    This algorithm is mainly for testing.\n    It uses the 'ind' array in a way that is similar to\n    its usage in algorithm 2.4.  This algorithm 2.2 may be easier to test,\n    so it gives a chance of uncovering bugs related to indexing\n    which could have propagated less noticeably to algorithm 2.4.\n\n    ")
    
    # Assigning a Call to a Name (line 261):
    
    # Assigning a Call to a Name (line 261):
    
    # Call to aslinearoperator(...): (line 261)
    # Processing the call arguments (line 261)
    # Getting the type of 'A' (line 261)
    A_391040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 41), 'A', False)
    # Processing the call keyword arguments (line 261)
    kwargs_391041 = {}
    # Getting the type of 'aslinearoperator' (line 261)
    aslinearoperator_391039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 24), 'aslinearoperator', False)
    # Calling aslinearoperator(args, kwargs) (line 261)
    aslinearoperator_call_result_391042 = invoke(stypy.reporting.localization.Localization(__file__, 261, 24), aslinearoperator_391039, *[A_391040], **kwargs_391041)
    
    # Assigning a type to the variable 'A_linear_operator' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'A_linear_operator', aslinearoperator_call_result_391042)
    
    # Assigning a Call to a Name (line 262):
    
    # Assigning a Call to a Name (line 262):
    
    # Call to aslinearoperator(...): (line 262)
    # Processing the call arguments (line 262)
    # Getting the type of 'AT' (line 262)
    AT_391044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 42), 'AT', False)
    # Processing the call keyword arguments (line 262)
    kwargs_391045 = {}
    # Getting the type of 'aslinearoperator' (line 262)
    aslinearoperator_391043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 25), 'aslinearoperator', False)
    # Calling aslinearoperator(args, kwargs) (line 262)
    aslinearoperator_call_result_391046 = invoke(stypy.reporting.localization.Localization(__file__, 262, 25), aslinearoperator_391043, *[AT_391044], **kwargs_391045)
    
    # Assigning a type to the variable 'AT_linear_operator' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'AT_linear_operator', aslinearoperator_call_result_391046)
    
    # Assigning a Subscript to a Name (line 263):
    
    # Assigning a Subscript to a Name (line 263):
    
    # Obtaining the type of the subscript
    int_391047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 32), 'int')
    # Getting the type of 'A_linear_operator' (line 263)
    A_linear_operator_391048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'A_linear_operator')
    # Obtaining the member 'shape' of a type (line 263)
    shape_391049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 8), A_linear_operator_391048, 'shape')
    # Obtaining the member '__getitem__' of a type (line 263)
    getitem___391050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 8), shape_391049, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 263)
    subscript_call_result_391051 = invoke(stypy.reporting.localization.Localization(__file__, 263, 8), getitem___391050, int_391047)
    
    # Assigning a type to the variable 'n' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'n', subscript_call_result_391051)
    
    # Assigning a Call to a Name (line 266):
    
    # Assigning a Call to a Name (line 266):
    
    # Call to ones(...): (line 266)
    # Processing the call arguments (line 266)
    
    # Obtaining an instance of the builtin type 'tuple' (line 266)
    tuple_391054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 266)
    # Adding element type (line 266)
    # Getting the type of 'n' (line 266)
    n_391055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 17), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 17), tuple_391054, n_391055)
    # Adding element type (line 266)
    # Getting the type of 't' (line 266)
    t_391056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 20), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 17), tuple_391054, t_391056)
    
    # Processing the call keyword arguments (line 266)
    kwargs_391057 = {}
    # Getting the type of 'np' (line 266)
    np_391052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'np', False)
    # Obtaining the member 'ones' of a type (line 266)
    ones_391053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 8), np_391052, 'ones')
    # Calling ones(args, kwargs) (line 266)
    ones_call_result_391058 = invoke(stypy.reporting.localization.Localization(__file__, 266, 8), ones_391053, *[tuple_391054], **kwargs_391057)
    
    # Assigning a type to the variable 'X' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'X', ones_call_result_391058)
    
    
    # Getting the type of 't' (line 267)
    t_391059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 7), 't')
    int_391060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 11), 'int')
    # Applying the binary operator '>' (line 267)
    result_gt_391061 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 7), '>', t_391059, int_391060)
    
    # Testing the type of an if condition (line 267)
    if_condition_391062 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 267, 4), result_gt_391061)
    # Assigning a type to the variable 'if_condition_391062' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'if_condition_391062', if_condition_391062)
    # SSA begins for if statement (line 267)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 268):
    
    # Assigning a BinOp to a Subscript (line 268):
    
    # Call to randint(...): (line 268)
    # Processing the call arguments (line 268)
    int_391066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 37), 'int')
    int_391067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 40), 'int')
    # Processing the call keyword arguments (line 268)
    
    # Obtaining an instance of the builtin type 'tuple' (line 268)
    tuple_391068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 268)
    # Adding element type (line 268)
    # Getting the type of 'n' (line 268)
    n_391069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 49), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 49), tuple_391068, n_391069)
    # Adding element type (line 268)
    # Getting the type of 't' (line 268)
    t_391070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 52), 't', False)
    int_391071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 54), 'int')
    # Applying the binary operator '-' (line 268)
    result_sub_391072 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 52), '-', t_391070, int_391071)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 49), tuple_391068, result_sub_391072)
    
    keyword_391073 = tuple_391068
    kwargs_391074 = {'size': keyword_391073}
    # Getting the type of 'np' (line 268)
    np_391063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 19), 'np', False)
    # Obtaining the member 'random' of a type (line 268)
    random_391064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 19), np_391063, 'random')
    # Obtaining the member 'randint' of a type (line 268)
    randint_391065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 19), random_391064, 'randint')
    # Calling randint(args, kwargs) (line 268)
    randint_call_result_391075 = invoke(stypy.reporting.localization.Localization(__file__, 268, 19), randint_391065, *[int_391066, int_391067], **kwargs_391074)
    
    int_391076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 58), 'int')
    # Applying the binary operator '*' (line 268)
    result_mul_391077 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 19), '*', randint_call_result_391075, int_391076)
    
    int_391078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 62), 'int')
    # Applying the binary operator '-' (line 268)
    result_sub_391079 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 19), '-', result_mul_391077, int_391078)
    
    # Getting the type of 'X' (line 268)
    X_391080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'X')
    slice_391081 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 268, 8), None, None, None)
    int_391082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 13), 'int')
    slice_391083 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 268, 8), int_391082, None, None)
    # Storing an element on a container (line 268)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 8), X_391080, ((slice_391081, slice_391083), result_sub_391079))
    # SSA join for if statement (line 267)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'X' (line 269)
    X_391084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'X')
    
    # Call to float(...): (line 269)
    # Processing the call arguments (line 269)
    # Getting the type of 'n' (line 269)
    n_391086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 15), 'n', False)
    # Processing the call keyword arguments (line 269)
    kwargs_391087 = {}
    # Getting the type of 'float' (line 269)
    float_391085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 9), 'float', False)
    # Calling float(args, kwargs) (line 269)
    float_call_result_391088 = invoke(stypy.reporting.localization.Localization(__file__, 269, 9), float_391085, *[n_391086], **kwargs_391087)
    
    # Applying the binary operator 'div=' (line 269)
    result_div_391089 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 4), 'div=', X_391084, float_call_result_391088)
    # Assigning a type to the variable 'X' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'X', result_div_391089)
    
    
    # Assigning a Name to a Name (line 273):
    
    # Assigning a Name to a Name (line 273):
    # Getting the type of 'None' (line 273)
    None_391090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 13), 'None')
    # Assigning a type to the variable 'g_prev' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'g_prev', None_391090)
    
    # Assigning a Name to a Name (line 274):
    
    # Assigning a Name to a Name (line 274):
    # Getting the type of 'None' (line 274)
    None_391091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 13), 'None')
    # Assigning a type to the variable 'h_prev' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'h_prev', None_391091)
    
    # Assigning a Num to a Name (line 275):
    
    # Assigning a Num to a Name (line 275):
    int_391092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 8), 'int')
    # Assigning a type to the variable 'k' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'k', int_391092)
    
    # Assigning a Call to a Name (line 276):
    
    # Assigning a Call to a Name (line 276):
    
    # Call to range(...): (line 276)
    # Processing the call arguments (line 276)
    # Getting the type of 't' (line 276)
    t_391094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 16), 't', False)
    # Processing the call keyword arguments (line 276)
    kwargs_391095 = {}
    # Getting the type of 'range' (line 276)
    range_391093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 10), 'range', False)
    # Calling range(args, kwargs) (line 276)
    range_call_result_391096 = invoke(stypy.reporting.localization.Localization(__file__, 276, 10), range_391093, *[t_391094], **kwargs_391095)
    
    # Assigning a type to the variable 'ind' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'ind', range_call_result_391096)
    
    # Getting the type of 'True' (line 277)
    True_391097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 10), 'True')
    # Testing the type of an if condition (line 277)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 277, 4), True_391097)
    # SSA begins for while statement (line 277)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 278):
    
    # Assigning a Call to a Name (line 278):
    
    # Call to asarray(...): (line 278)
    # Processing the call arguments (line 278)
    
    # Call to matmat(...): (line 278)
    # Processing the call arguments (line 278)
    # Getting the type of 'X' (line 278)
    X_391102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 48), 'X', False)
    # Processing the call keyword arguments (line 278)
    kwargs_391103 = {}
    # Getting the type of 'A_linear_operator' (line 278)
    A_linear_operator_391100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 23), 'A_linear_operator', False)
    # Obtaining the member 'matmat' of a type (line 278)
    matmat_391101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 23), A_linear_operator_391100, 'matmat')
    # Calling matmat(args, kwargs) (line 278)
    matmat_call_result_391104 = invoke(stypy.reporting.localization.Localization(__file__, 278, 23), matmat_391101, *[X_391102], **kwargs_391103)
    
    # Processing the call keyword arguments (line 278)
    kwargs_391105 = {}
    # Getting the type of 'np' (line 278)
    np_391098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 278)
    asarray_391099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 12), np_391098, 'asarray')
    # Calling asarray(args, kwargs) (line 278)
    asarray_call_result_391106 = invoke(stypy.reporting.localization.Localization(__file__, 278, 12), asarray_391099, *[matmat_call_result_391104], **kwargs_391105)
    
    # Assigning a type to the variable 'Y' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'Y', asarray_call_result_391106)
    
    # Assigning a Call to a Name (line 279):
    
    # Assigning a Call to a Name (line 279):
    
    # Call to _sum_abs_axis0(...): (line 279)
    # Processing the call arguments (line 279)
    # Getting the type of 'Y' (line 279)
    Y_391108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 27), 'Y', False)
    # Processing the call keyword arguments (line 279)
    kwargs_391109 = {}
    # Getting the type of '_sum_abs_axis0' (line 279)
    _sum_abs_axis0_391107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), '_sum_abs_axis0', False)
    # Calling _sum_abs_axis0(args, kwargs) (line 279)
    _sum_abs_axis0_call_result_391110 = invoke(stypy.reporting.localization.Localization(__file__, 279, 12), _sum_abs_axis0_391107, *[Y_391108], **kwargs_391109)
    
    # Assigning a type to the variable 'g' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'g', _sum_abs_axis0_call_result_391110)
    
    # Assigning a Call to a Name (line 280):
    
    # Assigning a Call to a Name (line 280):
    
    # Call to argmax(...): (line 280)
    # Processing the call arguments (line 280)
    # Getting the type of 'g' (line 280)
    g_391113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 27), 'g', False)
    # Processing the call keyword arguments (line 280)
    kwargs_391114 = {}
    # Getting the type of 'np' (line 280)
    np_391111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 17), 'np', False)
    # Obtaining the member 'argmax' of a type (line 280)
    argmax_391112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 17), np_391111, 'argmax')
    # Calling argmax(args, kwargs) (line 280)
    argmax_call_result_391115 = invoke(stypy.reporting.localization.Localization(__file__, 280, 17), argmax_391112, *[g_391113], **kwargs_391114)
    
    # Assigning a type to the variable 'best_j' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'best_j', argmax_call_result_391115)
    
    # Call to sort(...): (line 281)
    # Processing the call keyword arguments (line 281)
    kwargs_391118 = {}
    # Getting the type of 'g' (line 281)
    g_391116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'g', False)
    # Obtaining the member 'sort' of a type (line 281)
    sort_391117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 8), g_391116, 'sort')
    # Calling sort(args, kwargs) (line 281)
    sort_call_result_391119 = invoke(stypy.reporting.localization.Localization(__file__, 281, 8), sort_391117, *[], **kwargs_391118)
    
    
    # Assigning a Subscript to a Name (line 282):
    
    # Assigning a Subscript to a Name (line 282):
    
    # Obtaining the type of the subscript
    int_391120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 16), 'int')
    slice_391121 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 282, 12), None, None, int_391120)
    # Getting the type of 'g' (line 282)
    g_391122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'g')
    # Obtaining the member '__getitem__' of a type (line 282)
    getitem___391123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 12), g_391122, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 282)
    subscript_call_result_391124 = invoke(stypy.reporting.localization.Localization(__file__, 282, 12), getitem___391123, slice_391121)
    
    # Assigning a type to the variable 'g' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'g', subscript_call_result_391124)
    
    # Assigning a Call to a Name (line 283):
    
    # Assigning a Call to a Name (line 283):
    
    # Call to sign_round_up(...): (line 283)
    # Processing the call arguments (line 283)
    # Getting the type of 'Y' (line 283)
    Y_391126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 26), 'Y', False)
    # Processing the call keyword arguments (line 283)
    kwargs_391127 = {}
    # Getting the type of 'sign_round_up' (line 283)
    sign_round_up_391125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'sign_round_up', False)
    # Calling sign_round_up(args, kwargs) (line 283)
    sign_round_up_call_result_391128 = invoke(stypy.reporting.localization.Localization(__file__, 283, 12), sign_round_up_391125, *[Y_391126], **kwargs_391127)
    
    # Assigning a type to the variable 'S' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'S', sign_round_up_call_result_391128)
    
    # Assigning a Call to a Name (line 284):
    
    # Assigning a Call to a Name (line 284):
    
    # Call to asarray(...): (line 284)
    # Processing the call arguments (line 284)
    
    # Call to matmat(...): (line 284)
    # Processing the call arguments (line 284)
    # Getting the type of 'S' (line 284)
    S_391133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 49), 'S', False)
    # Processing the call keyword arguments (line 284)
    kwargs_391134 = {}
    # Getting the type of 'AT_linear_operator' (line 284)
    AT_linear_operator_391131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 23), 'AT_linear_operator', False)
    # Obtaining the member 'matmat' of a type (line 284)
    matmat_391132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 23), AT_linear_operator_391131, 'matmat')
    # Calling matmat(args, kwargs) (line 284)
    matmat_call_result_391135 = invoke(stypy.reporting.localization.Localization(__file__, 284, 23), matmat_391132, *[S_391133], **kwargs_391134)
    
    # Processing the call keyword arguments (line 284)
    kwargs_391136 = {}
    # Getting the type of 'np' (line 284)
    np_391129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 284)
    asarray_391130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 12), np_391129, 'asarray')
    # Calling asarray(args, kwargs) (line 284)
    asarray_call_result_391137 = invoke(stypy.reporting.localization.Localization(__file__, 284, 12), asarray_391130, *[matmat_call_result_391135], **kwargs_391136)
    
    # Assigning a type to the variable 'Z' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'Z', asarray_call_result_391137)
    
    # Assigning a Call to a Name (line 285):
    
    # Assigning a Call to a Name (line 285):
    
    # Call to _max_abs_axis1(...): (line 285)
    # Processing the call arguments (line 285)
    # Getting the type of 'Z' (line 285)
    Z_391139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 27), 'Z', False)
    # Processing the call keyword arguments (line 285)
    kwargs_391140 = {}
    # Getting the type of '_max_abs_axis1' (line 285)
    _max_abs_axis1_391138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), '_max_abs_axis1', False)
    # Calling _max_abs_axis1(args, kwargs) (line 285)
    _max_abs_axis1_call_result_391141 = invoke(stypy.reporting.localization.Localization(__file__, 285, 12), _max_abs_axis1_391138, *[Z_391139], **kwargs_391140)
    
    # Assigning a type to the variable 'h' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'h', _max_abs_axis1_call_result_391141)
    
    
    # Getting the type of 'k' (line 295)
    k_391142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 11), 'k')
    int_391143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 16), 'int')
    # Applying the binary operator '>=' (line 295)
    result_ge_391144 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 11), '>=', k_391142, int_391143)
    
    # Testing the type of an if condition (line 295)
    if_condition_391145 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 295, 8), result_ge_391144)
    # Assigning a type to the variable 'if_condition_391145' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'if_condition_391145', if_condition_391145)
    # SSA begins for if statement (line 295)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to less_than_or_close(...): (line 296)
    # Processing the call arguments (line 296)
    
    # Call to max(...): (line 296)
    # Processing the call arguments (line 296)
    # Getting the type of 'h' (line 296)
    h_391148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 38), 'h', False)
    # Processing the call keyword arguments (line 296)
    kwargs_391149 = {}
    # Getting the type of 'max' (line 296)
    max_391147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 34), 'max', False)
    # Calling max(args, kwargs) (line 296)
    max_call_result_391150 = invoke(stypy.reporting.localization.Localization(__file__, 296, 34), max_391147, *[h_391148], **kwargs_391149)
    
    
    # Call to dot(...): (line 296)
    # Processing the call arguments (line 296)
    
    # Obtaining the type of the subscript
    slice_391153 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 296, 49), None, None, None)
    # Getting the type of 'best_j' (line 296)
    best_j_391154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 54), 'best_j', False)
    # Getting the type of 'Z' (line 296)
    Z_391155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 49), 'Z', False)
    # Obtaining the member '__getitem__' of a type (line 296)
    getitem___391156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 49), Z_391155, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 296)
    subscript_call_result_391157 = invoke(stypy.reporting.localization.Localization(__file__, 296, 49), getitem___391156, (slice_391153, best_j_391154))
    
    
    # Obtaining the type of the subscript
    slice_391158 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 296, 63), None, None, None)
    # Getting the type of 'best_j' (line 296)
    best_j_391159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 68), 'best_j', False)
    # Getting the type of 'X' (line 296)
    X_391160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 63), 'X', False)
    # Obtaining the member '__getitem__' of a type (line 296)
    getitem___391161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 63), X_391160, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 296)
    subscript_call_result_391162 = invoke(stypy.reporting.localization.Localization(__file__, 296, 63), getitem___391161, (slice_391158, best_j_391159))
    
    # Processing the call keyword arguments (line 296)
    kwargs_391163 = {}
    # Getting the type of 'np' (line 296)
    np_391151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 42), 'np', False)
    # Obtaining the member 'dot' of a type (line 296)
    dot_391152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 42), np_391151, 'dot')
    # Calling dot(args, kwargs) (line 296)
    dot_call_result_391164 = invoke(stypy.reporting.localization.Localization(__file__, 296, 42), dot_391152, *[subscript_call_result_391157, subscript_call_result_391162], **kwargs_391163)
    
    # Processing the call keyword arguments (line 296)
    kwargs_391165 = {}
    # Getting the type of 'less_than_or_close' (line 296)
    less_than_or_close_391146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 15), 'less_than_or_close', False)
    # Calling less_than_or_close(args, kwargs) (line 296)
    less_than_or_close_call_result_391166 = invoke(stypy.reporting.localization.Localization(__file__, 296, 15), less_than_or_close_391146, *[max_call_result_391150, dot_call_result_391164], **kwargs_391165)
    
    # Testing the type of an if condition (line 296)
    if_condition_391167 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 296, 12), less_than_or_close_call_result_391166)
    # Assigning a type to the variable 'if_condition_391167' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'if_condition_391167', if_condition_391167)
    # SSA begins for if statement (line 296)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 296)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 295)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 298):
    
    # Assigning a Subscript to a Name (line 298):
    
    # Obtaining the type of the subscript
    # Getting the type of 't' (line 298)
    t_391168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 35), 't')
    slice_391169 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 298, 14), None, t_391168, None)
    
    # Obtaining the type of the subscript
    int_391170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 30), 'int')
    slice_391171 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 298, 14), None, None, int_391170)
    
    # Call to argsort(...): (line 298)
    # Processing the call arguments (line 298)
    # Getting the type of 'h' (line 298)
    h_391174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 25), 'h', False)
    # Processing the call keyword arguments (line 298)
    kwargs_391175 = {}
    # Getting the type of 'np' (line 298)
    np_391172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 14), 'np', False)
    # Obtaining the member 'argsort' of a type (line 298)
    argsort_391173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 14), np_391172, 'argsort')
    # Calling argsort(args, kwargs) (line 298)
    argsort_call_result_391176 = invoke(stypy.reporting.localization.Localization(__file__, 298, 14), argsort_391173, *[h_391174], **kwargs_391175)
    
    # Obtaining the member '__getitem__' of a type (line 298)
    getitem___391177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 14), argsort_call_result_391176, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 298)
    subscript_call_result_391178 = invoke(stypy.reporting.localization.Localization(__file__, 298, 14), getitem___391177, slice_391171)
    
    # Obtaining the member '__getitem__' of a type (line 298)
    getitem___391179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 14), subscript_call_result_391178, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 298)
    subscript_call_result_391180 = invoke(stypy.reporting.localization.Localization(__file__, 298, 14), getitem___391179, slice_391169)
    
    # Assigning a type to the variable 'ind' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'ind', subscript_call_result_391180)
    
    # Assigning a Subscript to a Name (line 299):
    
    # Assigning a Subscript to a Name (line 299):
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 299)
    ind_391181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 14), 'ind')
    # Getting the type of 'h' (line 299)
    h_391182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'h')
    # Obtaining the member '__getitem__' of a type (line 299)
    getitem___391183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 12), h_391182, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
    subscript_call_result_391184 = invoke(stypy.reporting.localization.Localization(__file__, 299, 12), getitem___391183, ind_391181)
    
    # Assigning a type to the variable 'h' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'h', subscript_call_result_391184)
    
    
    # Call to range(...): (line 300)
    # Processing the call arguments (line 300)
    # Getting the type of 't' (line 300)
    t_391186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 23), 't', False)
    # Processing the call keyword arguments (line 300)
    kwargs_391187 = {}
    # Getting the type of 'range' (line 300)
    range_391185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 17), 'range', False)
    # Calling range(args, kwargs) (line 300)
    range_call_result_391188 = invoke(stypy.reporting.localization.Localization(__file__, 300, 17), range_391185, *[t_391186], **kwargs_391187)
    
    # Testing the type of a for loop iterable (line 300)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 300, 8), range_call_result_391188)
    # Getting the type of the for loop variable (line 300)
    for_loop_var_391189 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 300, 8), range_call_result_391188)
    # Assigning a type to the variable 'j' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'j', for_loop_var_391189)
    # SSA begins for a for statement (line 300)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 301):
    
    # Assigning a Call to a Subscript (line 301):
    
    # Call to elementary_vector(...): (line 301)
    # Processing the call arguments (line 301)
    # Getting the type of 'n' (line 301)
    n_391191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 40), 'n', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 301)
    j_391192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 47), 'j', False)
    # Getting the type of 'ind' (line 301)
    ind_391193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 43), 'ind', False)
    # Obtaining the member '__getitem__' of a type (line 301)
    getitem___391194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 43), ind_391193, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 301)
    subscript_call_result_391195 = invoke(stypy.reporting.localization.Localization(__file__, 301, 43), getitem___391194, j_391192)
    
    # Processing the call keyword arguments (line 301)
    kwargs_391196 = {}
    # Getting the type of 'elementary_vector' (line 301)
    elementary_vector_391190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 22), 'elementary_vector', False)
    # Calling elementary_vector(args, kwargs) (line 301)
    elementary_vector_call_result_391197 = invoke(stypy.reporting.localization.Localization(__file__, 301, 22), elementary_vector_391190, *[n_391191, subscript_call_result_391195], **kwargs_391196)
    
    # Getting the type of 'X' (line 301)
    X_391198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'X')
    slice_391199 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 301, 12), None, None, None)
    # Getting the type of 'j' (line 301)
    j_391200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 17), 'j')
    # Storing an element on a container (line 301)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), X_391198, ((slice_391199, j_391200), elementary_vector_call_result_391197))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'k' (line 304)
    k_391201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 11), 'k')
    int_391202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 16), 'int')
    # Applying the binary operator '>=' (line 304)
    result_ge_391203 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 11), '>=', k_391201, int_391202)
    
    # Testing the type of an if condition (line 304)
    if_condition_391204 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 304, 8), result_ge_391203)
    # Assigning a type to the variable 'if_condition_391204' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'if_condition_391204', if_condition_391204)
    # SSA begins for if statement (line 304)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Call to less_than_or_close(...): (line 305)
    # Processing the call arguments (line 305)
    
    # Obtaining the type of the subscript
    int_391206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 45), 'int')
    # Getting the type of 'g_prev' (line 305)
    g_prev_391207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 38), 'g_prev', False)
    # Obtaining the member '__getitem__' of a type (line 305)
    getitem___391208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 38), g_prev_391207, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 305)
    subscript_call_result_391209 = invoke(stypy.reporting.localization.Localization(__file__, 305, 38), getitem___391208, int_391206)
    
    
    # Obtaining the type of the subscript
    int_391210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 56), 'int')
    # Getting the type of 'h_prev' (line 305)
    h_prev_391211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 49), 'h_prev', False)
    # Obtaining the member '__getitem__' of a type (line 305)
    getitem___391212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 49), h_prev_391211, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 305)
    subscript_call_result_391213 = invoke(stypy.reporting.localization.Localization(__file__, 305, 49), getitem___391212, int_391210)
    
    # Processing the call keyword arguments (line 305)
    kwargs_391214 = {}
    # Getting the type of 'less_than_or_close' (line 305)
    less_than_or_close_391205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 19), 'less_than_or_close', False)
    # Calling less_than_or_close(args, kwargs) (line 305)
    less_than_or_close_call_result_391215 = invoke(stypy.reporting.localization.Localization(__file__, 305, 19), less_than_or_close_391205, *[subscript_call_result_391209, subscript_call_result_391213], **kwargs_391214)
    
    # Applying the 'not' unary operator (line 305)
    result_not__391216 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 15), 'not', less_than_or_close_call_result_391215)
    
    # Testing the type of an if condition (line 305)
    if_condition_391217 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 12), result_not__391216)
    # Assigning a type to the variable 'if_condition_391217' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'if_condition_391217', if_condition_391217)
    # SSA begins for if statement (line 305)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 306)
    # Processing the call arguments (line 306)
    str_391219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 32), 'str', 'invariant (2.2) is violated')
    # Processing the call keyword arguments (line 306)
    kwargs_391220 = {}
    # Getting the type of 'Exception' (line 306)
    Exception_391218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 22), 'Exception', False)
    # Calling Exception(args, kwargs) (line 306)
    Exception_call_result_391221 = invoke(stypy.reporting.localization.Localization(__file__, 306, 22), Exception_391218, *[str_391219], **kwargs_391220)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 306, 16), Exception_call_result_391221, 'raise parameter', BaseException)
    # SSA join for if statement (line 305)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to less_than_or_close(...): (line 307)
    # Processing the call arguments (line 307)
    
    # Obtaining the type of the subscript
    int_391223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 45), 'int')
    # Getting the type of 'h_prev' (line 307)
    h_prev_391224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 38), 'h_prev', False)
    # Obtaining the member '__getitem__' of a type (line 307)
    getitem___391225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 38), h_prev_391224, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 307)
    subscript_call_result_391226 = invoke(stypy.reporting.localization.Localization(__file__, 307, 38), getitem___391225, int_391223)
    
    
    # Obtaining the type of the subscript
    int_391227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 51), 'int')
    # Getting the type of 'g' (line 307)
    g_391228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 49), 'g', False)
    # Obtaining the member '__getitem__' of a type (line 307)
    getitem___391229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 49), g_391228, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 307)
    subscript_call_result_391230 = invoke(stypy.reporting.localization.Localization(__file__, 307, 49), getitem___391229, int_391227)
    
    # Processing the call keyword arguments (line 307)
    kwargs_391231 = {}
    # Getting the type of 'less_than_or_close' (line 307)
    less_than_or_close_391222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 19), 'less_than_or_close', False)
    # Calling less_than_or_close(args, kwargs) (line 307)
    less_than_or_close_call_result_391232 = invoke(stypy.reporting.localization.Localization(__file__, 307, 19), less_than_or_close_391222, *[subscript_call_result_391226, subscript_call_result_391230], **kwargs_391231)
    
    # Applying the 'not' unary operator (line 307)
    result_not__391233 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 15), 'not', less_than_or_close_call_result_391232)
    
    # Testing the type of an if condition (line 307)
    if_condition_391234 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 307, 12), result_not__391233)
    # Assigning a type to the variable 'if_condition_391234' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'if_condition_391234', if_condition_391234)
    # SSA begins for if statement (line 307)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 308)
    # Processing the call arguments (line 308)
    str_391236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 32), 'str', 'invariant (2.2) is violated')
    # Processing the call keyword arguments (line 308)
    kwargs_391237 = {}
    # Getting the type of 'Exception' (line 308)
    Exception_391235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 22), 'Exception', False)
    # Calling Exception(args, kwargs) (line 308)
    Exception_call_result_391238 = invoke(stypy.reporting.localization.Localization(__file__, 308, 22), Exception_391235, *[str_391236], **kwargs_391237)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 308, 16), Exception_call_result_391238, 'raise parameter', BaseException)
    # SSA join for if statement (line 307)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 304)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'k' (line 311)
    k_391239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 11), 'k')
    int_391240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 16), 'int')
    # Applying the binary operator '>=' (line 311)
    result_ge_391241 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 11), '>=', k_391239, int_391240)
    
    # Testing the type of an if condition (line 311)
    if_condition_391242 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 311, 8), result_ge_391241)
    # Assigning a type to the variable 'if_condition_391242' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'if_condition_391242', if_condition_391242)
    # SSA begins for if statement (line 311)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to range(...): (line 312)
    # Processing the call arguments (line 312)
    # Getting the type of 't' (line 312)
    t_391244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 27), 't', False)
    # Processing the call keyword arguments (line 312)
    kwargs_391245 = {}
    # Getting the type of 'range' (line 312)
    range_391243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 21), 'range', False)
    # Calling range(args, kwargs) (line 312)
    range_call_result_391246 = invoke(stypy.reporting.localization.Localization(__file__, 312, 21), range_391243, *[t_391244], **kwargs_391245)
    
    # Testing the type of a for loop iterable (line 312)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 312, 12), range_call_result_391246)
    # Getting the type of the for loop variable (line 312)
    for_loop_var_391247 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 312, 12), range_call_result_391246)
    # Assigning a type to the variable 'j' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'j', for_loop_var_391247)
    # SSA begins for a for statement (line 312)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Call to less_than_or_close(...): (line 313)
    # Processing the call arguments (line 313)
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 313)
    j_391249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 44), 'j', False)
    # Getting the type of 'g' (line 313)
    g_391250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 42), 'g', False)
    # Obtaining the member '__getitem__' of a type (line 313)
    getitem___391251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 42), g_391250, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 313)
    subscript_call_result_391252 = invoke(stypy.reporting.localization.Localization(__file__, 313, 42), getitem___391251, j_391249)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 313)
    j_391253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 55), 'j', False)
    # Getting the type of 'g_prev' (line 313)
    g_prev_391254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 48), 'g_prev', False)
    # Obtaining the member '__getitem__' of a type (line 313)
    getitem___391255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 48), g_prev_391254, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 313)
    subscript_call_result_391256 = invoke(stypy.reporting.localization.Localization(__file__, 313, 48), getitem___391255, j_391253)
    
    # Processing the call keyword arguments (line 313)
    kwargs_391257 = {}
    # Getting the type of 'less_than_or_close' (line 313)
    less_than_or_close_391248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 23), 'less_than_or_close', False)
    # Calling less_than_or_close(args, kwargs) (line 313)
    less_than_or_close_call_result_391258 = invoke(stypy.reporting.localization.Localization(__file__, 313, 23), less_than_or_close_391248, *[subscript_call_result_391252, subscript_call_result_391256], **kwargs_391257)
    
    # Applying the 'not' unary operator (line 313)
    result_not__391259 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 19), 'not', less_than_or_close_call_result_391258)
    
    # Testing the type of an if condition (line 313)
    if_condition_391260 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 313, 16), result_not__391259)
    # Assigning a type to the variable 'if_condition_391260' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 16), 'if_condition_391260', if_condition_391260)
    # SSA begins for if statement (line 313)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 314)
    # Processing the call arguments (line 314)
    str_391262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 36), 'str', 'invariant (2.3) is violated')
    # Processing the call keyword arguments (line 314)
    kwargs_391263 = {}
    # Getting the type of 'Exception' (line 314)
    Exception_391261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 26), 'Exception', False)
    # Calling Exception(args, kwargs) (line 314)
    Exception_call_result_391264 = invoke(stypy.reporting.localization.Localization(__file__, 314, 26), Exception_391261, *[str_391262], **kwargs_391263)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 314, 20), Exception_call_result_391264, 'raise parameter', BaseException)
    # SSA join for if statement (line 313)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 311)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 317):
    
    # Assigning a Name to a Name (line 317):
    # Getting the type of 'g' (line 317)
    g_391265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 17), 'g')
    # Assigning a type to the variable 'g_prev' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'g_prev', g_391265)
    
    # Assigning a Name to a Name (line 318):
    
    # Assigning a Name to a Name (line 318):
    # Getting the type of 'h' (line 318)
    h_391266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 17), 'h')
    # Assigning a type to the variable 'h_prev' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'h_prev', h_391266)
    
    # Getting the type of 'k' (line 319)
    k_391267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'k')
    int_391268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 13), 'int')
    # Applying the binary operator '+=' (line 319)
    result_iadd_391269 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 8), '+=', k_391267, int_391268)
    # Assigning a type to the variable 'k' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'k', result_iadd_391269)
    
    # SSA join for while statement (line 277)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 322)
    tuple_391270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 322)
    # Adding element type (line 322)
    # Getting the type of 'g' (line 322)
    g_391271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 11), 'g')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 11), tuple_391270, g_391271)
    # Adding element type (line 322)
    # Getting the type of 'ind' (line 322)
    ind_391272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 14), 'ind')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 11), tuple_391270, ind_391272)
    
    # Assigning a type to the variable 'stypy_return_type' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'stypy_return_type', tuple_391270)
    
    # ################# End of '_algorithm_2_2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_algorithm_2_2' in the type store
    # Getting the type of 'stypy_return_type' (line 222)
    stypy_return_type_391273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_391273)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_algorithm_2_2'
    return stypy_return_type_391273

# Assigning a type to the variable '_algorithm_2_2' (line 222)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 0), '_algorithm_2_2', _algorithm_2_2)

@norecursion
def _onenormest_core(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_onenormest_core'
    module_type_store = module_type_store.open_function_context('_onenormest_core', 325, 0, False)
    
    # Passed parameters checking function
    _onenormest_core.stypy_localization = localization
    _onenormest_core.stypy_type_of_self = None
    _onenormest_core.stypy_type_store = module_type_store
    _onenormest_core.stypy_function_name = '_onenormest_core'
    _onenormest_core.stypy_param_names_list = ['A', 'AT', 't', 'itmax']
    _onenormest_core.stypy_varargs_param_name = None
    _onenormest_core.stypy_kwargs_param_name = None
    _onenormest_core.stypy_call_defaults = defaults
    _onenormest_core.stypy_call_varargs = varargs
    _onenormest_core.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_onenormest_core', ['A', 'AT', 't', 'itmax'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_onenormest_core', localization, ['A', 'AT', 't', 'itmax'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_onenormest_core(...)' code ##################

    str_391274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, (-1)), 'str', '\n    Compute a lower bound of the 1-norm of a sparse matrix.\n\n    Parameters\n    ----------\n    A : ndarray or other linear operator\n        A linear operator that can produce matrix products.\n    AT : ndarray or other linear operator\n        The transpose of A.\n    t : int, optional\n        A positive parameter controlling the tradeoff between\n        accuracy versus time and memory usage.\n    itmax : int, optional\n        Use at most this many iterations.\n\n    Returns\n    -------\n    est : float\n        An underestimate of the 1-norm of the sparse matrix.\n    v : ndarray, optional\n        The vector such that ||Av||_1 == est*||v||_1.\n        It can be thought of as an input to the linear operator\n        that gives an output with particularly large norm.\n    w : ndarray, optional\n        The vector Av which has relatively large 1-norm.\n        It can be thought of as an output of the linear operator\n        that is relatively large in norm compared to the input.\n    nmults : int, optional\n        The number of matrix products that were computed.\n    nresamples : int, optional\n        The number of times a parallel column was observed,\n        necessitating a re-randomization of the column.\n\n    Notes\n    -----\n    This is algorithm 2.4.\n\n    ')
    
    # Assigning a Call to a Name (line 366):
    
    # Assigning a Call to a Name (line 366):
    
    # Call to aslinearoperator(...): (line 366)
    # Processing the call arguments (line 366)
    # Getting the type of 'A' (line 366)
    A_391276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 41), 'A', False)
    # Processing the call keyword arguments (line 366)
    kwargs_391277 = {}
    # Getting the type of 'aslinearoperator' (line 366)
    aslinearoperator_391275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 24), 'aslinearoperator', False)
    # Calling aslinearoperator(args, kwargs) (line 366)
    aslinearoperator_call_result_391278 = invoke(stypy.reporting.localization.Localization(__file__, 366, 24), aslinearoperator_391275, *[A_391276], **kwargs_391277)
    
    # Assigning a type to the variable 'A_linear_operator' (line 366)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'A_linear_operator', aslinearoperator_call_result_391278)
    
    # Assigning a Call to a Name (line 367):
    
    # Assigning a Call to a Name (line 367):
    
    # Call to aslinearoperator(...): (line 367)
    # Processing the call arguments (line 367)
    # Getting the type of 'AT' (line 367)
    AT_391280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 42), 'AT', False)
    # Processing the call keyword arguments (line 367)
    kwargs_391281 = {}
    # Getting the type of 'aslinearoperator' (line 367)
    aslinearoperator_391279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 25), 'aslinearoperator', False)
    # Calling aslinearoperator(args, kwargs) (line 367)
    aslinearoperator_call_result_391282 = invoke(stypy.reporting.localization.Localization(__file__, 367, 25), aslinearoperator_391279, *[AT_391280], **kwargs_391281)
    
    # Assigning a type to the variable 'AT_linear_operator' (line 367)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'AT_linear_operator', aslinearoperator_call_result_391282)
    
    
    # Getting the type of 'itmax' (line 368)
    itmax_391283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 7), 'itmax')
    int_391284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 15), 'int')
    # Applying the binary operator '<' (line 368)
    result_lt_391285 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 7), '<', itmax_391283, int_391284)
    
    # Testing the type of an if condition (line 368)
    if_condition_391286 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 368, 4), result_lt_391285)
    # Assigning a type to the variable 'if_condition_391286' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'if_condition_391286', if_condition_391286)
    # SSA begins for if statement (line 368)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 369)
    # Processing the call arguments (line 369)
    str_391288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 25), 'str', 'at least two iterations are required')
    # Processing the call keyword arguments (line 369)
    kwargs_391289 = {}
    # Getting the type of 'ValueError' (line 369)
    ValueError_391287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 369)
    ValueError_call_result_391290 = invoke(stypy.reporting.localization.Localization(__file__, 369, 14), ValueError_391287, *[str_391288], **kwargs_391289)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 369, 8), ValueError_call_result_391290, 'raise parameter', BaseException)
    # SSA join for if statement (line 368)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 't' (line 370)
    t_391291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 7), 't')
    int_391292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 11), 'int')
    # Applying the binary operator '<' (line 370)
    result_lt_391293 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 7), '<', t_391291, int_391292)
    
    # Testing the type of an if condition (line 370)
    if_condition_391294 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 370, 4), result_lt_391293)
    # Assigning a type to the variable 'if_condition_391294' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'if_condition_391294', if_condition_391294)
    # SSA begins for if statement (line 370)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 371)
    # Processing the call arguments (line 371)
    str_391296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 25), 'str', 'at least one column is required')
    # Processing the call keyword arguments (line 371)
    kwargs_391297 = {}
    # Getting the type of 'ValueError' (line 371)
    ValueError_391295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 371)
    ValueError_call_result_391298 = invoke(stypy.reporting.localization.Localization(__file__, 371, 14), ValueError_391295, *[str_391296], **kwargs_391297)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 371, 8), ValueError_call_result_391298, 'raise parameter', BaseException)
    # SSA join for if statement (line 370)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 372):
    
    # Assigning a Subscript to a Name (line 372):
    
    # Obtaining the type of the subscript
    int_391299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 16), 'int')
    # Getting the type of 'A' (line 372)
    A_391300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'A')
    # Obtaining the member 'shape' of a type (line 372)
    shape_391301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 8), A_391300, 'shape')
    # Obtaining the member '__getitem__' of a type (line 372)
    getitem___391302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 8), shape_391301, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 372)
    subscript_call_result_391303 = invoke(stypy.reporting.localization.Localization(__file__, 372, 8), getitem___391302, int_391299)
    
    # Assigning a type to the variable 'n' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'n', subscript_call_result_391303)
    
    
    # Getting the type of 't' (line 373)
    t_391304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 7), 't')
    # Getting the type of 'n' (line 373)
    n_391305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'n')
    # Applying the binary operator '>=' (line 373)
    result_ge_391306 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 7), '>=', t_391304, n_391305)
    
    # Testing the type of an if condition (line 373)
    if_condition_391307 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 373, 4), result_ge_391306)
    # Assigning a type to the variable 'if_condition_391307' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'if_condition_391307', if_condition_391307)
    # SSA begins for if statement (line 373)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 374)
    # Processing the call arguments (line 374)
    str_391309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 25), 'str', 't should be smaller than the order of A')
    # Processing the call keyword arguments (line 374)
    kwargs_391310 = {}
    # Getting the type of 'ValueError' (line 374)
    ValueError_391308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 374)
    ValueError_call_result_391311 = invoke(stypy.reporting.localization.Localization(__file__, 374, 14), ValueError_391308, *[str_391309], **kwargs_391310)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 374, 8), ValueError_call_result_391311, 'raise parameter', BaseException)
    # SSA join for if statement (line 373)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 377):
    
    # Assigning a Num to a Name (line 377):
    int_391312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 13), 'int')
    # Assigning a type to the variable 'nmults' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'nmults', int_391312)
    
    # Assigning a Num to a Name (line 378):
    
    # Assigning a Num to a Name (line 378):
    int_391313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 17), 'int')
    # Assigning a type to the variable 'nresamples' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'nresamples', int_391313)
    
    # Assigning a Call to a Name (line 384):
    
    # Assigning a Call to a Name (line 384):
    
    # Call to ones(...): (line 384)
    # Processing the call arguments (line 384)
    
    # Obtaining an instance of the builtin type 'tuple' (line 384)
    tuple_391316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 384)
    # Adding element type (line 384)
    # Getting the type of 'n' (line 384)
    n_391317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 17), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 17), tuple_391316, n_391317)
    # Adding element type (line 384)
    # Getting the type of 't' (line 384)
    t_391318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 20), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 17), tuple_391316, t_391318)
    
    # Processing the call keyword arguments (line 384)
    # Getting the type of 'float' (line 384)
    float_391319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 30), 'float', False)
    keyword_391320 = float_391319
    kwargs_391321 = {'dtype': keyword_391320}
    # Getting the type of 'np' (line 384)
    np_391314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'np', False)
    # Obtaining the member 'ones' of a type (line 384)
    ones_391315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 8), np_391314, 'ones')
    # Calling ones(args, kwargs) (line 384)
    ones_call_result_391322 = invoke(stypy.reporting.localization.Localization(__file__, 384, 8), ones_391315, *[tuple_391316], **kwargs_391321)
    
    # Assigning a type to the variable 'X' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'X', ones_call_result_391322)
    
    
    # Getting the type of 't' (line 388)
    t_391323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 7), 't')
    int_391324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 11), 'int')
    # Applying the binary operator '>' (line 388)
    result_gt_391325 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 7), '>', t_391323, int_391324)
    
    # Testing the type of an if condition (line 388)
    if_condition_391326 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 388, 4), result_gt_391325)
    # Assigning a type to the variable 'if_condition_391326' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'if_condition_391326', if_condition_391326)
    # SSA begins for if statement (line 388)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to range(...): (line 389)
    # Processing the call arguments (line 389)
    int_391328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 23), 'int')
    # Getting the type of 't' (line 389)
    t_391329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 26), 't', False)
    # Processing the call keyword arguments (line 389)
    kwargs_391330 = {}
    # Getting the type of 'range' (line 389)
    range_391327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 17), 'range', False)
    # Calling range(args, kwargs) (line 389)
    range_call_result_391331 = invoke(stypy.reporting.localization.Localization(__file__, 389, 17), range_391327, *[int_391328, t_391329], **kwargs_391330)
    
    # Testing the type of a for loop iterable (line 389)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 389, 8), range_call_result_391331)
    # Getting the type of the for loop variable (line 389)
    for_loop_var_391332 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 389, 8), range_call_result_391331)
    # Assigning a type to the variable 'i' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'i', for_loop_var_391332)
    # SSA begins for a for statement (line 389)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to resample_column(...): (line 392)
    # Processing the call arguments (line 392)
    # Getting the type of 'i' (line 392)
    i_391334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 28), 'i', False)
    # Getting the type of 'X' (line 392)
    X_391335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 31), 'X', False)
    # Processing the call keyword arguments (line 392)
    kwargs_391336 = {}
    # Getting the type of 'resample_column' (line 392)
    resample_column_391333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'resample_column', False)
    # Calling resample_column(args, kwargs) (line 392)
    resample_column_call_result_391337 = invoke(stypy.reporting.localization.Localization(__file__, 392, 12), resample_column_391333, *[i_391334, X_391335], **kwargs_391336)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 393)
    # Processing the call arguments (line 393)
    # Getting the type of 't' (line 393)
    t_391339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 23), 't', False)
    # Processing the call keyword arguments (line 393)
    kwargs_391340 = {}
    # Getting the type of 'range' (line 393)
    range_391338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 17), 'range', False)
    # Calling range(args, kwargs) (line 393)
    range_call_result_391341 = invoke(stypy.reporting.localization.Localization(__file__, 393, 17), range_391338, *[t_391339], **kwargs_391340)
    
    # Testing the type of a for loop iterable (line 393)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 393, 8), range_call_result_391341)
    # Getting the type of the for loop variable (line 393)
    for_loop_var_391342 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 393, 8), range_call_result_391341)
    # Assigning a type to the variable 'i' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'i', for_loop_var_391342)
    # SSA begins for a for statement (line 393)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to column_needs_resampling(...): (line 394)
    # Processing the call arguments (line 394)
    # Getting the type of 'i' (line 394)
    i_391344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 42), 'i', False)
    # Getting the type of 'X' (line 394)
    X_391345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 45), 'X', False)
    # Processing the call keyword arguments (line 394)
    kwargs_391346 = {}
    # Getting the type of 'column_needs_resampling' (line 394)
    column_needs_resampling_391343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 18), 'column_needs_resampling', False)
    # Calling column_needs_resampling(args, kwargs) (line 394)
    column_needs_resampling_call_result_391347 = invoke(stypy.reporting.localization.Localization(__file__, 394, 18), column_needs_resampling_391343, *[i_391344, X_391345], **kwargs_391346)
    
    # Testing the type of an if condition (line 394)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 394, 12), column_needs_resampling_call_result_391347)
    # SSA begins for while statement (line 394)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Call to resample_column(...): (line 395)
    # Processing the call arguments (line 395)
    # Getting the type of 'i' (line 395)
    i_391349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 32), 'i', False)
    # Getting the type of 'X' (line 395)
    X_391350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 35), 'X', False)
    # Processing the call keyword arguments (line 395)
    kwargs_391351 = {}
    # Getting the type of 'resample_column' (line 395)
    resample_column_391348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 16), 'resample_column', False)
    # Calling resample_column(args, kwargs) (line 395)
    resample_column_call_result_391352 = invoke(stypy.reporting.localization.Localization(__file__, 395, 16), resample_column_391348, *[i_391349, X_391350], **kwargs_391351)
    
    
    # Getting the type of 'nresamples' (line 396)
    nresamples_391353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 16), 'nresamples')
    int_391354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 30), 'int')
    # Applying the binary operator '+=' (line 396)
    result_iadd_391355 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 16), '+=', nresamples_391353, int_391354)
    # Assigning a type to the variable 'nresamples' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 16), 'nresamples', result_iadd_391355)
    
    # SSA join for while statement (line 394)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 388)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'X' (line 398)
    X_391356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'X')
    
    # Call to float(...): (line 398)
    # Processing the call arguments (line 398)
    # Getting the type of 'n' (line 398)
    n_391358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 15), 'n', False)
    # Processing the call keyword arguments (line 398)
    kwargs_391359 = {}
    # Getting the type of 'float' (line 398)
    float_391357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 9), 'float', False)
    # Calling float(args, kwargs) (line 398)
    float_call_result_391360 = invoke(stypy.reporting.localization.Localization(__file__, 398, 9), float_391357, *[n_391358], **kwargs_391359)
    
    # Applying the binary operator 'div=' (line 398)
    result_div_391361 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 4), 'div=', X_391356, float_call_result_391360)
    # Assigning a type to the variable 'X' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'X', result_div_391361)
    
    
    # Assigning a Call to a Name (line 400):
    
    # Assigning a Call to a Name (line 400):
    
    # Call to zeros(...): (line 400)
    # Processing the call arguments (line 400)
    int_391364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 24), 'int')
    # Processing the call keyword arguments (line 400)
    # Getting the type of 'np' (line 400)
    np_391365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 33), 'np', False)
    # Obtaining the member 'intp' of a type (line 400)
    intp_391366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 33), np_391365, 'intp')
    keyword_391367 = intp_391366
    kwargs_391368 = {'dtype': keyword_391367}
    # Getting the type of 'np' (line 400)
    np_391362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 15), 'np', False)
    # Obtaining the member 'zeros' of a type (line 400)
    zeros_391363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 15), np_391362, 'zeros')
    # Calling zeros(args, kwargs) (line 400)
    zeros_call_result_391369 = invoke(stypy.reporting.localization.Localization(__file__, 400, 15), zeros_391363, *[int_391364], **kwargs_391368)
    
    # Assigning a type to the variable 'ind_hist' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'ind_hist', zeros_call_result_391369)
    
    # Assigning a Num to a Name (line 401):
    
    # Assigning a Num to a Name (line 401):
    int_391370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 14), 'int')
    # Assigning a type to the variable 'est_old' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'est_old', int_391370)
    
    # Assigning a Call to a Name (line 402):
    
    # Assigning a Call to a Name (line 402):
    
    # Call to zeros(...): (line 402)
    # Processing the call arguments (line 402)
    
    # Obtaining an instance of the builtin type 'tuple' (line 402)
    tuple_391373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 402)
    # Adding element type (line 402)
    # Getting the type of 'n' (line 402)
    n_391374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 18), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 18), tuple_391373, n_391374)
    # Adding element type (line 402)
    # Getting the type of 't' (line 402)
    t_391375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 21), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 18), tuple_391373, t_391375)
    
    # Processing the call keyword arguments (line 402)
    # Getting the type of 'float' (line 402)
    float_391376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 31), 'float', False)
    keyword_391377 = float_391376
    kwargs_391378 = {'dtype': keyword_391377}
    # Getting the type of 'np' (line 402)
    np_391371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 402)
    zeros_391372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 8), np_391371, 'zeros')
    # Calling zeros(args, kwargs) (line 402)
    zeros_call_result_391379 = invoke(stypy.reporting.localization.Localization(__file__, 402, 8), zeros_391372, *[tuple_391373], **kwargs_391378)
    
    # Assigning a type to the variable 'S' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'S', zeros_call_result_391379)
    
    # Assigning a Num to a Name (line 403):
    
    # Assigning a Num to a Name (line 403):
    int_391380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 8), 'int')
    # Assigning a type to the variable 'k' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'k', int_391380)
    
    # Assigning a Name to a Name (line 404):
    
    # Assigning a Name to a Name (line 404):
    # Getting the type of 'None' (line 404)
    None_391381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 10), 'None')
    # Assigning a type to the variable 'ind' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'ind', None_391381)
    
    # Getting the type of 'True' (line 405)
    True_391382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 10), 'True')
    # Testing the type of an if condition (line 405)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 405, 4), True_391382)
    # SSA begins for while statement (line 405)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 406):
    
    # Assigning a Call to a Name (line 406):
    
    # Call to asarray(...): (line 406)
    # Processing the call arguments (line 406)
    
    # Call to matmat(...): (line 406)
    # Processing the call arguments (line 406)
    # Getting the type of 'X' (line 406)
    X_391387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 48), 'X', False)
    # Processing the call keyword arguments (line 406)
    kwargs_391388 = {}
    # Getting the type of 'A_linear_operator' (line 406)
    A_linear_operator_391385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 23), 'A_linear_operator', False)
    # Obtaining the member 'matmat' of a type (line 406)
    matmat_391386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 23), A_linear_operator_391385, 'matmat')
    # Calling matmat(args, kwargs) (line 406)
    matmat_call_result_391389 = invoke(stypy.reporting.localization.Localization(__file__, 406, 23), matmat_391386, *[X_391387], **kwargs_391388)
    
    # Processing the call keyword arguments (line 406)
    kwargs_391390 = {}
    # Getting the type of 'np' (line 406)
    np_391383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 406)
    asarray_391384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 12), np_391383, 'asarray')
    # Calling asarray(args, kwargs) (line 406)
    asarray_call_result_391391 = invoke(stypy.reporting.localization.Localization(__file__, 406, 12), asarray_391384, *[matmat_call_result_391389], **kwargs_391390)
    
    # Assigning a type to the variable 'Y' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'Y', asarray_call_result_391391)
    
    # Getting the type of 'nmults' (line 407)
    nmults_391392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'nmults')
    int_391393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 18), 'int')
    # Applying the binary operator '+=' (line 407)
    result_iadd_391394 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 8), '+=', nmults_391392, int_391393)
    # Assigning a type to the variable 'nmults' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'nmults', result_iadd_391394)
    
    
    # Assigning a Call to a Name (line 408):
    
    # Assigning a Call to a Name (line 408):
    
    # Call to _sum_abs_axis0(...): (line 408)
    # Processing the call arguments (line 408)
    # Getting the type of 'Y' (line 408)
    Y_391396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 30), 'Y', False)
    # Processing the call keyword arguments (line 408)
    kwargs_391397 = {}
    # Getting the type of '_sum_abs_axis0' (line 408)
    _sum_abs_axis0_391395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 15), '_sum_abs_axis0', False)
    # Calling _sum_abs_axis0(args, kwargs) (line 408)
    _sum_abs_axis0_call_result_391398 = invoke(stypy.reporting.localization.Localization(__file__, 408, 15), _sum_abs_axis0_391395, *[Y_391396], **kwargs_391397)
    
    # Assigning a type to the variable 'mags' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'mags', _sum_abs_axis0_call_result_391398)
    
    # Assigning a Call to a Name (line 409):
    
    # Assigning a Call to a Name (line 409):
    
    # Call to max(...): (line 409)
    # Processing the call arguments (line 409)
    # Getting the type of 'mags' (line 409)
    mags_391401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 21), 'mags', False)
    # Processing the call keyword arguments (line 409)
    kwargs_391402 = {}
    # Getting the type of 'np' (line 409)
    np_391399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 14), 'np', False)
    # Obtaining the member 'max' of a type (line 409)
    max_391400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 14), np_391399, 'max')
    # Calling max(args, kwargs) (line 409)
    max_call_result_391403 = invoke(stypy.reporting.localization.Localization(__file__, 409, 14), max_391400, *[mags_391401], **kwargs_391402)
    
    # Assigning a type to the variable 'est' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'est', max_call_result_391403)
    
    # Assigning a Call to a Name (line 410):
    
    # Assigning a Call to a Name (line 410):
    
    # Call to argmax(...): (line 410)
    # Processing the call arguments (line 410)
    # Getting the type of 'mags' (line 410)
    mags_391406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 27), 'mags', False)
    # Processing the call keyword arguments (line 410)
    kwargs_391407 = {}
    # Getting the type of 'np' (line 410)
    np_391404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 17), 'np', False)
    # Obtaining the member 'argmax' of a type (line 410)
    argmax_391405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 17), np_391404, 'argmax')
    # Calling argmax(args, kwargs) (line 410)
    argmax_call_result_391408 = invoke(stypy.reporting.localization.Localization(__file__, 410, 17), argmax_391405, *[mags_391406], **kwargs_391407)
    
    # Assigning a type to the variable 'best_j' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'best_j', argmax_call_result_391408)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'est' (line 411)
    est_391409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 11), 'est')
    # Getting the type of 'est_old' (line 411)
    est_old_391410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 17), 'est_old')
    # Applying the binary operator '>' (line 411)
    result_gt_391411 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 11), '>', est_391409, est_old_391410)
    
    
    # Getting the type of 'k' (line 411)
    k_391412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 28), 'k')
    int_391413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 33), 'int')
    # Applying the binary operator '==' (line 411)
    result_eq_391414 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 28), '==', k_391412, int_391413)
    
    # Applying the binary operator 'or' (line 411)
    result_or_keyword_391415 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 11), 'or', result_gt_391411, result_eq_391414)
    
    # Testing the type of an if condition (line 411)
    if_condition_391416 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 411, 8), result_or_keyword_391415)
    # Assigning a type to the variable 'if_condition_391416' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'if_condition_391416', if_condition_391416)
    # SSA begins for if statement (line 411)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'k' (line 412)
    k_391417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 15), 'k')
    int_391418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 20), 'int')
    # Applying the binary operator '>=' (line 412)
    result_ge_391419 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 15), '>=', k_391417, int_391418)
    
    # Testing the type of an if condition (line 412)
    if_condition_391420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 412, 12), result_ge_391419)
    # Assigning a type to the variable 'if_condition_391420' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 12), 'if_condition_391420', if_condition_391420)
    # SSA begins for if statement (line 412)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 413):
    
    # Assigning a Subscript to a Name (line 413):
    
    # Obtaining the type of the subscript
    # Getting the type of 'best_j' (line 413)
    best_j_391421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 31), 'best_j')
    # Getting the type of 'ind' (line 413)
    ind_391422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 27), 'ind')
    # Obtaining the member '__getitem__' of a type (line 413)
    getitem___391423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 27), ind_391422, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 413)
    subscript_call_result_391424 = invoke(stypy.reporting.localization.Localization(__file__, 413, 27), getitem___391423, best_j_391421)
    
    # Assigning a type to the variable 'ind_best' (line 413)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 16), 'ind_best', subscript_call_result_391424)
    # SSA join for if statement (line 412)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 414):
    
    # Assigning a Subscript to a Name (line 414):
    
    # Obtaining the type of the subscript
    slice_391425 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 414, 16), None, None, None)
    # Getting the type of 'best_j' (line 414)
    best_j_391426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 21), 'best_j')
    # Getting the type of 'Y' (line 414)
    Y_391427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 16), 'Y')
    # Obtaining the member '__getitem__' of a type (line 414)
    getitem___391428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 16), Y_391427, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 414)
    subscript_call_result_391429 = invoke(stypy.reporting.localization.Localization(__file__, 414, 16), getitem___391428, (slice_391425, best_j_391426))
    
    # Assigning a type to the variable 'w' (line 414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'w', subscript_call_result_391429)
    # SSA join for if statement (line 411)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'k' (line 416)
    k_391430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 11), 'k')
    int_391431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 16), 'int')
    # Applying the binary operator '>=' (line 416)
    result_ge_391432 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 11), '>=', k_391430, int_391431)
    
    
    # Getting the type of 'est' (line 416)
    est_391433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 22), 'est')
    # Getting the type of 'est_old' (line 416)
    est_old_391434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 29), 'est_old')
    # Applying the binary operator '<=' (line 416)
    result_le_391435 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 22), '<=', est_391433, est_old_391434)
    
    # Applying the binary operator 'and' (line 416)
    result_and_keyword_391436 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 11), 'and', result_ge_391432, result_le_391435)
    
    # Testing the type of an if condition (line 416)
    if_condition_391437 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 416, 8), result_and_keyword_391436)
    # Assigning a type to the variable 'if_condition_391437' (line 416)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'if_condition_391437', if_condition_391437)
    # SSA begins for if statement (line 416)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 417):
    
    # Assigning a Name to a Name (line 417):
    # Getting the type of 'est_old' (line 417)
    est_old_391438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 18), 'est_old')
    # Assigning a type to the variable 'est' (line 417)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'est', est_old_391438)
    # SSA join for if statement (line 416)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 419):
    
    # Assigning a Name to a Name (line 419):
    # Getting the type of 'est' (line 419)
    est_391439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 18), 'est')
    # Assigning a type to the variable 'est_old' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'est_old', est_391439)
    
    # Assigning a Name to a Name (line 420):
    
    # Assigning a Name to a Name (line 420):
    # Getting the type of 'S' (line 420)
    S_391440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 16), 'S')
    # Assigning a type to the variable 'S_old' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'S_old', S_391440)
    
    
    # Getting the type of 'k' (line 421)
    k_391441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 11), 'k')
    # Getting the type of 'itmax' (line 421)
    itmax_391442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 15), 'itmax')
    # Applying the binary operator '>' (line 421)
    result_gt_391443 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 11), '>', k_391441, itmax_391442)
    
    # Testing the type of an if condition (line 421)
    if_condition_391444 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 421, 8), result_gt_391443)
    # Assigning a type to the variable 'if_condition_391444' (line 421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'if_condition_391444', if_condition_391444)
    # SSA begins for if statement (line 421)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 421)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 423):
    
    # Assigning a Call to a Name (line 423):
    
    # Call to sign_round_up(...): (line 423)
    # Processing the call arguments (line 423)
    # Getting the type of 'Y' (line 423)
    Y_391446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 26), 'Y', False)
    # Processing the call keyword arguments (line 423)
    kwargs_391447 = {}
    # Getting the type of 'sign_round_up' (line 423)
    sign_round_up_391445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'sign_round_up', False)
    # Calling sign_round_up(args, kwargs) (line 423)
    sign_round_up_call_result_391448 = invoke(stypy.reporting.localization.Localization(__file__, 423, 12), sign_round_up_391445, *[Y_391446], **kwargs_391447)
    
    # Assigning a type to the variable 'S' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'S', sign_round_up_call_result_391448)
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 424, 8), module_type_store, 'Y')
    
    
    # Call to every_col_of_X_is_parallel_to_a_col_of_Y(...): (line 426)
    # Processing the call arguments (line 426)
    # Getting the type of 'S' (line 426)
    S_391450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 52), 'S', False)
    # Getting the type of 'S_old' (line 426)
    S_old_391451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 55), 'S_old', False)
    # Processing the call keyword arguments (line 426)
    kwargs_391452 = {}
    # Getting the type of 'every_col_of_X_is_parallel_to_a_col_of_Y' (line 426)
    every_col_of_X_is_parallel_to_a_col_of_Y_391449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 11), 'every_col_of_X_is_parallel_to_a_col_of_Y', False)
    # Calling every_col_of_X_is_parallel_to_a_col_of_Y(args, kwargs) (line 426)
    every_col_of_X_is_parallel_to_a_col_of_Y_call_result_391453 = invoke(stypy.reporting.localization.Localization(__file__, 426, 11), every_col_of_X_is_parallel_to_a_col_of_Y_391449, *[S_391450, S_old_391451], **kwargs_391452)
    
    # Testing the type of an if condition (line 426)
    if_condition_391454 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 426, 8), every_col_of_X_is_parallel_to_a_col_of_Y_call_result_391453)
    # Assigning a type to the variable 'if_condition_391454' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'if_condition_391454', if_condition_391454)
    # SSA begins for if statement (line 426)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 426)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 't' (line 428)
    t_391455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 11), 't')
    int_391456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 15), 'int')
    # Applying the binary operator '>' (line 428)
    result_gt_391457 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 11), '>', t_391455, int_391456)
    
    # Testing the type of an if condition (line 428)
    if_condition_391458 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 428, 8), result_gt_391457)
    # Assigning a type to the variable 'if_condition_391458' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'if_condition_391458', if_condition_391458)
    # SSA begins for if statement (line 428)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to range(...): (line 431)
    # Processing the call arguments (line 431)
    # Getting the type of 't' (line 431)
    t_391460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 27), 't', False)
    # Processing the call keyword arguments (line 431)
    kwargs_391461 = {}
    # Getting the type of 'range' (line 431)
    range_391459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 21), 'range', False)
    # Calling range(args, kwargs) (line 431)
    range_call_result_391462 = invoke(stypy.reporting.localization.Localization(__file__, 431, 21), range_391459, *[t_391460], **kwargs_391461)
    
    # Testing the type of a for loop iterable (line 431)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 431, 12), range_call_result_391462)
    # Getting the type of the for loop variable (line 431)
    for_loop_var_391463 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 431, 12), range_call_result_391462)
    # Assigning a type to the variable 'i' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 12), 'i', for_loop_var_391463)
    # SSA begins for a for statement (line 431)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to column_needs_resampling(...): (line 432)
    # Processing the call arguments (line 432)
    # Getting the type of 'i' (line 432)
    i_391465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 46), 'i', False)
    # Getting the type of 'S' (line 432)
    S_391466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 49), 'S', False)
    # Getting the type of 'S_old' (line 432)
    S_old_391467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 52), 'S_old', False)
    # Processing the call keyword arguments (line 432)
    kwargs_391468 = {}
    # Getting the type of 'column_needs_resampling' (line 432)
    column_needs_resampling_391464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 22), 'column_needs_resampling', False)
    # Calling column_needs_resampling(args, kwargs) (line 432)
    column_needs_resampling_call_result_391469 = invoke(stypy.reporting.localization.Localization(__file__, 432, 22), column_needs_resampling_391464, *[i_391465, S_391466, S_old_391467], **kwargs_391468)
    
    # Testing the type of an if condition (line 432)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 432, 16), column_needs_resampling_call_result_391469)
    # SSA begins for while statement (line 432)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Call to resample_column(...): (line 433)
    # Processing the call arguments (line 433)
    # Getting the type of 'i' (line 433)
    i_391471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 36), 'i', False)
    # Getting the type of 'S' (line 433)
    S_391472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 39), 'S', False)
    # Processing the call keyword arguments (line 433)
    kwargs_391473 = {}
    # Getting the type of 'resample_column' (line 433)
    resample_column_391470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 20), 'resample_column', False)
    # Calling resample_column(args, kwargs) (line 433)
    resample_column_call_result_391474 = invoke(stypy.reporting.localization.Localization(__file__, 433, 20), resample_column_391470, *[i_391471, S_391472], **kwargs_391473)
    
    
    # Getting the type of 'nresamples' (line 434)
    nresamples_391475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 20), 'nresamples')
    int_391476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 34), 'int')
    # Applying the binary operator '+=' (line 434)
    result_iadd_391477 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 20), '+=', nresamples_391475, int_391476)
    # Assigning a type to the variable 'nresamples' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 20), 'nresamples', result_iadd_391477)
    
    # SSA join for while statement (line 432)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 428)
    module_type_store = module_type_store.join_ssa_context()
    
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 435, 8), module_type_store, 'S_old')
    
    # Assigning a Call to a Name (line 437):
    
    # Assigning a Call to a Name (line 437):
    
    # Call to asarray(...): (line 437)
    # Processing the call arguments (line 437)
    
    # Call to matmat(...): (line 437)
    # Processing the call arguments (line 437)
    # Getting the type of 'S' (line 437)
    S_391482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 49), 'S', False)
    # Processing the call keyword arguments (line 437)
    kwargs_391483 = {}
    # Getting the type of 'AT_linear_operator' (line 437)
    AT_linear_operator_391480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 23), 'AT_linear_operator', False)
    # Obtaining the member 'matmat' of a type (line 437)
    matmat_391481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 23), AT_linear_operator_391480, 'matmat')
    # Calling matmat(args, kwargs) (line 437)
    matmat_call_result_391484 = invoke(stypy.reporting.localization.Localization(__file__, 437, 23), matmat_391481, *[S_391482], **kwargs_391483)
    
    # Processing the call keyword arguments (line 437)
    kwargs_391485 = {}
    # Getting the type of 'np' (line 437)
    np_391478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 437)
    asarray_391479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 12), np_391478, 'asarray')
    # Calling asarray(args, kwargs) (line 437)
    asarray_call_result_391486 = invoke(stypy.reporting.localization.Localization(__file__, 437, 12), asarray_391479, *[matmat_call_result_391484], **kwargs_391485)
    
    # Assigning a type to the variable 'Z' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'Z', asarray_call_result_391486)
    
    # Getting the type of 'nmults' (line 438)
    nmults_391487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'nmults')
    int_391488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 18), 'int')
    # Applying the binary operator '+=' (line 438)
    result_iadd_391489 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 8), '+=', nmults_391487, int_391488)
    # Assigning a type to the variable 'nmults' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'nmults', result_iadd_391489)
    
    
    # Assigning a Call to a Name (line 439):
    
    # Assigning a Call to a Name (line 439):
    
    # Call to _max_abs_axis1(...): (line 439)
    # Processing the call arguments (line 439)
    # Getting the type of 'Z' (line 439)
    Z_391491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 27), 'Z', False)
    # Processing the call keyword arguments (line 439)
    kwargs_391492 = {}
    # Getting the type of '_max_abs_axis1' (line 439)
    _max_abs_axis1_391490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 12), '_max_abs_axis1', False)
    # Calling _max_abs_axis1(args, kwargs) (line 439)
    _max_abs_axis1_call_result_391493 = invoke(stypy.reporting.localization.Localization(__file__, 439, 12), _max_abs_axis1_391490, *[Z_391491], **kwargs_391492)
    
    # Assigning a type to the variable 'h' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'h', _max_abs_axis1_call_result_391493)
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 440, 8), module_type_store, 'Z')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'k' (line 442)
    k_391494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 11), 'k')
    int_391495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 16), 'int')
    # Applying the binary operator '>=' (line 442)
    result_ge_391496 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 11), '>=', k_391494, int_391495)
    
    
    
    # Call to max(...): (line 442)
    # Processing the call arguments (line 442)
    # Getting the type of 'h' (line 442)
    h_391498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 26), 'h', False)
    # Processing the call keyword arguments (line 442)
    kwargs_391499 = {}
    # Getting the type of 'max' (line 442)
    max_391497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 22), 'max', False)
    # Calling max(args, kwargs) (line 442)
    max_call_result_391500 = invoke(stypy.reporting.localization.Localization(__file__, 442, 22), max_391497, *[h_391498], **kwargs_391499)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind_best' (line 442)
    ind_best_391501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 34), 'ind_best')
    # Getting the type of 'h' (line 442)
    h_391502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 32), 'h')
    # Obtaining the member '__getitem__' of a type (line 442)
    getitem___391503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 32), h_391502, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 442)
    subscript_call_result_391504 = invoke(stypy.reporting.localization.Localization(__file__, 442, 32), getitem___391503, ind_best_391501)
    
    # Applying the binary operator '==' (line 442)
    result_eq_391505 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 22), '==', max_call_result_391500, subscript_call_result_391504)
    
    # Applying the binary operator 'and' (line 442)
    result_and_keyword_391506 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 11), 'and', result_ge_391496, result_eq_391505)
    
    # Testing the type of an if condition (line 442)
    if_condition_391507 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 442, 8), result_and_keyword_391506)
    # Assigning a type to the variable 'if_condition_391507' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'if_condition_391507', if_condition_391507)
    # SSA begins for if statement (line 442)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 442)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 449):
    
    # Assigning a Call to a Name (line 449):
    
    # Call to copy(...): (line 449)
    # Processing the call keyword arguments (line 449)
    kwargs_391527 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 't' (line 449)
    t_391508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 35), 't', False)
    
    # Call to len(...): (line 449)
    # Processing the call arguments (line 449)
    # Getting the type of 'ind_hist' (line 449)
    ind_hist_391510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 41), 'ind_hist', False)
    # Processing the call keyword arguments (line 449)
    kwargs_391511 = {}
    # Getting the type of 'len' (line 449)
    len_391509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 37), 'len', False)
    # Calling len(args, kwargs) (line 449)
    len_call_result_391512 = invoke(stypy.reporting.localization.Localization(__file__, 449, 37), len_391509, *[ind_hist_391510], **kwargs_391511)
    
    # Applying the binary operator '+' (line 449)
    result_add_391513 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 35), '+', t_391508, len_call_result_391512)
    
    slice_391514 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 449, 14), None, result_add_391513, None)
    
    # Obtaining the type of the subscript
    int_391515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 30), 'int')
    slice_391516 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 449, 14), None, None, int_391515)
    
    # Call to argsort(...): (line 449)
    # Processing the call arguments (line 449)
    # Getting the type of 'h' (line 449)
    h_391519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 25), 'h', False)
    # Processing the call keyword arguments (line 449)
    kwargs_391520 = {}
    # Getting the type of 'np' (line 449)
    np_391517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 14), 'np', False)
    # Obtaining the member 'argsort' of a type (line 449)
    argsort_391518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 14), np_391517, 'argsort')
    # Calling argsort(args, kwargs) (line 449)
    argsort_call_result_391521 = invoke(stypy.reporting.localization.Localization(__file__, 449, 14), argsort_391518, *[h_391519], **kwargs_391520)
    
    # Obtaining the member '__getitem__' of a type (line 449)
    getitem___391522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 14), argsort_call_result_391521, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 449)
    subscript_call_result_391523 = invoke(stypy.reporting.localization.Localization(__file__, 449, 14), getitem___391522, slice_391516)
    
    # Obtaining the member '__getitem__' of a type (line 449)
    getitem___391524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 14), subscript_call_result_391523, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 449)
    subscript_call_result_391525 = invoke(stypy.reporting.localization.Localization(__file__, 449, 14), getitem___391524, slice_391514)
    
    # Obtaining the member 'copy' of a type (line 449)
    copy_391526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 14), subscript_call_result_391525, 'copy')
    # Calling copy(args, kwargs) (line 449)
    copy_call_result_391528 = invoke(stypy.reporting.localization.Localization(__file__, 449, 14), copy_391526, *[], **kwargs_391527)
    
    # Assigning a type to the variable 'ind' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'ind', copy_call_result_391528)
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 450, 8), module_type_store, 'h')
    
    
    # Getting the type of 't' (line 451)
    t_391529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 11), 't')
    int_391530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 15), 'int')
    # Applying the binary operator '>' (line 451)
    result_gt_391531 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 11), '>', t_391529, int_391530)
    
    # Testing the type of an if condition (line 451)
    if_condition_391532 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 451, 8), result_gt_391531)
    # Assigning a type to the variable 'if_condition_391532' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'if_condition_391532', if_condition_391532)
    # SSA begins for if statement (line 451)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to all(...): (line 454)
    # Processing the call keyword arguments (line 454)
    kwargs_391544 = {}
    
    # Call to in1d(...): (line 454)
    # Processing the call arguments (line 454)
    
    # Obtaining the type of the subscript
    # Getting the type of 't' (line 454)
    t_391535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 28), 't', False)
    slice_391536 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 454, 23), None, t_391535, None)
    # Getting the type of 'ind' (line 454)
    ind_391537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 23), 'ind', False)
    # Obtaining the member '__getitem__' of a type (line 454)
    getitem___391538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 23), ind_391537, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 454)
    subscript_call_result_391539 = invoke(stypy.reporting.localization.Localization(__file__, 454, 23), getitem___391538, slice_391536)
    
    # Getting the type of 'ind_hist' (line 454)
    ind_hist_391540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 32), 'ind_hist', False)
    # Processing the call keyword arguments (line 454)
    kwargs_391541 = {}
    # Getting the type of 'np' (line 454)
    np_391533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 15), 'np', False)
    # Obtaining the member 'in1d' of a type (line 454)
    in1d_391534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 15), np_391533, 'in1d')
    # Calling in1d(args, kwargs) (line 454)
    in1d_call_result_391542 = invoke(stypy.reporting.localization.Localization(__file__, 454, 15), in1d_391534, *[subscript_call_result_391539, ind_hist_391540], **kwargs_391541)
    
    # Obtaining the member 'all' of a type (line 454)
    all_391543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 15), in1d_call_result_391542, 'all')
    # Calling all(args, kwargs) (line 454)
    all_call_result_391545 = invoke(stypy.reporting.localization.Localization(__file__, 454, 15), all_391543, *[], **kwargs_391544)
    
    # Testing the type of an if condition (line 454)
    if_condition_391546 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 454, 12), all_call_result_391545)
    # Assigning a type to the variable 'if_condition_391546' (line 454)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 12), 'if_condition_391546', if_condition_391546)
    # SSA begins for if statement (line 454)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 454)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 459):
    
    # Assigning a Call to a Name (line 459):
    
    # Call to in1d(...): (line 459)
    # Processing the call arguments (line 459)
    # Getting the type of 'ind' (line 459)
    ind_391549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 27), 'ind', False)
    # Getting the type of 'ind_hist' (line 459)
    ind_hist_391550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 32), 'ind_hist', False)
    # Processing the call keyword arguments (line 459)
    kwargs_391551 = {}
    # Getting the type of 'np' (line 459)
    np_391547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 19), 'np', False)
    # Obtaining the member 'in1d' of a type (line 459)
    in1d_391548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 19), np_391547, 'in1d')
    # Calling in1d(args, kwargs) (line 459)
    in1d_call_result_391552 = invoke(stypy.reporting.localization.Localization(__file__, 459, 19), in1d_391548, *[ind_391549, ind_hist_391550], **kwargs_391551)
    
    # Assigning a type to the variable 'seen' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'seen', in1d_call_result_391552)
    
    # Assigning a Call to a Name (line 460):
    
    # Assigning a Call to a Name (line 460):
    
    # Call to concatenate(...): (line 460)
    # Processing the call arguments (line 460)
    
    # Obtaining an instance of the builtin type 'tuple' (line 460)
    tuple_391555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 460)
    # Adding element type (line 460)
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'seen' (line 460)
    seen_391556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 39), 'seen', False)
    # Applying the '~' unary operator (line 460)
    result_inv_391557 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 38), '~', seen_391556)
    
    # Getting the type of 'ind' (line 460)
    ind_391558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 34), 'ind', False)
    # Obtaining the member '__getitem__' of a type (line 460)
    getitem___391559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 34), ind_391558, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 460)
    subscript_call_result_391560 = invoke(stypy.reporting.localization.Localization(__file__, 460, 34), getitem___391559, result_inv_391557)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 34), tuple_391555, subscript_call_result_391560)
    # Adding element type (line 460)
    
    # Obtaining the type of the subscript
    # Getting the type of 'seen' (line 460)
    seen_391561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 50), 'seen', False)
    # Getting the type of 'ind' (line 460)
    ind_391562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 46), 'ind', False)
    # Obtaining the member '__getitem__' of a type (line 460)
    getitem___391563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 46), ind_391562, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 460)
    subscript_call_result_391564 = invoke(stypy.reporting.localization.Localization(__file__, 460, 46), getitem___391563, seen_391561)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 34), tuple_391555, subscript_call_result_391564)
    
    # Processing the call keyword arguments (line 460)
    kwargs_391565 = {}
    # Getting the type of 'np' (line 460)
    np_391553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 18), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 460)
    concatenate_391554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 18), np_391553, 'concatenate')
    # Calling concatenate(args, kwargs) (line 460)
    concatenate_call_result_391566 = invoke(stypy.reporting.localization.Localization(__file__, 460, 18), concatenate_391554, *[tuple_391555], **kwargs_391565)
    
    # Assigning a type to the variable 'ind' (line 460)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'ind', concatenate_call_result_391566)
    # SSA join for if statement (line 451)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 461)
    # Processing the call arguments (line 461)
    # Getting the type of 't' (line 461)
    t_391568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 23), 't', False)
    # Processing the call keyword arguments (line 461)
    kwargs_391569 = {}
    # Getting the type of 'range' (line 461)
    range_391567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 17), 'range', False)
    # Calling range(args, kwargs) (line 461)
    range_call_result_391570 = invoke(stypy.reporting.localization.Localization(__file__, 461, 17), range_391567, *[t_391568], **kwargs_391569)
    
    # Testing the type of a for loop iterable (line 461)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 461, 8), range_call_result_391570)
    # Getting the type of the for loop variable (line 461)
    for_loop_var_391571 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 461, 8), range_call_result_391570)
    # Assigning a type to the variable 'j' (line 461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'j', for_loop_var_391571)
    # SSA begins for a for statement (line 461)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 462):
    
    # Assigning a Call to a Subscript (line 462):
    
    # Call to elementary_vector(...): (line 462)
    # Processing the call arguments (line 462)
    # Getting the type of 'n' (line 462)
    n_391573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 40), 'n', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 462)
    j_391574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 47), 'j', False)
    # Getting the type of 'ind' (line 462)
    ind_391575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 43), 'ind', False)
    # Obtaining the member '__getitem__' of a type (line 462)
    getitem___391576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 43), ind_391575, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 462)
    subscript_call_result_391577 = invoke(stypy.reporting.localization.Localization(__file__, 462, 43), getitem___391576, j_391574)
    
    # Processing the call keyword arguments (line 462)
    kwargs_391578 = {}
    # Getting the type of 'elementary_vector' (line 462)
    elementary_vector_391572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 22), 'elementary_vector', False)
    # Calling elementary_vector(args, kwargs) (line 462)
    elementary_vector_call_result_391579 = invoke(stypy.reporting.localization.Localization(__file__, 462, 22), elementary_vector_391572, *[n_391573, subscript_call_result_391577], **kwargs_391578)
    
    # Getting the type of 'X' (line 462)
    X_391580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 12), 'X')
    slice_391581 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 462, 12), None, None, None)
    # Getting the type of 'j' (line 462)
    j_391582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 17), 'j')
    # Storing an element on a container (line 462)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 12), X_391580, ((slice_391581, j_391582), elementary_vector_call_result_391579))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 464):
    
    # Assigning a Subscript to a Name (line 464):
    
    # Obtaining the type of the subscript
    
    
    # Call to in1d(...): (line 464)
    # Processing the call arguments (line 464)
    
    # Obtaining the type of the subscript
    # Getting the type of 't' (line 464)
    t_391585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 40), 't', False)
    slice_391586 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 464, 35), None, t_391585, None)
    # Getting the type of 'ind' (line 464)
    ind_391587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 35), 'ind', False)
    # Obtaining the member '__getitem__' of a type (line 464)
    getitem___391588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 35), ind_391587, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 464)
    subscript_call_result_391589 = invoke(stypy.reporting.localization.Localization(__file__, 464, 35), getitem___391588, slice_391586)
    
    # Getting the type of 'ind_hist' (line 464)
    ind_hist_391590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 44), 'ind_hist', False)
    # Processing the call keyword arguments (line 464)
    kwargs_391591 = {}
    # Getting the type of 'np' (line 464)
    np_391583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 27), 'np', False)
    # Obtaining the member 'in1d' of a type (line 464)
    in1d_391584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 27), np_391583, 'in1d')
    # Calling in1d(args, kwargs) (line 464)
    in1d_call_result_391592 = invoke(stypy.reporting.localization.Localization(__file__, 464, 27), in1d_391584, *[subscript_call_result_391589, ind_hist_391590], **kwargs_391591)
    
    # Applying the '~' unary operator (line 464)
    result_inv_391593 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 26), '~', in1d_call_result_391592)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 't' (line 464)
    t_391594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 23), 't')
    slice_391595 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 464, 18), None, t_391594, None)
    # Getting the type of 'ind' (line 464)
    ind_391596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 18), 'ind')
    # Obtaining the member '__getitem__' of a type (line 464)
    getitem___391597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 18), ind_391596, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 464)
    subscript_call_result_391598 = invoke(stypy.reporting.localization.Localization(__file__, 464, 18), getitem___391597, slice_391595)
    
    # Obtaining the member '__getitem__' of a type (line 464)
    getitem___391599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 18), subscript_call_result_391598, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 464)
    subscript_call_result_391600 = invoke(stypy.reporting.localization.Localization(__file__, 464, 18), getitem___391599, result_inv_391593)
    
    # Assigning a type to the variable 'new_ind' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'new_ind', subscript_call_result_391600)
    
    # Assigning a Call to a Name (line 465):
    
    # Assigning a Call to a Name (line 465):
    
    # Call to concatenate(...): (line 465)
    # Processing the call arguments (line 465)
    
    # Obtaining an instance of the builtin type 'tuple' (line 465)
    tuple_391603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 465)
    # Adding element type (line 465)
    # Getting the type of 'ind_hist' (line 465)
    ind_hist_391604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 35), 'ind_hist', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 35), tuple_391603, ind_hist_391604)
    # Adding element type (line 465)
    # Getting the type of 'new_ind' (line 465)
    new_ind_391605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 45), 'new_ind', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 35), tuple_391603, new_ind_391605)
    
    # Processing the call keyword arguments (line 465)
    kwargs_391606 = {}
    # Getting the type of 'np' (line 465)
    np_391601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 19), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 465)
    concatenate_391602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 19), np_391601, 'concatenate')
    # Calling concatenate(args, kwargs) (line 465)
    concatenate_call_result_391607 = invoke(stypy.reporting.localization.Localization(__file__, 465, 19), concatenate_391602, *[tuple_391603], **kwargs_391606)
    
    # Assigning a type to the variable 'ind_hist' (line 465)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'ind_hist', concatenate_call_result_391607)
    
    # Getting the type of 'k' (line 466)
    k_391608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'k')
    int_391609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 13), 'int')
    # Applying the binary operator '+=' (line 466)
    result_iadd_391610 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 8), '+=', k_391608, int_391609)
    # Assigning a type to the variable 'k' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'k', result_iadd_391610)
    
    # SSA join for while statement (line 405)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 467):
    
    # Assigning a Call to a Name (line 467):
    
    # Call to elementary_vector(...): (line 467)
    # Processing the call arguments (line 467)
    # Getting the type of 'n' (line 467)
    n_391612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 26), 'n', False)
    # Getting the type of 'ind_best' (line 467)
    ind_best_391613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 29), 'ind_best', False)
    # Processing the call keyword arguments (line 467)
    kwargs_391614 = {}
    # Getting the type of 'elementary_vector' (line 467)
    elementary_vector_391611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'elementary_vector', False)
    # Calling elementary_vector(args, kwargs) (line 467)
    elementary_vector_call_result_391615 = invoke(stypy.reporting.localization.Localization(__file__, 467, 8), elementary_vector_391611, *[n_391612, ind_best_391613], **kwargs_391614)
    
    # Assigning a type to the variable 'v' (line 467)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'v', elementary_vector_call_result_391615)
    
    # Obtaining an instance of the builtin type 'tuple' (line 468)
    tuple_391616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 468)
    # Adding element type (line 468)
    # Getting the type of 'est' (line 468)
    est_391617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 11), 'est')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 11), tuple_391616, est_391617)
    # Adding element type (line 468)
    # Getting the type of 'v' (line 468)
    v_391618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 16), 'v')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 11), tuple_391616, v_391618)
    # Adding element type (line 468)
    # Getting the type of 'w' (line 468)
    w_391619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 19), 'w')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 11), tuple_391616, w_391619)
    # Adding element type (line 468)
    # Getting the type of 'nmults' (line 468)
    nmults_391620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 22), 'nmults')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 11), tuple_391616, nmults_391620)
    # Adding element type (line 468)
    # Getting the type of 'nresamples' (line 468)
    nresamples_391621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 30), 'nresamples')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 11), tuple_391616, nresamples_391621)
    
    # Assigning a type to the variable 'stypy_return_type' (line 468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 4), 'stypy_return_type', tuple_391616)
    
    # ################# End of '_onenormest_core(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_onenormest_core' in the type store
    # Getting the type of 'stypy_return_type' (line 325)
    stypy_return_type_391622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_391622)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_onenormest_core'
    return stypy_return_type_391622

# Assigning a type to the variable '_onenormest_core' (line 325)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 0), '_onenormest_core', _onenormest_core)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
