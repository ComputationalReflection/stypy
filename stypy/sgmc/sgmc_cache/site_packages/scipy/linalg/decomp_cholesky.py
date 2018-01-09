
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Cholesky decomposition functions.'''
2: 
3: from __future__ import division, print_function, absolute_import
4: 
5: from numpy import asarray_chkfinite, asarray, atleast_2d
6: 
7: # Local imports
8: from .misc import LinAlgError, _datacopied
9: from .lapack import get_lapack_funcs
10: 
11: __all__ = ['cholesky', 'cho_factor', 'cho_solve', 'cholesky_banded',
12:            'cho_solve_banded']
13: 
14: 
15: def _cholesky(a, lower=False, overwrite_a=False, clean=True,
16:               check_finite=True):
17:     '''Common code for cholesky() and cho_factor().'''
18: 
19:     a1 = asarray_chkfinite(a) if check_finite else asarray(a)
20:     a1 = atleast_2d(a1)
21: 
22:     # Dimension check
23:     if a1.ndim != 2:
24:         raise ValueError('Input array needs to be 2 dimensional but received '
25:                          'a {}d-array.'.format(a1.ndim))
26:     # Squareness check
27:     if a1.shape[0] != a1.shape[1]:
28:         raise ValueError('Input array is expected to be square but has '
29:                          'the shape: {}.'.format(a1.shape))
30: 
31:     # Quick return for square empty array
32:     if a1.size == 0:
33:         return a1.copy(), lower
34: 
35:     overwrite_a = overwrite_a or _datacopied(a1, a)
36:     potrf, = get_lapack_funcs(('potrf',), (a1,))
37:     c, info = potrf(a1, lower=lower, overwrite_a=overwrite_a, clean=clean)
38:     if info > 0:
39:         raise LinAlgError("%d-th leading minor of the array is not positive "
40:                           "definite" % info)
41:     if info < 0:
42:         raise ValueError('LAPACK reported an illegal value in {}-th argument'
43:                          'on entry to "POTRF".'.format(-info))
44:     return c, lower
45: 
46: 
47: def cholesky(a, lower=False, overwrite_a=False, check_finite=True):
48:     '''
49:     Compute the Cholesky decomposition of a matrix.
50: 
51:     Returns the Cholesky decomposition, :math:`A = L L^*` or
52:     :math:`A = U^* U` of a Hermitian positive-definite matrix A.
53: 
54:     Parameters
55:     ----------
56:     a : (M, M) array_like
57:         Matrix to be decomposed
58:     lower : bool, optional
59:         Whether to compute the upper or lower triangular Cholesky
60:         factorization.  Default is upper-triangular.
61:     overwrite_a : bool, optional
62:         Whether to overwrite data in `a` (may improve performance).
63:     check_finite : bool, optional
64:         Whether to check that the input matrix contains only finite numbers.
65:         Disabling may give a performance gain, but may result in problems
66:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
67: 
68:     Returns
69:     -------
70:     c : (M, M) ndarray
71:         Upper- or lower-triangular Cholesky factor of `a`.
72: 
73:     Raises
74:     ------
75:     LinAlgError : if decomposition fails.
76: 
77:     Examples
78:     --------
79:     >>> from scipy import array, linalg, dot
80:     >>> a = array([[1,-2j],[2j,5]])
81:     >>> L = linalg.cholesky(a, lower=True)
82:     >>> L
83:     array([[ 1.+0.j,  0.+0.j],
84:            [ 0.+2.j,  1.+0.j]])
85:     >>> dot(L, L.T.conj())
86:     array([[ 1.+0.j,  0.-2.j],
87:            [ 0.+2.j,  5.+0.j]])
88: 
89:     '''
90:     c, lower = _cholesky(a, lower=lower, overwrite_a=overwrite_a, clean=True,
91:                          check_finite=check_finite)
92:     return c
93: 
94: 
95: def cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
96:     '''
97:     Compute the Cholesky decomposition of a matrix, to use in cho_solve
98: 
99:     Returns a matrix containing the Cholesky decomposition,
100:     ``A = L L*`` or ``A = U* U`` of a Hermitian positive-definite matrix `a`.
101:     The return value can be directly used as the first parameter to cho_solve.
102: 
103:     .. warning::
104:         The returned matrix also contains random data in the entries not
105:         used by the Cholesky decomposition. If you need to zero these
106:         entries, use the function `cholesky` instead.
107: 
108:     Parameters
109:     ----------
110:     a : (M, M) array_like
111:         Matrix to be decomposed
112:     lower : bool, optional
113:         Whether to compute the upper or lower triangular Cholesky factorization
114:         (Default: upper-triangular)
115:     overwrite_a : bool, optional
116:         Whether to overwrite data in a (may improve performance)
117:     check_finite : bool, optional
118:         Whether to check that the input matrix contains only finite numbers.
119:         Disabling may give a performance gain, but may result in problems
120:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
121: 
122:     Returns
123:     -------
124:     c : (M, M) ndarray
125:         Matrix whose upper or lower triangle contains the Cholesky factor
126:         of `a`. Other parts of the matrix contain random data.
127:     lower : bool
128:         Flag indicating whether the factor is in the lower or upper triangle
129: 
130:     Raises
131:     ------
132:     LinAlgError
133:         Raised if decomposition fails.
134: 
135:     See also
136:     --------
137:     cho_solve : Solve a linear set equations using the Cholesky factorization
138:                 of a matrix.
139: 
140:     '''
141:     c, lower = _cholesky(a, lower=lower, overwrite_a=overwrite_a, clean=False,
142:                          check_finite=check_finite)
143:     return c, lower
144: 
145: 
146: def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
147:     '''Solve the linear equations A x = b, given the Cholesky factorization of A.
148: 
149:     Parameters
150:     ----------
151:     (c, lower) : tuple, (array, bool)
152:         Cholesky factorization of a, as given by cho_factor
153:     b : array
154:         Right-hand side
155:     overwrite_b : bool, optional
156:         Whether to overwrite data in b (may improve performance)
157:     check_finite : bool, optional
158:         Whether to check that the input matrices contain only finite numbers.
159:         Disabling may give a performance gain, but may result in problems
160:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
161: 
162:     Returns
163:     -------
164:     x : array
165:         The solution to the system A x = b
166: 
167:     See also
168:     --------
169:     cho_factor : Cholesky factorization of a matrix
170: 
171:     '''
172:     (c, lower) = c_and_lower
173:     if check_finite:
174:         b1 = asarray_chkfinite(b)
175:         c = asarray_chkfinite(c)
176:     else:
177:         b1 = asarray(b)
178:         c = asarray(c)
179:     if c.ndim != 2 or c.shape[0] != c.shape[1]:
180:         raise ValueError("The factored matrix c is not square.")
181:     if c.shape[1] != b1.shape[0]:
182:         raise ValueError("incompatible dimensions.")
183: 
184:     overwrite_b = overwrite_b or _datacopied(b1, b)
185: 
186:     potrs, = get_lapack_funcs(('potrs',), (c, b1))
187:     x, info = potrs(c, b1, lower=lower, overwrite_b=overwrite_b)
188:     if info != 0:
189:         raise ValueError('illegal value in %d-th argument of internal potrs'
190:                          % -info)
191:     return x
192: 
193: 
194: def cholesky_banded(ab, overwrite_ab=False, lower=False, check_finite=True):
195:     '''
196:     Cholesky decompose a banded Hermitian positive-definite matrix
197: 
198:     The matrix a is stored in ab either in lower diagonal or upper
199:     diagonal ordered form::
200: 
201:         ab[u + i - j, j] == a[i,j]        (if upper form; i <= j)
202:         ab[    i - j, j] == a[i,j]        (if lower form; i >= j)
203: 
204:     Example of ab (shape of a is (6,6), u=2)::
205: 
206:         upper form:
207:         *   *   a02 a13 a24 a35
208:         *   a01 a12 a23 a34 a45
209:         a00 a11 a22 a33 a44 a55
210: 
211:         lower form:
212:         a00 a11 a22 a33 a44 a55
213:         a10 a21 a32 a43 a54 *
214:         a20 a31 a42 a53 *   *
215: 
216:     Parameters
217:     ----------
218:     ab : (u + 1, M) array_like
219:         Banded matrix
220:     overwrite_ab : bool, optional
221:         Discard data in ab (may enhance performance)
222:     lower : bool, optional
223:         Is the matrix in the lower form. (Default is upper form)
224:     check_finite : bool, optional
225:         Whether to check that the input matrix contains only finite numbers.
226:         Disabling may give a performance gain, but may result in problems
227:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
228: 
229:     Returns
230:     -------
231:     c : (u + 1, M) ndarray
232:         Cholesky factorization of a, in the same banded format as ab
233: 
234:     '''
235:     if check_finite:
236:         ab = asarray_chkfinite(ab)
237:     else:
238:         ab = asarray(ab)
239: 
240:     pbtrf, = get_lapack_funcs(('pbtrf',), (ab,))
241:     c, info = pbtrf(ab, lower=lower, overwrite_ab=overwrite_ab)
242:     if info > 0:
243:         raise LinAlgError("%d-th leading minor not positive definite" % info)
244:     if info < 0:
245:         raise ValueError('illegal value in %d-th argument of internal pbtrf'
246:                          % -info)
247:     return c
248: 
249: 
250: def cho_solve_banded(cb_and_lower, b, overwrite_b=False, check_finite=True):
251:     '''Solve the linear equations A x = b, given the Cholesky factorization of A.
252: 
253:     Parameters
254:     ----------
255:     (cb, lower) : tuple, (array, bool)
256:         `cb` is the Cholesky factorization of A, as given by cholesky_banded.
257:         `lower` must be the same value that was given to cholesky_banded.
258:     b : array
259:         Right-hand side
260:     overwrite_b : bool, optional
261:         If True, the function will overwrite the values in `b`.
262:     check_finite : bool, optional
263:         Whether to check that the input matrices contain only finite numbers.
264:         Disabling may give a performance gain, but may result in problems
265:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
266: 
267:     Returns
268:     -------
269:     x : array
270:         The solution to the system A x = b
271: 
272:     See also
273:     --------
274:     cholesky_banded : Cholesky factorization of a banded matrix
275: 
276:     Notes
277:     -----
278: 
279:     .. versionadded:: 0.8.0
280: 
281:     '''
282:     (cb, lower) = cb_and_lower
283:     if check_finite:
284:         cb = asarray_chkfinite(cb)
285:         b = asarray_chkfinite(b)
286:     else:
287:         cb = asarray(cb)
288:         b = asarray(b)
289: 
290:     # Validate shapes.
291:     if cb.shape[-1] != b.shape[0]:
292:         raise ValueError("shapes of cb and b are not compatible.")
293: 
294:     pbtrs, = get_lapack_funcs(('pbtrs',), (cb, b))
295:     x, info = pbtrs(cb, b, lower=lower, overwrite_b=overwrite_b)
296:     if info > 0:
297:         raise LinAlgError("%d-th leading minor not positive definite" % info)
298:     if info < 0:
299:         raise ValueError('illegal value in %d-th argument of internal pbtrs'
300:                          % -info)
301:     return x
302: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_17488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Cholesky decomposition functions.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy import asarray_chkfinite, asarray, atleast_2d' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_17489 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_17489) is not StypyTypeError):

    if (import_17489 != 'pyd_module'):
        __import__(import_17489)
        sys_modules_17490 = sys.modules[import_17489]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', sys_modules_17490.module_type_store, module_type_store, ['asarray_chkfinite', 'asarray', 'atleast_2d'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_17490, sys_modules_17490.module_type_store, module_type_store)
    else:
        from numpy import asarray_chkfinite, asarray, atleast_2d

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', None, module_type_store, ['asarray_chkfinite', 'asarray', 'atleast_2d'], [asarray_chkfinite, asarray, atleast_2d])

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_17489)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.linalg.misc import LinAlgError, _datacopied' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_17491 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg.misc')

if (type(import_17491) is not StypyTypeError):

    if (import_17491 != 'pyd_module'):
        __import__(import_17491)
        sys_modules_17492 = sys.modules[import_17491]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg.misc', sys_modules_17492.module_type_store, module_type_store, ['LinAlgError', '_datacopied'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_17492, sys_modules_17492.module_type_store, module_type_store)
    else:
        from scipy.linalg.misc import LinAlgError, _datacopied

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg.misc', None, module_type_store, ['LinAlgError', '_datacopied'], [LinAlgError, _datacopied])

else:
    # Assigning a type to the variable 'scipy.linalg.misc' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg.misc', import_17491)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.linalg.lapack import get_lapack_funcs' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_17493 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg.lapack')

if (type(import_17493) is not StypyTypeError):

    if (import_17493 != 'pyd_module'):
        __import__(import_17493)
        sys_modules_17494 = sys.modules[import_17493]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg.lapack', sys_modules_17494.module_type_store, module_type_store, ['get_lapack_funcs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_17494, sys_modules_17494.module_type_store, module_type_store)
    else:
        from scipy.linalg.lapack import get_lapack_funcs

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg.lapack', None, module_type_store, ['get_lapack_funcs'], [get_lapack_funcs])

else:
    # Assigning a type to the variable 'scipy.linalg.lapack' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg.lapack', import_17493)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')


# Assigning a List to a Name (line 11):

# Assigning a List to a Name (line 11):
__all__ = ['cholesky', 'cho_factor', 'cho_solve', 'cholesky_banded', 'cho_solve_banded']
module_type_store.set_exportable_members(['cholesky', 'cho_factor', 'cho_solve', 'cholesky_banded', 'cho_solve_banded'])

# Obtaining an instance of the builtin type 'list' (line 11)
list_17495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 11)
# Adding element type (line 11)
str_17496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 11), 'str', 'cholesky')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 10), list_17495, str_17496)
# Adding element type (line 11)
str_17497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 23), 'str', 'cho_factor')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 10), list_17495, str_17497)
# Adding element type (line 11)
str_17498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 37), 'str', 'cho_solve')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 10), list_17495, str_17498)
# Adding element type (line 11)
str_17499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 50), 'str', 'cholesky_banded')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 10), list_17495, str_17499)
# Adding element type (line 11)
str_17500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 11), 'str', 'cho_solve_banded')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 10), list_17495, str_17500)

# Assigning a type to the variable '__all__' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), '__all__', list_17495)

@norecursion
def _cholesky(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 15)
    False_17501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 23), 'False')
    # Getting the type of 'False' (line 15)
    False_17502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 42), 'False')
    # Getting the type of 'True' (line 15)
    True_17503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 55), 'True')
    # Getting the type of 'True' (line 16)
    True_17504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 27), 'True')
    defaults = [False_17501, False_17502, True_17503, True_17504]
    # Create a new context for function '_cholesky'
    module_type_store = module_type_store.open_function_context('_cholesky', 15, 0, False)
    
    # Passed parameters checking function
    _cholesky.stypy_localization = localization
    _cholesky.stypy_type_of_self = None
    _cholesky.stypy_type_store = module_type_store
    _cholesky.stypy_function_name = '_cholesky'
    _cholesky.stypy_param_names_list = ['a', 'lower', 'overwrite_a', 'clean', 'check_finite']
    _cholesky.stypy_varargs_param_name = None
    _cholesky.stypy_kwargs_param_name = None
    _cholesky.stypy_call_defaults = defaults
    _cholesky.stypy_call_varargs = varargs
    _cholesky.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_cholesky', ['a', 'lower', 'overwrite_a', 'clean', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_cholesky', localization, ['a', 'lower', 'overwrite_a', 'clean', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_cholesky(...)' code ##################

    str_17505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 4), 'str', 'Common code for cholesky() and cho_factor().')
    
    # Assigning a IfExp to a Name (line 19):
    
    # Assigning a IfExp to a Name (line 19):
    
    # Getting the type of 'check_finite' (line 19)
    check_finite_17506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 33), 'check_finite')
    # Testing the type of an if expression (line 19)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 19, 9), check_finite_17506)
    # SSA begins for if expression (line 19)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to asarray_chkfinite(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'a' (line 19)
    a_17508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 27), 'a', False)
    # Processing the call keyword arguments (line 19)
    kwargs_17509 = {}
    # Getting the type of 'asarray_chkfinite' (line 19)
    asarray_chkfinite_17507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 9), 'asarray_chkfinite', False)
    # Calling asarray_chkfinite(args, kwargs) (line 19)
    asarray_chkfinite_call_result_17510 = invoke(stypy.reporting.localization.Localization(__file__, 19, 9), asarray_chkfinite_17507, *[a_17508], **kwargs_17509)
    
    # SSA branch for the else part of an if expression (line 19)
    module_type_store.open_ssa_branch('if expression else')
    
    # Call to asarray(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'a' (line 19)
    a_17512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 59), 'a', False)
    # Processing the call keyword arguments (line 19)
    kwargs_17513 = {}
    # Getting the type of 'asarray' (line 19)
    asarray_17511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 51), 'asarray', False)
    # Calling asarray(args, kwargs) (line 19)
    asarray_call_result_17514 = invoke(stypy.reporting.localization.Localization(__file__, 19, 51), asarray_17511, *[a_17512], **kwargs_17513)
    
    # SSA join for if expression (line 19)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_17515 = union_type.UnionType.add(asarray_chkfinite_call_result_17510, asarray_call_result_17514)
    
    # Assigning a type to the variable 'a1' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'a1', if_exp_17515)
    
    # Assigning a Call to a Name (line 20):
    
    # Assigning a Call to a Name (line 20):
    
    # Call to atleast_2d(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'a1' (line 20)
    a1_17517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 20), 'a1', False)
    # Processing the call keyword arguments (line 20)
    kwargs_17518 = {}
    # Getting the type of 'atleast_2d' (line 20)
    atleast_2d_17516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 9), 'atleast_2d', False)
    # Calling atleast_2d(args, kwargs) (line 20)
    atleast_2d_call_result_17519 = invoke(stypy.reporting.localization.Localization(__file__, 20, 9), atleast_2d_17516, *[a1_17517], **kwargs_17518)
    
    # Assigning a type to the variable 'a1' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'a1', atleast_2d_call_result_17519)
    
    
    # Getting the type of 'a1' (line 23)
    a1_17520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 7), 'a1')
    # Obtaining the member 'ndim' of a type (line 23)
    ndim_17521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 7), a1_17520, 'ndim')
    int_17522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 18), 'int')
    # Applying the binary operator '!=' (line 23)
    result_ne_17523 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 7), '!=', ndim_17521, int_17522)
    
    # Testing the type of an if condition (line 23)
    if_condition_17524 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 23, 4), result_ne_17523)
    # Assigning a type to the variable 'if_condition_17524' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'if_condition_17524', if_condition_17524)
    # SSA begins for if statement (line 23)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 24)
    # Processing the call arguments (line 24)
    
    # Call to format(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'a1' (line 25)
    a1_17528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 47), 'a1', False)
    # Obtaining the member 'ndim' of a type (line 25)
    ndim_17529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 47), a1_17528, 'ndim')
    # Processing the call keyword arguments (line 24)
    kwargs_17530 = {}
    str_17526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'str', 'Input array needs to be 2 dimensional but received a {}d-array.')
    # Obtaining the member 'format' of a type (line 24)
    format_17527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 25), str_17526, 'format')
    # Calling format(args, kwargs) (line 24)
    format_call_result_17531 = invoke(stypy.reporting.localization.Localization(__file__, 24, 25), format_17527, *[ndim_17529], **kwargs_17530)
    
    # Processing the call keyword arguments (line 24)
    kwargs_17532 = {}
    # Getting the type of 'ValueError' (line 24)
    ValueError_17525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 24)
    ValueError_call_result_17533 = invoke(stypy.reporting.localization.Localization(__file__, 24, 14), ValueError_17525, *[format_call_result_17531], **kwargs_17532)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 24, 8), ValueError_call_result_17533, 'raise parameter', BaseException)
    # SSA join for if statement (line 23)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_17534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 16), 'int')
    # Getting the type of 'a1' (line 27)
    a1_17535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 7), 'a1')
    # Obtaining the member 'shape' of a type (line 27)
    shape_17536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 7), a1_17535, 'shape')
    # Obtaining the member '__getitem__' of a type (line 27)
    getitem___17537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 7), shape_17536, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 27)
    subscript_call_result_17538 = invoke(stypy.reporting.localization.Localization(__file__, 27, 7), getitem___17537, int_17534)
    
    
    # Obtaining the type of the subscript
    int_17539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 31), 'int')
    # Getting the type of 'a1' (line 27)
    a1_17540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 22), 'a1')
    # Obtaining the member 'shape' of a type (line 27)
    shape_17541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 22), a1_17540, 'shape')
    # Obtaining the member '__getitem__' of a type (line 27)
    getitem___17542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 22), shape_17541, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 27)
    subscript_call_result_17543 = invoke(stypy.reporting.localization.Localization(__file__, 27, 22), getitem___17542, int_17539)
    
    # Applying the binary operator '!=' (line 27)
    result_ne_17544 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 7), '!=', subscript_call_result_17538, subscript_call_result_17543)
    
    # Testing the type of an if condition (line 27)
    if_condition_17545 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 4), result_ne_17544)
    # Assigning a type to the variable 'if_condition_17545' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'if_condition_17545', if_condition_17545)
    # SSA begins for if statement (line 27)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 28)
    # Processing the call arguments (line 28)
    
    # Call to format(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'a1' (line 29)
    a1_17549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 49), 'a1', False)
    # Obtaining the member 'shape' of a type (line 29)
    shape_17550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 49), a1_17549, 'shape')
    # Processing the call keyword arguments (line 28)
    kwargs_17551 = {}
    str_17547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 25), 'str', 'Input array is expected to be square but has the shape: {}.')
    # Obtaining the member 'format' of a type (line 28)
    format_17548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 25), str_17547, 'format')
    # Calling format(args, kwargs) (line 28)
    format_call_result_17552 = invoke(stypy.reporting.localization.Localization(__file__, 28, 25), format_17548, *[shape_17550], **kwargs_17551)
    
    # Processing the call keyword arguments (line 28)
    kwargs_17553 = {}
    # Getting the type of 'ValueError' (line 28)
    ValueError_17546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 28)
    ValueError_call_result_17554 = invoke(stypy.reporting.localization.Localization(__file__, 28, 14), ValueError_17546, *[format_call_result_17552], **kwargs_17553)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 28, 8), ValueError_call_result_17554, 'raise parameter', BaseException)
    # SSA join for if statement (line 27)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'a1' (line 32)
    a1_17555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 7), 'a1')
    # Obtaining the member 'size' of a type (line 32)
    size_17556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 7), a1_17555, 'size')
    int_17557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 18), 'int')
    # Applying the binary operator '==' (line 32)
    result_eq_17558 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 7), '==', size_17556, int_17557)
    
    # Testing the type of an if condition (line 32)
    if_condition_17559 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 4), result_eq_17558)
    # Assigning a type to the variable 'if_condition_17559' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'if_condition_17559', if_condition_17559)
    # SSA begins for if statement (line 32)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 33)
    tuple_17560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 33)
    # Adding element type (line 33)
    
    # Call to copy(...): (line 33)
    # Processing the call keyword arguments (line 33)
    kwargs_17563 = {}
    # Getting the type of 'a1' (line 33)
    a1_17561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'a1', False)
    # Obtaining the member 'copy' of a type (line 33)
    copy_17562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 15), a1_17561, 'copy')
    # Calling copy(args, kwargs) (line 33)
    copy_call_result_17564 = invoke(stypy.reporting.localization.Localization(__file__, 33, 15), copy_17562, *[], **kwargs_17563)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 15), tuple_17560, copy_call_result_17564)
    # Adding element type (line 33)
    # Getting the type of 'lower' (line 33)
    lower_17565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 26), 'lower')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 15), tuple_17560, lower_17565)
    
    # Assigning a type to the variable 'stypy_return_type' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'stypy_return_type', tuple_17560)
    # SSA join for if statement (line 32)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 35):
    
    # Assigning a BoolOp to a Name (line 35):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_a' (line 35)
    overwrite_a_17566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 18), 'overwrite_a')
    
    # Call to _datacopied(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'a1' (line 35)
    a1_17568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 45), 'a1', False)
    # Getting the type of 'a' (line 35)
    a_17569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 49), 'a', False)
    # Processing the call keyword arguments (line 35)
    kwargs_17570 = {}
    # Getting the type of '_datacopied' (line 35)
    _datacopied_17567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 33), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 35)
    _datacopied_call_result_17571 = invoke(stypy.reporting.localization.Localization(__file__, 35, 33), _datacopied_17567, *[a1_17568, a_17569], **kwargs_17570)
    
    # Applying the binary operator 'or' (line 35)
    result_or_keyword_17572 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 18), 'or', overwrite_a_17566, _datacopied_call_result_17571)
    
    # Assigning a type to the variable 'overwrite_a' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'overwrite_a', result_or_keyword_17572)
    
    # Assigning a Call to a Tuple (line 36):
    
    # Assigning a Subscript to a Name (line 36):
    
    # Obtaining the type of the subscript
    int_17573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Obtaining an instance of the builtin type 'tuple' (line 36)
    tuple_17575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 36)
    # Adding element type (line 36)
    str_17576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 31), 'str', 'potrf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 31), tuple_17575, str_17576)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 36)
    tuple_17577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 36)
    # Adding element type (line 36)
    # Getting the type of 'a1' (line 36)
    a1_17578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 43), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 43), tuple_17577, a1_17578)
    
    # Processing the call keyword arguments (line 36)
    kwargs_17579 = {}
    # Getting the type of 'get_lapack_funcs' (line 36)
    get_lapack_funcs_17574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 13), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 36)
    get_lapack_funcs_call_result_17580 = invoke(stypy.reporting.localization.Localization(__file__, 36, 13), get_lapack_funcs_17574, *[tuple_17575, tuple_17577], **kwargs_17579)
    
    # Obtaining the member '__getitem__' of a type (line 36)
    getitem___17581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 4), get_lapack_funcs_call_result_17580, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 36)
    subscript_call_result_17582 = invoke(stypy.reporting.localization.Localization(__file__, 36, 4), getitem___17581, int_17573)
    
    # Assigning a type to the variable 'tuple_var_assignment_17468' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'tuple_var_assignment_17468', subscript_call_result_17582)
    
    # Assigning a Name to a Name (line 36):
    # Getting the type of 'tuple_var_assignment_17468' (line 36)
    tuple_var_assignment_17468_17583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'tuple_var_assignment_17468')
    # Assigning a type to the variable 'potrf' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'potrf', tuple_var_assignment_17468_17583)
    
    # Assigning a Call to a Tuple (line 37):
    
    # Assigning a Subscript to a Name (line 37):
    
    # Obtaining the type of the subscript
    int_17584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 4), 'int')
    
    # Call to potrf(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'a1' (line 37)
    a1_17586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 20), 'a1', False)
    # Processing the call keyword arguments (line 37)
    # Getting the type of 'lower' (line 37)
    lower_17587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 30), 'lower', False)
    keyword_17588 = lower_17587
    # Getting the type of 'overwrite_a' (line 37)
    overwrite_a_17589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 49), 'overwrite_a', False)
    keyword_17590 = overwrite_a_17589
    # Getting the type of 'clean' (line 37)
    clean_17591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 68), 'clean', False)
    keyword_17592 = clean_17591
    kwargs_17593 = {'lower': keyword_17588, 'overwrite_a': keyword_17590, 'clean': keyword_17592}
    # Getting the type of 'potrf' (line 37)
    potrf_17585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 14), 'potrf', False)
    # Calling potrf(args, kwargs) (line 37)
    potrf_call_result_17594 = invoke(stypy.reporting.localization.Localization(__file__, 37, 14), potrf_17585, *[a1_17586], **kwargs_17593)
    
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___17595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 4), potrf_call_result_17594, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_17596 = invoke(stypy.reporting.localization.Localization(__file__, 37, 4), getitem___17595, int_17584)
    
    # Assigning a type to the variable 'tuple_var_assignment_17469' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'tuple_var_assignment_17469', subscript_call_result_17596)
    
    # Assigning a Subscript to a Name (line 37):
    
    # Obtaining the type of the subscript
    int_17597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 4), 'int')
    
    # Call to potrf(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'a1' (line 37)
    a1_17599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 20), 'a1', False)
    # Processing the call keyword arguments (line 37)
    # Getting the type of 'lower' (line 37)
    lower_17600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 30), 'lower', False)
    keyword_17601 = lower_17600
    # Getting the type of 'overwrite_a' (line 37)
    overwrite_a_17602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 49), 'overwrite_a', False)
    keyword_17603 = overwrite_a_17602
    # Getting the type of 'clean' (line 37)
    clean_17604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 68), 'clean', False)
    keyword_17605 = clean_17604
    kwargs_17606 = {'lower': keyword_17601, 'overwrite_a': keyword_17603, 'clean': keyword_17605}
    # Getting the type of 'potrf' (line 37)
    potrf_17598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 14), 'potrf', False)
    # Calling potrf(args, kwargs) (line 37)
    potrf_call_result_17607 = invoke(stypy.reporting.localization.Localization(__file__, 37, 14), potrf_17598, *[a1_17599], **kwargs_17606)
    
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___17608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 4), potrf_call_result_17607, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_17609 = invoke(stypy.reporting.localization.Localization(__file__, 37, 4), getitem___17608, int_17597)
    
    # Assigning a type to the variable 'tuple_var_assignment_17470' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'tuple_var_assignment_17470', subscript_call_result_17609)
    
    # Assigning a Name to a Name (line 37):
    # Getting the type of 'tuple_var_assignment_17469' (line 37)
    tuple_var_assignment_17469_17610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'tuple_var_assignment_17469')
    # Assigning a type to the variable 'c' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'c', tuple_var_assignment_17469_17610)
    
    # Assigning a Name to a Name (line 37):
    # Getting the type of 'tuple_var_assignment_17470' (line 37)
    tuple_var_assignment_17470_17611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'tuple_var_assignment_17470')
    # Assigning a type to the variable 'info' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 7), 'info', tuple_var_assignment_17470_17611)
    
    
    # Getting the type of 'info' (line 38)
    info_17612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 7), 'info')
    int_17613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 14), 'int')
    # Applying the binary operator '>' (line 38)
    result_gt_17614 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 7), '>', info_17612, int_17613)
    
    # Testing the type of an if condition (line 38)
    if_condition_17615 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 4), result_gt_17614)
    # Assigning a type to the variable 'if_condition_17615' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'if_condition_17615', if_condition_17615)
    # SSA begins for if statement (line 38)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 39)
    # Processing the call arguments (line 39)
    str_17617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 26), 'str', '%d-th leading minor of the array is not positive definite')
    # Getting the type of 'info' (line 40)
    info_17618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 39), 'info', False)
    # Applying the binary operator '%' (line 39)
    result_mod_17619 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 26), '%', str_17617, info_17618)
    
    # Processing the call keyword arguments (line 39)
    kwargs_17620 = {}
    # Getting the type of 'LinAlgError' (line 39)
    LinAlgError_17616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 39)
    LinAlgError_call_result_17621 = invoke(stypy.reporting.localization.Localization(__file__, 39, 14), LinAlgError_17616, *[result_mod_17619], **kwargs_17620)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 39, 8), LinAlgError_call_result_17621, 'raise parameter', BaseException)
    # SSA join for if statement (line 38)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'info' (line 41)
    info_17622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 7), 'info')
    int_17623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 14), 'int')
    # Applying the binary operator '<' (line 41)
    result_lt_17624 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 7), '<', info_17622, int_17623)
    
    # Testing the type of an if condition (line 41)
    if_condition_17625 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 4), result_lt_17624)
    # Assigning a type to the variable 'if_condition_17625' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'if_condition_17625', if_condition_17625)
    # SSA begins for if statement (line 41)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Call to format(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Getting the type of 'info' (line 43)
    info_17629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 56), 'info', False)
    # Applying the 'usub' unary operator (line 43)
    result___neg___17630 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 55), 'usub', info_17629)
    
    # Processing the call keyword arguments (line 42)
    kwargs_17631 = {}
    str_17627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 25), 'str', 'LAPACK reported an illegal value in {}-th argumenton entry to "POTRF".')
    # Obtaining the member 'format' of a type (line 42)
    format_17628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 25), str_17627, 'format')
    # Calling format(args, kwargs) (line 42)
    format_call_result_17632 = invoke(stypy.reporting.localization.Localization(__file__, 42, 25), format_17628, *[result___neg___17630], **kwargs_17631)
    
    # Processing the call keyword arguments (line 42)
    kwargs_17633 = {}
    # Getting the type of 'ValueError' (line 42)
    ValueError_17626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 42)
    ValueError_call_result_17634 = invoke(stypy.reporting.localization.Localization(__file__, 42, 14), ValueError_17626, *[format_call_result_17632], **kwargs_17633)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 42, 8), ValueError_call_result_17634, 'raise parameter', BaseException)
    # SSA join for if statement (line 41)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 44)
    tuple_17635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 44)
    # Adding element type (line 44)
    # Getting the type of 'c' (line 44)
    c_17636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 11), tuple_17635, c_17636)
    # Adding element type (line 44)
    # Getting the type of 'lower' (line 44)
    lower_17637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 14), 'lower')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 11), tuple_17635, lower_17637)
    
    # Assigning a type to the variable 'stypy_return_type' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type', tuple_17635)
    
    # ################# End of '_cholesky(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_cholesky' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_17638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17638)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_cholesky'
    return stypy_return_type_17638

# Assigning a type to the variable '_cholesky' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), '_cholesky', _cholesky)

@norecursion
def cholesky(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 47)
    False_17639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 22), 'False')
    # Getting the type of 'False' (line 47)
    False_17640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 41), 'False')
    # Getting the type of 'True' (line 47)
    True_17641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 61), 'True')
    defaults = [False_17639, False_17640, True_17641]
    # Create a new context for function 'cholesky'
    module_type_store = module_type_store.open_function_context('cholesky', 47, 0, False)
    
    # Passed parameters checking function
    cholesky.stypy_localization = localization
    cholesky.stypy_type_of_self = None
    cholesky.stypy_type_store = module_type_store
    cholesky.stypy_function_name = 'cholesky'
    cholesky.stypy_param_names_list = ['a', 'lower', 'overwrite_a', 'check_finite']
    cholesky.stypy_varargs_param_name = None
    cholesky.stypy_kwargs_param_name = None
    cholesky.stypy_call_defaults = defaults
    cholesky.stypy_call_varargs = varargs
    cholesky.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cholesky', ['a', 'lower', 'overwrite_a', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cholesky', localization, ['a', 'lower', 'overwrite_a', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cholesky(...)' code ##################

    str_17642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, (-1)), 'str', '\n    Compute the Cholesky decomposition of a matrix.\n\n    Returns the Cholesky decomposition, :math:`A = L L^*` or\n    :math:`A = U^* U` of a Hermitian positive-definite matrix A.\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        Matrix to be decomposed\n    lower : bool, optional\n        Whether to compute the upper or lower triangular Cholesky\n        factorization.  Default is upper-triangular.\n    overwrite_a : bool, optional\n        Whether to overwrite data in `a` (may improve performance).\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    c : (M, M) ndarray\n        Upper- or lower-triangular Cholesky factor of `a`.\n\n    Raises\n    ------\n    LinAlgError : if decomposition fails.\n\n    Examples\n    --------\n    >>> from scipy import array, linalg, dot\n    >>> a = array([[1,-2j],[2j,5]])\n    >>> L = linalg.cholesky(a, lower=True)\n    >>> L\n    array([[ 1.+0.j,  0.+0.j],\n           [ 0.+2.j,  1.+0.j]])\n    >>> dot(L, L.T.conj())\n    array([[ 1.+0.j,  0.-2.j],\n           [ 0.+2.j,  5.+0.j]])\n\n    ')
    
    # Assigning a Call to a Tuple (line 90):
    
    # Assigning a Subscript to a Name (line 90):
    
    # Obtaining the type of the subscript
    int_17643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 4), 'int')
    
    # Call to _cholesky(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'a' (line 90)
    a_17645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'a', False)
    # Processing the call keyword arguments (line 90)
    # Getting the type of 'lower' (line 90)
    lower_17646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 34), 'lower', False)
    keyword_17647 = lower_17646
    # Getting the type of 'overwrite_a' (line 90)
    overwrite_a_17648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 53), 'overwrite_a', False)
    keyword_17649 = overwrite_a_17648
    # Getting the type of 'True' (line 90)
    True_17650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 72), 'True', False)
    keyword_17651 = True_17650
    # Getting the type of 'check_finite' (line 91)
    check_finite_17652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 38), 'check_finite', False)
    keyword_17653 = check_finite_17652
    kwargs_17654 = {'lower': keyword_17647, 'overwrite_a': keyword_17649, 'check_finite': keyword_17653, 'clean': keyword_17651}
    # Getting the type of '_cholesky' (line 90)
    _cholesky_17644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 15), '_cholesky', False)
    # Calling _cholesky(args, kwargs) (line 90)
    _cholesky_call_result_17655 = invoke(stypy.reporting.localization.Localization(__file__, 90, 15), _cholesky_17644, *[a_17645], **kwargs_17654)
    
    # Obtaining the member '__getitem__' of a type (line 90)
    getitem___17656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 4), _cholesky_call_result_17655, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 90)
    subscript_call_result_17657 = invoke(stypy.reporting.localization.Localization(__file__, 90, 4), getitem___17656, int_17643)
    
    # Assigning a type to the variable 'tuple_var_assignment_17471' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'tuple_var_assignment_17471', subscript_call_result_17657)
    
    # Assigning a Subscript to a Name (line 90):
    
    # Obtaining the type of the subscript
    int_17658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 4), 'int')
    
    # Call to _cholesky(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'a' (line 90)
    a_17660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'a', False)
    # Processing the call keyword arguments (line 90)
    # Getting the type of 'lower' (line 90)
    lower_17661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 34), 'lower', False)
    keyword_17662 = lower_17661
    # Getting the type of 'overwrite_a' (line 90)
    overwrite_a_17663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 53), 'overwrite_a', False)
    keyword_17664 = overwrite_a_17663
    # Getting the type of 'True' (line 90)
    True_17665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 72), 'True', False)
    keyword_17666 = True_17665
    # Getting the type of 'check_finite' (line 91)
    check_finite_17667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 38), 'check_finite', False)
    keyword_17668 = check_finite_17667
    kwargs_17669 = {'lower': keyword_17662, 'overwrite_a': keyword_17664, 'check_finite': keyword_17668, 'clean': keyword_17666}
    # Getting the type of '_cholesky' (line 90)
    _cholesky_17659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 15), '_cholesky', False)
    # Calling _cholesky(args, kwargs) (line 90)
    _cholesky_call_result_17670 = invoke(stypy.reporting.localization.Localization(__file__, 90, 15), _cholesky_17659, *[a_17660], **kwargs_17669)
    
    # Obtaining the member '__getitem__' of a type (line 90)
    getitem___17671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 4), _cholesky_call_result_17670, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 90)
    subscript_call_result_17672 = invoke(stypy.reporting.localization.Localization(__file__, 90, 4), getitem___17671, int_17658)
    
    # Assigning a type to the variable 'tuple_var_assignment_17472' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'tuple_var_assignment_17472', subscript_call_result_17672)
    
    # Assigning a Name to a Name (line 90):
    # Getting the type of 'tuple_var_assignment_17471' (line 90)
    tuple_var_assignment_17471_17673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'tuple_var_assignment_17471')
    # Assigning a type to the variable 'c' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'c', tuple_var_assignment_17471_17673)
    
    # Assigning a Name to a Name (line 90):
    # Getting the type of 'tuple_var_assignment_17472' (line 90)
    tuple_var_assignment_17472_17674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'tuple_var_assignment_17472')
    # Assigning a type to the variable 'lower' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 7), 'lower', tuple_var_assignment_17472_17674)
    # Getting the type of 'c' (line 92)
    c_17675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type', c_17675)
    
    # ################# End of 'cholesky(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cholesky' in the type store
    # Getting the type of 'stypy_return_type' (line 47)
    stypy_return_type_17676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17676)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cholesky'
    return stypy_return_type_17676

# Assigning a type to the variable 'cholesky' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'cholesky', cholesky)

@norecursion
def cho_factor(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 95)
    False_17677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 24), 'False')
    # Getting the type of 'False' (line 95)
    False_17678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 43), 'False')
    # Getting the type of 'True' (line 95)
    True_17679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 63), 'True')
    defaults = [False_17677, False_17678, True_17679]
    # Create a new context for function 'cho_factor'
    module_type_store = module_type_store.open_function_context('cho_factor', 95, 0, False)
    
    # Passed parameters checking function
    cho_factor.stypy_localization = localization
    cho_factor.stypy_type_of_self = None
    cho_factor.stypy_type_store = module_type_store
    cho_factor.stypy_function_name = 'cho_factor'
    cho_factor.stypy_param_names_list = ['a', 'lower', 'overwrite_a', 'check_finite']
    cho_factor.stypy_varargs_param_name = None
    cho_factor.stypy_kwargs_param_name = None
    cho_factor.stypy_call_defaults = defaults
    cho_factor.stypy_call_varargs = varargs
    cho_factor.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cho_factor', ['a', 'lower', 'overwrite_a', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cho_factor', localization, ['a', 'lower', 'overwrite_a', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cho_factor(...)' code ##################

    str_17680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, (-1)), 'str', '\n    Compute the Cholesky decomposition of a matrix, to use in cho_solve\n\n    Returns a matrix containing the Cholesky decomposition,\n    ``A = L L*`` or ``A = U* U`` of a Hermitian positive-definite matrix `a`.\n    The return value can be directly used as the first parameter to cho_solve.\n\n    .. warning::\n        The returned matrix also contains random data in the entries not\n        used by the Cholesky decomposition. If you need to zero these\n        entries, use the function `cholesky` instead.\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        Matrix to be decomposed\n    lower : bool, optional\n        Whether to compute the upper or lower triangular Cholesky factorization\n        (Default: upper-triangular)\n    overwrite_a : bool, optional\n        Whether to overwrite data in a (may improve performance)\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    c : (M, M) ndarray\n        Matrix whose upper or lower triangle contains the Cholesky factor\n        of `a`. Other parts of the matrix contain random data.\n    lower : bool\n        Flag indicating whether the factor is in the lower or upper triangle\n\n    Raises\n    ------\n    LinAlgError\n        Raised if decomposition fails.\n\n    See also\n    --------\n    cho_solve : Solve a linear set equations using the Cholesky factorization\n                of a matrix.\n\n    ')
    
    # Assigning a Call to a Tuple (line 141):
    
    # Assigning a Subscript to a Name (line 141):
    
    # Obtaining the type of the subscript
    int_17681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 4), 'int')
    
    # Call to _cholesky(...): (line 141)
    # Processing the call arguments (line 141)
    # Getting the type of 'a' (line 141)
    a_17683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 25), 'a', False)
    # Processing the call keyword arguments (line 141)
    # Getting the type of 'lower' (line 141)
    lower_17684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 34), 'lower', False)
    keyword_17685 = lower_17684
    # Getting the type of 'overwrite_a' (line 141)
    overwrite_a_17686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 53), 'overwrite_a', False)
    keyword_17687 = overwrite_a_17686
    # Getting the type of 'False' (line 141)
    False_17688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 72), 'False', False)
    keyword_17689 = False_17688
    # Getting the type of 'check_finite' (line 142)
    check_finite_17690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 38), 'check_finite', False)
    keyword_17691 = check_finite_17690
    kwargs_17692 = {'lower': keyword_17685, 'overwrite_a': keyword_17687, 'check_finite': keyword_17691, 'clean': keyword_17689}
    # Getting the type of '_cholesky' (line 141)
    _cholesky_17682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), '_cholesky', False)
    # Calling _cholesky(args, kwargs) (line 141)
    _cholesky_call_result_17693 = invoke(stypy.reporting.localization.Localization(__file__, 141, 15), _cholesky_17682, *[a_17683], **kwargs_17692)
    
    # Obtaining the member '__getitem__' of a type (line 141)
    getitem___17694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 4), _cholesky_call_result_17693, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 141)
    subscript_call_result_17695 = invoke(stypy.reporting.localization.Localization(__file__, 141, 4), getitem___17694, int_17681)
    
    # Assigning a type to the variable 'tuple_var_assignment_17473' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'tuple_var_assignment_17473', subscript_call_result_17695)
    
    # Assigning a Subscript to a Name (line 141):
    
    # Obtaining the type of the subscript
    int_17696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 4), 'int')
    
    # Call to _cholesky(...): (line 141)
    # Processing the call arguments (line 141)
    # Getting the type of 'a' (line 141)
    a_17698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 25), 'a', False)
    # Processing the call keyword arguments (line 141)
    # Getting the type of 'lower' (line 141)
    lower_17699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 34), 'lower', False)
    keyword_17700 = lower_17699
    # Getting the type of 'overwrite_a' (line 141)
    overwrite_a_17701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 53), 'overwrite_a', False)
    keyword_17702 = overwrite_a_17701
    # Getting the type of 'False' (line 141)
    False_17703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 72), 'False', False)
    keyword_17704 = False_17703
    # Getting the type of 'check_finite' (line 142)
    check_finite_17705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 38), 'check_finite', False)
    keyword_17706 = check_finite_17705
    kwargs_17707 = {'lower': keyword_17700, 'overwrite_a': keyword_17702, 'check_finite': keyword_17706, 'clean': keyword_17704}
    # Getting the type of '_cholesky' (line 141)
    _cholesky_17697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), '_cholesky', False)
    # Calling _cholesky(args, kwargs) (line 141)
    _cholesky_call_result_17708 = invoke(stypy.reporting.localization.Localization(__file__, 141, 15), _cholesky_17697, *[a_17698], **kwargs_17707)
    
    # Obtaining the member '__getitem__' of a type (line 141)
    getitem___17709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 4), _cholesky_call_result_17708, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 141)
    subscript_call_result_17710 = invoke(stypy.reporting.localization.Localization(__file__, 141, 4), getitem___17709, int_17696)
    
    # Assigning a type to the variable 'tuple_var_assignment_17474' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'tuple_var_assignment_17474', subscript_call_result_17710)
    
    # Assigning a Name to a Name (line 141):
    # Getting the type of 'tuple_var_assignment_17473' (line 141)
    tuple_var_assignment_17473_17711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'tuple_var_assignment_17473')
    # Assigning a type to the variable 'c' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'c', tuple_var_assignment_17473_17711)
    
    # Assigning a Name to a Name (line 141):
    # Getting the type of 'tuple_var_assignment_17474' (line 141)
    tuple_var_assignment_17474_17712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'tuple_var_assignment_17474')
    # Assigning a type to the variable 'lower' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 7), 'lower', tuple_var_assignment_17474_17712)
    
    # Obtaining an instance of the builtin type 'tuple' (line 143)
    tuple_17713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 143)
    # Adding element type (line 143)
    # Getting the type of 'c' (line 143)
    c_17714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 11), 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 11), tuple_17713, c_17714)
    # Adding element type (line 143)
    # Getting the type of 'lower' (line 143)
    lower_17715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 14), 'lower')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 11), tuple_17713, lower_17715)
    
    # Assigning a type to the variable 'stypy_return_type' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type', tuple_17713)
    
    # ################# End of 'cho_factor(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cho_factor' in the type store
    # Getting the type of 'stypy_return_type' (line 95)
    stypy_return_type_17716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17716)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cho_factor'
    return stypy_return_type_17716

# Assigning a type to the variable 'cho_factor' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'cho_factor', cho_factor)

@norecursion
def cho_solve(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 146)
    False_17717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 42), 'False')
    # Getting the type of 'True' (line 146)
    True_17718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 62), 'True')
    defaults = [False_17717, True_17718]
    # Create a new context for function 'cho_solve'
    module_type_store = module_type_store.open_function_context('cho_solve', 146, 0, False)
    
    # Passed parameters checking function
    cho_solve.stypy_localization = localization
    cho_solve.stypy_type_of_self = None
    cho_solve.stypy_type_store = module_type_store
    cho_solve.stypy_function_name = 'cho_solve'
    cho_solve.stypy_param_names_list = ['c_and_lower', 'b', 'overwrite_b', 'check_finite']
    cho_solve.stypy_varargs_param_name = None
    cho_solve.stypy_kwargs_param_name = None
    cho_solve.stypy_call_defaults = defaults
    cho_solve.stypy_call_varargs = varargs
    cho_solve.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cho_solve', ['c_and_lower', 'b', 'overwrite_b', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cho_solve', localization, ['c_and_lower', 'b', 'overwrite_b', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cho_solve(...)' code ##################

    str_17719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, (-1)), 'str', 'Solve the linear equations A x = b, given the Cholesky factorization of A.\n\n    Parameters\n    ----------\n    (c, lower) : tuple, (array, bool)\n        Cholesky factorization of a, as given by cho_factor\n    b : array\n        Right-hand side\n    overwrite_b : bool, optional\n        Whether to overwrite data in b (may improve performance)\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    x : array\n        The solution to the system A x = b\n\n    See also\n    --------\n    cho_factor : Cholesky factorization of a matrix\n\n    ')
    
    # Assigning a Name to a Tuple (line 172):
    
    # Assigning a Subscript to a Name (line 172):
    
    # Obtaining the type of the subscript
    int_17720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 4), 'int')
    # Getting the type of 'c_and_lower' (line 172)
    c_and_lower_17721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 17), 'c_and_lower')
    # Obtaining the member '__getitem__' of a type (line 172)
    getitem___17722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 4), c_and_lower_17721, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 172)
    subscript_call_result_17723 = invoke(stypy.reporting.localization.Localization(__file__, 172, 4), getitem___17722, int_17720)
    
    # Assigning a type to the variable 'tuple_var_assignment_17475' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'tuple_var_assignment_17475', subscript_call_result_17723)
    
    # Assigning a Subscript to a Name (line 172):
    
    # Obtaining the type of the subscript
    int_17724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 4), 'int')
    # Getting the type of 'c_and_lower' (line 172)
    c_and_lower_17725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 17), 'c_and_lower')
    # Obtaining the member '__getitem__' of a type (line 172)
    getitem___17726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 4), c_and_lower_17725, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 172)
    subscript_call_result_17727 = invoke(stypy.reporting.localization.Localization(__file__, 172, 4), getitem___17726, int_17724)
    
    # Assigning a type to the variable 'tuple_var_assignment_17476' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'tuple_var_assignment_17476', subscript_call_result_17727)
    
    # Assigning a Name to a Name (line 172):
    # Getting the type of 'tuple_var_assignment_17475' (line 172)
    tuple_var_assignment_17475_17728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'tuple_var_assignment_17475')
    # Assigning a type to the variable 'c' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 5), 'c', tuple_var_assignment_17475_17728)
    
    # Assigning a Name to a Name (line 172):
    # Getting the type of 'tuple_var_assignment_17476' (line 172)
    tuple_var_assignment_17476_17729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'tuple_var_assignment_17476')
    # Assigning a type to the variable 'lower' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'lower', tuple_var_assignment_17476_17729)
    
    # Getting the type of 'check_finite' (line 173)
    check_finite_17730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 7), 'check_finite')
    # Testing the type of an if condition (line 173)
    if_condition_17731 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 173, 4), check_finite_17730)
    # Assigning a type to the variable 'if_condition_17731' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'if_condition_17731', if_condition_17731)
    # SSA begins for if statement (line 173)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 174):
    
    # Assigning a Call to a Name (line 174):
    
    # Call to asarray_chkfinite(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'b' (line 174)
    b_17733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 31), 'b', False)
    # Processing the call keyword arguments (line 174)
    kwargs_17734 = {}
    # Getting the type of 'asarray_chkfinite' (line 174)
    asarray_chkfinite_17732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 13), 'asarray_chkfinite', False)
    # Calling asarray_chkfinite(args, kwargs) (line 174)
    asarray_chkfinite_call_result_17735 = invoke(stypy.reporting.localization.Localization(__file__, 174, 13), asarray_chkfinite_17732, *[b_17733], **kwargs_17734)
    
    # Assigning a type to the variable 'b1' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'b1', asarray_chkfinite_call_result_17735)
    
    # Assigning a Call to a Name (line 175):
    
    # Assigning a Call to a Name (line 175):
    
    # Call to asarray_chkfinite(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 'c' (line 175)
    c_17737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 30), 'c', False)
    # Processing the call keyword arguments (line 175)
    kwargs_17738 = {}
    # Getting the type of 'asarray_chkfinite' (line 175)
    asarray_chkfinite_17736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'asarray_chkfinite', False)
    # Calling asarray_chkfinite(args, kwargs) (line 175)
    asarray_chkfinite_call_result_17739 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), asarray_chkfinite_17736, *[c_17737], **kwargs_17738)
    
    # Assigning a type to the variable 'c' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'c', asarray_chkfinite_call_result_17739)
    # SSA branch for the else part of an if statement (line 173)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 177):
    
    # Assigning a Call to a Name (line 177):
    
    # Call to asarray(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'b' (line 177)
    b_17741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 21), 'b', False)
    # Processing the call keyword arguments (line 177)
    kwargs_17742 = {}
    # Getting the type of 'asarray' (line 177)
    asarray_17740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 13), 'asarray', False)
    # Calling asarray(args, kwargs) (line 177)
    asarray_call_result_17743 = invoke(stypy.reporting.localization.Localization(__file__, 177, 13), asarray_17740, *[b_17741], **kwargs_17742)
    
    # Assigning a type to the variable 'b1' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'b1', asarray_call_result_17743)
    
    # Assigning a Call to a Name (line 178):
    
    # Assigning a Call to a Name (line 178):
    
    # Call to asarray(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'c' (line 178)
    c_17745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 20), 'c', False)
    # Processing the call keyword arguments (line 178)
    kwargs_17746 = {}
    # Getting the type of 'asarray' (line 178)
    asarray_17744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'asarray', False)
    # Calling asarray(args, kwargs) (line 178)
    asarray_call_result_17747 = invoke(stypy.reporting.localization.Localization(__file__, 178, 12), asarray_17744, *[c_17745], **kwargs_17746)
    
    # Assigning a type to the variable 'c' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'c', asarray_call_result_17747)
    # SSA join for if statement (line 173)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'c' (line 179)
    c_17748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 7), 'c')
    # Obtaining the member 'ndim' of a type (line 179)
    ndim_17749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 7), c_17748, 'ndim')
    int_17750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 17), 'int')
    # Applying the binary operator '!=' (line 179)
    result_ne_17751 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 7), '!=', ndim_17749, int_17750)
    
    
    
    # Obtaining the type of the subscript
    int_17752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 30), 'int')
    # Getting the type of 'c' (line 179)
    c_17753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 22), 'c')
    # Obtaining the member 'shape' of a type (line 179)
    shape_17754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 22), c_17753, 'shape')
    # Obtaining the member '__getitem__' of a type (line 179)
    getitem___17755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 22), shape_17754, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 179)
    subscript_call_result_17756 = invoke(stypy.reporting.localization.Localization(__file__, 179, 22), getitem___17755, int_17752)
    
    
    # Obtaining the type of the subscript
    int_17757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 44), 'int')
    # Getting the type of 'c' (line 179)
    c_17758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 36), 'c')
    # Obtaining the member 'shape' of a type (line 179)
    shape_17759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 36), c_17758, 'shape')
    # Obtaining the member '__getitem__' of a type (line 179)
    getitem___17760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 36), shape_17759, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 179)
    subscript_call_result_17761 = invoke(stypy.reporting.localization.Localization(__file__, 179, 36), getitem___17760, int_17757)
    
    # Applying the binary operator '!=' (line 179)
    result_ne_17762 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 22), '!=', subscript_call_result_17756, subscript_call_result_17761)
    
    # Applying the binary operator 'or' (line 179)
    result_or_keyword_17763 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 7), 'or', result_ne_17751, result_ne_17762)
    
    # Testing the type of an if condition (line 179)
    if_condition_17764 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 179, 4), result_or_keyword_17763)
    # Assigning a type to the variable 'if_condition_17764' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'if_condition_17764', if_condition_17764)
    # SSA begins for if statement (line 179)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 180)
    # Processing the call arguments (line 180)
    str_17766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 25), 'str', 'The factored matrix c is not square.')
    # Processing the call keyword arguments (line 180)
    kwargs_17767 = {}
    # Getting the type of 'ValueError' (line 180)
    ValueError_17765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 180)
    ValueError_call_result_17768 = invoke(stypy.reporting.localization.Localization(__file__, 180, 14), ValueError_17765, *[str_17766], **kwargs_17767)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 180, 8), ValueError_call_result_17768, 'raise parameter', BaseException)
    # SSA join for if statement (line 179)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_17769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 15), 'int')
    # Getting the type of 'c' (line 181)
    c_17770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 7), 'c')
    # Obtaining the member 'shape' of a type (line 181)
    shape_17771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 7), c_17770, 'shape')
    # Obtaining the member '__getitem__' of a type (line 181)
    getitem___17772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 7), shape_17771, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 181)
    subscript_call_result_17773 = invoke(stypy.reporting.localization.Localization(__file__, 181, 7), getitem___17772, int_17769)
    
    
    # Obtaining the type of the subscript
    int_17774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 30), 'int')
    # Getting the type of 'b1' (line 181)
    b1_17775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 21), 'b1')
    # Obtaining the member 'shape' of a type (line 181)
    shape_17776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 21), b1_17775, 'shape')
    # Obtaining the member '__getitem__' of a type (line 181)
    getitem___17777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 21), shape_17776, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 181)
    subscript_call_result_17778 = invoke(stypy.reporting.localization.Localization(__file__, 181, 21), getitem___17777, int_17774)
    
    # Applying the binary operator '!=' (line 181)
    result_ne_17779 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 7), '!=', subscript_call_result_17773, subscript_call_result_17778)
    
    # Testing the type of an if condition (line 181)
    if_condition_17780 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 4), result_ne_17779)
    # Assigning a type to the variable 'if_condition_17780' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'if_condition_17780', if_condition_17780)
    # SSA begins for if statement (line 181)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 182)
    # Processing the call arguments (line 182)
    str_17782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 25), 'str', 'incompatible dimensions.')
    # Processing the call keyword arguments (line 182)
    kwargs_17783 = {}
    # Getting the type of 'ValueError' (line 182)
    ValueError_17781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 182)
    ValueError_call_result_17784 = invoke(stypy.reporting.localization.Localization(__file__, 182, 14), ValueError_17781, *[str_17782], **kwargs_17783)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 182, 8), ValueError_call_result_17784, 'raise parameter', BaseException)
    # SSA join for if statement (line 181)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 184):
    
    # Assigning a BoolOp to a Name (line 184):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_b' (line 184)
    overwrite_b_17785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 18), 'overwrite_b')
    
    # Call to _datacopied(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'b1' (line 184)
    b1_17787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 45), 'b1', False)
    # Getting the type of 'b' (line 184)
    b_17788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 49), 'b', False)
    # Processing the call keyword arguments (line 184)
    kwargs_17789 = {}
    # Getting the type of '_datacopied' (line 184)
    _datacopied_17786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 33), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 184)
    _datacopied_call_result_17790 = invoke(stypy.reporting.localization.Localization(__file__, 184, 33), _datacopied_17786, *[b1_17787, b_17788], **kwargs_17789)
    
    # Applying the binary operator 'or' (line 184)
    result_or_keyword_17791 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 18), 'or', overwrite_b_17785, _datacopied_call_result_17790)
    
    # Assigning a type to the variable 'overwrite_b' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'overwrite_b', result_or_keyword_17791)
    
    # Assigning a Call to a Tuple (line 186):
    
    # Assigning a Subscript to a Name (line 186):
    
    # Obtaining the type of the subscript
    int_17792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 186)
    # Processing the call arguments (line 186)
    
    # Obtaining an instance of the builtin type 'tuple' (line 186)
    tuple_17794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 186)
    # Adding element type (line 186)
    str_17795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 31), 'str', 'potrs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 31), tuple_17794, str_17795)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 186)
    tuple_17796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 186)
    # Adding element type (line 186)
    # Getting the type of 'c' (line 186)
    c_17797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 43), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 43), tuple_17796, c_17797)
    # Adding element type (line 186)
    # Getting the type of 'b1' (line 186)
    b1_17798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 46), 'b1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 43), tuple_17796, b1_17798)
    
    # Processing the call keyword arguments (line 186)
    kwargs_17799 = {}
    # Getting the type of 'get_lapack_funcs' (line 186)
    get_lapack_funcs_17793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 13), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 186)
    get_lapack_funcs_call_result_17800 = invoke(stypy.reporting.localization.Localization(__file__, 186, 13), get_lapack_funcs_17793, *[tuple_17794, tuple_17796], **kwargs_17799)
    
    # Obtaining the member '__getitem__' of a type (line 186)
    getitem___17801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 4), get_lapack_funcs_call_result_17800, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 186)
    subscript_call_result_17802 = invoke(stypy.reporting.localization.Localization(__file__, 186, 4), getitem___17801, int_17792)
    
    # Assigning a type to the variable 'tuple_var_assignment_17477' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'tuple_var_assignment_17477', subscript_call_result_17802)
    
    # Assigning a Name to a Name (line 186):
    # Getting the type of 'tuple_var_assignment_17477' (line 186)
    tuple_var_assignment_17477_17803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'tuple_var_assignment_17477')
    # Assigning a type to the variable 'potrs' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'potrs', tuple_var_assignment_17477_17803)
    
    # Assigning a Call to a Tuple (line 187):
    
    # Assigning a Subscript to a Name (line 187):
    
    # Obtaining the type of the subscript
    int_17804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 4), 'int')
    
    # Call to potrs(...): (line 187)
    # Processing the call arguments (line 187)
    # Getting the type of 'c' (line 187)
    c_17806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 20), 'c', False)
    # Getting the type of 'b1' (line 187)
    b1_17807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 23), 'b1', False)
    # Processing the call keyword arguments (line 187)
    # Getting the type of 'lower' (line 187)
    lower_17808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 33), 'lower', False)
    keyword_17809 = lower_17808
    # Getting the type of 'overwrite_b' (line 187)
    overwrite_b_17810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 52), 'overwrite_b', False)
    keyword_17811 = overwrite_b_17810
    kwargs_17812 = {'lower': keyword_17809, 'overwrite_b': keyword_17811}
    # Getting the type of 'potrs' (line 187)
    potrs_17805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 14), 'potrs', False)
    # Calling potrs(args, kwargs) (line 187)
    potrs_call_result_17813 = invoke(stypy.reporting.localization.Localization(__file__, 187, 14), potrs_17805, *[c_17806, b1_17807], **kwargs_17812)
    
    # Obtaining the member '__getitem__' of a type (line 187)
    getitem___17814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 4), potrs_call_result_17813, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 187)
    subscript_call_result_17815 = invoke(stypy.reporting.localization.Localization(__file__, 187, 4), getitem___17814, int_17804)
    
    # Assigning a type to the variable 'tuple_var_assignment_17478' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'tuple_var_assignment_17478', subscript_call_result_17815)
    
    # Assigning a Subscript to a Name (line 187):
    
    # Obtaining the type of the subscript
    int_17816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 4), 'int')
    
    # Call to potrs(...): (line 187)
    # Processing the call arguments (line 187)
    # Getting the type of 'c' (line 187)
    c_17818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 20), 'c', False)
    # Getting the type of 'b1' (line 187)
    b1_17819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 23), 'b1', False)
    # Processing the call keyword arguments (line 187)
    # Getting the type of 'lower' (line 187)
    lower_17820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 33), 'lower', False)
    keyword_17821 = lower_17820
    # Getting the type of 'overwrite_b' (line 187)
    overwrite_b_17822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 52), 'overwrite_b', False)
    keyword_17823 = overwrite_b_17822
    kwargs_17824 = {'lower': keyword_17821, 'overwrite_b': keyword_17823}
    # Getting the type of 'potrs' (line 187)
    potrs_17817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 14), 'potrs', False)
    # Calling potrs(args, kwargs) (line 187)
    potrs_call_result_17825 = invoke(stypy.reporting.localization.Localization(__file__, 187, 14), potrs_17817, *[c_17818, b1_17819], **kwargs_17824)
    
    # Obtaining the member '__getitem__' of a type (line 187)
    getitem___17826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 4), potrs_call_result_17825, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 187)
    subscript_call_result_17827 = invoke(stypy.reporting.localization.Localization(__file__, 187, 4), getitem___17826, int_17816)
    
    # Assigning a type to the variable 'tuple_var_assignment_17479' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'tuple_var_assignment_17479', subscript_call_result_17827)
    
    # Assigning a Name to a Name (line 187):
    # Getting the type of 'tuple_var_assignment_17478' (line 187)
    tuple_var_assignment_17478_17828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'tuple_var_assignment_17478')
    # Assigning a type to the variable 'x' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'x', tuple_var_assignment_17478_17828)
    
    # Assigning a Name to a Name (line 187):
    # Getting the type of 'tuple_var_assignment_17479' (line 187)
    tuple_var_assignment_17479_17829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'tuple_var_assignment_17479')
    # Assigning a type to the variable 'info' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 7), 'info', tuple_var_assignment_17479_17829)
    
    
    # Getting the type of 'info' (line 188)
    info_17830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 7), 'info')
    int_17831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 15), 'int')
    # Applying the binary operator '!=' (line 188)
    result_ne_17832 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 7), '!=', info_17830, int_17831)
    
    # Testing the type of an if condition (line 188)
    if_condition_17833 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 4), result_ne_17832)
    # Assigning a type to the variable 'if_condition_17833' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'if_condition_17833', if_condition_17833)
    # SSA begins for if statement (line 188)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 189)
    # Processing the call arguments (line 189)
    str_17835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 25), 'str', 'illegal value in %d-th argument of internal potrs')
    
    # Getting the type of 'info' (line 190)
    info_17836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 28), 'info', False)
    # Applying the 'usub' unary operator (line 190)
    result___neg___17837 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 27), 'usub', info_17836)
    
    # Applying the binary operator '%' (line 189)
    result_mod_17838 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 25), '%', str_17835, result___neg___17837)
    
    # Processing the call keyword arguments (line 189)
    kwargs_17839 = {}
    # Getting the type of 'ValueError' (line 189)
    ValueError_17834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 189)
    ValueError_call_result_17840 = invoke(stypy.reporting.localization.Localization(__file__, 189, 14), ValueError_17834, *[result_mod_17838], **kwargs_17839)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 189, 8), ValueError_call_result_17840, 'raise parameter', BaseException)
    # SSA join for if statement (line 188)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 191)
    x_17841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'stypy_return_type', x_17841)
    
    # ################# End of 'cho_solve(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cho_solve' in the type store
    # Getting the type of 'stypy_return_type' (line 146)
    stypy_return_type_17842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17842)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cho_solve'
    return stypy_return_type_17842

# Assigning a type to the variable 'cho_solve' (line 146)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 0), 'cho_solve', cho_solve)

@norecursion
def cholesky_banded(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 194)
    False_17843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 37), 'False')
    # Getting the type of 'False' (line 194)
    False_17844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 50), 'False')
    # Getting the type of 'True' (line 194)
    True_17845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 70), 'True')
    defaults = [False_17843, False_17844, True_17845]
    # Create a new context for function 'cholesky_banded'
    module_type_store = module_type_store.open_function_context('cholesky_banded', 194, 0, False)
    
    # Passed parameters checking function
    cholesky_banded.stypy_localization = localization
    cholesky_banded.stypy_type_of_self = None
    cholesky_banded.stypy_type_store = module_type_store
    cholesky_banded.stypy_function_name = 'cholesky_banded'
    cholesky_banded.stypy_param_names_list = ['ab', 'overwrite_ab', 'lower', 'check_finite']
    cholesky_banded.stypy_varargs_param_name = None
    cholesky_banded.stypy_kwargs_param_name = None
    cholesky_banded.stypy_call_defaults = defaults
    cholesky_banded.stypy_call_varargs = varargs
    cholesky_banded.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cholesky_banded', ['ab', 'overwrite_ab', 'lower', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cholesky_banded', localization, ['ab', 'overwrite_ab', 'lower', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cholesky_banded(...)' code ##################

    str_17846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, (-1)), 'str', '\n    Cholesky decompose a banded Hermitian positive-definite matrix\n\n    The matrix a is stored in ab either in lower diagonal or upper\n    diagonal ordered form::\n\n        ab[u + i - j, j] == a[i,j]        (if upper form; i <= j)\n        ab[    i - j, j] == a[i,j]        (if lower form; i >= j)\n\n    Example of ab (shape of a is (6,6), u=2)::\n\n        upper form:\n        *   *   a02 a13 a24 a35\n        *   a01 a12 a23 a34 a45\n        a00 a11 a22 a33 a44 a55\n\n        lower form:\n        a00 a11 a22 a33 a44 a55\n        a10 a21 a32 a43 a54 *\n        a20 a31 a42 a53 *   *\n\n    Parameters\n    ----------\n    ab : (u + 1, M) array_like\n        Banded matrix\n    overwrite_ab : bool, optional\n        Discard data in ab (may enhance performance)\n    lower : bool, optional\n        Is the matrix in the lower form. (Default is upper form)\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    c : (u + 1, M) ndarray\n        Cholesky factorization of a, in the same banded format as ab\n\n    ')
    
    # Getting the type of 'check_finite' (line 235)
    check_finite_17847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 7), 'check_finite')
    # Testing the type of an if condition (line 235)
    if_condition_17848 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 4), check_finite_17847)
    # Assigning a type to the variable 'if_condition_17848' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'if_condition_17848', if_condition_17848)
    # SSA begins for if statement (line 235)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 236):
    
    # Assigning a Call to a Name (line 236):
    
    # Call to asarray_chkfinite(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'ab' (line 236)
    ab_17850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 31), 'ab', False)
    # Processing the call keyword arguments (line 236)
    kwargs_17851 = {}
    # Getting the type of 'asarray_chkfinite' (line 236)
    asarray_chkfinite_17849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 13), 'asarray_chkfinite', False)
    # Calling asarray_chkfinite(args, kwargs) (line 236)
    asarray_chkfinite_call_result_17852 = invoke(stypy.reporting.localization.Localization(__file__, 236, 13), asarray_chkfinite_17849, *[ab_17850], **kwargs_17851)
    
    # Assigning a type to the variable 'ab' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'ab', asarray_chkfinite_call_result_17852)
    # SSA branch for the else part of an if statement (line 235)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 238):
    
    # Assigning a Call to a Name (line 238):
    
    # Call to asarray(...): (line 238)
    # Processing the call arguments (line 238)
    # Getting the type of 'ab' (line 238)
    ab_17854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 21), 'ab', False)
    # Processing the call keyword arguments (line 238)
    kwargs_17855 = {}
    # Getting the type of 'asarray' (line 238)
    asarray_17853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 13), 'asarray', False)
    # Calling asarray(args, kwargs) (line 238)
    asarray_call_result_17856 = invoke(stypy.reporting.localization.Localization(__file__, 238, 13), asarray_17853, *[ab_17854], **kwargs_17855)
    
    # Assigning a type to the variable 'ab' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'ab', asarray_call_result_17856)
    # SSA join for if statement (line 235)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 240):
    
    # Assigning a Subscript to a Name (line 240):
    
    # Obtaining the type of the subscript
    int_17857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 240)
    # Processing the call arguments (line 240)
    
    # Obtaining an instance of the builtin type 'tuple' (line 240)
    tuple_17859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 240)
    # Adding element type (line 240)
    str_17860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 31), 'str', 'pbtrf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 31), tuple_17859, str_17860)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 240)
    tuple_17861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 240)
    # Adding element type (line 240)
    # Getting the type of 'ab' (line 240)
    ab_17862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 43), 'ab', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 43), tuple_17861, ab_17862)
    
    # Processing the call keyword arguments (line 240)
    kwargs_17863 = {}
    # Getting the type of 'get_lapack_funcs' (line 240)
    get_lapack_funcs_17858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 13), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 240)
    get_lapack_funcs_call_result_17864 = invoke(stypy.reporting.localization.Localization(__file__, 240, 13), get_lapack_funcs_17858, *[tuple_17859, tuple_17861], **kwargs_17863)
    
    # Obtaining the member '__getitem__' of a type (line 240)
    getitem___17865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 4), get_lapack_funcs_call_result_17864, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 240)
    subscript_call_result_17866 = invoke(stypy.reporting.localization.Localization(__file__, 240, 4), getitem___17865, int_17857)
    
    # Assigning a type to the variable 'tuple_var_assignment_17480' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'tuple_var_assignment_17480', subscript_call_result_17866)
    
    # Assigning a Name to a Name (line 240):
    # Getting the type of 'tuple_var_assignment_17480' (line 240)
    tuple_var_assignment_17480_17867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'tuple_var_assignment_17480')
    # Assigning a type to the variable 'pbtrf' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'pbtrf', tuple_var_assignment_17480_17867)
    
    # Assigning a Call to a Tuple (line 241):
    
    # Assigning a Subscript to a Name (line 241):
    
    # Obtaining the type of the subscript
    int_17868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 4), 'int')
    
    # Call to pbtrf(...): (line 241)
    # Processing the call arguments (line 241)
    # Getting the type of 'ab' (line 241)
    ab_17870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 20), 'ab', False)
    # Processing the call keyword arguments (line 241)
    # Getting the type of 'lower' (line 241)
    lower_17871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 30), 'lower', False)
    keyword_17872 = lower_17871
    # Getting the type of 'overwrite_ab' (line 241)
    overwrite_ab_17873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 50), 'overwrite_ab', False)
    keyword_17874 = overwrite_ab_17873
    kwargs_17875 = {'lower': keyword_17872, 'overwrite_ab': keyword_17874}
    # Getting the type of 'pbtrf' (line 241)
    pbtrf_17869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 14), 'pbtrf', False)
    # Calling pbtrf(args, kwargs) (line 241)
    pbtrf_call_result_17876 = invoke(stypy.reporting.localization.Localization(__file__, 241, 14), pbtrf_17869, *[ab_17870], **kwargs_17875)
    
    # Obtaining the member '__getitem__' of a type (line 241)
    getitem___17877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 4), pbtrf_call_result_17876, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 241)
    subscript_call_result_17878 = invoke(stypy.reporting.localization.Localization(__file__, 241, 4), getitem___17877, int_17868)
    
    # Assigning a type to the variable 'tuple_var_assignment_17481' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'tuple_var_assignment_17481', subscript_call_result_17878)
    
    # Assigning a Subscript to a Name (line 241):
    
    # Obtaining the type of the subscript
    int_17879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 4), 'int')
    
    # Call to pbtrf(...): (line 241)
    # Processing the call arguments (line 241)
    # Getting the type of 'ab' (line 241)
    ab_17881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 20), 'ab', False)
    # Processing the call keyword arguments (line 241)
    # Getting the type of 'lower' (line 241)
    lower_17882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 30), 'lower', False)
    keyword_17883 = lower_17882
    # Getting the type of 'overwrite_ab' (line 241)
    overwrite_ab_17884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 50), 'overwrite_ab', False)
    keyword_17885 = overwrite_ab_17884
    kwargs_17886 = {'lower': keyword_17883, 'overwrite_ab': keyword_17885}
    # Getting the type of 'pbtrf' (line 241)
    pbtrf_17880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 14), 'pbtrf', False)
    # Calling pbtrf(args, kwargs) (line 241)
    pbtrf_call_result_17887 = invoke(stypy.reporting.localization.Localization(__file__, 241, 14), pbtrf_17880, *[ab_17881], **kwargs_17886)
    
    # Obtaining the member '__getitem__' of a type (line 241)
    getitem___17888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 4), pbtrf_call_result_17887, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 241)
    subscript_call_result_17889 = invoke(stypy.reporting.localization.Localization(__file__, 241, 4), getitem___17888, int_17879)
    
    # Assigning a type to the variable 'tuple_var_assignment_17482' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'tuple_var_assignment_17482', subscript_call_result_17889)
    
    # Assigning a Name to a Name (line 241):
    # Getting the type of 'tuple_var_assignment_17481' (line 241)
    tuple_var_assignment_17481_17890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'tuple_var_assignment_17481')
    # Assigning a type to the variable 'c' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'c', tuple_var_assignment_17481_17890)
    
    # Assigning a Name to a Name (line 241):
    # Getting the type of 'tuple_var_assignment_17482' (line 241)
    tuple_var_assignment_17482_17891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'tuple_var_assignment_17482')
    # Assigning a type to the variable 'info' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 7), 'info', tuple_var_assignment_17482_17891)
    
    
    # Getting the type of 'info' (line 242)
    info_17892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 7), 'info')
    int_17893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 14), 'int')
    # Applying the binary operator '>' (line 242)
    result_gt_17894 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 7), '>', info_17892, int_17893)
    
    # Testing the type of an if condition (line 242)
    if_condition_17895 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 4), result_gt_17894)
    # Assigning a type to the variable 'if_condition_17895' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'if_condition_17895', if_condition_17895)
    # SSA begins for if statement (line 242)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 243)
    # Processing the call arguments (line 243)
    str_17897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 26), 'str', '%d-th leading minor not positive definite')
    # Getting the type of 'info' (line 243)
    info_17898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 72), 'info', False)
    # Applying the binary operator '%' (line 243)
    result_mod_17899 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 26), '%', str_17897, info_17898)
    
    # Processing the call keyword arguments (line 243)
    kwargs_17900 = {}
    # Getting the type of 'LinAlgError' (line 243)
    LinAlgError_17896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 243)
    LinAlgError_call_result_17901 = invoke(stypy.reporting.localization.Localization(__file__, 243, 14), LinAlgError_17896, *[result_mod_17899], **kwargs_17900)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 243, 8), LinAlgError_call_result_17901, 'raise parameter', BaseException)
    # SSA join for if statement (line 242)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'info' (line 244)
    info_17902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 7), 'info')
    int_17903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 14), 'int')
    # Applying the binary operator '<' (line 244)
    result_lt_17904 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 7), '<', info_17902, int_17903)
    
    # Testing the type of an if condition (line 244)
    if_condition_17905 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 244, 4), result_lt_17904)
    # Assigning a type to the variable 'if_condition_17905' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'if_condition_17905', if_condition_17905)
    # SSA begins for if statement (line 244)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 245)
    # Processing the call arguments (line 245)
    str_17907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 25), 'str', 'illegal value in %d-th argument of internal pbtrf')
    
    # Getting the type of 'info' (line 246)
    info_17908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 28), 'info', False)
    # Applying the 'usub' unary operator (line 246)
    result___neg___17909 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 27), 'usub', info_17908)
    
    # Applying the binary operator '%' (line 245)
    result_mod_17910 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 25), '%', str_17907, result___neg___17909)
    
    # Processing the call keyword arguments (line 245)
    kwargs_17911 = {}
    # Getting the type of 'ValueError' (line 245)
    ValueError_17906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 245)
    ValueError_call_result_17912 = invoke(stypy.reporting.localization.Localization(__file__, 245, 14), ValueError_17906, *[result_mod_17910], **kwargs_17911)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 245, 8), ValueError_call_result_17912, 'raise parameter', BaseException)
    # SSA join for if statement (line 244)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'c' (line 247)
    c_17913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'stypy_return_type', c_17913)
    
    # ################# End of 'cholesky_banded(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cholesky_banded' in the type store
    # Getting the type of 'stypy_return_type' (line 194)
    stypy_return_type_17914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17914)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cholesky_banded'
    return stypy_return_type_17914

# Assigning a type to the variable 'cholesky_banded' (line 194)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 0), 'cholesky_banded', cholesky_banded)

@norecursion
def cho_solve_banded(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 250)
    False_17915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 50), 'False')
    # Getting the type of 'True' (line 250)
    True_17916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 70), 'True')
    defaults = [False_17915, True_17916]
    # Create a new context for function 'cho_solve_banded'
    module_type_store = module_type_store.open_function_context('cho_solve_banded', 250, 0, False)
    
    # Passed parameters checking function
    cho_solve_banded.stypy_localization = localization
    cho_solve_banded.stypy_type_of_self = None
    cho_solve_banded.stypy_type_store = module_type_store
    cho_solve_banded.stypy_function_name = 'cho_solve_banded'
    cho_solve_banded.stypy_param_names_list = ['cb_and_lower', 'b', 'overwrite_b', 'check_finite']
    cho_solve_banded.stypy_varargs_param_name = None
    cho_solve_banded.stypy_kwargs_param_name = None
    cho_solve_banded.stypy_call_defaults = defaults
    cho_solve_banded.stypy_call_varargs = varargs
    cho_solve_banded.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cho_solve_banded', ['cb_and_lower', 'b', 'overwrite_b', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cho_solve_banded', localization, ['cb_and_lower', 'b', 'overwrite_b', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cho_solve_banded(...)' code ##################

    str_17917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, (-1)), 'str', 'Solve the linear equations A x = b, given the Cholesky factorization of A.\n\n    Parameters\n    ----------\n    (cb, lower) : tuple, (array, bool)\n        `cb` is the Cholesky factorization of A, as given by cholesky_banded.\n        `lower` must be the same value that was given to cholesky_banded.\n    b : array\n        Right-hand side\n    overwrite_b : bool, optional\n        If True, the function will overwrite the values in `b`.\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    x : array\n        The solution to the system A x = b\n\n    See also\n    --------\n    cholesky_banded : Cholesky factorization of a banded matrix\n\n    Notes\n    -----\n\n    .. versionadded:: 0.8.0\n\n    ')
    
    # Assigning a Name to a Tuple (line 282):
    
    # Assigning a Subscript to a Name (line 282):
    
    # Obtaining the type of the subscript
    int_17918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 4), 'int')
    # Getting the type of 'cb_and_lower' (line 282)
    cb_and_lower_17919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 18), 'cb_and_lower')
    # Obtaining the member '__getitem__' of a type (line 282)
    getitem___17920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 4), cb_and_lower_17919, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 282)
    subscript_call_result_17921 = invoke(stypy.reporting.localization.Localization(__file__, 282, 4), getitem___17920, int_17918)
    
    # Assigning a type to the variable 'tuple_var_assignment_17483' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'tuple_var_assignment_17483', subscript_call_result_17921)
    
    # Assigning a Subscript to a Name (line 282):
    
    # Obtaining the type of the subscript
    int_17922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 4), 'int')
    # Getting the type of 'cb_and_lower' (line 282)
    cb_and_lower_17923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 18), 'cb_and_lower')
    # Obtaining the member '__getitem__' of a type (line 282)
    getitem___17924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 4), cb_and_lower_17923, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 282)
    subscript_call_result_17925 = invoke(stypy.reporting.localization.Localization(__file__, 282, 4), getitem___17924, int_17922)
    
    # Assigning a type to the variable 'tuple_var_assignment_17484' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'tuple_var_assignment_17484', subscript_call_result_17925)
    
    # Assigning a Name to a Name (line 282):
    # Getting the type of 'tuple_var_assignment_17483' (line 282)
    tuple_var_assignment_17483_17926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'tuple_var_assignment_17483')
    # Assigning a type to the variable 'cb' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 5), 'cb', tuple_var_assignment_17483_17926)
    
    # Assigning a Name to a Name (line 282):
    # Getting the type of 'tuple_var_assignment_17484' (line 282)
    tuple_var_assignment_17484_17927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'tuple_var_assignment_17484')
    # Assigning a type to the variable 'lower' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 9), 'lower', tuple_var_assignment_17484_17927)
    
    # Getting the type of 'check_finite' (line 283)
    check_finite_17928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 7), 'check_finite')
    # Testing the type of an if condition (line 283)
    if_condition_17929 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 283, 4), check_finite_17928)
    # Assigning a type to the variable 'if_condition_17929' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'if_condition_17929', if_condition_17929)
    # SSA begins for if statement (line 283)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 284):
    
    # Assigning a Call to a Name (line 284):
    
    # Call to asarray_chkfinite(...): (line 284)
    # Processing the call arguments (line 284)
    # Getting the type of 'cb' (line 284)
    cb_17931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 31), 'cb', False)
    # Processing the call keyword arguments (line 284)
    kwargs_17932 = {}
    # Getting the type of 'asarray_chkfinite' (line 284)
    asarray_chkfinite_17930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 13), 'asarray_chkfinite', False)
    # Calling asarray_chkfinite(args, kwargs) (line 284)
    asarray_chkfinite_call_result_17933 = invoke(stypy.reporting.localization.Localization(__file__, 284, 13), asarray_chkfinite_17930, *[cb_17931], **kwargs_17932)
    
    # Assigning a type to the variable 'cb' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'cb', asarray_chkfinite_call_result_17933)
    
    # Assigning a Call to a Name (line 285):
    
    # Assigning a Call to a Name (line 285):
    
    # Call to asarray_chkfinite(...): (line 285)
    # Processing the call arguments (line 285)
    # Getting the type of 'b' (line 285)
    b_17935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 30), 'b', False)
    # Processing the call keyword arguments (line 285)
    kwargs_17936 = {}
    # Getting the type of 'asarray_chkfinite' (line 285)
    asarray_chkfinite_17934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'asarray_chkfinite', False)
    # Calling asarray_chkfinite(args, kwargs) (line 285)
    asarray_chkfinite_call_result_17937 = invoke(stypy.reporting.localization.Localization(__file__, 285, 12), asarray_chkfinite_17934, *[b_17935], **kwargs_17936)
    
    # Assigning a type to the variable 'b' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'b', asarray_chkfinite_call_result_17937)
    # SSA branch for the else part of an if statement (line 283)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 287):
    
    # Assigning a Call to a Name (line 287):
    
    # Call to asarray(...): (line 287)
    # Processing the call arguments (line 287)
    # Getting the type of 'cb' (line 287)
    cb_17939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 21), 'cb', False)
    # Processing the call keyword arguments (line 287)
    kwargs_17940 = {}
    # Getting the type of 'asarray' (line 287)
    asarray_17938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 13), 'asarray', False)
    # Calling asarray(args, kwargs) (line 287)
    asarray_call_result_17941 = invoke(stypy.reporting.localization.Localization(__file__, 287, 13), asarray_17938, *[cb_17939], **kwargs_17940)
    
    # Assigning a type to the variable 'cb' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'cb', asarray_call_result_17941)
    
    # Assigning a Call to a Name (line 288):
    
    # Assigning a Call to a Name (line 288):
    
    # Call to asarray(...): (line 288)
    # Processing the call arguments (line 288)
    # Getting the type of 'b' (line 288)
    b_17943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 20), 'b', False)
    # Processing the call keyword arguments (line 288)
    kwargs_17944 = {}
    # Getting the type of 'asarray' (line 288)
    asarray_17942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'asarray', False)
    # Calling asarray(args, kwargs) (line 288)
    asarray_call_result_17945 = invoke(stypy.reporting.localization.Localization(__file__, 288, 12), asarray_17942, *[b_17943], **kwargs_17944)
    
    # Assigning a type to the variable 'b' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'b', asarray_call_result_17945)
    # SSA join for if statement (line 283)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_17946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 16), 'int')
    # Getting the type of 'cb' (line 291)
    cb_17947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 7), 'cb')
    # Obtaining the member 'shape' of a type (line 291)
    shape_17948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 7), cb_17947, 'shape')
    # Obtaining the member '__getitem__' of a type (line 291)
    getitem___17949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 7), shape_17948, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 291)
    subscript_call_result_17950 = invoke(stypy.reporting.localization.Localization(__file__, 291, 7), getitem___17949, int_17946)
    
    
    # Obtaining the type of the subscript
    int_17951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 31), 'int')
    # Getting the type of 'b' (line 291)
    b_17952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 23), 'b')
    # Obtaining the member 'shape' of a type (line 291)
    shape_17953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 23), b_17952, 'shape')
    # Obtaining the member '__getitem__' of a type (line 291)
    getitem___17954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 23), shape_17953, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 291)
    subscript_call_result_17955 = invoke(stypy.reporting.localization.Localization(__file__, 291, 23), getitem___17954, int_17951)
    
    # Applying the binary operator '!=' (line 291)
    result_ne_17956 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 7), '!=', subscript_call_result_17950, subscript_call_result_17955)
    
    # Testing the type of an if condition (line 291)
    if_condition_17957 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 291, 4), result_ne_17956)
    # Assigning a type to the variable 'if_condition_17957' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'if_condition_17957', if_condition_17957)
    # SSA begins for if statement (line 291)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 292)
    # Processing the call arguments (line 292)
    str_17959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 25), 'str', 'shapes of cb and b are not compatible.')
    # Processing the call keyword arguments (line 292)
    kwargs_17960 = {}
    # Getting the type of 'ValueError' (line 292)
    ValueError_17958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 292)
    ValueError_call_result_17961 = invoke(stypy.reporting.localization.Localization(__file__, 292, 14), ValueError_17958, *[str_17959], **kwargs_17960)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 292, 8), ValueError_call_result_17961, 'raise parameter', BaseException)
    # SSA join for if statement (line 291)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 294):
    
    # Assigning a Subscript to a Name (line 294):
    
    # Obtaining the type of the subscript
    int_17962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 294)
    # Processing the call arguments (line 294)
    
    # Obtaining an instance of the builtin type 'tuple' (line 294)
    tuple_17964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 294)
    # Adding element type (line 294)
    str_17965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 31), 'str', 'pbtrs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 31), tuple_17964, str_17965)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 294)
    tuple_17966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 294)
    # Adding element type (line 294)
    # Getting the type of 'cb' (line 294)
    cb_17967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 43), 'cb', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 43), tuple_17966, cb_17967)
    # Adding element type (line 294)
    # Getting the type of 'b' (line 294)
    b_17968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 47), 'b', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 43), tuple_17966, b_17968)
    
    # Processing the call keyword arguments (line 294)
    kwargs_17969 = {}
    # Getting the type of 'get_lapack_funcs' (line 294)
    get_lapack_funcs_17963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 13), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 294)
    get_lapack_funcs_call_result_17970 = invoke(stypy.reporting.localization.Localization(__file__, 294, 13), get_lapack_funcs_17963, *[tuple_17964, tuple_17966], **kwargs_17969)
    
    # Obtaining the member '__getitem__' of a type (line 294)
    getitem___17971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 4), get_lapack_funcs_call_result_17970, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 294)
    subscript_call_result_17972 = invoke(stypy.reporting.localization.Localization(__file__, 294, 4), getitem___17971, int_17962)
    
    # Assigning a type to the variable 'tuple_var_assignment_17485' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'tuple_var_assignment_17485', subscript_call_result_17972)
    
    # Assigning a Name to a Name (line 294):
    # Getting the type of 'tuple_var_assignment_17485' (line 294)
    tuple_var_assignment_17485_17973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'tuple_var_assignment_17485')
    # Assigning a type to the variable 'pbtrs' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'pbtrs', tuple_var_assignment_17485_17973)
    
    # Assigning a Call to a Tuple (line 295):
    
    # Assigning a Subscript to a Name (line 295):
    
    # Obtaining the type of the subscript
    int_17974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 4), 'int')
    
    # Call to pbtrs(...): (line 295)
    # Processing the call arguments (line 295)
    # Getting the type of 'cb' (line 295)
    cb_17976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 20), 'cb', False)
    # Getting the type of 'b' (line 295)
    b_17977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 24), 'b', False)
    # Processing the call keyword arguments (line 295)
    # Getting the type of 'lower' (line 295)
    lower_17978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 33), 'lower', False)
    keyword_17979 = lower_17978
    # Getting the type of 'overwrite_b' (line 295)
    overwrite_b_17980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 52), 'overwrite_b', False)
    keyword_17981 = overwrite_b_17980
    kwargs_17982 = {'lower': keyword_17979, 'overwrite_b': keyword_17981}
    # Getting the type of 'pbtrs' (line 295)
    pbtrs_17975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 14), 'pbtrs', False)
    # Calling pbtrs(args, kwargs) (line 295)
    pbtrs_call_result_17983 = invoke(stypy.reporting.localization.Localization(__file__, 295, 14), pbtrs_17975, *[cb_17976, b_17977], **kwargs_17982)
    
    # Obtaining the member '__getitem__' of a type (line 295)
    getitem___17984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 4), pbtrs_call_result_17983, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 295)
    subscript_call_result_17985 = invoke(stypy.reporting.localization.Localization(__file__, 295, 4), getitem___17984, int_17974)
    
    # Assigning a type to the variable 'tuple_var_assignment_17486' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'tuple_var_assignment_17486', subscript_call_result_17985)
    
    # Assigning a Subscript to a Name (line 295):
    
    # Obtaining the type of the subscript
    int_17986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 4), 'int')
    
    # Call to pbtrs(...): (line 295)
    # Processing the call arguments (line 295)
    # Getting the type of 'cb' (line 295)
    cb_17988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 20), 'cb', False)
    # Getting the type of 'b' (line 295)
    b_17989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 24), 'b', False)
    # Processing the call keyword arguments (line 295)
    # Getting the type of 'lower' (line 295)
    lower_17990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 33), 'lower', False)
    keyword_17991 = lower_17990
    # Getting the type of 'overwrite_b' (line 295)
    overwrite_b_17992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 52), 'overwrite_b', False)
    keyword_17993 = overwrite_b_17992
    kwargs_17994 = {'lower': keyword_17991, 'overwrite_b': keyword_17993}
    # Getting the type of 'pbtrs' (line 295)
    pbtrs_17987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 14), 'pbtrs', False)
    # Calling pbtrs(args, kwargs) (line 295)
    pbtrs_call_result_17995 = invoke(stypy.reporting.localization.Localization(__file__, 295, 14), pbtrs_17987, *[cb_17988, b_17989], **kwargs_17994)
    
    # Obtaining the member '__getitem__' of a type (line 295)
    getitem___17996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 4), pbtrs_call_result_17995, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 295)
    subscript_call_result_17997 = invoke(stypy.reporting.localization.Localization(__file__, 295, 4), getitem___17996, int_17986)
    
    # Assigning a type to the variable 'tuple_var_assignment_17487' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'tuple_var_assignment_17487', subscript_call_result_17997)
    
    # Assigning a Name to a Name (line 295):
    # Getting the type of 'tuple_var_assignment_17486' (line 295)
    tuple_var_assignment_17486_17998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'tuple_var_assignment_17486')
    # Assigning a type to the variable 'x' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'x', tuple_var_assignment_17486_17998)
    
    # Assigning a Name to a Name (line 295):
    # Getting the type of 'tuple_var_assignment_17487' (line 295)
    tuple_var_assignment_17487_17999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'tuple_var_assignment_17487')
    # Assigning a type to the variable 'info' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 7), 'info', tuple_var_assignment_17487_17999)
    
    
    # Getting the type of 'info' (line 296)
    info_18000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 7), 'info')
    int_18001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 14), 'int')
    # Applying the binary operator '>' (line 296)
    result_gt_18002 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 7), '>', info_18000, int_18001)
    
    # Testing the type of an if condition (line 296)
    if_condition_18003 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 296, 4), result_gt_18002)
    # Assigning a type to the variable 'if_condition_18003' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'if_condition_18003', if_condition_18003)
    # SSA begins for if statement (line 296)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 297)
    # Processing the call arguments (line 297)
    str_18005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 26), 'str', '%d-th leading minor not positive definite')
    # Getting the type of 'info' (line 297)
    info_18006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 72), 'info', False)
    # Applying the binary operator '%' (line 297)
    result_mod_18007 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 26), '%', str_18005, info_18006)
    
    # Processing the call keyword arguments (line 297)
    kwargs_18008 = {}
    # Getting the type of 'LinAlgError' (line 297)
    LinAlgError_18004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 297)
    LinAlgError_call_result_18009 = invoke(stypy.reporting.localization.Localization(__file__, 297, 14), LinAlgError_18004, *[result_mod_18007], **kwargs_18008)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 297, 8), LinAlgError_call_result_18009, 'raise parameter', BaseException)
    # SSA join for if statement (line 296)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'info' (line 298)
    info_18010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 7), 'info')
    int_18011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 14), 'int')
    # Applying the binary operator '<' (line 298)
    result_lt_18012 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 7), '<', info_18010, int_18011)
    
    # Testing the type of an if condition (line 298)
    if_condition_18013 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 298, 4), result_lt_18012)
    # Assigning a type to the variable 'if_condition_18013' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'if_condition_18013', if_condition_18013)
    # SSA begins for if statement (line 298)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 299)
    # Processing the call arguments (line 299)
    str_18015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 25), 'str', 'illegal value in %d-th argument of internal pbtrs')
    
    # Getting the type of 'info' (line 300)
    info_18016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 28), 'info', False)
    # Applying the 'usub' unary operator (line 300)
    result___neg___18017 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 27), 'usub', info_18016)
    
    # Applying the binary operator '%' (line 299)
    result_mod_18018 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 25), '%', str_18015, result___neg___18017)
    
    # Processing the call keyword arguments (line 299)
    kwargs_18019 = {}
    # Getting the type of 'ValueError' (line 299)
    ValueError_18014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 299)
    ValueError_call_result_18020 = invoke(stypy.reporting.localization.Localization(__file__, 299, 14), ValueError_18014, *[result_mod_18018], **kwargs_18019)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 299, 8), ValueError_call_result_18020, 'raise parameter', BaseException)
    # SSA join for if statement (line 298)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 301)
    x_18021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'stypy_return_type', x_18021)
    
    # ################# End of 'cho_solve_banded(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cho_solve_banded' in the type store
    # Getting the type of 'stypy_return_type' (line 250)
    stypy_return_type_18022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18022)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cho_solve_banded'
    return stypy_return_type_18022

# Assigning a type to the variable 'cho_solve_banded' (line 250)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 0), 'cho_solve_banded', cho_solve_banded)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
