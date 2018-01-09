
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Schur decomposition functions.'''
2: from __future__ import division, print_function, absolute_import
3: 
4: import numpy
5: from numpy import asarray_chkfinite, single, asarray
6: 
7: from scipy._lib.six import callable
8: 
9: # Local imports.
10: from . import misc
11: from .misc import LinAlgError, _datacopied
12: from .lapack import get_lapack_funcs
13: from .decomp import eigvals
14: 
15: __all__ = ['schur', 'rsf2csf']
16: 
17: _double_precision = ['i','l','d']
18: 
19: 
20: def schur(a, output='real', lwork=None, overwrite_a=False, sort=None,
21:           check_finite=True):
22:     '''
23:     Compute Schur decomposition of a matrix.
24: 
25:     The Schur decomposition is::
26: 
27:         A = Z T Z^H
28: 
29:     where Z is unitary and T is either upper-triangular, or for real
30:     Schur decomposition (output='real'), quasi-upper triangular.  In
31:     the quasi-triangular form, 2x2 blocks describing complex-valued
32:     eigenvalue pairs may extrude from the diagonal.
33: 
34:     Parameters
35:     ----------
36:     a : (M, M) array_like
37:         Matrix to decompose
38:     output : {'real', 'complex'}, optional
39:         Construct the real or complex Schur decomposition (for real matrices).
40:     lwork : int, optional
41:         Work array size. If None or -1, it is automatically computed.
42:     overwrite_a : bool, optional
43:         Whether to overwrite data in a (may improve performance).
44:     sort : {None, callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional
45:         Specifies whether the upper eigenvalues should be sorted.  A callable
46:         may be passed that, given a eigenvalue, returns a boolean denoting
47:         whether the eigenvalue should be sorted to the top-left (True).
48:         Alternatively, string parameters may be used::
49: 
50:             'lhp'   Left-hand plane (x.real < 0.0)
51:             'rhp'   Right-hand plane (x.real > 0.0)
52:             'iuc'   Inside the unit circle (x*x.conjugate() <= 1.0)
53:             'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)
54: 
55:         Defaults to None (no sorting).
56:     check_finite : bool, optional
57:         Whether to check that the input matrix contains only finite numbers.
58:         Disabling may give a performance gain, but may result in problems
59:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
60: 
61:     Returns
62:     -------
63:     T : (M, M) ndarray
64:         Schur form of A. It is real-valued for the real Schur decomposition.
65:     Z : (M, M) ndarray
66:         An unitary Schur transformation matrix for A.
67:         It is real-valued for the real Schur decomposition.
68:     sdim : int
69:         If and only if sorting was requested, a third return value will
70:         contain the number of eigenvalues satisfying the sort condition.
71: 
72:     Raises
73:     ------
74:     LinAlgError
75:         Error raised under three conditions:
76: 
77:         1. The algorithm failed due to a failure of the QR algorithm to
78:            compute all eigenvalues
79:         2. If eigenvalue sorting was requested, the eigenvalues could not be
80:            reordered due to a failure to separate eigenvalues, usually because
81:            of poor conditioning
82:         3. If eigenvalue sorting was requested, roundoff errors caused the
83:            leading eigenvalues to no longer satisfy the sorting condition
84: 
85:     See also
86:     --------
87:     rsf2csf : Convert real Schur form to complex Schur form
88: 
89:     '''
90:     if output not in ['real','complex','r','c']:
91:         raise ValueError("argument must be 'real', or 'complex'")
92:     if check_finite:
93:         a1 = asarray_chkfinite(a)
94:     else:
95:         a1 = asarray(a)
96:     if len(a1.shape) != 2 or (a1.shape[0] != a1.shape[1]):
97:         raise ValueError('expected square matrix')
98:     typ = a1.dtype.char
99:     if output in ['complex','c'] and typ not in ['F','D']:
100:         if typ in _double_precision:
101:             a1 = a1.astype('D')
102:             typ = 'D'
103:         else:
104:             a1 = a1.astype('F')
105:             typ = 'F'
106:     overwrite_a = overwrite_a or (_datacopied(a1, a))
107:     gees, = get_lapack_funcs(('gees',), (a1,))
108:     if lwork is None or lwork == -1:
109:         # get optimal work array
110:         result = gees(lambda x: None, a1, lwork=-1)
111:         lwork = result[-2][0].real.astype(numpy.int)
112: 
113:     if sort is None:
114:         sort_t = 0
115:         sfunction = lambda x: None
116:     else:
117:         sort_t = 1
118:         if callable(sort):
119:             sfunction = sort
120:         elif sort == 'lhp':
121:             sfunction = lambda x: (numpy.real(x) < 0.0)
122:         elif sort == 'rhp':
123:             sfunction = lambda x: (numpy.real(x) >= 0.0)
124:         elif sort == 'iuc':
125:             sfunction = lambda x: (abs(x) <= 1.0)
126:         elif sort == 'ouc':
127:             sfunction = lambda x: (abs(x) > 1.0)
128:         else:
129:             raise ValueError("sort parameter must be None, a callable, or " +
130:                 "one of ('lhp','rhp','iuc','ouc')")
131: 
132:     result = gees(sfunction, a1, lwork=lwork, overwrite_a=overwrite_a,
133:         sort_t=sort_t)
134: 
135:     info = result[-1]
136:     if info < 0:
137:         raise ValueError('illegal value in %d-th argument of internal gees'
138:                                                                     % -info)
139:     elif info == a1.shape[0] + 1:
140:         raise LinAlgError('Eigenvalues could not be separated for reordering.')
141:     elif info == a1.shape[0] + 2:
142:         raise LinAlgError('Leading eigenvalues do not satisfy sort condition.')
143:     elif info > 0:
144:         raise LinAlgError("Schur form not found.  Possibly ill-conditioned.")
145: 
146:     if sort_t == 0:
147:         return result[0], result[-3]
148:     else:
149:         return result[0], result[-3], result[1]
150: 
151: 
152: eps = numpy.finfo(float).eps
153: feps = numpy.finfo(single).eps
154: 
155: _array_kind = {'b':0, 'h':0, 'B': 0, 'i':0, 'l': 0, 'f': 0, 'd': 0, 'F': 1, 'D': 1}
156: _array_precision = {'i': 1, 'l': 1, 'f': 0, 'd': 1, 'F': 0, 'D': 1}
157: _array_type = [['f', 'd'], ['F', 'D']]
158: 
159: 
160: def _commonType(*arrays):
161:     kind = 0
162:     precision = 0
163:     for a in arrays:
164:         t = a.dtype.char
165:         kind = max(kind, _array_kind[t])
166:         precision = max(precision, _array_precision[t])
167:     return _array_type[kind][precision]
168: 
169: 
170: def _castCopy(type, *arrays):
171:     cast_arrays = ()
172:     for a in arrays:
173:         if a.dtype.char == type:
174:             cast_arrays = cast_arrays + (a.copy(),)
175:         else:
176:             cast_arrays = cast_arrays + (a.astype(type),)
177:     if len(cast_arrays) == 1:
178:         return cast_arrays[0]
179:     else:
180:         return cast_arrays
181: 
182: 
183: def rsf2csf(T, Z, check_finite=True):
184:     '''
185:     Convert real Schur form to complex Schur form.
186: 
187:     Convert a quasi-diagonal real-valued Schur form to the upper triangular
188:     complex-valued Schur form.
189: 
190:     Parameters
191:     ----------
192:     T : (M, M) array_like
193:         Real Schur form of the original matrix
194:     Z : (M, M) array_like
195:         Schur transformation matrix
196:     check_finite : bool, optional
197:         Whether to check that the input matrices contain only finite numbers.
198:         Disabling may give a performance gain, but may result in problems
199:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
200: 
201:     Returns
202:     -------
203:     T : (M, M) ndarray
204:         Complex Schur form of the original matrix
205:     Z : (M, M) ndarray
206:         Schur transformation matrix corresponding to the complex form
207: 
208:     See also
209:     --------
210:     schur : Schur decompose a matrix
211: 
212:     '''
213:     if check_finite:
214:         Z, T = map(asarray_chkfinite, (Z, T))
215:     else:
216:         Z,T = map(asarray, (Z,T))
217:     if len(Z.shape) != 2 or Z.shape[0] != Z.shape[1]:
218:         raise ValueError("matrix must be square.")
219:     if len(T.shape) != 2 or T.shape[0] != T.shape[1]:
220:         raise ValueError("matrix must be square.")
221:     if T.shape[0] != Z.shape[0]:
222:         raise ValueError("matrices must be same dimension.")
223:     N = T.shape[0]
224:     arr = numpy.array
225:     t = _commonType(Z, T, arr([3.0],'F'))
226:     Z, T = _castCopy(t, Z, T)
227:     conj = numpy.conj
228:     dot = numpy.dot
229:     r_ = numpy.r_
230:     transp = numpy.transpose
231:     for m in range(N-1, 0, -1):
232:         if abs(T[m,m-1]) > eps*(abs(T[m-1,m-1]) + abs(T[m,m])):
233:             k = slice(m-1, m+1)
234:             mu = eigvals(T[k,k]) - T[m,m]
235:             r = misc.norm([mu[0], T[m,m-1]])
236:             c = mu[0] / r
237:             s = T[m,m-1] / r
238:             G = r_[arr([[conj(c), s]], dtype=t), arr([[-s, c]], dtype=t)]
239:             Gc = conj(transp(G))
240:             j = slice(m-1, N)
241:             T[k,j] = dot(G, T[k,j])
242:             i = slice(0, m+1)
243:             T[i,k] = dot(T[i,k], Gc)
244:             i = slice(0, N)
245:             Z[i,k] = dot(Z[i,k], Gc)
246:         T[m,m-1] = 0.0
247:     return T, Z
248: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_19298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Schur decomposition functions.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_19299 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_19299) is not StypyTypeError):

    if (import_19299 != 'pyd_module'):
        __import__(import_19299)
        sys_modules_19300 = sys.modules[import_19299]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', sys_modules_19300.module_type_store, module_type_store)
    else:
        import numpy

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_19299)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy import asarray_chkfinite, single, asarray' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_19301 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_19301) is not StypyTypeError):

    if (import_19301 != 'pyd_module'):
        __import__(import_19301)
        sys_modules_19302 = sys.modules[import_19301]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', sys_modules_19302.module_type_store, module_type_store, ['asarray_chkfinite', 'single', 'asarray'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_19302, sys_modules_19302.module_type_store, module_type_store)
    else:
        from numpy import asarray_chkfinite, single, asarray

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', None, module_type_store, ['asarray_chkfinite', 'single', 'asarray'], [asarray_chkfinite, single, asarray])

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_19301)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy._lib.six import callable' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_19303 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib.six')

if (type(import_19303) is not StypyTypeError):

    if (import_19303 != 'pyd_module'):
        __import__(import_19303)
        sys_modules_19304 = sys.modules[import_19303]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib.six', sys_modules_19304.module_type_store, module_type_store, ['callable'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_19304, sys_modules_19304.module_type_store, module_type_store)
    else:
        from scipy._lib.six import callable

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib.six', None, module_type_store, ['callable'], [callable])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib.six', import_19303)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.linalg import misc' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_19305 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.linalg')

if (type(import_19305) is not StypyTypeError):

    if (import_19305 != 'pyd_module'):
        __import__(import_19305)
        sys_modules_19306 = sys.modules[import_19305]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.linalg', sys_modules_19306.module_type_store, module_type_store, ['misc'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_19306, sys_modules_19306.module_type_store, module_type_store)
    else:
        from scipy.linalg import misc

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.linalg', None, module_type_store, ['misc'], [misc])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.linalg', import_19305)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.linalg.misc import LinAlgError, _datacopied' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_19307 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg.misc')

if (type(import_19307) is not StypyTypeError):

    if (import_19307 != 'pyd_module'):
        __import__(import_19307)
        sys_modules_19308 = sys.modules[import_19307]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg.misc', sys_modules_19308.module_type_store, module_type_store, ['LinAlgError', '_datacopied'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_19308, sys_modules_19308.module_type_store, module_type_store)
    else:
        from scipy.linalg.misc import LinAlgError, _datacopied

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg.misc', None, module_type_store, ['LinAlgError', '_datacopied'], [LinAlgError, _datacopied])

else:
    # Assigning a type to the variable 'scipy.linalg.misc' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg.misc', import_19307)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.linalg.lapack import get_lapack_funcs' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_19309 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.linalg.lapack')

if (type(import_19309) is not StypyTypeError):

    if (import_19309 != 'pyd_module'):
        __import__(import_19309)
        sys_modules_19310 = sys.modules[import_19309]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.linalg.lapack', sys_modules_19310.module_type_store, module_type_store, ['get_lapack_funcs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_19310, sys_modules_19310.module_type_store, module_type_store)
    else:
        from scipy.linalg.lapack import get_lapack_funcs

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.linalg.lapack', None, module_type_store, ['get_lapack_funcs'], [get_lapack_funcs])

else:
    # Assigning a type to the variable 'scipy.linalg.lapack' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.linalg.lapack', import_19309)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.linalg.decomp import eigvals' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_19311 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.linalg.decomp')

if (type(import_19311) is not StypyTypeError):

    if (import_19311 != 'pyd_module'):
        __import__(import_19311)
        sys_modules_19312 = sys.modules[import_19311]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.linalg.decomp', sys_modules_19312.module_type_store, module_type_store, ['eigvals'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_19312, sys_modules_19312.module_type_store, module_type_store)
    else:
        from scipy.linalg.decomp import eigvals

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.linalg.decomp', None, module_type_store, ['eigvals'], [eigvals])

else:
    # Assigning a type to the variable 'scipy.linalg.decomp' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.linalg.decomp', import_19311)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')


# Assigning a List to a Name (line 15):

# Assigning a List to a Name (line 15):
__all__ = ['schur', 'rsf2csf']
module_type_store.set_exportable_members(['schur', 'rsf2csf'])

# Obtaining an instance of the builtin type 'list' (line 15)
list_19313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
str_19314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 11), 'str', 'schur')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_19313, str_19314)
# Adding element type (line 15)
str_19315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 20), 'str', 'rsf2csf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_19313, str_19315)

# Assigning a type to the variable '__all__' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), '__all__', list_19313)

# Assigning a List to a Name (line 17):

# Assigning a List to a Name (line 17):

# Obtaining an instance of the builtin type 'list' (line 17)
list_19316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
str_19317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 21), 'str', 'i')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 20), list_19316, str_19317)
# Adding element type (line 17)
str_19318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 25), 'str', 'l')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 20), list_19316, str_19318)
# Adding element type (line 17)
str_19319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 29), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 20), list_19316, str_19319)

# Assigning a type to the variable '_double_precision' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), '_double_precision', list_19316)

@norecursion
def schur(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_19320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 20), 'str', 'real')
    # Getting the type of 'None' (line 20)
    None_19321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 34), 'None')
    # Getting the type of 'False' (line 20)
    False_19322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 52), 'False')
    # Getting the type of 'None' (line 20)
    None_19323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 64), 'None')
    # Getting the type of 'True' (line 21)
    True_19324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 23), 'True')
    defaults = [str_19320, None_19321, False_19322, None_19323, True_19324]
    # Create a new context for function 'schur'
    module_type_store = module_type_store.open_function_context('schur', 20, 0, False)
    
    # Passed parameters checking function
    schur.stypy_localization = localization
    schur.stypy_type_of_self = None
    schur.stypy_type_store = module_type_store
    schur.stypy_function_name = 'schur'
    schur.stypy_param_names_list = ['a', 'output', 'lwork', 'overwrite_a', 'sort', 'check_finite']
    schur.stypy_varargs_param_name = None
    schur.stypy_kwargs_param_name = None
    schur.stypy_call_defaults = defaults
    schur.stypy_call_varargs = varargs
    schur.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'schur', ['a', 'output', 'lwork', 'overwrite_a', 'sort', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'schur', localization, ['a', 'output', 'lwork', 'overwrite_a', 'sort', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'schur(...)' code ##################

    str_19325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, (-1)), 'str', "\n    Compute Schur decomposition of a matrix.\n\n    The Schur decomposition is::\n\n        A = Z T Z^H\n\n    where Z is unitary and T is either upper-triangular, or for real\n    Schur decomposition (output='real'), quasi-upper triangular.  In\n    the quasi-triangular form, 2x2 blocks describing complex-valued\n    eigenvalue pairs may extrude from the diagonal.\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        Matrix to decompose\n    output : {'real', 'complex'}, optional\n        Construct the real or complex Schur decomposition (for real matrices).\n    lwork : int, optional\n        Work array size. If None or -1, it is automatically computed.\n    overwrite_a : bool, optional\n        Whether to overwrite data in a (may improve performance).\n    sort : {None, callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional\n        Specifies whether the upper eigenvalues should be sorted.  A callable\n        may be passed that, given a eigenvalue, returns a boolean denoting\n        whether the eigenvalue should be sorted to the top-left (True).\n        Alternatively, string parameters may be used::\n\n            'lhp'   Left-hand plane (x.real < 0.0)\n            'rhp'   Right-hand plane (x.real > 0.0)\n            'iuc'   Inside the unit circle (x*x.conjugate() <= 1.0)\n            'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)\n\n        Defaults to None (no sorting).\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    T : (M, M) ndarray\n        Schur form of A. It is real-valued for the real Schur decomposition.\n    Z : (M, M) ndarray\n        An unitary Schur transformation matrix for A.\n        It is real-valued for the real Schur decomposition.\n    sdim : int\n        If and only if sorting was requested, a third return value will\n        contain the number of eigenvalues satisfying the sort condition.\n\n    Raises\n    ------\n    LinAlgError\n        Error raised under three conditions:\n\n        1. The algorithm failed due to a failure of the QR algorithm to\n           compute all eigenvalues\n        2. If eigenvalue sorting was requested, the eigenvalues could not be\n           reordered due to a failure to separate eigenvalues, usually because\n           of poor conditioning\n        3. If eigenvalue sorting was requested, roundoff errors caused the\n           leading eigenvalues to no longer satisfy the sorting condition\n\n    See also\n    --------\n    rsf2csf : Convert real Schur form to complex Schur form\n\n    ")
    
    
    # Getting the type of 'output' (line 90)
    output_19326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 7), 'output')
    
    # Obtaining an instance of the builtin type 'list' (line 90)
    list_19327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 90)
    # Adding element type (line 90)
    str_19328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 22), 'str', 'real')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 21), list_19327, str_19328)
    # Adding element type (line 90)
    str_19329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 29), 'str', 'complex')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 21), list_19327, str_19329)
    # Adding element type (line 90)
    str_19330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 39), 'str', 'r')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 21), list_19327, str_19330)
    # Adding element type (line 90)
    str_19331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 43), 'str', 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 21), list_19327, str_19331)
    
    # Applying the binary operator 'notin' (line 90)
    result_contains_19332 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 7), 'notin', output_19326, list_19327)
    
    # Testing the type of an if condition (line 90)
    if_condition_19333 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 4), result_contains_19332)
    # Assigning a type to the variable 'if_condition_19333' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'if_condition_19333', if_condition_19333)
    # SSA begins for if statement (line 90)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 91)
    # Processing the call arguments (line 91)
    str_19335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 25), 'str', "argument must be 'real', or 'complex'")
    # Processing the call keyword arguments (line 91)
    kwargs_19336 = {}
    # Getting the type of 'ValueError' (line 91)
    ValueError_19334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 91)
    ValueError_call_result_19337 = invoke(stypy.reporting.localization.Localization(__file__, 91, 14), ValueError_19334, *[str_19335], **kwargs_19336)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 91, 8), ValueError_call_result_19337, 'raise parameter', BaseException)
    # SSA join for if statement (line 90)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'check_finite' (line 92)
    check_finite_19338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 7), 'check_finite')
    # Testing the type of an if condition (line 92)
    if_condition_19339 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 4), check_finite_19338)
    # Assigning a type to the variable 'if_condition_19339' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'if_condition_19339', if_condition_19339)
    # SSA begins for if statement (line 92)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 93):
    
    # Assigning a Call to a Name (line 93):
    
    # Call to asarray_chkfinite(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'a' (line 93)
    a_19341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 31), 'a', False)
    # Processing the call keyword arguments (line 93)
    kwargs_19342 = {}
    # Getting the type of 'asarray_chkfinite' (line 93)
    asarray_chkfinite_19340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 13), 'asarray_chkfinite', False)
    # Calling asarray_chkfinite(args, kwargs) (line 93)
    asarray_chkfinite_call_result_19343 = invoke(stypy.reporting.localization.Localization(__file__, 93, 13), asarray_chkfinite_19340, *[a_19341], **kwargs_19342)
    
    # Assigning a type to the variable 'a1' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'a1', asarray_chkfinite_call_result_19343)
    # SSA branch for the else part of an if statement (line 92)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 95):
    
    # Assigning a Call to a Name (line 95):
    
    # Call to asarray(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'a' (line 95)
    a_19345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 21), 'a', False)
    # Processing the call keyword arguments (line 95)
    kwargs_19346 = {}
    # Getting the type of 'asarray' (line 95)
    asarray_19344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 13), 'asarray', False)
    # Calling asarray(args, kwargs) (line 95)
    asarray_call_result_19347 = invoke(stypy.reporting.localization.Localization(__file__, 95, 13), asarray_19344, *[a_19345], **kwargs_19346)
    
    # Assigning a type to the variable 'a1' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'a1', asarray_call_result_19347)
    # SSA join for if statement (line 92)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'a1' (line 96)
    a1_19349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 11), 'a1', False)
    # Obtaining the member 'shape' of a type (line 96)
    shape_19350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 11), a1_19349, 'shape')
    # Processing the call keyword arguments (line 96)
    kwargs_19351 = {}
    # Getting the type of 'len' (line 96)
    len_19348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 7), 'len', False)
    # Calling len(args, kwargs) (line 96)
    len_call_result_19352 = invoke(stypy.reporting.localization.Localization(__file__, 96, 7), len_19348, *[shape_19350], **kwargs_19351)
    
    int_19353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 24), 'int')
    # Applying the binary operator '!=' (line 96)
    result_ne_19354 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 7), '!=', len_call_result_19352, int_19353)
    
    
    
    # Obtaining the type of the subscript
    int_19355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 39), 'int')
    # Getting the type of 'a1' (line 96)
    a1_19356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 30), 'a1')
    # Obtaining the member 'shape' of a type (line 96)
    shape_19357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 30), a1_19356, 'shape')
    # Obtaining the member '__getitem__' of a type (line 96)
    getitem___19358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 30), shape_19357, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 96)
    subscript_call_result_19359 = invoke(stypy.reporting.localization.Localization(__file__, 96, 30), getitem___19358, int_19355)
    
    
    # Obtaining the type of the subscript
    int_19360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 54), 'int')
    # Getting the type of 'a1' (line 96)
    a1_19361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 45), 'a1')
    # Obtaining the member 'shape' of a type (line 96)
    shape_19362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 45), a1_19361, 'shape')
    # Obtaining the member '__getitem__' of a type (line 96)
    getitem___19363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 45), shape_19362, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 96)
    subscript_call_result_19364 = invoke(stypy.reporting.localization.Localization(__file__, 96, 45), getitem___19363, int_19360)
    
    # Applying the binary operator '!=' (line 96)
    result_ne_19365 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 30), '!=', subscript_call_result_19359, subscript_call_result_19364)
    
    # Applying the binary operator 'or' (line 96)
    result_or_keyword_19366 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 7), 'or', result_ne_19354, result_ne_19365)
    
    # Testing the type of an if condition (line 96)
    if_condition_19367 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 4), result_or_keyword_19366)
    # Assigning a type to the variable 'if_condition_19367' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'if_condition_19367', if_condition_19367)
    # SSA begins for if statement (line 96)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 97)
    # Processing the call arguments (line 97)
    str_19369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 25), 'str', 'expected square matrix')
    # Processing the call keyword arguments (line 97)
    kwargs_19370 = {}
    # Getting the type of 'ValueError' (line 97)
    ValueError_19368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 97)
    ValueError_call_result_19371 = invoke(stypy.reporting.localization.Localization(__file__, 97, 14), ValueError_19368, *[str_19369], **kwargs_19370)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 97, 8), ValueError_call_result_19371, 'raise parameter', BaseException)
    # SSA join for if statement (line 96)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 98):
    
    # Assigning a Attribute to a Name (line 98):
    # Getting the type of 'a1' (line 98)
    a1_19372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 10), 'a1')
    # Obtaining the member 'dtype' of a type (line 98)
    dtype_19373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 10), a1_19372, 'dtype')
    # Obtaining the member 'char' of a type (line 98)
    char_19374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 10), dtype_19373, 'char')
    # Assigning a type to the variable 'typ' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'typ', char_19374)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'output' (line 99)
    output_19375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 7), 'output')
    
    # Obtaining an instance of the builtin type 'list' (line 99)
    list_19376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 99)
    # Adding element type (line 99)
    str_19377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 18), 'str', 'complex')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 17), list_19376, str_19377)
    # Adding element type (line 99)
    str_19378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 28), 'str', 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 17), list_19376, str_19378)
    
    # Applying the binary operator 'in' (line 99)
    result_contains_19379 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 7), 'in', output_19375, list_19376)
    
    
    # Getting the type of 'typ' (line 99)
    typ_19380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 37), 'typ')
    
    # Obtaining an instance of the builtin type 'list' (line 99)
    list_19381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 48), 'list')
    # Adding type elements to the builtin type 'list' instance (line 99)
    # Adding element type (line 99)
    str_19382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 49), 'str', 'F')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 48), list_19381, str_19382)
    # Adding element type (line 99)
    str_19383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 53), 'str', 'D')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 48), list_19381, str_19383)
    
    # Applying the binary operator 'notin' (line 99)
    result_contains_19384 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 37), 'notin', typ_19380, list_19381)
    
    # Applying the binary operator 'and' (line 99)
    result_and_keyword_19385 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 7), 'and', result_contains_19379, result_contains_19384)
    
    # Testing the type of an if condition (line 99)
    if_condition_19386 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 4), result_and_keyword_19385)
    # Assigning a type to the variable 'if_condition_19386' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'if_condition_19386', if_condition_19386)
    # SSA begins for if statement (line 99)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'typ' (line 100)
    typ_19387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 11), 'typ')
    # Getting the type of '_double_precision' (line 100)
    _double_precision_19388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 18), '_double_precision')
    # Applying the binary operator 'in' (line 100)
    result_contains_19389 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 11), 'in', typ_19387, _double_precision_19388)
    
    # Testing the type of an if condition (line 100)
    if_condition_19390 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 8), result_contains_19389)
    # Assigning a type to the variable 'if_condition_19390' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'if_condition_19390', if_condition_19390)
    # SSA begins for if statement (line 100)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 101):
    
    # Assigning a Call to a Name (line 101):
    
    # Call to astype(...): (line 101)
    # Processing the call arguments (line 101)
    str_19393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 27), 'str', 'D')
    # Processing the call keyword arguments (line 101)
    kwargs_19394 = {}
    # Getting the type of 'a1' (line 101)
    a1_19391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 'a1', False)
    # Obtaining the member 'astype' of a type (line 101)
    astype_19392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 17), a1_19391, 'astype')
    # Calling astype(args, kwargs) (line 101)
    astype_call_result_19395 = invoke(stypy.reporting.localization.Localization(__file__, 101, 17), astype_19392, *[str_19393], **kwargs_19394)
    
    # Assigning a type to the variable 'a1' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'a1', astype_call_result_19395)
    
    # Assigning a Str to a Name (line 102):
    
    # Assigning a Str to a Name (line 102):
    str_19396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 18), 'str', 'D')
    # Assigning a type to the variable 'typ' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'typ', str_19396)
    # SSA branch for the else part of an if statement (line 100)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 104):
    
    # Assigning a Call to a Name (line 104):
    
    # Call to astype(...): (line 104)
    # Processing the call arguments (line 104)
    str_19399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 27), 'str', 'F')
    # Processing the call keyword arguments (line 104)
    kwargs_19400 = {}
    # Getting the type of 'a1' (line 104)
    a1_19397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 17), 'a1', False)
    # Obtaining the member 'astype' of a type (line 104)
    astype_19398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 17), a1_19397, 'astype')
    # Calling astype(args, kwargs) (line 104)
    astype_call_result_19401 = invoke(stypy.reporting.localization.Localization(__file__, 104, 17), astype_19398, *[str_19399], **kwargs_19400)
    
    # Assigning a type to the variable 'a1' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'a1', astype_call_result_19401)
    
    # Assigning a Str to a Name (line 105):
    
    # Assigning a Str to a Name (line 105):
    str_19402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 18), 'str', 'F')
    # Assigning a type to the variable 'typ' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'typ', str_19402)
    # SSA join for if statement (line 100)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 99)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 106):
    
    # Assigning a BoolOp to a Name (line 106):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_a' (line 106)
    overwrite_a_19403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), 'overwrite_a')
    
    # Call to _datacopied(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'a1' (line 106)
    a1_19405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 46), 'a1', False)
    # Getting the type of 'a' (line 106)
    a_19406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 50), 'a', False)
    # Processing the call keyword arguments (line 106)
    kwargs_19407 = {}
    # Getting the type of '_datacopied' (line 106)
    _datacopied_19404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 34), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 106)
    _datacopied_call_result_19408 = invoke(stypy.reporting.localization.Localization(__file__, 106, 34), _datacopied_19404, *[a1_19405, a_19406], **kwargs_19407)
    
    # Applying the binary operator 'or' (line 106)
    result_or_keyword_19409 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 18), 'or', overwrite_a_19403, _datacopied_call_result_19408)
    
    # Assigning a type to the variable 'overwrite_a' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'overwrite_a', result_or_keyword_19409)
    
    # Assigning a Call to a Tuple (line 107):
    
    # Assigning a Subscript to a Name (line 107):
    
    # Obtaining the type of the subscript
    int_19410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 107)
    # Processing the call arguments (line 107)
    
    # Obtaining an instance of the builtin type 'tuple' (line 107)
    tuple_19412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 107)
    # Adding element type (line 107)
    str_19413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 30), 'str', 'gees')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 30), tuple_19412, str_19413)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 107)
    tuple_19414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 107)
    # Adding element type (line 107)
    # Getting the type of 'a1' (line 107)
    a1_19415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 41), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 41), tuple_19414, a1_19415)
    
    # Processing the call keyword arguments (line 107)
    kwargs_19416 = {}
    # Getting the type of 'get_lapack_funcs' (line 107)
    get_lapack_funcs_19411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 107)
    get_lapack_funcs_call_result_19417 = invoke(stypy.reporting.localization.Localization(__file__, 107, 12), get_lapack_funcs_19411, *[tuple_19412, tuple_19414], **kwargs_19416)
    
    # Obtaining the member '__getitem__' of a type (line 107)
    getitem___19418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 4), get_lapack_funcs_call_result_19417, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 107)
    subscript_call_result_19419 = invoke(stypy.reporting.localization.Localization(__file__, 107, 4), getitem___19418, int_19410)
    
    # Assigning a type to the variable 'tuple_var_assignment_19291' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'tuple_var_assignment_19291', subscript_call_result_19419)
    
    # Assigning a Name to a Name (line 107):
    # Getting the type of 'tuple_var_assignment_19291' (line 107)
    tuple_var_assignment_19291_19420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'tuple_var_assignment_19291')
    # Assigning a type to the variable 'gees' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'gees', tuple_var_assignment_19291_19420)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'lwork' (line 108)
    lwork_19421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 7), 'lwork')
    # Getting the type of 'None' (line 108)
    None_19422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'None')
    # Applying the binary operator 'is' (line 108)
    result_is__19423 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 7), 'is', lwork_19421, None_19422)
    
    
    # Getting the type of 'lwork' (line 108)
    lwork_19424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 24), 'lwork')
    int_19425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 33), 'int')
    # Applying the binary operator '==' (line 108)
    result_eq_19426 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 24), '==', lwork_19424, int_19425)
    
    # Applying the binary operator 'or' (line 108)
    result_or_keyword_19427 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 7), 'or', result_is__19423, result_eq_19426)
    
    # Testing the type of an if condition (line 108)
    if_condition_19428 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 4), result_or_keyword_19427)
    # Assigning a type to the variable 'if_condition_19428' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'if_condition_19428', if_condition_19428)
    # SSA begins for if statement (line 108)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 110):
    
    # Assigning a Call to a Name (line 110):
    
    # Call to gees(...): (line 110)
    # Processing the call arguments (line 110)

    @norecursion
    def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_1'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 110, 22, True)
        # Passed parameters checking function
        _stypy_temp_lambda_1.stypy_localization = localization
        _stypy_temp_lambda_1.stypy_type_of_self = None
        _stypy_temp_lambda_1.stypy_type_store = module_type_store
        _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
        _stypy_temp_lambda_1.stypy_param_names_list = ['x']
        _stypy_temp_lambda_1.stypy_varargs_param_name = None
        _stypy_temp_lambda_1.stypy_kwargs_param_name = None
        _stypy_temp_lambda_1.stypy_call_defaults = defaults
        _stypy_temp_lambda_1.stypy_call_varargs = varargs
        _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_1', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        # Getting the type of 'None' (line 110)
        None_19430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 32), 'None', False)
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 22), 'stypy_return_type', None_19430)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_1' in the type store
        # Getting the type of 'stypy_return_type' (line 110)
        stypy_return_type_19431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 22), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19431)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_1'
        return stypy_return_type_19431

    # Assigning a type to the variable '_stypy_temp_lambda_1' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 22), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
    # Getting the type of '_stypy_temp_lambda_1' (line 110)
    _stypy_temp_lambda_1_19432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 22), '_stypy_temp_lambda_1')
    # Getting the type of 'a1' (line 110)
    a1_19433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 38), 'a1', False)
    # Processing the call keyword arguments (line 110)
    int_19434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 48), 'int')
    keyword_19435 = int_19434
    kwargs_19436 = {'lwork': keyword_19435}
    # Getting the type of 'gees' (line 110)
    gees_19429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 17), 'gees', False)
    # Calling gees(args, kwargs) (line 110)
    gees_call_result_19437 = invoke(stypy.reporting.localization.Localization(__file__, 110, 17), gees_19429, *[_stypy_temp_lambda_1_19432, a1_19433], **kwargs_19436)
    
    # Assigning a type to the variable 'result' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'result', gees_call_result_19437)
    
    # Assigning a Call to a Name (line 111):
    
    # Assigning a Call to a Name (line 111):
    
    # Call to astype(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'numpy' (line 111)
    numpy_19447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 42), 'numpy', False)
    # Obtaining the member 'int' of a type (line 111)
    int_19448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 42), numpy_19447, 'int')
    # Processing the call keyword arguments (line 111)
    kwargs_19449 = {}
    
    # Obtaining the type of the subscript
    int_19438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 27), 'int')
    
    # Obtaining the type of the subscript
    int_19439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 23), 'int')
    # Getting the type of 'result' (line 111)
    result_19440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'result', False)
    # Obtaining the member '__getitem__' of a type (line 111)
    getitem___19441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 16), result_19440, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
    subscript_call_result_19442 = invoke(stypy.reporting.localization.Localization(__file__, 111, 16), getitem___19441, int_19439)
    
    # Obtaining the member '__getitem__' of a type (line 111)
    getitem___19443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 16), subscript_call_result_19442, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
    subscript_call_result_19444 = invoke(stypy.reporting.localization.Localization(__file__, 111, 16), getitem___19443, int_19438)
    
    # Obtaining the member 'real' of a type (line 111)
    real_19445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 16), subscript_call_result_19444, 'real')
    # Obtaining the member 'astype' of a type (line 111)
    astype_19446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 16), real_19445, 'astype')
    # Calling astype(args, kwargs) (line 111)
    astype_call_result_19450 = invoke(stypy.reporting.localization.Localization(__file__, 111, 16), astype_19446, *[int_19448], **kwargs_19449)
    
    # Assigning a type to the variable 'lwork' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'lwork', astype_call_result_19450)
    # SSA join for if statement (line 108)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 113)
    # Getting the type of 'sort' (line 113)
    sort_19451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 7), 'sort')
    # Getting the type of 'None' (line 113)
    None_19452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 15), 'None')
    
    (may_be_19453, more_types_in_union_19454) = may_be_none(sort_19451, None_19452)

    if may_be_19453:

        if more_types_in_union_19454:
            # Runtime conditional SSA (line 113)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Num to a Name (line 114):
        
        # Assigning a Num to a Name (line 114):
        int_19455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 17), 'int')
        # Assigning a type to the variable 'sort_t' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'sort_t', int_19455)
        
        # Assigning a Lambda to a Name (line 115):
        
        # Assigning a Lambda to a Name (line 115):

        @norecursion
        def _stypy_temp_lambda_2(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_2'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_2', 115, 20, True)
            # Passed parameters checking function
            _stypy_temp_lambda_2.stypy_localization = localization
            _stypy_temp_lambda_2.stypy_type_of_self = None
            _stypy_temp_lambda_2.stypy_type_store = module_type_store
            _stypy_temp_lambda_2.stypy_function_name = '_stypy_temp_lambda_2'
            _stypy_temp_lambda_2.stypy_param_names_list = ['x']
            _stypy_temp_lambda_2.stypy_varargs_param_name = None
            _stypy_temp_lambda_2.stypy_kwargs_param_name = None
            _stypy_temp_lambda_2.stypy_call_defaults = defaults
            _stypy_temp_lambda_2.stypy_call_varargs = varargs
            _stypy_temp_lambda_2.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_2', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_2', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'None' (line 115)
            None_19456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 30), 'None')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 20), 'stypy_return_type', None_19456)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_2' in the type store
            # Getting the type of 'stypy_return_type' (line 115)
            stypy_return_type_19457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 20), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_19457)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_2'
            return stypy_return_type_19457

        # Assigning a type to the variable '_stypy_temp_lambda_2' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 20), '_stypy_temp_lambda_2', _stypy_temp_lambda_2)
        # Getting the type of '_stypy_temp_lambda_2' (line 115)
        _stypy_temp_lambda_2_19458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 20), '_stypy_temp_lambda_2')
        # Assigning a type to the variable 'sfunction' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'sfunction', _stypy_temp_lambda_2_19458)

        if more_types_in_union_19454:
            # Runtime conditional SSA for else branch (line 113)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_19453) or more_types_in_union_19454):
        
        # Assigning a Num to a Name (line 117):
        
        # Assigning a Num to a Name (line 117):
        int_19459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 17), 'int')
        # Assigning a type to the variable 'sort_t' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'sort_t', int_19459)
        
        
        # Call to callable(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'sort' (line 118)
        sort_19461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 20), 'sort', False)
        # Processing the call keyword arguments (line 118)
        kwargs_19462 = {}
        # Getting the type of 'callable' (line 118)
        callable_19460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 11), 'callable', False)
        # Calling callable(args, kwargs) (line 118)
        callable_call_result_19463 = invoke(stypy.reporting.localization.Localization(__file__, 118, 11), callable_19460, *[sort_19461], **kwargs_19462)
        
        # Testing the type of an if condition (line 118)
        if_condition_19464 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 8), callable_call_result_19463)
        # Assigning a type to the variable 'if_condition_19464' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'if_condition_19464', if_condition_19464)
        # SSA begins for if statement (line 118)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 119):
        
        # Assigning a Name to a Name (line 119):
        # Getting the type of 'sort' (line 119)
        sort_19465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 24), 'sort')
        # Assigning a type to the variable 'sfunction' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'sfunction', sort_19465)
        # SSA branch for the else part of an if statement (line 118)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'sort' (line 120)
        sort_19466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 13), 'sort')
        str_19467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 21), 'str', 'lhp')
        # Applying the binary operator '==' (line 120)
        result_eq_19468 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 13), '==', sort_19466, str_19467)
        
        # Testing the type of an if condition (line 120)
        if_condition_19469 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 120, 13), result_eq_19468)
        # Assigning a type to the variable 'if_condition_19469' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 13), 'if_condition_19469', if_condition_19469)
        # SSA begins for if statement (line 120)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Lambda to a Name (line 121):
        
        # Assigning a Lambda to a Name (line 121):

        @norecursion
        def _stypy_temp_lambda_3(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_3'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_3', 121, 24, True)
            # Passed parameters checking function
            _stypy_temp_lambda_3.stypy_localization = localization
            _stypy_temp_lambda_3.stypy_type_of_self = None
            _stypy_temp_lambda_3.stypy_type_store = module_type_store
            _stypy_temp_lambda_3.stypy_function_name = '_stypy_temp_lambda_3'
            _stypy_temp_lambda_3.stypy_param_names_list = ['x']
            _stypy_temp_lambda_3.stypy_varargs_param_name = None
            _stypy_temp_lambda_3.stypy_kwargs_param_name = None
            _stypy_temp_lambda_3.stypy_call_defaults = defaults
            _stypy_temp_lambda_3.stypy_call_varargs = varargs
            _stypy_temp_lambda_3.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_3', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_3', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            
            # Call to real(...): (line 121)
            # Processing the call arguments (line 121)
            # Getting the type of 'x' (line 121)
            x_19472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 46), 'x', False)
            # Processing the call keyword arguments (line 121)
            kwargs_19473 = {}
            # Getting the type of 'numpy' (line 121)
            numpy_19470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 35), 'numpy', False)
            # Obtaining the member 'real' of a type (line 121)
            real_19471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 35), numpy_19470, 'real')
            # Calling real(args, kwargs) (line 121)
            real_call_result_19474 = invoke(stypy.reporting.localization.Localization(__file__, 121, 35), real_19471, *[x_19472], **kwargs_19473)
            
            float_19475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 51), 'float')
            # Applying the binary operator '<' (line 121)
            result_lt_19476 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 35), '<', real_call_result_19474, float_19475)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 121)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 24), 'stypy_return_type', result_lt_19476)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_3' in the type store
            # Getting the type of 'stypy_return_type' (line 121)
            stypy_return_type_19477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 24), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_19477)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_3'
            return stypy_return_type_19477

        # Assigning a type to the variable '_stypy_temp_lambda_3' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 24), '_stypy_temp_lambda_3', _stypy_temp_lambda_3)
        # Getting the type of '_stypy_temp_lambda_3' (line 121)
        _stypy_temp_lambda_3_19478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 24), '_stypy_temp_lambda_3')
        # Assigning a type to the variable 'sfunction' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'sfunction', _stypy_temp_lambda_3_19478)
        # SSA branch for the else part of an if statement (line 120)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'sort' (line 122)
        sort_19479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 13), 'sort')
        str_19480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 21), 'str', 'rhp')
        # Applying the binary operator '==' (line 122)
        result_eq_19481 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 13), '==', sort_19479, str_19480)
        
        # Testing the type of an if condition (line 122)
        if_condition_19482 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 13), result_eq_19481)
        # Assigning a type to the variable 'if_condition_19482' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 13), 'if_condition_19482', if_condition_19482)
        # SSA begins for if statement (line 122)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Lambda to a Name (line 123):
        
        # Assigning a Lambda to a Name (line 123):

        @norecursion
        def _stypy_temp_lambda_4(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_4'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_4', 123, 24, True)
            # Passed parameters checking function
            _stypy_temp_lambda_4.stypy_localization = localization
            _stypy_temp_lambda_4.stypy_type_of_self = None
            _stypy_temp_lambda_4.stypy_type_store = module_type_store
            _stypy_temp_lambda_4.stypy_function_name = '_stypy_temp_lambda_4'
            _stypy_temp_lambda_4.stypy_param_names_list = ['x']
            _stypy_temp_lambda_4.stypy_varargs_param_name = None
            _stypy_temp_lambda_4.stypy_kwargs_param_name = None
            _stypy_temp_lambda_4.stypy_call_defaults = defaults
            _stypy_temp_lambda_4.stypy_call_varargs = varargs
            _stypy_temp_lambda_4.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_4', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_4', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            
            # Call to real(...): (line 123)
            # Processing the call arguments (line 123)
            # Getting the type of 'x' (line 123)
            x_19485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 46), 'x', False)
            # Processing the call keyword arguments (line 123)
            kwargs_19486 = {}
            # Getting the type of 'numpy' (line 123)
            numpy_19483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 35), 'numpy', False)
            # Obtaining the member 'real' of a type (line 123)
            real_19484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 35), numpy_19483, 'real')
            # Calling real(args, kwargs) (line 123)
            real_call_result_19487 = invoke(stypy.reporting.localization.Localization(__file__, 123, 35), real_19484, *[x_19485], **kwargs_19486)
            
            float_19488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 52), 'float')
            # Applying the binary operator '>=' (line 123)
            result_ge_19489 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 35), '>=', real_call_result_19487, float_19488)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 123)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 24), 'stypy_return_type', result_ge_19489)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_4' in the type store
            # Getting the type of 'stypy_return_type' (line 123)
            stypy_return_type_19490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 24), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_19490)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_4'
            return stypy_return_type_19490

        # Assigning a type to the variable '_stypy_temp_lambda_4' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 24), '_stypy_temp_lambda_4', _stypy_temp_lambda_4)
        # Getting the type of '_stypy_temp_lambda_4' (line 123)
        _stypy_temp_lambda_4_19491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 24), '_stypy_temp_lambda_4')
        # Assigning a type to the variable 'sfunction' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'sfunction', _stypy_temp_lambda_4_19491)
        # SSA branch for the else part of an if statement (line 122)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'sort' (line 124)
        sort_19492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 13), 'sort')
        str_19493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 21), 'str', 'iuc')
        # Applying the binary operator '==' (line 124)
        result_eq_19494 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 13), '==', sort_19492, str_19493)
        
        # Testing the type of an if condition (line 124)
        if_condition_19495 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 13), result_eq_19494)
        # Assigning a type to the variable 'if_condition_19495' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 13), 'if_condition_19495', if_condition_19495)
        # SSA begins for if statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Lambda to a Name (line 125):
        
        # Assigning a Lambda to a Name (line 125):

        @norecursion
        def _stypy_temp_lambda_5(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_5'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_5', 125, 24, True)
            # Passed parameters checking function
            _stypy_temp_lambda_5.stypy_localization = localization
            _stypy_temp_lambda_5.stypy_type_of_self = None
            _stypy_temp_lambda_5.stypy_type_store = module_type_store
            _stypy_temp_lambda_5.stypy_function_name = '_stypy_temp_lambda_5'
            _stypy_temp_lambda_5.stypy_param_names_list = ['x']
            _stypy_temp_lambda_5.stypy_varargs_param_name = None
            _stypy_temp_lambda_5.stypy_kwargs_param_name = None
            _stypy_temp_lambda_5.stypy_call_defaults = defaults
            _stypy_temp_lambda_5.stypy_call_varargs = varargs
            _stypy_temp_lambda_5.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_5', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_5', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            
            # Call to abs(...): (line 125)
            # Processing the call arguments (line 125)
            # Getting the type of 'x' (line 125)
            x_19497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 39), 'x', False)
            # Processing the call keyword arguments (line 125)
            kwargs_19498 = {}
            # Getting the type of 'abs' (line 125)
            abs_19496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 35), 'abs', False)
            # Calling abs(args, kwargs) (line 125)
            abs_call_result_19499 = invoke(stypy.reporting.localization.Localization(__file__, 125, 35), abs_19496, *[x_19497], **kwargs_19498)
            
            float_19500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 45), 'float')
            # Applying the binary operator '<=' (line 125)
            result_le_19501 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 35), '<=', abs_call_result_19499, float_19500)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 24), 'stypy_return_type', result_le_19501)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_5' in the type store
            # Getting the type of 'stypy_return_type' (line 125)
            stypy_return_type_19502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 24), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_19502)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_5'
            return stypy_return_type_19502

        # Assigning a type to the variable '_stypy_temp_lambda_5' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 24), '_stypy_temp_lambda_5', _stypy_temp_lambda_5)
        # Getting the type of '_stypy_temp_lambda_5' (line 125)
        _stypy_temp_lambda_5_19503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 24), '_stypy_temp_lambda_5')
        # Assigning a type to the variable 'sfunction' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'sfunction', _stypy_temp_lambda_5_19503)
        # SSA branch for the else part of an if statement (line 124)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'sort' (line 126)
        sort_19504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 13), 'sort')
        str_19505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 21), 'str', 'ouc')
        # Applying the binary operator '==' (line 126)
        result_eq_19506 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 13), '==', sort_19504, str_19505)
        
        # Testing the type of an if condition (line 126)
        if_condition_19507 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 13), result_eq_19506)
        # Assigning a type to the variable 'if_condition_19507' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 13), 'if_condition_19507', if_condition_19507)
        # SSA begins for if statement (line 126)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Lambda to a Name (line 127):
        
        # Assigning a Lambda to a Name (line 127):

        @norecursion
        def _stypy_temp_lambda_6(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_6'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_6', 127, 24, True)
            # Passed parameters checking function
            _stypy_temp_lambda_6.stypy_localization = localization
            _stypy_temp_lambda_6.stypy_type_of_self = None
            _stypy_temp_lambda_6.stypy_type_store = module_type_store
            _stypy_temp_lambda_6.stypy_function_name = '_stypy_temp_lambda_6'
            _stypy_temp_lambda_6.stypy_param_names_list = ['x']
            _stypy_temp_lambda_6.stypy_varargs_param_name = None
            _stypy_temp_lambda_6.stypy_kwargs_param_name = None
            _stypy_temp_lambda_6.stypy_call_defaults = defaults
            _stypy_temp_lambda_6.stypy_call_varargs = varargs
            _stypy_temp_lambda_6.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_6', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_6', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            
            # Call to abs(...): (line 127)
            # Processing the call arguments (line 127)
            # Getting the type of 'x' (line 127)
            x_19509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 39), 'x', False)
            # Processing the call keyword arguments (line 127)
            kwargs_19510 = {}
            # Getting the type of 'abs' (line 127)
            abs_19508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 35), 'abs', False)
            # Calling abs(args, kwargs) (line 127)
            abs_call_result_19511 = invoke(stypy.reporting.localization.Localization(__file__, 127, 35), abs_19508, *[x_19509], **kwargs_19510)
            
            float_19512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 44), 'float')
            # Applying the binary operator '>' (line 127)
            result_gt_19513 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 35), '>', abs_call_result_19511, float_19512)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 127)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 24), 'stypy_return_type', result_gt_19513)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_6' in the type store
            # Getting the type of 'stypy_return_type' (line 127)
            stypy_return_type_19514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 24), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_19514)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_6'
            return stypy_return_type_19514

        # Assigning a type to the variable '_stypy_temp_lambda_6' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 24), '_stypy_temp_lambda_6', _stypy_temp_lambda_6)
        # Getting the type of '_stypy_temp_lambda_6' (line 127)
        _stypy_temp_lambda_6_19515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 24), '_stypy_temp_lambda_6')
        # Assigning a type to the variable 'sfunction' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'sfunction', _stypy_temp_lambda_6_19515)
        # SSA branch for the else part of an if statement (line 126)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 129)
        # Processing the call arguments (line 129)
        str_19517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 29), 'str', 'sort parameter must be None, a callable, or ')
        str_19518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 16), 'str', "one of ('lhp','rhp','iuc','ouc')")
        # Applying the binary operator '+' (line 129)
        result_add_19519 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 29), '+', str_19517, str_19518)
        
        # Processing the call keyword arguments (line 129)
        kwargs_19520 = {}
        # Getting the type of 'ValueError' (line 129)
        ValueError_19516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 129)
        ValueError_call_result_19521 = invoke(stypy.reporting.localization.Localization(__file__, 129, 18), ValueError_19516, *[result_add_19519], **kwargs_19520)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 129, 12), ValueError_call_result_19521, 'raise parameter', BaseException)
        # SSA join for if statement (line 126)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 124)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 122)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 120)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 118)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_19453 and more_types_in_union_19454):
            # SSA join for if statement (line 113)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 132):
    
    # Assigning a Call to a Name (line 132):
    
    # Call to gees(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'sfunction' (line 132)
    sfunction_19523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 18), 'sfunction', False)
    # Getting the type of 'a1' (line 132)
    a1_19524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 29), 'a1', False)
    # Processing the call keyword arguments (line 132)
    # Getting the type of 'lwork' (line 132)
    lwork_19525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 39), 'lwork', False)
    keyword_19526 = lwork_19525
    # Getting the type of 'overwrite_a' (line 132)
    overwrite_a_19527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 58), 'overwrite_a', False)
    keyword_19528 = overwrite_a_19527
    # Getting the type of 'sort_t' (line 133)
    sort_t_19529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'sort_t', False)
    keyword_19530 = sort_t_19529
    kwargs_19531 = {'sort_t': keyword_19530, 'overwrite_a': keyword_19528, 'lwork': keyword_19526}
    # Getting the type of 'gees' (line 132)
    gees_19522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 13), 'gees', False)
    # Calling gees(args, kwargs) (line 132)
    gees_call_result_19532 = invoke(stypy.reporting.localization.Localization(__file__, 132, 13), gees_19522, *[sfunction_19523, a1_19524], **kwargs_19531)
    
    # Assigning a type to the variable 'result' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'result', gees_call_result_19532)
    
    # Assigning a Subscript to a Name (line 135):
    
    # Assigning a Subscript to a Name (line 135):
    
    # Obtaining the type of the subscript
    int_19533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 18), 'int')
    # Getting the type of 'result' (line 135)
    result_19534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 11), 'result')
    # Obtaining the member '__getitem__' of a type (line 135)
    getitem___19535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 11), result_19534, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 135)
    subscript_call_result_19536 = invoke(stypy.reporting.localization.Localization(__file__, 135, 11), getitem___19535, int_19533)
    
    # Assigning a type to the variable 'info' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'info', subscript_call_result_19536)
    
    
    # Getting the type of 'info' (line 136)
    info_19537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 7), 'info')
    int_19538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 14), 'int')
    # Applying the binary operator '<' (line 136)
    result_lt_19539 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 7), '<', info_19537, int_19538)
    
    # Testing the type of an if condition (line 136)
    if_condition_19540 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 136, 4), result_lt_19539)
    # Assigning a type to the variable 'if_condition_19540' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'if_condition_19540', if_condition_19540)
    # SSA begins for if statement (line 136)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 137)
    # Processing the call arguments (line 137)
    str_19542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 25), 'str', 'illegal value in %d-th argument of internal gees')
    
    # Getting the type of 'info' (line 138)
    info_19543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 71), 'info', False)
    # Applying the 'usub' unary operator (line 138)
    result___neg___19544 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 70), 'usub', info_19543)
    
    # Applying the binary operator '%' (line 137)
    result_mod_19545 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 25), '%', str_19542, result___neg___19544)
    
    # Processing the call keyword arguments (line 137)
    kwargs_19546 = {}
    # Getting the type of 'ValueError' (line 137)
    ValueError_19541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 137)
    ValueError_call_result_19547 = invoke(stypy.reporting.localization.Localization(__file__, 137, 14), ValueError_19541, *[result_mod_19545], **kwargs_19546)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 137, 8), ValueError_call_result_19547, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 136)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'info' (line 139)
    info_19548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 9), 'info')
    
    # Obtaining the type of the subscript
    int_19549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 26), 'int')
    # Getting the type of 'a1' (line 139)
    a1_19550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 17), 'a1')
    # Obtaining the member 'shape' of a type (line 139)
    shape_19551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 17), a1_19550, 'shape')
    # Obtaining the member '__getitem__' of a type (line 139)
    getitem___19552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 17), shape_19551, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 139)
    subscript_call_result_19553 = invoke(stypy.reporting.localization.Localization(__file__, 139, 17), getitem___19552, int_19549)
    
    int_19554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 31), 'int')
    # Applying the binary operator '+' (line 139)
    result_add_19555 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 17), '+', subscript_call_result_19553, int_19554)
    
    # Applying the binary operator '==' (line 139)
    result_eq_19556 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 9), '==', info_19548, result_add_19555)
    
    # Testing the type of an if condition (line 139)
    if_condition_19557 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 9), result_eq_19556)
    # Assigning a type to the variable 'if_condition_19557' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 9), 'if_condition_19557', if_condition_19557)
    # SSA begins for if statement (line 139)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 140)
    # Processing the call arguments (line 140)
    str_19559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 26), 'str', 'Eigenvalues could not be separated for reordering.')
    # Processing the call keyword arguments (line 140)
    kwargs_19560 = {}
    # Getting the type of 'LinAlgError' (line 140)
    LinAlgError_19558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 140)
    LinAlgError_call_result_19561 = invoke(stypy.reporting.localization.Localization(__file__, 140, 14), LinAlgError_19558, *[str_19559], **kwargs_19560)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 140, 8), LinAlgError_call_result_19561, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 139)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'info' (line 141)
    info_19562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 9), 'info')
    
    # Obtaining the type of the subscript
    int_19563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 26), 'int')
    # Getting the type of 'a1' (line 141)
    a1_19564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 17), 'a1')
    # Obtaining the member 'shape' of a type (line 141)
    shape_19565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 17), a1_19564, 'shape')
    # Obtaining the member '__getitem__' of a type (line 141)
    getitem___19566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 17), shape_19565, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 141)
    subscript_call_result_19567 = invoke(stypy.reporting.localization.Localization(__file__, 141, 17), getitem___19566, int_19563)
    
    int_19568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 31), 'int')
    # Applying the binary operator '+' (line 141)
    result_add_19569 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 17), '+', subscript_call_result_19567, int_19568)
    
    # Applying the binary operator '==' (line 141)
    result_eq_19570 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 9), '==', info_19562, result_add_19569)
    
    # Testing the type of an if condition (line 141)
    if_condition_19571 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 9), result_eq_19570)
    # Assigning a type to the variable 'if_condition_19571' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 9), 'if_condition_19571', if_condition_19571)
    # SSA begins for if statement (line 141)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 142)
    # Processing the call arguments (line 142)
    str_19573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 26), 'str', 'Leading eigenvalues do not satisfy sort condition.')
    # Processing the call keyword arguments (line 142)
    kwargs_19574 = {}
    # Getting the type of 'LinAlgError' (line 142)
    LinAlgError_19572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 142)
    LinAlgError_call_result_19575 = invoke(stypy.reporting.localization.Localization(__file__, 142, 14), LinAlgError_19572, *[str_19573], **kwargs_19574)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 142, 8), LinAlgError_call_result_19575, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 141)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'info' (line 143)
    info_19576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 9), 'info')
    int_19577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 16), 'int')
    # Applying the binary operator '>' (line 143)
    result_gt_19578 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 9), '>', info_19576, int_19577)
    
    # Testing the type of an if condition (line 143)
    if_condition_19579 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 9), result_gt_19578)
    # Assigning a type to the variable 'if_condition_19579' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 9), 'if_condition_19579', if_condition_19579)
    # SSA begins for if statement (line 143)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 144)
    # Processing the call arguments (line 144)
    str_19581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 26), 'str', 'Schur form not found.  Possibly ill-conditioned.')
    # Processing the call keyword arguments (line 144)
    kwargs_19582 = {}
    # Getting the type of 'LinAlgError' (line 144)
    LinAlgError_19580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 144)
    LinAlgError_call_result_19583 = invoke(stypy.reporting.localization.Localization(__file__, 144, 14), LinAlgError_19580, *[str_19581], **kwargs_19582)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 144, 8), LinAlgError_call_result_19583, 'raise parameter', BaseException)
    # SSA join for if statement (line 143)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 141)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 139)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 136)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'sort_t' (line 146)
    sort_t_19584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 7), 'sort_t')
    int_19585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 17), 'int')
    # Applying the binary operator '==' (line 146)
    result_eq_19586 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 7), '==', sort_t_19584, int_19585)
    
    # Testing the type of an if condition (line 146)
    if_condition_19587 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 4), result_eq_19586)
    # Assigning a type to the variable 'if_condition_19587' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'if_condition_19587', if_condition_19587)
    # SSA begins for if statement (line 146)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 147)
    tuple_19588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 147)
    # Adding element type (line 147)
    
    # Obtaining the type of the subscript
    int_19589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 22), 'int')
    # Getting the type of 'result' (line 147)
    result_19590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 15), 'result')
    # Obtaining the member '__getitem__' of a type (line 147)
    getitem___19591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 15), result_19590, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 147)
    subscript_call_result_19592 = invoke(stypy.reporting.localization.Localization(__file__, 147, 15), getitem___19591, int_19589)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 15), tuple_19588, subscript_call_result_19592)
    # Adding element type (line 147)
    
    # Obtaining the type of the subscript
    int_19593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 33), 'int')
    # Getting the type of 'result' (line 147)
    result_19594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 26), 'result')
    # Obtaining the member '__getitem__' of a type (line 147)
    getitem___19595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 26), result_19594, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 147)
    subscript_call_result_19596 = invoke(stypy.reporting.localization.Localization(__file__, 147, 26), getitem___19595, int_19593)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 15), tuple_19588, subscript_call_result_19596)
    
    # Assigning a type to the variable 'stypy_return_type' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'stypy_return_type', tuple_19588)
    # SSA branch for the else part of an if statement (line 146)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 149)
    tuple_19597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 149)
    # Adding element type (line 149)
    
    # Obtaining the type of the subscript
    int_19598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 22), 'int')
    # Getting the type of 'result' (line 149)
    result_19599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'result')
    # Obtaining the member '__getitem__' of a type (line 149)
    getitem___19600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 15), result_19599, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 149)
    subscript_call_result_19601 = invoke(stypy.reporting.localization.Localization(__file__, 149, 15), getitem___19600, int_19598)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 15), tuple_19597, subscript_call_result_19601)
    # Adding element type (line 149)
    
    # Obtaining the type of the subscript
    int_19602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 33), 'int')
    # Getting the type of 'result' (line 149)
    result_19603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 26), 'result')
    # Obtaining the member '__getitem__' of a type (line 149)
    getitem___19604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 26), result_19603, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 149)
    subscript_call_result_19605 = invoke(stypy.reporting.localization.Localization(__file__, 149, 26), getitem___19604, int_19602)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 15), tuple_19597, subscript_call_result_19605)
    # Adding element type (line 149)
    
    # Obtaining the type of the subscript
    int_19606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 45), 'int')
    # Getting the type of 'result' (line 149)
    result_19607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 38), 'result')
    # Obtaining the member '__getitem__' of a type (line 149)
    getitem___19608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 38), result_19607, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 149)
    subscript_call_result_19609 = invoke(stypy.reporting.localization.Localization(__file__, 149, 38), getitem___19608, int_19606)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 15), tuple_19597, subscript_call_result_19609)
    
    # Assigning a type to the variable 'stypy_return_type' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'stypy_return_type', tuple_19597)
    # SSA join for if statement (line 146)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'schur(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'schur' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_19610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_19610)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'schur'
    return stypy_return_type_19610

# Assigning a type to the variable 'schur' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'schur', schur)

# Assigning a Attribute to a Name (line 152):

# Assigning a Attribute to a Name (line 152):

# Call to finfo(...): (line 152)
# Processing the call arguments (line 152)
# Getting the type of 'float' (line 152)
float_19613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 18), 'float', False)
# Processing the call keyword arguments (line 152)
kwargs_19614 = {}
# Getting the type of 'numpy' (line 152)
numpy_19611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 6), 'numpy', False)
# Obtaining the member 'finfo' of a type (line 152)
finfo_19612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 6), numpy_19611, 'finfo')
# Calling finfo(args, kwargs) (line 152)
finfo_call_result_19615 = invoke(stypy.reporting.localization.Localization(__file__, 152, 6), finfo_19612, *[float_19613], **kwargs_19614)

# Obtaining the member 'eps' of a type (line 152)
eps_19616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 6), finfo_call_result_19615, 'eps')
# Assigning a type to the variable 'eps' (line 152)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'eps', eps_19616)

# Assigning a Attribute to a Name (line 153):

# Assigning a Attribute to a Name (line 153):

# Call to finfo(...): (line 153)
# Processing the call arguments (line 153)
# Getting the type of 'single' (line 153)
single_19619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 19), 'single', False)
# Processing the call keyword arguments (line 153)
kwargs_19620 = {}
# Getting the type of 'numpy' (line 153)
numpy_19617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 7), 'numpy', False)
# Obtaining the member 'finfo' of a type (line 153)
finfo_19618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 7), numpy_19617, 'finfo')
# Calling finfo(args, kwargs) (line 153)
finfo_call_result_19621 = invoke(stypy.reporting.localization.Localization(__file__, 153, 7), finfo_19618, *[single_19619], **kwargs_19620)

# Obtaining the member 'eps' of a type (line 153)
eps_19622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 7), finfo_call_result_19621, 'eps')
# Assigning a type to the variable 'feps' (line 153)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 0), 'feps', eps_19622)

# Assigning a Dict to a Name (line 155):

# Assigning a Dict to a Name (line 155):

# Obtaining an instance of the builtin type 'dict' (line 155)
dict_19623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 155)
# Adding element type (key, value) (line 155)
str_19624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 15), 'str', 'b')
int_19625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 19), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 14), dict_19623, (str_19624, int_19625))
# Adding element type (key, value) (line 155)
str_19626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 22), 'str', 'h')
int_19627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 26), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 14), dict_19623, (str_19626, int_19627))
# Adding element type (key, value) (line 155)
str_19628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 29), 'str', 'B')
int_19629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 34), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 14), dict_19623, (str_19628, int_19629))
# Adding element type (key, value) (line 155)
str_19630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 37), 'str', 'i')
int_19631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 41), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 14), dict_19623, (str_19630, int_19631))
# Adding element type (key, value) (line 155)
str_19632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 44), 'str', 'l')
int_19633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 49), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 14), dict_19623, (str_19632, int_19633))
# Adding element type (key, value) (line 155)
str_19634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 52), 'str', 'f')
int_19635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 57), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 14), dict_19623, (str_19634, int_19635))
# Adding element type (key, value) (line 155)
str_19636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 60), 'str', 'd')
int_19637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 65), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 14), dict_19623, (str_19636, int_19637))
# Adding element type (key, value) (line 155)
str_19638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 68), 'str', 'F')
int_19639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 73), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 14), dict_19623, (str_19638, int_19639))
# Adding element type (key, value) (line 155)
str_19640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 76), 'str', 'D')
int_19641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 81), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 14), dict_19623, (str_19640, int_19641))

# Assigning a type to the variable '_array_kind' (line 155)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 0), '_array_kind', dict_19623)

# Assigning a Dict to a Name (line 156):

# Assigning a Dict to a Name (line 156):

# Obtaining an instance of the builtin type 'dict' (line 156)
dict_19642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 156)
# Adding element type (key, value) (line 156)
str_19643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 20), 'str', 'i')
int_19644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 25), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 19), dict_19642, (str_19643, int_19644))
# Adding element type (key, value) (line 156)
str_19645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 28), 'str', 'l')
int_19646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 33), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 19), dict_19642, (str_19645, int_19646))
# Adding element type (key, value) (line 156)
str_19647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 36), 'str', 'f')
int_19648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 41), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 19), dict_19642, (str_19647, int_19648))
# Adding element type (key, value) (line 156)
str_19649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 44), 'str', 'd')
int_19650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 49), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 19), dict_19642, (str_19649, int_19650))
# Adding element type (key, value) (line 156)
str_19651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 52), 'str', 'F')
int_19652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 57), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 19), dict_19642, (str_19651, int_19652))
# Adding element type (key, value) (line 156)
str_19653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 60), 'str', 'D')
int_19654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 65), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 19), dict_19642, (str_19653, int_19654))

# Assigning a type to the variable '_array_precision' (line 156)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 0), '_array_precision', dict_19642)

# Assigning a List to a Name (line 157):

# Assigning a List to a Name (line 157):

# Obtaining an instance of the builtin type 'list' (line 157)
list_19655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 157)
# Adding element type (line 157)

# Obtaining an instance of the builtin type 'list' (line 157)
list_19656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 157)
# Adding element type (line 157)
str_19657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 16), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 15), list_19656, str_19657)
# Adding element type (line 157)
str_19658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 21), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 15), list_19656, str_19658)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 14), list_19655, list_19656)
# Adding element type (line 157)

# Obtaining an instance of the builtin type 'list' (line 157)
list_19659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 157)
# Adding element type (line 157)
str_19660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 28), 'str', 'F')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 27), list_19659, str_19660)
# Adding element type (line 157)
str_19661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 33), 'str', 'D')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 27), list_19659, str_19661)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 14), list_19655, list_19659)

# Assigning a type to the variable '_array_type' (line 157)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), '_array_type', list_19655)

@norecursion
def _commonType(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_commonType'
    module_type_store = module_type_store.open_function_context('_commonType', 160, 0, False)
    
    # Passed parameters checking function
    _commonType.stypy_localization = localization
    _commonType.stypy_type_of_self = None
    _commonType.stypy_type_store = module_type_store
    _commonType.stypy_function_name = '_commonType'
    _commonType.stypy_param_names_list = []
    _commonType.stypy_varargs_param_name = 'arrays'
    _commonType.stypy_kwargs_param_name = None
    _commonType.stypy_call_defaults = defaults
    _commonType.stypy_call_varargs = varargs
    _commonType.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_commonType', [], 'arrays', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_commonType', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_commonType(...)' code ##################

    
    # Assigning a Num to a Name (line 161):
    
    # Assigning a Num to a Name (line 161):
    int_19662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 11), 'int')
    # Assigning a type to the variable 'kind' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'kind', int_19662)
    
    # Assigning a Num to a Name (line 162):
    
    # Assigning a Num to a Name (line 162):
    int_19663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 16), 'int')
    # Assigning a type to the variable 'precision' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'precision', int_19663)
    
    # Getting the type of 'arrays' (line 163)
    arrays_19664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 13), 'arrays')
    # Testing the type of a for loop iterable (line 163)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 163, 4), arrays_19664)
    # Getting the type of the for loop variable (line 163)
    for_loop_var_19665 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 163, 4), arrays_19664)
    # Assigning a type to the variable 'a' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'a', for_loop_var_19665)
    # SSA begins for a for statement (line 163)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Attribute to a Name (line 164):
    
    # Assigning a Attribute to a Name (line 164):
    # Getting the type of 'a' (line 164)
    a_19666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'a')
    # Obtaining the member 'dtype' of a type (line 164)
    dtype_19667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 12), a_19666, 'dtype')
    # Obtaining the member 'char' of a type (line 164)
    char_19668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 12), dtype_19667, 'char')
    # Assigning a type to the variable 't' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 't', char_19668)
    
    # Assigning a Call to a Name (line 165):
    
    # Assigning a Call to a Name (line 165):
    
    # Call to max(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'kind' (line 165)
    kind_19670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 19), 'kind', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 't' (line 165)
    t_19671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 37), 't', False)
    # Getting the type of '_array_kind' (line 165)
    _array_kind_19672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 25), '_array_kind', False)
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___19673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 25), _array_kind_19672, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_19674 = invoke(stypy.reporting.localization.Localization(__file__, 165, 25), getitem___19673, t_19671)
    
    # Processing the call keyword arguments (line 165)
    kwargs_19675 = {}
    # Getting the type of 'max' (line 165)
    max_19669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 15), 'max', False)
    # Calling max(args, kwargs) (line 165)
    max_call_result_19676 = invoke(stypy.reporting.localization.Localization(__file__, 165, 15), max_19669, *[kind_19670, subscript_call_result_19674], **kwargs_19675)
    
    # Assigning a type to the variable 'kind' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'kind', max_call_result_19676)
    
    # Assigning a Call to a Name (line 166):
    
    # Assigning a Call to a Name (line 166):
    
    # Call to max(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'precision' (line 166)
    precision_19678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 24), 'precision', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 't' (line 166)
    t_19679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 52), 't', False)
    # Getting the type of '_array_precision' (line 166)
    _array_precision_19680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 35), '_array_precision', False)
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___19681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 35), _array_precision_19680, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_19682 = invoke(stypy.reporting.localization.Localization(__file__, 166, 35), getitem___19681, t_19679)
    
    # Processing the call keyword arguments (line 166)
    kwargs_19683 = {}
    # Getting the type of 'max' (line 166)
    max_19677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 20), 'max', False)
    # Calling max(args, kwargs) (line 166)
    max_call_result_19684 = invoke(stypy.reporting.localization.Localization(__file__, 166, 20), max_19677, *[precision_19678, subscript_call_result_19682], **kwargs_19683)
    
    # Assigning a type to the variable 'precision' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'precision', max_call_result_19684)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'precision' (line 167)
    precision_19685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 29), 'precision')
    
    # Obtaining the type of the subscript
    # Getting the type of 'kind' (line 167)
    kind_19686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 23), 'kind')
    # Getting the type of '_array_type' (line 167)
    _array_type_19687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), '_array_type')
    # Obtaining the member '__getitem__' of a type (line 167)
    getitem___19688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 11), _array_type_19687, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 167)
    subscript_call_result_19689 = invoke(stypy.reporting.localization.Localization(__file__, 167, 11), getitem___19688, kind_19686)
    
    # Obtaining the member '__getitem__' of a type (line 167)
    getitem___19690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 11), subscript_call_result_19689, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 167)
    subscript_call_result_19691 = invoke(stypy.reporting.localization.Localization(__file__, 167, 11), getitem___19690, precision_19685)
    
    # Assigning a type to the variable 'stypy_return_type' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type', subscript_call_result_19691)
    
    # ################# End of '_commonType(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_commonType' in the type store
    # Getting the type of 'stypy_return_type' (line 160)
    stypy_return_type_19692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_19692)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_commonType'
    return stypy_return_type_19692

# Assigning a type to the variable '_commonType' (line 160)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), '_commonType', _commonType)

@norecursion
def _castCopy(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_castCopy'
    module_type_store = module_type_store.open_function_context('_castCopy', 170, 0, False)
    
    # Passed parameters checking function
    _castCopy.stypy_localization = localization
    _castCopy.stypy_type_of_self = None
    _castCopy.stypy_type_store = module_type_store
    _castCopy.stypy_function_name = '_castCopy'
    _castCopy.stypy_param_names_list = ['type']
    _castCopy.stypy_varargs_param_name = 'arrays'
    _castCopy.stypy_kwargs_param_name = None
    _castCopy.stypy_call_defaults = defaults
    _castCopy.stypy_call_varargs = varargs
    _castCopy.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_castCopy', ['type'], 'arrays', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_castCopy', localization, ['type'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_castCopy(...)' code ##################

    
    # Assigning a Tuple to a Name (line 171):
    
    # Assigning a Tuple to a Name (line 171):
    
    # Obtaining an instance of the builtin type 'tuple' (line 171)
    tuple_19693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 171)
    
    # Assigning a type to the variable 'cast_arrays' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'cast_arrays', tuple_19693)
    
    # Getting the type of 'arrays' (line 172)
    arrays_19694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 13), 'arrays')
    # Testing the type of a for loop iterable (line 172)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 172, 4), arrays_19694)
    # Getting the type of the for loop variable (line 172)
    for_loop_var_19695 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 172, 4), arrays_19694)
    # Assigning a type to the variable 'a' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'a', for_loop_var_19695)
    # SSA begins for a for statement (line 172)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'a' (line 173)
    a_19696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 11), 'a')
    # Obtaining the member 'dtype' of a type (line 173)
    dtype_19697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 11), a_19696, 'dtype')
    # Obtaining the member 'char' of a type (line 173)
    char_19698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 11), dtype_19697, 'char')
    # Getting the type of 'type' (line 173)
    type_19699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 27), 'type')
    # Applying the binary operator '==' (line 173)
    result_eq_19700 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 11), '==', char_19698, type_19699)
    
    # Testing the type of an if condition (line 173)
    if_condition_19701 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 173, 8), result_eq_19700)
    # Assigning a type to the variable 'if_condition_19701' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'if_condition_19701', if_condition_19701)
    # SSA begins for if statement (line 173)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 174):
    
    # Assigning a BinOp to a Name (line 174):
    # Getting the type of 'cast_arrays' (line 174)
    cast_arrays_19702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 26), 'cast_arrays')
    
    # Obtaining an instance of the builtin type 'tuple' (line 174)
    tuple_19703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 174)
    # Adding element type (line 174)
    
    # Call to copy(...): (line 174)
    # Processing the call keyword arguments (line 174)
    kwargs_19706 = {}
    # Getting the type of 'a' (line 174)
    a_19704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 41), 'a', False)
    # Obtaining the member 'copy' of a type (line 174)
    copy_19705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 41), a_19704, 'copy')
    # Calling copy(args, kwargs) (line 174)
    copy_call_result_19707 = invoke(stypy.reporting.localization.Localization(__file__, 174, 41), copy_19705, *[], **kwargs_19706)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 41), tuple_19703, copy_call_result_19707)
    
    # Applying the binary operator '+' (line 174)
    result_add_19708 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 26), '+', cast_arrays_19702, tuple_19703)
    
    # Assigning a type to the variable 'cast_arrays' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'cast_arrays', result_add_19708)
    # SSA branch for the else part of an if statement (line 173)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 176):
    
    # Assigning a BinOp to a Name (line 176):
    # Getting the type of 'cast_arrays' (line 176)
    cast_arrays_19709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 26), 'cast_arrays')
    
    # Obtaining an instance of the builtin type 'tuple' (line 176)
    tuple_19710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 176)
    # Adding element type (line 176)
    
    # Call to astype(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'type' (line 176)
    type_19713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 50), 'type', False)
    # Processing the call keyword arguments (line 176)
    kwargs_19714 = {}
    # Getting the type of 'a' (line 176)
    a_19711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 41), 'a', False)
    # Obtaining the member 'astype' of a type (line 176)
    astype_19712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 41), a_19711, 'astype')
    # Calling astype(args, kwargs) (line 176)
    astype_call_result_19715 = invoke(stypy.reporting.localization.Localization(__file__, 176, 41), astype_19712, *[type_19713], **kwargs_19714)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 41), tuple_19710, astype_call_result_19715)
    
    # Applying the binary operator '+' (line 176)
    result_add_19716 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 26), '+', cast_arrays_19709, tuple_19710)
    
    # Assigning a type to the variable 'cast_arrays' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'cast_arrays', result_add_19716)
    # SSA join for if statement (line 173)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'cast_arrays' (line 177)
    cast_arrays_19718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 11), 'cast_arrays', False)
    # Processing the call keyword arguments (line 177)
    kwargs_19719 = {}
    # Getting the type of 'len' (line 177)
    len_19717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 7), 'len', False)
    # Calling len(args, kwargs) (line 177)
    len_call_result_19720 = invoke(stypy.reporting.localization.Localization(__file__, 177, 7), len_19717, *[cast_arrays_19718], **kwargs_19719)
    
    int_19721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 27), 'int')
    # Applying the binary operator '==' (line 177)
    result_eq_19722 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 7), '==', len_call_result_19720, int_19721)
    
    # Testing the type of an if condition (line 177)
    if_condition_19723 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 177, 4), result_eq_19722)
    # Assigning a type to the variable 'if_condition_19723' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'if_condition_19723', if_condition_19723)
    # SSA begins for if statement (line 177)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    int_19724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 27), 'int')
    # Getting the type of 'cast_arrays' (line 178)
    cast_arrays_19725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 15), 'cast_arrays')
    # Obtaining the member '__getitem__' of a type (line 178)
    getitem___19726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 15), cast_arrays_19725, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 178)
    subscript_call_result_19727 = invoke(stypy.reporting.localization.Localization(__file__, 178, 15), getitem___19726, int_19724)
    
    # Assigning a type to the variable 'stypy_return_type' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'stypy_return_type', subscript_call_result_19727)
    # SSA branch for the else part of an if statement (line 177)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'cast_arrays' (line 180)
    cast_arrays_19728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 15), 'cast_arrays')
    # Assigning a type to the variable 'stypy_return_type' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'stypy_return_type', cast_arrays_19728)
    # SSA join for if statement (line 177)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_castCopy(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_castCopy' in the type store
    # Getting the type of 'stypy_return_type' (line 170)
    stypy_return_type_19729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_19729)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_castCopy'
    return stypy_return_type_19729

# Assigning a type to the variable '_castCopy' (line 170)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), '_castCopy', _castCopy)

@norecursion
def rsf2csf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 183)
    True_19730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 31), 'True')
    defaults = [True_19730]
    # Create a new context for function 'rsf2csf'
    module_type_store = module_type_store.open_function_context('rsf2csf', 183, 0, False)
    
    # Passed parameters checking function
    rsf2csf.stypy_localization = localization
    rsf2csf.stypy_type_of_self = None
    rsf2csf.stypy_type_store = module_type_store
    rsf2csf.stypy_function_name = 'rsf2csf'
    rsf2csf.stypy_param_names_list = ['T', 'Z', 'check_finite']
    rsf2csf.stypy_varargs_param_name = None
    rsf2csf.stypy_kwargs_param_name = None
    rsf2csf.stypy_call_defaults = defaults
    rsf2csf.stypy_call_varargs = varargs
    rsf2csf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rsf2csf', ['T', 'Z', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rsf2csf', localization, ['T', 'Z', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rsf2csf(...)' code ##################

    str_19731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, (-1)), 'str', '\n    Convert real Schur form to complex Schur form.\n\n    Convert a quasi-diagonal real-valued Schur form to the upper triangular\n    complex-valued Schur form.\n\n    Parameters\n    ----------\n    T : (M, M) array_like\n        Real Schur form of the original matrix\n    Z : (M, M) array_like\n        Schur transformation matrix\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    T : (M, M) ndarray\n        Complex Schur form of the original matrix\n    Z : (M, M) ndarray\n        Schur transformation matrix corresponding to the complex form\n\n    See also\n    --------\n    schur : Schur decompose a matrix\n\n    ')
    
    # Getting the type of 'check_finite' (line 213)
    check_finite_19732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 7), 'check_finite')
    # Testing the type of an if condition (line 213)
    if_condition_19733 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 4), check_finite_19732)
    # Assigning a type to the variable 'if_condition_19733' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'if_condition_19733', if_condition_19733)
    # SSA begins for if statement (line 213)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 214):
    
    # Assigning a Subscript to a Name (line 214):
    
    # Obtaining the type of the subscript
    int_19734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 8), 'int')
    
    # Call to map(...): (line 214)
    # Processing the call arguments (line 214)
    # Getting the type of 'asarray_chkfinite' (line 214)
    asarray_chkfinite_19736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 19), 'asarray_chkfinite', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 214)
    tuple_19737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 214)
    # Adding element type (line 214)
    # Getting the type of 'Z' (line 214)
    Z_19738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 39), 'Z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 39), tuple_19737, Z_19738)
    # Adding element type (line 214)
    # Getting the type of 'T' (line 214)
    T_19739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 42), 'T', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 39), tuple_19737, T_19739)
    
    # Processing the call keyword arguments (line 214)
    kwargs_19740 = {}
    # Getting the type of 'map' (line 214)
    map_19735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 15), 'map', False)
    # Calling map(args, kwargs) (line 214)
    map_call_result_19741 = invoke(stypy.reporting.localization.Localization(__file__, 214, 15), map_19735, *[asarray_chkfinite_19736, tuple_19737], **kwargs_19740)
    
    # Obtaining the member '__getitem__' of a type (line 214)
    getitem___19742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), map_call_result_19741, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 214)
    subscript_call_result_19743 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), getitem___19742, int_19734)
    
    # Assigning a type to the variable 'tuple_var_assignment_19292' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'tuple_var_assignment_19292', subscript_call_result_19743)
    
    # Assigning a Subscript to a Name (line 214):
    
    # Obtaining the type of the subscript
    int_19744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 8), 'int')
    
    # Call to map(...): (line 214)
    # Processing the call arguments (line 214)
    # Getting the type of 'asarray_chkfinite' (line 214)
    asarray_chkfinite_19746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 19), 'asarray_chkfinite', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 214)
    tuple_19747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 214)
    # Adding element type (line 214)
    # Getting the type of 'Z' (line 214)
    Z_19748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 39), 'Z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 39), tuple_19747, Z_19748)
    # Adding element type (line 214)
    # Getting the type of 'T' (line 214)
    T_19749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 42), 'T', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 39), tuple_19747, T_19749)
    
    # Processing the call keyword arguments (line 214)
    kwargs_19750 = {}
    # Getting the type of 'map' (line 214)
    map_19745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 15), 'map', False)
    # Calling map(args, kwargs) (line 214)
    map_call_result_19751 = invoke(stypy.reporting.localization.Localization(__file__, 214, 15), map_19745, *[asarray_chkfinite_19746, tuple_19747], **kwargs_19750)
    
    # Obtaining the member '__getitem__' of a type (line 214)
    getitem___19752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), map_call_result_19751, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 214)
    subscript_call_result_19753 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), getitem___19752, int_19744)
    
    # Assigning a type to the variable 'tuple_var_assignment_19293' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'tuple_var_assignment_19293', subscript_call_result_19753)
    
    # Assigning a Name to a Name (line 214):
    # Getting the type of 'tuple_var_assignment_19292' (line 214)
    tuple_var_assignment_19292_19754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'tuple_var_assignment_19292')
    # Assigning a type to the variable 'Z' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'Z', tuple_var_assignment_19292_19754)
    
    # Assigning a Name to a Name (line 214):
    # Getting the type of 'tuple_var_assignment_19293' (line 214)
    tuple_var_assignment_19293_19755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'tuple_var_assignment_19293')
    # Assigning a type to the variable 'T' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 11), 'T', tuple_var_assignment_19293_19755)
    # SSA branch for the else part of an if statement (line 213)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 216):
    
    # Assigning a Subscript to a Name (line 216):
    
    # Obtaining the type of the subscript
    int_19756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 8), 'int')
    
    # Call to map(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'asarray' (line 216)
    asarray_19758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 18), 'asarray', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 216)
    tuple_19759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 216)
    # Adding element type (line 216)
    # Getting the type of 'Z' (line 216)
    Z_19760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 28), 'Z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 28), tuple_19759, Z_19760)
    # Adding element type (line 216)
    # Getting the type of 'T' (line 216)
    T_19761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 30), 'T', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 28), tuple_19759, T_19761)
    
    # Processing the call keyword arguments (line 216)
    kwargs_19762 = {}
    # Getting the type of 'map' (line 216)
    map_19757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 14), 'map', False)
    # Calling map(args, kwargs) (line 216)
    map_call_result_19763 = invoke(stypy.reporting.localization.Localization(__file__, 216, 14), map_19757, *[asarray_19758, tuple_19759], **kwargs_19762)
    
    # Obtaining the member '__getitem__' of a type (line 216)
    getitem___19764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), map_call_result_19763, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 216)
    subscript_call_result_19765 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), getitem___19764, int_19756)
    
    # Assigning a type to the variable 'tuple_var_assignment_19294' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'tuple_var_assignment_19294', subscript_call_result_19765)
    
    # Assigning a Subscript to a Name (line 216):
    
    # Obtaining the type of the subscript
    int_19766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 8), 'int')
    
    # Call to map(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'asarray' (line 216)
    asarray_19768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 18), 'asarray', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 216)
    tuple_19769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 216)
    # Adding element type (line 216)
    # Getting the type of 'Z' (line 216)
    Z_19770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 28), 'Z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 28), tuple_19769, Z_19770)
    # Adding element type (line 216)
    # Getting the type of 'T' (line 216)
    T_19771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 30), 'T', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 28), tuple_19769, T_19771)
    
    # Processing the call keyword arguments (line 216)
    kwargs_19772 = {}
    # Getting the type of 'map' (line 216)
    map_19767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 14), 'map', False)
    # Calling map(args, kwargs) (line 216)
    map_call_result_19773 = invoke(stypy.reporting.localization.Localization(__file__, 216, 14), map_19767, *[asarray_19768, tuple_19769], **kwargs_19772)
    
    # Obtaining the member '__getitem__' of a type (line 216)
    getitem___19774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), map_call_result_19773, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 216)
    subscript_call_result_19775 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), getitem___19774, int_19766)
    
    # Assigning a type to the variable 'tuple_var_assignment_19295' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'tuple_var_assignment_19295', subscript_call_result_19775)
    
    # Assigning a Name to a Name (line 216):
    # Getting the type of 'tuple_var_assignment_19294' (line 216)
    tuple_var_assignment_19294_19776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'tuple_var_assignment_19294')
    # Assigning a type to the variable 'Z' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'Z', tuple_var_assignment_19294_19776)
    
    # Assigning a Name to a Name (line 216):
    # Getting the type of 'tuple_var_assignment_19295' (line 216)
    tuple_var_assignment_19295_19777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'tuple_var_assignment_19295')
    # Assigning a type to the variable 'T' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 10), 'T', tuple_var_assignment_19295_19777)
    # SSA join for if statement (line 213)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'Z' (line 217)
    Z_19779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 11), 'Z', False)
    # Obtaining the member 'shape' of a type (line 217)
    shape_19780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 11), Z_19779, 'shape')
    # Processing the call keyword arguments (line 217)
    kwargs_19781 = {}
    # Getting the type of 'len' (line 217)
    len_19778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 7), 'len', False)
    # Calling len(args, kwargs) (line 217)
    len_call_result_19782 = invoke(stypy.reporting.localization.Localization(__file__, 217, 7), len_19778, *[shape_19780], **kwargs_19781)
    
    int_19783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 23), 'int')
    # Applying the binary operator '!=' (line 217)
    result_ne_19784 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 7), '!=', len_call_result_19782, int_19783)
    
    
    
    # Obtaining the type of the subscript
    int_19785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 36), 'int')
    # Getting the type of 'Z' (line 217)
    Z_19786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 28), 'Z')
    # Obtaining the member 'shape' of a type (line 217)
    shape_19787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 28), Z_19786, 'shape')
    # Obtaining the member '__getitem__' of a type (line 217)
    getitem___19788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 28), shape_19787, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 217)
    subscript_call_result_19789 = invoke(stypy.reporting.localization.Localization(__file__, 217, 28), getitem___19788, int_19785)
    
    
    # Obtaining the type of the subscript
    int_19790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 50), 'int')
    # Getting the type of 'Z' (line 217)
    Z_19791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 42), 'Z')
    # Obtaining the member 'shape' of a type (line 217)
    shape_19792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 42), Z_19791, 'shape')
    # Obtaining the member '__getitem__' of a type (line 217)
    getitem___19793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 42), shape_19792, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 217)
    subscript_call_result_19794 = invoke(stypy.reporting.localization.Localization(__file__, 217, 42), getitem___19793, int_19790)
    
    # Applying the binary operator '!=' (line 217)
    result_ne_19795 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 28), '!=', subscript_call_result_19789, subscript_call_result_19794)
    
    # Applying the binary operator 'or' (line 217)
    result_or_keyword_19796 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 7), 'or', result_ne_19784, result_ne_19795)
    
    # Testing the type of an if condition (line 217)
    if_condition_19797 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 4), result_or_keyword_19796)
    # Assigning a type to the variable 'if_condition_19797' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'if_condition_19797', if_condition_19797)
    # SSA begins for if statement (line 217)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 218)
    # Processing the call arguments (line 218)
    str_19799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 25), 'str', 'matrix must be square.')
    # Processing the call keyword arguments (line 218)
    kwargs_19800 = {}
    # Getting the type of 'ValueError' (line 218)
    ValueError_19798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 218)
    ValueError_call_result_19801 = invoke(stypy.reporting.localization.Localization(__file__, 218, 14), ValueError_19798, *[str_19799], **kwargs_19800)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 218, 8), ValueError_call_result_19801, 'raise parameter', BaseException)
    # SSA join for if statement (line 217)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'T' (line 219)
    T_19803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 11), 'T', False)
    # Obtaining the member 'shape' of a type (line 219)
    shape_19804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 11), T_19803, 'shape')
    # Processing the call keyword arguments (line 219)
    kwargs_19805 = {}
    # Getting the type of 'len' (line 219)
    len_19802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 7), 'len', False)
    # Calling len(args, kwargs) (line 219)
    len_call_result_19806 = invoke(stypy.reporting.localization.Localization(__file__, 219, 7), len_19802, *[shape_19804], **kwargs_19805)
    
    int_19807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 23), 'int')
    # Applying the binary operator '!=' (line 219)
    result_ne_19808 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 7), '!=', len_call_result_19806, int_19807)
    
    
    
    # Obtaining the type of the subscript
    int_19809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 36), 'int')
    # Getting the type of 'T' (line 219)
    T_19810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 28), 'T')
    # Obtaining the member 'shape' of a type (line 219)
    shape_19811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 28), T_19810, 'shape')
    # Obtaining the member '__getitem__' of a type (line 219)
    getitem___19812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 28), shape_19811, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 219)
    subscript_call_result_19813 = invoke(stypy.reporting.localization.Localization(__file__, 219, 28), getitem___19812, int_19809)
    
    
    # Obtaining the type of the subscript
    int_19814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 50), 'int')
    # Getting the type of 'T' (line 219)
    T_19815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 42), 'T')
    # Obtaining the member 'shape' of a type (line 219)
    shape_19816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 42), T_19815, 'shape')
    # Obtaining the member '__getitem__' of a type (line 219)
    getitem___19817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 42), shape_19816, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 219)
    subscript_call_result_19818 = invoke(stypy.reporting.localization.Localization(__file__, 219, 42), getitem___19817, int_19814)
    
    # Applying the binary operator '!=' (line 219)
    result_ne_19819 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 28), '!=', subscript_call_result_19813, subscript_call_result_19818)
    
    # Applying the binary operator 'or' (line 219)
    result_or_keyword_19820 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 7), 'or', result_ne_19808, result_ne_19819)
    
    # Testing the type of an if condition (line 219)
    if_condition_19821 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 219, 4), result_or_keyword_19820)
    # Assigning a type to the variable 'if_condition_19821' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'if_condition_19821', if_condition_19821)
    # SSA begins for if statement (line 219)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 220)
    # Processing the call arguments (line 220)
    str_19823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 25), 'str', 'matrix must be square.')
    # Processing the call keyword arguments (line 220)
    kwargs_19824 = {}
    # Getting the type of 'ValueError' (line 220)
    ValueError_19822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 220)
    ValueError_call_result_19825 = invoke(stypy.reporting.localization.Localization(__file__, 220, 14), ValueError_19822, *[str_19823], **kwargs_19824)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 220, 8), ValueError_call_result_19825, 'raise parameter', BaseException)
    # SSA join for if statement (line 219)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_19826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 15), 'int')
    # Getting the type of 'T' (line 221)
    T_19827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 7), 'T')
    # Obtaining the member 'shape' of a type (line 221)
    shape_19828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 7), T_19827, 'shape')
    # Obtaining the member '__getitem__' of a type (line 221)
    getitem___19829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 7), shape_19828, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 221)
    subscript_call_result_19830 = invoke(stypy.reporting.localization.Localization(__file__, 221, 7), getitem___19829, int_19826)
    
    
    # Obtaining the type of the subscript
    int_19831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 29), 'int')
    # Getting the type of 'Z' (line 221)
    Z_19832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 21), 'Z')
    # Obtaining the member 'shape' of a type (line 221)
    shape_19833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 21), Z_19832, 'shape')
    # Obtaining the member '__getitem__' of a type (line 221)
    getitem___19834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 21), shape_19833, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 221)
    subscript_call_result_19835 = invoke(stypy.reporting.localization.Localization(__file__, 221, 21), getitem___19834, int_19831)
    
    # Applying the binary operator '!=' (line 221)
    result_ne_19836 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 7), '!=', subscript_call_result_19830, subscript_call_result_19835)
    
    # Testing the type of an if condition (line 221)
    if_condition_19837 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 4), result_ne_19836)
    # Assigning a type to the variable 'if_condition_19837' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'if_condition_19837', if_condition_19837)
    # SSA begins for if statement (line 221)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 222)
    # Processing the call arguments (line 222)
    str_19839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 25), 'str', 'matrices must be same dimension.')
    # Processing the call keyword arguments (line 222)
    kwargs_19840 = {}
    # Getting the type of 'ValueError' (line 222)
    ValueError_19838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 222)
    ValueError_call_result_19841 = invoke(stypy.reporting.localization.Localization(__file__, 222, 14), ValueError_19838, *[str_19839], **kwargs_19840)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 222, 8), ValueError_call_result_19841, 'raise parameter', BaseException)
    # SSA join for if statement (line 221)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 223):
    
    # Assigning a Subscript to a Name (line 223):
    
    # Obtaining the type of the subscript
    int_19842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 16), 'int')
    # Getting the type of 'T' (line 223)
    T_19843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'T')
    # Obtaining the member 'shape' of a type (line 223)
    shape_19844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), T_19843, 'shape')
    # Obtaining the member '__getitem__' of a type (line 223)
    getitem___19845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), shape_19844, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 223)
    subscript_call_result_19846 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), getitem___19845, int_19842)
    
    # Assigning a type to the variable 'N' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'N', subscript_call_result_19846)
    
    # Assigning a Attribute to a Name (line 224):
    
    # Assigning a Attribute to a Name (line 224):
    # Getting the type of 'numpy' (line 224)
    numpy_19847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 10), 'numpy')
    # Obtaining the member 'array' of a type (line 224)
    array_19848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 10), numpy_19847, 'array')
    # Assigning a type to the variable 'arr' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'arr', array_19848)
    
    # Assigning a Call to a Name (line 225):
    
    # Assigning a Call to a Name (line 225):
    
    # Call to _commonType(...): (line 225)
    # Processing the call arguments (line 225)
    # Getting the type of 'Z' (line 225)
    Z_19850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 20), 'Z', False)
    # Getting the type of 'T' (line 225)
    T_19851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 23), 'T', False)
    
    # Call to arr(...): (line 225)
    # Processing the call arguments (line 225)
    
    # Obtaining an instance of the builtin type 'list' (line 225)
    list_19853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 225)
    # Adding element type (line 225)
    float_19854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 31), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 30), list_19853, float_19854)
    
    str_19855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 36), 'str', 'F')
    # Processing the call keyword arguments (line 225)
    kwargs_19856 = {}
    # Getting the type of 'arr' (line 225)
    arr_19852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 26), 'arr', False)
    # Calling arr(args, kwargs) (line 225)
    arr_call_result_19857 = invoke(stypy.reporting.localization.Localization(__file__, 225, 26), arr_19852, *[list_19853, str_19855], **kwargs_19856)
    
    # Processing the call keyword arguments (line 225)
    kwargs_19858 = {}
    # Getting the type of '_commonType' (line 225)
    _commonType_19849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), '_commonType', False)
    # Calling _commonType(args, kwargs) (line 225)
    _commonType_call_result_19859 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), _commonType_19849, *[Z_19850, T_19851, arr_call_result_19857], **kwargs_19858)
    
    # Assigning a type to the variable 't' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 't', _commonType_call_result_19859)
    
    # Assigning a Call to a Tuple (line 226):
    
    # Assigning a Subscript to a Name (line 226):
    
    # Obtaining the type of the subscript
    int_19860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 4), 'int')
    
    # Call to _castCopy(...): (line 226)
    # Processing the call arguments (line 226)
    # Getting the type of 't' (line 226)
    t_19862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 21), 't', False)
    # Getting the type of 'Z' (line 226)
    Z_19863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 24), 'Z', False)
    # Getting the type of 'T' (line 226)
    T_19864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 27), 'T', False)
    # Processing the call keyword arguments (line 226)
    kwargs_19865 = {}
    # Getting the type of '_castCopy' (line 226)
    _castCopy_19861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 11), '_castCopy', False)
    # Calling _castCopy(args, kwargs) (line 226)
    _castCopy_call_result_19866 = invoke(stypy.reporting.localization.Localization(__file__, 226, 11), _castCopy_19861, *[t_19862, Z_19863, T_19864], **kwargs_19865)
    
    # Obtaining the member '__getitem__' of a type (line 226)
    getitem___19867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 4), _castCopy_call_result_19866, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 226)
    subscript_call_result_19868 = invoke(stypy.reporting.localization.Localization(__file__, 226, 4), getitem___19867, int_19860)
    
    # Assigning a type to the variable 'tuple_var_assignment_19296' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'tuple_var_assignment_19296', subscript_call_result_19868)
    
    # Assigning a Subscript to a Name (line 226):
    
    # Obtaining the type of the subscript
    int_19869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 4), 'int')
    
    # Call to _castCopy(...): (line 226)
    # Processing the call arguments (line 226)
    # Getting the type of 't' (line 226)
    t_19871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 21), 't', False)
    # Getting the type of 'Z' (line 226)
    Z_19872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 24), 'Z', False)
    # Getting the type of 'T' (line 226)
    T_19873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 27), 'T', False)
    # Processing the call keyword arguments (line 226)
    kwargs_19874 = {}
    # Getting the type of '_castCopy' (line 226)
    _castCopy_19870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 11), '_castCopy', False)
    # Calling _castCopy(args, kwargs) (line 226)
    _castCopy_call_result_19875 = invoke(stypy.reporting.localization.Localization(__file__, 226, 11), _castCopy_19870, *[t_19871, Z_19872, T_19873], **kwargs_19874)
    
    # Obtaining the member '__getitem__' of a type (line 226)
    getitem___19876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 4), _castCopy_call_result_19875, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 226)
    subscript_call_result_19877 = invoke(stypy.reporting.localization.Localization(__file__, 226, 4), getitem___19876, int_19869)
    
    # Assigning a type to the variable 'tuple_var_assignment_19297' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'tuple_var_assignment_19297', subscript_call_result_19877)
    
    # Assigning a Name to a Name (line 226):
    # Getting the type of 'tuple_var_assignment_19296' (line 226)
    tuple_var_assignment_19296_19878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'tuple_var_assignment_19296')
    # Assigning a type to the variable 'Z' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'Z', tuple_var_assignment_19296_19878)
    
    # Assigning a Name to a Name (line 226):
    # Getting the type of 'tuple_var_assignment_19297' (line 226)
    tuple_var_assignment_19297_19879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'tuple_var_assignment_19297')
    # Assigning a type to the variable 'T' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 7), 'T', tuple_var_assignment_19297_19879)
    
    # Assigning a Attribute to a Name (line 227):
    
    # Assigning a Attribute to a Name (line 227):
    # Getting the type of 'numpy' (line 227)
    numpy_19880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 11), 'numpy')
    # Obtaining the member 'conj' of a type (line 227)
    conj_19881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 11), numpy_19880, 'conj')
    # Assigning a type to the variable 'conj' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'conj', conj_19881)
    
    # Assigning a Attribute to a Name (line 228):
    
    # Assigning a Attribute to a Name (line 228):
    # Getting the type of 'numpy' (line 228)
    numpy_19882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 10), 'numpy')
    # Obtaining the member 'dot' of a type (line 228)
    dot_19883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 10), numpy_19882, 'dot')
    # Assigning a type to the variable 'dot' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'dot', dot_19883)
    
    # Assigning a Attribute to a Name (line 229):
    
    # Assigning a Attribute to a Name (line 229):
    # Getting the type of 'numpy' (line 229)
    numpy_19884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 9), 'numpy')
    # Obtaining the member 'r_' of a type (line 229)
    r__19885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 9), numpy_19884, 'r_')
    # Assigning a type to the variable 'r_' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'r_', r__19885)
    
    # Assigning a Attribute to a Name (line 230):
    
    # Assigning a Attribute to a Name (line 230):
    # Getting the type of 'numpy' (line 230)
    numpy_19886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 13), 'numpy')
    # Obtaining the member 'transpose' of a type (line 230)
    transpose_19887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 13), numpy_19886, 'transpose')
    # Assigning a type to the variable 'transp' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'transp', transpose_19887)
    
    
    # Call to range(...): (line 231)
    # Processing the call arguments (line 231)
    # Getting the type of 'N' (line 231)
    N_19889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 19), 'N', False)
    int_19890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 21), 'int')
    # Applying the binary operator '-' (line 231)
    result_sub_19891 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 19), '-', N_19889, int_19890)
    
    int_19892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 24), 'int')
    int_19893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 27), 'int')
    # Processing the call keyword arguments (line 231)
    kwargs_19894 = {}
    # Getting the type of 'range' (line 231)
    range_19888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 13), 'range', False)
    # Calling range(args, kwargs) (line 231)
    range_call_result_19895 = invoke(stypy.reporting.localization.Localization(__file__, 231, 13), range_19888, *[result_sub_19891, int_19892, int_19893], **kwargs_19894)
    
    # Testing the type of a for loop iterable (line 231)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 231, 4), range_call_result_19895)
    # Getting the type of the for loop variable (line 231)
    for_loop_var_19896 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 231, 4), range_call_result_19895)
    # Assigning a type to the variable 'm' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'm', for_loop_var_19896)
    # SSA begins for a for statement (line 231)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Call to abs(...): (line 232)
    # Processing the call arguments (line 232)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 232)
    tuple_19898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 232)
    # Adding element type (line 232)
    # Getting the type of 'm' (line 232)
    m_19899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 17), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 17), tuple_19898, m_19899)
    # Adding element type (line 232)
    # Getting the type of 'm' (line 232)
    m_19900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 19), 'm', False)
    int_19901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 21), 'int')
    # Applying the binary operator '-' (line 232)
    result_sub_19902 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 19), '-', m_19900, int_19901)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 17), tuple_19898, result_sub_19902)
    
    # Getting the type of 'T' (line 232)
    T_19903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 15), 'T', False)
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___19904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 15), T_19903, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_19905 = invoke(stypy.reporting.localization.Localization(__file__, 232, 15), getitem___19904, tuple_19898)
    
    # Processing the call keyword arguments (line 232)
    kwargs_19906 = {}
    # Getting the type of 'abs' (line 232)
    abs_19897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 11), 'abs', False)
    # Calling abs(args, kwargs) (line 232)
    abs_call_result_19907 = invoke(stypy.reporting.localization.Localization(__file__, 232, 11), abs_19897, *[subscript_call_result_19905], **kwargs_19906)
    
    # Getting the type of 'eps' (line 232)
    eps_19908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 27), 'eps')
    
    # Call to abs(...): (line 232)
    # Processing the call arguments (line 232)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 232)
    tuple_19910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 232)
    # Adding element type (line 232)
    # Getting the type of 'm' (line 232)
    m_19911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 38), 'm', False)
    int_19912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 40), 'int')
    # Applying the binary operator '-' (line 232)
    result_sub_19913 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 38), '-', m_19911, int_19912)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 38), tuple_19910, result_sub_19913)
    # Adding element type (line 232)
    # Getting the type of 'm' (line 232)
    m_19914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 42), 'm', False)
    int_19915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 44), 'int')
    # Applying the binary operator '-' (line 232)
    result_sub_19916 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 42), '-', m_19914, int_19915)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 38), tuple_19910, result_sub_19916)
    
    # Getting the type of 'T' (line 232)
    T_19917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 36), 'T', False)
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___19918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 36), T_19917, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_19919 = invoke(stypy.reporting.localization.Localization(__file__, 232, 36), getitem___19918, tuple_19910)
    
    # Processing the call keyword arguments (line 232)
    kwargs_19920 = {}
    # Getting the type of 'abs' (line 232)
    abs_19909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 32), 'abs', False)
    # Calling abs(args, kwargs) (line 232)
    abs_call_result_19921 = invoke(stypy.reporting.localization.Localization(__file__, 232, 32), abs_19909, *[subscript_call_result_19919], **kwargs_19920)
    
    
    # Call to abs(...): (line 232)
    # Processing the call arguments (line 232)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 232)
    tuple_19923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 56), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 232)
    # Adding element type (line 232)
    # Getting the type of 'm' (line 232)
    m_19924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 56), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 56), tuple_19923, m_19924)
    # Adding element type (line 232)
    # Getting the type of 'm' (line 232)
    m_19925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 58), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 56), tuple_19923, m_19925)
    
    # Getting the type of 'T' (line 232)
    T_19926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 54), 'T', False)
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___19927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 54), T_19926, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_19928 = invoke(stypy.reporting.localization.Localization(__file__, 232, 54), getitem___19927, tuple_19923)
    
    # Processing the call keyword arguments (line 232)
    kwargs_19929 = {}
    # Getting the type of 'abs' (line 232)
    abs_19922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 50), 'abs', False)
    # Calling abs(args, kwargs) (line 232)
    abs_call_result_19930 = invoke(stypy.reporting.localization.Localization(__file__, 232, 50), abs_19922, *[subscript_call_result_19928], **kwargs_19929)
    
    # Applying the binary operator '+' (line 232)
    result_add_19931 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 32), '+', abs_call_result_19921, abs_call_result_19930)
    
    # Applying the binary operator '*' (line 232)
    result_mul_19932 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 27), '*', eps_19908, result_add_19931)
    
    # Applying the binary operator '>' (line 232)
    result_gt_19933 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 11), '>', abs_call_result_19907, result_mul_19932)
    
    # Testing the type of an if condition (line 232)
    if_condition_19934 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 232, 8), result_gt_19933)
    # Assigning a type to the variable 'if_condition_19934' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'if_condition_19934', if_condition_19934)
    # SSA begins for if statement (line 232)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 233):
    
    # Assigning a Call to a Name (line 233):
    
    # Call to slice(...): (line 233)
    # Processing the call arguments (line 233)
    # Getting the type of 'm' (line 233)
    m_19936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 22), 'm', False)
    int_19937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 24), 'int')
    # Applying the binary operator '-' (line 233)
    result_sub_19938 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 22), '-', m_19936, int_19937)
    
    # Getting the type of 'm' (line 233)
    m_19939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 27), 'm', False)
    int_19940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 29), 'int')
    # Applying the binary operator '+' (line 233)
    result_add_19941 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 27), '+', m_19939, int_19940)
    
    # Processing the call keyword arguments (line 233)
    kwargs_19942 = {}
    # Getting the type of 'slice' (line 233)
    slice_19935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 16), 'slice', False)
    # Calling slice(args, kwargs) (line 233)
    slice_call_result_19943 = invoke(stypy.reporting.localization.Localization(__file__, 233, 16), slice_19935, *[result_sub_19938, result_add_19941], **kwargs_19942)
    
    # Assigning a type to the variable 'k' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'k', slice_call_result_19943)
    
    # Assigning a BinOp to a Name (line 234):
    
    # Assigning a BinOp to a Name (line 234):
    
    # Call to eigvals(...): (line 234)
    # Processing the call arguments (line 234)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 234)
    tuple_19945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 234)
    # Adding element type (line 234)
    # Getting the type of 'k' (line 234)
    k_19946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 27), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 27), tuple_19945, k_19946)
    # Adding element type (line 234)
    # Getting the type of 'k' (line 234)
    k_19947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 29), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 27), tuple_19945, k_19947)
    
    # Getting the type of 'T' (line 234)
    T_19948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 25), 'T', False)
    # Obtaining the member '__getitem__' of a type (line 234)
    getitem___19949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 25), T_19948, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 234)
    subscript_call_result_19950 = invoke(stypy.reporting.localization.Localization(__file__, 234, 25), getitem___19949, tuple_19945)
    
    # Processing the call keyword arguments (line 234)
    kwargs_19951 = {}
    # Getting the type of 'eigvals' (line 234)
    eigvals_19944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 17), 'eigvals', False)
    # Calling eigvals(args, kwargs) (line 234)
    eigvals_call_result_19952 = invoke(stypy.reporting.localization.Localization(__file__, 234, 17), eigvals_19944, *[subscript_call_result_19950], **kwargs_19951)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 234)
    tuple_19953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 234)
    # Adding element type (line 234)
    # Getting the type of 'm' (line 234)
    m_19954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 37), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 37), tuple_19953, m_19954)
    # Adding element type (line 234)
    # Getting the type of 'm' (line 234)
    m_19955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 39), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 37), tuple_19953, m_19955)
    
    # Getting the type of 'T' (line 234)
    T_19956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 35), 'T')
    # Obtaining the member '__getitem__' of a type (line 234)
    getitem___19957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 35), T_19956, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 234)
    subscript_call_result_19958 = invoke(stypy.reporting.localization.Localization(__file__, 234, 35), getitem___19957, tuple_19953)
    
    # Applying the binary operator '-' (line 234)
    result_sub_19959 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 17), '-', eigvals_call_result_19952, subscript_call_result_19958)
    
    # Assigning a type to the variable 'mu' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'mu', result_sub_19959)
    
    # Assigning a Call to a Name (line 235):
    
    # Assigning a Call to a Name (line 235):
    
    # Call to norm(...): (line 235)
    # Processing the call arguments (line 235)
    
    # Obtaining an instance of the builtin type 'list' (line 235)
    list_19962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 235)
    # Adding element type (line 235)
    
    # Obtaining the type of the subscript
    int_19963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 30), 'int')
    # Getting the type of 'mu' (line 235)
    mu_19964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 27), 'mu', False)
    # Obtaining the member '__getitem__' of a type (line 235)
    getitem___19965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 27), mu_19964, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 235)
    subscript_call_result_19966 = invoke(stypy.reporting.localization.Localization(__file__, 235, 27), getitem___19965, int_19963)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 26), list_19962, subscript_call_result_19966)
    # Adding element type (line 235)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 235)
    tuple_19967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 235)
    # Adding element type (line 235)
    # Getting the type of 'm' (line 235)
    m_19968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 36), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 36), tuple_19967, m_19968)
    # Adding element type (line 235)
    # Getting the type of 'm' (line 235)
    m_19969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 38), 'm', False)
    int_19970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 40), 'int')
    # Applying the binary operator '-' (line 235)
    result_sub_19971 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 38), '-', m_19969, int_19970)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 36), tuple_19967, result_sub_19971)
    
    # Getting the type of 'T' (line 235)
    T_19972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 34), 'T', False)
    # Obtaining the member '__getitem__' of a type (line 235)
    getitem___19973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 34), T_19972, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 235)
    subscript_call_result_19974 = invoke(stypy.reporting.localization.Localization(__file__, 235, 34), getitem___19973, tuple_19967)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 26), list_19962, subscript_call_result_19974)
    
    # Processing the call keyword arguments (line 235)
    kwargs_19975 = {}
    # Getting the type of 'misc' (line 235)
    misc_19960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'misc', False)
    # Obtaining the member 'norm' of a type (line 235)
    norm_19961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 16), misc_19960, 'norm')
    # Calling norm(args, kwargs) (line 235)
    norm_call_result_19976 = invoke(stypy.reporting.localization.Localization(__file__, 235, 16), norm_19961, *[list_19962], **kwargs_19975)
    
    # Assigning a type to the variable 'r' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'r', norm_call_result_19976)
    
    # Assigning a BinOp to a Name (line 236):
    
    # Assigning a BinOp to a Name (line 236):
    
    # Obtaining the type of the subscript
    int_19977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 19), 'int')
    # Getting the type of 'mu' (line 236)
    mu_19978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 16), 'mu')
    # Obtaining the member '__getitem__' of a type (line 236)
    getitem___19979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 16), mu_19978, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 236)
    subscript_call_result_19980 = invoke(stypy.reporting.localization.Localization(__file__, 236, 16), getitem___19979, int_19977)
    
    # Getting the type of 'r' (line 236)
    r_19981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 24), 'r')
    # Applying the binary operator 'div' (line 236)
    result_div_19982 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 16), 'div', subscript_call_result_19980, r_19981)
    
    # Assigning a type to the variable 'c' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'c', result_div_19982)
    
    # Assigning a BinOp to a Name (line 237):
    
    # Assigning a BinOp to a Name (line 237):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 237)
    tuple_19983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 237)
    # Adding element type (line 237)
    # Getting the type of 'm' (line 237)
    m_19984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 18), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 18), tuple_19983, m_19984)
    # Adding element type (line 237)
    # Getting the type of 'm' (line 237)
    m_19985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 20), 'm')
    int_19986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 22), 'int')
    # Applying the binary operator '-' (line 237)
    result_sub_19987 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 20), '-', m_19985, int_19986)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 18), tuple_19983, result_sub_19987)
    
    # Getting the type of 'T' (line 237)
    T_19988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 16), 'T')
    # Obtaining the member '__getitem__' of a type (line 237)
    getitem___19989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 16), T_19988, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 237)
    subscript_call_result_19990 = invoke(stypy.reporting.localization.Localization(__file__, 237, 16), getitem___19989, tuple_19983)
    
    # Getting the type of 'r' (line 237)
    r_19991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 27), 'r')
    # Applying the binary operator 'div' (line 237)
    result_div_19992 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 16), 'div', subscript_call_result_19990, r_19991)
    
    # Assigning a type to the variable 's' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 's', result_div_19992)
    
    # Assigning a Subscript to a Name (line 238):
    
    # Assigning a Subscript to a Name (line 238):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 238)
    tuple_19993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 238)
    # Adding element type (line 238)
    
    # Call to arr(...): (line 238)
    # Processing the call arguments (line 238)
    
    # Obtaining an instance of the builtin type 'list' (line 238)
    list_19995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 238)
    # Adding element type (line 238)
    
    # Obtaining an instance of the builtin type 'list' (line 238)
    list_19996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 238)
    # Adding element type (line 238)
    
    # Call to conj(...): (line 238)
    # Processing the call arguments (line 238)
    # Getting the type of 'c' (line 238)
    c_19998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 30), 'c', False)
    # Processing the call keyword arguments (line 238)
    kwargs_19999 = {}
    # Getting the type of 'conj' (line 238)
    conj_19997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 25), 'conj', False)
    # Calling conj(args, kwargs) (line 238)
    conj_call_result_20000 = invoke(stypy.reporting.localization.Localization(__file__, 238, 25), conj_19997, *[c_19998], **kwargs_19999)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 24), list_19996, conj_call_result_20000)
    # Adding element type (line 238)
    # Getting the type of 's' (line 238)
    s_20001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 34), 's', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 24), list_19996, s_20001)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 23), list_19995, list_19996)
    
    # Processing the call keyword arguments (line 238)
    # Getting the type of 't' (line 238)
    t_20002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 45), 't', False)
    keyword_20003 = t_20002
    kwargs_20004 = {'dtype': keyword_20003}
    # Getting the type of 'arr' (line 238)
    arr_19994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 19), 'arr', False)
    # Calling arr(args, kwargs) (line 238)
    arr_call_result_20005 = invoke(stypy.reporting.localization.Localization(__file__, 238, 19), arr_19994, *[list_19995], **kwargs_20004)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 19), tuple_19993, arr_call_result_20005)
    # Adding element type (line 238)
    
    # Call to arr(...): (line 238)
    # Processing the call arguments (line 238)
    
    # Obtaining an instance of the builtin type 'list' (line 238)
    list_20007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 53), 'list')
    # Adding type elements to the builtin type 'list' instance (line 238)
    # Adding element type (line 238)
    
    # Obtaining an instance of the builtin type 'list' (line 238)
    list_20008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 54), 'list')
    # Adding type elements to the builtin type 'list' instance (line 238)
    # Adding element type (line 238)
    
    # Getting the type of 's' (line 238)
    s_20009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 56), 's', False)
    # Applying the 'usub' unary operator (line 238)
    result___neg___20010 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 55), 'usub', s_20009)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 54), list_20008, result___neg___20010)
    # Adding element type (line 238)
    # Getting the type of 'c' (line 238)
    c_20011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 59), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 54), list_20008, c_20011)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 53), list_20007, list_20008)
    
    # Processing the call keyword arguments (line 238)
    # Getting the type of 't' (line 238)
    t_20012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 70), 't', False)
    keyword_20013 = t_20012
    kwargs_20014 = {'dtype': keyword_20013}
    # Getting the type of 'arr' (line 238)
    arr_20006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 49), 'arr', False)
    # Calling arr(args, kwargs) (line 238)
    arr_call_result_20015 = invoke(stypy.reporting.localization.Localization(__file__, 238, 49), arr_20006, *[list_20007], **kwargs_20014)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 19), tuple_19993, arr_call_result_20015)
    
    # Getting the type of 'r_' (line 238)
    r__20016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'r_')
    # Obtaining the member '__getitem__' of a type (line 238)
    getitem___20017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 16), r__20016, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 238)
    subscript_call_result_20018 = invoke(stypy.reporting.localization.Localization(__file__, 238, 16), getitem___20017, tuple_19993)
    
    # Assigning a type to the variable 'G' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'G', subscript_call_result_20018)
    
    # Assigning a Call to a Name (line 239):
    
    # Assigning a Call to a Name (line 239):
    
    # Call to conj(...): (line 239)
    # Processing the call arguments (line 239)
    
    # Call to transp(...): (line 239)
    # Processing the call arguments (line 239)
    # Getting the type of 'G' (line 239)
    G_20021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 29), 'G', False)
    # Processing the call keyword arguments (line 239)
    kwargs_20022 = {}
    # Getting the type of 'transp' (line 239)
    transp_20020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 22), 'transp', False)
    # Calling transp(args, kwargs) (line 239)
    transp_call_result_20023 = invoke(stypy.reporting.localization.Localization(__file__, 239, 22), transp_20020, *[G_20021], **kwargs_20022)
    
    # Processing the call keyword arguments (line 239)
    kwargs_20024 = {}
    # Getting the type of 'conj' (line 239)
    conj_20019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 17), 'conj', False)
    # Calling conj(args, kwargs) (line 239)
    conj_call_result_20025 = invoke(stypy.reporting.localization.Localization(__file__, 239, 17), conj_20019, *[transp_call_result_20023], **kwargs_20024)
    
    # Assigning a type to the variable 'Gc' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'Gc', conj_call_result_20025)
    
    # Assigning a Call to a Name (line 240):
    
    # Assigning a Call to a Name (line 240):
    
    # Call to slice(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'm' (line 240)
    m_20027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 22), 'm', False)
    int_20028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 24), 'int')
    # Applying the binary operator '-' (line 240)
    result_sub_20029 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 22), '-', m_20027, int_20028)
    
    # Getting the type of 'N' (line 240)
    N_20030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 27), 'N', False)
    # Processing the call keyword arguments (line 240)
    kwargs_20031 = {}
    # Getting the type of 'slice' (line 240)
    slice_20026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 16), 'slice', False)
    # Calling slice(args, kwargs) (line 240)
    slice_call_result_20032 = invoke(stypy.reporting.localization.Localization(__file__, 240, 16), slice_20026, *[result_sub_20029, N_20030], **kwargs_20031)
    
    # Assigning a type to the variable 'j' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'j', slice_call_result_20032)
    
    # Assigning a Call to a Subscript (line 241):
    
    # Assigning a Call to a Subscript (line 241):
    
    # Call to dot(...): (line 241)
    # Processing the call arguments (line 241)
    # Getting the type of 'G' (line 241)
    G_20034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 25), 'G', False)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 241)
    tuple_20035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 241)
    # Adding element type (line 241)
    # Getting the type of 'k' (line 241)
    k_20036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 30), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 30), tuple_20035, k_20036)
    # Adding element type (line 241)
    # Getting the type of 'j' (line 241)
    j_20037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 32), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 30), tuple_20035, j_20037)
    
    # Getting the type of 'T' (line 241)
    T_20038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 28), 'T', False)
    # Obtaining the member '__getitem__' of a type (line 241)
    getitem___20039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 28), T_20038, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 241)
    subscript_call_result_20040 = invoke(stypy.reporting.localization.Localization(__file__, 241, 28), getitem___20039, tuple_20035)
    
    # Processing the call keyword arguments (line 241)
    kwargs_20041 = {}
    # Getting the type of 'dot' (line 241)
    dot_20033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 21), 'dot', False)
    # Calling dot(args, kwargs) (line 241)
    dot_call_result_20042 = invoke(stypy.reporting.localization.Localization(__file__, 241, 21), dot_20033, *[G_20034, subscript_call_result_20040], **kwargs_20041)
    
    # Getting the type of 'T' (line 241)
    T_20043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'T')
    
    # Obtaining an instance of the builtin type 'tuple' (line 241)
    tuple_20044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 241)
    # Adding element type (line 241)
    # Getting the type of 'k' (line 241)
    k_20045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 14), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 14), tuple_20044, k_20045)
    # Adding element type (line 241)
    # Getting the type of 'j' (line 241)
    j_20046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 16), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 14), tuple_20044, j_20046)
    
    # Storing an element on a container (line 241)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 12), T_20043, (tuple_20044, dot_call_result_20042))
    
    # Assigning a Call to a Name (line 242):
    
    # Assigning a Call to a Name (line 242):
    
    # Call to slice(...): (line 242)
    # Processing the call arguments (line 242)
    int_20048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 22), 'int')
    # Getting the type of 'm' (line 242)
    m_20049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 25), 'm', False)
    int_20050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 27), 'int')
    # Applying the binary operator '+' (line 242)
    result_add_20051 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 25), '+', m_20049, int_20050)
    
    # Processing the call keyword arguments (line 242)
    kwargs_20052 = {}
    # Getting the type of 'slice' (line 242)
    slice_20047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'slice', False)
    # Calling slice(args, kwargs) (line 242)
    slice_call_result_20053 = invoke(stypy.reporting.localization.Localization(__file__, 242, 16), slice_20047, *[int_20048, result_add_20051], **kwargs_20052)
    
    # Assigning a type to the variable 'i' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'i', slice_call_result_20053)
    
    # Assigning a Call to a Subscript (line 243):
    
    # Assigning a Call to a Subscript (line 243):
    
    # Call to dot(...): (line 243)
    # Processing the call arguments (line 243)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 243)
    tuple_20055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 243)
    # Adding element type (line 243)
    # Getting the type of 'i' (line 243)
    i_20056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 27), 'i', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 27), tuple_20055, i_20056)
    # Adding element type (line 243)
    # Getting the type of 'k' (line 243)
    k_20057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 29), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 27), tuple_20055, k_20057)
    
    # Getting the type of 'T' (line 243)
    T_20058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 25), 'T', False)
    # Obtaining the member '__getitem__' of a type (line 243)
    getitem___20059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 25), T_20058, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 243)
    subscript_call_result_20060 = invoke(stypy.reporting.localization.Localization(__file__, 243, 25), getitem___20059, tuple_20055)
    
    # Getting the type of 'Gc' (line 243)
    Gc_20061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 33), 'Gc', False)
    # Processing the call keyword arguments (line 243)
    kwargs_20062 = {}
    # Getting the type of 'dot' (line 243)
    dot_20054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 21), 'dot', False)
    # Calling dot(args, kwargs) (line 243)
    dot_call_result_20063 = invoke(stypy.reporting.localization.Localization(__file__, 243, 21), dot_20054, *[subscript_call_result_20060, Gc_20061], **kwargs_20062)
    
    # Getting the type of 'T' (line 243)
    T_20064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'T')
    
    # Obtaining an instance of the builtin type 'tuple' (line 243)
    tuple_20065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 243)
    # Adding element type (line 243)
    # Getting the type of 'i' (line 243)
    i_20066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 14), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 14), tuple_20065, i_20066)
    # Adding element type (line 243)
    # Getting the type of 'k' (line 243)
    k_20067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 16), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 14), tuple_20065, k_20067)
    
    # Storing an element on a container (line 243)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 12), T_20064, (tuple_20065, dot_call_result_20063))
    
    # Assigning a Call to a Name (line 244):
    
    # Assigning a Call to a Name (line 244):
    
    # Call to slice(...): (line 244)
    # Processing the call arguments (line 244)
    int_20069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 22), 'int')
    # Getting the type of 'N' (line 244)
    N_20070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 25), 'N', False)
    # Processing the call keyword arguments (line 244)
    kwargs_20071 = {}
    # Getting the type of 'slice' (line 244)
    slice_20068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 16), 'slice', False)
    # Calling slice(args, kwargs) (line 244)
    slice_call_result_20072 = invoke(stypy.reporting.localization.Localization(__file__, 244, 16), slice_20068, *[int_20069, N_20070], **kwargs_20071)
    
    # Assigning a type to the variable 'i' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'i', slice_call_result_20072)
    
    # Assigning a Call to a Subscript (line 245):
    
    # Assigning a Call to a Subscript (line 245):
    
    # Call to dot(...): (line 245)
    # Processing the call arguments (line 245)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 245)
    tuple_20074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 245)
    # Adding element type (line 245)
    # Getting the type of 'i' (line 245)
    i_20075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 27), 'i', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 27), tuple_20074, i_20075)
    # Adding element type (line 245)
    # Getting the type of 'k' (line 245)
    k_20076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 29), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 27), tuple_20074, k_20076)
    
    # Getting the type of 'Z' (line 245)
    Z_20077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 25), 'Z', False)
    # Obtaining the member '__getitem__' of a type (line 245)
    getitem___20078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 25), Z_20077, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 245)
    subscript_call_result_20079 = invoke(stypy.reporting.localization.Localization(__file__, 245, 25), getitem___20078, tuple_20074)
    
    # Getting the type of 'Gc' (line 245)
    Gc_20080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 33), 'Gc', False)
    # Processing the call keyword arguments (line 245)
    kwargs_20081 = {}
    # Getting the type of 'dot' (line 245)
    dot_20073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 21), 'dot', False)
    # Calling dot(args, kwargs) (line 245)
    dot_call_result_20082 = invoke(stypy.reporting.localization.Localization(__file__, 245, 21), dot_20073, *[subscript_call_result_20079, Gc_20080], **kwargs_20081)
    
    # Getting the type of 'Z' (line 245)
    Z_20083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'Z')
    
    # Obtaining an instance of the builtin type 'tuple' (line 245)
    tuple_20084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 245)
    # Adding element type (line 245)
    # Getting the type of 'i' (line 245)
    i_20085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 14), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 14), tuple_20084, i_20085)
    # Adding element type (line 245)
    # Getting the type of 'k' (line 245)
    k_20086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 16), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 14), tuple_20084, k_20086)
    
    # Storing an element on a container (line 245)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 12), Z_20083, (tuple_20084, dot_call_result_20082))
    # SSA join for if statement (line 232)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Subscript (line 246):
    
    # Assigning a Num to a Subscript (line 246):
    float_20087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 19), 'float')
    # Getting the type of 'T' (line 246)
    T_20088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'T')
    
    # Obtaining an instance of the builtin type 'tuple' (line 246)
    tuple_20089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 246)
    # Adding element type (line 246)
    # Getting the type of 'm' (line 246)
    m_20090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 10), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 10), tuple_20089, m_20090)
    # Adding element type (line 246)
    # Getting the type of 'm' (line 246)
    m_20091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'm')
    int_20092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 14), 'int')
    # Applying the binary operator '-' (line 246)
    result_sub_20093 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 12), '-', m_20091, int_20092)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 10), tuple_20089, result_sub_20093)
    
    # Storing an element on a container (line 246)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 8), T_20088, (tuple_20089, float_20087))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 247)
    tuple_20094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 247)
    # Adding element type (line 247)
    # Getting the type of 'T' (line 247)
    T_20095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 11), 'T')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 11), tuple_20094, T_20095)
    # Adding element type (line 247)
    # Getting the type of 'Z' (line 247)
    Z_20096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 14), 'Z')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 11), tuple_20094, Z_20096)
    
    # Assigning a type to the variable 'stypy_return_type' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'stypy_return_type', tuple_20094)
    
    # ################# End of 'rsf2csf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rsf2csf' in the type store
    # Getting the type of 'stypy_return_type' (line 183)
    stypy_return_type_20097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20097)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rsf2csf'
    return stypy_return_type_20097

# Assigning a type to the variable 'rsf2csf' (line 183)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'rsf2csf', rsf2csf)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
