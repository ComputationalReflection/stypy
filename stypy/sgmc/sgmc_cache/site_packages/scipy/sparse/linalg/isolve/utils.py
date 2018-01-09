
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: __docformat__ = "restructuredtext en"
4: 
5: __all__ = []
6: 
7: from warnings import warn
8: 
9: from numpy import asanyarray, asarray, asmatrix, array, matrix, zeros
10: 
11: from scipy.sparse.linalg.interface import aslinearoperator, LinearOperator, \
12:      IdentityOperator
13: 
14: _coerce_rules = {('f','f'):'f', ('f','d'):'d', ('f','F'):'F',
15:                  ('f','D'):'D', ('d','f'):'d', ('d','d'):'d',
16:                  ('d','F'):'D', ('d','D'):'D', ('F','f'):'F',
17:                  ('F','d'):'D', ('F','F'):'F', ('F','D'):'D',
18:                  ('D','f'):'D', ('D','d'):'D', ('D','F'):'D',
19:                  ('D','D'):'D'}
20: 
21: 
22: def coerce(x,y):
23:     if x not in 'fdFD':
24:         x = 'd'
25:     if y not in 'fdFD':
26:         y = 'd'
27:     return _coerce_rules[x,y]
28: 
29: 
30: def id(x):
31:     return x
32: 
33: 
34: def make_system(A, M, x0, b):
35:     '''Make a linear system Ax=b
36: 
37:     Parameters
38:     ----------
39:     A : LinearOperator
40:         sparse or dense matrix (or any valid input to aslinearoperator)
41:     M : {LinearOperator, Nones}
42:         preconditioner
43:         sparse or dense matrix (or any valid input to aslinearoperator)
44:     x0 : {array_like, None}
45:         initial guess to iterative method
46:     b : array_like
47:         right hand side
48: 
49:     Returns
50:     -------
51:     (A, M, x, b, postprocess)
52:         A : LinearOperator
53:             matrix of the linear system
54:         M : LinearOperator
55:             preconditioner
56:         x : rank 1 ndarray
57:             initial guess
58:         b : rank 1 ndarray
59:             right hand side
60:         postprocess : function
61:             converts the solution vector to the appropriate
62:             type and dimensions (e.g. (N,1) matrix)
63: 
64:     '''
65:     A_ = A
66:     A = aslinearoperator(A)
67: 
68:     if A.shape[0] != A.shape[1]:
69:         raise ValueError('expected square matrix, but got shape=%s' % (A.shape,))
70: 
71:     N = A.shape[0]
72: 
73:     b = asanyarray(b)
74: 
75:     if not (b.shape == (N,1) or b.shape == (N,)):
76:         raise ValueError('A and b have incompatible dimensions')
77: 
78:     if b.dtype.char not in 'fdFD':
79:         b = b.astype('d')  # upcast non-FP types to double
80: 
81:     def postprocess(x):
82:         if isinstance(b,matrix):
83:             x = asmatrix(x)
84:         return x.reshape(b.shape)
85: 
86:     if hasattr(A,'dtype'):
87:         xtype = A.dtype.char
88:     else:
89:         xtype = A.matvec(b).dtype.char
90:     xtype = coerce(xtype, b.dtype.char)
91: 
92:     b = asarray(b,dtype=xtype)  # make b the same type as x
93:     b = b.ravel()
94: 
95:     if x0 is None:
96:         x = zeros(N, dtype=xtype)
97:     else:
98:         x = array(x0, dtype=xtype)
99:         if not (x.shape == (N,1) or x.shape == (N,)):
100:             raise ValueError('A and x have incompatible dimensions')
101:         x = x.ravel()
102: 
103:     # process preconditioner
104:     if M is None:
105:         if hasattr(A_,'psolve'):
106:             psolve = A_.psolve
107:         else:
108:             psolve = id
109:         if hasattr(A_,'rpsolve'):
110:             rpsolve = A_.rpsolve
111:         else:
112:             rpsolve = id
113:         if psolve is id and rpsolve is id:
114:             M = IdentityOperator(shape=A.shape, dtype=A.dtype)
115:         else:
116:             M = LinearOperator(A.shape, matvec=psolve, rmatvec=rpsolve,
117:                                dtype=A.dtype)
118:     else:
119:         M = aslinearoperator(M)
120:         if A.shape != M.shape:
121:             raise ValueError('matrix and preconditioner have different shapes')
122: 
123:     return A, M, x, b, postprocess
124: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_414303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 16), 'str', 'restructuredtext en')
# Assigning a type to the variable '__docformat__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__docformat__', str_414303)

# Assigning a List to a Name (line 5):
__all__ = []
module_type_store.set_exportable_members([])

# Obtaining an instance of the builtin type 'list' (line 5)
list_414304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)

# Assigning a type to the variable '__all__' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), '__all__', list_414304)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from warnings import warn' statement (line 7)
try:
    from warnings import warn

except:
    warn = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'warnings', None, module_type_store, ['warn'], [warn])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy import asanyarray, asarray, asmatrix, array, matrix, zeros' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_414305 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_414305) is not StypyTypeError):

    if (import_414305 != 'pyd_module'):
        __import__(import_414305)
        sys_modules_414306 = sys.modules[import_414305]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', sys_modules_414306.module_type_store, module_type_store, ['asanyarray', 'asarray', 'asmatrix', 'array', 'matrix', 'zeros'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_414306, sys_modules_414306.module_type_store, module_type_store)
    else:
        from numpy import asanyarray, asarray, asmatrix, array, matrix, zeros

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', None, module_type_store, ['asanyarray', 'asarray', 'asmatrix', 'array', 'matrix', 'zeros'], [asanyarray, asarray, asmatrix, array, matrix, zeros])

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_414305)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.sparse.linalg.interface import aslinearoperator, LinearOperator, IdentityOperator' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_414307 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg.interface')

if (type(import_414307) is not StypyTypeError):

    if (import_414307 != 'pyd_module'):
        __import__(import_414307)
        sys_modules_414308 = sys.modules[import_414307]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg.interface', sys_modules_414308.module_type_store, module_type_store, ['aslinearoperator', 'LinearOperator', 'IdentityOperator'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_414308, sys_modules_414308.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.interface import aslinearoperator, LinearOperator, IdentityOperator

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg.interface', None, module_type_store, ['aslinearoperator', 'LinearOperator', 'IdentityOperator'], [aslinearoperator, LinearOperator, IdentityOperator])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.interface' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg.interface', import_414307)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')


# Assigning a Dict to a Name (line 14):

# Obtaining an instance of the builtin type 'dict' (line 14)
dict_414309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 14)
# Adding element type (key, value) (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 14)
tuple_414310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 14)
# Adding element type (line 14)
str_414311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 18), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 18), tuple_414310, str_414311)
# Adding element type (line 14)
str_414312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 22), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 18), tuple_414310, str_414312)

str_414313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 27), 'str', 'f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), dict_414309, (tuple_414310, str_414313))
# Adding element type (key, value) (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 14)
tuple_414314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 33), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 14)
# Adding element type (line 14)
str_414315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 33), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 33), tuple_414314, str_414315)
# Adding element type (line 14)
str_414316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 37), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 33), tuple_414314, str_414316)

str_414317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 42), 'str', 'd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), dict_414309, (tuple_414314, str_414317))
# Adding element type (key, value) (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 14)
tuple_414318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 14)
# Adding element type (line 14)
str_414319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 48), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 48), tuple_414318, str_414319)
# Adding element type (line 14)
str_414320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 52), 'str', 'F')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 48), tuple_414318, str_414320)

str_414321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 57), 'str', 'F')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), dict_414309, (tuple_414318, str_414321))
# Adding element type (key, value) (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 15)
tuple_414322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 15)
# Adding element type (line 15)
str_414323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 18), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 18), tuple_414322, str_414323)
# Adding element type (line 15)
str_414324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 22), 'str', 'D')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 18), tuple_414322, str_414324)

str_414325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 27), 'str', 'D')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), dict_414309, (tuple_414322, str_414325))
# Adding element type (key, value) (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 15)
tuple_414326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 33), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 15)
# Adding element type (line 15)
str_414327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 33), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 33), tuple_414326, str_414327)
# Adding element type (line 15)
str_414328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 37), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 33), tuple_414326, str_414328)

str_414329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 42), 'str', 'd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), dict_414309, (tuple_414326, str_414329))
# Adding element type (key, value) (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 15)
tuple_414330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 15)
# Adding element type (line 15)
str_414331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 48), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 48), tuple_414330, str_414331)
# Adding element type (line 15)
str_414332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 52), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 48), tuple_414330, str_414332)

str_414333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 57), 'str', 'd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), dict_414309, (tuple_414330, str_414333))
# Adding element type (key, value) (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 16)
tuple_414334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 16)
# Adding element type (line 16)
str_414335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 18), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 18), tuple_414334, str_414335)
# Adding element type (line 16)
str_414336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 22), 'str', 'F')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 18), tuple_414334, str_414336)

str_414337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 27), 'str', 'D')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), dict_414309, (tuple_414334, str_414337))
# Adding element type (key, value) (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 16)
tuple_414338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 33), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 16)
# Adding element type (line 16)
str_414339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 33), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 33), tuple_414338, str_414339)
# Adding element type (line 16)
str_414340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 37), 'str', 'D')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 33), tuple_414338, str_414340)

str_414341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 42), 'str', 'D')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), dict_414309, (tuple_414338, str_414341))
# Adding element type (key, value) (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 16)
tuple_414342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 16)
# Adding element type (line 16)
str_414343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 48), 'str', 'F')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 48), tuple_414342, str_414343)
# Adding element type (line 16)
str_414344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 52), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 48), tuple_414342, str_414344)

str_414345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 57), 'str', 'F')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), dict_414309, (tuple_414342, str_414345))
# Adding element type (key, value) (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 17)
tuple_414346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 17)
# Adding element type (line 17)
str_414347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 18), 'str', 'F')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 18), tuple_414346, str_414347)
# Adding element type (line 17)
str_414348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 22), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 18), tuple_414346, str_414348)

str_414349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 27), 'str', 'D')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), dict_414309, (tuple_414346, str_414349))
# Adding element type (key, value) (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 17)
tuple_414350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 33), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 17)
# Adding element type (line 17)
str_414351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 33), 'str', 'F')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 33), tuple_414350, str_414351)
# Adding element type (line 17)
str_414352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 37), 'str', 'F')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 33), tuple_414350, str_414352)

str_414353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 42), 'str', 'F')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), dict_414309, (tuple_414350, str_414353))
# Adding element type (key, value) (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 17)
tuple_414354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 17)
# Adding element type (line 17)
str_414355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 48), 'str', 'F')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 48), tuple_414354, str_414355)
# Adding element type (line 17)
str_414356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 52), 'str', 'D')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 48), tuple_414354, str_414356)

str_414357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 57), 'str', 'D')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), dict_414309, (tuple_414354, str_414357))
# Adding element type (key, value) (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 18)
tuple_414358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 18)
# Adding element type (line 18)
str_414359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 18), 'str', 'D')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 18), tuple_414358, str_414359)
# Adding element type (line 18)
str_414360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 22), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 18), tuple_414358, str_414360)

str_414361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 27), 'str', 'D')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), dict_414309, (tuple_414358, str_414361))
# Adding element type (key, value) (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 18)
tuple_414362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 33), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 18)
# Adding element type (line 18)
str_414363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 33), 'str', 'D')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 33), tuple_414362, str_414363)
# Adding element type (line 18)
str_414364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 37), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 33), tuple_414362, str_414364)

str_414365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 42), 'str', 'D')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), dict_414309, (tuple_414362, str_414365))
# Adding element type (key, value) (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 18)
tuple_414366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 18)
# Adding element type (line 18)
str_414367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 48), 'str', 'D')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 48), tuple_414366, str_414367)
# Adding element type (line 18)
str_414368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 52), 'str', 'F')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 48), tuple_414366, str_414368)

str_414369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 57), 'str', 'D')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), dict_414309, (tuple_414366, str_414369))
# Adding element type (key, value) (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 19)
tuple_414370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 19)
# Adding element type (line 19)
str_414371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 18), 'str', 'D')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 18), tuple_414370, str_414371)
# Adding element type (line 19)
str_414372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 22), 'str', 'D')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 18), tuple_414370, str_414372)

str_414373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 27), 'str', 'D')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), dict_414309, (tuple_414370, str_414373))

# Assigning a type to the variable '_coerce_rules' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), '_coerce_rules', dict_414309)

@norecursion
def coerce(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'coerce'
    module_type_store = module_type_store.open_function_context('coerce', 22, 0, False)
    
    # Passed parameters checking function
    coerce.stypy_localization = localization
    coerce.stypy_type_of_self = None
    coerce.stypy_type_store = module_type_store
    coerce.stypy_function_name = 'coerce'
    coerce.stypy_param_names_list = ['x', 'y']
    coerce.stypy_varargs_param_name = None
    coerce.stypy_kwargs_param_name = None
    coerce.stypy_call_defaults = defaults
    coerce.stypy_call_varargs = varargs
    coerce.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'coerce', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'coerce', localization, ['x', 'y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'coerce(...)' code ##################

    
    
    # Getting the type of 'x' (line 23)
    x_414374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 7), 'x')
    str_414375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 16), 'str', 'fdFD')
    # Applying the binary operator 'notin' (line 23)
    result_contains_414376 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 7), 'notin', x_414374, str_414375)
    
    # Testing the type of an if condition (line 23)
    if_condition_414377 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 23, 4), result_contains_414376)
    # Assigning a type to the variable 'if_condition_414377' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'if_condition_414377', if_condition_414377)
    # SSA begins for if statement (line 23)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 24):
    str_414378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 12), 'str', 'd')
    # Assigning a type to the variable 'x' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'x', str_414378)
    # SSA join for if statement (line 23)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'y' (line 25)
    y_414379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 7), 'y')
    str_414380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 16), 'str', 'fdFD')
    # Applying the binary operator 'notin' (line 25)
    result_contains_414381 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 7), 'notin', y_414379, str_414380)
    
    # Testing the type of an if condition (line 25)
    if_condition_414382 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 25, 4), result_contains_414381)
    # Assigning a type to the variable 'if_condition_414382' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'if_condition_414382', if_condition_414382)
    # SSA begins for if statement (line 25)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 26):
    str_414383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 12), 'str', 'd')
    # Assigning a type to the variable 'y' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'y', str_414383)
    # SSA join for if statement (line 25)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 27)
    tuple_414384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 27)
    # Adding element type (line 27)
    # Getting the type of 'x' (line 27)
    x_414385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 25), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 25), tuple_414384, x_414385)
    # Adding element type (line 27)
    # Getting the type of 'y' (line 27)
    y_414386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 27), 'y')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 25), tuple_414384, y_414386)
    
    # Getting the type of '_coerce_rules' (line 27)
    _coerce_rules_414387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 11), '_coerce_rules')
    # Obtaining the member '__getitem__' of a type (line 27)
    getitem___414388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 11), _coerce_rules_414387, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 27)
    subscript_call_result_414389 = invoke(stypy.reporting.localization.Localization(__file__, 27, 11), getitem___414388, tuple_414384)
    
    # Assigning a type to the variable 'stypy_return_type' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type', subscript_call_result_414389)
    
    # ################# End of 'coerce(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'coerce' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_414390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_414390)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'coerce'
    return stypy_return_type_414390

# Assigning a type to the variable 'coerce' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'coerce', coerce)

@norecursion
def id(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'id'
    module_type_store = module_type_store.open_function_context('id', 30, 0, False)
    
    # Passed parameters checking function
    id.stypy_localization = localization
    id.stypy_type_of_self = None
    id.stypy_type_store = module_type_store
    id.stypy_function_name = 'id'
    id.stypy_param_names_list = ['x']
    id.stypy_varargs_param_name = None
    id.stypy_kwargs_param_name = None
    id.stypy_call_defaults = defaults
    id.stypy_call_varargs = varargs
    id.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'id', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'id', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'id(...)' code ##################

    # Getting the type of 'x' (line 31)
    x_414391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type', x_414391)
    
    # ################# End of 'id(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'id' in the type store
    # Getting the type of 'stypy_return_type' (line 30)
    stypy_return_type_414392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_414392)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'id'
    return stypy_return_type_414392

# Assigning a type to the variable 'id' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'id', id)

@norecursion
def make_system(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'make_system'
    module_type_store = module_type_store.open_function_context('make_system', 34, 0, False)
    
    # Passed parameters checking function
    make_system.stypy_localization = localization
    make_system.stypy_type_of_self = None
    make_system.stypy_type_store = module_type_store
    make_system.stypy_function_name = 'make_system'
    make_system.stypy_param_names_list = ['A', 'M', 'x0', 'b']
    make_system.stypy_varargs_param_name = None
    make_system.stypy_kwargs_param_name = None
    make_system.stypy_call_defaults = defaults
    make_system.stypy_call_varargs = varargs
    make_system.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'make_system', ['A', 'M', 'x0', 'b'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'make_system', localization, ['A', 'M', 'x0', 'b'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'make_system(...)' code ##################

    str_414393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, (-1)), 'str', 'Make a linear system Ax=b\n\n    Parameters\n    ----------\n    A : LinearOperator\n        sparse or dense matrix (or any valid input to aslinearoperator)\n    M : {LinearOperator, Nones}\n        preconditioner\n        sparse or dense matrix (or any valid input to aslinearoperator)\n    x0 : {array_like, None}\n        initial guess to iterative method\n    b : array_like\n        right hand side\n\n    Returns\n    -------\n    (A, M, x, b, postprocess)\n        A : LinearOperator\n            matrix of the linear system\n        M : LinearOperator\n            preconditioner\n        x : rank 1 ndarray\n            initial guess\n        b : rank 1 ndarray\n            right hand side\n        postprocess : function\n            converts the solution vector to the appropriate\n            type and dimensions (e.g. (N,1) matrix)\n\n    ')
    
    # Assigning a Name to a Name (line 65):
    # Getting the type of 'A' (line 65)
    A_414394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 9), 'A')
    # Assigning a type to the variable 'A_' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'A_', A_414394)
    
    # Assigning a Call to a Name (line 66):
    
    # Call to aslinearoperator(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'A' (line 66)
    A_414396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'A', False)
    # Processing the call keyword arguments (line 66)
    kwargs_414397 = {}
    # Getting the type of 'aslinearoperator' (line 66)
    aslinearoperator_414395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'aslinearoperator', False)
    # Calling aslinearoperator(args, kwargs) (line 66)
    aslinearoperator_call_result_414398 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), aslinearoperator_414395, *[A_414396], **kwargs_414397)
    
    # Assigning a type to the variable 'A' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'A', aslinearoperator_call_result_414398)
    
    
    
    # Obtaining the type of the subscript
    int_414399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 15), 'int')
    # Getting the type of 'A' (line 68)
    A_414400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 7), 'A')
    # Obtaining the member 'shape' of a type (line 68)
    shape_414401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 7), A_414400, 'shape')
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___414402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 7), shape_414401, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_414403 = invoke(stypy.reporting.localization.Localization(__file__, 68, 7), getitem___414402, int_414399)
    
    
    # Obtaining the type of the subscript
    int_414404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 29), 'int')
    # Getting the type of 'A' (line 68)
    A_414405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 21), 'A')
    # Obtaining the member 'shape' of a type (line 68)
    shape_414406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 21), A_414405, 'shape')
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___414407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 21), shape_414406, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_414408 = invoke(stypy.reporting.localization.Localization(__file__, 68, 21), getitem___414407, int_414404)
    
    # Applying the binary operator '!=' (line 68)
    result_ne_414409 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 7), '!=', subscript_call_result_414403, subscript_call_result_414408)
    
    # Testing the type of an if condition (line 68)
    if_condition_414410 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 4), result_ne_414409)
    # Assigning a type to the variable 'if_condition_414410' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'if_condition_414410', if_condition_414410)
    # SSA begins for if statement (line 68)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 69)
    # Processing the call arguments (line 69)
    str_414412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 25), 'str', 'expected square matrix, but got shape=%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 69)
    tuple_414413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 71), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 69)
    # Adding element type (line 69)
    # Getting the type of 'A' (line 69)
    A_414414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 71), 'A', False)
    # Obtaining the member 'shape' of a type (line 69)
    shape_414415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 71), A_414414, 'shape')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 71), tuple_414413, shape_414415)
    
    # Applying the binary operator '%' (line 69)
    result_mod_414416 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 25), '%', str_414412, tuple_414413)
    
    # Processing the call keyword arguments (line 69)
    kwargs_414417 = {}
    # Getting the type of 'ValueError' (line 69)
    ValueError_414411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 69)
    ValueError_call_result_414418 = invoke(stypy.reporting.localization.Localization(__file__, 69, 14), ValueError_414411, *[result_mod_414416], **kwargs_414417)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 69, 8), ValueError_call_result_414418, 'raise parameter', BaseException)
    # SSA join for if statement (line 68)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 71):
    
    # Obtaining the type of the subscript
    int_414419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 16), 'int')
    # Getting the type of 'A' (line 71)
    A_414420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'A')
    # Obtaining the member 'shape' of a type (line 71)
    shape_414421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), A_414420, 'shape')
    # Obtaining the member '__getitem__' of a type (line 71)
    getitem___414422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), shape_414421, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
    subscript_call_result_414423 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), getitem___414422, int_414419)
    
    # Assigning a type to the variable 'N' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'N', subscript_call_result_414423)
    
    # Assigning a Call to a Name (line 73):
    
    # Call to asanyarray(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'b' (line 73)
    b_414425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 19), 'b', False)
    # Processing the call keyword arguments (line 73)
    kwargs_414426 = {}
    # Getting the type of 'asanyarray' (line 73)
    asanyarray_414424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 73)
    asanyarray_call_result_414427 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), asanyarray_414424, *[b_414425], **kwargs_414426)
    
    # Assigning a type to the variable 'b' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'b', asanyarray_call_result_414427)
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'b' (line 75)
    b_414428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'b')
    # Obtaining the member 'shape' of a type (line 75)
    shape_414429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 12), b_414428, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 75)
    tuple_414430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 75)
    # Adding element type (line 75)
    # Getting the type of 'N' (line 75)
    N_414431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 24), 'N')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 24), tuple_414430, N_414431)
    # Adding element type (line 75)
    int_414432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 24), tuple_414430, int_414432)
    
    # Applying the binary operator '==' (line 75)
    result_eq_414433 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 12), '==', shape_414429, tuple_414430)
    
    
    # Getting the type of 'b' (line 75)
    b_414434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 32), 'b')
    # Obtaining the member 'shape' of a type (line 75)
    shape_414435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 32), b_414434, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 75)
    tuple_414436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 75)
    # Adding element type (line 75)
    # Getting the type of 'N' (line 75)
    N_414437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 44), 'N')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 44), tuple_414436, N_414437)
    
    # Applying the binary operator '==' (line 75)
    result_eq_414438 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 32), '==', shape_414435, tuple_414436)
    
    # Applying the binary operator 'or' (line 75)
    result_or_keyword_414439 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 12), 'or', result_eq_414433, result_eq_414438)
    
    # Applying the 'not' unary operator (line 75)
    result_not__414440 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 7), 'not', result_or_keyword_414439)
    
    # Testing the type of an if condition (line 75)
    if_condition_414441 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 4), result_not__414440)
    # Assigning a type to the variable 'if_condition_414441' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'if_condition_414441', if_condition_414441)
    # SSA begins for if statement (line 75)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 76)
    # Processing the call arguments (line 76)
    str_414443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 25), 'str', 'A and b have incompatible dimensions')
    # Processing the call keyword arguments (line 76)
    kwargs_414444 = {}
    # Getting the type of 'ValueError' (line 76)
    ValueError_414442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 76)
    ValueError_call_result_414445 = invoke(stypy.reporting.localization.Localization(__file__, 76, 14), ValueError_414442, *[str_414443], **kwargs_414444)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 76, 8), ValueError_call_result_414445, 'raise parameter', BaseException)
    # SSA join for if statement (line 75)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'b' (line 78)
    b_414446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 7), 'b')
    # Obtaining the member 'dtype' of a type (line 78)
    dtype_414447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 7), b_414446, 'dtype')
    # Obtaining the member 'char' of a type (line 78)
    char_414448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 7), dtype_414447, 'char')
    str_414449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 27), 'str', 'fdFD')
    # Applying the binary operator 'notin' (line 78)
    result_contains_414450 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 7), 'notin', char_414448, str_414449)
    
    # Testing the type of an if condition (line 78)
    if_condition_414451 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 4), result_contains_414450)
    # Assigning a type to the variable 'if_condition_414451' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'if_condition_414451', if_condition_414451)
    # SSA begins for if statement (line 78)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 79):
    
    # Call to astype(...): (line 79)
    # Processing the call arguments (line 79)
    str_414454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 21), 'str', 'd')
    # Processing the call keyword arguments (line 79)
    kwargs_414455 = {}
    # Getting the type of 'b' (line 79)
    b_414452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'b', False)
    # Obtaining the member 'astype' of a type (line 79)
    astype_414453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), b_414452, 'astype')
    # Calling astype(args, kwargs) (line 79)
    astype_call_result_414456 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), astype_414453, *[str_414454], **kwargs_414455)
    
    # Assigning a type to the variable 'b' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'b', astype_call_result_414456)
    # SSA join for if statement (line 78)
    module_type_store = module_type_store.join_ssa_context()
    

    @norecursion
    def postprocess(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'postprocess'
        module_type_store = module_type_store.open_function_context('postprocess', 81, 4, False)
        
        # Passed parameters checking function
        postprocess.stypy_localization = localization
        postprocess.stypy_type_of_self = None
        postprocess.stypy_type_store = module_type_store
        postprocess.stypy_function_name = 'postprocess'
        postprocess.stypy_param_names_list = ['x']
        postprocess.stypy_varargs_param_name = None
        postprocess.stypy_kwargs_param_name = None
        postprocess.stypy_call_defaults = defaults
        postprocess.stypy_call_varargs = varargs
        postprocess.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'postprocess', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'postprocess', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'postprocess(...)' code ##################

        
        
        # Call to isinstance(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'b' (line 82)
        b_414458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 22), 'b', False)
        # Getting the type of 'matrix' (line 82)
        matrix_414459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 24), 'matrix', False)
        # Processing the call keyword arguments (line 82)
        kwargs_414460 = {}
        # Getting the type of 'isinstance' (line 82)
        isinstance_414457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 82)
        isinstance_call_result_414461 = invoke(stypy.reporting.localization.Localization(__file__, 82, 11), isinstance_414457, *[b_414458, matrix_414459], **kwargs_414460)
        
        # Testing the type of an if condition (line 82)
        if_condition_414462 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 8), isinstance_call_result_414461)
        # Assigning a type to the variable 'if_condition_414462' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'if_condition_414462', if_condition_414462)
        # SSA begins for if statement (line 82)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 83):
        
        # Call to asmatrix(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'x' (line 83)
        x_414464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 25), 'x', False)
        # Processing the call keyword arguments (line 83)
        kwargs_414465 = {}
        # Getting the type of 'asmatrix' (line 83)
        asmatrix_414463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'asmatrix', False)
        # Calling asmatrix(args, kwargs) (line 83)
        asmatrix_call_result_414466 = invoke(stypy.reporting.localization.Localization(__file__, 83, 16), asmatrix_414463, *[x_414464], **kwargs_414465)
        
        # Assigning a type to the variable 'x' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'x', asmatrix_call_result_414466)
        # SSA join for if statement (line 82)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to reshape(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'b' (line 84)
        b_414469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 25), 'b', False)
        # Obtaining the member 'shape' of a type (line 84)
        shape_414470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 25), b_414469, 'shape')
        # Processing the call keyword arguments (line 84)
        kwargs_414471 = {}
        # Getting the type of 'x' (line 84)
        x_414467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 15), 'x', False)
        # Obtaining the member 'reshape' of a type (line 84)
        reshape_414468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 15), x_414467, 'reshape')
        # Calling reshape(args, kwargs) (line 84)
        reshape_call_result_414472 = invoke(stypy.reporting.localization.Localization(__file__, 84, 15), reshape_414468, *[shape_414470], **kwargs_414471)
        
        # Assigning a type to the variable 'stypy_return_type' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'stypy_return_type', reshape_call_result_414472)
        
        # ################# End of 'postprocess(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'postprocess' in the type store
        # Getting the type of 'stypy_return_type' (line 81)
        stypy_return_type_414473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_414473)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'postprocess'
        return stypy_return_type_414473

    # Assigning a type to the variable 'postprocess' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'postprocess', postprocess)
    
    # Type idiom detected: calculating its left and rigth part (line 86)
    str_414474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 17), 'str', 'dtype')
    # Getting the type of 'A' (line 86)
    A_414475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'A')
    
    (may_be_414476, more_types_in_union_414477) = may_provide_member(str_414474, A_414475)

    if may_be_414476:

        if more_types_in_union_414477:
            # Runtime conditional SSA (line 86)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'A' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'A', remove_not_member_provider_from_union(A_414475, 'dtype'))
        
        # Assigning a Attribute to a Name (line 87):
        # Getting the type of 'A' (line 87)
        A_414478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'A')
        # Obtaining the member 'dtype' of a type (line 87)
        dtype_414479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 16), A_414478, 'dtype')
        # Obtaining the member 'char' of a type (line 87)
        char_414480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 16), dtype_414479, 'char')
        # Assigning a type to the variable 'xtype' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'xtype', char_414480)

        if more_types_in_union_414477:
            # Runtime conditional SSA for else branch (line 86)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_414476) or more_types_in_union_414477):
        # Assigning a type to the variable 'A' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'A', remove_member_provider_from_union(A_414475, 'dtype'))
        
        # Assigning a Attribute to a Name (line 89):
        
        # Call to matvec(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'b' (line 89)
        b_414483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 25), 'b', False)
        # Processing the call keyword arguments (line 89)
        kwargs_414484 = {}
        # Getting the type of 'A' (line 89)
        A_414481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'A', False)
        # Obtaining the member 'matvec' of a type (line 89)
        matvec_414482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 16), A_414481, 'matvec')
        # Calling matvec(args, kwargs) (line 89)
        matvec_call_result_414485 = invoke(stypy.reporting.localization.Localization(__file__, 89, 16), matvec_414482, *[b_414483], **kwargs_414484)
        
        # Obtaining the member 'dtype' of a type (line 89)
        dtype_414486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 16), matvec_call_result_414485, 'dtype')
        # Obtaining the member 'char' of a type (line 89)
        char_414487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 16), dtype_414486, 'char')
        # Assigning a type to the variable 'xtype' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'xtype', char_414487)

        if (may_be_414476 and more_types_in_union_414477):
            # SSA join for if statement (line 86)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 90):
    
    # Call to coerce(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'xtype' (line 90)
    xtype_414489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 19), 'xtype', False)
    # Getting the type of 'b' (line 90)
    b_414490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 26), 'b', False)
    # Obtaining the member 'dtype' of a type (line 90)
    dtype_414491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 26), b_414490, 'dtype')
    # Obtaining the member 'char' of a type (line 90)
    char_414492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 26), dtype_414491, 'char')
    # Processing the call keyword arguments (line 90)
    kwargs_414493 = {}
    # Getting the type of 'coerce' (line 90)
    coerce_414488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'coerce', False)
    # Calling coerce(args, kwargs) (line 90)
    coerce_call_result_414494 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), coerce_414488, *[xtype_414489, char_414492], **kwargs_414493)
    
    # Assigning a type to the variable 'xtype' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'xtype', coerce_call_result_414494)
    
    # Assigning a Call to a Name (line 92):
    
    # Call to asarray(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'b' (line 92)
    b_414496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'b', False)
    # Processing the call keyword arguments (line 92)
    # Getting the type of 'xtype' (line 92)
    xtype_414497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'xtype', False)
    keyword_414498 = xtype_414497
    kwargs_414499 = {'dtype': keyword_414498}
    # Getting the type of 'asarray' (line 92)
    asarray_414495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'asarray', False)
    # Calling asarray(args, kwargs) (line 92)
    asarray_call_result_414500 = invoke(stypy.reporting.localization.Localization(__file__, 92, 8), asarray_414495, *[b_414496], **kwargs_414499)
    
    # Assigning a type to the variable 'b' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'b', asarray_call_result_414500)
    
    # Assigning a Call to a Name (line 93):
    
    # Call to ravel(...): (line 93)
    # Processing the call keyword arguments (line 93)
    kwargs_414503 = {}
    # Getting the type of 'b' (line 93)
    b_414501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'b', False)
    # Obtaining the member 'ravel' of a type (line 93)
    ravel_414502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), b_414501, 'ravel')
    # Calling ravel(args, kwargs) (line 93)
    ravel_call_result_414504 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), ravel_414502, *[], **kwargs_414503)
    
    # Assigning a type to the variable 'b' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'b', ravel_call_result_414504)
    
    # Type idiom detected: calculating its left and rigth part (line 95)
    # Getting the type of 'x0' (line 95)
    x0_414505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 7), 'x0')
    # Getting the type of 'None' (line 95)
    None_414506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 13), 'None')
    
    (may_be_414507, more_types_in_union_414508) = may_be_none(x0_414505, None_414506)

    if may_be_414507:

        if more_types_in_union_414508:
            # Runtime conditional SSA (line 95)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 96):
        
        # Call to zeros(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'N' (line 96)
        N_414510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 18), 'N', False)
        # Processing the call keyword arguments (line 96)
        # Getting the type of 'xtype' (line 96)
        xtype_414511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 27), 'xtype', False)
        keyword_414512 = xtype_414511
        kwargs_414513 = {'dtype': keyword_414512}
        # Getting the type of 'zeros' (line 96)
        zeros_414509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'zeros', False)
        # Calling zeros(args, kwargs) (line 96)
        zeros_call_result_414514 = invoke(stypy.reporting.localization.Localization(__file__, 96, 12), zeros_414509, *[N_414510], **kwargs_414513)
        
        # Assigning a type to the variable 'x' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'x', zeros_call_result_414514)

        if more_types_in_union_414508:
            # Runtime conditional SSA for else branch (line 95)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_414507) or more_types_in_union_414508):
        
        # Assigning a Call to a Name (line 98):
        
        # Call to array(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'x0' (line 98)
        x0_414516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 18), 'x0', False)
        # Processing the call keyword arguments (line 98)
        # Getting the type of 'xtype' (line 98)
        xtype_414517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 28), 'xtype', False)
        keyword_414518 = xtype_414517
        kwargs_414519 = {'dtype': keyword_414518}
        # Getting the type of 'array' (line 98)
        array_414515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'array', False)
        # Calling array(args, kwargs) (line 98)
        array_call_result_414520 = invoke(stypy.reporting.localization.Localization(__file__, 98, 12), array_414515, *[x0_414516], **kwargs_414519)
        
        # Assigning a type to the variable 'x' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'x', array_call_result_414520)
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'x' (line 99)
        x_414521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'x')
        # Obtaining the member 'shape' of a type (line 99)
        shape_414522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), x_414521, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 99)
        tuple_414523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 99)
        # Adding element type (line 99)
        # Getting the type of 'N' (line 99)
        N_414524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 28), 'N')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 28), tuple_414523, N_414524)
        # Adding element type (line 99)
        int_414525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 28), tuple_414523, int_414525)
        
        # Applying the binary operator '==' (line 99)
        result_eq_414526 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 16), '==', shape_414522, tuple_414523)
        
        
        # Getting the type of 'x' (line 99)
        x_414527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 36), 'x')
        # Obtaining the member 'shape' of a type (line 99)
        shape_414528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 36), x_414527, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 99)
        tuple_414529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 99)
        # Adding element type (line 99)
        # Getting the type of 'N' (line 99)
        N_414530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 48), 'N')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 48), tuple_414529, N_414530)
        
        # Applying the binary operator '==' (line 99)
        result_eq_414531 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 36), '==', shape_414528, tuple_414529)
        
        # Applying the binary operator 'or' (line 99)
        result_or_keyword_414532 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 16), 'or', result_eq_414526, result_eq_414531)
        
        # Applying the 'not' unary operator (line 99)
        result_not__414533 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 11), 'not', result_or_keyword_414532)
        
        # Testing the type of an if condition (line 99)
        if_condition_414534 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 8), result_not__414533)
        # Assigning a type to the variable 'if_condition_414534' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'if_condition_414534', if_condition_414534)
        # SSA begins for if statement (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 100)
        # Processing the call arguments (line 100)
        str_414536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 29), 'str', 'A and x have incompatible dimensions')
        # Processing the call keyword arguments (line 100)
        kwargs_414537 = {}
        # Getting the type of 'ValueError' (line 100)
        ValueError_414535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 100)
        ValueError_call_result_414538 = invoke(stypy.reporting.localization.Localization(__file__, 100, 18), ValueError_414535, *[str_414536], **kwargs_414537)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 100, 12), ValueError_call_result_414538, 'raise parameter', BaseException)
        # SSA join for if statement (line 99)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 101):
        
        # Call to ravel(...): (line 101)
        # Processing the call keyword arguments (line 101)
        kwargs_414541 = {}
        # Getting the type of 'x' (line 101)
        x_414539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'x', False)
        # Obtaining the member 'ravel' of a type (line 101)
        ravel_414540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), x_414539, 'ravel')
        # Calling ravel(args, kwargs) (line 101)
        ravel_call_result_414542 = invoke(stypy.reporting.localization.Localization(__file__, 101, 12), ravel_414540, *[], **kwargs_414541)
        
        # Assigning a type to the variable 'x' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'x', ravel_call_result_414542)

        if (may_be_414507 and more_types_in_union_414508):
            # SSA join for if statement (line 95)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 104)
    # Getting the type of 'M' (line 104)
    M_414543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 7), 'M')
    # Getting the type of 'None' (line 104)
    None_414544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'None')
    
    (may_be_414545, more_types_in_union_414546) = may_be_none(M_414543, None_414544)

    if may_be_414545:

        if more_types_in_union_414546:
            # Runtime conditional SSA (line 104)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Type idiom detected: calculating its left and rigth part (line 105)
        str_414547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 22), 'str', 'psolve')
        # Getting the type of 'A_' (line 105)
        A__414548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 19), 'A_')
        
        (may_be_414549, more_types_in_union_414550) = may_provide_member(str_414547, A__414548)

        if may_be_414549:

            if more_types_in_union_414550:
                # Runtime conditional SSA (line 105)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'A_' (line 105)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'A_', remove_not_member_provider_from_union(A__414548, 'psolve'))
            
            # Assigning a Attribute to a Name (line 106):
            # Getting the type of 'A_' (line 106)
            A__414551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 21), 'A_')
            # Obtaining the member 'psolve' of a type (line 106)
            psolve_414552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 21), A__414551, 'psolve')
            # Assigning a type to the variable 'psolve' (line 106)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'psolve', psolve_414552)

            if more_types_in_union_414550:
                # Runtime conditional SSA for else branch (line 105)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_414549) or more_types_in_union_414550):
            # Assigning a type to the variable 'A_' (line 105)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'A_', remove_member_provider_from_union(A__414548, 'psolve'))
            
            # Assigning a Name to a Name (line 108):
            # Getting the type of 'id' (line 108)
            id_414553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 21), 'id')
            # Assigning a type to the variable 'psolve' (line 108)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'psolve', id_414553)

            if (may_be_414549 and more_types_in_union_414550):
                # SSA join for if statement (line 105)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 109)
        str_414554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 22), 'str', 'rpsolve')
        # Getting the type of 'A_' (line 109)
        A__414555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'A_')
        
        (may_be_414556, more_types_in_union_414557) = may_provide_member(str_414554, A__414555)

        if may_be_414556:

            if more_types_in_union_414557:
                # Runtime conditional SSA (line 109)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'A_' (line 109)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'A_', remove_not_member_provider_from_union(A__414555, 'rpsolve'))
            
            # Assigning a Attribute to a Name (line 110):
            # Getting the type of 'A_' (line 110)
            A__414558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 22), 'A_')
            # Obtaining the member 'rpsolve' of a type (line 110)
            rpsolve_414559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 22), A__414558, 'rpsolve')
            # Assigning a type to the variable 'rpsolve' (line 110)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'rpsolve', rpsolve_414559)

            if more_types_in_union_414557:
                # Runtime conditional SSA for else branch (line 109)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_414556) or more_types_in_union_414557):
            # Assigning a type to the variable 'A_' (line 109)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'A_', remove_member_provider_from_union(A__414555, 'rpsolve'))
            
            # Assigning a Name to a Name (line 112):
            # Getting the type of 'id' (line 112)
            id_414560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 22), 'id')
            # Assigning a type to the variable 'rpsolve' (line 112)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'rpsolve', id_414560)

            if (may_be_414556 and more_types_in_union_414557):
                # SSA join for if statement (line 109)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'psolve' (line 113)
        psolve_414561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 11), 'psolve')
        # Getting the type of 'id' (line 113)
        id_414562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 21), 'id')
        # Applying the binary operator 'is' (line 113)
        result_is__414563 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 11), 'is', psolve_414561, id_414562)
        
        
        # Getting the type of 'rpsolve' (line 113)
        rpsolve_414564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 28), 'rpsolve')
        # Getting the type of 'id' (line 113)
        id_414565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 39), 'id')
        # Applying the binary operator 'is' (line 113)
        result_is__414566 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 28), 'is', rpsolve_414564, id_414565)
        
        # Applying the binary operator 'and' (line 113)
        result_and_keyword_414567 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 11), 'and', result_is__414563, result_is__414566)
        
        # Testing the type of an if condition (line 113)
        if_condition_414568 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 8), result_and_keyword_414567)
        # Assigning a type to the variable 'if_condition_414568' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'if_condition_414568', if_condition_414568)
        # SSA begins for if statement (line 113)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 114):
        
        # Call to IdentityOperator(...): (line 114)
        # Processing the call keyword arguments (line 114)
        # Getting the type of 'A' (line 114)
        A_414570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 39), 'A', False)
        # Obtaining the member 'shape' of a type (line 114)
        shape_414571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 39), A_414570, 'shape')
        keyword_414572 = shape_414571
        # Getting the type of 'A' (line 114)
        A_414573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 54), 'A', False)
        # Obtaining the member 'dtype' of a type (line 114)
        dtype_414574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 54), A_414573, 'dtype')
        keyword_414575 = dtype_414574
        kwargs_414576 = {'dtype': keyword_414575, 'shape': keyword_414572}
        # Getting the type of 'IdentityOperator' (line 114)
        IdentityOperator_414569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'IdentityOperator', False)
        # Calling IdentityOperator(args, kwargs) (line 114)
        IdentityOperator_call_result_414577 = invoke(stypy.reporting.localization.Localization(__file__, 114, 16), IdentityOperator_414569, *[], **kwargs_414576)
        
        # Assigning a type to the variable 'M' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'M', IdentityOperator_call_result_414577)
        # SSA branch for the else part of an if statement (line 113)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 116):
        
        # Call to LinearOperator(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'A' (line 116)
        A_414579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 31), 'A', False)
        # Obtaining the member 'shape' of a type (line 116)
        shape_414580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 31), A_414579, 'shape')
        # Processing the call keyword arguments (line 116)
        # Getting the type of 'psolve' (line 116)
        psolve_414581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 47), 'psolve', False)
        keyword_414582 = psolve_414581
        # Getting the type of 'rpsolve' (line 116)
        rpsolve_414583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 63), 'rpsolve', False)
        keyword_414584 = rpsolve_414583
        # Getting the type of 'A' (line 117)
        A_414585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 37), 'A', False)
        # Obtaining the member 'dtype' of a type (line 117)
        dtype_414586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 37), A_414585, 'dtype')
        keyword_414587 = dtype_414586
        kwargs_414588 = {'dtype': keyword_414587, 'rmatvec': keyword_414584, 'matvec': keyword_414582}
        # Getting the type of 'LinearOperator' (line 116)
        LinearOperator_414578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'LinearOperator', False)
        # Calling LinearOperator(args, kwargs) (line 116)
        LinearOperator_call_result_414589 = invoke(stypy.reporting.localization.Localization(__file__, 116, 16), LinearOperator_414578, *[shape_414580], **kwargs_414588)
        
        # Assigning a type to the variable 'M' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'M', LinearOperator_call_result_414589)
        # SSA join for if statement (line 113)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_414546:
            # Runtime conditional SSA for else branch (line 104)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_414545) or more_types_in_union_414546):
        
        # Assigning a Call to a Name (line 119):
        
        # Call to aslinearoperator(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'M' (line 119)
        M_414591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 29), 'M', False)
        # Processing the call keyword arguments (line 119)
        kwargs_414592 = {}
        # Getting the type of 'aslinearoperator' (line 119)
        aslinearoperator_414590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'aslinearoperator', False)
        # Calling aslinearoperator(args, kwargs) (line 119)
        aslinearoperator_call_result_414593 = invoke(stypy.reporting.localization.Localization(__file__, 119, 12), aslinearoperator_414590, *[M_414591], **kwargs_414592)
        
        # Assigning a type to the variable 'M' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'M', aslinearoperator_call_result_414593)
        
        
        # Getting the type of 'A' (line 120)
        A_414594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 11), 'A')
        # Obtaining the member 'shape' of a type (line 120)
        shape_414595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 11), A_414594, 'shape')
        # Getting the type of 'M' (line 120)
        M_414596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 22), 'M')
        # Obtaining the member 'shape' of a type (line 120)
        shape_414597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 22), M_414596, 'shape')
        # Applying the binary operator '!=' (line 120)
        result_ne_414598 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 11), '!=', shape_414595, shape_414597)
        
        # Testing the type of an if condition (line 120)
        if_condition_414599 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 120, 8), result_ne_414598)
        # Assigning a type to the variable 'if_condition_414599' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'if_condition_414599', if_condition_414599)
        # SSA begins for if statement (line 120)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 121)
        # Processing the call arguments (line 121)
        str_414601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 29), 'str', 'matrix and preconditioner have different shapes')
        # Processing the call keyword arguments (line 121)
        kwargs_414602 = {}
        # Getting the type of 'ValueError' (line 121)
        ValueError_414600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 121)
        ValueError_call_result_414603 = invoke(stypy.reporting.localization.Localization(__file__, 121, 18), ValueError_414600, *[str_414601], **kwargs_414602)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 121, 12), ValueError_call_result_414603, 'raise parameter', BaseException)
        # SSA join for if statement (line 120)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_414545 and more_types_in_union_414546):
            # SSA join for if statement (line 104)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Obtaining an instance of the builtin type 'tuple' (line 123)
    tuple_414604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 123)
    # Adding element type (line 123)
    # Getting the type of 'A' (line 123)
    A_414605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 'A')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 11), tuple_414604, A_414605)
    # Adding element type (line 123)
    # Getting the type of 'M' (line 123)
    M_414606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 14), 'M')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 11), tuple_414604, M_414606)
    # Adding element type (line 123)
    # Getting the type of 'x' (line 123)
    x_414607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 17), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 11), tuple_414604, x_414607)
    # Adding element type (line 123)
    # Getting the type of 'b' (line 123)
    b_414608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 20), 'b')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 11), tuple_414604, b_414608)
    # Adding element type (line 123)
    # Getting the type of 'postprocess' (line 123)
    postprocess_414609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 23), 'postprocess')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 11), tuple_414604, postprocess_414609)
    
    # Assigning a type to the variable 'stypy_return_type' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type', tuple_414604)
    
    # ################# End of 'make_system(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'make_system' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_414610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_414610)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'make_system'
    return stypy_return_type_414610

# Assigning a type to the variable 'make_system' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'make_system', make_system)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
