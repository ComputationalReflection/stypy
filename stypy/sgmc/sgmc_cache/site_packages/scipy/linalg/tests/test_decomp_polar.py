
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.linalg import norm
5: from numpy.testing import (assert_, assert_allclose, assert_equal)
6: from scipy.linalg import polar, eigh
7: 
8: 
9: diag2 = np.array([[2, 0], [0, 3]])
10: a13 = np.array([[1, 2, 2]])
11: 
12: precomputed_cases = [
13:     [[[0]], 'right', [[1]], [[0]]],
14:     [[[0]], 'left', [[1]], [[0]]],
15:     [[[9]], 'right', [[1]], [[9]]],
16:     [[[9]], 'left', [[1]], [[9]]],
17:     [diag2, 'right', np.eye(2), diag2],
18:     [diag2, 'left', np.eye(2), diag2],
19:     [a13, 'right', a13/norm(a13[0]), a13.T.dot(a13)/norm(a13[0])],
20: ]
21: 
22: verify_cases = [
23:     [[1, 2], [3, 4]],
24:     [[1, 2, 3]],
25:     [[1], [2], [3]],
26:     [[1, 2, 3], [3, 4, 0]],
27:     [[1, 2], [3, 4], [5, 5]],
28:     [[1, 2], [3, 4+5j]],
29:     [[1, 2, 3j]],
30:     [[1], [2], [3j]],
31:     [[1, 2, 3+2j], [3, 4-1j, -4j]],
32:     [[1, 2], [3-2j, 4+0.5j], [5, 5]],
33:     [[10000, 10, 1], [-1, 2, 3j], [0, 1, 2]],
34: ]
35: 
36: 
37: def check_precomputed_polar(a, side, expected_u, expected_p):
38:     # Compare the result of the polar decomposition to a
39:     # precomputed result.
40:     u, p = polar(a, side=side)
41:     assert_allclose(u, expected_u, atol=1e-15)
42:     assert_allclose(p, expected_p, atol=1e-15)
43: 
44: 
45: def verify_polar(a):
46:     # Compute the polar decomposition, and then verify that
47:     # the result has all the expected properties.
48:     product_atol = np.sqrt(np.finfo(float).eps)
49: 
50:     aa = np.asarray(a)
51:     m, n = aa.shape
52: 
53:     u, p = polar(a, side='right')
54:     assert_equal(u.shape, (m, n))
55:     assert_equal(p.shape, (n, n))
56:     # a = up
57:     assert_allclose(u.dot(p), a, atol=product_atol)
58:     if m >= n:
59:         assert_allclose(u.conj().T.dot(u), np.eye(n), atol=1e-15)
60:     else:
61:         assert_allclose(u.dot(u.conj().T), np.eye(m), atol=1e-15)
62:     # p is Hermitian positive semidefinite.
63:     assert_allclose(p.conj().T, p)
64:     evals = eigh(p, eigvals_only=True)
65:     nonzero_evals = evals[abs(evals) > 1e-14]
66:     assert_((nonzero_evals >= 0).all())
67: 
68:     u, p = polar(a, side='left')
69:     assert_equal(u.shape, (m, n))
70:     assert_equal(p.shape, (m, m))
71:     # a = pu
72:     assert_allclose(p.dot(u), a, atol=product_atol)
73:     if m >= n:
74:         assert_allclose(u.conj().T.dot(u), np.eye(n), atol=1e-15)
75:     else:
76:         assert_allclose(u.dot(u.conj().T), np.eye(m), atol=1e-15)
77:     # p is Hermitian positive semidefinite.
78:     assert_allclose(p.conj().T, p)
79:     evals = eigh(p, eigvals_only=True)
80:     nonzero_evals = evals[abs(evals) > 1e-14]
81:     assert_((nonzero_evals >= 0).all())
82: 
83: 
84: def test_precomputed_cases():
85:     for a, side, expected_u, expected_p in precomputed_cases:
86:         check_precomputed_polar(a, side, expected_u, expected_p)
87: 
88: 
89: def test_verify_cases():
90:     for a in verify_cases:
91:         verify_polar(a)
92: 
93: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_74446 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_74446) is not StypyTypeError):

    if (import_74446 != 'pyd_module'):
        __import__(import_74446)
        sys_modules_74447 = sys.modules[import_74446]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_74447.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_74446)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.linalg import norm' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_74448 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.linalg')

if (type(import_74448) is not StypyTypeError):

    if (import_74448 != 'pyd_module'):
        __import__(import_74448)
        sys_modules_74449 = sys.modules[import_74448]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.linalg', sys_modules_74449.module_type_store, module_type_store, ['norm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_74449, sys_modules_74449.module_type_store, module_type_store)
    else:
        from numpy.linalg import norm

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.linalg', None, module_type_store, ['norm'], [norm])

else:
    # Assigning a type to the variable 'numpy.linalg' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.linalg', import_74448)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy.testing import assert_, assert_allclose, assert_equal' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_74450 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing')

if (type(import_74450) is not StypyTypeError):

    if (import_74450 != 'pyd_module'):
        __import__(import_74450)
        sys_modules_74451 = sys.modules[import_74450]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', sys_modules_74451.module_type_store, module_type_store, ['assert_', 'assert_allclose', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_74451, sys_modules_74451.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_allclose, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_allclose', 'assert_equal'], [assert_, assert_allclose, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', import_74450)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.linalg import polar, eigh' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_74452 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.linalg')

if (type(import_74452) is not StypyTypeError):

    if (import_74452 != 'pyd_module'):
        __import__(import_74452)
        sys_modules_74453 = sys.modules[import_74452]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.linalg', sys_modules_74453.module_type_store, module_type_store, ['polar', 'eigh'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_74453, sys_modules_74453.module_type_store, module_type_store)
    else:
        from scipy.linalg import polar, eigh

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.linalg', None, module_type_store, ['polar', 'eigh'], [polar, eigh])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.linalg', import_74452)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')


# Assigning a Call to a Name (line 9):

# Assigning a Call to a Name (line 9):

# Call to array(...): (line 9)
# Processing the call arguments (line 9)

# Obtaining an instance of the builtin type 'list' (line 9)
list_74456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)

# Obtaining an instance of the builtin type 'list' (line 9)
list_74457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
int_74458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 18), list_74457, int_74458)
# Adding element type (line 9)
int_74459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 18), list_74457, int_74459)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 17), list_74456, list_74457)
# Adding element type (line 9)

# Obtaining an instance of the builtin type 'list' (line 9)
list_74460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 26), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
int_74461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 26), list_74460, int_74461)
# Adding element type (line 9)
int_74462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 26), list_74460, int_74462)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 17), list_74456, list_74460)

# Processing the call keyword arguments (line 9)
kwargs_74463 = {}
# Getting the type of 'np' (line 9)
np_74454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'np', False)
# Obtaining the member 'array' of a type (line 9)
array_74455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 8), np_74454, 'array')
# Calling array(args, kwargs) (line 9)
array_call_result_74464 = invoke(stypy.reporting.localization.Localization(__file__, 9, 8), array_74455, *[list_74456], **kwargs_74463)

# Assigning a type to the variable 'diag2' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'diag2', array_call_result_74464)

# Assigning a Call to a Name (line 10):

# Assigning a Call to a Name (line 10):

# Call to array(...): (line 10)
# Processing the call arguments (line 10)

# Obtaining an instance of the builtin type 'list' (line 10)
list_74467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)

# Obtaining an instance of the builtin type 'list' (line 10)
list_74468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)
int_74469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 16), list_74468, int_74469)
# Adding element type (line 10)
int_74470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 16), list_74468, int_74470)
# Adding element type (line 10)
int_74471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 16), list_74468, int_74471)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 15), list_74467, list_74468)

# Processing the call keyword arguments (line 10)
kwargs_74472 = {}
# Getting the type of 'np' (line 10)
np_74465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 6), 'np', False)
# Obtaining the member 'array' of a type (line 10)
array_74466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 6), np_74465, 'array')
# Calling array(args, kwargs) (line 10)
array_call_result_74473 = invoke(stypy.reporting.localization.Localization(__file__, 10, 6), array_74466, *[list_74467], **kwargs_74472)

# Assigning a type to the variable 'a13' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'a13', array_call_result_74473)

# Assigning a List to a Name (line 12):

# Assigning a List to a Name (line 12):

# Obtaining an instance of the builtin type 'list' (line 12)
list_74474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)

# Obtaining an instance of the builtin type 'list' (line 13)
list_74475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)

# Obtaining an instance of the builtin type 'list' (line 13)
list_74476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)

# Obtaining an instance of the builtin type 'list' (line 13)
list_74477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 6), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)
int_74478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 7), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 6), list_74477, int_74478)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 5), list_74476, list_74477)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 4), list_74475, list_74476)
# Adding element type (line 13)
str_74479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 12), 'str', 'right')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 4), list_74475, str_74479)
# Adding element type (line 13)

# Obtaining an instance of the builtin type 'list' (line 13)
list_74480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)

# Obtaining an instance of the builtin type 'list' (line 13)
list_74481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)
int_74482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 22), list_74481, int_74482)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 21), list_74480, list_74481)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 4), list_74475, list_74480)
# Adding element type (line 13)

# Obtaining an instance of the builtin type 'list' (line 13)
list_74483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)

# Obtaining an instance of the builtin type 'list' (line 13)
list_74484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 29), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)
int_74485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 29), list_74484, int_74485)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 28), list_74483, list_74484)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 4), list_74475, list_74483)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 20), list_74474, list_74475)
# Adding element type (line 12)

# Obtaining an instance of the builtin type 'list' (line 14)
list_74486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'list' (line 14)
list_74487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'list' (line 14)
list_74488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 6), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
int_74489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 7), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 6), list_74488, int_74489)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 5), list_74487, list_74488)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 4), list_74486, list_74487)
# Adding element type (line 14)
str_74490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 12), 'str', 'left')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 4), list_74486, str_74490)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'list' (line 14)
list_74491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'list' (line 14)
list_74492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
int_74493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 21), list_74492, int_74493)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 20), list_74491, list_74492)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 4), list_74486, list_74491)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'list' (line 14)
list_74494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'list' (line 14)
list_74495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
int_74496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 28), list_74495, int_74496)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 27), list_74494, list_74495)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 4), list_74486, list_74494)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 20), list_74474, list_74486)
# Adding element type (line 12)

# Obtaining an instance of the builtin type 'list' (line 15)
list_74497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)

# Obtaining an instance of the builtin type 'list' (line 15)
list_74498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)

# Obtaining an instance of the builtin type 'list' (line 15)
list_74499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 6), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
int_74500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 7), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 6), list_74499, int_74500)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 5), list_74498, list_74499)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 4), list_74497, list_74498)
# Adding element type (line 15)
str_74501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 12), 'str', 'right')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 4), list_74497, str_74501)
# Adding element type (line 15)

# Obtaining an instance of the builtin type 'list' (line 15)
list_74502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)

# Obtaining an instance of the builtin type 'list' (line 15)
list_74503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
int_74504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 22), list_74503, int_74504)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 21), list_74502, list_74503)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 4), list_74497, list_74502)
# Adding element type (line 15)

# Obtaining an instance of the builtin type 'list' (line 15)
list_74505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)

# Obtaining an instance of the builtin type 'list' (line 15)
list_74506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 29), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
int_74507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 29), list_74506, int_74507)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 28), list_74505, list_74506)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 4), list_74497, list_74505)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 20), list_74474, list_74497)
# Adding element type (line 12)

# Obtaining an instance of the builtin type 'list' (line 16)
list_74508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)

# Obtaining an instance of the builtin type 'list' (line 16)
list_74509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)

# Obtaining an instance of the builtin type 'list' (line 16)
list_74510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 6), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
int_74511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 7), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 6), list_74510, int_74511)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 5), list_74509, list_74510)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 4), list_74508, list_74509)
# Adding element type (line 16)
str_74512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 12), 'str', 'left')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 4), list_74508, str_74512)
# Adding element type (line 16)

# Obtaining an instance of the builtin type 'list' (line 16)
list_74513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)

# Obtaining an instance of the builtin type 'list' (line 16)
list_74514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
int_74515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 21), list_74514, int_74515)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 20), list_74513, list_74514)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 4), list_74508, list_74513)
# Adding element type (line 16)

# Obtaining an instance of the builtin type 'list' (line 16)
list_74516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)

# Obtaining an instance of the builtin type 'list' (line 16)
list_74517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
int_74518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 28), list_74517, int_74518)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 27), list_74516, list_74517)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 4), list_74508, list_74516)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 20), list_74474, list_74508)
# Adding element type (line 12)

# Obtaining an instance of the builtin type 'list' (line 17)
list_74519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
# Getting the type of 'diag2' (line 17)
diag2_74520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'diag2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 4), list_74519, diag2_74520)
# Adding element type (line 17)
str_74521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 12), 'str', 'right')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 4), list_74519, str_74521)
# Adding element type (line 17)

# Call to eye(...): (line 17)
# Processing the call arguments (line 17)
int_74524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 28), 'int')
# Processing the call keyword arguments (line 17)
kwargs_74525 = {}
# Getting the type of 'np' (line 17)
np_74522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 21), 'np', False)
# Obtaining the member 'eye' of a type (line 17)
eye_74523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 21), np_74522, 'eye')
# Calling eye(args, kwargs) (line 17)
eye_call_result_74526 = invoke(stypy.reporting.localization.Localization(__file__, 17, 21), eye_74523, *[int_74524], **kwargs_74525)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 4), list_74519, eye_call_result_74526)
# Adding element type (line 17)
# Getting the type of 'diag2' (line 17)
diag2_74527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 32), 'diag2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 4), list_74519, diag2_74527)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 20), list_74474, list_74519)
# Adding element type (line 12)

# Obtaining an instance of the builtin type 'list' (line 18)
list_74528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)
# Getting the type of 'diag2' (line 18)
diag2_74529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 5), 'diag2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 4), list_74528, diag2_74529)
# Adding element type (line 18)
str_74530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 12), 'str', 'left')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 4), list_74528, str_74530)
# Adding element type (line 18)

# Call to eye(...): (line 18)
# Processing the call arguments (line 18)
int_74533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 27), 'int')
# Processing the call keyword arguments (line 18)
kwargs_74534 = {}
# Getting the type of 'np' (line 18)
np_74531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 20), 'np', False)
# Obtaining the member 'eye' of a type (line 18)
eye_74532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 20), np_74531, 'eye')
# Calling eye(args, kwargs) (line 18)
eye_call_result_74535 = invoke(stypy.reporting.localization.Localization(__file__, 18, 20), eye_74532, *[int_74533], **kwargs_74534)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 4), list_74528, eye_call_result_74535)
# Adding element type (line 18)
# Getting the type of 'diag2' (line 18)
diag2_74536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 31), 'diag2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 4), list_74528, diag2_74536)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 20), list_74474, list_74528)
# Adding element type (line 12)

# Obtaining an instance of the builtin type 'list' (line 19)
list_74537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)
# Getting the type of 'a13' (line 19)
a13_74538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'a13')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 4), list_74537, a13_74538)
# Adding element type (line 19)
str_74539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 10), 'str', 'right')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 4), list_74537, str_74539)
# Adding element type (line 19)
# Getting the type of 'a13' (line 19)
a13_74540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 19), 'a13')

# Call to norm(...): (line 19)
# Processing the call arguments (line 19)

# Obtaining the type of the subscript
int_74542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 32), 'int')
# Getting the type of 'a13' (line 19)
a13_74543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 28), 'a13', False)
# Obtaining the member '__getitem__' of a type (line 19)
getitem___74544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 28), a13_74543, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 19)
subscript_call_result_74545 = invoke(stypy.reporting.localization.Localization(__file__, 19, 28), getitem___74544, int_74542)

# Processing the call keyword arguments (line 19)
kwargs_74546 = {}
# Getting the type of 'norm' (line 19)
norm_74541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 23), 'norm', False)
# Calling norm(args, kwargs) (line 19)
norm_call_result_74547 = invoke(stypy.reporting.localization.Localization(__file__, 19, 23), norm_74541, *[subscript_call_result_74545], **kwargs_74546)

# Applying the binary operator 'div' (line 19)
result_div_74548 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 19), 'div', a13_74540, norm_call_result_74547)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 4), list_74537, result_div_74548)
# Adding element type (line 19)

# Call to dot(...): (line 19)
# Processing the call arguments (line 19)
# Getting the type of 'a13' (line 19)
a13_74552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 47), 'a13', False)
# Processing the call keyword arguments (line 19)
kwargs_74553 = {}
# Getting the type of 'a13' (line 19)
a13_74549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 37), 'a13', False)
# Obtaining the member 'T' of a type (line 19)
T_74550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 37), a13_74549, 'T')
# Obtaining the member 'dot' of a type (line 19)
dot_74551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 37), T_74550, 'dot')
# Calling dot(args, kwargs) (line 19)
dot_call_result_74554 = invoke(stypy.reporting.localization.Localization(__file__, 19, 37), dot_74551, *[a13_74552], **kwargs_74553)


# Call to norm(...): (line 19)
# Processing the call arguments (line 19)

# Obtaining the type of the subscript
int_74556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 61), 'int')
# Getting the type of 'a13' (line 19)
a13_74557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 57), 'a13', False)
# Obtaining the member '__getitem__' of a type (line 19)
getitem___74558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 57), a13_74557, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 19)
subscript_call_result_74559 = invoke(stypy.reporting.localization.Localization(__file__, 19, 57), getitem___74558, int_74556)

# Processing the call keyword arguments (line 19)
kwargs_74560 = {}
# Getting the type of 'norm' (line 19)
norm_74555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 52), 'norm', False)
# Calling norm(args, kwargs) (line 19)
norm_call_result_74561 = invoke(stypy.reporting.localization.Localization(__file__, 19, 52), norm_74555, *[subscript_call_result_74559], **kwargs_74560)

# Applying the binary operator 'div' (line 19)
result_div_74562 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 37), 'div', dot_call_result_74554, norm_call_result_74561)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 4), list_74537, result_div_74562)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 20), list_74474, list_74537)

# Assigning a type to the variable 'precomputed_cases' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'precomputed_cases', list_74474)

# Assigning a List to a Name (line 22):

# Assigning a List to a Name (line 22):

# Obtaining an instance of the builtin type 'list' (line 22)
list_74563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 22)
# Adding element type (line 22)

# Obtaining an instance of the builtin type 'list' (line 23)
list_74564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)
# Adding element type (line 23)

# Obtaining an instance of the builtin type 'list' (line 23)
list_74565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)
# Adding element type (line 23)
int_74566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 5), list_74565, int_74566)
# Adding element type (line 23)
int_74567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 5), list_74565, int_74567)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 4), list_74564, list_74565)
# Adding element type (line 23)

# Obtaining an instance of the builtin type 'list' (line 23)
list_74568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)
# Adding element type (line 23)
int_74569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 13), list_74568, int_74569)
# Adding element type (line 23)
int_74570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 13), list_74568, int_74570)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 4), list_74564, list_74568)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), list_74563, list_74564)
# Adding element type (line 22)

# Obtaining an instance of the builtin type 'list' (line 24)
list_74571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 24)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'list' (line 24)
list_74572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 24)
# Adding element type (line 24)
int_74573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 5), list_74572, int_74573)
# Adding element type (line 24)
int_74574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 5), list_74572, int_74574)
# Adding element type (line 24)
int_74575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 12), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 5), list_74572, int_74575)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 4), list_74571, list_74572)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), list_74563, list_74571)
# Adding element type (line 22)

# Obtaining an instance of the builtin type 'list' (line 25)
list_74576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 25)
# Adding element type (line 25)

# Obtaining an instance of the builtin type 'list' (line 25)
list_74577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 25)
# Adding element type (line 25)
int_74578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 5), list_74577, int_74578)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 4), list_74576, list_74577)
# Adding element type (line 25)

# Obtaining an instance of the builtin type 'list' (line 25)
list_74579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 25)
# Adding element type (line 25)
int_74580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 10), list_74579, int_74580)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 4), list_74576, list_74579)
# Adding element type (line 25)

# Obtaining an instance of the builtin type 'list' (line 25)
list_74581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 25)
# Adding element type (line 25)
int_74582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_74581, int_74582)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 4), list_74576, list_74581)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), list_74563, list_74576)
# Adding element type (line 22)

# Obtaining an instance of the builtin type 'list' (line 26)
list_74583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)
# Adding element type (line 26)

# Obtaining an instance of the builtin type 'list' (line 26)
list_74584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)
# Adding element type (line 26)
int_74585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 5), list_74584, int_74585)
# Adding element type (line 26)
int_74586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 5), list_74584, int_74586)
# Adding element type (line 26)
int_74587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 12), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 5), list_74584, int_74587)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 4), list_74583, list_74584)
# Adding element type (line 26)

# Obtaining an instance of the builtin type 'list' (line 26)
list_74588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)
# Adding element type (line 26)
int_74589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 16), list_74588, int_74589)
# Adding element type (line 26)
int_74590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 16), list_74588, int_74590)
# Adding element type (line 26)
int_74591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 16), list_74588, int_74591)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 4), list_74583, list_74588)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), list_74563, list_74583)
# Adding element type (line 22)

# Obtaining an instance of the builtin type 'list' (line 27)
list_74592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 27)
# Adding element type (line 27)

# Obtaining an instance of the builtin type 'list' (line 27)
list_74593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 27)
# Adding element type (line 27)
int_74594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 5), list_74593, int_74594)
# Adding element type (line 27)
int_74595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 5), list_74593, int_74595)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 4), list_74592, list_74593)
# Adding element type (line 27)

# Obtaining an instance of the builtin type 'list' (line 27)
list_74596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 27)
# Adding element type (line 27)
int_74597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 13), list_74596, int_74597)
# Adding element type (line 27)
int_74598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 13), list_74596, int_74598)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 4), list_74592, list_74596)
# Adding element type (line 27)

# Obtaining an instance of the builtin type 'list' (line 27)
list_74599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 27)
# Adding element type (line 27)
int_74600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 21), list_74599, int_74600)
# Adding element type (line 27)
int_74601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 21), list_74599, int_74601)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 4), list_74592, list_74599)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), list_74563, list_74592)
# Adding element type (line 22)

# Obtaining an instance of the builtin type 'list' (line 28)
list_74602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 28)
# Adding element type (line 28)

# Obtaining an instance of the builtin type 'list' (line 28)
list_74603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 28)
# Adding element type (line 28)
int_74604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 5), list_74603, int_74604)
# Adding element type (line 28)
int_74605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 5), list_74603, int_74605)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 4), list_74602, list_74603)
# Adding element type (line 28)

# Obtaining an instance of the builtin type 'list' (line 28)
list_74606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 28)
# Adding element type (line 28)
int_74607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 13), list_74606, int_74607)
# Adding element type (line 28)
int_74608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 17), 'int')
complex_74609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 19), 'complex')
# Applying the binary operator '+' (line 28)
result_add_74610 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 17), '+', int_74608, complex_74609)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 13), list_74606, result_add_74610)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 4), list_74602, list_74606)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), list_74563, list_74602)
# Adding element type (line 22)

# Obtaining an instance of the builtin type 'list' (line 29)
list_74611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 29)
# Adding element type (line 29)

# Obtaining an instance of the builtin type 'list' (line 29)
list_74612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 29)
# Adding element type (line 29)
int_74613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 5), list_74612, int_74613)
# Adding element type (line 29)
int_74614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 5), list_74612, int_74614)
# Adding element type (line 29)
complex_74615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 12), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 5), list_74612, complex_74615)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 4), list_74611, list_74612)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), list_74563, list_74611)
# Adding element type (line 22)

# Obtaining an instance of the builtin type 'list' (line 30)
list_74616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 30)
# Adding element type (line 30)

# Obtaining an instance of the builtin type 'list' (line 30)
list_74617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 30)
# Adding element type (line 30)
int_74618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 5), list_74617, int_74618)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 4), list_74616, list_74617)
# Adding element type (line 30)

# Obtaining an instance of the builtin type 'list' (line 30)
list_74619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 30)
# Adding element type (line 30)
int_74620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 10), list_74619, int_74620)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 4), list_74616, list_74619)
# Adding element type (line 30)

# Obtaining an instance of the builtin type 'list' (line 30)
list_74621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 30)
# Adding element type (line 30)
complex_74622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 16), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 15), list_74621, complex_74622)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 4), list_74616, list_74621)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), list_74563, list_74616)
# Adding element type (line 22)

# Obtaining an instance of the builtin type 'list' (line 31)
list_74623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 31)
# Adding element type (line 31)

# Obtaining an instance of the builtin type 'list' (line 31)
list_74624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 31)
# Adding element type (line 31)
int_74625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 5), list_74624, int_74625)
# Adding element type (line 31)
int_74626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 5), list_74624, int_74626)
# Adding element type (line 31)
int_74627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 12), 'int')
complex_74628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 14), 'complex')
# Applying the binary operator '+' (line 31)
result_add_74629 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 12), '+', int_74627, complex_74628)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 5), list_74624, result_add_74629)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 4), list_74623, list_74624)
# Adding element type (line 31)

# Obtaining an instance of the builtin type 'list' (line 31)
list_74630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 31)
# Adding element type (line 31)
int_74631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), list_74630, int_74631)
# Adding element type (line 31)
int_74632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 23), 'int')
complex_74633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 25), 'complex')
# Applying the binary operator '-' (line 31)
result_sub_74634 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 23), '-', int_74632, complex_74633)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), list_74630, result_sub_74634)
# Adding element type (line 31)
complex_74635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 29), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), list_74630, complex_74635)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 4), list_74623, list_74630)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), list_74563, list_74623)
# Adding element type (line 22)

# Obtaining an instance of the builtin type 'list' (line 32)
list_74636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 32)
# Adding element type (line 32)

# Obtaining an instance of the builtin type 'list' (line 32)
list_74637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 32)
# Adding element type (line 32)
int_74638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 5), list_74637, int_74638)
# Adding element type (line 32)
int_74639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 5), list_74637, int_74639)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 4), list_74636, list_74637)
# Adding element type (line 32)

# Obtaining an instance of the builtin type 'list' (line 32)
list_74640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 32)
# Adding element type (line 32)
int_74641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 14), 'int')
complex_74642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 16), 'complex')
# Applying the binary operator '-' (line 32)
result_sub_74643 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 14), '-', int_74641, complex_74642)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 13), list_74640, result_sub_74643)
# Adding element type (line 32)
int_74644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 20), 'int')
complex_74645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 22), 'complex')
# Applying the binary operator '+' (line 32)
result_add_74646 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 20), '+', int_74644, complex_74645)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 13), list_74640, result_add_74646)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 4), list_74636, list_74640)
# Adding element type (line 32)

# Obtaining an instance of the builtin type 'list' (line 32)
list_74647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 29), 'list')
# Adding type elements to the builtin type 'list' instance (line 32)
# Adding element type (line 32)
int_74648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 29), list_74647, int_74648)
# Adding element type (line 32)
int_74649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 29), list_74647, int_74649)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 4), list_74636, list_74647)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), list_74563, list_74636)
# Adding element type (line 22)

# Obtaining an instance of the builtin type 'list' (line 33)
list_74650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 33)
# Adding element type (line 33)

# Obtaining an instance of the builtin type 'list' (line 33)
list_74651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 33)
# Adding element type (line 33)
int_74652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 5), list_74651, int_74652)
# Adding element type (line 33)
int_74653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 5), list_74651, int_74653)
# Adding element type (line 33)
int_74654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 5), list_74651, int_74654)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 4), list_74650, list_74651)
# Adding element type (line 33)

# Obtaining an instance of the builtin type 'list' (line 33)
list_74655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 33)
# Adding element type (line 33)
int_74656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), list_74655, int_74656)
# Adding element type (line 33)
int_74657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), list_74655, int_74657)
# Adding element type (line 33)
complex_74658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 29), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), list_74655, complex_74658)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 4), list_74650, list_74655)
# Adding element type (line 33)

# Obtaining an instance of the builtin type 'list' (line 33)
list_74659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 34), 'list')
# Adding type elements to the builtin type 'list' instance (line 33)
# Adding element type (line 33)
int_74660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 34), list_74659, int_74660)
# Adding element type (line 33)
int_74661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 34), list_74659, int_74661)
# Adding element type (line 33)
int_74662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 41), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 34), list_74659, int_74662)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 4), list_74650, list_74659)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), list_74563, list_74650)

# Assigning a type to the variable 'verify_cases' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'verify_cases', list_74563)

@norecursion
def check_precomputed_polar(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_precomputed_polar'
    module_type_store = module_type_store.open_function_context('check_precomputed_polar', 37, 0, False)
    
    # Passed parameters checking function
    check_precomputed_polar.stypy_localization = localization
    check_precomputed_polar.stypy_type_of_self = None
    check_precomputed_polar.stypy_type_store = module_type_store
    check_precomputed_polar.stypy_function_name = 'check_precomputed_polar'
    check_precomputed_polar.stypy_param_names_list = ['a', 'side', 'expected_u', 'expected_p']
    check_precomputed_polar.stypy_varargs_param_name = None
    check_precomputed_polar.stypy_kwargs_param_name = None
    check_precomputed_polar.stypy_call_defaults = defaults
    check_precomputed_polar.stypy_call_varargs = varargs
    check_precomputed_polar.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_precomputed_polar', ['a', 'side', 'expected_u', 'expected_p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_precomputed_polar', localization, ['a', 'side', 'expected_u', 'expected_p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_precomputed_polar(...)' code ##################

    
    # Assigning a Call to a Tuple (line 40):
    
    # Assigning a Subscript to a Name (line 40):
    
    # Obtaining the type of the subscript
    int_74663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 4), 'int')
    
    # Call to polar(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'a' (line 40)
    a_74665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 17), 'a', False)
    # Processing the call keyword arguments (line 40)
    # Getting the type of 'side' (line 40)
    side_74666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 25), 'side', False)
    keyword_74667 = side_74666
    kwargs_74668 = {'side': keyword_74667}
    # Getting the type of 'polar' (line 40)
    polar_74664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'polar', False)
    # Calling polar(args, kwargs) (line 40)
    polar_call_result_74669 = invoke(stypy.reporting.localization.Localization(__file__, 40, 11), polar_74664, *[a_74665], **kwargs_74668)
    
    # Obtaining the member '__getitem__' of a type (line 40)
    getitem___74670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 4), polar_call_result_74669, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
    subscript_call_result_74671 = invoke(stypy.reporting.localization.Localization(__file__, 40, 4), getitem___74670, int_74663)
    
    # Assigning a type to the variable 'tuple_var_assignment_74438' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'tuple_var_assignment_74438', subscript_call_result_74671)
    
    # Assigning a Subscript to a Name (line 40):
    
    # Obtaining the type of the subscript
    int_74672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 4), 'int')
    
    # Call to polar(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'a' (line 40)
    a_74674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 17), 'a', False)
    # Processing the call keyword arguments (line 40)
    # Getting the type of 'side' (line 40)
    side_74675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 25), 'side', False)
    keyword_74676 = side_74675
    kwargs_74677 = {'side': keyword_74676}
    # Getting the type of 'polar' (line 40)
    polar_74673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'polar', False)
    # Calling polar(args, kwargs) (line 40)
    polar_call_result_74678 = invoke(stypy.reporting.localization.Localization(__file__, 40, 11), polar_74673, *[a_74674], **kwargs_74677)
    
    # Obtaining the member '__getitem__' of a type (line 40)
    getitem___74679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 4), polar_call_result_74678, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
    subscript_call_result_74680 = invoke(stypy.reporting.localization.Localization(__file__, 40, 4), getitem___74679, int_74672)
    
    # Assigning a type to the variable 'tuple_var_assignment_74439' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'tuple_var_assignment_74439', subscript_call_result_74680)
    
    # Assigning a Name to a Name (line 40):
    # Getting the type of 'tuple_var_assignment_74438' (line 40)
    tuple_var_assignment_74438_74681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'tuple_var_assignment_74438')
    # Assigning a type to the variable 'u' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'u', tuple_var_assignment_74438_74681)
    
    # Assigning a Name to a Name (line 40):
    # Getting the type of 'tuple_var_assignment_74439' (line 40)
    tuple_var_assignment_74439_74682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'tuple_var_assignment_74439')
    # Assigning a type to the variable 'p' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 7), 'p', tuple_var_assignment_74439_74682)
    
    # Call to assert_allclose(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'u' (line 41)
    u_74684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'u', False)
    # Getting the type of 'expected_u' (line 41)
    expected_u_74685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 23), 'expected_u', False)
    # Processing the call keyword arguments (line 41)
    float_74686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 40), 'float')
    keyword_74687 = float_74686
    kwargs_74688 = {'atol': keyword_74687}
    # Getting the type of 'assert_allclose' (line 41)
    assert_allclose_74683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 41)
    assert_allclose_call_result_74689 = invoke(stypy.reporting.localization.Localization(__file__, 41, 4), assert_allclose_74683, *[u_74684, expected_u_74685], **kwargs_74688)
    
    
    # Call to assert_allclose(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'p' (line 42)
    p_74691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'p', False)
    # Getting the type of 'expected_p' (line 42)
    expected_p_74692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 23), 'expected_p', False)
    # Processing the call keyword arguments (line 42)
    float_74693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 40), 'float')
    keyword_74694 = float_74693
    kwargs_74695 = {'atol': keyword_74694}
    # Getting the type of 'assert_allclose' (line 42)
    assert_allclose_74690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 42)
    assert_allclose_call_result_74696 = invoke(stypy.reporting.localization.Localization(__file__, 42, 4), assert_allclose_74690, *[p_74691, expected_p_74692], **kwargs_74695)
    
    
    # ################# End of 'check_precomputed_polar(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_precomputed_polar' in the type store
    # Getting the type of 'stypy_return_type' (line 37)
    stypy_return_type_74697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_74697)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_precomputed_polar'
    return stypy_return_type_74697

# Assigning a type to the variable 'check_precomputed_polar' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'check_precomputed_polar', check_precomputed_polar)

@norecursion
def verify_polar(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'verify_polar'
    module_type_store = module_type_store.open_function_context('verify_polar', 45, 0, False)
    
    # Passed parameters checking function
    verify_polar.stypy_localization = localization
    verify_polar.stypy_type_of_self = None
    verify_polar.stypy_type_store = module_type_store
    verify_polar.stypy_function_name = 'verify_polar'
    verify_polar.stypy_param_names_list = ['a']
    verify_polar.stypy_varargs_param_name = None
    verify_polar.stypy_kwargs_param_name = None
    verify_polar.stypy_call_defaults = defaults
    verify_polar.stypy_call_varargs = varargs
    verify_polar.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'verify_polar', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'verify_polar', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'verify_polar(...)' code ##################

    
    # Assigning a Call to a Name (line 48):
    
    # Assigning a Call to a Name (line 48):
    
    # Call to sqrt(...): (line 48)
    # Processing the call arguments (line 48)
    
    # Call to finfo(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'float' (line 48)
    float_74702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 36), 'float', False)
    # Processing the call keyword arguments (line 48)
    kwargs_74703 = {}
    # Getting the type of 'np' (line 48)
    np_74700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 27), 'np', False)
    # Obtaining the member 'finfo' of a type (line 48)
    finfo_74701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 27), np_74700, 'finfo')
    # Calling finfo(args, kwargs) (line 48)
    finfo_call_result_74704 = invoke(stypy.reporting.localization.Localization(__file__, 48, 27), finfo_74701, *[float_74702], **kwargs_74703)
    
    # Obtaining the member 'eps' of a type (line 48)
    eps_74705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 27), finfo_call_result_74704, 'eps')
    # Processing the call keyword arguments (line 48)
    kwargs_74706 = {}
    # Getting the type of 'np' (line 48)
    np_74698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 48)
    sqrt_74699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 19), np_74698, 'sqrt')
    # Calling sqrt(args, kwargs) (line 48)
    sqrt_call_result_74707 = invoke(stypy.reporting.localization.Localization(__file__, 48, 19), sqrt_74699, *[eps_74705], **kwargs_74706)
    
    # Assigning a type to the variable 'product_atol' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'product_atol', sqrt_call_result_74707)
    
    # Assigning a Call to a Name (line 50):
    
    # Assigning a Call to a Name (line 50):
    
    # Call to asarray(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'a' (line 50)
    a_74710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 20), 'a', False)
    # Processing the call keyword arguments (line 50)
    kwargs_74711 = {}
    # Getting the type of 'np' (line 50)
    np_74708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 9), 'np', False)
    # Obtaining the member 'asarray' of a type (line 50)
    asarray_74709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 9), np_74708, 'asarray')
    # Calling asarray(args, kwargs) (line 50)
    asarray_call_result_74712 = invoke(stypy.reporting.localization.Localization(__file__, 50, 9), asarray_74709, *[a_74710], **kwargs_74711)
    
    # Assigning a type to the variable 'aa' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'aa', asarray_call_result_74712)
    
    # Assigning a Attribute to a Tuple (line 51):
    
    # Assigning a Subscript to a Name (line 51):
    
    # Obtaining the type of the subscript
    int_74713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 4), 'int')
    # Getting the type of 'aa' (line 51)
    aa_74714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 11), 'aa')
    # Obtaining the member 'shape' of a type (line 51)
    shape_74715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 11), aa_74714, 'shape')
    # Obtaining the member '__getitem__' of a type (line 51)
    getitem___74716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 4), shape_74715, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 51)
    subscript_call_result_74717 = invoke(stypy.reporting.localization.Localization(__file__, 51, 4), getitem___74716, int_74713)
    
    # Assigning a type to the variable 'tuple_var_assignment_74440' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'tuple_var_assignment_74440', subscript_call_result_74717)
    
    # Assigning a Subscript to a Name (line 51):
    
    # Obtaining the type of the subscript
    int_74718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 4), 'int')
    # Getting the type of 'aa' (line 51)
    aa_74719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 11), 'aa')
    # Obtaining the member 'shape' of a type (line 51)
    shape_74720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 11), aa_74719, 'shape')
    # Obtaining the member '__getitem__' of a type (line 51)
    getitem___74721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 4), shape_74720, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 51)
    subscript_call_result_74722 = invoke(stypy.reporting.localization.Localization(__file__, 51, 4), getitem___74721, int_74718)
    
    # Assigning a type to the variable 'tuple_var_assignment_74441' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'tuple_var_assignment_74441', subscript_call_result_74722)
    
    # Assigning a Name to a Name (line 51):
    # Getting the type of 'tuple_var_assignment_74440' (line 51)
    tuple_var_assignment_74440_74723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'tuple_var_assignment_74440')
    # Assigning a type to the variable 'm' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'm', tuple_var_assignment_74440_74723)
    
    # Assigning a Name to a Name (line 51):
    # Getting the type of 'tuple_var_assignment_74441' (line 51)
    tuple_var_assignment_74441_74724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'tuple_var_assignment_74441')
    # Assigning a type to the variable 'n' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 7), 'n', tuple_var_assignment_74441_74724)
    
    # Assigning a Call to a Tuple (line 53):
    
    # Assigning a Subscript to a Name (line 53):
    
    # Obtaining the type of the subscript
    int_74725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 4), 'int')
    
    # Call to polar(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'a' (line 53)
    a_74727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'a', False)
    # Processing the call keyword arguments (line 53)
    str_74728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 25), 'str', 'right')
    keyword_74729 = str_74728
    kwargs_74730 = {'side': keyword_74729}
    # Getting the type of 'polar' (line 53)
    polar_74726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 11), 'polar', False)
    # Calling polar(args, kwargs) (line 53)
    polar_call_result_74731 = invoke(stypy.reporting.localization.Localization(__file__, 53, 11), polar_74726, *[a_74727], **kwargs_74730)
    
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___74732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 4), polar_call_result_74731, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_74733 = invoke(stypy.reporting.localization.Localization(__file__, 53, 4), getitem___74732, int_74725)
    
    # Assigning a type to the variable 'tuple_var_assignment_74442' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'tuple_var_assignment_74442', subscript_call_result_74733)
    
    # Assigning a Subscript to a Name (line 53):
    
    # Obtaining the type of the subscript
    int_74734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 4), 'int')
    
    # Call to polar(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'a' (line 53)
    a_74736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'a', False)
    # Processing the call keyword arguments (line 53)
    str_74737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 25), 'str', 'right')
    keyword_74738 = str_74737
    kwargs_74739 = {'side': keyword_74738}
    # Getting the type of 'polar' (line 53)
    polar_74735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 11), 'polar', False)
    # Calling polar(args, kwargs) (line 53)
    polar_call_result_74740 = invoke(stypy.reporting.localization.Localization(__file__, 53, 11), polar_74735, *[a_74736], **kwargs_74739)
    
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___74741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 4), polar_call_result_74740, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_74742 = invoke(stypy.reporting.localization.Localization(__file__, 53, 4), getitem___74741, int_74734)
    
    # Assigning a type to the variable 'tuple_var_assignment_74443' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'tuple_var_assignment_74443', subscript_call_result_74742)
    
    # Assigning a Name to a Name (line 53):
    # Getting the type of 'tuple_var_assignment_74442' (line 53)
    tuple_var_assignment_74442_74743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'tuple_var_assignment_74442')
    # Assigning a type to the variable 'u' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'u', tuple_var_assignment_74442_74743)
    
    # Assigning a Name to a Name (line 53):
    # Getting the type of 'tuple_var_assignment_74443' (line 53)
    tuple_var_assignment_74443_74744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'tuple_var_assignment_74443')
    # Assigning a type to the variable 'p' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 7), 'p', tuple_var_assignment_74443_74744)
    
    # Call to assert_equal(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'u' (line 54)
    u_74746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 17), 'u', False)
    # Obtaining the member 'shape' of a type (line 54)
    shape_74747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 17), u_74746, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 54)
    tuple_74748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 54)
    # Adding element type (line 54)
    # Getting the type of 'm' (line 54)
    m_74749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 27), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 27), tuple_74748, m_74749)
    # Adding element type (line 54)
    # Getting the type of 'n' (line 54)
    n_74750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 30), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 27), tuple_74748, n_74750)
    
    # Processing the call keyword arguments (line 54)
    kwargs_74751 = {}
    # Getting the type of 'assert_equal' (line 54)
    assert_equal_74745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 54)
    assert_equal_call_result_74752 = invoke(stypy.reporting.localization.Localization(__file__, 54, 4), assert_equal_74745, *[shape_74747, tuple_74748], **kwargs_74751)
    
    
    # Call to assert_equal(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'p' (line 55)
    p_74754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 17), 'p', False)
    # Obtaining the member 'shape' of a type (line 55)
    shape_74755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 17), p_74754, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 55)
    tuple_74756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 55)
    # Adding element type (line 55)
    # Getting the type of 'n' (line 55)
    n_74757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 27), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 27), tuple_74756, n_74757)
    # Adding element type (line 55)
    # Getting the type of 'n' (line 55)
    n_74758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 30), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 27), tuple_74756, n_74758)
    
    # Processing the call keyword arguments (line 55)
    kwargs_74759 = {}
    # Getting the type of 'assert_equal' (line 55)
    assert_equal_74753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 55)
    assert_equal_call_result_74760 = invoke(stypy.reporting.localization.Localization(__file__, 55, 4), assert_equal_74753, *[shape_74755, tuple_74756], **kwargs_74759)
    
    
    # Call to assert_allclose(...): (line 57)
    # Processing the call arguments (line 57)
    
    # Call to dot(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'p' (line 57)
    p_74764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 26), 'p', False)
    # Processing the call keyword arguments (line 57)
    kwargs_74765 = {}
    # Getting the type of 'u' (line 57)
    u_74762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 20), 'u', False)
    # Obtaining the member 'dot' of a type (line 57)
    dot_74763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 20), u_74762, 'dot')
    # Calling dot(args, kwargs) (line 57)
    dot_call_result_74766 = invoke(stypy.reporting.localization.Localization(__file__, 57, 20), dot_74763, *[p_74764], **kwargs_74765)
    
    # Getting the type of 'a' (line 57)
    a_74767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 30), 'a', False)
    # Processing the call keyword arguments (line 57)
    # Getting the type of 'product_atol' (line 57)
    product_atol_74768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 38), 'product_atol', False)
    keyword_74769 = product_atol_74768
    kwargs_74770 = {'atol': keyword_74769}
    # Getting the type of 'assert_allclose' (line 57)
    assert_allclose_74761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 57)
    assert_allclose_call_result_74771 = invoke(stypy.reporting.localization.Localization(__file__, 57, 4), assert_allclose_74761, *[dot_call_result_74766, a_74767], **kwargs_74770)
    
    
    
    # Getting the type of 'm' (line 58)
    m_74772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 7), 'm')
    # Getting the type of 'n' (line 58)
    n_74773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'n')
    # Applying the binary operator '>=' (line 58)
    result_ge_74774 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 7), '>=', m_74772, n_74773)
    
    # Testing the type of an if condition (line 58)
    if_condition_74775 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 4), result_ge_74774)
    # Assigning a type to the variable 'if_condition_74775' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'if_condition_74775', if_condition_74775)
    # SSA begins for if statement (line 58)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to assert_allclose(...): (line 59)
    # Processing the call arguments (line 59)
    
    # Call to dot(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'u' (line 59)
    u_74783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 39), 'u', False)
    # Processing the call keyword arguments (line 59)
    kwargs_74784 = {}
    
    # Call to conj(...): (line 59)
    # Processing the call keyword arguments (line 59)
    kwargs_74779 = {}
    # Getting the type of 'u' (line 59)
    u_74777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 24), 'u', False)
    # Obtaining the member 'conj' of a type (line 59)
    conj_74778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 24), u_74777, 'conj')
    # Calling conj(args, kwargs) (line 59)
    conj_call_result_74780 = invoke(stypy.reporting.localization.Localization(__file__, 59, 24), conj_74778, *[], **kwargs_74779)
    
    # Obtaining the member 'T' of a type (line 59)
    T_74781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 24), conj_call_result_74780, 'T')
    # Obtaining the member 'dot' of a type (line 59)
    dot_74782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 24), T_74781, 'dot')
    # Calling dot(args, kwargs) (line 59)
    dot_call_result_74785 = invoke(stypy.reporting.localization.Localization(__file__, 59, 24), dot_74782, *[u_74783], **kwargs_74784)
    
    
    # Call to eye(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'n' (line 59)
    n_74788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 50), 'n', False)
    # Processing the call keyword arguments (line 59)
    kwargs_74789 = {}
    # Getting the type of 'np' (line 59)
    np_74786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 43), 'np', False)
    # Obtaining the member 'eye' of a type (line 59)
    eye_74787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 43), np_74786, 'eye')
    # Calling eye(args, kwargs) (line 59)
    eye_call_result_74790 = invoke(stypy.reporting.localization.Localization(__file__, 59, 43), eye_74787, *[n_74788], **kwargs_74789)
    
    # Processing the call keyword arguments (line 59)
    float_74791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 59), 'float')
    keyword_74792 = float_74791
    kwargs_74793 = {'atol': keyword_74792}
    # Getting the type of 'assert_allclose' (line 59)
    assert_allclose_74776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 59)
    assert_allclose_call_result_74794 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), assert_allclose_74776, *[dot_call_result_74785, eye_call_result_74790], **kwargs_74793)
    
    # SSA branch for the else part of an if statement (line 58)
    module_type_store.open_ssa_branch('else')
    
    # Call to assert_allclose(...): (line 61)
    # Processing the call arguments (line 61)
    
    # Call to dot(...): (line 61)
    # Processing the call arguments (line 61)
    
    # Call to conj(...): (line 61)
    # Processing the call keyword arguments (line 61)
    kwargs_74800 = {}
    # Getting the type of 'u' (line 61)
    u_74798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 30), 'u', False)
    # Obtaining the member 'conj' of a type (line 61)
    conj_74799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 30), u_74798, 'conj')
    # Calling conj(args, kwargs) (line 61)
    conj_call_result_74801 = invoke(stypy.reporting.localization.Localization(__file__, 61, 30), conj_74799, *[], **kwargs_74800)
    
    # Obtaining the member 'T' of a type (line 61)
    T_74802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 30), conj_call_result_74801, 'T')
    # Processing the call keyword arguments (line 61)
    kwargs_74803 = {}
    # Getting the type of 'u' (line 61)
    u_74796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'u', False)
    # Obtaining the member 'dot' of a type (line 61)
    dot_74797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 24), u_74796, 'dot')
    # Calling dot(args, kwargs) (line 61)
    dot_call_result_74804 = invoke(stypy.reporting.localization.Localization(__file__, 61, 24), dot_74797, *[T_74802], **kwargs_74803)
    
    
    # Call to eye(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'm' (line 61)
    m_74807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 50), 'm', False)
    # Processing the call keyword arguments (line 61)
    kwargs_74808 = {}
    # Getting the type of 'np' (line 61)
    np_74805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 43), 'np', False)
    # Obtaining the member 'eye' of a type (line 61)
    eye_74806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 43), np_74805, 'eye')
    # Calling eye(args, kwargs) (line 61)
    eye_call_result_74809 = invoke(stypy.reporting.localization.Localization(__file__, 61, 43), eye_74806, *[m_74807], **kwargs_74808)
    
    # Processing the call keyword arguments (line 61)
    float_74810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 59), 'float')
    keyword_74811 = float_74810
    kwargs_74812 = {'atol': keyword_74811}
    # Getting the type of 'assert_allclose' (line 61)
    assert_allclose_74795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 61)
    assert_allclose_call_result_74813 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), assert_allclose_74795, *[dot_call_result_74804, eye_call_result_74809], **kwargs_74812)
    
    # SSA join for if statement (line 58)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to assert_allclose(...): (line 63)
    # Processing the call arguments (line 63)
    
    # Call to conj(...): (line 63)
    # Processing the call keyword arguments (line 63)
    kwargs_74817 = {}
    # Getting the type of 'p' (line 63)
    p_74815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'p', False)
    # Obtaining the member 'conj' of a type (line 63)
    conj_74816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 20), p_74815, 'conj')
    # Calling conj(args, kwargs) (line 63)
    conj_call_result_74818 = invoke(stypy.reporting.localization.Localization(__file__, 63, 20), conj_74816, *[], **kwargs_74817)
    
    # Obtaining the member 'T' of a type (line 63)
    T_74819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 20), conj_call_result_74818, 'T')
    # Getting the type of 'p' (line 63)
    p_74820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 32), 'p', False)
    # Processing the call keyword arguments (line 63)
    kwargs_74821 = {}
    # Getting the type of 'assert_allclose' (line 63)
    assert_allclose_74814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 63)
    assert_allclose_call_result_74822 = invoke(stypy.reporting.localization.Localization(__file__, 63, 4), assert_allclose_74814, *[T_74819, p_74820], **kwargs_74821)
    
    
    # Assigning a Call to a Name (line 64):
    
    # Assigning a Call to a Name (line 64):
    
    # Call to eigh(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'p' (line 64)
    p_74824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 17), 'p', False)
    # Processing the call keyword arguments (line 64)
    # Getting the type of 'True' (line 64)
    True_74825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 33), 'True', False)
    keyword_74826 = True_74825
    kwargs_74827 = {'eigvals_only': keyword_74826}
    # Getting the type of 'eigh' (line 64)
    eigh_74823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'eigh', False)
    # Calling eigh(args, kwargs) (line 64)
    eigh_call_result_74828 = invoke(stypy.reporting.localization.Localization(__file__, 64, 12), eigh_74823, *[p_74824], **kwargs_74827)
    
    # Assigning a type to the variable 'evals' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'evals', eigh_call_result_74828)
    
    # Assigning a Subscript to a Name (line 65):
    
    # Assigning a Subscript to a Name (line 65):
    
    # Obtaining the type of the subscript
    
    
    # Call to abs(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'evals' (line 65)
    evals_74830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 30), 'evals', False)
    # Processing the call keyword arguments (line 65)
    kwargs_74831 = {}
    # Getting the type of 'abs' (line 65)
    abs_74829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 26), 'abs', False)
    # Calling abs(args, kwargs) (line 65)
    abs_call_result_74832 = invoke(stypy.reporting.localization.Localization(__file__, 65, 26), abs_74829, *[evals_74830], **kwargs_74831)
    
    float_74833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 39), 'float')
    # Applying the binary operator '>' (line 65)
    result_gt_74834 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 26), '>', abs_call_result_74832, float_74833)
    
    # Getting the type of 'evals' (line 65)
    evals_74835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'evals')
    # Obtaining the member '__getitem__' of a type (line 65)
    getitem___74836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 20), evals_74835, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 65)
    subscript_call_result_74837 = invoke(stypy.reporting.localization.Localization(__file__, 65, 20), getitem___74836, result_gt_74834)
    
    # Assigning a type to the variable 'nonzero_evals' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'nonzero_evals', subscript_call_result_74837)
    
    # Call to assert_(...): (line 66)
    # Processing the call arguments (line 66)
    
    # Call to all(...): (line 66)
    # Processing the call keyword arguments (line 66)
    kwargs_74843 = {}
    
    # Getting the type of 'nonzero_evals' (line 66)
    nonzero_evals_74839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 13), 'nonzero_evals', False)
    int_74840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 30), 'int')
    # Applying the binary operator '>=' (line 66)
    result_ge_74841 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 13), '>=', nonzero_evals_74839, int_74840)
    
    # Obtaining the member 'all' of a type (line 66)
    all_74842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 13), result_ge_74841, 'all')
    # Calling all(args, kwargs) (line 66)
    all_call_result_74844 = invoke(stypy.reporting.localization.Localization(__file__, 66, 13), all_74842, *[], **kwargs_74843)
    
    # Processing the call keyword arguments (line 66)
    kwargs_74845 = {}
    # Getting the type of 'assert_' (line 66)
    assert__74838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 66)
    assert__call_result_74846 = invoke(stypy.reporting.localization.Localization(__file__, 66, 4), assert__74838, *[all_call_result_74844], **kwargs_74845)
    
    
    # Assigning a Call to a Tuple (line 68):
    
    # Assigning a Subscript to a Name (line 68):
    
    # Obtaining the type of the subscript
    int_74847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 4), 'int')
    
    # Call to polar(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'a' (line 68)
    a_74849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 17), 'a', False)
    # Processing the call keyword arguments (line 68)
    str_74850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 25), 'str', 'left')
    keyword_74851 = str_74850
    kwargs_74852 = {'side': keyword_74851}
    # Getting the type of 'polar' (line 68)
    polar_74848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'polar', False)
    # Calling polar(args, kwargs) (line 68)
    polar_call_result_74853 = invoke(stypy.reporting.localization.Localization(__file__, 68, 11), polar_74848, *[a_74849], **kwargs_74852)
    
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___74854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 4), polar_call_result_74853, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_74855 = invoke(stypy.reporting.localization.Localization(__file__, 68, 4), getitem___74854, int_74847)
    
    # Assigning a type to the variable 'tuple_var_assignment_74444' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'tuple_var_assignment_74444', subscript_call_result_74855)
    
    # Assigning a Subscript to a Name (line 68):
    
    # Obtaining the type of the subscript
    int_74856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 4), 'int')
    
    # Call to polar(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'a' (line 68)
    a_74858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 17), 'a', False)
    # Processing the call keyword arguments (line 68)
    str_74859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 25), 'str', 'left')
    keyword_74860 = str_74859
    kwargs_74861 = {'side': keyword_74860}
    # Getting the type of 'polar' (line 68)
    polar_74857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'polar', False)
    # Calling polar(args, kwargs) (line 68)
    polar_call_result_74862 = invoke(stypy.reporting.localization.Localization(__file__, 68, 11), polar_74857, *[a_74858], **kwargs_74861)
    
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___74863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 4), polar_call_result_74862, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_74864 = invoke(stypy.reporting.localization.Localization(__file__, 68, 4), getitem___74863, int_74856)
    
    # Assigning a type to the variable 'tuple_var_assignment_74445' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'tuple_var_assignment_74445', subscript_call_result_74864)
    
    # Assigning a Name to a Name (line 68):
    # Getting the type of 'tuple_var_assignment_74444' (line 68)
    tuple_var_assignment_74444_74865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'tuple_var_assignment_74444')
    # Assigning a type to the variable 'u' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'u', tuple_var_assignment_74444_74865)
    
    # Assigning a Name to a Name (line 68):
    # Getting the type of 'tuple_var_assignment_74445' (line 68)
    tuple_var_assignment_74445_74866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'tuple_var_assignment_74445')
    # Assigning a type to the variable 'p' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 7), 'p', tuple_var_assignment_74445_74866)
    
    # Call to assert_equal(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'u' (line 69)
    u_74868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 17), 'u', False)
    # Obtaining the member 'shape' of a type (line 69)
    shape_74869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 17), u_74868, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 69)
    tuple_74870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 69)
    # Adding element type (line 69)
    # Getting the type of 'm' (line 69)
    m_74871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 27), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 27), tuple_74870, m_74871)
    # Adding element type (line 69)
    # Getting the type of 'n' (line 69)
    n_74872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 30), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 27), tuple_74870, n_74872)
    
    # Processing the call keyword arguments (line 69)
    kwargs_74873 = {}
    # Getting the type of 'assert_equal' (line 69)
    assert_equal_74867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 69)
    assert_equal_call_result_74874 = invoke(stypy.reporting.localization.Localization(__file__, 69, 4), assert_equal_74867, *[shape_74869, tuple_74870], **kwargs_74873)
    
    
    # Call to assert_equal(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'p' (line 70)
    p_74876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 17), 'p', False)
    # Obtaining the member 'shape' of a type (line 70)
    shape_74877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 17), p_74876, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 70)
    tuple_74878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 70)
    # Adding element type (line 70)
    # Getting the type of 'm' (line 70)
    m_74879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 27), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 27), tuple_74878, m_74879)
    # Adding element type (line 70)
    # Getting the type of 'm' (line 70)
    m_74880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 30), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 27), tuple_74878, m_74880)
    
    # Processing the call keyword arguments (line 70)
    kwargs_74881 = {}
    # Getting the type of 'assert_equal' (line 70)
    assert_equal_74875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 70)
    assert_equal_call_result_74882 = invoke(stypy.reporting.localization.Localization(__file__, 70, 4), assert_equal_74875, *[shape_74877, tuple_74878], **kwargs_74881)
    
    
    # Call to assert_allclose(...): (line 72)
    # Processing the call arguments (line 72)
    
    # Call to dot(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'u' (line 72)
    u_74886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 26), 'u', False)
    # Processing the call keyword arguments (line 72)
    kwargs_74887 = {}
    # Getting the type of 'p' (line 72)
    p_74884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 20), 'p', False)
    # Obtaining the member 'dot' of a type (line 72)
    dot_74885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 20), p_74884, 'dot')
    # Calling dot(args, kwargs) (line 72)
    dot_call_result_74888 = invoke(stypy.reporting.localization.Localization(__file__, 72, 20), dot_74885, *[u_74886], **kwargs_74887)
    
    # Getting the type of 'a' (line 72)
    a_74889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 30), 'a', False)
    # Processing the call keyword arguments (line 72)
    # Getting the type of 'product_atol' (line 72)
    product_atol_74890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 38), 'product_atol', False)
    keyword_74891 = product_atol_74890
    kwargs_74892 = {'atol': keyword_74891}
    # Getting the type of 'assert_allclose' (line 72)
    assert_allclose_74883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 72)
    assert_allclose_call_result_74893 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), assert_allclose_74883, *[dot_call_result_74888, a_74889], **kwargs_74892)
    
    
    
    # Getting the type of 'm' (line 73)
    m_74894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 7), 'm')
    # Getting the type of 'n' (line 73)
    n_74895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'n')
    # Applying the binary operator '>=' (line 73)
    result_ge_74896 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 7), '>=', m_74894, n_74895)
    
    # Testing the type of an if condition (line 73)
    if_condition_74897 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 4), result_ge_74896)
    # Assigning a type to the variable 'if_condition_74897' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'if_condition_74897', if_condition_74897)
    # SSA begins for if statement (line 73)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to assert_allclose(...): (line 74)
    # Processing the call arguments (line 74)
    
    # Call to dot(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'u' (line 74)
    u_74905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 39), 'u', False)
    # Processing the call keyword arguments (line 74)
    kwargs_74906 = {}
    
    # Call to conj(...): (line 74)
    # Processing the call keyword arguments (line 74)
    kwargs_74901 = {}
    # Getting the type of 'u' (line 74)
    u_74899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 24), 'u', False)
    # Obtaining the member 'conj' of a type (line 74)
    conj_74900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 24), u_74899, 'conj')
    # Calling conj(args, kwargs) (line 74)
    conj_call_result_74902 = invoke(stypy.reporting.localization.Localization(__file__, 74, 24), conj_74900, *[], **kwargs_74901)
    
    # Obtaining the member 'T' of a type (line 74)
    T_74903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 24), conj_call_result_74902, 'T')
    # Obtaining the member 'dot' of a type (line 74)
    dot_74904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 24), T_74903, 'dot')
    # Calling dot(args, kwargs) (line 74)
    dot_call_result_74907 = invoke(stypy.reporting.localization.Localization(__file__, 74, 24), dot_74904, *[u_74905], **kwargs_74906)
    
    
    # Call to eye(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'n' (line 74)
    n_74910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 50), 'n', False)
    # Processing the call keyword arguments (line 74)
    kwargs_74911 = {}
    # Getting the type of 'np' (line 74)
    np_74908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 43), 'np', False)
    # Obtaining the member 'eye' of a type (line 74)
    eye_74909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 43), np_74908, 'eye')
    # Calling eye(args, kwargs) (line 74)
    eye_call_result_74912 = invoke(stypy.reporting.localization.Localization(__file__, 74, 43), eye_74909, *[n_74910], **kwargs_74911)
    
    # Processing the call keyword arguments (line 74)
    float_74913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 59), 'float')
    keyword_74914 = float_74913
    kwargs_74915 = {'atol': keyword_74914}
    # Getting the type of 'assert_allclose' (line 74)
    assert_allclose_74898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 74)
    assert_allclose_call_result_74916 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), assert_allclose_74898, *[dot_call_result_74907, eye_call_result_74912], **kwargs_74915)
    
    # SSA branch for the else part of an if statement (line 73)
    module_type_store.open_ssa_branch('else')
    
    # Call to assert_allclose(...): (line 76)
    # Processing the call arguments (line 76)
    
    # Call to dot(...): (line 76)
    # Processing the call arguments (line 76)
    
    # Call to conj(...): (line 76)
    # Processing the call keyword arguments (line 76)
    kwargs_74922 = {}
    # Getting the type of 'u' (line 76)
    u_74920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 30), 'u', False)
    # Obtaining the member 'conj' of a type (line 76)
    conj_74921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 30), u_74920, 'conj')
    # Calling conj(args, kwargs) (line 76)
    conj_call_result_74923 = invoke(stypy.reporting.localization.Localization(__file__, 76, 30), conj_74921, *[], **kwargs_74922)
    
    # Obtaining the member 'T' of a type (line 76)
    T_74924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 30), conj_call_result_74923, 'T')
    # Processing the call keyword arguments (line 76)
    kwargs_74925 = {}
    # Getting the type of 'u' (line 76)
    u_74918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'u', False)
    # Obtaining the member 'dot' of a type (line 76)
    dot_74919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 24), u_74918, 'dot')
    # Calling dot(args, kwargs) (line 76)
    dot_call_result_74926 = invoke(stypy.reporting.localization.Localization(__file__, 76, 24), dot_74919, *[T_74924], **kwargs_74925)
    
    
    # Call to eye(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'm' (line 76)
    m_74929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 50), 'm', False)
    # Processing the call keyword arguments (line 76)
    kwargs_74930 = {}
    # Getting the type of 'np' (line 76)
    np_74927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 43), 'np', False)
    # Obtaining the member 'eye' of a type (line 76)
    eye_74928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 43), np_74927, 'eye')
    # Calling eye(args, kwargs) (line 76)
    eye_call_result_74931 = invoke(stypy.reporting.localization.Localization(__file__, 76, 43), eye_74928, *[m_74929], **kwargs_74930)
    
    # Processing the call keyword arguments (line 76)
    float_74932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 59), 'float')
    keyword_74933 = float_74932
    kwargs_74934 = {'atol': keyword_74933}
    # Getting the type of 'assert_allclose' (line 76)
    assert_allclose_74917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 76)
    assert_allclose_call_result_74935 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), assert_allclose_74917, *[dot_call_result_74926, eye_call_result_74931], **kwargs_74934)
    
    # SSA join for if statement (line 73)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to assert_allclose(...): (line 78)
    # Processing the call arguments (line 78)
    
    # Call to conj(...): (line 78)
    # Processing the call keyword arguments (line 78)
    kwargs_74939 = {}
    # Getting the type of 'p' (line 78)
    p_74937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 20), 'p', False)
    # Obtaining the member 'conj' of a type (line 78)
    conj_74938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 20), p_74937, 'conj')
    # Calling conj(args, kwargs) (line 78)
    conj_call_result_74940 = invoke(stypy.reporting.localization.Localization(__file__, 78, 20), conj_74938, *[], **kwargs_74939)
    
    # Obtaining the member 'T' of a type (line 78)
    T_74941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 20), conj_call_result_74940, 'T')
    # Getting the type of 'p' (line 78)
    p_74942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 32), 'p', False)
    # Processing the call keyword arguments (line 78)
    kwargs_74943 = {}
    # Getting the type of 'assert_allclose' (line 78)
    assert_allclose_74936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 78)
    assert_allclose_call_result_74944 = invoke(stypy.reporting.localization.Localization(__file__, 78, 4), assert_allclose_74936, *[T_74941, p_74942], **kwargs_74943)
    
    
    # Assigning a Call to a Name (line 79):
    
    # Assigning a Call to a Name (line 79):
    
    # Call to eigh(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'p' (line 79)
    p_74946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 17), 'p', False)
    # Processing the call keyword arguments (line 79)
    # Getting the type of 'True' (line 79)
    True_74947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 33), 'True', False)
    keyword_74948 = True_74947
    kwargs_74949 = {'eigvals_only': keyword_74948}
    # Getting the type of 'eigh' (line 79)
    eigh_74945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'eigh', False)
    # Calling eigh(args, kwargs) (line 79)
    eigh_call_result_74950 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), eigh_74945, *[p_74946], **kwargs_74949)
    
    # Assigning a type to the variable 'evals' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'evals', eigh_call_result_74950)
    
    # Assigning a Subscript to a Name (line 80):
    
    # Assigning a Subscript to a Name (line 80):
    
    # Obtaining the type of the subscript
    
    
    # Call to abs(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'evals' (line 80)
    evals_74952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 30), 'evals', False)
    # Processing the call keyword arguments (line 80)
    kwargs_74953 = {}
    # Getting the type of 'abs' (line 80)
    abs_74951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 26), 'abs', False)
    # Calling abs(args, kwargs) (line 80)
    abs_call_result_74954 = invoke(stypy.reporting.localization.Localization(__file__, 80, 26), abs_74951, *[evals_74952], **kwargs_74953)
    
    float_74955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 39), 'float')
    # Applying the binary operator '>' (line 80)
    result_gt_74956 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 26), '>', abs_call_result_74954, float_74955)
    
    # Getting the type of 'evals' (line 80)
    evals_74957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'evals')
    # Obtaining the member '__getitem__' of a type (line 80)
    getitem___74958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 20), evals_74957, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 80)
    subscript_call_result_74959 = invoke(stypy.reporting.localization.Localization(__file__, 80, 20), getitem___74958, result_gt_74956)
    
    # Assigning a type to the variable 'nonzero_evals' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'nonzero_evals', subscript_call_result_74959)
    
    # Call to assert_(...): (line 81)
    # Processing the call arguments (line 81)
    
    # Call to all(...): (line 81)
    # Processing the call keyword arguments (line 81)
    kwargs_74965 = {}
    
    # Getting the type of 'nonzero_evals' (line 81)
    nonzero_evals_74961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 13), 'nonzero_evals', False)
    int_74962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 30), 'int')
    # Applying the binary operator '>=' (line 81)
    result_ge_74963 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 13), '>=', nonzero_evals_74961, int_74962)
    
    # Obtaining the member 'all' of a type (line 81)
    all_74964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 13), result_ge_74963, 'all')
    # Calling all(args, kwargs) (line 81)
    all_call_result_74966 = invoke(stypy.reporting.localization.Localization(__file__, 81, 13), all_74964, *[], **kwargs_74965)
    
    # Processing the call keyword arguments (line 81)
    kwargs_74967 = {}
    # Getting the type of 'assert_' (line 81)
    assert__74960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 81)
    assert__call_result_74968 = invoke(stypy.reporting.localization.Localization(__file__, 81, 4), assert__74960, *[all_call_result_74966], **kwargs_74967)
    
    
    # ################# End of 'verify_polar(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'verify_polar' in the type store
    # Getting the type of 'stypy_return_type' (line 45)
    stypy_return_type_74969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_74969)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'verify_polar'
    return stypy_return_type_74969

# Assigning a type to the variable 'verify_polar' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'verify_polar', verify_polar)

@norecursion
def test_precomputed_cases(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_precomputed_cases'
    module_type_store = module_type_store.open_function_context('test_precomputed_cases', 84, 0, False)
    
    # Passed parameters checking function
    test_precomputed_cases.stypy_localization = localization
    test_precomputed_cases.stypy_type_of_self = None
    test_precomputed_cases.stypy_type_store = module_type_store
    test_precomputed_cases.stypy_function_name = 'test_precomputed_cases'
    test_precomputed_cases.stypy_param_names_list = []
    test_precomputed_cases.stypy_varargs_param_name = None
    test_precomputed_cases.stypy_kwargs_param_name = None
    test_precomputed_cases.stypy_call_defaults = defaults
    test_precomputed_cases.stypy_call_varargs = varargs
    test_precomputed_cases.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_precomputed_cases', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_precomputed_cases', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_precomputed_cases(...)' code ##################

    
    # Getting the type of 'precomputed_cases' (line 85)
    precomputed_cases_74970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 43), 'precomputed_cases')
    # Testing the type of a for loop iterable (line 85)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 85, 4), precomputed_cases_74970)
    # Getting the type of the for loop variable (line 85)
    for_loop_var_74971 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 85, 4), precomputed_cases_74970)
    # Assigning a type to the variable 'a' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 4), for_loop_var_74971))
    # Assigning a type to the variable 'side' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'side', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 4), for_loop_var_74971))
    # Assigning a type to the variable 'expected_u' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'expected_u', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 4), for_loop_var_74971))
    # Assigning a type to the variable 'expected_p' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'expected_p', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 4), for_loop_var_74971))
    # SSA begins for a for statement (line 85)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check_precomputed_polar(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'a' (line 86)
    a_74973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 32), 'a', False)
    # Getting the type of 'side' (line 86)
    side_74974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 35), 'side', False)
    # Getting the type of 'expected_u' (line 86)
    expected_u_74975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 41), 'expected_u', False)
    # Getting the type of 'expected_p' (line 86)
    expected_p_74976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 53), 'expected_p', False)
    # Processing the call keyword arguments (line 86)
    kwargs_74977 = {}
    # Getting the type of 'check_precomputed_polar' (line 86)
    check_precomputed_polar_74972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'check_precomputed_polar', False)
    # Calling check_precomputed_polar(args, kwargs) (line 86)
    check_precomputed_polar_call_result_74978 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), check_precomputed_polar_74972, *[a_74973, side_74974, expected_u_74975, expected_p_74976], **kwargs_74977)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_precomputed_cases(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_precomputed_cases' in the type store
    # Getting the type of 'stypy_return_type' (line 84)
    stypy_return_type_74979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_74979)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_precomputed_cases'
    return stypy_return_type_74979

# Assigning a type to the variable 'test_precomputed_cases' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'test_precomputed_cases', test_precomputed_cases)

@norecursion
def test_verify_cases(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_verify_cases'
    module_type_store = module_type_store.open_function_context('test_verify_cases', 89, 0, False)
    
    # Passed parameters checking function
    test_verify_cases.stypy_localization = localization
    test_verify_cases.stypy_type_of_self = None
    test_verify_cases.stypy_type_store = module_type_store
    test_verify_cases.stypy_function_name = 'test_verify_cases'
    test_verify_cases.stypy_param_names_list = []
    test_verify_cases.stypy_varargs_param_name = None
    test_verify_cases.stypy_kwargs_param_name = None
    test_verify_cases.stypy_call_defaults = defaults
    test_verify_cases.stypy_call_varargs = varargs
    test_verify_cases.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_verify_cases', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_verify_cases', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_verify_cases(...)' code ##################

    
    # Getting the type of 'verify_cases' (line 90)
    verify_cases_74980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 13), 'verify_cases')
    # Testing the type of a for loop iterable (line 90)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 90, 4), verify_cases_74980)
    # Getting the type of the for loop variable (line 90)
    for_loop_var_74981 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 90, 4), verify_cases_74980)
    # Assigning a type to the variable 'a' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'a', for_loop_var_74981)
    # SSA begins for a for statement (line 90)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to verify_polar(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'a' (line 91)
    a_74983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 21), 'a', False)
    # Processing the call keyword arguments (line 91)
    kwargs_74984 = {}
    # Getting the type of 'verify_polar' (line 91)
    verify_polar_74982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'verify_polar', False)
    # Calling verify_polar(args, kwargs) (line 91)
    verify_polar_call_result_74985 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), verify_polar_74982, *[a_74983], **kwargs_74984)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_verify_cases(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_verify_cases' in the type store
    # Getting the type of 'stypy_return_type' (line 89)
    stypy_return_type_74986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_74986)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_verify_cases'
    return stypy_return_type_74986

# Assigning a type to the variable 'test_verify_cases' (line 89)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), 'test_verify_cases', test_verify_cases)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
