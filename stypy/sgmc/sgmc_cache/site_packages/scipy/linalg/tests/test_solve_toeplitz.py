
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Test functions for linalg._solve_toeplitz module
2: '''
3: from __future__ import division, print_function, absolute_import
4: 
5: import numpy as np
6: from scipy.linalg._solve_toeplitz import levinson
7: from scipy.linalg import solve, toeplitz, solve_toeplitz
8: from numpy.testing import assert_equal, assert_allclose
9: 
10: import pytest
11: from pytest import raises as assert_raises
12: 
13: 
14: def test_solve_equivalence():
15:     # For toeplitz matrices, solve_toeplitz() should be equivalent to solve().
16:     random = np.random.RandomState(1234)
17:     for n in (1, 2, 3, 10):
18:         c = random.randn(n)
19:         if random.rand() < 0.5:
20:             c = c + 1j * random.randn(n)
21:         r = random.randn(n)
22:         if random.rand() < 0.5:
23:             r = r + 1j * random.randn(n)
24:         y = random.randn(n)
25:         if random.rand() < 0.5:
26:             y = y + 1j * random.randn(n)
27: 
28:         # Check equivalence when both the column and row are provided.
29:         actual = solve_toeplitz((c,r), y)
30:         desired = solve(toeplitz(c, r=r), y)
31:         assert_allclose(actual, desired)
32: 
33:         # Check equivalence when the column is provided but not the row.
34:         actual = solve_toeplitz(c, b=y)
35:         desired = solve(toeplitz(c), y)
36:         assert_allclose(actual, desired)
37: 
38: 
39: def test_multiple_rhs():
40:     random = np.random.RandomState(1234)
41:     c = random.randn(4)
42:     r = random.randn(4)
43:     for offset in [0, 1j]:
44:         for yshape in ((4,), (4, 3), (4, 3, 2)):
45:             y = random.randn(*yshape) + offset
46:             actual = solve_toeplitz((c,r), b=y)
47:             desired = solve(toeplitz(c, r=r), y)
48:             assert_equal(actual.shape, yshape)
49:             assert_equal(desired.shape, yshape)
50:             assert_allclose(actual, desired)
51:             
52:             
53: def test_native_list_arguments():
54:     c = [1,2,4,7]
55:     r = [1,3,9,12]
56:     y = [5,1,4,2]
57:     actual = solve_toeplitz((c,r), y)
58:     desired = solve(toeplitz(c, r=r), y)
59:     assert_allclose(actual, desired)
60: 
61: 
62: def test_zero_diag_error():
63:     # The Levinson-Durbin implementation fails when the diagonal is zero.
64:     random = np.random.RandomState(1234)
65:     n = 4
66:     c = random.randn(n)
67:     r = random.randn(n)
68:     y = random.randn(n)
69:     c[0] = 0
70:     assert_raises(np.linalg.LinAlgError,
71:         solve_toeplitz, (c, r), b=y)
72: 
73: 
74: def test_wikipedia_counterexample():
75:     # The Levinson-Durbin implementation also fails in other cases.
76:     # This example is from the talk page of the wikipedia article.
77:     random = np.random.RandomState(1234)
78:     c = [2, 2, 1]
79:     y = random.randn(3)
80:     assert_raises(np.linalg.LinAlgError, solve_toeplitz, c, b=y)
81: 
82: 
83: def test_reflection_coeffs():
84:     # check that that the partial solutions are given by the reflection
85:     # coefficients
86: 
87:     random = np.random.RandomState(1234)
88:     y_d = random.randn(10)
89:     y_z = random.randn(10) + 1j
90:     reflection_coeffs_d = [1]
91:     reflection_coeffs_z = [1]
92:     for i in range(2, 10):
93:         reflection_coeffs_d.append(solve_toeplitz(y_d[:(i-1)], b=y_d[1:i])[-1])
94:         reflection_coeffs_z.append(solve_toeplitz(y_z[:(i-1)], b=y_z[1:i])[-1])
95: 
96:     y_d_concat = np.concatenate((y_d[-2:0:-1], y_d[:-1]))
97:     y_z_concat = np.concatenate((y_z[-2:0:-1].conj(), y_z[:-1]))
98:     _, ref_d = levinson(y_d_concat, b=y_d[1:])
99:     _, ref_z = levinson(y_z_concat, b=y_z[1:])
100: 
101:     assert_allclose(reflection_coeffs_d, ref_d[:-1])
102:     assert_allclose(reflection_coeffs_z, ref_z[:-1])
103: 
104: 
105: @pytest.mark.xfail(reason='Instability of Levinson iteration')
106: def test_unstable():
107:     # this is a "Gaussian Toeplitz matrix", as mentioned in Example 2 of
108:     # I. Gohbert, T. Kailath and V. Olshevsky "Fast Gaussian Elimination with
109:     # Partial Pivoting for Matrices with Displacement Structure"
110:     # Mathematics of Computation, 64, 212 (1995), pp 1557-1576
111:     # which can be unstable for levinson recursion.
112: 
113:     # other fast toeplitz solvers such as GKO or Burg should be better.
114:     random = np.random.RandomState(1234)
115:     n = 100
116:     c = 0.9 ** (np.arange(n)**2)
117:     y = random.randn(n)
118: 
119:     solution1 = solve_toeplitz(c, b=y)
120:     solution2 = solve(toeplitz(c), y)
121: 
122:     assert_allclose(solution1, solution2)
123: 
124: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_110434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', 'Test functions for linalg._solve_toeplitz module\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_110435 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_110435) is not StypyTypeError):

    if (import_110435 != 'pyd_module'):
        __import__(import_110435)
        sys_modules_110436 = sys.modules[import_110435]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_110436.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_110435)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.linalg._solve_toeplitz import levinson' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_110437 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.linalg._solve_toeplitz')

if (type(import_110437) is not StypyTypeError):

    if (import_110437 != 'pyd_module'):
        __import__(import_110437)
        sys_modules_110438 = sys.modules[import_110437]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.linalg._solve_toeplitz', sys_modules_110438.module_type_store, module_type_store, ['levinson'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_110438, sys_modules_110438.module_type_store, module_type_store)
    else:
        from scipy.linalg._solve_toeplitz import levinson

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.linalg._solve_toeplitz', None, module_type_store, ['levinson'], [levinson])

else:
    # Assigning a type to the variable 'scipy.linalg._solve_toeplitz' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.linalg._solve_toeplitz', import_110437)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.linalg import solve, toeplitz, solve_toeplitz' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_110439 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg')

if (type(import_110439) is not StypyTypeError):

    if (import_110439 != 'pyd_module'):
        __import__(import_110439)
        sys_modules_110440 = sys.modules[import_110439]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg', sys_modules_110440.module_type_store, module_type_store, ['solve', 'toeplitz', 'solve_toeplitz'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_110440, sys_modules_110440.module_type_store, module_type_store)
    else:
        from scipy.linalg import solve, toeplitz, solve_toeplitz

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg', None, module_type_store, ['solve', 'toeplitz', 'solve_toeplitz'], [solve, toeplitz, solve_toeplitz])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg', import_110439)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from numpy.testing import assert_equal, assert_allclose' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_110441 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing')

if (type(import_110441) is not StypyTypeError):

    if (import_110441 != 'pyd_module'):
        __import__(import_110441)
        sys_modules_110442 = sys.modules[import_110441]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', sys_modules_110442.module_type_store, module_type_store, ['assert_equal', 'assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_110442, sys_modules_110442.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_allclose'], [assert_equal, assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', import_110441)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import pytest' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_110443 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest')

if (type(import_110443) is not StypyTypeError):

    if (import_110443 != 'pyd_module'):
        __import__(import_110443)
        sys_modules_110444 = sys.modules[import_110443]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest', sys_modules_110444.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest', import_110443)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from pytest import assert_raises' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_110445 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'pytest')

if (type(import_110445) is not StypyTypeError):

    if (import_110445 != 'pyd_module'):
        __import__(import_110445)
        sys_modules_110446 = sys.modules[import_110445]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'pytest', sys_modules_110446.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_110446, sys_modules_110446.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'pytest', import_110445)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')


@norecursion
def test_solve_equivalence(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_solve_equivalence'
    module_type_store = module_type_store.open_function_context('test_solve_equivalence', 14, 0, False)
    
    # Passed parameters checking function
    test_solve_equivalence.stypy_localization = localization
    test_solve_equivalence.stypy_type_of_self = None
    test_solve_equivalence.stypy_type_store = module_type_store
    test_solve_equivalence.stypy_function_name = 'test_solve_equivalence'
    test_solve_equivalence.stypy_param_names_list = []
    test_solve_equivalence.stypy_varargs_param_name = None
    test_solve_equivalence.stypy_kwargs_param_name = None
    test_solve_equivalence.stypy_call_defaults = defaults
    test_solve_equivalence.stypy_call_varargs = varargs
    test_solve_equivalence.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_solve_equivalence', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_solve_equivalence', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_solve_equivalence(...)' code ##################

    
    # Assigning a Call to a Name (line 16):
    
    # Assigning a Call to a Name (line 16):
    
    # Call to RandomState(...): (line 16)
    # Processing the call arguments (line 16)
    int_110450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 35), 'int')
    # Processing the call keyword arguments (line 16)
    kwargs_110451 = {}
    # Getting the type of 'np' (line 16)
    np_110447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 13), 'np', False)
    # Obtaining the member 'random' of a type (line 16)
    random_110448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 13), np_110447, 'random')
    # Obtaining the member 'RandomState' of a type (line 16)
    RandomState_110449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 13), random_110448, 'RandomState')
    # Calling RandomState(args, kwargs) (line 16)
    RandomState_call_result_110452 = invoke(stypy.reporting.localization.Localization(__file__, 16, 13), RandomState_110449, *[int_110450], **kwargs_110451)
    
    # Assigning a type to the variable 'random' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'random', RandomState_call_result_110452)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 17)
    tuple_110453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 17)
    # Adding element type (line 17)
    int_110454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 14), tuple_110453, int_110454)
    # Adding element type (line 17)
    int_110455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 14), tuple_110453, int_110455)
    # Adding element type (line 17)
    int_110456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 14), tuple_110453, int_110456)
    # Adding element type (line 17)
    int_110457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 14), tuple_110453, int_110457)
    
    # Testing the type of a for loop iterable (line 17)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 17, 4), tuple_110453)
    # Getting the type of the for loop variable (line 17)
    for_loop_var_110458 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 17, 4), tuple_110453)
    # Assigning a type to the variable 'n' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'n', for_loop_var_110458)
    # SSA begins for a for statement (line 17)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 18):
    
    # Assigning a Call to a Name (line 18):
    
    # Call to randn(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'n' (line 18)
    n_110461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 25), 'n', False)
    # Processing the call keyword arguments (line 18)
    kwargs_110462 = {}
    # Getting the type of 'random' (line 18)
    random_110459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'random', False)
    # Obtaining the member 'randn' of a type (line 18)
    randn_110460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 12), random_110459, 'randn')
    # Calling randn(args, kwargs) (line 18)
    randn_call_result_110463 = invoke(stypy.reporting.localization.Localization(__file__, 18, 12), randn_110460, *[n_110461], **kwargs_110462)
    
    # Assigning a type to the variable 'c' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'c', randn_call_result_110463)
    
    
    
    # Call to rand(...): (line 19)
    # Processing the call keyword arguments (line 19)
    kwargs_110466 = {}
    # Getting the type of 'random' (line 19)
    random_110464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 11), 'random', False)
    # Obtaining the member 'rand' of a type (line 19)
    rand_110465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 11), random_110464, 'rand')
    # Calling rand(args, kwargs) (line 19)
    rand_call_result_110467 = invoke(stypy.reporting.localization.Localization(__file__, 19, 11), rand_110465, *[], **kwargs_110466)
    
    float_110468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 27), 'float')
    # Applying the binary operator '<' (line 19)
    result_lt_110469 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 11), '<', rand_call_result_110467, float_110468)
    
    # Testing the type of an if condition (line 19)
    if_condition_110470 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 19, 8), result_lt_110469)
    # Assigning a type to the variable 'if_condition_110470' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'if_condition_110470', if_condition_110470)
    # SSA begins for if statement (line 19)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 20):
    
    # Assigning a BinOp to a Name (line 20):
    # Getting the type of 'c' (line 20)
    c_110471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'c')
    complex_110472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 20), 'complex')
    
    # Call to randn(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'n' (line 20)
    n_110475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 38), 'n', False)
    # Processing the call keyword arguments (line 20)
    kwargs_110476 = {}
    # Getting the type of 'random' (line 20)
    random_110473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 25), 'random', False)
    # Obtaining the member 'randn' of a type (line 20)
    randn_110474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 25), random_110473, 'randn')
    # Calling randn(args, kwargs) (line 20)
    randn_call_result_110477 = invoke(stypy.reporting.localization.Localization(__file__, 20, 25), randn_110474, *[n_110475], **kwargs_110476)
    
    # Applying the binary operator '*' (line 20)
    result_mul_110478 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 20), '*', complex_110472, randn_call_result_110477)
    
    # Applying the binary operator '+' (line 20)
    result_add_110479 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 16), '+', c_110471, result_mul_110478)
    
    # Assigning a type to the variable 'c' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'c', result_add_110479)
    # SSA join for if statement (line 19)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 21):
    
    # Assigning a Call to a Name (line 21):
    
    # Call to randn(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'n' (line 21)
    n_110482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 25), 'n', False)
    # Processing the call keyword arguments (line 21)
    kwargs_110483 = {}
    # Getting the type of 'random' (line 21)
    random_110480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'random', False)
    # Obtaining the member 'randn' of a type (line 21)
    randn_110481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 12), random_110480, 'randn')
    # Calling randn(args, kwargs) (line 21)
    randn_call_result_110484 = invoke(stypy.reporting.localization.Localization(__file__, 21, 12), randn_110481, *[n_110482], **kwargs_110483)
    
    # Assigning a type to the variable 'r' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'r', randn_call_result_110484)
    
    
    
    # Call to rand(...): (line 22)
    # Processing the call keyword arguments (line 22)
    kwargs_110487 = {}
    # Getting the type of 'random' (line 22)
    random_110485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'random', False)
    # Obtaining the member 'rand' of a type (line 22)
    rand_110486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 11), random_110485, 'rand')
    # Calling rand(args, kwargs) (line 22)
    rand_call_result_110488 = invoke(stypy.reporting.localization.Localization(__file__, 22, 11), rand_110486, *[], **kwargs_110487)
    
    float_110489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 27), 'float')
    # Applying the binary operator '<' (line 22)
    result_lt_110490 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 11), '<', rand_call_result_110488, float_110489)
    
    # Testing the type of an if condition (line 22)
    if_condition_110491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 22, 8), result_lt_110490)
    # Assigning a type to the variable 'if_condition_110491' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'if_condition_110491', if_condition_110491)
    # SSA begins for if statement (line 22)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 23):
    
    # Assigning a BinOp to a Name (line 23):
    # Getting the type of 'r' (line 23)
    r_110492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 16), 'r')
    complex_110493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 20), 'complex')
    
    # Call to randn(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'n' (line 23)
    n_110496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 38), 'n', False)
    # Processing the call keyword arguments (line 23)
    kwargs_110497 = {}
    # Getting the type of 'random' (line 23)
    random_110494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 25), 'random', False)
    # Obtaining the member 'randn' of a type (line 23)
    randn_110495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 25), random_110494, 'randn')
    # Calling randn(args, kwargs) (line 23)
    randn_call_result_110498 = invoke(stypy.reporting.localization.Localization(__file__, 23, 25), randn_110495, *[n_110496], **kwargs_110497)
    
    # Applying the binary operator '*' (line 23)
    result_mul_110499 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 20), '*', complex_110493, randn_call_result_110498)
    
    # Applying the binary operator '+' (line 23)
    result_add_110500 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 16), '+', r_110492, result_mul_110499)
    
    # Assigning a type to the variable 'r' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'r', result_add_110500)
    # SSA join for if statement (line 22)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 24):
    
    # Assigning a Call to a Name (line 24):
    
    # Call to randn(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'n' (line 24)
    n_110503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 25), 'n', False)
    # Processing the call keyword arguments (line 24)
    kwargs_110504 = {}
    # Getting the type of 'random' (line 24)
    random_110501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'random', False)
    # Obtaining the member 'randn' of a type (line 24)
    randn_110502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 12), random_110501, 'randn')
    # Calling randn(args, kwargs) (line 24)
    randn_call_result_110505 = invoke(stypy.reporting.localization.Localization(__file__, 24, 12), randn_110502, *[n_110503], **kwargs_110504)
    
    # Assigning a type to the variable 'y' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'y', randn_call_result_110505)
    
    
    
    # Call to rand(...): (line 25)
    # Processing the call keyword arguments (line 25)
    kwargs_110508 = {}
    # Getting the type of 'random' (line 25)
    random_110506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 11), 'random', False)
    # Obtaining the member 'rand' of a type (line 25)
    rand_110507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 11), random_110506, 'rand')
    # Calling rand(args, kwargs) (line 25)
    rand_call_result_110509 = invoke(stypy.reporting.localization.Localization(__file__, 25, 11), rand_110507, *[], **kwargs_110508)
    
    float_110510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 27), 'float')
    # Applying the binary operator '<' (line 25)
    result_lt_110511 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 11), '<', rand_call_result_110509, float_110510)
    
    # Testing the type of an if condition (line 25)
    if_condition_110512 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 25, 8), result_lt_110511)
    # Assigning a type to the variable 'if_condition_110512' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'if_condition_110512', if_condition_110512)
    # SSA begins for if statement (line 25)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 26):
    
    # Assigning a BinOp to a Name (line 26):
    # Getting the type of 'y' (line 26)
    y_110513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), 'y')
    complex_110514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 20), 'complex')
    
    # Call to randn(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'n' (line 26)
    n_110517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 38), 'n', False)
    # Processing the call keyword arguments (line 26)
    kwargs_110518 = {}
    # Getting the type of 'random' (line 26)
    random_110515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 25), 'random', False)
    # Obtaining the member 'randn' of a type (line 26)
    randn_110516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 25), random_110515, 'randn')
    # Calling randn(args, kwargs) (line 26)
    randn_call_result_110519 = invoke(stypy.reporting.localization.Localization(__file__, 26, 25), randn_110516, *[n_110517], **kwargs_110518)
    
    # Applying the binary operator '*' (line 26)
    result_mul_110520 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 20), '*', complex_110514, randn_call_result_110519)
    
    # Applying the binary operator '+' (line 26)
    result_add_110521 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 16), '+', y_110513, result_mul_110520)
    
    # Assigning a type to the variable 'y' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'y', result_add_110521)
    # SSA join for if statement (line 25)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 29):
    
    # Assigning a Call to a Name (line 29):
    
    # Call to solve_toeplitz(...): (line 29)
    # Processing the call arguments (line 29)
    
    # Obtaining an instance of the builtin type 'tuple' (line 29)
    tuple_110523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 29)
    # Adding element type (line 29)
    # Getting the type of 'c' (line 29)
    c_110524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 33), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 33), tuple_110523, c_110524)
    # Adding element type (line 29)
    # Getting the type of 'r' (line 29)
    r_110525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 35), 'r', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 33), tuple_110523, r_110525)
    
    # Getting the type of 'y' (line 29)
    y_110526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 39), 'y', False)
    # Processing the call keyword arguments (line 29)
    kwargs_110527 = {}
    # Getting the type of 'solve_toeplitz' (line 29)
    solve_toeplitz_110522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 17), 'solve_toeplitz', False)
    # Calling solve_toeplitz(args, kwargs) (line 29)
    solve_toeplitz_call_result_110528 = invoke(stypy.reporting.localization.Localization(__file__, 29, 17), solve_toeplitz_110522, *[tuple_110523, y_110526], **kwargs_110527)
    
    # Assigning a type to the variable 'actual' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'actual', solve_toeplitz_call_result_110528)
    
    # Assigning a Call to a Name (line 30):
    
    # Assigning a Call to a Name (line 30):
    
    # Call to solve(...): (line 30)
    # Processing the call arguments (line 30)
    
    # Call to toeplitz(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'c' (line 30)
    c_110531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 33), 'c', False)
    # Processing the call keyword arguments (line 30)
    # Getting the type of 'r' (line 30)
    r_110532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 38), 'r', False)
    keyword_110533 = r_110532
    kwargs_110534 = {'r': keyword_110533}
    # Getting the type of 'toeplitz' (line 30)
    toeplitz_110530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 24), 'toeplitz', False)
    # Calling toeplitz(args, kwargs) (line 30)
    toeplitz_call_result_110535 = invoke(stypy.reporting.localization.Localization(__file__, 30, 24), toeplitz_110530, *[c_110531], **kwargs_110534)
    
    # Getting the type of 'y' (line 30)
    y_110536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 42), 'y', False)
    # Processing the call keyword arguments (line 30)
    kwargs_110537 = {}
    # Getting the type of 'solve' (line 30)
    solve_110529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 18), 'solve', False)
    # Calling solve(args, kwargs) (line 30)
    solve_call_result_110538 = invoke(stypy.reporting.localization.Localization(__file__, 30, 18), solve_110529, *[toeplitz_call_result_110535, y_110536], **kwargs_110537)
    
    # Assigning a type to the variable 'desired' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'desired', solve_call_result_110538)
    
    # Call to assert_allclose(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'actual' (line 31)
    actual_110540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 24), 'actual', False)
    # Getting the type of 'desired' (line 31)
    desired_110541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 32), 'desired', False)
    # Processing the call keyword arguments (line 31)
    kwargs_110542 = {}
    # Getting the type of 'assert_allclose' (line 31)
    assert_allclose_110539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 31)
    assert_allclose_call_result_110543 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), assert_allclose_110539, *[actual_110540, desired_110541], **kwargs_110542)
    
    
    # Assigning a Call to a Name (line 34):
    
    # Assigning a Call to a Name (line 34):
    
    # Call to solve_toeplitz(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'c' (line 34)
    c_110545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 32), 'c', False)
    # Processing the call keyword arguments (line 34)
    # Getting the type of 'y' (line 34)
    y_110546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 37), 'y', False)
    keyword_110547 = y_110546
    kwargs_110548 = {'b': keyword_110547}
    # Getting the type of 'solve_toeplitz' (line 34)
    solve_toeplitz_110544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 17), 'solve_toeplitz', False)
    # Calling solve_toeplitz(args, kwargs) (line 34)
    solve_toeplitz_call_result_110549 = invoke(stypy.reporting.localization.Localization(__file__, 34, 17), solve_toeplitz_110544, *[c_110545], **kwargs_110548)
    
    # Assigning a type to the variable 'actual' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'actual', solve_toeplitz_call_result_110549)
    
    # Assigning a Call to a Name (line 35):
    
    # Assigning a Call to a Name (line 35):
    
    # Call to solve(...): (line 35)
    # Processing the call arguments (line 35)
    
    # Call to toeplitz(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'c' (line 35)
    c_110552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 33), 'c', False)
    # Processing the call keyword arguments (line 35)
    kwargs_110553 = {}
    # Getting the type of 'toeplitz' (line 35)
    toeplitz_110551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 24), 'toeplitz', False)
    # Calling toeplitz(args, kwargs) (line 35)
    toeplitz_call_result_110554 = invoke(stypy.reporting.localization.Localization(__file__, 35, 24), toeplitz_110551, *[c_110552], **kwargs_110553)
    
    # Getting the type of 'y' (line 35)
    y_110555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 37), 'y', False)
    # Processing the call keyword arguments (line 35)
    kwargs_110556 = {}
    # Getting the type of 'solve' (line 35)
    solve_110550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 18), 'solve', False)
    # Calling solve(args, kwargs) (line 35)
    solve_call_result_110557 = invoke(stypy.reporting.localization.Localization(__file__, 35, 18), solve_110550, *[toeplitz_call_result_110554, y_110555], **kwargs_110556)
    
    # Assigning a type to the variable 'desired' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'desired', solve_call_result_110557)
    
    # Call to assert_allclose(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'actual' (line 36)
    actual_110559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 24), 'actual', False)
    # Getting the type of 'desired' (line 36)
    desired_110560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 32), 'desired', False)
    # Processing the call keyword arguments (line 36)
    kwargs_110561 = {}
    # Getting the type of 'assert_allclose' (line 36)
    assert_allclose_110558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 36)
    assert_allclose_call_result_110562 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), assert_allclose_110558, *[actual_110559, desired_110560], **kwargs_110561)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_solve_equivalence(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_solve_equivalence' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_110563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_110563)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_solve_equivalence'
    return stypy_return_type_110563

# Assigning a type to the variable 'test_solve_equivalence' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'test_solve_equivalence', test_solve_equivalence)

@norecursion
def test_multiple_rhs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_multiple_rhs'
    module_type_store = module_type_store.open_function_context('test_multiple_rhs', 39, 0, False)
    
    # Passed parameters checking function
    test_multiple_rhs.stypy_localization = localization
    test_multiple_rhs.stypy_type_of_self = None
    test_multiple_rhs.stypy_type_store = module_type_store
    test_multiple_rhs.stypy_function_name = 'test_multiple_rhs'
    test_multiple_rhs.stypy_param_names_list = []
    test_multiple_rhs.stypy_varargs_param_name = None
    test_multiple_rhs.stypy_kwargs_param_name = None
    test_multiple_rhs.stypy_call_defaults = defaults
    test_multiple_rhs.stypy_call_varargs = varargs
    test_multiple_rhs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_multiple_rhs', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_multiple_rhs', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_multiple_rhs(...)' code ##################

    
    # Assigning a Call to a Name (line 40):
    
    # Assigning a Call to a Name (line 40):
    
    # Call to RandomState(...): (line 40)
    # Processing the call arguments (line 40)
    int_110567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 35), 'int')
    # Processing the call keyword arguments (line 40)
    kwargs_110568 = {}
    # Getting the type of 'np' (line 40)
    np_110564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 13), 'np', False)
    # Obtaining the member 'random' of a type (line 40)
    random_110565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 13), np_110564, 'random')
    # Obtaining the member 'RandomState' of a type (line 40)
    RandomState_110566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 13), random_110565, 'RandomState')
    # Calling RandomState(args, kwargs) (line 40)
    RandomState_call_result_110569 = invoke(stypy.reporting.localization.Localization(__file__, 40, 13), RandomState_110566, *[int_110567], **kwargs_110568)
    
    # Assigning a type to the variable 'random' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'random', RandomState_call_result_110569)
    
    # Assigning a Call to a Name (line 41):
    
    # Assigning a Call to a Name (line 41):
    
    # Call to randn(...): (line 41)
    # Processing the call arguments (line 41)
    int_110572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 21), 'int')
    # Processing the call keyword arguments (line 41)
    kwargs_110573 = {}
    # Getting the type of 'random' (line 41)
    random_110570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'random', False)
    # Obtaining the member 'randn' of a type (line 41)
    randn_110571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), random_110570, 'randn')
    # Calling randn(args, kwargs) (line 41)
    randn_call_result_110574 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), randn_110571, *[int_110572], **kwargs_110573)
    
    # Assigning a type to the variable 'c' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'c', randn_call_result_110574)
    
    # Assigning a Call to a Name (line 42):
    
    # Assigning a Call to a Name (line 42):
    
    # Call to randn(...): (line 42)
    # Processing the call arguments (line 42)
    int_110577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 21), 'int')
    # Processing the call keyword arguments (line 42)
    kwargs_110578 = {}
    # Getting the type of 'random' (line 42)
    random_110575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'random', False)
    # Obtaining the member 'randn' of a type (line 42)
    randn_110576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), random_110575, 'randn')
    # Calling randn(args, kwargs) (line 42)
    randn_call_result_110579 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), randn_110576, *[int_110577], **kwargs_110578)
    
    # Assigning a type to the variable 'r' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'r', randn_call_result_110579)
    
    
    # Obtaining an instance of the builtin type 'list' (line 43)
    list_110580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 43)
    # Adding element type (line 43)
    int_110581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), list_110580, int_110581)
    # Adding element type (line 43)
    complex_110582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 22), 'complex')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), list_110580, complex_110582)
    
    # Testing the type of a for loop iterable (line 43)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 43, 4), list_110580)
    # Getting the type of the for loop variable (line 43)
    for_loop_var_110583 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 43, 4), list_110580)
    # Assigning a type to the variable 'offset' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'offset', for_loop_var_110583)
    # SSA begins for a for statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 44)
    tuple_110584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 44)
    # Adding element type (line 44)
    
    # Obtaining an instance of the builtin type 'tuple' (line 44)
    tuple_110585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 44)
    # Adding element type (line 44)
    int_110586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 24), tuple_110585, int_110586)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 23), tuple_110584, tuple_110585)
    # Adding element type (line 44)
    
    # Obtaining an instance of the builtin type 'tuple' (line 44)
    tuple_110587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 44)
    # Adding element type (line 44)
    int_110588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 30), tuple_110587, int_110588)
    # Adding element type (line 44)
    int_110589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 30), tuple_110587, int_110589)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 23), tuple_110584, tuple_110587)
    # Adding element type (line 44)
    
    # Obtaining an instance of the builtin type 'tuple' (line 44)
    tuple_110590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 44)
    # Adding element type (line 44)
    int_110591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 38), tuple_110590, int_110591)
    # Adding element type (line 44)
    int_110592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 38), tuple_110590, int_110592)
    # Adding element type (line 44)
    int_110593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 38), tuple_110590, int_110593)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 23), tuple_110584, tuple_110590)
    
    # Testing the type of a for loop iterable (line 44)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 44, 8), tuple_110584)
    # Getting the type of the for loop variable (line 44)
    for_loop_var_110594 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 44, 8), tuple_110584)
    # Assigning a type to the variable 'yshape' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'yshape', for_loop_var_110594)
    # SSA begins for a for statement (line 44)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 45):
    
    # Assigning a BinOp to a Name (line 45):
    
    # Call to randn(...): (line 45)
    # Getting the type of 'yshape' (line 45)
    yshape_110597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 30), 'yshape', False)
    # Processing the call keyword arguments (line 45)
    kwargs_110598 = {}
    # Getting the type of 'random' (line 45)
    random_110595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 16), 'random', False)
    # Obtaining the member 'randn' of a type (line 45)
    randn_110596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 16), random_110595, 'randn')
    # Calling randn(args, kwargs) (line 45)
    randn_call_result_110599 = invoke(stypy.reporting.localization.Localization(__file__, 45, 16), randn_110596, *[yshape_110597], **kwargs_110598)
    
    # Getting the type of 'offset' (line 45)
    offset_110600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 40), 'offset')
    # Applying the binary operator '+' (line 45)
    result_add_110601 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 16), '+', randn_call_result_110599, offset_110600)
    
    # Assigning a type to the variable 'y' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'y', result_add_110601)
    
    # Assigning a Call to a Name (line 46):
    
    # Assigning a Call to a Name (line 46):
    
    # Call to solve_toeplitz(...): (line 46)
    # Processing the call arguments (line 46)
    
    # Obtaining an instance of the builtin type 'tuple' (line 46)
    tuple_110603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 46)
    # Adding element type (line 46)
    # Getting the type of 'c' (line 46)
    c_110604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 37), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 37), tuple_110603, c_110604)
    # Adding element type (line 46)
    # Getting the type of 'r' (line 46)
    r_110605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 39), 'r', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 37), tuple_110603, r_110605)
    
    # Processing the call keyword arguments (line 46)
    # Getting the type of 'y' (line 46)
    y_110606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 45), 'y', False)
    keyword_110607 = y_110606
    kwargs_110608 = {'b': keyword_110607}
    # Getting the type of 'solve_toeplitz' (line 46)
    solve_toeplitz_110602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 21), 'solve_toeplitz', False)
    # Calling solve_toeplitz(args, kwargs) (line 46)
    solve_toeplitz_call_result_110609 = invoke(stypy.reporting.localization.Localization(__file__, 46, 21), solve_toeplitz_110602, *[tuple_110603], **kwargs_110608)
    
    # Assigning a type to the variable 'actual' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'actual', solve_toeplitz_call_result_110609)
    
    # Assigning a Call to a Name (line 47):
    
    # Assigning a Call to a Name (line 47):
    
    # Call to solve(...): (line 47)
    # Processing the call arguments (line 47)
    
    # Call to toeplitz(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'c' (line 47)
    c_110612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 37), 'c', False)
    # Processing the call keyword arguments (line 47)
    # Getting the type of 'r' (line 47)
    r_110613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 42), 'r', False)
    keyword_110614 = r_110613
    kwargs_110615 = {'r': keyword_110614}
    # Getting the type of 'toeplitz' (line 47)
    toeplitz_110611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 28), 'toeplitz', False)
    # Calling toeplitz(args, kwargs) (line 47)
    toeplitz_call_result_110616 = invoke(stypy.reporting.localization.Localization(__file__, 47, 28), toeplitz_110611, *[c_110612], **kwargs_110615)
    
    # Getting the type of 'y' (line 47)
    y_110617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 46), 'y', False)
    # Processing the call keyword arguments (line 47)
    kwargs_110618 = {}
    # Getting the type of 'solve' (line 47)
    solve_110610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 22), 'solve', False)
    # Calling solve(args, kwargs) (line 47)
    solve_call_result_110619 = invoke(stypy.reporting.localization.Localization(__file__, 47, 22), solve_110610, *[toeplitz_call_result_110616, y_110617], **kwargs_110618)
    
    # Assigning a type to the variable 'desired' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'desired', solve_call_result_110619)
    
    # Call to assert_equal(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'actual' (line 48)
    actual_110621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 25), 'actual', False)
    # Obtaining the member 'shape' of a type (line 48)
    shape_110622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 25), actual_110621, 'shape')
    # Getting the type of 'yshape' (line 48)
    yshape_110623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 39), 'yshape', False)
    # Processing the call keyword arguments (line 48)
    kwargs_110624 = {}
    # Getting the type of 'assert_equal' (line 48)
    assert_equal_110620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 48)
    assert_equal_call_result_110625 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), assert_equal_110620, *[shape_110622, yshape_110623], **kwargs_110624)
    
    
    # Call to assert_equal(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'desired' (line 49)
    desired_110627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'desired', False)
    # Obtaining the member 'shape' of a type (line 49)
    shape_110628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 25), desired_110627, 'shape')
    # Getting the type of 'yshape' (line 49)
    yshape_110629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 40), 'yshape', False)
    # Processing the call keyword arguments (line 49)
    kwargs_110630 = {}
    # Getting the type of 'assert_equal' (line 49)
    assert_equal_110626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 49)
    assert_equal_call_result_110631 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), assert_equal_110626, *[shape_110628, yshape_110629], **kwargs_110630)
    
    
    # Call to assert_allclose(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'actual' (line 50)
    actual_110633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 28), 'actual', False)
    # Getting the type of 'desired' (line 50)
    desired_110634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 36), 'desired', False)
    # Processing the call keyword arguments (line 50)
    kwargs_110635 = {}
    # Getting the type of 'assert_allclose' (line 50)
    assert_allclose_110632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 50)
    assert_allclose_call_result_110636 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), assert_allclose_110632, *[actual_110633, desired_110634], **kwargs_110635)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_multiple_rhs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_multiple_rhs' in the type store
    # Getting the type of 'stypy_return_type' (line 39)
    stypy_return_type_110637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_110637)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_multiple_rhs'
    return stypy_return_type_110637

# Assigning a type to the variable 'test_multiple_rhs' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'test_multiple_rhs', test_multiple_rhs)

@norecursion
def test_native_list_arguments(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_native_list_arguments'
    module_type_store = module_type_store.open_function_context('test_native_list_arguments', 53, 0, False)
    
    # Passed parameters checking function
    test_native_list_arguments.stypy_localization = localization
    test_native_list_arguments.stypy_type_of_self = None
    test_native_list_arguments.stypy_type_store = module_type_store
    test_native_list_arguments.stypy_function_name = 'test_native_list_arguments'
    test_native_list_arguments.stypy_param_names_list = []
    test_native_list_arguments.stypy_varargs_param_name = None
    test_native_list_arguments.stypy_kwargs_param_name = None
    test_native_list_arguments.stypy_call_defaults = defaults
    test_native_list_arguments.stypy_call_varargs = varargs
    test_native_list_arguments.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_native_list_arguments', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_native_list_arguments', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_native_list_arguments(...)' code ##################

    
    # Assigning a List to a Name (line 54):
    
    # Assigning a List to a Name (line 54):
    
    # Obtaining an instance of the builtin type 'list' (line 54)
    list_110638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 54)
    # Adding element type (line 54)
    int_110639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 8), list_110638, int_110639)
    # Adding element type (line 54)
    int_110640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 8), list_110638, int_110640)
    # Adding element type (line 54)
    int_110641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 8), list_110638, int_110641)
    # Adding element type (line 54)
    int_110642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 8), list_110638, int_110642)
    
    # Assigning a type to the variable 'c' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'c', list_110638)
    
    # Assigning a List to a Name (line 55):
    
    # Assigning a List to a Name (line 55):
    
    # Obtaining an instance of the builtin type 'list' (line 55)
    list_110643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 55)
    # Adding element type (line 55)
    int_110644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 8), list_110643, int_110644)
    # Adding element type (line 55)
    int_110645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 8), list_110643, int_110645)
    # Adding element type (line 55)
    int_110646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 8), list_110643, int_110646)
    # Adding element type (line 55)
    int_110647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 8), list_110643, int_110647)
    
    # Assigning a type to the variable 'r' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'r', list_110643)
    
    # Assigning a List to a Name (line 56):
    
    # Assigning a List to a Name (line 56):
    
    # Obtaining an instance of the builtin type 'list' (line 56)
    list_110648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 56)
    # Adding element type (line 56)
    int_110649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 8), list_110648, int_110649)
    # Adding element type (line 56)
    int_110650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 8), list_110648, int_110650)
    # Adding element type (line 56)
    int_110651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 8), list_110648, int_110651)
    # Adding element type (line 56)
    int_110652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 8), list_110648, int_110652)
    
    # Assigning a type to the variable 'y' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'y', list_110648)
    
    # Assigning a Call to a Name (line 57):
    
    # Assigning a Call to a Name (line 57):
    
    # Call to solve_toeplitz(...): (line 57)
    # Processing the call arguments (line 57)
    
    # Obtaining an instance of the builtin type 'tuple' (line 57)
    tuple_110654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 57)
    # Adding element type (line 57)
    # Getting the type of 'c' (line 57)
    c_110655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 29), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 29), tuple_110654, c_110655)
    # Adding element type (line 57)
    # Getting the type of 'r' (line 57)
    r_110656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 31), 'r', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 29), tuple_110654, r_110656)
    
    # Getting the type of 'y' (line 57)
    y_110657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 35), 'y', False)
    # Processing the call keyword arguments (line 57)
    kwargs_110658 = {}
    # Getting the type of 'solve_toeplitz' (line 57)
    solve_toeplitz_110653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 13), 'solve_toeplitz', False)
    # Calling solve_toeplitz(args, kwargs) (line 57)
    solve_toeplitz_call_result_110659 = invoke(stypy.reporting.localization.Localization(__file__, 57, 13), solve_toeplitz_110653, *[tuple_110654, y_110657], **kwargs_110658)
    
    # Assigning a type to the variable 'actual' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'actual', solve_toeplitz_call_result_110659)
    
    # Assigning a Call to a Name (line 58):
    
    # Assigning a Call to a Name (line 58):
    
    # Call to solve(...): (line 58)
    # Processing the call arguments (line 58)
    
    # Call to toeplitz(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'c' (line 58)
    c_110662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 29), 'c', False)
    # Processing the call keyword arguments (line 58)
    # Getting the type of 'r' (line 58)
    r_110663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 34), 'r', False)
    keyword_110664 = r_110663
    kwargs_110665 = {'r': keyword_110664}
    # Getting the type of 'toeplitz' (line 58)
    toeplitz_110661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 20), 'toeplitz', False)
    # Calling toeplitz(args, kwargs) (line 58)
    toeplitz_call_result_110666 = invoke(stypy.reporting.localization.Localization(__file__, 58, 20), toeplitz_110661, *[c_110662], **kwargs_110665)
    
    # Getting the type of 'y' (line 58)
    y_110667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 38), 'y', False)
    # Processing the call keyword arguments (line 58)
    kwargs_110668 = {}
    # Getting the type of 'solve' (line 58)
    solve_110660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 14), 'solve', False)
    # Calling solve(args, kwargs) (line 58)
    solve_call_result_110669 = invoke(stypy.reporting.localization.Localization(__file__, 58, 14), solve_110660, *[toeplitz_call_result_110666, y_110667], **kwargs_110668)
    
    # Assigning a type to the variable 'desired' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'desired', solve_call_result_110669)
    
    # Call to assert_allclose(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'actual' (line 59)
    actual_110671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'actual', False)
    # Getting the type of 'desired' (line 59)
    desired_110672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 28), 'desired', False)
    # Processing the call keyword arguments (line 59)
    kwargs_110673 = {}
    # Getting the type of 'assert_allclose' (line 59)
    assert_allclose_110670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 59)
    assert_allclose_call_result_110674 = invoke(stypy.reporting.localization.Localization(__file__, 59, 4), assert_allclose_110670, *[actual_110671, desired_110672], **kwargs_110673)
    
    
    # ################# End of 'test_native_list_arguments(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_native_list_arguments' in the type store
    # Getting the type of 'stypy_return_type' (line 53)
    stypy_return_type_110675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_110675)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_native_list_arguments'
    return stypy_return_type_110675

# Assigning a type to the variable 'test_native_list_arguments' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'test_native_list_arguments', test_native_list_arguments)

@norecursion
def test_zero_diag_error(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_zero_diag_error'
    module_type_store = module_type_store.open_function_context('test_zero_diag_error', 62, 0, False)
    
    # Passed parameters checking function
    test_zero_diag_error.stypy_localization = localization
    test_zero_diag_error.stypy_type_of_self = None
    test_zero_diag_error.stypy_type_store = module_type_store
    test_zero_diag_error.stypy_function_name = 'test_zero_diag_error'
    test_zero_diag_error.stypy_param_names_list = []
    test_zero_diag_error.stypy_varargs_param_name = None
    test_zero_diag_error.stypy_kwargs_param_name = None
    test_zero_diag_error.stypy_call_defaults = defaults
    test_zero_diag_error.stypy_call_varargs = varargs
    test_zero_diag_error.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_zero_diag_error', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_zero_diag_error', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_zero_diag_error(...)' code ##################

    
    # Assigning a Call to a Name (line 64):
    
    # Assigning a Call to a Name (line 64):
    
    # Call to RandomState(...): (line 64)
    # Processing the call arguments (line 64)
    int_110679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 35), 'int')
    # Processing the call keyword arguments (line 64)
    kwargs_110680 = {}
    # Getting the type of 'np' (line 64)
    np_110676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 13), 'np', False)
    # Obtaining the member 'random' of a type (line 64)
    random_110677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 13), np_110676, 'random')
    # Obtaining the member 'RandomState' of a type (line 64)
    RandomState_110678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 13), random_110677, 'RandomState')
    # Calling RandomState(args, kwargs) (line 64)
    RandomState_call_result_110681 = invoke(stypy.reporting.localization.Localization(__file__, 64, 13), RandomState_110678, *[int_110679], **kwargs_110680)
    
    # Assigning a type to the variable 'random' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'random', RandomState_call_result_110681)
    
    # Assigning a Num to a Name (line 65):
    
    # Assigning a Num to a Name (line 65):
    int_110682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 8), 'int')
    # Assigning a type to the variable 'n' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'n', int_110682)
    
    # Assigning a Call to a Name (line 66):
    
    # Assigning a Call to a Name (line 66):
    
    # Call to randn(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'n' (line 66)
    n_110685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 21), 'n', False)
    # Processing the call keyword arguments (line 66)
    kwargs_110686 = {}
    # Getting the type of 'random' (line 66)
    random_110683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'random', False)
    # Obtaining the member 'randn' of a type (line 66)
    randn_110684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), random_110683, 'randn')
    # Calling randn(args, kwargs) (line 66)
    randn_call_result_110687 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), randn_110684, *[n_110685], **kwargs_110686)
    
    # Assigning a type to the variable 'c' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'c', randn_call_result_110687)
    
    # Assigning a Call to a Name (line 67):
    
    # Assigning a Call to a Name (line 67):
    
    # Call to randn(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'n' (line 67)
    n_110690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 21), 'n', False)
    # Processing the call keyword arguments (line 67)
    kwargs_110691 = {}
    # Getting the type of 'random' (line 67)
    random_110688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'random', False)
    # Obtaining the member 'randn' of a type (line 67)
    randn_110689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), random_110688, 'randn')
    # Calling randn(args, kwargs) (line 67)
    randn_call_result_110692 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), randn_110689, *[n_110690], **kwargs_110691)
    
    # Assigning a type to the variable 'r' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'r', randn_call_result_110692)
    
    # Assigning a Call to a Name (line 68):
    
    # Assigning a Call to a Name (line 68):
    
    # Call to randn(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'n' (line 68)
    n_110695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 21), 'n', False)
    # Processing the call keyword arguments (line 68)
    kwargs_110696 = {}
    # Getting the type of 'random' (line 68)
    random_110693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'random', False)
    # Obtaining the member 'randn' of a type (line 68)
    randn_110694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), random_110693, 'randn')
    # Calling randn(args, kwargs) (line 68)
    randn_call_result_110697 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), randn_110694, *[n_110695], **kwargs_110696)
    
    # Assigning a type to the variable 'y' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'y', randn_call_result_110697)
    
    # Assigning a Num to a Subscript (line 69):
    
    # Assigning a Num to a Subscript (line 69):
    int_110698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 11), 'int')
    # Getting the type of 'c' (line 69)
    c_110699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'c')
    int_110700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 6), 'int')
    # Storing an element on a container (line 69)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 4), c_110699, (int_110700, int_110698))
    
    # Call to assert_raises(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'np' (line 70)
    np_110702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 18), 'np', False)
    # Obtaining the member 'linalg' of a type (line 70)
    linalg_110703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 18), np_110702, 'linalg')
    # Obtaining the member 'LinAlgError' of a type (line 70)
    LinAlgError_110704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 18), linalg_110703, 'LinAlgError')
    # Getting the type of 'solve_toeplitz' (line 71)
    solve_toeplitz_110705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'solve_toeplitz', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 71)
    tuple_110706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 71)
    # Adding element type (line 71)
    # Getting the type of 'c' (line 71)
    c_110707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 25), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 25), tuple_110706, c_110707)
    # Adding element type (line 71)
    # Getting the type of 'r' (line 71)
    r_110708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 28), 'r', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 25), tuple_110706, r_110708)
    
    # Processing the call keyword arguments (line 70)
    # Getting the type of 'y' (line 71)
    y_110709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 34), 'y', False)
    keyword_110710 = y_110709
    kwargs_110711 = {'b': keyword_110710}
    # Getting the type of 'assert_raises' (line 70)
    assert_raises_110701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 70)
    assert_raises_call_result_110712 = invoke(stypy.reporting.localization.Localization(__file__, 70, 4), assert_raises_110701, *[LinAlgError_110704, solve_toeplitz_110705, tuple_110706], **kwargs_110711)
    
    
    # ################# End of 'test_zero_diag_error(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_zero_diag_error' in the type store
    # Getting the type of 'stypy_return_type' (line 62)
    stypy_return_type_110713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_110713)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_zero_diag_error'
    return stypy_return_type_110713

# Assigning a type to the variable 'test_zero_diag_error' (line 62)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'test_zero_diag_error', test_zero_diag_error)

@norecursion
def test_wikipedia_counterexample(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_wikipedia_counterexample'
    module_type_store = module_type_store.open_function_context('test_wikipedia_counterexample', 74, 0, False)
    
    # Passed parameters checking function
    test_wikipedia_counterexample.stypy_localization = localization
    test_wikipedia_counterexample.stypy_type_of_self = None
    test_wikipedia_counterexample.stypy_type_store = module_type_store
    test_wikipedia_counterexample.stypy_function_name = 'test_wikipedia_counterexample'
    test_wikipedia_counterexample.stypy_param_names_list = []
    test_wikipedia_counterexample.stypy_varargs_param_name = None
    test_wikipedia_counterexample.stypy_kwargs_param_name = None
    test_wikipedia_counterexample.stypy_call_defaults = defaults
    test_wikipedia_counterexample.stypy_call_varargs = varargs
    test_wikipedia_counterexample.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_wikipedia_counterexample', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_wikipedia_counterexample', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_wikipedia_counterexample(...)' code ##################

    
    # Assigning a Call to a Name (line 77):
    
    # Assigning a Call to a Name (line 77):
    
    # Call to RandomState(...): (line 77)
    # Processing the call arguments (line 77)
    int_110717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 35), 'int')
    # Processing the call keyword arguments (line 77)
    kwargs_110718 = {}
    # Getting the type of 'np' (line 77)
    np_110714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 13), 'np', False)
    # Obtaining the member 'random' of a type (line 77)
    random_110715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 13), np_110714, 'random')
    # Obtaining the member 'RandomState' of a type (line 77)
    RandomState_110716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 13), random_110715, 'RandomState')
    # Calling RandomState(args, kwargs) (line 77)
    RandomState_call_result_110719 = invoke(stypy.reporting.localization.Localization(__file__, 77, 13), RandomState_110716, *[int_110717], **kwargs_110718)
    
    # Assigning a type to the variable 'random' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'random', RandomState_call_result_110719)
    
    # Assigning a List to a Name (line 78):
    
    # Assigning a List to a Name (line 78):
    
    # Obtaining an instance of the builtin type 'list' (line 78)
    list_110720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 78)
    # Adding element type (line 78)
    int_110721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 8), list_110720, int_110721)
    # Adding element type (line 78)
    int_110722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 8), list_110720, int_110722)
    # Adding element type (line 78)
    int_110723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 8), list_110720, int_110723)
    
    # Assigning a type to the variable 'c' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'c', list_110720)
    
    # Assigning a Call to a Name (line 79):
    
    # Assigning a Call to a Name (line 79):
    
    # Call to randn(...): (line 79)
    # Processing the call arguments (line 79)
    int_110726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 21), 'int')
    # Processing the call keyword arguments (line 79)
    kwargs_110727 = {}
    # Getting the type of 'random' (line 79)
    random_110724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'random', False)
    # Obtaining the member 'randn' of a type (line 79)
    randn_110725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), random_110724, 'randn')
    # Calling randn(args, kwargs) (line 79)
    randn_call_result_110728 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), randn_110725, *[int_110726], **kwargs_110727)
    
    # Assigning a type to the variable 'y' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'y', randn_call_result_110728)
    
    # Call to assert_raises(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'np' (line 80)
    np_110730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 18), 'np', False)
    # Obtaining the member 'linalg' of a type (line 80)
    linalg_110731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 18), np_110730, 'linalg')
    # Obtaining the member 'LinAlgError' of a type (line 80)
    LinAlgError_110732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 18), linalg_110731, 'LinAlgError')
    # Getting the type of 'solve_toeplitz' (line 80)
    solve_toeplitz_110733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 41), 'solve_toeplitz', False)
    # Getting the type of 'c' (line 80)
    c_110734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 57), 'c', False)
    # Processing the call keyword arguments (line 80)
    # Getting the type of 'y' (line 80)
    y_110735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 62), 'y', False)
    keyword_110736 = y_110735
    kwargs_110737 = {'b': keyword_110736}
    # Getting the type of 'assert_raises' (line 80)
    assert_raises_110729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 80)
    assert_raises_call_result_110738 = invoke(stypy.reporting.localization.Localization(__file__, 80, 4), assert_raises_110729, *[LinAlgError_110732, solve_toeplitz_110733, c_110734], **kwargs_110737)
    
    
    # ################# End of 'test_wikipedia_counterexample(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_wikipedia_counterexample' in the type store
    # Getting the type of 'stypy_return_type' (line 74)
    stypy_return_type_110739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_110739)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_wikipedia_counterexample'
    return stypy_return_type_110739

# Assigning a type to the variable 'test_wikipedia_counterexample' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'test_wikipedia_counterexample', test_wikipedia_counterexample)

@norecursion
def test_reflection_coeffs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_reflection_coeffs'
    module_type_store = module_type_store.open_function_context('test_reflection_coeffs', 83, 0, False)
    
    # Passed parameters checking function
    test_reflection_coeffs.stypy_localization = localization
    test_reflection_coeffs.stypy_type_of_self = None
    test_reflection_coeffs.stypy_type_store = module_type_store
    test_reflection_coeffs.stypy_function_name = 'test_reflection_coeffs'
    test_reflection_coeffs.stypy_param_names_list = []
    test_reflection_coeffs.stypy_varargs_param_name = None
    test_reflection_coeffs.stypy_kwargs_param_name = None
    test_reflection_coeffs.stypy_call_defaults = defaults
    test_reflection_coeffs.stypy_call_varargs = varargs
    test_reflection_coeffs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_reflection_coeffs', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_reflection_coeffs', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_reflection_coeffs(...)' code ##################

    
    # Assigning a Call to a Name (line 87):
    
    # Assigning a Call to a Name (line 87):
    
    # Call to RandomState(...): (line 87)
    # Processing the call arguments (line 87)
    int_110743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 35), 'int')
    # Processing the call keyword arguments (line 87)
    kwargs_110744 = {}
    # Getting the type of 'np' (line 87)
    np_110740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'np', False)
    # Obtaining the member 'random' of a type (line 87)
    random_110741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 13), np_110740, 'random')
    # Obtaining the member 'RandomState' of a type (line 87)
    RandomState_110742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 13), random_110741, 'RandomState')
    # Calling RandomState(args, kwargs) (line 87)
    RandomState_call_result_110745 = invoke(stypy.reporting.localization.Localization(__file__, 87, 13), RandomState_110742, *[int_110743], **kwargs_110744)
    
    # Assigning a type to the variable 'random' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'random', RandomState_call_result_110745)
    
    # Assigning a Call to a Name (line 88):
    
    # Assigning a Call to a Name (line 88):
    
    # Call to randn(...): (line 88)
    # Processing the call arguments (line 88)
    int_110748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 23), 'int')
    # Processing the call keyword arguments (line 88)
    kwargs_110749 = {}
    # Getting the type of 'random' (line 88)
    random_110746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 10), 'random', False)
    # Obtaining the member 'randn' of a type (line 88)
    randn_110747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 10), random_110746, 'randn')
    # Calling randn(args, kwargs) (line 88)
    randn_call_result_110750 = invoke(stypy.reporting.localization.Localization(__file__, 88, 10), randn_110747, *[int_110748], **kwargs_110749)
    
    # Assigning a type to the variable 'y_d' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'y_d', randn_call_result_110750)
    
    # Assigning a BinOp to a Name (line 89):
    
    # Assigning a BinOp to a Name (line 89):
    
    # Call to randn(...): (line 89)
    # Processing the call arguments (line 89)
    int_110753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 23), 'int')
    # Processing the call keyword arguments (line 89)
    kwargs_110754 = {}
    # Getting the type of 'random' (line 89)
    random_110751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 10), 'random', False)
    # Obtaining the member 'randn' of a type (line 89)
    randn_110752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 10), random_110751, 'randn')
    # Calling randn(args, kwargs) (line 89)
    randn_call_result_110755 = invoke(stypy.reporting.localization.Localization(__file__, 89, 10), randn_110752, *[int_110753], **kwargs_110754)
    
    complex_110756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 29), 'complex')
    # Applying the binary operator '+' (line 89)
    result_add_110757 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 10), '+', randn_call_result_110755, complex_110756)
    
    # Assigning a type to the variable 'y_z' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'y_z', result_add_110757)
    
    # Assigning a List to a Name (line 90):
    
    # Assigning a List to a Name (line 90):
    
    # Obtaining an instance of the builtin type 'list' (line 90)
    list_110758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 90)
    # Adding element type (line 90)
    int_110759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 26), list_110758, int_110759)
    
    # Assigning a type to the variable 'reflection_coeffs_d' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'reflection_coeffs_d', list_110758)
    
    # Assigning a List to a Name (line 91):
    
    # Assigning a List to a Name (line 91):
    
    # Obtaining an instance of the builtin type 'list' (line 91)
    list_110760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 91)
    # Adding element type (line 91)
    int_110761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 26), list_110760, int_110761)
    
    # Assigning a type to the variable 'reflection_coeffs_z' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'reflection_coeffs_z', list_110760)
    
    
    # Call to range(...): (line 92)
    # Processing the call arguments (line 92)
    int_110763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 19), 'int')
    int_110764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 22), 'int')
    # Processing the call keyword arguments (line 92)
    kwargs_110765 = {}
    # Getting the type of 'range' (line 92)
    range_110762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 13), 'range', False)
    # Calling range(args, kwargs) (line 92)
    range_call_result_110766 = invoke(stypy.reporting.localization.Localization(__file__, 92, 13), range_110762, *[int_110763, int_110764], **kwargs_110765)
    
    # Testing the type of a for loop iterable (line 92)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 92, 4), range_call_result_110766)
    # Getting the type of the for loop variable (line 92)
    for_loop_var_110767 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 92, 4), range_call_result_110766)
    # Assigning a type to the variable 'i' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'i', for_loop_var_110767)
    # SSA begins for a for statement (line 92)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 93)
    # Processing the call arguments (line 93)
    
    # Obtaining the type of the subscript
    int_110770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 75), 'int')
    
    # Call to solve_toeplitz(...): (line 93)
    # Processing the call arguments (line 93)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 93)
    i_110772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 56), 'i', False)
    int_110773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 58), 'int')
    # Applying the binary operator '-' (line 93)
    result_sub_110774 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 56), '-', i_110772, int_110773)
    
    slice_110775 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 93, 50), None, result_sub_110774, None)
    # Getting the type of 'y_d' (line 93)
    y_d_110776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 50), 'y_d', False)
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___110777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 50), y_d_110776, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_110778 = invoke(stypy.reporting.localization.Localization(__file__, 93, 50), getitem___110777, slice_110775)
    
    # Processing the call keyword arguments (line 93)
    
    # Obtaining the type of the subscript
    int_110779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 69), 'int')
    # Getting the type of 'i' (line 93)
    i_110780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 71), 'i', False)
    slice_110781 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 93, 65), int_110779, i_110780, None)
    # Getting the type of 'y_d' (line 93)
    y_d_110782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 65), 'y_d', False)
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___110783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 65), y_d_110782, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_110784 = invoke(stypy.reporting.localization.Localization(__file__, 93, 65), getitem___110783, slice_110781)
    
    keyword_110785 = subscript_call_result_110784
    kwargs_110786 = {'b': keyword_110785}
    # Getting the type of 'solve_toeplitz' (line 93)
    solve_toeplitz_110771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 35), 'solve_toeplitz', False)
    # Calling solve_toeplitz(args, kwargs) (line 93)
    solve_toeplitz_call_result_110787 = invoke(stypy.reporting.localization.Localization(__file__, 93, 35), solve_toeplitz_110771, *[subscript_call_result_110778], **kwargs_110786)
    
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___110788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 35), solve_toeplitz_call_result_110787, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_110789 = invoke(stypy.reporting.localization.Localization(__file__, 93, 35), getitem___110788, int_110770)
    
    # Processing the call keyword arguments (line 93)
    kwargs_110790 = {}
    # Getting the type of 'reflection_coeffs_d' (line 93)
    reflection_coeffs_d_110768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'reflection_coeffs_d', False)
    # Obtaining the member 'append' of a type (line 93)
    append_110769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), reflection_coeffs_d_110768, 'append')
    # Calling append(args, kwargs) (line 93)
    append_call_result_110791 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), append_110769, *[subscript_call_result_110789], **kwargs_110790)
    
    
    # Call to append(...): (line 94)
    # Processing the call arguments (line 94)
    
    # Obtaining the type of the subscript
    int_110794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 75), 'int')
    
    # Call to solve_toeplitz(...): (line 94)
    # Processing the call arguments (line 94)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 94)
    i_110796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 56), 'i', False)
    int_110797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 58), 'int')
    # Applying the binary operator '-' (line 94)
    result_sub_110798 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 56), '-', i_110796, int_110797)
    
    slice_110799 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 94, 50), None, result_sub_110798, None)
    # Getting the type of 'y_z' (line 94)
    y_z_110800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 50), 'y_z', False)
    # Obtaining the member '__getitem__' of a type (line 94)
    getitem___110801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 50), y_z_110800, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 94)
    subscript_call_result_110802 = invoke(stypy.reporting.localization.Localization(__file__, 94, 50), getitem___110801, slice_110799)
    
    # Processing the call keyword arguments (line 94)
    
    # Obtaining the type of the subscript
    int_110803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 69), 'int')
    # Getting the type of 'i' (line 94)
    i_110804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 71), 'i', False)
    slice_110805 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 94, 65), int_110803, i_110804, None)
    # Getting the type of 'y_z' (line 94)
    y_z_110806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 65), 'y_z', False)
    # Obtaining the member '__getitem__' of a type (line 94)
    getitem___110807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 65), y_z_110806, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 94)
    subscript_call_result_110808 = invoke(stypy.reporting.localization.Localization(__file__, 94, 65), getitem___110807, slice_110805)
    
    keyword_110809 = subscript_call_result_110808
    kwargs_110810 = {'b': keyword_110809}
    # Getting the type of 'solve_toeplitz' (line 94)
    solve_toeplitz_110795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 35), 'solve_toeplitz', False)
    # Calling solve_toeplitz(args, kwargs) (line 94)
    solve_toeplitz_call_result_110811 = invoke(stypy.reporting.localization.Localization(__file__, 94, 35), solve_toeplitz_110795, *[subscript_call_result_110802], **kwargs_110810)
    
    # Obtaining the member '__getitem__' of a type (line 94)
    getitem___110812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 35), solve_toeplitz_call_result_110811, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 94)
    subscript_call_result_110813 = invoke(stypy.reporting.localization.Localization(__file__, 94, 35), getitem___110812, int_110794)
    
    # Processing the call keyword arguments (line 94)
    kwargs_110814 = {}
    # Getting the type of 'reflection_coeffs_z' (line 94)
    reflection_coeffs_z_110792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'reflection_coeffs_z', False)
    # Obtaining the member 'append' of a type (line 94)
    append_110793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), reflection_coeffs_z_110792, 'append')
    # Calling append(args, kwargs) (line 94)
    append_call_result_110815 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), append_110793, *[subscript_call_result_110813], **kwargs_110814)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 96):
    
    # Assigning a Call to a Name (line 96):
    
    # Call to concatenate(...): (line 96)
    # Processing the call arguments (line 96)
    
    # Obtaining an instance of the builtin type 'tuple' (line 96)
    tuple_110818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 96)
    # Adding element type (line 96)
    
    # Obtaining the type of the subscript
    int_110819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 37), 'int')
    int_110820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 40), 'int')
    int_110821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 42), 'int')
    slice_110822 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 96, 33), int_110819, int_110820, int_110821)
    # Getting the type of 'y_d' (line 96)
    y_d_110823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 33), 'y_d', False)
    # Obtaining the member '__getitem__' of a type (line 96)
    getitem___110824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 33), y_d_110823, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 96)
    subscript_call_result_110825 = invoke(stypy.reporting.localization.Localization(__file__, 96, 33), getitem___110824, slice_110822)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 33), tuple_110818, subscript_call_result_110825)
    # Adding element type (line 96)
    
    # Obtaining the type of the subscript
    int_110826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 52), 'int')
    slice_110827 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 96, 47), None, int_110826, None)
    # Getting the type of 'y_d' (line 96)
    y_d_110828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 47), 'y_d', False)
    # Obtaining the member '__getitem__' of a type (line 96)
    getitem___110829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 47), y_d_110828, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 96)
    subscript_call_result_110830 = invoke(stypy.reporting.localization.Localization(__file__, 96, 47), getitem___110829, slice_110827)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 33), tuple_110818, subscript_call_result_110830)
    
    # Processing the call keyword arguments (line 96)
    kwargs_110831 = {}
    # Getting the type of 'np' (line 96)
    np_110816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 17), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 96)
    concatenate_110817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 17), np_110816, 'concatenate')
    # Calling concatenate(args, kwargs) (line 96)
    concatenate_call_result_110832 = invoke(stypy.reporting.localization.Localization(__file__, 96, 17), concatenate_110817, *[tuple_110818], **kwargs_110831)
    
    # Assigning a type to the variable 'y_d_concat' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'y_d_concat', concatenate_call_result_110832)
    
    # Assigning a Call to a Name (line 97):
    
    # Assigning a Call to a Name (line 97):
    
    # Call to concatenate(...): (line 97)
    # Processing the call arguments (line 97)
    
    # Obtaining an instance of the builtin type 'tuple' (line 97)
    tuple_110835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 97)
    # Adding element type (line 97)
    
    # Call to conj(...): (line 97)
    # Processing the call keyword arguments (line 97)
    kwargs_110844 = {}
    
    # Obtaining the type of the subscript
    int_110836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 37), 'int')
    int_110837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 40), 'int')
    int_110838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 42), 'int')
    slice_110839 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 97, 33), int_110836, int_110837, int_110838)
    # Getting the type of 'y_z' (line 97)
    y_z_110840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 33), 'y_z', False)
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___110841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 33), y_z_110840, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_110842 = invoke(stypy.reporting.localization.Localization(__file__, 97, 33), getitem___110841, slice_110839)
    
    # Obtaining the member 'conj' of a type (line 97)
    conj_110843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 33), subscript_call_result_110842, 'conj')
    # Calling conj(args, kwargs) (line 97)
    conj_call_result_110845 = invoke(stypy.reporting.localization.Localization(__file__, 97, 33), conj_110843, *[], **kwargs_110844)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 33), tuple_110835, conj_call_result_110845)
    # Adding element type (line 97)
    
    # Obtaining the type of the subscript
    int_110846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 59), 'int')
    slice_110847 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 97, 54), None, int_110846, None)
    # Getting the type of 'y_z' (line 97)
    y_z_110848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 54), 'y_z', False)
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___110849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 54), y_z_110848, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_110850 = invoke(stypy.reporting.localization.Localization(__file__, 97, 54), getitem___110849, slice_110847)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 33), tuple_110835, subscript_call_result_110850)
    
    # Processing the call keyword arguments (line 97)
    kwargs_110851 = {}
    # Getting the type of 'np' (line 97)
    np_110833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 17), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 97)
    concatenate_110834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 17), np_110833, 'concatenate')
    # Calling concatenate(args, kwargs) (line 97)
    concatenate_call_result_110852 = invoke(stypy.reporting.localization.Localization(__file__, 97, 17), concatenate_110834, *[tuple_110835], **kwargs_110851)
    
    # Assigning a type to the variable 'y_z_concat' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'y_z_concat', concatenate_call_result_110852)
    
    # Assigning a Call to a Tuple (line 98):
    
    # Assigning a Subscript to a Name (line 98):
    
    # Obtaining the type of the subscript
    int_110853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 4), 'int')
    
    # Call to levinson(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'y_d_concat' (line 98)
    y_d_concat_110855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 24), 'y_d_concat', False)
    # Processing the call keyword arguments (line 98)
    
    # Obtaining the type of the subscript
    int_110856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 42), 'int')
    slice_110857 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 98, 38), int_110856, None, None)
    # Getting the type of 'y_d' (line 98)
    y_d_110858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 38), 'y_d', False)
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___110859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 38), y_d_110858, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_110860 = invoke(stypy.reporting.localization.Localization(__file__, 98, 38), getitem___110859, slice_110857)
    
    keyword_110861 = subscript_call_result_110860
    kwargs_110862 = {'b': keyword_110861}
    # Getting the type of 'levinson' (line 98)
    levinson_110854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 15), 'levinson', False)
    # Calling levinson(args, kwargs) (line 98)
    levinson_call_result_110863 = invoke(stypy.reporting.localization.Localization(__file__, 98, 15), levinson_110854, *[y_d_concat_110855], **kwargs_110862)
    
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___110864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 4), levinson_call_result_110863, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_110865 = invoke(stypy.reporting.localization.Localization(__file__, 98, 4), getitem___110864, int_110853)
    
    # Assigning a type to the variable 'tuple_var_assignment_110430' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_110430', subscript_call_result_110865)
    
    # Assigning a Subscript to a Name (line 98):
    
    # Obtaining the type of the subscript
    int_110866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 4), 'int')
    
    # Call to levinson(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'y_d_concat' (line 98)
    y_d_concat_110868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 24), 'y_d_concat', False)
    # Processing the call keyword arguments (line 98)
    
    # Obtaining the type of the subscript
    int_110869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 42), 'int')
    slice_110870 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 98, 38), int_110869, None, None)
    # Getting the type of 'y_d' (line 98)
    y_d_110871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 38), 'y_d', False)
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___110872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 38), y_d_110871, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_110873 = invoke(stypy.reporting.localization.Localization(__file__, 98, 38), getitem___110872, slice_110870)
    
    keyword_110874 = subscript_call_result_110873
    kwargs_110875 = {'b': keyword_110874}
    # Getting the type of 'levinson' (line 98)
    levinson_110867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 15), 'levinson', False)
    # Calling levinson(args, kwargs) (line 98)
    levinson_call_result_110876 = invoke(stypy.reporting.localization.Localization(__file__, 98, 15), levinson_110867, *[y_d_concat_110868], **kwargs_110875)
    
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___110877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 4), levinson_call_result_110876, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_110878 = invoke(stypy.reporting.localization.Localization(__file__, 98, 4), getitem___110877, int_110866)
    
    # Assigning a type to the variable 'tuple_var_assignment_110431' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_110431', subscript_call_result_110878)
    
    # Assigning a Name to a Name (line 98):
    # Getting the type of 'tuple_var_assignment_110430' (line 98)
    tuple_var_assignment_110430_110879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_110430')
    # Assigning a type to the variable '_' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), '_', tuple_var_assignment_110430_110879)
    
    # Assigning a Name to a Name (line 98):
    # Getting the type of 'tuple_var_assignment_110431' (line 98)
    tuple_var_assignment_110431_110880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_110431')
    # Assigning a type to the variable 'ref_d' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 7), 'ref_d', tuple_var_assignment_110431_110880)
    
    # Assigning a Call to a Tuple (line 99):
    
    # Assigning a Subscript to a Name (line 99):
    
    # Obtaining the type of the subscript
    int_110881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 4), 'int')
    
    # Call to levinson(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'y_z_concat' (line 99)
    y_z_concat_110883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 24), 'y_z_concat', False)
    # Processing the call keyword arguments (line 99)
    
    # Obtaining the type of the subscript
    int_110884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 42), 'int')
    slice_110885 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 99, 38), int_110884, None, None)
    # Getting the type of 'y_z' (line 99)
    y_z_110886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 38), 'y_z', False)
    # Obtaining the member '__getitem__' of a type (line 99)
    getitem___110887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 38), y_z_110886, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 99)
    subscript_call_result_110888 = invoke(stypy.reporting.localization.Localization(__file__, 99, 38), getitem___110887, slice_110885)
    
    keyword_110889 = subscript_call_result_110888
    kwargs_110890 = {'b': keyword_110889}
    # Getting the type of 'levinson' (line 99)
    levinson_110882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 15), 'levinson', False)
    # Calling levinson(args, kwargs) (line 99)
    levinson_call_result_110891 = invoke(stypy.reporting.localization.Localization(__file__, 99, 15), levinson_110882, *[y_z_concat_110883], **kwargs_110890)
    
    # Obtaining the member '__getitem__' of a type (line 99)
    getitem___110892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 4), levinson_call_result_110891, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 99)
    subscript_call_result_110893 = invoke(stypy.reporting.localization.Localization(__file__, 99, 4), getitem___110892, int_110881)
    
    # Assigning a type to the variable 'tuple_var_assignment_110432' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'tuple_var_assignment_110432', subscript_call_result_110893)
    
    # Assigning a Subscript to a Name (line 99):
    
    # Obtaining the type of the subscript
    int_110894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 4), 'int')
    
    # Call to levinson(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'y_z_concat' (line 99)
    y_z_concat_110896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 24), 'y_z_concat', False)
    # Processing the call keyword arguments (line 99)
    
    # Obtaining the type of the subscript
    int_110897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 42), 'int')
    slice_110898 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 99, 38), int_110897, None, None)
    # Getting the type of 'y_z' (line 99)
    y_z_110899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 38), 'y_z', False)
    # Obtaining the member '__getitem__' of a type (line 99)
    getitem___110900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 38), y_z_110899, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 99)
    subscript_call_result_110901 = invoke(stypy.reporting.localization.Localization(__file__, 99, 38), getitem___110900, slice_110898)
    
    keyword_110902 = subscript_call_result_110901
    kwargs_110903 = {'b': keyword_110902}
    # Getting the type of 'levinson' (line 99)
    levinson_110895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 15), 'levinson', False)
    # Calling levinson(args, kwargs) (line 99)
    levinson_call_result_110904 = invoke(stypy.reporting.localization.Localization(__file__, 99, 15), levinson_110895, *[y_z_concat_110896], **kwargs_110903)
    
    # Obtaining the member '__getitem__' of a type (line 99)
    getitem___110905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 4), levinson_call_result_110904, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 99)
    subscript_call_result_110906 = invoke(stypy.reporting.localization.Localization(__file__, 99, 4), getitem___110905, int_110894)
    
    # Assigning a type to the variable 'tuple_var_assignment_110433' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'tuple_var_assignment_110433', subscript_call_result_110906)
    
    # Assigning a Name to a Name (line 99):
    # Getting the type of 'tuple_var_assignment_110432' (line 99)
    tuple_var_assignment_110432_110907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'tuple_var_assignment_110432')
    # Assigning a type to the variable '_' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), '_', tuple_var_assignment_110432_110907)
    
    # Assigning a Name to a Name (line 99):
    # Getting the type of 'tuple_var_assignment_110433' (line 99)
    tuple_var_assignment_110433_110908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'tuple_var_assignment_110433')
    # Assigning a type to the variable 'ref_z' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 7), 'ref_z', tuple_var_assignment_110433_110908)
    
    # Call to assert_allclose(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'reflection_coeffs_d' (line 101)
    reflection_coeffs_d_110910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 20), 'reflection_coeffs_d', False)
    
    # Obtaining the type of the subscript
    int_110911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 48), 'int')
    slice_110912 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 101, 41), None, int_110911, None)
    # Getting the type of 'ref_d' (line 101)
    ref_d_110913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 41), 'ref_d', False)
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___110914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 41), ref_d_110913, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_110915 = invoke(stypy.reporting.localization.Localization(__file__, 101, 41), getitem___110914, slice_110912)
    
    # Processing the call keyword arguments (line 101)
    kwargs_110916 = {}
    # Getting the type of 'assert_allclose' (line 101)
    assert_allclose_110909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 101)
    assert_allclose_call_result_110917 = invoke(stypy.reporting.localization.Localization(__file__, 101, 4), assert_allclose_110909, *[reflection_coeffs_d_110910, subscript_call_result_110915], **kwargs_110916)
    
    
    # Call to assert_allclose(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'reflection_coeffs_z' (line 102)
    reflection_coeffs_z_110919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 20), 'reflection_coeffs_z', False)
    
    # Obtaining the type of the subscript
    int_110920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 48), 'int')
    slice_110921 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 102, 41), None, int_110920, None)
    # Getting the type of 'ref_z' (line 102)
    ref_z_110922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 41), 'ref_z', False)
    # Obtaining the member '__getitem__' of a type (line 102)
    getitem___110923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 41), ref_z_110922, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 102)
    subscript_call_result_110924 = invoke(stypy.reporting.localization.Localization(__file__, 102, 41), getitem___110923, slice_110921)
    
    # Processing the call keyword arguments (line 102)
    kwargs_110925 = {}
    # Getting the type of 'assert_allclose' (line 102)
    assert_allclose_110918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 102)
    assert_allclose_call_result_110926 = invoke(stypy.reporting.localization.Localization(__file__, 102, 4), assert_allclose_110918, *[reflection_coeffs_z_110919, subscript_call_result_110924], **kwargs_110925)
    
    
    # ################# End of 'test_reflection_coeffs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_reflection_coeffs' in the type store
    # Getting the type of 'stypy_return_type' (line 83)
    stypy_return_type_110927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_110927)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_reflection_coeffs'
    return stypy_return_type_110927

# Assigning a type to the variable 'test_reflection_coeffs' (line 83)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'test_reflection_coeffs', test_reflection_coeffs)

@norecursion
def test_unstable(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_unstable'
    module_type_store = module_type_store.open_function_context('test_unstable', 105, 0, False)
    
    # Passed parameters checking function
    test_unstable.stypy_localization = localization
    test_unstable.stypy_type_of_self = None
    test_unstable.stypy_type_store = module_type_store
    test_unstable.stypy_function_name = 'test_unstable'
    test_unstable.stypy_param_names_list = []
    test_unstable.stypy_varargs_param_name = None
    test_unstable.stypy_kwargs_param_name = None
    test_unstable.stypy_call_defaults = defaults
    test_unstable.stypy_call_varargs = varargs
    test_unstable.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_unstable', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_unstable', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_unstable(...)' code ##################

    
    # Assigning a Call to a Name (line 114):
    
    # Assigning a Call to a Name (line 114):
    
    # Call to RandomState(...): (line 114)
    # Processing the call arguments (line 114)
    int_110931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 35), 'int')
    # Processing the call keyword arguments (line 114)
    kwargs_110932 = {}
    # Getting the type of 'np' (line 114)
    np_110928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 13), 'np', False)
    # Obtaining the member 'random' of a type (line 114)
    random_110929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 13), np_110928, 'random')
    # Obtaining the member 'RandomState' of a type (line 114)
    RandomState_110930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 13), random_110929, 'RandomState')
    # Calling RandomState(args, kwargs) (line 114)
    RandomState_call_result_110933 = invoke(stypy.reporting.localization.Localization(__file__, 114, 13), RandomState_110930, *[int_110931], **kwargs_110932)
    
    # Assigning a type to the variable 'random' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'random', RandomState_call_result_110933)
    
    # Assigning a Num to a Name (line 115):
    
    # Assigning a Num to a Name (line 115):
    int_110934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 8), 'int')
    # Assigning a type to the variable 'n' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'n', int_110934)
    
    # Assigning a BinOp to a Name (line 116):
    
    # Assigning a BinOp to a Name (line 116):
    float_110935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 8), 'float')
    
    # Call to arange(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'n' (line 116)
    n_110938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 26), 'n', False)
    # Processing the call keyword arguments (line 116)
    kwargs_110939 = {}
    # Getting the type of 'np' (line 116)
    np_110936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'np', False)
    # Obtaining the member 'arange' of a type (line 116)
    arange_110937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 16), np_110936, 'arange')
    # Calling arange(args, kwargs) (line 116)
    arange_call_result_110940 = invoke(stypy.reporting.localization.Localization(__file__, 116, 16), arange_110937, *[n_110938], **kwargs_110939)
    
    int_110941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 30), 'int')
    # Applying the binary operator '**' (line 116)
    result_pow_110942 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 16), '**', arange_call_result_110940, int_110941)
    
    # Applying the binary operator '**' (line 116)
    result_pow_110943 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 8), '**', float_110935, result_pow_110942)
    
    # Assigning a type to the variable 'c' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'c', result_pow_110943)
    
    # Assigning a Call to a Name (line 117):
    
    # Assigning a Call to a Name (line 117):
    
    # Call to randn(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'n' (line 117)
    n_110946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 21), 'n', False)
    # Processing the call keyword arguments (line 117)
    kwargs_110947 = {}
    # Getting the type of 'random' (line 117)
    random_110944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'random', False)
    # Obtaining the member 'randn' of a type (line 117)
    randn_110945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), random_110944, 'randn')
    # Calling randn(args, kwargs) (line 117)
    randn_call_result_110948 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), randn_110945, *[n_110946], **kwargs_110947)
    
    # Assigning a type to the variable 'y' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'y', randn_call_result_110948)
    
    # Assigning a Call to a Name (line 119):
    
    # Assigning a Call to a Name (line 119):
    
    # Call to solve_toeplitz(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'c' (line 119)
    c_110950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 31), 'c', False)
    # Processing the call keyword arguments (line 119)
    # Getting the type of 'y' (line 119)
    y_110951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 36), 'y', False)
    keyword_110952 = y_110951
    kwargs_110953 = {'b': keyword_110952}
    # Getting the type of 'solve_toeplitz' (line 119)
    solve_toeplitz_110949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'solve_toeplitz', False)
    # Calling solve_toeplitz(args, kwargs) (line 119)
    solve_toeplitz_call_result_110954 = invoke(stypy.reporting.localization.Localization(__file__, 119, 16), solve_toeplitz_110949, *[c_110950], **kwargs_110953)
    
    # Assigning a type to the variable 'solution1' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'solution1', solve_toeplitz_call_result_110954)
    
    # Assigning a Call to a Name (line 120):
    
    # Assigning a Call to a Name (line 120):
    
    # Call to solve(...): (line 120)
    # Processing the call arguments (line 120)
    
    # Call to toeplitz(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'c' (line 120)
    c_110957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 31), 'c', False)
    # Processing the call keyword arguments (line 120)
    kwargs_110958 = {}
    # Getting the type of 'toeplitz' (line 120)
    toeplitz_110956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 22), 'toeplitz', False)
    # Calling toeplitz(args, kwargs) (line 120)
    toeplitz_call_result_110959 = invoke(stypy.reporting.localization.Localization(__file__, 120, 22), toeplitz_110956, *[c_110957], **kwargs_110958)
    
    # Getting the type of 'y' (line 120)
    y_110960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 35), 'y', False)
    # Processing the call keyword arguments (line 120)
    kwargs_110961 = {}
    # Getting the type of 'solve' (line 120)
    solve_110955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'solve', False)
    # Calling solve(args, kwargs) (line 120)
    solve_call_result_110962 = invoke(stypy.reporting.localization.Localization(__file__, 120, 16), solve_110955, *[toeplitz_call_result_110959, y_110960], **kwargs_110961)
    
    # Assigning a type to the variable 'solution2' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'solution2', solve_call_result_110962)
    
    # Call to assert_allclose(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'solution1' (line 122)
    solution1_110964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 20), 'solution1', False)
    # Getting the type of 'solution2' (line 122)
    solution2_110965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 31), 'solution2', False)
    # Processing the call keyword arguments (line 122)
    kwargs_110966 = {}
    # Getting the type of 'assert_allclose' (line 122)
    assert_allclose_110963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 122)
    assert_allclose_call_result_110967 = invoke(stypy.reporting.localization.Localization(__file__, 122, 4), assert_allclose_110963, *[solution1_110964, solution2_110965], **kwargs_110966)
    
    
    # ################# End of 'test_unstable(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_unstable' in the type store
    # Getting the type of 'stypy_return_type' (line 105)
    stypy_return_type_110968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_110968)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_unstable'
    return stypy_return_type_110968

# Assigning a type to the variable 'test_unstable' (line 105)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'test_unstable', test_unstable)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
