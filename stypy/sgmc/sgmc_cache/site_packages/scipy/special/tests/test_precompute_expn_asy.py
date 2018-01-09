
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from numpy.testing import assert_equal
4: 
5: from scipy.special._testutils import check_version, MissingModule
6: from scipy.special._precompute.expn_asy import generate_A
7: 
8: try:
9:     import sympy
10:     from sympy import Poly
11: except ImportError:
12:     sympy = MissingModule("sympy")
13: 
14: 
15: @check_version(sympy, "1.0")
16: def test_generate_A():
17:     # Data from DLMF 8.20.5
18:     x = sympy.symbols('x')
19:     Astd = [Poly(1, x),
20:             Poly(1, x),
21:             Poly(1 - 2*x),
22:             Poly(1 - 8*x + 6*x**2)]
23:     Ares = generate_A(len(Astd))
24: 
25:     for p, q in zip(Astd, Ares):
26:         assert_equal(p, q)
27: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from numpy.testing import assert_equal' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559093 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing')

if (type(import_559093) is not StypyTypeError):

    if (import_559093 != 'pyd_module'):
        __import__(import_559093)
        sys_modules_559094 = sys.modules[import_559093]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', sys_modules_559094.module_type_store, module_type_store, ['assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_559094, sys_modules_559094.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', None, module_type_store, ['assert_equal'], [assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', import_559093)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.special._testutils import check_version, MissingModule' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559095 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special._testutils')

if (type(import_559095) is not StypyTypeError):

    if (import_559095 != 'pyd_module'):
        __import__(import_559095)
        sys_modules_559096 = sys.modules[import_559095]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special._testutils', sys_modules_559096.module_type_store, module_type_store, ['check_version', 'MissingModule'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_559096, sys_modules_559096.module_type_store, module_type_store)
    else:
        from scipy.special._testutils import check_version, MissingModule

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special._testutils', None, module_type_store, ['check_version', 'MissingModule'], [check_version, MissingModule])

else:
    # Assigning a type to the variable 'scipy.special._testutils' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special._testutils', import_559095)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.special._precompute.expn_asy import generate_A' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559097 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._precompute.expn_asy')

if (type(import_559097) is not StypyTypeError):

    if (import_559097 != 'pyd_module'):
        __import__(import_559097)
        sys_modules_559098 = sys.modules[import_559097]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._precompute.expn_asy', sys_modules_559098.module_type_store, module_type_store, ['generate_A'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_559098, sys_modules_559098.module_type_store, module_type_store)
    else:
        from scipy.special._precompute.expn_asy import generate_A

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._precompute.expn_asy', None, module_type_store, ['generate_A'], [generate_A])

else:
    # Assigning a type to the variable 'scipy.special._precompute.expn_asy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._precompute.expn_asy', import_559097)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')



# SSA begins for try-except statement (line 8)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 4))

# 'import sympy' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559099 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'sympy')

if (type(import_559099) is not StypyTypeError):

    if (import_559099 != 'pyd_module'):
        __import__(import_559099)
        sys_modules_559100 = sys.modules[import_559099]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'sympy', sys_modules_559100.module_type_store, module_type_store)
    else:
        import sympy

        import_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'sympy', sympy, module_type_store)

else:
    # Assigning a type to the variable 'sympy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'sympy', import_559099)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 4))

# 'from sympy import Poly' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559101 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'sympy')

if (type(import_559101) is not StypyTypeError):

    if (import_559101 != 'pyd_module'):
        __import__(import_559101)
        sys_modules_559102 = sys.modules[import_559101]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'sympy', sys_modules_559102.module_type_store, module_type_store, ['Poly'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 4), __file__, sys_modules_559102, sys_modules_559102.module_type_store, module_type_store)
    else:
        from sympy import Poly

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'sympy', None, module_type_store, ['Poly'], [Poly])

else:
    # Assigning a type to the variable 'sympy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'sympy', import_559101)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

# SSA branch for the except part of a try statement (line 8)
# SSA branch for the except 'ImportError' branch of a try statement (line 8)
module_type_store.open_ssa_branch('except')

# Assigning a Call to a Name (line 12):

# Call to MissingModule(...): (line 12)
# Processing the call arguments (line 12)
str_559104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 26), 'str', 'sympy')
# Processing the call keyword arguments (line 12)
kwargs_559105 = {}
# Getting the type of 'MissingModule' (line 12)
MissingModule_559103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'MissingModule', False)
# Calling MissingModule(args, kwargs) (line 12)
MissingModule_call_result_559106 = invoke(stypy.reporting.localization.Localization(__file__, 12, 12), MissingModule_559103, *[str_559104], **kwargs_559105)

# Assigning a type to the variable 'sympy' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'sympy', MissingModule_call_result_559106)
# SSA join for try-except statement (line 8)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def test_generate_A(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_generate_A'
    module_type_store = module_type_store.open_function_context('test_generate_A', 15, 0, False)
    
    # Passed parameters checking function
    test_generate_A.stypy_localization = localization
    test_generate_A.stypy_type_of_self = None
    test_generate_A.stypy_type_store = module_type_store
    test_generate_A.stypy_function_name = 'test_generate_A'
    test_generate_A.stypy_param_names_list = []
    test_generate_A.stypy_varargs_param_name = None
    test_generate_A.stypy_kwargs_param_name = None
    test_generate_A.stypy_call_defaults = defaults
    test_generate_A.stypy_call_varargs = varargs
    test_generate_A.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_generate_A', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_generate_A', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_generate_A(...)' code ##################

    
    # Assigning a Call to a Name (line 18):
    
    # Call to symbols(...): (line 18)
    # Processing the call arguments (line 18)
    str_559109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 22), 'str', 'x')
    # Processing the call keyword arguments (line 18)
    kwargs_559110 = {}
    # Getting the type of 'sympy' (line 18)
    sympy_559107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'sympy', False)
    # Obtaining the member 'symbols' of a type (line 18)
    symbols_559108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), sympy_559107, 'symbols')
    # Calling symbols(args, kwargs) (line 18)
    symbols_call_result_559111 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), symbols_559108, *[str_559109], **kwargs_559110)
    
    # Assigning a type to the variable 'x' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'x', symbols_call_result_559111)
    
    # Assigning a List to a Name (line 19):
    
    # Obtaining an instance of the builtin type 'list' (line 19)
    list_559112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 19)
    # Adding element type (line 19)
    
    # Call to Poly(...): (line 19)
    # Processing the call arguments (line 19)
    int_559114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 17), 'int')
    # Getting the type of 'x' (line 19)
    x_559115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'x', False)
    # Processing the call keyword arguments (line 19)
    kwargs_559116 = {}
    # Getting the type of 'Poly' (line 19)
    Poly_559113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'Poly', False)
    # Calling Poly(args, kwargs) (line 19)
    Poly_call_result_559117 = invoke(stypy.reporting.localization.Localization(__file__, 19, 12), Poly_559113, *[int_559114, x_559115], **kwargs_559116)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 11), list_559112, Poly_call_result_559117)
    # Adding element type (line 19)
    
    # Call to Poly(...): (line 20)
    # Processing the call arguments (line 20)
    int_559119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 17), 'int')
    # Getting the type of 'x' (line 20)
    x_559120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 20), 'x', False)
    # Processing the call keyword arguments (line 20)
    kwargs_559121 = {}
    # Getting the type of 'Poly' (line 20)
    Poly_559118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'Poly', False)
    # Calling Poly(args, kwargs) (line 20)
    Poly_call_result_559122 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), Poly_559118, *[int_559119, x_559120], **kwargs_559121)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 11), list_559112, Poly_call_result_559122)
    # Adding element type (line 19)
    
    # Call to Poly(...): (line 21)
    # Processing the call arguments (line 21)
    int_559124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 17), 'int')
    int_559125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 21), 'int')
    # Getting the type of 'x' (line 21)
    x_559126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 23), 'x', False)
    # Applying the binary operator '*' (line 21)
    result_mul_559127 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 21), '*', int_559125, x_559126)
    
    # Applying the binary operator '-' (line 21)
    result_sub_559128 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 17), '-', int_559124, result_mul_559127)
    
    # Processing the call keyword arguments (line 21)
    kwargs_559129 = {}
    # Getting the type of 'Poly' (line 21)
    Poly_559123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'Poly', False)
    # Calling Poly(args, kwargs) (line 21)
    Poly_call_result_559130 = invoke(stypy.reporting.localization.Localization(__file__, 21, 12), Poly_559123, *[result_sub_559128], **kwargs_559129)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 11), list_559112, Poly_call_result_559130)
    # Adding element type (line 19)
    
    # Call to Poly(...): (line 22)
    # Processing the call arguments (line 22)
    int_559132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 17), 'int')
    int_559133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 21), 'int')
    # Getting the type of 'x' (line 22)
    x_559134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 23), 'x', False)
    # Applying the binary operator '*' (line 22)
    result_mul_559135 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 21), '*', int_559133, x_559134)
    
    # Applying the binary operator '-' (line 22)
    result_sub_559136 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 17), '-', int_559132, result_mul_559135)
    
    int_559137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 27), 'int')
    # Getting the type of 'x' (line 22)
    x_559138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 29), 'x', False)
    int_559139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 32), 'int')
    # Applying the binary operator '**' (line 22)
    result_pow_559140 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 29), '**', x_559138, int_559139)
    
    # Applying the binary operator '*' (line 22)
    result_mul_559141 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 27), '*', int_559137, result_pow_559140)
    
    # Applying the binary operator '+' (line 22)
    result_add_559142 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 25), '+', result_sub_559136, result_mul_559141)
    
    # Processing the call keyword arguments (line 22)
    kwargs_559143 = {}
    # Getting the type of 'Poly' (line 22)
    Poly_559131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'Poly', False)
    # Calling Poly(args, kwargs) (line 22)
    Poly_call_result_559144 = invoke(stypy.reporting.localization.Localization(__file__, 22, 12), Poly_559131, *[result_add_559142], **kwargs_559143)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 11), list_559112, Poly_call_result_559144)
    
    # Assigning a type to the variable 'Astd' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'Astd', list_559112)
    
    # Assigning a Call to a Name (line 23):
    
    # Call to generate_A(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Call to len(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'Astd' (line 23)
    Astd_559147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 26), 'Astd', False)
    # Processing the call keyword arguments (line 23)
    kwargs_559148 = {}
    # Getting the type of 'len' (line 23)
    len_559146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 22), 'len', False)
    # Calling len(args, kwargs) (line 23)
    len_call_result_559149 = invoke(stypy.reporting.localization.Localization(__file__, 23, 22), len_559146, *[Astd_559147], **kwargs_559148)
    
    # Processing the call keyword arguments (line 23)
    kwargs_559150 = {}
    # Getting the type of 'generate_A' (line 23)
    generate_A_559145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), 'generate_A', False)
    # Calling generate_A(args, kwargs) (line 23)
    generate_A_call_result_559151 = invoke(stypy.reporting.localization.Localization(__file__, 23, 11), generate_A_559145, *[len_call_result_559149], **kwargs_559150)
    
    # Assigning a type to the variable 'Ares' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'Ares', generate_A_call_result_559151)
    
    
    # Call to zip(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'Astd' (line 25)
    Astd_559153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'Astd', False)
    # Getting the type of 'Ares' (line 25)
    Ares_559154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 26), 'Ares', False)
    # Processing the call keyword arguments (line 25)
    kwargs_559155 = {}
    # Getting the type of 'zip' (line 25)
    zip_559152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'zip', False)
    # Calling zip(args, kwargs) (line 25)
    zip_call_result_559156 = invoke(stypy.reporting.localization.Localization(__file__, 25, 16), zip_559152, *[Astd_559153, Ares_559154], **kwargs_559155)
    
    # Testing the type of a for loop iterable (line 25)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 25, 4), zip_call_result_559156)
    # Getting the type of the for loop variable (line 25)
    for_loop_var_559157 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 25, 4), zip_call_result_559156)
    # Assigning a type to the variable 'p' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'p', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 4), for_loop_var_559157))
    # Assigning a type to the variable 'q' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'q', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 4), for_loop_var_559157))
    # SSA begins for a for statement (line 25)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_equal(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'p' (line 26)
    p_559159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 21), 'p', False)
    # Getting the type of 'q' (line 26)
    q_559160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 24), 'q', False)
    # Processing the call keyword arguments (line 26)
    kwargs_559161 = {}
    # Getting the type of 'assert_equal' (line 26)
    assert_equal_559158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 26)
    assert_equal_call_result_559162 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), assert_equal_559158, *[p_559159, q_559160], **kwargs_559161)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_generate_A(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_generate_A' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_559163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_559163)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_generate_A'
    return stypy_return_type_559163

# Assigning a type to the variable 'test_generate_A' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'test_generate_A', test_generate_A)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
