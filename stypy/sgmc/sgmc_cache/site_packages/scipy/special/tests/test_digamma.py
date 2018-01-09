
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division
2: import numpy as np
3: from numpy import pi, log, sqrt
4: from scipy.special._testutils import FuncData
5: from scipy.special import digamma
6: 
7: # Euler-Mascheroni constant
8: euler = 0.57721566490153286
9: 
10: 
11: def test_consistency():
12:     # Make sure the implementation of digamma for real arguments
13:     # agrees with the implementation of digamma for complex arguments.
14: 
15:     # It's all poles after -1e16
16:     x = np.r_[-np.logspace(15, -30, 200), np.logspace(-30, 300, 200)]
17:     dataset = np.vstack((x + 0j, digamma(x))).T
18:     FuncData(digamma, dataset, 0, 1, rtol=5e-14, nan_ok=True).check()
19: 
20: 
21: def test_special_values():
22:     # Test special values from Gauss's digamma theorem. See
23:     #
24:     # https://en.wikipedia.org/wiki/Digamma_function
25: 
26:     dataset = [(1, -euler),
27:                (0.5, -2*log(2) - euler),
28:                (1/3, -pi/(2*sqrt(3)) - 3*log(3)/2 - euler),
29:                (1/4, -pi/2 - 3*log(2) - euler),
30:                (1/6, -pi*sqrt(3)/2 - 2*log(2) - 3*log(3)/2 - euler),
31:                (1/8, -pi/2 - 4*log(2) - (pi + log(2 + sqrt(2)) - log(2 - sqrt(2)))/sqrt(2) - euler)]
32: 
33:     dataset = np.asarray(dataset)
34:     FuncData(digamma, dataset, 0, 1, rtol=1e-14).check()
35: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import numpy' statement (line 2)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_537187 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy')

if (type(import_537187) is not StypyTypeError):

    if (import_537187 != 'pyd_module'):
        __import__(import_537187)
        sys_modules_537188 = sys.modules[import_537187]
        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'np', sys_modules_537188.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy', import_537187)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from numpy import pi, log, sqrt' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_537189 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_537189) is not StypyTypeError):

    if (import_537189 != 'pyd_module'):
        __import__(import_537189)
        sys_modules_537190 = sys.modules[import_537189]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', sys_modules_537190.module_type_store, module_type_store, ['pi', 'log', 'sqrt'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_537190, sys_modules_537190.module_type_store, module_type_store)
    else:
        from numpy import pi, log, sqrt

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', None, module_type_store, ['pi', 'log', 'sqrt'], [pi, log, sqrt])

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_537189)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from scipy.special._testutils import FuncData' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_537191 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.special._testutils')

if (type(import_537191) is not StypyTypeError):

    if (import_537191 != 'pyd_module'):
        __import__(import_537191)
        sys_modules_537192 = sys.modules[import_537191]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.special._testutils', sys_modules_537192.module_type_store, module_type_store, ['FuncData'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_537192, sys_modules_537192.module_type_store, module_type_store)
    else:
        from scipy.special._testutils import FuncData

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.special._testutils', None, module_type_store, ['FuncData'], [FuncData])

else:
    # Assigning a type to the variable 'scipy.special._testutils' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.special._testutils', import_537191)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.special import digamma' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_537193 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special')

if (type(import_537193) is not StypyTypeError):

    if (import_537193 != 'pyd_module'):
        __import__(import_537193)
        sys_modules_537194 = sys.modules[import_537193]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special', sys_modules_537194.module_type_store, module_type_store, ['digamma'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_537194, sys_modules_537194.module_type_store, module_type_store)
    else:
        from scipy.special import digamma

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special', None, module_type_store, ['digamma'], [digamma])

else:
    # Assigning a type to the variable 'scipy.special' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special', import_537193)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')


# Assigning a Num to a Name (line 8):
float_537195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 8), 'float')
# Assigning a type to the variable 'euler' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'euler', float_537195)

@norecursion
def test_consistency(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_consistency'
    module_type_store = module_type_store.open_function_context('test_consistency', 11, 0, False)
    
    # Passed parameters checking function
    test_consistency.stypy_localization = localization
    test_consistency.stypy_type_of_self = None
    test_consistency.stypy_type_store = module_type_store
    test_consistency.stypy_function_name = 'test_consistency'
    test_consistency.stypy_param_names_list = []
    test_consistency.stypy_varargs_param_name = None
    test_consistency.stypy_kwargs_param_name = None
    test_consistency.stypy_call_defaults = defaults
    test_consistency.stypy_call_varargs = varargs
    test_consistency.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_consistency', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_consistency', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_consistency(...)' code ##################

    
    # Assigning a Subscript to a Name (line 16):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 16)
    tuple_537196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 16)
    # Adding element type (line 16)
    
    
    # Call to logspace(...): (line 16)
    # Processing the call arguments (line 16)
    int_537199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 27), 'int')
    int_537200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 31), 'int')
    int_537201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 36), 'int')
    # Processing the call keyword arguments (line 16)
    kwargs_537202 = {}
    # Getting the type of 'np' (line 16)
    np_537197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 15), 'np', False)
    # Obtaining the member 'logspace' of a type (line 16)
    logspace_537198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 15), np_537197, 'logspace')
    # Calling logspace(args, kwargs) (line 16)
    logspace_call_result_537203 = invoke(stypy.reporting.localization.Localization(__file__, 16, 15), logspace_537198, *[int_537199, int_537200, int_537201], **kwargs_537202)
    
    # Applying the 'usub' unary operator (line 16)
    result___neg___537204 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 14), 'usub', logspace_call_result_537203)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 14), tuple_537196, result___neg___537204)
    # Adding element type (line 16)
    
    # Call to logspace(...): (line 16)
    # Processing the call arguments (line 16)
    int_537207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 54), 'int')
    int_537208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 59), 'int')
    int_537209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 64), 'int')
    # Processing the call keyword arguments (line 16)
    kwargs_537210 = {}
    # Getting the type of 'np' (line 16)
    np_537205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 42), 'np', False)
    # Obtaining the member 'logspace' of a type (line 16)
    logspace_537206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 42), np_537205, 'logspace')
    # Calling logspace(args, kwargs) (line 16)
    logspace_call_result_537211 = invoke(stypy.reporting.localization.Localization(__file__, 16, 42), logspace_537206, *[int_537207, int_537208, int_537209], **kwargs_537210)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 14), tuple_537196, logspace_call_result_537211)
    
    # Getting the type of 'np' (line 16)
    np_537212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'np')
    # Obtaining the member 'r_' of a type (line 16)
    r__537213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), np_537212, 'r_')
    # Obtaining the member '__getitem__' of a type (line 16)
    getitem___537214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), r__537213, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 16)
    subscript_call_result_537215 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), getitem___537214, tuple_537196)
    
    # Assigning a type to the variable 'x' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'x', subscript_call_result_537215)
    
    # Assigning a Attribute to a Name (line 17):
    
    # Call to vstack(...): (line 17)
    # Processing the call arguments (line 17)
    
    # Obtaining an instance of the builtin type 'tuple' (line 17)
    tuple_537218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 17)
    # Adding element type (line 17)
    # Getting the type of 'x' (line 17)
    x_537219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 25), 'x', False)
    complex_537220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 29), 'complex')
    # Applying the binary operator '+' (line 17)
    result_add_537221 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 25), '+', x_537219, complex_537220)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 25), tuple_537218, result_add_537221)
    # Adding element type (line 17)
    
    # Call to digamma(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'x' (line 17)
    x_537223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 41), 'x', False)
    # Processing the call keyword arguments (line 17)
    kwargs_537224 = {}
    # Getting the type of 'digamma' (line 17)
    digamma_537222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 33), 'digamma', False)
    # Calling digamma(args, kwargs) (line 17)
    digamma_call_result_537225 = invoke(stypy.reporting.localization.Localization(__file__, 17, 33), digamma_537222, *[x_537223], **kwargs_537224)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 25), tuple_537218, digamma_call_result_537225)
    
    # Processing the call keyword arguments (line 17)
    kwargs_537226 = {}
    # Getting the type of 'np' (line 17)
    np_537216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 14), 'np', False)
    # Obtaining the member 'vstack' of a type (line 17)
    vstack_537217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 14), np_537216, 'vstack')
    # Calling vstack(args, kwargs) (line 17)
    vstack_call_result_537227 = invoke(stypy.reporting.localization.Localization(__file__, 17, 14), vstack_537217, *[tuple_537218], **kwargs_537226)
    
    # Obtaining the member 'T' of a type (line 17)
    T_537228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 14), vstack_call_result_537227, 'T')
    # Assigning a type to the variable 'dataset' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'dataset', T_537228)
    
    # Call to check(...): (line 18)
    # Processing the call keyword arguments (line 18)
    kwargs_537241 = {}
    
    # Call to FuncData(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'digamma' (line 18)
    digamma_537230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 13), 'digamma', False)
    # Getting the type of 'dataset' (line 18)
    dataset_537231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 22), 'dataset', False)
    int_537232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 31), 'int')
    int_537233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 34), 'int')
    # Processing the call keyword arguments (line 18)
    float_537234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 42), 'float')
    keyword_537235 = float_537234
    # Getting the type of 'True' (line 18)
    True_537236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 56), 'True', False)
    keyword_537237 = True_537236
    kwargs_537238 = {'rtol': keyword_537235, 'nan_ok': keyword_537237}
    # Getting the type of 'FuncData' (line 18)
    FuncData_537229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'FuncData', False)
    # Calling FuncData(args, kwargs) (line 18)
    FuncData_call_result_537239 = invoke(stypy.reporting.localization.Localization(__file__, 18, 4), FuncData_537229, *[digamma_537230, dataset_537231, int_537232, int_537233], **kwargs_537238)
    
    # Obtaining the member 'check' of a type (line 18)
    check_537240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 4), FuncData_call_result_537239, 'check')
    # Calling check(args, kwargs) (line 18)
    check_call_result_537242 = invoke(stypy.reporting.localization.Localization(__file__, 18, 4), check_537240, *[], **kwargs_537241)
    
    
    # ################# End of 'test_consistency(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_consistency' in the type store
    # Getting the type of 'stypy_return_type' (line 11)
    stypy_return_type_537243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_537243)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_consistency'
    return stypy_return_type_537243

# Assigning a type to the variable 'test_consistency' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'test_consistency', test_consistency)

@norecursion
def test_special_values(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_special_values'
    module_type_store = module_type_store.open_function_context('test_special_values', 21, 0, False)
    
    # Passed parameters checking function
    test_special_values.stypy_localization = localization
    test_special_values.stypy_type_of_self = None
    test_special_values.stypy_type_store = module_type_store
    test_special_values.stypy_function_name = 'test_special_values'
    test_special_values.stypy_param_names_list = []
    test_special_values.stypy_varargs_param_name = None
    test_special_values.stypy_kwargs_param_name = None
    test_special_values.stypy_call_defaults = defaults
    test_special_values.stypy_call_varargs = varargs
    test_special_values.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_special_values', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_special_values', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_special_values(...)' code ##################

    
    # Assigning a List to a Name (line 26):
    
    # Obtaining an instance of the builtin type 'list' (line 26)
    list_537244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 26)
    # Adding element type (line 26)
    
    # Obtaining an instance of the builtin type 'tuple' (line 26)
    tuple_537245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 26)
    # Adding element type (line 26)
    int_537246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 16), tuple_537245, int_537246)
    # Adding element type (line 26)
    
    # Getting the type of 'euler' (line 26)
    euler_537247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 20), 'euler')
    # Applying the 'usub' unary operator (line 26)
    result___neg___537248 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 19), 'usub', euler_537247)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 16), tuple_537245, result___neg___537248)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 14), list_537244, tuple_537245)
    # Adding element type (line 26)
    
    # Obtaining an instance of the builtin type 'tuple' (line 27)
    tuple_537249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 27)
    # Adding element type (line 27)
    float_537250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 16), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 16), tuple_537249, float_537250)
    # Adding element type (line 27)
    int_537251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 21), 'int')
    
    # Call to log(...): (line 27)
    # Processing the call arguments (line 27)
    int_537253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 28), 'int')
    # Processing the call keyword arguments (line 27)
    kwargs_537254 = {}
    # Getting the type of 'log' (line 27)
    log_537252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 24), 'log', False)
    # Calling log(args, kwargs) (line 27)
    log_call_result_537255 = invoke(stypy.reporting.localization.Localization(__file__, 27, 24), log_537252, *[int_537253], **kwargs_537254)
    
    # Applying the binary operator '*' (line 27)
    result_mul_537256 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 21), '*', int_537251, log_call_result_537255)
    
    # Getting the type of 'euler' (line 27)
    euler_537257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 33), 'euler')
    # Applying the binary operator '-' (line 27)
    result_sub_537258 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 21), '-', result_mul_537256, euler_537257)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 16), tuple_537249, result_sub_537258)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 14), list_537244, tuple_537249)
    # Adding element type (line 26)
    
    # Obtaining an instance of the builtin type 'tuple' (line 28)
    tuple_537259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 28)
    # Adding element type (line 28)
    int_537260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 16), 'int')
    int_537261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 18), 'int')
    # Applying the binary operator 'div' (line 28)
    result_div_537262 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 16), 'div', int_537260, int_537261)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 16), tuple_537259, result_div_537262)
    # Adding element type (line 28)
    
    # Getting the type of 'pi' (line 28)
    pi_537263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 22), 'pi')
    # Applying the 'usub' unary operator (line 28)
    result___neg___537264 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 21), 'usub', pi_537263)
    
    int_537265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 26), 'int')
    
    # Call to sqrt(...): (line 28)
    # Processing the call arguments (line 28)
    int_537267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 33), 'int')
    # Processing the call keyword arguments (line 28)
    kwargs_537268 = {}
    # Getting the type of 'sqrt' (line 28)
    sqrt_537266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 28), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 28)
    sqrt_call_result_537269 = invoke(stypy.reporting.localization.Localization(__file__, 28, 28), sqrt_537266, *[int_537267], **kwargs_537268)
    
    # Applying the binary operator '*' (line 28)
    result_mul_537270 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 26), '*', int_537265, sqrt_call_result_537269)
    
    # Applying the binary operator 'div' (line 28)
    result_div_537271 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 21), 'div', result___neg___537264, result_mul_537270)
    
    int_537272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 39), 'int')
    
    # Call to log(...): (line 28)
    # Processing the call arguments (line 28)
    int_537274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 45), 'int')
    # Processing the call keyword arguments (line 28)
    kwargs_537275 = {}
    # Getting the type of 'log' (line 28)
    log_537273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 41), 'log', False)
    # Calling log(args, kwargs) (line 28)
    log_call_result_537276 = invoke(stypy.reporting.localization.Localization(__file__, 28, 41), log_537273, *[int_537274], **kwargs_537275)
    
    # Applying the binary operator '*' (line 28)
    result_mul_537277 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 39), '*', int_537272, log_call_result_537276)
    
    int_537278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 48), 'int')
    # Applying the binary operator 'div' (line 28)
    result_div_537279 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 47), 'div', result_mul_537277, int_537278)
    
    # Applying the binary operator '-' (line 28)
    result_sub_537280 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 21), '-', result_div_537271, result_div_537279)
    
    # Getting the type of 'euler' (line 28)
    euler_537281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 52), 'euler')
    # Applying the binary operator '-' (line 28)
    result_sub_537282 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 50), '-', result_sub_537280, euler_537281)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 16), tuple_537259, result_sub_537282)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 14), list_537244, tuple_537259)
    # Adding element type (line 26)
    
    # Obtaining an instance of the builtin type 'tuple' (line 29)
    tuple_537283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 29)
    # Adding element type (line 29)
    int_537284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 16), 'int')
    int_537285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 18), 'int')
    # Applying the binary operator 'div' (line 29)
    result_div_537286 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 16), 'div', int_537284, int_537285)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 16), tuple_537283, result_div_537286)
    # Adding element type (line 29)
    
    # Getting the type of 'pi' (line 29)
    pi_537287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 22), 'pi')
    # Applying the 'usub' unary operator (line 29)
    result___neg___537288 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 21), 'usub', pi_537287)
    
    int_537289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 25), 'int')
    # Applying the binary operator 'div' (line 29)
    result_div_537290 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 21), 'div', result___neg___537288, int_537289)
    
    int_537291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 29), 'int')
    
    # Call to log(...): (line 29)
    # Processing the call arguments (line 29)
    int_537293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 35), 'int')
    # Processing the call keyword arguments (line 29)
    kwargs_537294 = {}
    # Getting the type of 'log' (line 29)
    log_537292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 31), 'log', False)
    # Calling log(args, kwargs) (line 29)
    log_call_result_537295 = invoke(stypy.reporting.localization.Localization(__file__, 29, 31), log_537292, *[int_537293], **kwargs_537294)
    
    # Applying the binary operator '*' (line 29)
    result_mul_537296 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 29), '*', int_537291, log_call_result_537295)
    
    # Applying the binary operator '-' (line 29)
    result_sub_537297 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 21), '-', result_div_537290, result_mul_537296)
    
    # Getting the type of 'euler' (line 29)
    euler_537298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 40), 'euler')
    # Applying the binary operator '-' (line 29)
    result_sub_537299 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 38), '-', result_sub_537297, euler_537298)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 16), tuple_537283, result_sub_537299)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 14), list_537244, tuple_537283)
    # Adding element type (line 26)
    
    # Obtaining an instance of the builtin type 'tuple' (line 30)
    tuple_537300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 30)
    # Adding element type (line 30)
    int_537301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 16), 'int')
    int_537302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 18), 'int')
    # Applying the binary operator 'div' (line 30)
    result_div_537303 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 16), 'div', int_537301, int_537302)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 16), tuple_537300, result_div_537303)
    # Adding element type (line 30)
    
    # Getting the type of 'pi' (line 30)
    pi_537304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 22), 'pi')
    # Applying the 'usub' unary operator (line 30)
    result___neg___537305 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 21), 'usub', pi_537304)
    
    
    # Call to sqrt(...): (line 30)
    # Processing the call arguments (line 30)
    int_537307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 30), 'int')
    # Processing the call keyword arguments (line 30)
    kwargs_537308 = {}
    # Getting the type of 'sqrt' (line 30)
    sqrt_537306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 25), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 30)
    sqrt_call_result_537309 = invoke(stypy.reporting.localization.Localization(__file__, 30, 25), sqrt_537306, *[int_537307], **kwargs_537308)
    
    # Applying the binary operator '*' (line 30)
    result_mul_537310 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 21), '*', result___neg___537305, sqrt_call_result_537309)
    
    int_537311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 33), 'int')
    # Applying the binary operator 'div' (line 30)
    result_div_537312 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 32), 'div', result_mul_537310, int_537311)
    
    int_537313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 37), 'int')
    
    # Call to log(...): (line 30)
    # Processing the call arguments (line 30)
    int_537315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 43), 'int')
    # Processing the call keyword arguments (line 30)
    kwargs_537316 = {}
    # Getting the type of 'log' (line 30)
    log_537314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 39), 'log', False)
    # Calling log(args, kwargs) (line 30)
    log_call_result_537317 = invoke(stypy.reporting.localization.Localization(__file__, 30, 39), log_537314, *[int_537315], **kwargs_537316)
    
    # Applying the binary operator '*' (line 30)
    result_mul_537318 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 37), '*', int_537313, log_call_result_537317)
    
    # Applying the binary operator '-' (line 30)
    result_sub_537319 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 21), '-', result_div_537312, result_mul_537318)
    
    int_537320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 48), 'int')
    
    # Call to log(...): (line 30)
    # Processing the call arguments (line 30)
    int_537322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 54), 'int')
    # Processing the call keyword arguments (line 30)
    kwargs_537323 = {}
    # Getting the type of 'log' (line 30)
    log_537321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 50), 'log', False)
    # Calling log(args, kwargs) (line 30)
    log_call_result_537324 = invoke(stypy.reporting.localization.Localization(__file__, 30, 50), log_537321, *[int_537322], **kwargs_537323)
    
    # Applying the binary operator '*' (line 30)
    result_mul_537325 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 48), '*', int_537320, log_call_result_537324)
    
    int_537326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 57), 'int')
    # Applying the binary operator 'div' (line 30)
    result_div_537327 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 56), 'div', result_mul_537325, int_537326)
    
    # Applying the binary operator '-' (line 30)
    result_sub_537328 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 46), '-', result_sub_537319, result_div_537327)
    
    # Getting the type of 'euler' (line 30)
    euler_537329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 61), 'euler')
    # Applying the binary operator '-' (line 30)
    result_sub_537330 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 59), '-', result_sub_537328, euler_537329)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 16), tuple_537300, result_sub_537330)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 14), list_537244, tuple_537300)
    # Adding element type (line 26)
    
    # Obtaining an instance of the builtin type 'tuple' (line 31)
    tuple_537331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 31)
    # Adding element type (line 31)
    int_537332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 16), 'int')
    int_537333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 18), 'int')
    # Applying the binary operator 'div' (line 31)
    result_div_537334 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 16), 'div', int_537332, int_537333)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 16), tuple_537331, result_div_537334)
    # Adding element type (line 31)
    
    # Getting the type of 'pi' (line 31)
    pi_537335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 22), 'pi')
    # Applying the 'usub' unary operator (line 31)
    result___neg___537336 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 21), 'usub', pi_537335)
    
    int_537337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 25), 'int')
    # Applying the binary operator 'div' (line 31)
    result_div_537338 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 21), 'div', result___neg___537336, int_537337)
    
    int_537339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 29), 'int')
    
    # Call to log(...): (line 31)
    # Processing the call arguments (line 31)
    int_537341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 35), 'int')
    # Processing the call keyword arguments (line 31)
    kwargs_537342 = {}
    # Getting the type of 'log' (line 31)
    log_537340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 31), 'log', False)
    # Calling log(args, kwargs) (line 31)
    log_call_result_537343 = invoke(stypy.reporting.localization.Localization(__file__, 31, 31), log_537340, *[int_537341], **kwargs_537342)
    
    # Applying the binary operator '*' (line 31)
    result_mul_537344 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 29), '*', int_537339, log_call_result_537343)
    
    # Applying the binary operator '-' (line 31)
    result_sub_537345 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 21), '-', result_div_537338, result_mul_537344)
    
    # Getting the type of 'pi' (line 31)
    pi_537346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 41), 'pi')
    
    # Call to log(...): (line 31)
    # Processing the call arguments (line 31)
    int_537348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 50), 'int')
    
    # Call to sqrt(...): (line 31)
    # Processing the call arguments (line 31)
    int_537350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 59), 'int')
    # Processing the call keyword arguments (line 31)
    kwargs_537351 = {}
    # Getting the type of 'sqrt' (line 31)
    sqrt_537349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 54), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 31)
    sqrt_call_result_537352 = invoke(stypy.reporting.localization.Localization(__file__, 31, 54), sqrt_537349, *[int_537350], **kwargs_537351)
    
    # Applying the binary operator '+' (line 31)
    result_add_537353 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 50), '+', int_537348, sqrt_call_result_537352)
    
    # Processing the call keyword arguments (line 31)
    kwargs_537354 = {}
    # Getting the type of 'log' (line 31)
    log_537347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 46), 'log', False)
    # Calling log(args, kwargs) (line 31)
    log_call_result_537355 = invoke(stypy.reporting.localization.Localization(__file__, 31, 46), log_537347, *[result_add_537353], **kwargs_537354)
    
    # Applying the binary operator '+' (line 31)
    result_add_537356 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 41), '+', pi_537346, log_call_result_537355)
    
    
    # Call to log(...): (line 31)
    # Processing the call arguments (line 31)
    int_537358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 69), 'int')
    
    # Call to sqrt(...): (line 31)
    # Processing the call arguments (line 31)
    int_537360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 78), 'int')
    # Processing the call keyword arguments (line 31)
    kwargs_537361 = {}
    # Getting the type of 'sqrt' (line 31)
    sqrt_537359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 73), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 31)
    sqrt_call_result_537362 = invoke(stypy.reporting.localization.Localization(__file__, 31, 73), sqrt_537359, *[int_537360], **kwargs_537361)
    
    # Applying the binary operator '-' (line 31)
    result_sub_537363 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 69), '-', int_537358, sqrt_call_result_537362)
    
    # Processing the call keyword arguments (line 31)
    kwargs_537364 = {}
    # Getting the type of 'log' (line 31)
    log_537357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 65), 'log', False)
    # Calling log(args, kwargs) (line 31)
    log_call_result_537365 = invoke(stypy.reporting.localization.Localization(__file__, 31, 65), log_537357, *[result_sub_537363], **kwargs_537364)
    
    # Applying the binary operator '-' (line 31)
    result_sub_537366 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 63), '-', result_add_537356, log_call_result_537365)
    
    
    # Call to sqrt(...): (line 31)
    # Processing the call arguments (line 31)
    int_537368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 88), 'int')
    # Processing the call keyword arguments (line 31)
    kwargs_537369 = {}
    # Getting the type of 'sqrt' (line 31)
    sqrt_537367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 83), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 31)
    sqrt_call_result_537370 = invoke(stypy.reporting.localization.Localization(__file__, 31, 83), sqrt_537367, *[int_537368], **kwargs_537369)
    
    # Applying the binary operator 'div' (line 31)
    result_div_537371 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 40), 'div', result_sub_537366, sqrt_call_result_537370)
    
    # Applying the binary operator '-' (line 31)
    result_sub_537372 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 38), '-', result_sub_537345, result_div_537371)
    
    # Getting the type of 'euler' (line 31)
    euler_537373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 93), 'euler')
    # Applying the binary operator '-' (line 31)
    result_sub_537374 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 91), '-', result_sub_537372, euler_537373)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 16), tuple_537331, result_sub_537374)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 14), list_537244, tuple_537331)
    
    # Assigning a type to the variable 'dataset' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'dataset', list_537244)
    
    # Assigning a Call to a Name (line 33):
    
    # Call to asarray(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'dataset' (line 33)
    dataset_537377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 25), 'dataset', False)
    # Processing the call keyword arguments (line 33)
    kwargs_537378 = {}
    # Getting the type of 'np' (line 33)
    np_537375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 14), 'np', False)
    # Obtaining the member 'asarray' of a type (line 33)
    asarray_537376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 14), np_537375, 'asarray')
    # Calling asarray(args, kwargs) (line 33)
    asarray_call_result_537379 = invoke(stypy.reporting.localization.Localization(__file__, 33, 14), asarray_537376, *[dataset_537377], **kwargs_537378)
    
    # Assigning a type to the variable 'dataset' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'dataset', asarray_call_result_537379)
    
    # Call to check(...): (line 34)
    # Processing the call keyword arguments (line 34)
    kwargs_537390 = {}
    
    # Call to FuncData(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'digamma' (line 34)
    digamma_537381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 13), 'digamma', False)
    # Getting the type of 'dataset' (line 34)
    dataset_537382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 22), 'dataset', False)
    int_537383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 31), 'int')
    int_537384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 34), 'int')
    # Processing the call keyword arguments (line 34)
    float_537385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 42), 'float')
    keyword_537386 = float_537385
    kwargs_537387 = {'rtol': keyword_537386}
    # Getting the type of 'FuncData' (line 34)
    FuncData_537380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'FuncData', False)
    # Calling FuncData(args, kwargs) (line 34)
    FuncData_call_result_537388 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), FuncData_537380, *[digamma_537381, dataset_537382, int_537383, int_537384], **kwargs_537387)
    
    # Obtaining the member 'check' of a type (line 34)
    check_537389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 4), FuncData_call_result_537388, 'check')
    # Calling check(args, kwargs) (line 34)
    check_call_result_537391 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), check_537389, *[], **kwargs_537390)
    
    
    # ################# End of 'test_special_values(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_special_values' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_537392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_537392)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_special_values'
    return stypy_return_type_537392

# Assigning a type to the variable 'test_special_values' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'test_special_values', test_special_values)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
