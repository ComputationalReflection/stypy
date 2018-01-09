
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: 
5: import scipy.special as sc
6: from scipy.special._testutils import FuncData
7: 
8: 
9: def test_sici_consistency():
10:     # Make sure the implementation of sici for real arguments agrees
11:     # with the implementation of sici for complex arguments.
12: 
13:     # On the negative real axis Cephes drops the imaginary part in ci
14:     def sici(x):
15:         si, ci = sc.sici(x + 0j)
16:         return si.real, ci.real
17:     
18:     x = np.r_[-np.logspace(8, -30, 200), 0, np.logspace(-30, 8, 200)]
19:     si, ci = sc.sici(x)
20:     dataset = np.column_stack((x, si, ci))
21:     FuncData(sici, dataset, 0, (1, 2), rtol=1e-12).check()
22: 
23: 
24: def test_shichi_consistency():
25:     # Make sure the implementation of shichi for real arguments agrees
26:     # with the implementation of shichi for complex arguments.
27: 
28:     # On the negative real axis Cephes drops the imaginary part in chi
29:     def shichi(x):
30:         shi, chi = sc.shichi(x + 0j)
31:         return shi.real, chi.real
32: 
33:     # Overflow happens quickly, so limit range
34:     x = np.r_[-np.logspace(np.log10(700), -30, 200), 0,
35:               np.logspace(-30, np.log10(700), 200)]
36:     shi, chi = sc.shichi(x)
37:     dataset = np.column_stack((x, shi, chi))
38:     FuncData(shichi, dataset, 0, (1, 2), rtol=1e-14).check()
39: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_560295 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_560295) is not StypyTypeError):

    if (import_560295 != 'pyd_module'):
        __import__(import_560295)
        sys_modules_560296 = sys.modules[import_560295]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_560296.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_560295)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import scipy.special' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_560297 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special')

if (type(import_560297) is not StypyTypeError):

    if (import_560297 != 'pyd_module'):
        __import__(import_560297)
        sys_modules_560298 = sys.modules[import_560297]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sc', sys_modules_560298.module_type_store, module_type_store)
    else:
        import scipy.special as sc

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sc', scipy.special, module_type_store)

else:
    # Assigning a type to the variable 'scipy.special' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special', import_560297)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.special._testutils import FuncData' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_560299 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._testutils')

if (type(import_560299) is not StypyTypeError):

    if (import_560299 != 'pyd_module'):
        __import__(import_560299)
        sys_modules_560300 = sys.modules[import_560299]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._testutils', sys_modules_560300.module_type_store, module_type_store, ['FuncData'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_560300, sys_modules_560300.module_type_store, module_type_store)
    else:
        from scipy.special._testutils import FuncData

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._testutils', None, module_type_store, ['FuncData'], [FuncData])

else:
    # Assigning a type to the variable 'scipy.special._testutils' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._testutils', import_560299)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')


@norecursion
def test_sici_consistency(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_sici_consistency'
    module_type_store = module_type_store.open_function_context('test_sici_consistency', 9, 0, False)
    
    # Passed parameters checking function
    test_sici_consistency.stypy_localization = localization
    test_sici_consistency.stypy_type_of_self = None
    test_sici_consistency.stypy_type_store = module_type_store
    test_sici_consistency.stypy_function_name = 'test_sici_consistency'
    test_sici_consistency.stypy_param_names_list = []
    test_sici_consistency.stypy_varargs_param_name = None
    test_sici_consistency.stypy_kwargs_param_name = None
    test_sici_consistency.stypy_call_defaults = defaults
    test_sici_consistency.stypy_call_varargs = varargs
    test_sici_consistency.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_sici_consistency', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_sici_consistency', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_sici_consistency(...)' code ##################


    @norecursion
    def sici(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sici'
        module_type_store = module_type_store.open_function_context('sici', 14, 4, False)
        
        # Passed parameters checking function
        sici.stypy_localization = localization
        sici.stypy_type_of_self = None
        sici.stypy_type_store = module_type_store
        sici.stypy_function_name = 'sici'
        sici.stypy_param_names_list = ['x']
        sici.stypy_varargs_param_name = None
        sici.stypy_kwargs_param_name = None
        sici.stypy_call_defaults = defaults
        sici.stypy_call_varargs = varargs
        sici.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'sici', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sici', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sici(...)' code ##################

        
        # Assigning a Call to a Tuple (line 15):
        
        # Assigning a Subscript to a Name (line 15):
        
        # Obtaining the type of the subscript
        int_560301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 8), 'int')
        
        # Call to sici(...): (line 15)
        # Processing the call arguments (line 15)
        # Getting the type of 'x' (line 15)
        x_560304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 25), 'x', False)
        complex_560305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 29), 'complex')
        # Applying the binary operator '+' (line 15)
        result_add_560306 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 25), '+', x_560304, complex_560305)
        
        # Processing the call keyword arguments (line 15)
        kwargs_560307 = {}
        # Getting the type of 'sc' (line 15)
        sc_560302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 17), 'sc', False)
        # Obtaining the member 'sici' of a type (line 15)
        sici_560303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 17), sc_560302, 'sici')
        # Calling sici(args, kwargs) (line 15)
        sici_call_result_560308 = invoke(stypy.reporting.localization.Localization(__file__, 15, 17), sici_560303, *[result_add_560306], **kwargs_560307)
        
        # Obtaining the member '__getitem__' of a type (line 15)
        getitem___560309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), sici_call_result_560308, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 15)
        subscript_call_result_560310 = invoke(stypy.reporting.localization.Localization(__file__, 15, 8), getitem___560309, int_560301)
        
        # Assigning a type to the variable 'tuple_var_assignment_560287' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'tuple_var_assignment_560287', subscript_call_result_560310)
        
        # Assigning a Subscript to a Name (line 15):
        
        # Obtaining the type of the subscript
        int_560311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 8), 'int')
        
        # Call to sici(...): (line 15)
        # Processing the call arguments (line 15)
        # Getting the type of 'x' (line 15)
        x_560314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 25), 'x', False)
        complex_560315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 29), 'complex')
        # Applying the binary operator '+' (line 15)
        result_add_560316 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 25), '+', x_560314, complex_560315)
        
        # Processing the call keyword arguments (line 15)
        kwargs_560317 = {}
        # Getting the type of 'sc' (line 15)
        sc_560312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 17), 'sc', False)
        # Obtaining the member 'sici' of a type (line 15)
        sici_560313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 17), sc_560312, 'sici')
        # Calling sici(args, kwargs) (line 15)
        sici_call_result_560318 = invoke(stypy.reporting.localization.Localization(__file__, 15, 17), sici_560313, *[result_add_560316], **kwargs_560317)
        
        # Obtaining the member '__getitem__' of a type (line 15)
        getitem___560319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), sici_call_result_560318, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 15)
        subscript_call_result_560320 = invoke(stypy.reporting.localization.Localization(__file__, 15, 8), getitem___560319, int_560311)
        
        # Assigning a type to the variable 'tuple_var_assignment_560288' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'tuple_var_assignment_560288', subscript_call_result_560320)
        
        # Assigning a Name to a Name (line 15):
        # Getting the type of 'tuple_var_assignment_560287' (line 15)
        tuple_var_assignment_560287_560321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'tuple_var_assignment_560287')
        # Assigning a type to the variable 'si' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'si', tuple_var_assignment_560287_560321)
        
        # Assigning a Name to a Name (line 15):
        # Getting the type of 'tuple_var_assignment_560288' (line 15)
        tuple_var_assignment_560288_560322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'tuple_var_assignment_560288')
        # Assigning a type to the variable 'ci' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'ci', tuple_var_assignment_560288_560322)
        
        # Obtaining an instance of the builtin type 'tuple' (line 16)
        tuple_560323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 16)
        # Adding element type (line 16)
        # Getting the type of 'si' (line 16)
        si_560324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 15), 'si')
        # Obtaining the member 'real' of a type (line 16)
        real_560325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 15), si_560324, 'real')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 15), tuple_560323, real_560325)
        # Adding element type (line 16)
        # Getting the type of 'ci' (line 16)
        ci_560326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 24), 'ci')
        # Obtaining the member 'real' of a type (line 16)
        real_560327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 24), ci_560326, 'real')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 15), tuple_560323, real_560327)
        
        # Assigning a type to the variable 'stypy_return_type' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'stypy_return_type', tuple_560323)
        
        # ################# End of 'sici(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sici' in the type store
        # Getting the type of 'stypy_return_type' (line 14)
        stypy_return_type_560328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_560328)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sici'
        return stypy_return_type_560328

    # Assigning a type to the variable 'sici' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'sici', sici)
    
    # Assigning a Subscript to a Name (line 18):
    
    # Assigning a Subscript to a Name (line 18):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 18)
    tuple_560329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 18)
    # Adding element type (line 18)
    
    
    # Call to logspace(...): (line 18)
    # Processing the call arguments (line 18)
    int_560332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 27), 'int')
    int_560333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 30), 'int')
    int_560334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 35), 'int')
    # Processing the call keyword arguments (line 18)
    kwargs_560335 = {}
    # Getting the type of 'np' (line 18)
    np_560330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 15), 'np', False)
    # Obtaining the member 'logspace' of a type (line 18)
    logspace_560331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 15), np_560330, 'logspace')
    # Calling logspace(args, kwargs) (line 18)
    logspace_call_result_560336 = invoke(stypy.reporting.localization.Localization(__file__, 18, 15), logspace_560331, *[int_560332, int_560333, int_560334], **kwargs_560335)
    
    # Applying the 'usub' unary operator (line 18)
    result___neg___560337 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 14), 'usub', logspace_call_result_560336)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 14), tuple_560329, result___neg___560337)
    # Adding element type (line 18)
    int_560338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 14), tuple_560329, int_560338)
    # Adding element type (line 18)
    
    # Call to logspace(...): (line 18)
    # Processing the call arguments (line 18)
    int_560341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 56), 'int')
    int_560342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 61), 'int')
    int_560343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 64), 'int')
    # Processing the call keyword arguments (line 18)
    kwargs_560344 = {}
    # Getting the type of 'np' (line 18)
    np_560339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 44), 'np', False)
    # Obtaining the member 'logspace' of a type (line 18)
    logspace_560340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 44), np_560339, 'logspace')
    # Calling logspace(args, kwargs) (line 18)
    logspace_call_result_560345 = invoke(stypy.reporting.localization.Localization(__file__, 18, 44), logspace_560340, *[int_560341, int_560342, int_560343], **kwargs_560344)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 14), tuple_560329, logspace_call_result_560345)
    
    # Getting the type of 'np' (line 18)
    np_560346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'np')
    # Obtaining the member 'r_' of a type (line 18)
    r__560347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), np_560346, 'r_')
    # Obtaining the member '__getitem__' of a type (line 18)
    getitem___560348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), r__560347, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 18)
    subscript_call_result_560349 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), getitem___560348, tuple_560329)
    
    # Assigning a type to the variable 'x' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'x', subscript_call_result_560349)
    
    # Assigning a Call to a Tuple (line 19):
    
    # Assigning a Subscript to a Name (line 19):
    
    # Obtaining the type of the subscript
    int_560350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 4), 'int')
    
    # Call to sici(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'x' (line 19)
    x_560353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 21), 'x', False)
    # Processing the call keyword arguments (line 19)
    kwargs_560354 = {}
    # Getting the type of 'sc' (line 19)
    sc_560351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 13), 'sc', False)
    # Obtaining the member 'sici' of a type (line 19)
    sici_560352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 13), sc_560351, 'sici')
    # Calling sici(args, kwargs) (line 19)
    sici_call_result_560355 = invoke(stypy.reporting.localization.Localization(__file__, 19, 13), sici_560352, *[x_560353], **kwargs_560354)
    
    # Obtaining the member '__getitem__' of a type (line 19)
    getitem___560356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 4), sici_call_result_560355, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 19)
    subscript_call_result_560357 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), getitem___560356, int_560350)
    
    # Assigning a type to the variable 'tuple_var_assignment_560289' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'tuple_var_assignment_560289', subscript_call_result_560357)
    
    # Assigning a Subscript to a Name (line 19):
    
    # Obtaining the type of the subscript
    int_560358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 4), 'int')
    
    # Call to sici(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'x' (line 19)
    x_560361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 21), 'x', False)
    # Processing the call keyword arguments (line 19)
    kwargs_560362 = {}
    # Getting the type of 'sc' (line 19)
    sc_560359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 13), 'sc', False)
    # Obtaining the member 'sici' of a type (line 19)
    sici_560360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 13), sc_560359, 'sici')
    # Calling sici(args, kwargs) (line 19)
    sici_call_result_560363 = invoke(stypy.reporting.localization.Localization(__file__, 19, 13), sici_560360, *[x_560361], **kwargs_560362)
    
    # Obtaining the member '__getitem__' of a type (line 19)
    getitem___560364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 4), sici_call_result_560363, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 19)
    subscript_call_result_560365 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), getitem___560364, int_560358)
    
    # Assigning a type to the variable 'tuple_var_assignment_560290' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'tuple_var_assignment_560290', subscript_call_result_560365)
    
    # Assigning a Name to a Name (line 19):
    # Getting the type of 'tuple_var_assignment_560289' (line 19)
    tuple_var_assignment_560289_560366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'tuple_var_assignment_560289')
    # Assigning a type to the variable 'si' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'si', tuple_var_assignment_560289_560366)
    
    # Assigning a Name to a Name (line 19):
    # Getting the type of 'tuple_var_assignment_560290' (line 19)
    tuple_var_assignment_560290_560367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'tuple_var_assignment_560290')
    # Assigning a type to the variable 'ci' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'ci', tuple_var_assignment_560290_560367)
    
    # Assigning a Call to a Name (line 20):
    
    # Assigning a Call to a Name (line 20):
    
    # Call to column_stack(...): (line 20)
    # Processing the call arguments (line 20)
    
    # Obtaining an instance of the builtin type 'tuple' (line 20)
    tuple_560370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 20)
    # Adding element type (line 20)
    # Getting the type of 'x' (line 20)
    x_560371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 31), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 31), tuple_560370, x_560371)
    # Adding element type (line 20)
    # Getting the type of 'si' (line 20)
    si_560372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 34), 'si', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 31), tuple_560370, si_560372)
    # Adding element type (line 20)
    # Getting the type of 'ci' (line 20)
    ci_560373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 38), 'ci', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 31), tuple_560370, ci_560373)
    
    # Processing the call keyword arguments (line 20)
    kwargs_560374 = {}
    # Getting the type of 'np' (line 20)
    np_560368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 14), 'np', False)
    # Obtaining the member 'column_stack' of a type (line 20)
    column_stack_560369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 14), np_560368, 'column_stack')
    # Calling column_stack(args, kwargs) (line 20)
    column_stack_call_result_560375 = invoke(stypy.reporting.localization.Localization(__file__, 20, 14), column_stack_560369, *[tuple_560370], **kwargs_560374)
    
    # Assigning a type to the variable 'dataset' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'dataset', column_stack_call_result_560375)
    
    # Call to check(...): (line 21)
    # Processing the call keyword arguments (line 21)
    kwargs_560388 = {}
    
    # Call to FuncData(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'sici' (line 21)
    sici_560377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 13), 'sici', False)
    # Getting the type of 'dataset' (line 21)
    dataset_560378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 19), 'dataset', False)
    int_560379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 28), 'int')
    
    # Obtaining an instance of the builtin type 'tuple' (line 21)
    tuple_560380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 21)
    # Adding element type (line 21)
    int_560381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 32), tuple_560380, int_560381)
    # Adding element type (line 21)
    int_560382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 32), tuple_560380, int_560382)
    
    # Processing the call keyword arguments (line 21)
    float_560383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 44), 'float')
    keyword_560384 = float_560383
    kwargs_560385 = {'rtol': keyword_560384}
    # Getting the type of 'FuncData' (line 21)
    FuncData_560376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'FuncData', False)
    # Calling FuncData(args, kwargs) (line 21)
    FuncData_call_result_560386 = invoke(stypy.reporting.localization.Localization(__file__, 21, 4), FuncData_560376, *[sici_560377, dataset_560378, int_560379, tuple_560380], **kwargs_560385)
    
    # Obtaining the member 'check' of a type (line 21)
    check_560387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 4), FuncData_call_result_560386, 'check')
    # Calling check(args, kwargs) (line 21)
    check_call_result_560389 = invoke(stypy.reporting.localization.Localization(__file__, 21, 4), check_560387, *[], **kwargs_560388)
    
    
    # ################# End of 'test_sici_consistency(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_sici_consistency' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_560390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_560390)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_sici_consistency'
    return stypy_return_type_560390

# Assigning a type to the variable 'test_sici_consistency' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'test_sici_consistency', test_sici_consistency)

@norecursion
def test_shichi_consistency(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_shichi_consistency'
    module_type_store = module_type_store.open_function_context('test_shichi_consistency', 24, 0, False)
    
    # Passed parameters checking function
    test_shichi_consistency.stypy_localization = localization
    test_shichi_consistency.stypy_type_of_self = None
    test_shichi_consistency.stypy_type_store = module_type_store
    test_shichi_consistency.stypy_function_name = 'test_shichi_consistency'
    test_shichi_consistency.stypy_param_names_list = []
    test_shichi_consistency.stypy_varargs_param_name = None
    test_shichi_consistency.stypy_kwargs_param_name = None
    test_shichi_consistency.stypy_call_defaults = defaults
    test_shichi_consistency.stypy_call_varargs = varargs
    test_shichi_consistency.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_shichi_consistency', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_shichi_consistency', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_shichi_consistency(...)' code ##################


    @norecursion
    def shichi(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'shichi'
        module_type_store = module_type_store.open_function_context('shichi', 29, 4, False)
        
        # Passed parameters checking function
        shichi.stypy_localization = localization
        shichi.stypy_type_of_self = None
        shichi.stypy_type_store = module_type_store
        shichi.stypy_function_name = 'shichi'
        shichi.stypy_param_names_list = ['x']
        shichi.stypy_varargs_param_name = None
        shichi.stypy_kwargs_param_name = None
        shichi.stypy_call_defaults = defaults
        shichi.stypy_call_varargs = varargs
        shichi.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'shichi', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'shichi', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'shichi(...)' code ##################

        
        # Assigning a Call to a Tuple (line 30):
        
        # Assigning a Subscript to a Name (line 30):
        
        # Obtaining the type of the subscript
        int_560391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 8), 'int')
        
        # Call to shichi(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'x' (line 30)
        x_560394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 29), 'x', False)
        complex_560395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 33), 'complex')
        # Applying the binary operator '+' (line 30)
        result_add_560396 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 29), '+', x_560394, complex_560395)
        
        # Processing the call keyword arguments (line 30)
        kwargs_560397 = {}
        # Getting the type of 'sc' (line 30)
        sc_560392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 19), 'sc', False)
        # Obtaining the member 'shichi' of a type (line 30)
        shichi_560393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 19), sc_560392, 'shichi')
        # Calling shichi(args, kwargs) (line 30)
        shichi_call_result_560398 = invoke(stypy.reporting.localization.Localization(__file__, 30, 19), shichi_560393, *[result_add_560396], **kwargs_560397)
        
        # Obtaining the member '__getitem__' of a type (line 30)
        getitem___560399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), shichi_call_result_560398, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 30)
        subscript_call_result_560400 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), getitem___560399, int_560391)
        
        # Assigning a type to the variable 'tuple_var_assignment_560291' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'tuple_var_assignment_560291', subscript_call_result_560400)
        
        # Assigning a Subscript to a Name (line 30):
        
        # Obtaining the type of the subscript
        int_560401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 8), 'int')
        
        # Call to shichi(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'x' (line 30)
        x_560404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 29), 'x', False)
        complex_560405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 33), 'complex')
        # Applying the binary operator '+' (line 30)
        result_add_560406 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 29), '+', x_560404, complex_560405)
        
        # Processing the call keyword arguments (line 30)
        kwargs_560407 = {}
        # Getting the type of 'sc' (line 30)
        sc_560402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 19), 'sc', False)
        # Obtaining the member 'shichi' of a type (line 30)
        shichi_560403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 19), sc_560402, 'shichi')
        # Calling shichi(args, kwargs) (line 30)
        shichi_call_result_560408 = invoke(stypy.reporting.localization.Localization(__file__, 30, 19), shichi_560403, *[result_add_560406], **kwargs_560407)
        
        # Obtaining the member '__getitem__' of a type (line 30)
        getitem___560409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), shichi_call_result_560408, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 30)
        subscript_call_result_560410 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), getitem___560409, int_560401)
        
        # Assigning a type to the variable 'tuple_var_assignment_560292' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'tuple_var_assignment_560292', subscript_call_result_560410)
        
        # Assigning a Name to a Name (line 30):
        # Getting the type of 'tuple_var_assignment_560291' (line 30)
        tuple_var_assignment_560291_560411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'tuple_var_assignment_560291')
        # Assigning a type to the variable 'shi' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'shi', tuple_var_assignment_560291_560411)
        
        # Assigning a Name to a Name (line 30):
        # Getting the type of 'tuple_var_assignment_560292' (line 30)
        tuple_var_assignment_560292_560412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'tuple_var_assignment_560292')
        # Assigning a type to the variable 'chi' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 13), 'chi', tuple_var_assignment_560292_560412)
        
        # Obtaining an instance of the builtin type 'tuple' (line 31)
        tuple_560413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 31)
        # Adding element type (line 31)
        # Getting the type of 'shi' (line 31)
        shi_560414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'shi')
        # Obtaining the member 'real' of a type (line 31)
        real_560415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 15), shi_560414, 'real')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 15), tuple_560413, real_560415)
        # Adding element type (line 31)
        # Getting the type of 'chi' (line 31)
        chi_560416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 25), 'chi')
        # Obtaining the member 'real' of a type (line 31)
        real_560417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 25), chi_560416, 'real')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 15), tuple_560413, real_560417)
        
        # Assigning a type to the variable 'stypy_return_type' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'stypy_return_type', tuple_560413)
        
        # ################# End of 'shichi(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'shichi' in the type store
        # Getting the type of 'stypy_return_type' (line 29)
        stypy_return_type_560418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_560418)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'shichi'
        return stypy_return_type_560418

    # Assigning a type to the variable 'shichi' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'shichi', shichi)
    
    # Assigning a Subscript to a Name (line 34):
    
    # Assigning a Subscript to a Name (line 34):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 34)
    tuple_560419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 34)
    # Adding element type (line 34)
    
    
    # Call to logspace(...): (line 34)
    # Processing the call arguments (line 34)
    
    # Call to log10(...): (line 34)
    # Processing the call arguments (line 34)
    int_560424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 36), 'int')
    # Processing the call keyword arguments (line 34)
    kwargs_560425 = {}
    # Getting the type of 'np' (line 34)
    np_560422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 27), 'np', False)
    # Obtaining the member 'log10' of a type (line 34)
    log10_560423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 27), np_560422, 'log10')
    # Calling log10(args, kwargs) (line 34)
    log10_call_result_560426 = invoke(stypy.reporting.localization.Localization(__file__, 34, 27), log10_560423, *[int_560424], **kwargs_560425)
    
    int_560427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 42), 'int')
    int_560428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 47), 'int')
    # Processing the call keyword arguments (line 34)
    kwargs_560429 = {}
    # Getting the type of 'np' (line 34)
    np_560420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'np', False)
    # Obtaining the member 'logspace' of a type (line 34)
    logspace_560421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 15), np_560420, 'logspace')
    # Calling logspace(args, kwargs) (line 34)
    logspace_call_result_560430 = invoke(stypy.reporting.localization.Localization(__file__, 34, 15), logspace_560421, *[log10_call_result_560426, int_560427, int_560428], **kwargs_560429)
    
    # Applying the 'usub' unary operator (line 34)
    result___neg___560431 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 14), 'usub', logspace_call_result_560430)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 14), tuple_560419, result___neg___560431)
    # Adding element type (line 34)
    int_560432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 53), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 14), tuple_560419, int_560432)
    # Adding element type (line 34)
    
    # Call to logspace(...): (line 35)
    # Processing the call arguments (line 35)
    int_560435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 26), 'int')
    
    # Call to log10(...): (line 35)
    # Processing the call arguments (line 35)
    int_560438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 40), 'int')
    # Processing the call keyword arguments (line 35)
    kwargs_560439 = {}
    # Getting the type of 'np' (line 35)
    np_560436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 31), 'np', False)
    # Obtaining the member 'log10' of a type (line 35)
    log10_560437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 31), np_560436, 'log10')
    # Calling log10(args, kwargs) (line 35)
    log10_call_result_560440 = invoke(stypy.reporting.localization.Localization(__file__, 35, 31), log10_560437, *[int_560438], **kwargs_560439)
    
    int_560441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 46), 'int')
    # Processing the call keyword arguments (line 35)
    kwargs_560442 = {}
    # Getting the type of 'np' (line 35)
    np_560433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 14), 'np', False)
    # Obtaining the member 'logspace' of a type (line 35)
    logspace_560434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 14), np_560433, 'logspace')
    # Calling logspace(args, kwargs) (line 35)
    logspace_call_result_560443 = invoke(stypy.reporting.localization.Localization(__file__, 35, 14), logspace_560434, *[int_560435, log10_call_result_560440, int_560441], **kwargs_560442)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 14), tuple_560419, logspace_call_result_560443)
    
    # Getting the type of 'np' (line 34)
    np_560444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'np')
    # Obtaining the member 'r_' of a type (line 34)
    r__560445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), np_560444, 'r_')
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___560446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), r__560445, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_560447 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), getitem___560446, tuple_560419)
    
    # Assigning a type to the variable 'x' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'x', subscript_call_result_560447)
    
    # Assigning a Call to a Tuple (line 36):
    
    # Assigning a Subscript to a Name (line 36):
    
    # Obtaining the type of the subscript
    int_560448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 4), 'int')
    
    # Call to shichi(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'x' (line 36)
    x_560451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 25), 'x', False)
    # Processing the call keyword arguments (line 36)
    kwargs_560452 = {}
    # Getting the type of 'sc' (line 36)
    sc_560449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'sc', False)
    # Obtaining the member 'shichi' of a type (line 36)
    shichi_560450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 15), sc_560449, 'shichi')
    # Calling shichi(args, kwargs) (line 36)
    shichi_call_result_560453 = invoke(stypy.reporting.localization.Localization(__file__, 36, 15), shichi_560450, *[x_560451], **kwargs_560452)
    
    # Obtaining the member '__getitem__' of a type (line 36)
    getitem___560454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 4), shichi_call_result_560453, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 36)
    subscript_call_result_560455 = invoke(stypy.reporting.localization.Localization(__file__, 36, 4), getitem___560454, int_560448)
    
    # Assigning a type to the variable 'tuple_var_assignment_560293' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'tuple_var_assignment_560293', subscript_call_result_560455)
    
    # Assigning a Subscript to a Name (line 36):
    
    # Obtaining the type of the subscript
    int_560456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 4), 'int')
    
    # Call to shichi(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'x' (line 36)
    x_560459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 25), 'x', False)
    # Processing the call keyword arguments (line 36)
    kwargs_560460 = {}
    # Getting the type of 'sc' (line 36)
    sc_560457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'sc', False)
    # Obtaining the member 'shichi' of a type (line 36)
    shichi_560458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 15), sc_560457, 'shichi')
    # Calling shichi(args, kwargs) (line 36)
    shichi_call_result_560461 = invoke(stypy.reporting.localization.Localization(__file__, 36, 15), shichi_560458, *[x_560459], **kwargs_560460)
    
    # Obtaining the member '__getitem__' of a type (line 36)
    getitem___560462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 4), shichi_call_result_560461, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 36)
    subscript_call_result_560463 = invoke(stypy.reporting.localization.Localization(__file__, 36, 4), getitem___560462, int_560456)
    
    # Assigning a type to the variable 'tuple_var_assignment_560294' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'tuple_var_assignment_560294', subscript_call_result_560463)
    
    # Assigning a Name to a Name (line 36):
    # Getting the type of 'tuple_var_assignment_560293' (line 36)
    tuple_var_assignment_560293_560464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'tuple_var_assignment_560293')
    # Assigning a type to the variable 'shi' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'shi', tuple_var_assignment_560293_560464)
    
    # Assigning a Name to a Name (line 36):
    # Getting the type of 'tuple_var_assignment_560294' (line 36)
    tuple_var_assignment_560294_560465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'tuple_var_assignment_560294')
    # Assigning a type to the variable 'chi' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 9), 'chi', tuple_var_assignment_560294_560465)
    
    # Assigning a Call to a Name (line 37):
    
    # Assigning a Call to a Name (line 37):
    
    # Call to column_stack(...): (line 37)
    # Processing the call arguments (line 37)
    
    # Obtaining an instance of the builtin type 'tuple' (line 37)
    tuple_560468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 37)
    # Adding element type (line 37)
    # Getting the type of 'x' (line 37)
    x_560469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 31), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 31), tuple_560468, x_560469)
    # Adding element type (line 37)
    # Getting the type of 'shi' (line 37)
    shi_560470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 34), 'shi', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 31), tuple_560468, shi_560470)
    # Adding element type (line 37)
    # Getting the type of 'chi' (line 37)
    chi_560471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 39), 'chi', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 31), tuple_560468, chi_560471)
    
    # Processing the call keyword arguments (line 37)
    kwargs_560472 = {}
    # Getting the type of 'np' (line 37)
    np_560466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 14), 'np', False)
    # Obtaining the member 'column_stack' of a type (line 37)
    column_stack_560467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 14), np_560466, 'column_stack')
    # Calling column_stack(args, kwargs) (line 37)
    column_stack_call_result_560473 = invoke(stypy.reporting.localization.Localization(__file__, 37, 14), column_stack_560467, *[tuple_560468], **kwargs_560472)
    
    # Assigning a type to the variable 'dataset' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'dataset', column_stack_call_result_560473)
    
    # Call to check(...): (line 38)
    # Processing the call keyword arguments (line 38)
    kwargs_560486 = {}
    
    # Call to FuncData(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'shichi' (line 38)
    shichi_560475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 13), 'shichi', False)
    # Getting the type of 'dataset' (line 38)
    dataset_560476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 21), 'dataset', False)
    int_560477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 30), 'int')
    
    # Obtaining an instance of the builtin type 'tuple' (line 38)
    tuple_560478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 38)
    # Adding element type (line 38)
    int_560479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 34), tuple_560478, int_560479)
    # Adding element type (line 38)
    int_560480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 34), tuple_560478, int_560480)
    
    # Processing the call keyword arguments (line 38)
    float_560481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 46), 'float')
    keyword_560482 = float_560481
    kwargs_560483 = {'rtol': keyword_560482}
    # Getting the type of 'FuncData' (line 38)
    FuncData_560474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'FuncData', False)
    # Calling FuncData(args, kwargs) (line 38)
    FuncData_call_result_560484 = invoke(stypy.reporting.localization.Localization(__file__, 38, 4), FuncData_560474, *[shichi_560475, dataset_560476, int_560477, tuple_560478], **kwargs_560483)
    
    # Obtaining the member 'check' of a type (line 38)
    check_560485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 4), FuncData_call_result_560484, 'check')
    # Calling check(args, kwargs) (line 38)
    check_call_result_560487 = invoke(stypy.reporting.localization.Localization(__file__, 38, 4), check_560485, *[], **kwargs_560486)
    
    
    # ################# End of 'test_shichi_consistency(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_shichi_consistency' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_560488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_560488)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_shichi_consistency'
    return stypy_return_type_560488

# Assigning a type to the variable 'test_shichi_consistency' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'test_shichi_consistency', test_shichi_consistency)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
