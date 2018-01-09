
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from pytest import raises as assert_raises
5: 
6: from scipy.sparse.linalg import utils
7: 
8: 
9: def test_make_system_bad_shape():
10:     assert_raises(ValueError, utils.make_system, np.zeros((5,3)), None, np.zeros(4), np.zeros(4))
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_422343 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_422343) is not StypyTypeError):

    if (import_422343 != 'pyd_module'):
        __import__(import_422343)
        sys_modules_422344 = sys.modules[import_422343]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_422344.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_422343)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from pytest import assert_raises' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_422345 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest')

if (type(import_422345) is not StypyTypeError):

    if (import_422345 != 'pyd_module'):
        __import__(import_422345)
        sys_modules_422346 = sys.modules[import_422345]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', sys_modules_422346.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_422346, sys_modules_422346.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', import_422345)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.sparse.linalg import utils' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_422347 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse.linalg')

if (type(import_422347) is not StypyTypeError):

    if (import_422347 != 'pyd_module'):
        __import__(import_422347)
        sys_modules_422348 = sys.modules[import_422347]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse.linalg', sys_modules_422348.module_type_store, module_type_store, ['utils'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_422348, sys_modules_422348.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import utils

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse.linalg', None, module_type_store, ['utils'], [utils])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse.linalg', import_422347)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')


@norecursion
def test_make_system_bad_shape(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_make_system_bad_shape'
    module_type_store = module_type_store.open_function_context('test_make_system_bad_shape', 9, 0, False)
    
    # Passed parameters checking function
    test_make_system_bad_shape.stypy_localization = localization
    test_make_system_bad_shape.stypy_type_of_self = None
    test_make_system_bad_shape.stypy_type_store = module_type_store
    test_make_system_bad_shape.stypy_function_name = 'test_make_system_bad_shape'
    test_make_system_bad_shape.stypy_param_names_list = []
    test_make_system_bad_shape.stypy_varargs_param_name = None
    test_make_system_bad_shape.stypy_kwargs_param_name = None
    test_make_system_bad_shape.stypy_call_defaults = defaults
    test_make_system_bad_shape.stypy_call_varargs = varargs
    test_make_system_bad_shape.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_make_system_bad_shape', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_make_system_bad_shape', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_make_system_bad_shape(...)' code ##################

    
    # Call to assert_raises(...): (line 10)
    # Processing the call arguments (line 10)
    # Getting the type of 'ValueError' (line 10)
    ValueError_422350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 18), 'ValueError', False)
    # Getting the type of 'utils' (line 10)
    utils_422351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 30), 'utils', False)
    # Obtaining the member 'make_system' of a type (line 10)
    make_system_422352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 30), utils_422351, 'make_system')
    
    # Call to zeros(...): (line 10)
    # Processing the call arguments (line 10)
    
    # Obtaining an instance of the builtin type 'tuple' (line 10)
    tuple_422355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 59), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 10)
    # Adding element type (line 10)
    int_422356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 59), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 59), tuple_422355, int_422356)
    # Adding element type (line 10)
    int_422357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 61), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 59), tuple_422355, int_422357)
    
    # Processing the call keyword arguments (line 10)
    kwargs_422358 = {}
    # Getting the type of 'np' (line 10)
    np_422353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 49), 'np', False)
    # Obtaining the member 'zeros' of a type (line 10)
    zeros_422354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 49), np_422353, 'zeros')
    # Calling zeros(args, kwargs) (line 10)
    zeros_call_result_422359 = invoke(stypy.reporting.localization.Localization(__file__, 10, 49), zeros_422354, *[tuple_422355], **kwargs_422358)
    
    # Getting the type of 'None' (line 10)
    None_422360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 66), 'None', False)
    
    # Call to zeros(...): (line 10)
    # Processing the call arguments (line 10)
    int_422363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 81), 'int')
    # Processing the call keyword arguments (line 10)
    kwargs_422364 = {}
    # Getting the type of 'np' (line 10)
    np_422361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 72), 'np', False)
    # Obtaining the member 'zeros' of a type (line 10)
    zeros_422362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 72), np_422361, 'zeros')
    # Calling zeros(args, kwargs) (line 10)
    zeros_call_result_422365 = invoke(stypy.reporting.localization.Localization(__file__, 10, 72), zeros_422362, *[int_422363], **kwargs_422364)
    
    
    # Call to zeros(...): (line 10)
    # Processing the call arguments (line 10)
    int_422368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 94), 'int')
    # Processing the call keyword arguments (line 10)
    kwargs_422369 = {}
    # Getting the type of 'np' (line 10)
    np_422366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 85), 'np', False)
    # Obtaining the member 'zeros' of a type (line 10)
    zeros_422367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 85), np_422366, 'zeros')
    # Calling zeros(args, kwargs) (line 10)
    zeros_call_result_422370 = invoke(stypy.reporting.localization.Localization(__file__, 10, 85), zeros_422367, *[int_422368], **kwargs_422369)
    
    # Processing the call keyword arguments (line 10)
    kwargs_422371 = {}
    # Getting the type of 'assert_raises' (line 10)
    assert_raises_422349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 10)
    assert_raises_call_result_422372 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), assert_raises_422349, *[ValueError_422350, make_system_422352, zeros_call_result_422359, None_422360, zeros_call_result_422365, zeros_call_result_422370], **kwargs_422371)
    
    
    # ################# End of 'test_make_system_bad_shape(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_make_system_bad_shape' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_422373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_422373)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_make_system_bad_shape'
    return stypy_return_type_422373

# Assigning a type to the variable 'test_make_system_bad_shape' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'test_make_system_bad_shape', test_make_system_bad_shape)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
