
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import module_to_import
2: 
3: def f():
4:     return 3
5: 
6: z = f()
7: 
8: x = module_to_import.global_a
9: y = module_to_import.f_parent()
10: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import module_to_import' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')
import_5008 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'module_to_import')

if (type(import_5008) is not StypyTypeError):

    if (import_5008 != 'pyd_module'):
        __import__(import_5008)
        sys_modules_5009 = sys.modules[import_5008]
        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'module_to_import', sys_modules_5009.module_type_store, module_type_store)
    else:
        import module_to_import

        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'module_to_import', module_to_import, module_type_store)

else:
    # Assigning a type to the variable 'module_to_import' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'module_to_import', import_5008)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')


@norecursion
def f(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f'
    module_type_store = module_type_store.open_function_context('f', 3, 0, False)
    
    # Passed parameters checking function
    f.stypy_localization = localization
    f.stypy_type_of_self = None
    f.stypy_type_store = module_type_store
    f.stypy_function_name = 'f'
    f.stypy_param_names_list = []
    f.stypy_varargs_param_name = None
    f.stypy_kwargs_param_name = None
    f.stypy_call_defaults = defaults
    f.stypy_call_varargs = varargs
    f.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f(...)' code ##################

    int_5010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'stypy_return_type', int_5010)
    
    # ################# End of 'f(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f' in the type store
    # Getting the type of 'stypy_return_type' (line 3)
    stypy_return_type_5011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5011)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f'
    return stypy_return_type_5011

# Assigning a type to the variable 'f' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'f', f)

# Assigning a Call to a Name (line 6):

# Call to f(...): (line 6)
# Processing the call keyword arguments (line 6)
kwargs_5013 = {}
# Getting the type of 'f' (line 6)
f_5012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'f', False)
# Calling f(args, kwargs) (line 6)
f_call_result_5014 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), f_5012, *[], **kwargs_5013)

# Assigning a type to the variable 'z' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'z', f_call_result_5014)

# Assigning a Attribute to a Name (line 8):
# Getting the type of 'module_to_import' (line 8)
module_to_import_5015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'module_to_import')
# Obtaining the member 'global_a' of a type (line 8)
global_a_5016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 4), module_to_import_5015, 'global_a')
# Assigning a type to the variable 'x' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'x', global_a_5016)

# Assigning a Call to a Name (line 9):

# Call to f_parent(...): (line 9)
# Processing the call keyword arguments (line 9)
kwargs_5019 = {}
# Getting the type of 'module_to_import' (line 9)
module_to_import_5017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'module_to_import', False)
# Obtaining the member 'f_parent' of a type (line 9)
f_parent_5018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), module_to_import_5017, 'f_parent')
# Calling f_parent(args, kwargs) (line 9)
f_parent_call_result_5020 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), f_parent_5018, *[], **kwargs_5019)

# Assigning a type to the variable 'y' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'y', f_parent_call_result_5020)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
