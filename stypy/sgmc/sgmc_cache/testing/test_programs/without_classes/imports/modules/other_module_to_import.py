
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from sub.submodule import *
2: 
3: global_a2 = 1
4: 
5: 
6: def f_parent2():
7:     local_a = 2
8: 
9:     return local_a
10: 
11: var1 = submodule_var
12: var2 = submodule_func()

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from sub.submodule import ' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/modules/')
import_5184 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'sub.submodule')

if (type(import_5184) is not StypyTypeError):

    if (import_5184 != 'pyd_module'):
        __import__(import_5184)
        sys_modules_5185 = sys.modules[import_5184]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'sub.submodule', sys_modules_5185.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_5185, sys_modules_5185.module_type_store, module_type_store)
    else:
        from sub.submodule import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'sub.submodule', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'sub.submodule' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'sub.submodule', import_5184)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/modules/')


# Assigning a Num to a Name (line 3):
int_5186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 12), 'int')
# Assigning a type to the variable 'global_a2' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'global_a2', int_5186)

@norecursion
def f_parent2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f_parent2'
    module_type_store = module_type_store.open_function_context('f_parent2', 6, 0, False)
    
    # Passed parameters checking function
    f_parent2.stypy_localization = localization
    f_parent2.stypy_type_of_self = None
    f_parent2.stypy_type_store = module_type_store
    f_parent2.stypy_function_name = 'f_parent2'
    f_parent2.stypy_param_names_list = []
    f_parent2.stypy_varargs_param_name = None
    f_parent2.stypy_kwargs_param_name = None
    f_parent2.stypy_call_defaults = defaults
    f_parent2.stypy_call_varargs = varargs
    f_parent2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f_parent2', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f_parent2', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f_parent2(...)' code ##################

    
    # Assigning a Num to a Name (line 7):
    int_5187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 14), 'int')
    # Assigning a type to the variable 'local_a' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'local_a', int_5187)
    # Getting the type of 'local_a' (line 9)
    local_a_5188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 11), 'local_a')
    # Assigning a type to the variable 'stypy_return_type' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'stypy_return_type', local_a_5188)
    
    # ################# End of 'f_parent2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f_parent2' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_5189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5189)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f_parent2'
    return stypy_return_type_5189

# Assigning a type to the variable 'f_parent2' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'f_parent2', f_parent2)

# Assigning a Name to a Name (line 11):
# Getting the type of 'submodule_var' (line 11)
submodule_var_5190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 7), 'submodule_var')
# Assigning a type to the variable 'var1' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'var1', submodule_var_5190)

# Assigning a Call to a Name (line 12):

# Call to submodule_func(...): (line 12)
# Processing the call keyword arguments (line 12)
kwargs_5192 = {}
# Getting the type of 'submodule_func' (line 12)
submodule_func_5191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 7), 'submodule_func', False)
# Calling submodule_func(args, kwargs) (line 12)
submodule_func_call_result_5193 = invoke(stypy.reporting.localization.Localization(__file__, 12, 7), submodule_func_5191, *[], **kwargs_5192)

# Assigning a type to the variable 'var2' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'var2', submodule_func_call_result_5193)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
