
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: sparsetools is not a public module in scipy.sparse, but this file is
3: for backward compatibility if someone happens to use it.
4: '''
5: from numpy import deprecate
6: 
7: # This file shouldn't be imported by scipy --- Scipy code should use
8: # internally scipy.sparse._sparsetools
9: 
10: 
11: @deprecate(old_name="scipy.sparse.sparsetools",
12:            message=("scipy.sparse.sparsetools is a private module for scipy.sparse, "
13:                     "and should not be used."))
14: def _deprecated():
15:     pass
16: 
17: del deprecate
18: 
19: try:
20:     _deprecated()
21: except DeprecationWarning as e:
22:     # don't fail import if DeprecationWarnings raise error -- works around
23:     # the situation with Numpy's test framework
24:     pass
25: 
26: from ._sparsetools import *
27: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_379375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', '\nsparsetools is not a public module in scipy.sparse, but this file is\nfor backward compatibility if someone happens to use it.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy import deprecate' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_379376 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_379376) is not StypyTypeError):

    if (import_379376 != 'pyd_module'):
        __import__(import_379376)
        sys_modules_379377 = sys.modules[import_379376]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', sys_modules_379377.module_type_store, module_type_store, ['deprecate'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_379377, sys_modules_379377.module_type_store, module_type_store)
    else:
        from numpy import deprecate

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', None, module_type_store, ['deprecate'], [deprecate])

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_379376)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')


@norecursion
def _deprecated(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_deprecated'
    module_type_store = module_type_store.open_function_context('_deprecated', 11, 0, False)
    
    # Passed parameters checking function
    _deprecated.stypy_localization = localization
    _deprecated.stypy_type_of_self = None
    _deprecated.stypy_type_store = module_type_store
    _deprecated.stypy_function_name = '_deprecated'
    _deprecated.stypy_param_names_list = []
    _deprecated.stypy_varargs_param_name = None
    _deprecated.stypy_kwargs_param_name = None
    _deprecated.stypy_call_defaults = defaults
    _deprecated.stypy_call_varargs = varargs
    _deprecated.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_deprecated', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_deprecated', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_deprecated(...)' code ##################

    pass
    
    # ################# End of '_deprecated(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_deprecated' in the type store
    # Getting the type of 'stypy_return_type' (line 11)
    stypy_return_type_379378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_379378)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_deprecated'
    return stypy_return_type_379378

# Assigning a type to the variable '_deprecated' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), '_deprecated', _deprecated)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 17, 0), module_type_store, 'deprecate')


# SSA begins for try-except statement (line 19)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Call to _deprecated(...): (line 20)
# Processing the call keyword arguments (line 20)
kwargs_379380 = {}
# Getting the type of '_deprecated' (line 20)
_deprecated_379379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), '_deprecated', False)
# Calling _deprecated(args, kwargs) (line 20)
_deprecated_call_result_379381 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), _deprecated_379379, *[], **kwargs_379380)

# SSA branch for the except part of a try statement (line 19)
# SSA branch for the except 'DeprecationWarning' branch of a try statement (line 19)
# Storing handler type
module_type_store.open_ssa_branch('except')
# Getting the type of 'DeprecationWarning' (line 21)
DeprecationWarning_379382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 7), 'DeprecationWarning')
# Assigning a type to the variable 'e' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'e', DeprecationWarning_379382)
pass
# SSA join for try-except statement (line 19)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from scipy.sparse._sparsetools import ' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_379383 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.sparse._sparsetools')

if (type(import_379383) is not StypyTypeError):

    if (import_379383 != 'pyd_module'):
        __import__(import_379383)
        sys_modules_379384 = sys.modules[import_379383]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.sparse._sparsetools', sys_modules_379384.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_379384, sys_modules_379384.module_type_store, module_type_store)
    else:
        from scipy.sparse._sparsetools import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.sparse._sparsetools', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse._sparsetools' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.sparse._sparsetools', import_379383)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
