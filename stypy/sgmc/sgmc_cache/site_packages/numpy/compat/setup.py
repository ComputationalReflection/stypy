
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: from __future__ import division, print_function
3: 
4: 
5: def configuration(parent_package='',top_path=None):
6:     from numpy.distutils.misc_util import Configuration
7:     config = Configuration('compat', parent_package, top_path)
8:     return config
9: 
10: if __name__ == '__main__':
11:     from numpy.distutils.core import setup
12:     setup(configuration=configuration)
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_25825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 33), 'str', '')
    # Getting the type of 'None' (line 5)
    None_25826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 45), 'None')
    defaults = [str_25825, None_25826]
    # Create a new context for function 'configuration'
    module_type_store = module_type_store.open_function_context('configuration', 5, 0, False)
    
    # Passed parameters checking function
    configuration.stypy_localization = localization
    configuration.stypy_type_of_self = None
    configuration.stypy_type_store = module_type_store
    configuration.stypy_function_name = 'configuration'
    configuration.stypy_param_names_list = ['parent_package', 'top_path']
    configuration.stypy_varargs_param_name = None
    configuration.stypy_kwargs_param_name = None
    configuration.stypy_call_defaults = defaults
    configuration.stypy_call_varargs = varargs
    configuration.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'configuration', ['parent_package', 'top_path'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'configuration', localization, ['parent_package', 'top_path'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'configuration(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 4))
    
    # 'from numpy.distutils.misc_util import Configuration' statement (line 6)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/compat/')
    import_25827 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'numpy.distutils.misc_util')

    if (type(import_25827) is not StypyTypeError):

        if (import_25827 != 'pyd_module'):
            __import__(import_25827)
            sys_modules_25828 = sys.modules[import_25827]
            import_from_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'numpy.distutils.misc_util', sys_modules_25828.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 6, 4), __file__, sys_modules_25828, sys_modules_25828.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'numpy.distutils.misc_util', import_25827)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/compat/')
    
    
    # Assigning a Call to a Name (line 7):
    
    # Call to Configuration(...): (line 7)
    # Processing the call arguments (line 7)
    str_25830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 27), 'str', 'compat')
    # Getting the type of 'parent_package' (line 7)
    parent_package_25831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 37), 'parent_package', False)
    # Getting the type of 'top_path' (line 7)
    top_path_25832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 53), 'top_path', False)
    # Processing the call keyword arguments (line 7)
    kwargs_25833 = {}
    # Getting the type of 'Configuration' (line 7)
    Configuration_25829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 7)
    Configuration_call_result_25834 = invoke(stypy.reporting.localization.Localization(__file__, 7, 13), Configuration_25829, *[str_25830, parent_package_25831, top_path_25832], **kwargs_25833)
    
    # Assigning a type to the variable 'config' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'config', Configuration_call_result_25834)
    # Getting the type of 'config' (line 8)
    config_25835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type', config_25835)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 5)
    stypy_return_type_25836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25836)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_25836

# Assigning a type to the variable 'configuration' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 11)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/compat/')
    import_25837 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.core')

    if (type(import_25837) is not StypyTypeError):

        if (import_25837 != 'pyd_module'):
            __import__(import_25837)
            sys_modules_25838 = sys.modules[import_25837]
            import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.core', sys_modules_25838.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 11, 4), __file__, sys_modules_25838, sys_modules_25838.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.core', import_25837)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/compat/')
    
    
    # Call to setup(...): (line 12)
    # Processing the call keyword arguments (line 12)
    # Getting the type of 'configuration' (line 12)
    configuration_25840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 24), 'configuration', False)
    keyword_25841 = configuration_25840
    kwargs_25842 = {'configuration': keyword_25841}
    # Getting the type of 'setup' (line 12)
    setup_25839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 12)
    setup_call_result_25843 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), setup_25839, *[], **kwargs_25842)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
