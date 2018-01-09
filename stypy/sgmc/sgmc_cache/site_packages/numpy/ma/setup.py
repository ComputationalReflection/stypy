
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: from __future__ import division, print_function
3: 
4: def configuration(parent_package='',top_path=None):
5:     from numpy.distutils.misc_util import Configuration
6:     config = Configuration('ma', parent_package, top_path)
7:     config.add_data_dir('tests')
8:     return config
9: 
10: if __name__ == "__main__":
11:     from numpy.distutils.core import setup
12:     config = configuration(top_path='').todict()
13:     setup(**config)
14: 

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
    str_157023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 33), 'str', '')
    # Getting the type of 'None' (line 4)
    None_157024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 45), 'None')
    defaults = [str_157023, None_157024]
    # Create a new context for function 'configuration'
    module_type_store = module_type_store.open_function_context('configuration', 4, 0, False)
    
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

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 4))
    
    # 'from numpy.distutils.misc_util import Configuration' statement (line 5)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
    import_157025 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util')

    if (type(import_157025) is not StypyTypeError):

        if (import_157025 != 'pyd_module'):
            __import__(import_157025)
            sys_modules_157026 = sys.modules[import_157025]
            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', sys_modules_157026.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 5, 4), __file__, sys_modules_157026, sys_modules_157026.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', import_157025)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')
    
    
    # Assigning a Call to a Name (line 6):
    
    # Call to Configuration(...): (line 6)
    # Processing the call arguments (line 6)
    str_157028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 27), 'str', 'ma')
    # Getting the type of 'parent_package' (line 6)
    parent_package_157029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 33), 'parent_package', False)
    # Getting the type of 'top_path' (line 6)
    top_path_157030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 49), 'top_path', False)
    # Processing the call keyword arguments (line 6)
    kwargs_157031 = {}
    # Getting the type of 'Configuration' (line 6)
    Configuration_157027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 6)
    Configuration_call_result_157032 = invoke(stypy.reporting.localization.Localization(__file__, 6, 13), Configuration_157027, *[str_157028, parent_package_157029, top_path_157030], **kwargs_157031)
    
    # Assigning a type to the variable 'config' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'config', Configuration_call_result_157032)
    
    # Call to add_data_dir(...): (line 7)
    # Processing the call arguments (line 7)
    str_157035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 7)
    kwargs_157036 = {}
    # Getting the type of 'config' (line 7)
    config_157033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 7)
    add_data_dir_157034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), config_157033, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 7)
    add_data_dir_call_result_157037 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), add_data_dir_157034, *[str_157035], **kwargs_157036)
    
    # Getting the type of 'config' (line 8)
    config_157038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type', config_157038)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_157039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_157039)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_157039

# Assigning a type to the variable 'configuration' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 11)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
    import_157040 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.core')

    if (type(import_157040) is not StypyTypeError):

        if (import_157040 != 'pyd_module'):
            __import__(import_157040)
            sys_modules_157041 = sys.modules[import_157040]
            import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.core', sys_modules_157041.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 11, 4), __file__, sys_modules_157041, sys_modules_157041.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.core', import_157040)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')
    
    
    # Assigning a Call to a Name (line 12):
    
    # Call to todict(...): (line 12)
    # Processing the call keyword arguments (line 12)
    kwargs_157048 = {}
    
    # Call to configuration(...): (line 12)
    # Processing the call keyword arguments (line 12)
    str_157043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 36), 'str', '')
    keyword_157044 = str_157043
    kwargs_157045 = {'top_path': keyword_157044}
    # Getting the type of 'configuration' (line 12)
    configuration_157042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 13), 'configuration', False)
    # Calling configuration(args, kwargs) (line 12)
    configuration_call_result_157046 = invoke(stypy.reporting.localization.Localization(__file__, 12, 13), configuration_157042, *[], **kwargs_157045)
    
    # Obtaining the member 'todict' of a type (line 12)
    todict_157047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 13), configuration_call_result_157046, 'todict')
    # Calling todict(args, kwargs) (line 12)
    todict_call_result_157049 = invoke(stypy.reporting.localization.Localization(__file__, 12, 13), todict_157047, *[], **kwargs_157048)
    
    # Assigning a type to the variable 'config' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'config', todict_call_result_157049)
    
    # Call to setup(...): (line 13)
    # Processing the call keyword arguments (line 13)
    # Getting the type of 'config' (line 13)
    config_157051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'config', False)
    kwargs_157052 = {'config_157051': config_157051}
    # Getting the type of 'setup' (line 13)
    setup_157050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 13)
    setup_call_result_157053 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), setup_157050, *[], **kwargs_157052)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
