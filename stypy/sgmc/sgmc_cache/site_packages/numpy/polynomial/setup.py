
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function
2: 
3: def configuration(parent_package='',top_path=None):
4:     from numpy.distutils.misc_util import Configuration
5:     config = Configuration('polynomial', parent_package, top_path)
6:     config.add_data_dir('tests')
7:     return config
8: 
9: if __name__ == '__main__':
10:     from numpy.distutils.core import setup
11:     setup(configuration=configuration)
12: 

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
    str_179158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 33), 'str', '')
    # Getting the type of 'None' (line 3)
    None_179159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 45), 'None')
    defaults = [str_179158, None_179159]
    # Create a new context for function 'configuration'
    module_type_store = module_type_store.open_function_context('configuration', 3, 0, False)
    
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

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 4))
    
    # 'from numpy.distutils.misc_util import Configuration' statement (line 4)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
    import_179160 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 4), 'numpy.distutils.misc_util')

    if (type(import_179160) is not StypyTypeError):

        if (import_179160 != 'pyd_module'):
            __import__(import_179160)
            sys_modules_179161 = sys.modules[import_179160]
            import_from_module(stypy.reporting.localization.Localization(__file__, 4, 4), 'numpy.distutils.misc_util', sys_modules_179161.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 4, 4), __file__, sys_modules_179161, sys_modules_179161.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 4, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 4)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'numpy.distutils.misc_util', import_179160)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')
    
    
    # Assigning a Call to a Name (line 5):
    
    # Call to Configuration(...): (line 5)
    # Processing the call arguments (line 5)
    str_179163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 27), 'str', 'polynomial')
    # Getting the type of 'parent_package' (line 5)
    parent_package_179164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 41), 'parent_package', False)
    # Getting the type of 'top_path' (line 5)
    top_path_179165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 57), 'top_path', False)
    # Processing the call keyword arguments (line 5)
    kwargs_179166 = {}
    # Getting the type of 'Configuration' (line 5)
    Configuration_179162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 5)
    Configuration_call_result_179167 = invoke(stypy.reporting.localization.Localization(__file__, 5, 13), Configuration_179162, *[str_179163, parent_package_179164, top_path_179165], **kwargs_179166)
    
    # Assigning a type to the variable 'config' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'config', Configuration_call_result_179167)
    
    # Call to add_data_dir(...): (line 6)
    # Processing the call arguments (line 6)
    str_179170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 6)
    kwargs_179171 = {}
    # Getting the type of 'config' (line 6)
    config_179168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 6)
    add_data_dir_179169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), config_179168, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 6)
    add_data_dir_call_result_179172 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), add_data_dir_179169, *[str_179170], **kwargs_179171)
    
    # Getting the type of 'config' (line 7)
    config_179173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'stypy_return_type', config_179173)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 3)
    stypy_return_type_179174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_179174)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_179174

# Assigning a type to the variable 'configuration' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 10)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
    import_179175 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.distutils.core')

    if (type(import_179175) is not StypyTypeError):

        if (import_179175 != 'pyd_module'):
            __import__(import_179175)
            sys_modules_179176 = sys.modules[import_179175]
            import_from_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.distutils.core', sys_modules_179176.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 10, 4), __file__, sys_modules_179176, sys_modules_179176.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.distutils.core', import_179175)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')
    
    
    # Call to setup(...): (line 11)
    # Processing the call keyword arguments (line 11)
    # Getting the type of 'configuration' (line 11)
    configuration_179178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 24), 'configuration', False)
    keyword_179179 = configuration_179178
    kwargs_179180 = {'configuration': keyword_179179}
    # Getting the type of 'setup' (line 11)
    setup_179177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 11)
    setup_call_result_179181 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), setup_179177, *[], **kwargs_179180)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
