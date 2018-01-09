
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function
2: 
3: def configuration(parent_package='',top_path=None):
4:     from numpy.distutils.misc_util import Configuration
5: 
6:     config = Configuration('lib', parent_package, top_path)
7:     config.add_data_dir('tests')
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
    str_125159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 33), 'str', '')
    # Getting the type of 'None' (line 3)
    None_125160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 45), 'None')
    defaults = [str_125159, None_125160]
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
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
    import_125161 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 4), 'numpy.distutils.misc_util')

    if (type(import_125161) is not StypyTypeError):

        if (import_125161 != 'pyd_module'):
            __import__(import_125161)
            sys_modules_125162 = sys.modules[import_125161]
            import_from_module(stypy.reporting.localization.Localization(__file__, 4, 4), 'numpy.distutils.misc_util', sys_modules_125162.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 4, 4), __file__, sys_modules_125162, sys_modules_125162.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 4, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 4)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'numpy.distutils.misc_util', import_125161)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')
    
    
    # Assigning a Call to a Name (line 6):
    
    # Call to Configuration(...): (line 6)
    # Processing the call arguments (line 6)
    str_125164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 27), 'str', 'lib')
    # Getting the type of 'parent_package' (line 6)
    parent_package_125165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 34), 'parent_package', False)
    # Getting the type of 'top_path' (line 6)
    top_path_125166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 50), 'top_path', False)
    # Processing the call keyword arguments (line 6)
    kwargs_125167 = {}
    # Getting the type of 'Configuration' (line 6)
    Configuration_125163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 6)
    Configuration_call_result_125168 = invoke(stypy.reporting.localization.Localization(__file__, 6, 13), Configuration_125163, *[str_125164, parent_package_125165, top_path_125166], **kwargs_125167)
    
    # Assigning a type to the variable 'config' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'config', Configuration_call_result_125168)
    
    # Call to add_data_dir(...): (line 7)
    # Processing the call arguments (line 7)
    str_125171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 7)
    kwargs_125172 = {}
    # Getting the type of 'config' (line 7)
    config_125169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 7)
    add_data_dir_125170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), config_125169, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 7)
    add_data_dir_call_result_125173 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), add_data_dir_125170, *[str_125171], **kwargs_125172)
    
    # Getting the type of 'config' (line 8)
    config_125174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type', config_125174)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 3)
    stypy_return_type_125175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125175)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_125175

# Assigning a type to the variable 'configuration' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 11)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
    import_125176 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.core')

    if (type(import_125176) is not StypyTypeError):

        if (import_125176 != 'pyd_module'):
            __import__(import_125176)
            sys_modules_125177 = sys.modules[import_125176]
            import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.core', sys_modules_125177.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 11, 4), __file__, sys_modules_125177, sys_modules_125177.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.core', import_125176)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')
    
    
    # Call to setup(...): (line 12)
    # Processing the call keyword arguments (line 12)
    # Getting the type of 'configuration' (line 12)
    configuration_125179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 24), 'configuration', False)
    keyword_125180 = configuration_125179
    kwargs_125181 = {'configuration': keyword_125180}
    # Getting the type of 'setup' (line 12)
    setup_125178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 12)
    setup_call_result_125182 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), setup_125178, *[], **kwargs_125181)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
