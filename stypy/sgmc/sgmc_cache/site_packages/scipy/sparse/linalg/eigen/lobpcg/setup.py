
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: 
4: def configuration(parent_package='',top_path=None):
5:     from numpy.distutils.misc_util import Configuration
6: 
7:     config = Configuration('lobpcg',parent_package,top_path)
8:     config.add_data_dir('tests')
9: 
10:     return config
11: 
12: if __name__ == '__main__':
13:     from numpy.distutils.core import setup
14:     setup(**configuration(top_path='').todict())
15: 

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
    str_406879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 33), 'str', '')
    # Getting the type of 'None' (line 4)
    None_406880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 45), 'None')
    defaults = [str_406879, None_406880]
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
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/')
    import_406881 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util')

    if (type(import_406881) is not StypyTypeError):

        if (import_406881 != 'pyd_module'):
            __import__(import_406881)
            sys_modules_406882 = sys.modules[import_406881]
            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', sys_modules_406882.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 5, 4), __file__, sys_modules_406882, sys_modules_406882.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', import_406881)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/')
    
    
    # Assigning a Call to a Name (line 7):
    
    # Call to Configuration(...): (line 7)
    # Processing the call arguments (line 7)
    str_406884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 27), 'str', 'lobpcg')
    # Getting the type of 'parent_package' (line 7)
    parent_package_406885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 36), 'parent_package', False)
    # Getting the type of 'top_path' (line 7)
    top_path_406886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 51), 'top_path', False)
    # Processing the call keyword arguments (line 7)
    kwargs_406887 = {}
    # Getting the type of 'Configuration' (line 7)
    Configuration_406883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 7)
    Configuration_call_result_406888 = invoke(stypy.reporting.localization.Localization(__file__, 7, 13), Configuration_406883, *[str_406884, parent_package_406885, top_path_406886], **kwargs_406887)
    
    # Assigning a type to the variable 'config' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'config', Configuration_call_result_406888)
    
    # Call to add_data_dir(...): (line 8)
    # Processing the call arguments (line 8)
    str_406891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 8)
    kwargs_406892 = {}
    # Getting the type of 'config' (line 8)
    config_406889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 8)
    add_data_dir_406890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 4), config_406889, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 8)
    add_data_dir_call_result_406893 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), add_data_dir_406890, *[str_406891], **kwargs_406892)
    
    # Getting the type of 'config' (line 10)
    config_406894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'stypy_return_type', config_406894)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_406895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_406895)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_406895

# Assigning a type to the variable 'configuration' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 13)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/')
    import_406896 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy.distutils.core')

    if (type(import_406896) is not StypyTypeError):

        if (import_406896 != 'pyd_module'):
            __import__(import_406896)
            sys_modules_406897 = sys.modules[import_406896]
            import_from_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy.distutils.core', sys_modules_406897.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 13, 4), __file__, sys_modules_406897, sys_modules_406897.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy.distutils.core', import_406896)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/')
    
    
    # Call to setup(...): (line 14)
    # Processing the call keyword arguments (line 14)
    
    # Call to todict(...): (line 14)
    # Processing the call keyword arguments (line 14)
    kwargs_406905 = {}
    
    # Call to configuration(...): (line 14)
    # Processing the call keyword arguments (line 14)
    str_406900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 35), 'str', '')
    keyword_406901 = str_406900
    kwargs_406902 = {'top_path': keyword_406901}
    # Getting the type of 'configuration' (line 14)
    configuration_406899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 14)
    configuration_call_result_406903 = invoke(stypy.reporting.localization.Localization(__file__, 14, 12), configuration_406899, *[], **kwargs_406902)
    
    # Obtaining the member 'todict' of a type (line 14)
    todict_406904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 12), configuration_call_result_406903, 'todict')
    # Calling todict(args, kwargs) (line 14)
    todict_call_result_406906 = invoke(stypy.reporting.localization.Localization(__file__, 14, 12), todict_406904, *[], **kwargs_406905)
    
    kwargs_406907 = {'todict_call_result_406906': todict_call_result_406906}
    # Getting the type of 'setup' (line 14)
    setup_406898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 14)
    setup_call_result_406908 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), setup_406898, *[], **kwargs_406907)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
