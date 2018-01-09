
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: 
4: def configuration(parent_package='',top_path=None):
5:     from numpy.distutils.misc_util import Configuration
6: 
7:     config = Configuration('eigen',parent_package,top_path)
8: 
9:     config.add_subpackage(('arpack'))
10:     config.add_subpackage(('lobpcg'))
11: 
12:     return config
13: 
14: if __name__ == '__main__':
15:     from numpy.distutils.core import setup
16:     setup(**configuration(top_path='').todict())
17: 

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
    str_395936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 33), 'str', '')
    # Getting the type of 'None' (line 4)
    None_395937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 45), 'None')
    defaults = [str_395936, None_395937]
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
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/')
    import_395938 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util')

    if (type(import_395938) is not StypyTypeError):

        if (import_395938 != 'pyd_module'):
            __import__(import_395938)
            sys_modules_395939 = sys.modules[import_395938]
            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', sys_modules_395939.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 5, 4), __file__, sys_modules_395939, sys_modules_395939.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', import_395938)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/')
    
    
    # Assigning a Call to a Name (line 7):
    
    # Call to Configuration(...): (line 7)
    # Processing the call arguments (line 7)
    str_395941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 27), 'str', 'eigen')
    # Getting the type of 'parent_package' (line 7)
    parent_package_395942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 35), 'parent_package', False)
    # Getting the type of 'top_path' (line 7)
    top_path_395943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 50), 'top_path', False)
    # Processing the call keyword arguments (line 7)
    kwargs_395944 = {}
    # Getting the type of 'Configuration' (line 7)
    Configuration_395940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 7)
    Configuration_call_result_395945 = invoke(stypy.reporting.localization.Localization(__file__, 7, 13), Configuration_395940, *[str_395941, parent_package_395942, top_path_395943], **kwargs_395944)
    
    # Assigning a type to the variable 'config' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'config', Configuration_call_result_395945)
    
    # Call to add_subpackage(...): (line 9)
    # Processing the call arguments (line 9)
    str_395948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 27), 'str', 'arpack')
    # Processing the call keyword arguments (line 9)
    kwargs_395949 = {}
    # Getting the type of 'config' (line 9)
    config_395946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 9)
    add_subpackage_395947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), config_395946, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 9)
    add_subpackage_call_result_395950 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), add_subpackage_395947, *[str_395948], **kwargs_395949)
    
    
    # Call to add_subpackage(...): (line 10)
    # Processing the call arguments (line 10)
    str_395953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 27), 'str', 'lobpcg')
    # Processing the call keyword arguments (line 10)
    kwargs_395954 = {}
    # Getting the type of 'config' (line 10)
    config_395951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 10)
    add_subpackage_395952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 4), config_395951, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 10)
    add_subpackage_call_result_395955 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), add_subpackage_395952, *[str_395953], **kwargs_395954)
    
    # Getting the type of 'config' (line 12)
    config_395956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'stypy_return_type', config_395956)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_395957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_395957)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_395957

# Assigning a type to the variable 'configuration' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 15)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/')
    import_395958 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'numpy.distutils.core')

    if (type(import_395958) is not StypyTypeError):

        if (import_395958 != 'pyd_module'):
            __import__(import_395958)
            sys_modules_395959 = sys.modules[import_395958]
            import_from_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'numpy.distutils.core', sys_modules_395959.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 15, 4), __file__, sys_modules_395959, sys_modules_395959.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'numpy.distutils.core', import_395958)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/')
    
    
    # Call to setup(...): (line 16)
    # Processing the call keyword arguments (line 16)
    
    # Call to todict(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_395967 = {}
    
    # Call to configuration(...): (line 16)
    # Processing the call keyword arguments (line 16)
    str_395962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 35), 'str', '')
    keyword_395963 = str_395962
    kwargs_395964 = {'top_path': keyword_395963}
    # Getting the type of 'configuration' (line 16)
    configuration_395961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 16)
    configuration_call_result_395965 = invoke(stypy.reporting.localization.Localization(__file__, 16, 12), configuration_395961, *[], **kwargs_395964)
    
    # Obtaining the member 'todict' of a type (line 16)
    todict_395966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 12), configuration_call_result_395965, 'todict')
    # Calling todict(args, kwargs) (line 16)
    todict_call_result_395968 = invoke(stypy.reporting.localization.Localization(__file__, 16, 12), todict_395966, *[], **kwargs_395967)
    
    kwargs_395969 = {'todict_call_result_395968': todict_call_result_395968}
    # Getting the type of 'setup' (line 16)
    setup_395960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 16)
    setup_call_result_395970 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), setup_395960, *[], **kwargs_395969)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
