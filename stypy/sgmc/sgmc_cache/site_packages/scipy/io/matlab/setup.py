
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: 
4: def configuration(parent_package='io',top_path=None):
5:     from numpy.distutils.misc_util import Configuration
6:     config = Configuration('matlab', parent_package, top_path)
7:     config.add_extension('streams', sources=['streams.c'])
8:     config.add_extension('mio_utils', sources=['mio_utils.c'])
9:     config.add_extension('mio5_utils', sources=['mio5_utils.c'])
10:     config.add_data_dir('tests')
11:     return config
12: 
13: if __name__ == '__main__':
14:     from numpy.distutils.core import setup
15:     setup(**configuration(top_path='').todict())
16: 

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
    str_137947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 33), 'str', 'io')
    # Getting the type of 'None' (line 4)
    None_137948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 47), 'None')
    defaults = [str_137947, None_137948]
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
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
    import_137949 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util')

    if (type(import_137949) is not StypyTypeError):

        if (import_137949 != 'pyd_module'):
            __import__(import_137949)
            sys_modules_137950 = sys.modules[import_137949]
            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', sys_modules_137950.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 5, 4), __file__, sys_modules_137950, sys_modules_137950.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', import_137949)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')
    
    
    # Assigning a Call to a Name (line 6):
    
    # Call to Configuration(...): (line 6)
    # Processing the call arguments (line 6)
    str_137952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 27), 'str', 'matlab')
    # Getting the type of 'parent_package' (line 6)
    parent_package_137953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 37), 'parent_package', False)
    # Getting the type of 'top_path' (line 6)
    top_path_137954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 53), 'top_path', False)
    # Processing the call keyword arguments (line 6)
    kwargs_137955 = {}
    # Getting the type of 'Configuration' (line 6)
    Configuration_137951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 6)
    Configuration_call_result_137956 = invoke(stypy.reporting.localization.Localization(__file__, 6, 13), Configuration_137951, *[str_137952, parent_package_137953, top_path_137954], **kwargs_137955)
    
    # Assigning a type to the variable 'config' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'config', Configuration_call_result_137956)
    
    # Call to add_extension(...): (line 7)
    # Processing the call arguments (line 7)
    str_137959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 25), 'str', 'streams')
    # Processing the call keyword arguments (line 7)
    
    # Obtaining an instance of the builtin type 'list' (line 7)
    list_137960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 44), 'list')
    # Adding type elements to the builtin type 'list' instance (line 7)
    # Adding element type (line 7)
    str_137961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 45), 'str', 'streams.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 44), list_137960, str_137961)
    
    keyword_137962 = list_137960
    kwargs_137963 = {'sources': keyword_137962}
    # Getting the type of 'config' (line 7)
    config_137957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 7)
    add_extension_137958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), config_137957, 'add_extension')
    # Calling add_extension(args, kwargs) (line 7)
    add_extension_call_result_137964 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), add_extension_137958, *[str_137959], **kwargs_137963)
    
    
    # Call to add_extension(...): (line 8)
    # Processing the call arguments (line 8)
    str_137967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 25), 'str', 'mio_utils')
    # Processing the call keyword arguments (line 8)
    
    # Obtaining an instance of the builtin type 'list' (line 8)
    list_137968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 46), 'list')
    # Adding type elements to the builtin type 'list' instance (line 8)
    # Adding element type (line 8)
    str_137969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 47), 'str', 'mio_utils.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 46), list_137968, str_137969)
    
    keyword_137970 = list_137968
    kwargs_137971 = {'sources': keyword_137970}
    # Getting the type of 'config' (line 8)
    config_137965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 8)
    add_extension_137966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 4), config_137965, 'add_extension')
    # Calling add_extension(args, kwargs) (line 8)
    add_extension_call_result_137972 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), add_extension_137966, *[str_137967], **kwargs_137971)
    
    
    # Call to add_extension(...): (line 9)
    # Processing the call arguments (line 9)
    str_137975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 25), 'str', 'mio5_utils')
    # Processing the call keyword arguments (line 9)
    
    # Obtaining an instance of the builtin type 'list' (line 9)
    list_137976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 47), 'list')
    # Adding type elements to the builtin type 'list' instance (line 9)
    # Adding element type (line 9)
    str_137977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 48), 'str', 'mio5_utils.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 47), list_137976, str_137977)
    
    keyword_137978 = list_137976
    kwargs_137979 = {'sources': keyword_137978}
    # Getting the type of 'config' (line 9)
    config_137973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 9)
    add_extension_137974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), config_137973, 'add_extension')
    # Calling add_extension(args, kwargs) (line 9)
    add_extension_call_result_137980 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), add_extension_137974, *[str_137975], **kwargs_137979)
    
    
    # Call to add_data_dir(...): (line 10)
    # Processing the call arguments (line 10)
    str_137983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 10)
    kwargs_137984 = {}
    # Getting the type of 'config' (line 10)
    config_137981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 10)
    add_data_dir_137982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 4), config_137981, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 10)
    add_data_dir_call_result_137985 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), add_data_dir_137982, *[str_137983], **kwargs_137984)
    
    # Getting the type of 'config' (line 11)
    config_137986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type', config_137986)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_137987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_137987)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_137987

# Assigning a type to the variable 'configuration' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 14)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
    import_137988 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 4), 'numpy.distutils.core')

    if (type(import_137988) is not StypyTypeError):

        if (import_137988 != 'pyd_module'):
            __import__(import_137988)
            sys_modules_137989 = sys.modules[import_137988]
            import_from_module(stypy.reporting.localization.Localization(__file__, 14, 4), 'numpy.distutils.core', sys_modules_137989.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 14, 4), __file__, sys_modules_137989, sys_modules_137989.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 14, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'numpy.distutils.core', import_137988)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')
    
    
    # Call to setup(...): (line 15)
    # Processing the call keyword arguments (line 15)
    
    # Call to todict(...): (line 15)
    # Processing the call keyword arguments (line 15)
    kwargs_137997 = {}
    
    # Call to configuration(...): (line 15)
    # Processing the call keyword arguments (line 15)
    str_137992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 35), 'str', '')
    keyword_137993 = str_137992
    kwargs_137994 = {'top_path': keyword_137993}
    # Getting the type of 'configuration' (line 15)
    configuration_137991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 15)
    configuration_call_result_137995 = invoke(stypy.reporting.localization.Localization(__file__, 15, 12), configuration_137991, *[], **kwargs_137994)
    
    # Obtaining the member 'todict' of a type (line 15)
    todict_137996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 12), configuration_call_result_137995, 'todict')
    # Calling todict(args, kwargs) (line 15)
    todict_call_result_137998 = invoke(stypy.reporting.localization.Localization(__file__, 15, 12), todict_137996, *[], **kwargs_137997)
    
    kwargs_137999 = {'todict_call_result_137998': todict_call_result_137998}
    # Getting the type of 'setup' (line 15)
    setup_137990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 15)
    setup_call_result_138000 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), setup_137990, *[], **kwargs_137999)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
