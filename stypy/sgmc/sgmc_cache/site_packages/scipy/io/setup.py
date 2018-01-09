
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: 
4: def configuration(parent_package='',top_path=None):
5:     from numpy.distutils.misc_util import Configuration
6:     config = Configuration('io', parent_package, top_path)
7: 
8:     config.add_extension('_test_fortran',
9:                          sources=['_test_fortran.pyf', '_test_fortran.f'])
10: 
11:     config.add_data_dir('tests')
12:     config.add_subpackage('matlab')
13:     config.add_subpackage('arff')
14:     config.add_subpackage('harwell_boeing')
15:     return config
16: 
17: if __name__ == '__main__':
18:     from numpy.distutils.core import setup
19:     setup(**configuration(top_path='').todict())
20: 

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
    str_126940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 33), 'str', '')
    # Getting the type of 'None' (line 4)
    None_126941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 45), 'None')
    defaults = [str_126940, None_126941]
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
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/')
    import_126942 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util')

    if (type(import_126942) is not StypyTypeError):

        if (import_126942 != 'pyd_module'):
            __import__(import_126942)
            sys_modules_126943 = sys.modules[import_126942]
            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', sys_modules_126943.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 5, 4), __file__, sys_modules_126943, sys_modules_126943.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', import_126942)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/')
    
    
    # Assigning a Call to a Name (line 6):
    
    # Call to Configuration(...): (line 6)
    # Processing the call arguments (line 6)
    str_126945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 27), 'str', 'io')
    # Getting the type of 'parent_package' (line 6)
    parent_package_126946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 33), 'parent_package', False)
    # Getting the type of 'top_path' (line 6)
    top_path_126947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 49), 'top_path', False)
    # Processing the call keyword arguments (line 6)
    kwargs_126948 = {}
    # Getting the type of 'Configuration' (line 6)
    Configuration_126944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 6)
    Configuration_call_result_126949 = invoke(stypy.reporting.localization.Localization(__file__, 6, 13), Configuration_126944, *[str_126945, parent_package_126946, top_path_126947], **kwargs_126948)
    
    # Assigning a type to the variable 'config' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'config', Configuration_call_result_126949)
    
    # Call to add_extension(...): (line 8)
    # Processing the call arguments (line 8)
    str_126952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 25), 'str', '_test_fortran')
    # Processing the call keyword arguments (line 8)
    
    # Obtaining an instance of the builtin type 'list' (line 9)
    list_126953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 9)
    # Adding element type (line 9)
    str_126954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 34), 'str', '_test_fortran.pyf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 33), list_126953, str_126954)
    # Adding element type (line 9)
    str_126955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 55), 'str', '_test_fortran.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 33), list_126953, str_126955)
    
    keyword_126956 = list_126953
    kwargs_126957 = {'sources': keyword_126956}
    # Getting the type of 'config' (line 8)
    config_126950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 8)
    add_extension_126951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 4), config_126950, 'add_extension')
    # Calling add_extension(args, kwargs) (line 8)
    add_extension_call_result_126958 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), add_extension_126951, *[str_126952], **kwargs_126957)
    
    
    # Call to add_data_dir(...): (line 11)
    # Processing the call arguments (line 11)
    str_126961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 11)
    kwargs_126962 = {}
    # Getting the type of 'config' (line 11)
    config_126959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 11)
    add_data_dir_126960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), config_126959, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 11)
    add_data_dir_call_result_126963 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), add_data_dir_126960, *[str_126961], **kwargs_126962)
    
    
    # Call to add_subpackage(...): (line 12)
    # Processing the call arguments (line 12)
    str_126966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 26), 'str', 'matlab')
    # Processing the call keyword arguments (line 12)
    kwargs_126967 = {}
    # Getting the type of 'config' (line 12)
    config_126964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 12)
    add_subpackage_126965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), config_126964, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 12)
    add_subpackage_call_result_126968 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), add_subpackage_126965, *[str_126966], **kwargs_126967)
    
    
    # Call to add_subpackage(...): (line 13)
    # Processing the call arguments (line 13)
    str_126971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 26), 'str', 'arff')
    # Processing the call keyword arguments (line 13)
    kwargs_126972 = {}
    # Getting the type of 'config' (line 13)
    config_126969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 13)
    add_subpackage_126970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 4), config_126969, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 13)
    add_subpackage_call_result_126973 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), add_subpackage_126970, *[str_126971], **kwargs_126972)
    
    
    # Call to add_subpackage(...): (line 14)
    # Processing the call arguments (line 14)
    str_126976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 26), 'str', 'harwell_boeing')
    # Processing the call keyword arguments (line 14)
    kwargs_126977 = {}
    # Getting the type of 'config' (line 14)
    config_126974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 14)
    add_subpackage_126975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 4), config_126974, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 14)
    add_subpackage_call_result_126978 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), add_subpackage_126975, *[str_126976], **kwargs_126977)
    
    # Getting the type of 'config' (line 15)
    config_126979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type', config_126979)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_126980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126980)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_126980

# Assigning a type to the variable 'configuration' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 18)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/')
    import_126981 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'numpy.distutils.core')

    if (type(import_126981) is not StypyTypeError):

        if (import_126981 != 'pyd_module'):
            __import__(import_126981)
            sys_modules_126982 = sys.modules[import_126981]
            import_from_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'numpy.distutils.core', sys_modules_126982.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 18, 4), __file__, sys_modules_126982, sys_modules_126982.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'numpy.distutils.core', import_126981)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/')
    
    
    # Call to setup(...): (line 19)
    # Processing the call keyword arguments (line 19)
    
    # Call to todict(...): (line 19)
    # Processing the call keyword arguments (line 19)
    kwargs_126990 = {}
    
    # Call to configuration(...): (line 19)
    # Processing the call keyword arguments (line 19)
    str_126985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 35), 'str', '')
    keyword_126986 = str_126985
    kwargs_126987 = {'top_path': keyword_126986}
    # Getting the type of 'configuration' (line 19)
    configuration_126984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 19)
    configuration_call_result_126988 = invoke(stypy.reporting.localization.Localization(__file__, 19, 12), configuration_126984, *[], **kwargs_126987)
    
    # Obtaining the member 'todict' of a type (line 19)
    todict_126989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 12), configuration_call_result_126988, 'todict')
    # Calling todict(args, kwargs) (line 19)
    todict_call_result_126991 = invoke(stypy.reporting.localization.Localization(__file__, 19, 12), todict_126989, *[], **kwargs_126990)
    
    kwargs_126992 = {'todict_call_result_126991': todict_call_result_126991}
    # Getting the type of 'setup' (line 19)
    setup_126983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 19)
    setup_call_result_126993 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), setup_126983, *[], **kwargs_126992)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
