
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from os.path import join
4: 
5: 
6: def configuration(parent_package='',top_path=None):
7:     from numpy.distutils.misc_util import Configuration
8:     config = Configuration('stats', parent_package, top_path)
9: 
10:     config.add_data_dir('tests')
11: 
12:     statlib_src = [join('statlib', '*.f')]
13:     config.add_library('statlib', sources=statlib_src)
14: 
15:     # add statlib module
16:     config.add_extension('statlib',
17:         sources=['statlib.pyf'],
18:         f2py_options=['--no-wrap-functions'],
19:         libraries=['statlib'],
20:         depends=statlib_src
21:     )
22: 
23:     # add _stats module
24:     config.add_extension('_stats',
25:         sources=['_stats.c'],
26:     )
27: 
28:     # add mvn module
29:     config.add_extension('mvn',
30:         sources=['mvn.pyf','mvndst.f'],
31:     )
32: 
33:     return config
34: 
35: if __name__ == '__main__':
36:     from numpy.distutils.core import setup
37:     setup(**configuration(top_path='').todict())
38: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from os.path import join' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_579767 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path')

if (type(import_579767) is not StypyTypeError):

    if (import_579767 != 'pyd_module'):
        __import__(import_579767)
        sys_modules_579768 = sys.modules[import_579767]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', sys_modules_579768.module_type_store, module_type_store, ['join'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_579768, sys_modules_579768.module_type_store, module_type_store)
    else:
        from os.path import join

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', None, module_type_store, ['join'], [join])

else:
    # Assigning a type to the variable 'os.path' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', import_579767)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')


@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_579769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 33), 'str', '')
    # Getting the type of 'None' (line 6)
    None_579770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 45), 'None')
    defaults = [str_579769, None_579770]
    # Create a new context for function 'configuration'
    module_type_store = module_type_store.open_function_context('configuration', 6, 0, False)
    
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

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 4))
    
    # 'from numpy.distutils.misc_util import Configuration' statement (line 7)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
    import_579771 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util')

    if (type(import_579771) is not StypyTypeError):

        if (import_579771 != 'pyd_module'):
            __import__(import_579771)
            sys_modules_579772 = sys.modules[import_579771]
            import_from_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util', sys_modules_579772.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 7, 4), __file__, sys_modules_579772, sys_modules_579772.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util', import_579771)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')
    
    
    # Assigning a Call to a Name (line 8):
    
    # Call to Configuration(...): (line 8)
    # Processing the call arguments (line 8)
    str_579774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 27), 'str', 'stats')
    # Getting the type of 'parent_package' (line 8)
    parent_package_579775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 36), 'parent_package', False)
    # Getting the type of 'top_path' (line 8)
    top_path_579776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 52), 'top_path', False)
    # Processing the call keyword arguments (line 8)
    kwargs_579777 = {}
    # Getting the type of 'Configuration' (line 8)
    Configuration_579773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 8)
    Configuration_call_result_579778 = invoke(stypy.reporting.localization.Localization(__file__, 8, 13), Configuration_579773, *[str_579774, parent_package_579775, top_path_579776], **kwargs_579777)
    
    # Assigning a type to the variable 'config' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'config', Configuration_call_result_579778)
    
    # Call to add_data_dir(...): (line 10)
    # Processing the call arguments (line 10)
    str_579781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 10)
    kwargs_579782 = {}
    # Getting the type of 'config' (line 10)
    config_579779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 10)
    add_data_dir_579780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 4), config_579779, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 10)
    add_data_dir_call_result_579783 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), add_data_dir_579780, *[str_579781], **kwargs_579782)
    
    
    # Assigning a List to a Name (line 12):
    
    # Obtaining an instance of the builtin type 'list' (line 12)
    list_579784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 12)
    # Adding element type (line 12)
    
    # Call to join(...): (line 12)
    # Processing the call arguments (line 12)
    str_579786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 24), 'str', 'statlib')
    str_579787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 35), 'str', '*.f')
    # Processing the call keyword arguments (line 12)
    kwargs_579788 = {}
    # Getting the type of 'join' (line 12)
    join_579785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 19), 'join', False)
    # Calling join(args, kwargs) (line 12)
    join_call_result_579789 = invoke(stypy.reporting.localization.Localization(__file__, 12, 19), join_579785, *[str_579786, str_579787], **kwargs_579788)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 18), list_579784, join_call_result_579789)
    
    # Assigning a type to the variable 'statlib_src' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'statlib_src', list_579784)
    
    # Call to add_library(...): (line 13)
    # Processing the call arguments (line 13)
    str_579792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 23), 'str', 'statlib')
    # Processing the call keyword arguments (line 13)
    # Getting the type of 'statlib_src' (line 13)
    statlib_src_579793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 42), 'statlib_src', False)
    keyword_579794 = statlib_src_579793
    kwargs_579795 = {'sources': keyword_579794}
    # Getting the type of 'config' (line 13)
    config_579790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'config', False)
    # Obtaining the member 'add_library' of a type (line 13)
    add_library_579791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 4), config_579790, 'add_library')
    # Calling add_library(args, kwargs) (line 13)
    add_library_call_result_579796 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), add_library_579791, *[str_579792], **kwargs_579795)
    
    
    # Call to add_extension(...): (line 16)
    # Processing the call arguments (line 16)
    str_579799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'str', 'statlib')
    # Processing the call keyword arguments (line 16)
    
    # Obtaining an instance of the builtin type 'list' (line 17)
    list_579800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 17)
    # Adding element type (line 17)
    str_579801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 17), 'str', 'statlib.pyf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 16), list_579800, str_579801)
    
    keyword_579802 = list_579800
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_579803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    # Adding element type (line 18)
    str_579804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 22), 'str', '--no-wrap-functions')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 21), list_579803, str_579804)
    
    keyword_579805 = list_579803
    
    # Obtaining an instance of the builtin type 'list' (line 19)
    list_579806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 19)
    # Adding element type (line 19)
    str_579807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 19), 'str', 'statlib')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 18), list_579806, str_579807)
    
    keyword_579808 = list_579806
    # Getting the type of 'statlib_src' (line 20)
    statlib_src_579809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'statlib_src', False)
    keyword_579810 = statlib_src_579809
    kwargs_579811 = {'libraries': keyword_579808, 'sources': keyword_579802, 'f2py_options': keyword_579805, 'depends': keyword_579810}
    # Getting the type of 'config' (line 16)
    config_579797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 16)
    add_extension_579798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), config_579797, 'add_extension')
    # Calling add_extension(args, kwargs) (line 16)
    add_extension_call_result_579812 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), add_extension_579798, *[str_579799], **kwargs_579811)
    
    
    # Call to add_extension(...): (line 24)
    # Processing the call arguments (line 24)
    str_579815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'str', '_stats')
    # Processing the call keyword arguments (line 24)
    
    # Obtaining an instance of the builtin type 'list' (line 25)
    list_579816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 25)
    # Adding element type (line 25)
    str_579817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 17), 'str', '_stats.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 16), list_579816, str_579817)
    
    keyword_579818 = list_579816
    kwargs_579819 = {'sources': keyword_579818}
    # Getting the type of 'config' (line 24)
    config_579813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 24)
    add_extension_579814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 4), config_579813, 'add_extension')
    # Calling add_extension(args, kwargs) (line 24)
    add_extension_call_result_579820 = invoke(stypy.reporting.localization.Localization(__file__, 24, 4), add_extension_579814, *[str_579815], **kwargs_579819)
    
    
    # Call to add_extension(...): (line 29)
    # Processing the call arguments (line 29)
    str_579823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 25), 'str', 'mvn')
    # Processing the call keyword arguments (line 29)
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_579824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    str_579825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 17), 'str', 'mvn.pyf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 16), list_579824, str_579825)
    # Adding element type (line 30)
    str_579826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 27), 'str', 'mvndst.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 16), list_579824, str_579826)
    
    keyword_579827 = list_579824
    kwargs_579828 = {'sources': keyword_579827}
    # Getting the type of 'config' (line 29)
    config_579821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 29)
    add_extension_579822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 4), config_579821, 'add_extension')
    # Calling add_extension(args, kwargs) (line 29)
    add_extension_call_result_579829 = invoke(stypy.reporting.localization.Localization(__file__, 29, 4), add_extension_579822, *[str_579823], **kwargs_579828)
    
    # Getting the type of 'config' (line 33)
    config_579830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type', config_579830)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_579831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_579831)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_579831

# Assigning a type to the variable 'configuration' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 36, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 36)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
    import_579832 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 36, 4), 'numpy.distutils.core')

    if (type(import_579832) is not StypyTypeError):

        if (import_579832 != 'pyd_module'):
            __import__(import_579832)
            sys_modules_579833 = sys.modules[import_579832]
            import_from_module(stypy.reporting.localization.Localization(__file__, 36, 4), 'numpy.distutils.core', sys_modules_579833.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 36, 4), __file__, sys_modules_579833, sys_modules_579833.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 36, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'numpy.distutils.core', import_579832)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')
    
    
    # Call to setup(...): (line 37)
    # Processing the call keyword arguments (line 37)
    
    # Call to todict(...): (line 37)
    # Processing the call keyword arguments (line 37)
    kwargs_579841 = {}
    
    # Call to configuration(...): (line 37)
    # Processing the call keyword arguments (line 37)
    str_579836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 35), 'str', '')
    keyword_579837 = str_579836
    kwargs_579838 = {'top_path': keyword_579837}
    # Getting the type of 'configuration' (line 37)
    configuration_579835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 37)
    configuration_call_result_579839 = invoke(stypy.reporting.localization.Localization(__file__, 37, 12), configuration_579835, *[], **kwargs_579838)
    
    # Obtaining the member 'todict' of a type (line 37)
    todict_579840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 12), configuration_call_result_579839, 'todict')
    # Calling todict(args, kwargs) (line 37)
    todict_call_result_579842 = invoke(stypy.reporting.localization.Localization(__file__, 37, 12), todict_579840, *[], **kwargs_579841)
    
    kwargs_579843 = {'todict_call_result_579842': todict_call_result_579842}
    # Getting the type of 'setup' (line 37)
    setup_579834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 37)
    setup_call_result_579844 = invoke(stypy.reporting.localization.Localization(__file__, 37, 4), setup_579834, *[], **kwargs_579843)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
