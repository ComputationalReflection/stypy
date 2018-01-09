
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: 
4: def configuration(parent_package='', top_path=None):
5:     import numpy
6:     from numpy.distutils.misc_util import Configuration
7: 
8:     config = Configuration('csgraph', parent_package, top_path)
9: 
10:     config.add_data_dir('tests')
11: 
12:     config.add_extension('_shortest_path',
13:          sources=['_shortest_path.c'],
14:          include_dirs=[numpy.get_include()])
15: 
16:     config.add_extension('_traversal',
17:          sources=['_traversal.c'],
18:          include_dirs=[numpy.get_include()])
19: 
20:     config.add_extension('_min_spanning_tree',
21:          sources=['_min_spanning_tree.c'],
22:          include_dirs=[numpy.get_include()])
23:     
24:     config.add_extension('_reordering',
25:          sources=['_reordering.c'],
26:          include_dirs=[numpy.get_include()])
27: 
28:     config.add_extension('_tools',
29:          sources=['_tools.c'],
30:          include_dirs=[numpy.get_include()])
31: 
32:     return config
33: 

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
    str_381109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 33), 'str', '')
    # Getting the type of 'None' (line 4)
    None_381110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 46), 'None')
    defaults = [str_381109, None_381110]
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
    
    # 'import numpy' statement (line 5)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')
    import_381111 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy')

    if (type(import_381111) is not StypyTypeError):

        if (import_381111 != 'pyd_module'):
            __import__(import_381111)
            sys_modules_381112 = sys.modules[import_381111]
            import_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy', sys_modules_381112.module_type_store, module_type_store)
        else:
            import numpy

            import_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy', numpy, module_type_store)

    else:
        # Assigning a type to the variable 'numpy' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy', import_381111)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 4))
    
    # 'from numpy.distutils.misc_util import Configuration' statement (line 6)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')
    import_381113 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'numpy.distutils.misc_util')

    if (type(import_381113) is not StypyTypeError):

        if (import_381113 != 'pyd_module'):
            __import__(import_381113)
            sys_modules_381114 = sys.modules[import_381113]
            import_from_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'numpy.distutils.misc_util', sys_modules_381114.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 6, 4), __file__, sys_modules_381114, sys_modules_381114.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'numpy.distutils.misc_util', import_381113)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')
    
    
    # Assigning a Call to a Name (line 8):
    
    # Call to Configuration(...): (line 8)
    # Processing the call arguments (line 8)
    str_381116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 27), 'str', 'csgraph')
    # Getting the type of 'parent_package' (line 8)
    parent_package_381117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 38), 'parent_package', False)
    # Getting the type of 'top_path' (line 8)
    top_path_381118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 54), 'top_path', False)
    # Processing the call keyword arguments (line 8)
    kwargs_381119 = {}
    # Getting the type of 'Configuration' (line 8)
    Configuration_381115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 8)
    Configuration_call_result_381120 = invoke(stypy.reporting.localization.Localization(__file__, 8, 13), Configuration_381115, *[str_381116, parent_package_381117, top_path_381118], **kwargs_381119)
    
    # Assigning a type to the variable 'config' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'config', Configuration_call_result_381120)
    
    # Call to add_data_dir(...): (line 10)
    # Processing the call arguments (line 10)
    str_381123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 10)
    kwargs_381124 = {}
    # Getting the type of 'config' (line 10)
    config_381121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 10)
    add_data_dir_381122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 4), config_381121, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 10)
    add_data_dir_call_result_381125 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), add_data_dir_381122, *[str_381123], **kwargs_381124)
    
    
    # Call to add_extension(...): (line 12)
    # Processing the call arguments (line 12)
    str_381128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 25), 'str', '_shortest_path')
    # Processing the call keyword arguments (line 12)
    
    # Obtaining an instance of the builtin type 'list' (line 13)
    list_381129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 13)
    # Adding element type (line 13)
    str_381130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 18), 'str', '_shortest_path.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 17), list_381129, str_381130)
    
    keyword_381131 = list_381129
    
    # Obtaining an instance of the builtin type 'list' (line 14)
    list_381132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 14)
    # Adding element type (line 14)
    
    # Call to get_include(...): (line 14)
    # Processing the call keyword arguments (line 14)
    kwargs_381135 = {}
    # Getting the type of 'numpy' (line 14)
    numpy_381133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 23), 'numpy', False)
    # Obtaining the member 'get_include' of a type (line 14)
    get_include_381134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 23), numpy_381133, 'get_include')
    # Calling get_include(args, kwargs) (line 14)
    get_include_call_result_381136 = invoke(stypy.reporting.localization.Localization(__file__, 14, 23), get_include_381134, *[], **kwargs_381135)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 22), list_381132, get_include_call_result_381136)
    
    keyword_381137 = list_381132
    kwargs_381138 = {'sources': keyword_381131, 'include_dirs': keyword_381137}
    # Getting the type of 'config' (line 12)
    config_381126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 12)
    add_extension_381127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), config_381126, 'add_extension')
    # Calling add_extension(args, kwargs) (line 12)
    add_extension_call_result_381139 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), add_extension_381127, *[str_381128], **kwargs_381138)
    
    
    # Call to add_extension(...): (line 16)
    # Processing the call arguments (line 16)
    str_381142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'str', '_traversal')
    # Processing the call keyword arguments (line 16)
    
    # Obtaining an instance of the builtin type 'list' (line 17)
    list_381143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 17)
    # Adding element type (line 17)
    str_381144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 18), 'str', '_traversal.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 17), list_381143, str_381144)
    
    keyword_381145 = list_381143
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_381146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    # Adding element type (line 18)
    
    # Call to get_include(...): (line 18)
    # Processing the call keyword arguments (line 18)
    kwargs_381149 = {}
    # Getting the type of 'numpy' (line 18)
    numpy_381147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 23), 'numpy', False)
    # Obtaining the member 'get_include' of a type (line 18)
    get_include_381148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 23), numpy_381147, 'get_include')
    # Calling get_include(args, kwargs) (line 18)
    get_include_call_result_381150 = invoke(stypy.reporting.localization.Localization(__file__, 18, 23), get_include_381148, *[], **kwargs_381149)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 22), list_381146, get_include_call_result_381150)
    
    keyword_381151 = list_381146
    kwargs_381152 = {'sources': keyword_381145, 'include_dirs': keyword_381151}
    # Getting the type of 'config' (line 16)
    config_381140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 16)
    add_extension_381141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), config_381140, 'add_extension')
    # Calling add_extension(args, kwargs) (line 16)
    add_extension_call_result_381153 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), add_extension_381141, *[str_381142], **kwargs_381152)
    
    
    # Call to add_extension(...): (line 20)
    # Processing the call arguments (line 20)
    str_381156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 25), 'str', '_min_spanning_tree')
    # Processing the call keyword arguments (line 20)
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_381157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    str_381158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 18), 'str', '_min_spanning_tree.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 17), list_381157, str_381158)
    
    keyword_381159 = list_381157
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_381160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    
    # Call to get_include(...): (line 22)
    # Processing the call keyword arguments (line 22)
    kwargs_381163 = {}
    # Getting the type of 'numpy' (line 22)
    numpy_381161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 23), 'numpy', False)
    # Obtaining the member 'get_include' of a type (line 22)
    get_include_381162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 23), numpy_381161, 'get_include')
    # Calling get_include(args, kwargs) (line 22)
    get_include_call_result_381164 = invoke(stypy.reporting.localization.Localization(__file__, 22, 23), get_include_381162, *[], **kwargs_381163)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 22), list_381160, get_include_call_result_381164)
    
    keyword_381165 = list_381160
    kwargs_381166 = {'sources': keyword_381159, 'include_dirs': keyword_381165}
    # Getting the type of 'config' (line 20)
    config_381154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 20)
    add_extension_381155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 4), config_381154, 'add_extension')
    # Calling add_extension(args, kwargs) (line 20)
    add_extension_call_result_381167 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), add_extension_381155, *[str_381156], **kwargs_381166)
    
    
    # Call to add_extension(...): (line 24)
    # Processing the call arguments (line 24)
    str_381170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'str', '_reordering')
    # Processing the call keyword arguments (line 24)
    
    # Obtaining an instance of the builtin type 'list' (line 25)
    list_381171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 25)
    # Adding element type (line 25)
    str_381172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 18), 'str', '_reordering.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 17), list_381171, str_381172)
    
    keyword_381173 = list_381171
    
    # Obtaining an instance of the builtin type 'list' (line 26)
    list_381174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 26)
    # Adding element type (line 26)
    
    # Call to get_include(...): (line 26)
    # Processing the call keyword arguments (line 26)
    kwargs_381177 = {}
    # Getting the type of 'numpy' (line 26)
    numpy_381175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 23), 'numpy', False)
    # Obtaining the member 'get_include' of a type (line 26)
    get_include_381176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 23), numpy_381175, 'get_include')
    # Calling get_include(args, kwargs) (line 26)
    get_include_call_result_381178 = invoke(stypy.reporting.localization.Localization(__file__, 26, 23), get_include_381176, *[], **kwargs_381177)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 22), list_381174, get_include_call_result_381178)
    
    keyword_381179 = list_381174
    kwargs_381180 = {'sources': keyword_381173, 'include_dirs': keyword_381179}
    # Getting the type of 'config' (line 24)
    config_381168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 24)
    add_extension_381169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 4), config_381168, 'add_extension')
    # Calling add_extension(args, kwargs) (line 24)
    add_extension_call_result_381181 = invoke(stypy.reporting.localization.Localization(__file__, 24, 4), add_extension_381169, *[str_381170], **kwargs_381180)
    
    
    # Call to add_extension(...): (line 28)
    # Processing the call arguments (line 28)
    str_381184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 25), 'str', '_tools')
    # Processing the call keyword arguments (line 28)
    
    # Obtaining an instance of the builtin type 'list' (line 29)
    list_381185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 29)
    # Adding element type (line 29)
    str_381186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 18), 'str', '_tools.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 17), list_381185, str_381186)
    
    keyword_381187 = list_381185
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_381188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    
    # Call to get_include(...): (line 30)
    # Processing the call keyword arguments (line 30)
    kwargs_381191 = {}
    # Getting the type of 'numpy' (line 30)
    numpy_381189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 23), 'numpy', False)
    # Obtaining the member 'get_include' of a type (line 30)
    get_include_381190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 23), numpy_381189, 'get_include')
    # Calling get_include(args, kwargs) (line 30)
    get_include_call_result_381192 = invoke(stypy.reporting.localization.Localization(__file__, 30, 23), get_include_381190, *[], **kwargs_381191)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 22), list_381188, get_include_call_result_381192)
    
    keyword_381193 = list_381188
    kwargs_381194 = {'sources': keyword_381187, 'include_dirs': keyword_381193}
    # Getting the type of 'config' (line 28)
    config_381182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 28)
    add_extension_381183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 4), config_381182, 'add_extension')
    # Calling add_extension(args, kwargs) (line 28)
    add_extension_call_result_381195 = invoke(stypy.reporting.localization.Localization(__file__, 28, 4), add_extension_381183, *[str_381184], **kwargs_381194)
    
    # Getting the type of 'config' (line 32)
    config_381196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type', config_381196)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_381197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_381197)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_381197

# Assigning a type to the variable 'configuration' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'configuration', configuration)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
