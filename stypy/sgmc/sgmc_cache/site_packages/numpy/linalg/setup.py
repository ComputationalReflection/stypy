
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function
2: 
3: import os
4: import sys
5: 
6: def configuration(parent_package='', top_path=None):
7:     from numpy.distutils.misc_util import Configuration
8:     from numpy.distutils.system_info import get_info
9:     config = Configuration('linalg', parent_package, top_path)
10: 
11:     config.add_data_dir('tests')
12: 
13:     # Configure lapack_lite
14: 
15:     src_dir = 'lapack_lite'
16:     lapack_lite_src = [
17:         os.path.join(src_dir, 'python_xerbla.c'),
18:         os.path.join(src_dir, 'zlapack_lite.c'),
19:         os.path.join(src_dir, 'dlapack_lite.c'),
20:         os.path.join(src_dir, 'blas_lite.c'),
21:         os.path.join(src_dir, 'dlamch.c'),
22:         os.path.join(src_dir, 'f2c_lite.c'),
23:     ]
24:     all_sources = config.paths(lapack_lite_src)
25: 
26:     lapack_info = get_info('lapack_opt', 0)  # and {}
27: 
28:     def get_lapack_lite_sources(ext, build_dir):
29:         if not lapack_info:
30:             print("### Warning:  Using unoptimized lapack ###")
31:             return all_sources
32:         else:
33:             if sys.platform == 'win32':
34:                 print("### Warning:  python_xerbla.c is disabled ###")
35:                 return []
36:             return [all_sources[0]]
37: 
38:     config.add_extension(
39:         'lapack_lite',
40:         sources=['lapack_litemodule.c', get_lapack_lite_sources],
41:         depends=['lapack_lite/f2c.h'],
42:         extra_info=lapack_info,
43:     )
44: 
45:     # umath_linalg module
46:     config.add_extension(
47:         '_umath_linalg',
48:         sources=['umath_linalg.c.src', get_lapack_lite_sources],
49:         depends=['lapack_lite/f2c.h'],
50:         extra_info=lapack_info,
51:         libraries=['npymath'],
52:     )
53:     return config
54: 
55: if __name__ == '__main__':
56:     from numpy.distutils.core import setup
57:     setup(configuration=configuration)
58: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)


@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_138394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 33), 'str', '')
    # Getting the type of 'None' (line 6)
    None_138395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 46), 'None')
    defaults = [str_138394, None_138395]
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
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/linalg/')
    import_138396 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util')

    if (type(import_138396) is not StypyTypeError):

        if (import_138396 != 'pyd_module'):
            __import__(import_138396)
            sys_modules_138397 = sys.modules[import_138396]
            import_from_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util', sys_modules_138397.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 7, 4), __file__, sys_modules_138397, sys_modules_138397.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util', import_138396)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/linalg/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 4))
    
    # 'from numpy.distutils.system_info import get_info' statement (line 8)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/linalg/')
    import_138398 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.system_info')

    if (type(import_138398) is not StypyTypeError):

        if (import_138398 != 'pyd_module'):
            __import__(import_138398)
            sys_modules_138399 = sys.modules[import_138398]
            import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.system_info', sys_modules_138399.module_type_store, module_type_store, ['get_info'])
            nest_module(stypy.reporting.localization.Localization(__file__, 8, 4), __file__, sys_modules_138399, sys_modules_138399.module_type_store, module_type_store)
        else:
            from numpy.distutils.system_info import get_info

            import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.system_info', None, module_type_store, ['get_info'], [get_info])

    else:
        # Assigning a type to the variable 'numpy.distutils.system_info' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.system_info', import_138398)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/linalg/')
    
    
    # Assigning a Call to a Name (line 9):
    
    # Call to Configuration(...): (line 9)
    # Processing the call arguments (line 9)
    str_138401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 27), 'str', 'linalg')
    # Getting the type of 'parent_package' (line 9)
    parent_package_138402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 37), 'parent_package', False)
    # Getting the type of 'top_path' (line 9)
    top_path_138403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 53), 'top_path', False)
    # Processing the call keyword arguments (line 9)
    kwargs_138404 = {}
    # Getting the type of 'Configuration' (line 9)
    Configuration_138400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 9)
    Configuration_call_result_138405 = invoke(stypy.reporting.localization.Localization(__file__, 9, 13), Configuration_138400, *[str_138401, parent_package_138402, top_path_138403], **kwargs_138404)
    
    # Assigning a type to the variable 'config' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'config', Configuration_call_result_138405)
    
    # Call to add_data_dir(...): (line 11)
    # Processing the call arguments (line 11)
    str_138408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 11)
    kwargs_138409 = {}
    # Getting the type of 'config' (line 11)
    config_138406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 11)
    add_data_dir_138407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), config_138406, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 11)
    add_data_dir_call_result_138410 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), add_data_dir_138407, *[str_138408], **kwargs_138409)
    
    
    # Assigning a Str to a Name (line 15):
    str_138411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 14), 'str', 'lapack_lite')
    # Assigning a type to the variable 'src_dir' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'src_dir', str_138411)
    
    # Assigning a List to a Name (line 16):
    
    # Obtaining an instance of the builtin type 'list' (line 16)
    list_138412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 16)
    # Adding element type (line 16)
    
    # Call to join(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'src_dir' (line 17)
    src_dir_138416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 21), 'src_dir', False)
    str_138417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 30), 'str', 'python_xerbla.c')
    # Processing the call keyword arguments (line 17)
    kwargs_138418 = {}
    # Getting the type of 'os' (line 17)
    os_138413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'os', False)
    # Obtaining the member 'path' of a type (line 17)
    path_138414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), os_138413, 'path')
    # Obtaining the member 'join' of a type (line 17)
    join_138415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), path_138414, 'join')
    # Calling join(args, kwargs) (line 17)
    join_call_result_138419 = invoke(stypy.reporting.localization.Localization(__file__, 17, 8), join_138415, *[src_dir_138416, str_138417], **kwargs_138418)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 22), list_138412, join_call_result_138419)
    # Adding element type (line 16)
    
    # Call to join(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'src_dir' (line 18)
    src_dir_138423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 21), 'src_dir', False)
    str_138424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 30), 'str', 'zlapack_lite.c')
    # Processing the call keyword arguments (line 18)
    kwargs_138425 = {}
    # Getting the type of 'os' (line 18)
    os_138420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'os', False)
    # Obtaining the member 'path' of a type (line 18)
    path_138421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), os_138420, 'path')
    # Obtaining the member 'join' of a type (line 18)
    join_138422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), path_138421, 'join')
    # Calling join(args, kwargs) (line 18)
    join_call_result_138426 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), join_138422, *[src_dir_138423, str_138424], **kwargs_138425)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 22), list_138412, join_call_result_138426)
    # Adding element type (line 16)
    
    # Call to join(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'src_dir' (line 19)
    src_dir_138430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 21), 'src_dir', False)
    str_138431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 30), 'str', 'dlapack_lite.c')
    # Processing the call keyword arguments (line 19)
    kwargs_138432 = {}
    # Getting the type of 'os' (line 19)
    os_138427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'os', False)
    # Obtaining the member 'path' of a type (line 19)
    path_138428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 8), os_138427, 'path')
    # Obtaining the member 'join' of a type (line 19)
    join_138429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 8), path_138428, 'join')
    # Calling join(args, kwargs) (line 19)
    join_call_result_138433 = invoke(stypy.reporting.localization.Localization(__file__, 19, 8), join_138429, *[src_dir_138430, str_138431], **kwargs_138432)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 22), list_138412, join_call_result_138433)
    # Adding element type (line 16)
    
    # Call to join(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'src_dir' (line 20)
    src_dir_138437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 21), 'src_dir', False)
    str_138438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 30), 'str', 'blas_lite.c')
    # Processing the call keyword arguments (line 20)
    kwargs_138439 = {}
    # Getting the type of 'os' (line 20)
    os_138434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'os', False)
    # Obtaining the member 'path' of a type (line 20)
    path_138435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), os_138434, 'path')
    # Obtaining the member 'join' of a type (line 20)
    join_138436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), path_138435, 'join')
    # Calling join(args, kwargs) (line 20)
    join_call_result_138440 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), join_138436, *[src_dir_138437, str_138438], **kwargs_138439)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 22), list_138412, join_call_result_138440)
    # Adding element type (line 16)
    
    # Call to join(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'src_dir' (line 21)
    src_dir_138444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 21), 'src_dir', False)
    str_138445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 30), 'str', 'dlamch.c')
    # Processing the call keyword arguments (line 21)
    kwargs_138446 = {}
    # Getting the type of 'os' (line 21)
    os_138441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'os', False)
    # Obtaining the member 'path' of a type (line 21)
    path_138442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), os_138441, 'path')
    # Obtaining the member 'join' of a type (line 21)
    join_138443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), path_138442, 'join')
    # Calling join(args, kwargs) (line 21)
    join_call_result_138447 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), join_138443, *[src_dir_138444, str_138445], **kwargs_138446)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 22), list_138412, join_call_result_138447)
    # Adding element type (line 16)
    
    # Call to join(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'src_dir' (line 22)
    src_dir_138451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 21), 'src_dir', False)
    str_138452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 30), 'str', 'f2c_lite.c')
    # Processing the call keyword arguments (line 22)
    kwargs_138453 = {}
    # Getting the type of 'os' (line 22)
    os_138448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'os', False)
    # Obtaining the member 'path' of a type (line 22)
    path_138449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), os_138448, 'path')
    # Obtaining the member 'join' of a type (line 22)
    join_138450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), path_138449, 'join')
    # Calling join(args, kwargs) (line 22)
    join_call_result_138454 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), join_138450, *[src_dir_138451, str_138452], **kwargs_138453)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 22), list_138412, join_call_result_138454)
    
    # Assigning a type to the variable 'lapack_lite_src' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'lapack_lite_src', list_138412)
    
    # Assigning a Call to a Name (line 24):
    
    # Call to paths(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'lapack_lite_src' (line 24)
    lapack_lite_src_138457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 31), 'lapack_lite_src', False)
    # Processing the call keyword arguments (line 24)
    kwargs_138458 = {}
    # Getting the type of 'config' (line 24)
    config_138455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 18), 'config', False)
    # Obtaining the member 'paths' of a type (line 24)
    paths_138456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 18), config_138455, 'paths')
    # Calling paths(args, kwargs) (line 24)
    paths_call_result_138459 = invoke(stypy.reporting.localization.Localization(__file__, 24, 18), paths_138456, *[lapack_lite_src_138457], **kwargs_138458)
    
    # Assigning a type to the variable 'all_sources' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'all_sources', paths_call_result_138459)
    
    # Assigning a Call to a Name (line 26):
    
    # Call to get_info(...): (line 26)
    # Processing the call arguments (line 26)
    str_138461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 27), 'str', 'lapack_opt')
    int_138462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 41), 'int')
    # Processing the call keyword arguments (line 26)
    kwargs_138463 = {}
    # Getting the type of 'get_info' (line 26)
    get_info_138460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), 'get_info', False)
    # Calling get_info(args, kwargs) (line 26)
    get_info_call_result_138464 = invoke(stypy.reporting.localization.Localization(__file__, 26, 18), get_info_138460, *[str_138461, int_138462], **kwargs_138463)
    
    # Assigning a type to the variable 'lapack_info' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'lapack_info', get_info_call_result_138464)

    @norecursion
    def get_lapack_lite_sources(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_lapack_lite_sources'
        module_type_store = module_type_store.open_function_context('get_lapack_lite_sources', 28, 4, False)
        
        # Passed parameters checking function
        get_lapack_lite_sources.stypy_localization = localization
        get_lapack_lite_sources.stypy_type_of_self = None
        get_lapack_lite_sources.stypy_type_store = module_type_store
        get_lapack_lite_sources.stypy_function_name = 'get_lapack_lite_sources'
        get_lapack_lite_sources.stypy_param_names_list = ['ext', 'build_dir']
        get_lapack_lite_sources.stypy_varargs_param_name = None
        get_lapack_lite_sources.stypy_kwargs_param_name = None
        get_lapack_lite_sources.stypy_call_defaults = defaults
        get_lapack_lite_sources.stypy_call_varargs = varargs
        get_lapack_lite_sources.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'get_lapack_lite_sources', ['ext', 'build_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_lapack_lite_sources', localization, ['ext', 'build_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_lapack_lite_sources(...)' code ##################

        
        
        # Getting the type of 'lapack_info' (line 29)
        lapack_info_138465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'lapack_info')
        # Applying the 'not' unary operator (line 29)
        result_not__138466 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 11), 'not', lapack_info_138465)
        
        # Testing the type of an if condition (line 29)
        if_condition_138467 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 8), result_not__138466)
        # Assigning a type to the variable 'if_condition_138467' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'if_condition_138467', if_condition_138467)
        # SSA begins for if statement (line 29)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 30)
        # Processing the call arguments (line 30)
        str_138469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 18), 'str', '### Warning:  Using unoptimized lapack ###')
        # Processing the call keyword arguments (line 30)
        kwargs_138470 = {}
        # Getting the type of 'print' (line 30)
        print_138468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'print', False)
        # Calling print(args, kwargs) (line 30)
        print_call_result_138471 = invoke(stypy.reporting.localization.Localization(__file__, 30, 12), print_138468, *[str_138469], **kwargs_138470)
        
        # Getting the type of 'all_sources' (line 31)
        all_sources_138472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 19), 'all_sources')
        # Assigning a type to the variable 'stypy_return_type' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'stypy_return_type', all_sources_138472)
        # SSA branch for the else part of an if statement (line 29)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'sys' (line 33)
        sys_138473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'sys')
        # Obtaining the member 'platform' of a type (line 33)
        platform_138474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 15), sys_138473, 'platform')
        str_138475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 31), 'str', 'win32')
        # Applying the binary operator '==' (line 33)
        result_eq_138476 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 15), '==', platform_138474, str_138475)
        
        # Testing the type of an if condition (line 33)
        if_condition_138477 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 12), result_eq_138476)
        # Assigning a type to the variable 'if_condition_138477' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'if_condition_138477', if_condition_138477)
        # SSA begins for if statement (line 33)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 34)
        # Processing the call arguments (line 34)
        str_138479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 22), 'str', '### Warning:  python_xerbla.c is disabled ###')
        # Processing the call keyword arguments (line 34)
        kwargs_138480 = {}
        # Getting the type of 'print' (line 34)
        print_138478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'print', False)
        # Calling print(args, kwargs) (line 34)
        print_call_result_138481 = invoke(stypy.reporting.localization.Localization(__file__, 34, 16), print_138478, *[str_138479], **kwargs_138480)
        
        
        # Obtaining an instance of the builtin type 'list' (line 35)
        list_138482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 35)
        
        # Assigning a type to the variable 'stypy_return_type' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'stypy_return_type', list_138482)
        # SSA join for if statement (line 33)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'list' (line 36)
        list_138483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 36)
        # Adding element type (line 36)
        
        # Obtaining the type of the subscript
        int_138484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 32), 'int')
        # Getting the type of 'all_sources' (line 36)
        all_sources_138485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 20), 'all_sources')
        # Obtaining the member '__getitem__' of a type (line 36)
        getitem___138486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 20), all_sources_138485, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 36)
        subscript_call_result_138487 = invoke(stypy.reporting.localization.Localization(__file__, 36, 20), getitem___138486, int_138484)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 19), list_138483, subscript_call_result_138487)
        
        # Assigning a type to the variable 'stypy_return_type' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'stypy_return_type', list_138483)
        # SSA join for if statement (line 29)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'get_lapack_lite_sources(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_lapack_lite_sources' in the type store
        # Getting the type of 'stypy_return_type' (line 28)
        stypy_return_type_138488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_138488)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_lapack_lite_sources'
        return stypy_return_type_138488

    # Assigning a type to the variable 'get_lapack_lite_sources' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'get_lapack_lite_sources', get_lapack_lite_sources)
    
    # Call to add_extension(...): (line 38)
    # Processing the call arguments (line 38)
    str_138491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 8), 'str', 'lapack_lite')
    # Processing the call keyword arguments (line 38)
    
    # Obtaining an instance of the builtin type 'list' (line 40)
    list_138492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 40)
    # Adding element type (line 40)
    str_138493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 17), 'str', 'lapack_litemodule.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 16), list_138492, str_138493)
    # Adding element type (line 40)
    # Getting the type of 'get_lapack_lite_sources' (line 40)
    get_lapack_lite_sources_138494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 40), 'get_lapack_lite_sources', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 16), list_138492, get_lapack_lite_sources_138494)
    
    keyword_138495 = list_138492
    
    # Obtaining an instance of the builtin type 'list' (line 41)
    list_138496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 41)
    # Adding element type (line 41)
    str_138497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 17), 'str', 'lapack_lite/f2c.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 16), list_138496, str_138497)
    
    keyword_138498 = list_138496
    # Getting the type of 'lapack_info' (line 42)
    lapack_info_138499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 19), 'lapack_info', False)
    keyword_138500 = lapack_info_138499
    kwargs_138501 = {'sources': keyword_138495, 'depends': keyword_138498, 'extra_info': keyword_138500}
    # Getting the type of 'config' (line 38)
    config_138489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 38)
    add_extension_138490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 4), config_138489, 'add_extension')
    # Calling add_extension(args, kwargs) (line 38)
    add_extension_call_result_138502 = invoke(stypy.reporting.localization.Localization(__file__, 38, 4), add_extension_138490, *[str_138491], **kwargs_138501)
    
    
    # Call to add_extension(...): (line 46)
    # Processing the call arguments (line 46)
    str_138505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 8), 'str', '_umath_linalg')
    # Processing the call keyword arguments (line 46)
    
    # Obtaining an instance of the builtin type 'list' (line 48)
    list_138506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 48)
    # Adding element type (line 48)
    str_138507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 17), 'str', 'umath_linalg.c.src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 16), list_138506, str_138507)
    # Adding element type (line 48)
    # Getting the type of 'get_lapack_lite_sources' (line 48)
    get_lapack_lite_sources_138508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 39), 'get_lapack_lite_sources', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 16), list_138506, get_lapack_lite_sources_138508)
    
    keyword_138509 = list_138506
    
    # Obtaining an instance of the builtin type 'list' (line 49)
    list_138510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 49)
    # Adding element type (line 49)
    str_138511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 17), 'str', 'lapack_lite/f2c.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 16), list_138510, str_138511)
    
    keyword_138512 = list_138510
    # Getting the type of 'lapack_info' (line 50)
    lapack_info_138513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 19), 'lapack_info', False)
    keyword_138514 = lapack_info_138513
    
    # Obtaining an instance of the builtin type 'list' (line 51)
    list_138515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 51)
    # Adding element type (line 51)
    str_138516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 19), 'str', 'npymath')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 18), list_138515, str_138516)
    
    keyword_138517 = list_138515
    kwargs_138518 = {'libraries': keyword_138517, 'sources': keyword_138509, 'depends': keyword_138512, 'extra_info': keyword_138514}
    # Getting the type of 'config' (line 46)
    config_138503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 46)
    add_extension_138504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 4), config_138503, 'add_extension')
    # Calling add_extension(args, kwargs) (line 46)
    add_extension_call_result_138519 = invoke(stypy.reporting.localization.Localization(__file__, 46, 4), add_extension_138504, *[str_138505], **kwargs_138518)
    
    # Getting the type of 'config' (line 53)
    config_138520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type', config_138520)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_138521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_138521)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_138521

# Assigning a type to the variable 'configuration' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 56, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 56)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/linalg/')
    import_138522 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 56, 4), 'numpy.distutils.core')

    if (type(import_138522) is not StypyTypeError):

        if (import_138522 != 'pyd_module'):
            __import__(import_138522)
            sys_modules_138523 = sys.modules[import_138522]
            import_from_module(stypy.reporting.localization.Localization(__file__, 56, 4), 'numpy.distutils.core', sys_modules_138523.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 56, 4), __file__, sys_modules_138523, sys_modules_138523.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 56, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'numpy.distutils.core', import_138522)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/linalg/')
    
    
    # Call to setup(...): (line 57)
    # Processing the call keyword arguments (line 57)
    # Getting the type of 'configuration' (line 57)
    configuration_138525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 24), 'configuration', False)
    keyword_138526 = configuration_138525
    kwargs_138527 = {'configuration': keyword_138526}
    # Getting the type of 'setup' (line 57)
    setup_138524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 57)
    setup_call_result_138528 = invoke(stypy.reporting.localization.Localization(__file__, 57, 4), setup_138524, *[], **kwargs_138527)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
