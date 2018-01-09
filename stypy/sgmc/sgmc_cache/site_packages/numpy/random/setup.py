
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function
2: 
3: from os.path import join, split, dirname
4: import os
5: import sys
6: from distutils.dep_util import newer
7: from distutils.msvccompiler import get_build_version as get_msvc_build_version
8: 
9: def needs_mingw_ftime_workaround():
10:     # We need the mingw workaround for _ftime if the msvc runtime version is
11:     # 7.1 or above and we build with mingw ...
12:     # ... but we can't easily detect compiler version outside distutils command
13:     # context, so we will need to detect in randomkit whether we build with gcc
14:     msver = get_msvc_build_version()
15:     if msver and msver >= 8:
16:         return True
17: 
18:     return False
19: 
20: def configuration(parent_package='',top_path=None):
21:     from numpy.distutils.misc_util import Configuration, get_mathlibs
22:     config = Configuration('random', parent_package, top_path)
23: 
24:     def generate_libraries(ext, build_dir):
25:         config_cmd = config.get_config_cmd()
26:         libs = get_mathlibs()
27:         if sys.platform == 'win32':
28:             libs.append('Advapi32')
29:         ext.libraries.extend(libs)
30:         return None
31: 
32:     # enable unix large file support on 32 bit systems
33:     # (64 bit off_t, lseek -> lseek64 etc.)
34:     defs = [('_FILE_OFFSET_BITS', '64'),
35:             ('_LARGEFILE_SOURCE', '1'),
36:             ('_LARGEFILE64_SOURCE', '1')]
37:     if needs_mingw_ftime_workaround():
38:         defs.append(("NPY_NEEDS_MINGW_TIME_WORKAROUND", None))
39: 
40:     libs = []
41:     # Configure mtrand
42:     config.add_extension('mtrand',
43:                          sources=[join('mtrand', x) for x in
44:                                   ['mtrand.c', 'randomkit.c', 'initarray.c',
45:                                    'distributions.c']]+[generate_libraries],
46:                          libraries=libs,
47:                          depends=[join('mtrand', '*.h'),
48:                                   join('mtrand', '*.pyx'),
49:                                   join('mtrand', '*.pxi'),],
50:                          define_macros=defs,
51:                          )
52: 
53:     config.add_data_files(('.', join('mtrand', 'randomkit.h')))
54:     config.add_data_dir('tests')
55: 
56:     return config
57: 
58: 
59: if __name__ == '__main__':
60:     from numpy.distutils.core import setup
61:     setup(configuration=configuration)
62: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from os.path import join, split, dirname' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/random/')
import_180625 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path')

if (type(import_180625) is not StypyTypeError):

    if (import_180625 != 'pyd_module'):
        __import__(import_180625)
        sys_modules_180626 = sys.modules[import_180625]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', sys_modules_180626.module_type_store, module_type_store, ['join', 'split', 'dirname'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_180626, sys_modules_180626.module_type_store, module_type_store)
    else:
        from os.path import join, split, dirname

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', None, module_type_store, ['join', 'split', 'dirname'], [join, split, dirname])

else:
    # Assigning a type to the variable 'os.path' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', import_180625)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/random/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import os' statement (line 4)
import os

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import sys' statement (line 5)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from distutils.dep_util import newer' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/random/')
import_180627 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.dep_util')

if (type(import_180627) is not StypyTypeError):

    if (import_180627 != 'pyd_module'):
        __import__(import_180627)
        sys_modules_180628 = sys.modules[import_180627]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.dep_util', sys_modules_180628.module_type_store, module_type_store, ['newer'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_180628, sys_modules_180628.module_type_store, module_type_store)
    else:
        from distutils.dep_util import newer

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.dep_util', None, module_type_store, ['newer'], [newer])

else:
    # Assigning a type to the variable 'distutils.dep_util' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.dep_util', import_180627)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/random/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from distutils.msvccompiler import get_msvc_build_version' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/random/')
import_180629 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.msvccompiler')

if (type(import_180629) is not StypyTypeError):

    if (import_180629 != 'pyd_module'):
        __import__(import_180629)
        sys_modules_180630 = sys.modules[import_180629]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.msvccompiler', sys_modules_180630.module_type_store, module_type_store, ['get_build_version'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_180630, sys_modules_180630.module_type_store, module_type_store)
    else:
        from distutils.msvccompiler import get_build_version as get_msvc_build_version

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.msvccompiler', None, module_type_store, ['get_build_version'], [get_msvc_build_version])

else:
    # Assigning a type to the variable 'distutils.msvccompiler' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.msvccompiler', import_180629)

# Adding an alias
module_type_store.add_alias('get_msvc_build_version', 'get_build_version')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/random/')


@norecursion
def needs_mingw_ftime_workaround(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'needs_mingw_ftime_workaround'
    module_type_store = module_type_store.open_function_context('needs_mingw_ftime_workaround', 9, 0, False)
    
    # Passed parameters checking function
    needs_mingw_ftime_workaround.stypy_localization = localization
    needs_mingw_ftime_workaround.stypy_type_of_self = None
    needs_mingw_ftime_workaround.stypy_type_store = module_type_store
    needs_mingw_ftime_workaround.stypy_function_name = 'needs_mingw_ftime_workaround'
    needs_mingw_ftime_workaround.stypy_param_names_list = []
    needs_mingw_ftime_workaround.stypy_varargs_param_name = None
    needs_mingw_ftime_workaround.stypy_kwargs_param_name = None
    needs_mingw_ftime_workaround.stypy_call_defaults = defaults
    needs_mingw_ftime_workaround.stypy_call_varargs = varargs
    needs_mingw_ftime_workaround.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'needs_mingw_ftime_workaround', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'needs_mingw_ftime_workaround', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'needs_mingw_ftime_workaround(...)' code ##################

    
    # Assigning a Call to a Name (line 14):
    
    # Call to get_msvc_build_version(...): (line 14)
    # Processing the call keyword arguments (line 14)
    kwargs_180632 = {}
    # Getting the type of 'get_msvc_build_version' (line 14)
    get_msvc_build_version_180631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 12), 'get_msvc_build_version', False)
    # Calling get_msvc_build_version(args, kwargs) (line 14)
    get_msvc_build_version_call_result_180633 = invoke(stypy.reporting.localization.Localization(__file__, 14, 12), get_msvc_build_version_180631, *[], **kwargs_180632)
    
    # Assigning a type to the variable 'msver' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'msver', get_msvc_build_version_call_result_180633)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'msver' (line 15)
    msver_180634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 7), 'msver')
    
    # Getting the type of 'msver' (line 15)
    msver_180635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 17), 'msver')
    int_180636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 26), 'int')
    # Applying the binary operator '>=' (line 15)
    result_ge_180637 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 17), '>=', msver_180635, int_180636)
    
    # Applying the binary operator 'and' (line 15)
    result_and_keyword_180638 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 7), 'and', msver_180634, result_ge_180637)
    
    # Testing the type of an if condition (line 15)
    if_condition_180639 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 15, 4), result_and_keyword_180638)
    # Assigning a type to the variable 'if_condition_180639' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'if_condition_180639', if_condition_180639)
    # SSA begins for if statement (line 15)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'True' (line 16)
    True_180640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 15), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'stypy_return_type', True_180640)
    # SSA join for if statement (line 15)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'False' (line 18)
    False_180641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type', False_180641)
    
    # ################# End of 'needs_mingw_ftime_workaround(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'needs_mingw_ftime_workaround' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_180642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_180642)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'needs_mingw_ftime_workaround'
    return stypy_return_type_180642

# Assigning a type to the variable 'needs_mingw_ftime_workaround' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'needs_mingw_ftime_workaround', needs_mingw_ftime_workaround)

@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_180643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 33), 'str', '')
    # Getting the type of 'None' (line 20)
    None_180644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 45), 'None')
    defaults = [str_180643, None_180644]
    # Create a new context for function 'configuration'
    module_type_store = module_type_store.open_function_context('configuration', 20, 0, False)
    
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

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 4))
    
    # 'from numpy.distutils.misc_util import Configuration, get_mathlibs' statement (line 21)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/random/')
    import_180645 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 4), 'numpy.distutils.misc_util')

    if (type(import_180645) is not StypyTypeError):

        if (import_180645 != 'pyd_module'):
            __import__(import_180645)
            sys_modules_180646 = sys.modules[import_180645]
            import_from_module(stypy.reporting.localization.Localization(__file__, 21, 4), 'numpy.distutils.misc_util', sys_modules_180646.module_type_store, module_type_store, ['Configuration', 'get_mathlibs'])
            nest_module(stypy.reporting.localization.Localization(__file__, 21, 4), __file__, sys_modules_180646, sys_modules_180646.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration, get_mathlibs

            import_from_module(stypy.reporting.localization.Localization(__file__, 21, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration', 'get_mathlibs'], [Configuration, get_mathlibs])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'numpy.distutils.misc_util', import_180645)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/random/')
    
    
    # Assigning a Call to a Name (line 22):
    
    # Call to Configuration(...): (line 22)
    # Processing the call arguments (line 22)
    str_180648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 27), 'str', 'random')
    # Getting the type of 'parent_package' (line 22)
    parent_package_180649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 37), 'parent_package', False)
    # Getting the type of 'top_path' (line 22)
    top_path_180650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 53), 'top_path', False)
    # Processing the call keyword arguments (line 22)
    kwargs_180651 = {}
    # Getting the type of 'Configuration' (line 22)
    Configuration_180647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 22)
    Configuration_call_result_180652 = invoke(stypy.reporting.localization.Localization(__file__, 22, 13), Configuration_180647, *[str_180648, parent_package_180649, top_path_180650], **kwargs_180651)
    
    # Assigning a type to the variable 'config' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'config', Configuration_call_result_180652)

    @norecursion
    def generate_libraries(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'generate_libraries'
        module_type_store = module_type_store.open_function_context('generate_libraries', 24, 4, False)
        
        # Passed parameters checking function
        generate_libraries.stypy_localization = localization
        generate_libraries.stypy_type_of_self = None
        generate_libraries.stypy_type_store = module_type_store
        generate_libraries.stypy_function_name = 'generate_libraries'
        generate_libraries.stypy_param_names_list = ['ext', 'build_dir']
        generate_libraries.stypy_varargs_param_name = None
        generate_libraries.stypy_kwargs_param_name = None
        generate_libraries.stypy_call_defaults = defaults
        generate_libraries.stypy_call_varargs = varargs
        generate_libraries.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'generate_libraries', ['ext', 'build_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'generate_libraries', localization, ['ext', 'build_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'generate_libraries(...)' code ##################

        
        # Assigning a Call to a Name (line 25):
        
        # Call to get_config_cmd(...): (line 25)
        # Processing the call keyword arguments (line 25)
        kwargs_180655 = {}
        # Getting the type of 'config' (line 25)
        config_180653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 21), 'config', False)
        # Obtaining the member 'get_config_cmd' of a type (line 25)
        get_config_cmd_180654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 21), config_180653, 'get_config_cmd')
        # Calling get_config_cmd(args, kwargs) (line 25)
        get_config_cmd_call_result_180656 = invoke(stypy.reporting.localization.Localization(__file__, 25, 21), get_config_cmd_180654, *[], **kwargs_180655)
        
        # Assigning a type to the variable 'config_cmd' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'config_cmd', get_config_cmd_call_result_180656)
        
        # Assigning a Call to a Name (line 26):
        
        # Call to get_mathlibs(...): (line 26)
        # Processing the call keyword arguments (line 26)
        kwargs_180658 = {}
        # Getting the type of 'get_mathlibs' (line 26)
        get_mathlibs_180657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'get_mathlibs', False)
        # Calling get_mathlibs(args, kwargs) (line 26)
        get_mathlibs_call_result_180659 = invoke(stypy.reporting.localization.Localization(__file__, 26, 15), get_mathlibs_180657, *[], **kwargs_180658)
        
        # Assigning a type to the variable 'libs' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'libs', get_mathlibs_call_result_180659)
        
        
        # Getting the type of 'sys' (line 27)
        sys_180660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 11), 'sys')
        # Obtaining the member 'platform' of a type (line 27)
        platform_180661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 11), sys_180660, 'platform')
        str_180662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 27), 'str', 'win32')
        # Applying the binary operator '==' (line 27)
        result_eq_180663 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 11), '==', platform_180661, str_180662)
        
        # Testing the type of an if condition (line 27)
        if_condition_180664 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 8), result_eq_180663)
        # Assigning a type to the variable 'if_condition_180664' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'if_condition_180664', if_condition_180664)
        # SSA begins for if statement (line 27)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 28)
        # Processing the call arguments (line 28)
        str_180667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 24), 'str', 'Advapi32')
        # Processing the call keyword arguments (line 28)
        kwargs_180668 = {}
        # Getting the type of 'libs' (line 28)
        libs_180665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'libs', False)
        # Obtaining the member 'append' of a type (line 28)
        append_180666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 12), libs_180665, 'append')
        # Calling append(args, kwargs) (line 28)
        append_call_result_180669 = invoke(stypy.reporting.localization.Localization(__file__, 28, 12), append_180666, *[str_180667], **kwargs_180668)
        
        # SSA join for if statement (line 27)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to extend(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'libs' (line 29)
        libs_180673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 29), 'libs', False)
        # Processing the call keyword arguments (line 29)
        kwargs_180674 = {}
        # Getting the type of 'ext' (line 29)
        ext_180670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'ext', False)
        # Obtaining the member 'libraries' of a type (line 29)
        libraries_180671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), ext_180670, 'libraries')
        # Obtaining the member 'extend' of a type (line 29)
        extend_180672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), libraries_180671, 'extend')
        # Calling extend(args, kwargs) (line 29)
        extend_call_result_180675 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), extend_180672, *[libs_180673], **kwargs_180674)
        
        # Getting the type of 'None' (line 30)
        None_180676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'stypy_return_type', None_180676)
        
        # ################# End of 'generate_libraries(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'generate_libraries' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_180677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_180677)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'generate_libraries'
        return stypy_return_type_180677

    # Assigning a type to the variable 'generate_libraries' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'generate_libraries', generate_libraries)
    
    # Assigning a List to a Name (line 34):
    
    # Obtaining an instance of the builtin type 'list' (line 34)
    list_180678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 34)
    # Adding element type (line 34)
    
    # Obtaining an instance of the builtin type 'tuple' (line 34)
    tuple_180679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 34)
    # Adding element type (line 34)
    str_180680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 13), 'str', '_FILE_OFFSET_BITS')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 13), tuple_180679, str_180680)
    # Adding element type (line 34)
    str_180681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 34), 'str', '64')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 13), tuple_180679, str_180681)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), list_180678, tuple_180679)
    # Adding element type (line 34)
    
    # Obtaining an instance of the builtin type 'tuple' (line 35)
    tuple_180682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 35)
    # Adding element type (line 35)
    str_180683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 13), 'str', '_LARGEFILE_SOURCE')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 13), tuple_180682, str_180683)
    # Adding element type (line 35)
    str_180684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 34), 'str', '1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 13), tuple_180682, str_180684)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), list_180678, tuple_180682)
    # Adding element type (line 34)
    
    # Obtaining an instance of the builtin type 'tuple' (line 36)
    tuple_180685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 36)
    # Adding element type (line 36)
    str_180686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 13), 'str', '_LARGEFILE64_SOURCE')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 13), tuple_180685, str_180686)
    # Adding element type (line 36)
    str_180687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 36), 'str', '1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 13), tuple_180685, str_180687)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), list_180678, tuple_180685)
    
    # Assigning a type to the variable 'defs' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'defs', list_180678)
    
    
    # Call to needs_mingw_ftime_workaround(...): (line 37)
    # Processing the call keyword arguments (line 37)
    kwargs_180689 = {}
    # Getting the type of 'needs_mingw_ftime_workaround' (line 37)
    needs_mingw_ftime_workaround_180688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 7), 'needs_mingw_ftime_workaround', False)
    # Calling needs_mingw_ftime_workaround(args, kwargs) (line 37)
    needs_mingw_ftime_workaround_call_result_180690 = invoke(stypy.reporting.localization.Localization(__file__, 37, 7), needs_mingw_ftime_workaround_180688, *[], **kwargs_180689)
    
    # Testing the type of an if condition (line 37)
    if_condition_180691 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 4), needs_mingw_ftime_workaround_call_result_180690)
    # Assigning a type to the variable 'if_condition_180691' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'if_condition_180691', if_condition_180691)
    # SSA begins for if statement (line 37)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 38)
    # Processing the call arguments (line 38)
    
    # Obtaining an instance of the builtin type 'tuple' (line 38)
    tuple_180694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 38)
    # Adding element type (line 38)
    str_180695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 21), 'str', 'NPY_NEEDS_MINGW_TIME_WORKAROUND')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 21), tuple_180694, str_180695)
    # Adding element type (line 38)
    # Getting the type of 'None' (line 38)
    None_180696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 56), 'None', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 21), tuple_180694, None_180696)
    
    # Processing the call keyword arguments (line 38)
    kwargs_180697 = {}
    # Getting the type of 'defs' (line 38)
    defs_180692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'defs', False)
    # Obtaining the member 'append' of a type (line 38)
    append_180693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), defs_180692, 'append')
    # Calling append(args, kwargs) (line 38)
    append_call_result_180698 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), append_180693, *[tuple_180694], **kwargs_180697)
    
    # SSA join for if statement (line 37)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 40):
    
    # Obtaining an instance of the builtin type 'list' (line 40)
    list_180699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 40)
    
    # Assigning a type to the variable 'libs' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'libs', list_180699)
    
    # Call to add_extension(...): (line 42)
    # Processing the call arguments (line 42)
    str_180702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 25), 'str', 'mtrand')
    # Processing the call keyword arguments (line 42)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 44)
    list_180708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 44)
    # Adding element type (line 44)
    str_180709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 35), 'str', 'mtrand.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 34), list_180708, str_180709)
    # Adding element type (line 44)
    str_180710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 47), 'str', 'randomkit.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 34), list_180708, str_180710)
    # Adding element type (line 44)
    str_180711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 62), 'str', 'initarray.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 34), list_180708, str_180711)
    # Adding element type (line 44)
    str_180712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 35), 'str', 'distributions.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 34), list_180708, str_180712)
    
    comprehension_180713 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 34), list_180708)
    # Assigning a type to the variable 'x' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 34), 'x', comprehension_180713)
    
    # Call to join(...): (line 43)
    # Processing the call arguments (line 43)
    str_180704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 39), 'str', 'mtrand')
    # Getting the type of 'x' (line 43)
    x_180705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 49), 'x', False)
    # Processing the call keyword arguments (line 43)
    kwargs_180706 = {}
    # Getting the type of 'join' (line 43)
    join_180703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 34), 'join', False)
    # Calling join(args, kwargs) (line 43)
    join_call_result_180707 = invoke(stypy.reporting.localization.Localization(__file__, 43, 34), join_180703, *[str_180704, x_180705], **kwargs_180706)
    
    list_180714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 34), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 34), list_180714, join_call_result_180707)
    
    # Obtaining an instance of the builtin type 'list' (line 45)
    list_180715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 55), 'list')
    # Adding type elements to the builtin type 'list' instance (line 45)
    # Adding element type (line 45)
    # Getting the type of 'generate_libraries' (line 45)
    generate_libraries_180716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 56), 'generate_libraries', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 55), list_180715, generate_libraries_180716)
    
    # Applying the binary operator '+' (line 43)
    result_add_180717 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 33), '+', list_180714, list_180715)
    
    keyword_180718 = result_add_180717
    # Getting the type of 'libs' (line 46)
    libs_180719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 35), 'libs', False)
    keyword_180720 = libs_180719
    
    # Obtaining an instance of the builtin type 'list' (line 47)
    list_180721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 47)
    # Adding element type (line 47)
    
    # Call to join(...): (line 47)
    # Processing the call arguments (line 47)
    str_180723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 39), 'str', 'mtrand')
    str_180724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 49), 'str', '*.h')
    # Processing the call keyword arguments (line 47)
    kwargs_180725 = {}
    # Getting the type of 'join' (line 47)
    join_180722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 34), 'join', False)
    # Calling join(args, kwargs) (line 47)
    join_call_result_180726 = invoke(stypy.reporting.localization.Localization(__file__, 47, 34), join_180722, *[str_180723, str_180724], **kwargs_180725)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 33), list_180721, join_call_result_180726)
    # Adding element type (line 47)
    
    # Call to join(...): (line 48)
    # Processing the call arguments (line 48)
    str_180728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 39), 'str', 'mtrand')
    str_180729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 49), 'str', '*.pyx')
    # Processing the call keyword arguments (line 48)
    kwargs_180730 = {}
    # Getting the type of 'join' (line 48)
    join_180727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 34), 'join', False)
    # Calling join(args, kwargs) (line 48)
    join_call_result_180731 = invoke(stypy.reporting.localization.Localization(__file__, 48, 34), join_180727, *[str_180728, str_180729], **kwargs_180730)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 33), list_180721, join_call_result_180731)
    # Adding element type (line 47)
    
    # Call to join(...): (line 49)
    # Processing the call arguments (line 49)
    str_180733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 39), 'str', 'mtrand')
    str_180734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 49), 'str', '*.pxi')
    # Processing the call keyword arguments (line 49)
    kwargs_180735 = {}
    # Getting the type of 'join' (line 49)
    join_180732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 34), 'join', False)
    # Calling join(args, kwargs) (line 49)
    join_call_result_180736 = invoke(stypy.reporting.localization.Localization(__file__, 49, 34), join_180732, *[str_180733, str_180734], **kwargs_180735)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 33), list_180721, join_call_result_180736)
    
    keyword_180737 = list_180721
    # Getting the type of 'defs' (line 50)
    defs_180738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 39), 'defs', False)
    keyword_180739 = defs_180738
    kwargs_180740 = {'libraries': keyword_180720, 'sources': keyword_180718, 'depends': keyword_180737, 'define_macros': keyword_180739}
    # Getting the type of 'config' (line 42)
    config_180700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 42)
    add_extension_180701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 4), config_180700, 'add_extension')
    # Calling add_extension(args, kwargs) (line 42)
    add_extension_call_result_180741 = invoke(stypy.reporting.localization.Localization(__file__, 42, 4), add_extension_180701, *[str_180702], **kwargs_180740)
    
    
    # Call to add_data_files(...): (line 53)
    # Processing the call arguments (line 53)
    
    # Obtaining an instance of the builtin type 'tuple' (line 53)
    tuple_180744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 53)
    # Adding element type (line 53)
    str_180745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 27), 'str', '.')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 27), tuple_180744, str_180745)
    # Adding element type (line 53)
    
    # Call to join(...): (line 53)
    # Processing the call arguments (line 53)
    str_180747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 37), 'str', 'mtrand')
    str_180748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 47), 'str', 'randomkit.h')
    # Processing the call keyword arguments (line 53)
    kwargs_180749 = {}
    # Getting the type of 'join' (line 53)
    join_180746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 32), 'join', False)
    # Calling join(args, kwargs) (line 53)
    join_call_result_180750 = invoke(stypy.reporting.localization.Localization(__file__, 53, 32), join_180746, *[str_180747, str_180748], **kwargs_180749)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 27), tuple_180744, join_call_result_180750)
    
    # Processing the call keyword arguments (line 53)
    kwargs_180751 = {}
    # Getting the type of 'config' (line 53)
    config_180742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'config', False)
    # Obtaining the member 'add_data_files' of a type (line 53)
    add_data_files_180743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 4), config_180742, 'add_data_files')
    # Calling add_data_files(args, kwargs) (line 53)
    add_data_files_call_result_180752 = invoke(stypy.reporting.localization.Localization(__file__, 53, 4), add_data_files_180743, *[tuple_180744], **kwargs_180751)
    
    
    # Call to add_data_dir(...): (line 54)
    # Processing the call arguments (line 54)
    str_180755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 54)
    kwargs_180756 = {}
    # Getting the type of 'config' (line 54)
    config_180753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 54)
    add_data_dir_180754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 4), config_180753, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 54)
    add_data_dir_call_result_180757 = invoke(stypy.reporting.localization.Localization(__file__, 54, 4), add_data_dir_180754, *[str_180755], **kwargs_180756)
    
    # Getting the type of 'config' (line 56)
    config_180758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type', config_180758)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_180759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_180759)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_180759

# Assigning a type to the variable 'configuration' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 60, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 60)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/random/')
    import_180760 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 60, 4), 'numpy.distutils.core')

    if (type(import_180760) is not StypyTypeError):

        if (import_180760 != 'pyd_module'):
            __import__(import_180760)
            sys_modules_180761 = sys.modules[import_180760]
            import_from_module(stypy.reporting.localization.Localization(__file__, 60, 4), 'numpy.distutils.core', sys_modules_180761.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 60, 4), __file__, sys_modules_180761, sys_modules_180761.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 60, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'numpy.distutils.core', import_180760)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/random/')
    
    
    # Call to setup(...): (line 61)
    # Processing the call keyword arguments (line 61)
    # Getting the type of 'configuration' (line 61)
    configuration_180763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'configuration', False)
    keyword_180764 = configuration_180763
    kwargs_180765 = {'configuration': keyword_180764}
    # Getting the type of 'setup' (line 61)
    setup_180762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 61)
    setup_call_result_180766 = invoke(stypy.reporting.localization.Localization(__file__, 61, 4), setup_180762, *[], **kwargs_180765)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
