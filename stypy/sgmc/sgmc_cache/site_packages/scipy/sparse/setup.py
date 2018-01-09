
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import os
4: import sys
5: import subprocess
6: 
7: 
8: def configuration(parent_package='',top_path=None):
9:     from numpy.distutils.misc_util import Configuration
10: 
11:     config = Configuration('sparse',parent_package,top_path)
12: 
13:     config.add_data_dir('tests')
14: 
15:     config.add_subpackage('linalg')
16:     config.add_subpackage('csgraph')
17: 
18:     config.add_extension('_csparsetools',
19:                          sources=['_csparsetools.c'])
20: 
21:     def get_sparsetools_sources(ext, build_dir):
22:         # Defer generation of source files
23:         subprocess.check_call([sys.executable,
24:                                os.path.join(os.path.dirname(__file__),
25:                                             'generate_sparsetools.py'),
26:                                '--no-force'])
27:         return []
28: 
29:     depends = ['sparsetools_impl.h',
30:                'bsr_impl.h',
31:                'csc_impl.h',
32:                'csr_impl.h',
33:                'other_impl.h',
34:                'bool_ops.h',
35:                'bsr.h',
36:                'complex_ops.h',
37:                'coo.h',
38:                'csc.h',
39:                'csgraph.h',
40:                'csr.h',
41:                'dense.h',
42:                'dia.h',
43:                'py3k.h',
44:                'sparsetools.h',
45:                'util.h']
46:     depends = [os.path.join('sparsetools', hdr) for hdr in depends],
47:     config.add_extension('_sparsetools',
48:                          define_macros=[('__STDC_FORMAT_MACROS', 1)],
49:                          depends=depends,
50:                          include_dirs=['sparsetools'],
51:                          sources=[os.path.join('sparsetools', 'sparsetools.cxx'),
52:                                   os.path.join('sparsetools', 'csr.cxx'),
53:                                   os.path.join('sparsetools', 'csc.cxx'),
54:                                   os.path.join('sparsetools', 'bsr.cxx'),
55:                                   os.path.join('sparsetools', 'other.cxx'),
56:                                   get_sparsetools_sources]
57:                          )
58: 
59:     return config
60: 
61: if __name__ == '__main__':
62:     from numpy.distutils.core import setup
63:     setup(**configuration(top_path='').todict())
64: 

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

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import subprocess' statement (line 5)
import subprocess

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'subprocess', subprocess, module_type_store)


@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_379223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 33), 'str', '')
    # Getting the type of 'None' (line 8)
    None_379224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 45), 'None')
    defaults = [str_379223, None_379224]
    # Create a new context for function 'configuration'
    module_type_store = module_type_store.open_function_context('configuration', 8, 0, False)
    
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

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 4))
    
    # 'from numpy.distutils.misc_util import Configuration' statement (line 9)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
    import_379225 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.misc_util')

    if (type(import_379225) is not StypyTypeError):

        if (import_379225 != 'pyd_module'):
            __import__(import_379225)
            sys_modules_379226 = sys.modules[import_379225]
            import_from_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.misc_util', sys_modules_379226.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 9, 4), __file__, sys_modules_379226, sys_modules_379226.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.misc_util', import_379225)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
    
    
    # Assigning a Call to a Name (line 11):
    
    # Call to Configuration(...): (line 11)
    # Processing the call arguments (line 11)
    str_379228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 27), 'str', 'sparse')
    # Getting the type of 'parent_package' (line 11)
    parent_package_379229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 36), 'parent_package', False)
    # Getting the type of 'top_path' (line 11)
    top_path_379230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 51), 'top_path', False)
    # Processing the call keyword arguments (line 11)
    kwargs_379231 = {}
    # Getting the type of 'Configuration' (line 11)
    Configuration_379227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 11)
    Configuration_call_result_379232 = invoke(stypy.reporting.localization.Localization(__file__, 11, 13), Configuration_379227, *[str_379228, parent_package_379229, top_path_379230], **kwargs_379231)
    
    # Assigning a type to the variable 'config' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'config', Configuration_call_result_379232)
    
    # Call to add_data_dir(...): (line 13)
    # Processing the call arguments (line 13)
    str_379235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 13)
    kwargs_379236 = {}
    # Getting the type of 'config' (line 13)
    config_379233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 13)
    add_data_dir_379234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 4), config_379233, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 13)
    add_data_dir_call_result_379237 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), add_data_dir_379234, *[str_379235], **kwargs_379236)
    
    
    # Call to add_subpackage(...): (line 15)
    # Processing the call arguments (line 15)
    str_379240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 26), 'str', 'linalg')
    # Processing the call keyword arguments (line 15)
    kwargs_379241 = {}
    # Getting the type of 'config' (line 15)
    config_379238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 15)
    add_subpackage_379239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 4), config_379238, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 15)
    add_subpackage_call_result_379242 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), add_subpackage_379239, *[str_379240], **kwargs_379241)
    
    
    # Call to add_subpackage(...): (line 16)
    # Processing the call arguments (line 16)
    str_379245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 26), 'str', 'csgraph')
    # Processing the call keyword arguments (line 16)
    kwargs_379246 = {}
    # Getting the type of 'config' (line 16)
    config_379243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 16)
    add_subpackage_379244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), config_379243, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 16)
    add_subpackage_call_result_379247 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), add_subpackage_379244, *[str_379245], **kwargs_379246)
    
    
    # Call to add_extension(...): (line 18)
    # Processing the call arguments (line 18)
    str_379250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'str', '_csparsetools')
    # Processing the call keyword arguments (line 18)
    
    # Obtaining an instance of the builtin type 'list' (line 19)
    list_379251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 19)
    # Adding element type (line 19)
    str_379252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 34), 'str', '_csparsetools.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 33), list_379251, str_379252)
    
    keyword_379253 = list_379251
    kwargs_379254 = {'sources': keyword_379253}
    # Getting the type of 'config' (line 18)
    config_379248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 18)
    add_extension_379249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 4), config_379248, 'add_extension')
    # Calling add_extension(args, kwargs) (line 18)
    add_extension_call_result_379255 = invoke(stypy.reporting.localization.Localization(__file__, 18, 4), add_extension_379249, *[str_379250], **kwargs_379254)
    

    @norecursion
    def get_sparsetools_sources(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_sparsetools_sources'
        module_type_store = module_type_store.open_function_context('get_sparsetools_sources', 21, 4, False)
        
        # Passed parameters checking function
        get_sparsetools_sources.stypy_localization = localization
        get_sparsetools_sources.stypy_type_of_self = None
        get_sparsetools_sources.stypy_type_store = module_type_store
        get_sparsetools_sources.stypy_function_name = 'get_sparsetools_sources'
        get_sparsetools_sources.stypy_param_names_list = ['ext', 'build_dir']
        get_sparsetools_sources.stypy_varargs_param_name = None
        get_sparsetools_sources.stypy_kwargs_param_name = None
        get_sparsetools_sources.stypy_call_defaults = defaults
        get_sparsetools_sources.stypy_call_varargs = varargs
        get_sparsetools_sources.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'get_sparsetools_sources', ['ext', 'build_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_sparsetools_sources', localization, ['ext', 'build_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_sparsetools_sources(...)' code ##################

        
        # Call to check_call(...): (line 23)
        # Processing the call arguments (line 23)
        
        # Obtaining an instance of the builtin type 'list' (line 23)
        list_379258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 23)
        # Adding element type (line 23)
        # Getting the type of 'sys' (line 23)
        sys_379259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 31), 'sys', False)
        # Obtaining the member 'executable' of a type (line 23)
        executable_379260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 31), sys_379259, 'executable')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 30), list_379258, executable_379260)
        # Adding element type (line 23)
        
        # Call to join(...): (line 24)
        # Processing the call arguments (line 24)
        
        # Call to dirname(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of '__file__' (line 24)
        file___379267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 60), '__file__', False)
        # Processing the call keyword arguments (line 24)
        kwargs_379268 = {}
        # Getting the type of 'os' (line 24)
        os_379264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 44), 'os', False)
        # Obtaining the member 'path' of a type (line 24)
        path_379265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 44), os_379264, 'path')
        # Obtaining the member 'dirname' of a type (line 24)
        dirname_379266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 44), path_379265, 'dirname')
        # Calling dirname(args, kwargs) (line 24)
        dirname_call_result_379269 = invoke(stypy.reporting.localization.Localization(__file__, 24, 44), dirname_379266, *[file___379267], **kwargs_379268)
        
        str_379270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 44), 'str', 'generate_sparsetools.py')
        # Processing the call keyword arguments (line 24)
        kwargs_379271 = {}
        # Getting the type of 'os' (line 24)
        os_379261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 31), 'os', False)
        # Obtaining the member 'path' of a type (line 24)
        path_379262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 31), os_379261, 'path')
        # Obtaining the member 'join' of a type (line 24)
        join_379263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 31), path_379262, 'join')
        # Calling join(args, kwargs) (line 24)
        join_call_result_379272 = invoke(stypy.reporting.localization.Localization(__file__, 24, 31), join_379263, *[dirname_call_result_379269, str_379270], **kwargs_379271)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 30), list_379258, join_call_result_379272)
        # Adding element type (line 23)
        str_379273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 31), 'str', '--no-force')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 30), list_379258, str_379273)
        
        # Processing the call keyword arguments (line 23)
        kwargs_379274 = {}
        # Getting the type of 'subprocess' (line 23)
        subprocess_379256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'subprocess', False)
        # Obtaining the member 'check_call' of a type (line 23)
        check_call_379257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), subprocess_379256, 'check_call')
        # Calling check_call(args, kwargs) (line 23)
        check_call_call_result_379275 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), check_call_379257, *[list_379258], **kwargs_379274)
        
        
        # Obtaining an instance of the builtin type 'list' (line 27)
        list_379276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 27)
        
        # Assigning a type to the variable 'stypy_return_type' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'stypy_return_type', list_379276)
        
        # ################# End of 'get_sparsetools_sources(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_sparsetools_sources' in the type store
        # Getting the type of 'stypy_return_type' (line 21)
        stypy_return_type_379277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_379277)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_sparsetools_sources'
        return stypy_return_type_379277

    # Assigning a type to the variable 'get_sparsetools_sources' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'get_sparsetools_sources', get_sparsetools_sources)
    
    # Assigning a List to a Name (line 29):
    
    # Obtaining an instance of the builtin type 'list' (line 29)
    list_379278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 29)
    # Adding element type (line 29)
    str_379279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 15), 'str', 'sparsetools_impl.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 14), list_379278, str_379279)
    # Adding element type (line 29)
    str_379280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 15), 'str', 'bsr_impl.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 14), list_379278, str_379280)
    # Adding element type (line 29)
    str_379281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 15), 'str', 'csc_impl.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 14), list_379278, str_379281)
    # Adding element type (line 29)
    str_379282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 15), 'str', 'csr_impl.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 14), list_379278, str_379282)
    # Adding element type (line 29)
    str_379283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 15), 'str', 'other_impl.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 14), list_379278, str_379283)
    # Adding element type (line 29)
    str_379284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 15), 'str', 'bool_ops.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 14), list_379278, str_379284)
    # Adding element type (line 29)
    str_379285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 15), 'str', 'bsr.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 14), list_379278, str_379285)
    # Adding element type (line 29)
    str_379286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 15), 'str', 'complex_ops.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 14), list_379278, str_379286)
    # Adding element type (line 29)
    str_379287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 15), 'str', 'coo.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 14), list_379278, str_379287)
    # Adding element type (line 29)
    str_379288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 15), 'str', 'csc.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 14), list_379278, str_379288)
    # Adding element type (line 29)
    str_379289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 15), 'str', 'csgraph.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 14), list_379278, str_379289)
    # Adding element type (line 29)
    str_379290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 15), 'str', 'csr.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 14), list_379278, str_379290)
    # Adding element type (line 29)
    str_379291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 15), 'str', 'dense.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 14), list_379278, str_379291)
    # Adding element type (line 29)
    str_379292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 15), 'str', 'dia.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 14), list_379278, str_379292)
    # Adding element type (line 29)
    str_379293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 15), 'str', 'py3k.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 14), list_379278, str_379293)
    # Adding element type (line 29)
    str_379294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 15), 'str', 'sparsetools.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 14), list_379278, str_379294)
    # Adding element type (line 29)
    str_379295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 15), 'str', 'util.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 14), list_379278, str_379295)
    
    # Assigning a type to the variable 'depends' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'depends', list_379278)
    
    # Assigning a Tuple to a Name (line 46):
    
    # Obtaining an instance of the builtin type 'tuple' (line 46)
    tuple_379296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 46)
    # Adding element type (line 46)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'depends' (line 46)
    depends_379304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 59), 'depends')
    comprehension_379305 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 15), depends_379304)
    # Assigning a type to the variable 'hdr' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 15), 'hdr', comprehension_379305)
    
    # Call to join(...): (line 46)
    # Processing the call arguments (line 46)
    str_379300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 28), 'str', 'sparsetools')
    # Getting the type of 'hdr' (line 46)
    hdr_379301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 43), 'hdr', False)
    # Processing the call keyword arguments (line 46)
    kwargs_379302 = {}
    # Getting the type of 'os' (line 46)
    os_379297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 46)
    path_379298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 15), os_379297, 'path')
    # Obtaining the member 'join' of a type (line 46)
    join_379299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 15), path_379298, 'join')
    # Calling join(args, kwargs) (line 46)
    join_call_result_379303 = invoke(stypy.reporting.localization.Localization(__file__, 46, 15), join_379299, *[str_379300, hdr_379301], **kwargs_379302)
    
    list_379306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 15), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 15), list_379306, join_call_result_379303)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 14), tuple_379296, list_379306)
    
    # Assigning a type to the variable 'depends' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'depends', tuple_379296)
    
    # Call to add_extension(...): (line 47)
    # Processing the call arguments (line 47)
    str_379309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 25), 'str', '_sparsetools')
    # Processing the call keyword arguments (line 47)
    
    # Obtaining an instance of the builtin type 'list' (line 48)
    list_379310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 48)
    # Adding element type (line 48)
    
    # Obtaining an instance of the builtin type 'tuple' (line 48)
    tuple_379311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 48)
    # Adding element type (line 48)
    str_379312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 41), 'str', '__STDC_FORMAT_MACROS')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 41), tuple_379311, str_379312)
    # Adding element type (line 48)
    int_379313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 65), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 41), tuple_379311, int_379313)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 39), list_379310, tuple_379311)
    
    keyword_379314 = list_379310
    # Getting the type of 'depends' (line 49)
    depends_379315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 33), 'depends', False)
    keyword_379316 = depends_379315
    
    # Obtaining an instance of the builtin type 'list' (line 50)
    list_379317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 50)
    # Adding element type (line 50)
    str_379318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 39), 'str', 'sparsetools')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 38), list_379317, str_379318)
    
    keyword_379319 = list_379317
    
    # Obtaining an instance of the builtin type 'list' (line 51)
    list_379320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 51)
    # Adding element type (line 51)
    
    # Call to join(...): (line 51)
    # Processing the call arguments (line 51)
    str_379324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 47), 'str', 'sparsetools')
    str_379325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 62), 'str', 'sparsetools.cxx')
    # Processing the call keyword arguments (line 51)
    kwargs_379326 = {}
    # Getting the type of 'os' (line 51)
    os_379321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 34), 'os', False)
    # Obtaining the member 'path' of a type (line 51)
    path_379322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 34), os_379321, 'path')
    # Obtaining the member 'join' of a type (line 51)
    join_379323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 34), path_379322, 'join')
    # Calling join(args, kwargs) (line 51)
    join_call_result_379327 = invoke(stypy.reporting.localization.Localization(__file__, 51, 34), join_379323, *[str_379324, str_379325], **kwargs_379326)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 33), list_379320, join_call_result_379327)
    # Adding element type (line 51)
    
    # Call to join(...): (line 52)
    # Processing the call arguments (line 52)
    str_379331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 47), 'str', 'sparsetools')
    str_379332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 62), 'str', 'csr.cxx')
    # Processing the call keyword arguments (line 52)
    kwargs_379333 = {}
    # Getting the type of 'os' (line 52)
    os_379328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 34), 'os', False)
    # Obtaining the member 'path' of a type (line 52)
    path_379329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 34), os_379328, 'path')
    # Obtaining the member 'join' of a type (line 52)
    join_379330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 34), path_379329, 'join')
    # Calling join(args, kwargs) (line 52)
    join_call_result_379334 = invoke(stypy.reporting.localization.Localization(__file__, 52, 34), join_379330, *[str_379331, str_379332], **kwargs_379333)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 33), list_379320, join_call_result_379334)
    # Adding element type (line 51)
    
    # Call to join(...): (line 53)
    # Processing the call arguments (line 53)
    str_379338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 47), 'str', 'sparsetools')
    str_379339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 62), 'str', 'csc.cxx')
    # Processing the call keyword arguments (line 53)
    kwargs_379340 = {}
    # Getting the type of 'os' (line 53)
    os_379335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 34), 'os', False)
    # Obtaining the member 'path' of a type (line 53)
    path_379336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 34), os_379335, 'path')
    # Obtaining the member 'join' of a type (line 53)
    join_379337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 34), path_379336, 'join')
    # Calling join(args, kwargs) (line 53)
    join_call_result_379341 = invoke(stypy.reporting.localization.Localization(__file__, 53, 34), join_379337, *[str_379338, str_379339], **kwargs_379340)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 33), list_379320, join_call_result_379341)
    # Adding element type (line 51)
    
    # Call to join(...): (line 54)
    # Processing the call arguments (line 54)
    str_379345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 47), 'str', 'sparsetools')
    str_379346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 62), 'str', 'bsr.cxx')
    # Processing the call keyword arguments (line 54)
    kwargs_379347 = {}
    # Getting the type of 'os' (line 54)
    os_379342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 34), 'os', False)
    # Obtaining the member 'path' of a type (line 54)
    path_379343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 34), os_379342, 'path')
    # Obtaining the member 'join' of a type (line 54)
    join_379344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 34), path_379343, 'join')
    # Calling join(args, kwargs) (line 54)
    join_call_result_379348 = invoke(stypy.reporting.localization.Localization(__file__, 54, 34), join_379344, *[str_379345, str_379346], **kwargs_379347)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 33), list_379320, join_call_result_379348)
    # Adding element type (line 51)
    
    # Call to join(...): (line 55)
    # Processing the call arguments (line 55)
    str_379352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 47), 'str', 'sparsetools')
    str_379353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 62), 'str', 'other.cxx')
    # Processing the call keyword arguments (line 55)
    kwargs_379354 = {}
    # Getting the type of 'os' (line 55)
    os_379349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 34), 'os', False)
    # Obtaining the member 'path' of a type (line 55)
    path_379350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 34), os_379349, 'path')
    # Obtaining the member 'join' of a type (line 55)
    join_379351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 34), path_379350, 'join')
    # Calling join(args, kwargs) (line 55)
    join_call_result_379355 = invoke(stypy.reporting.localization.Localization(__file__, 55, 34), join_379351, *[str_379352, str_379353], **kwargs_379354)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 33), list_379320, join_call_result_379355)
    # Adding element type (line 51)
    # Getting the type of 'get_sparsetools_sources' (line 56)
    get_sparsetools_sources_379356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 34), 'get_sparsetools_sources', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 33), list_379320, get_sparsetools_sources_379356)
    
    keyword_379357 = list_379320
    kwargs_379358 = {'sources': keyword_379357, 'depends': keyword_379316, 'define_macros': keyword_379314, 'include_dirs': keyword_379319}
    # Getting the type of 'config' (line 47)
    config_379307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 47)
    add_extension_379308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 4), config_379307, 'add_extension')
    # Calling add_extension(args, kwargs) (line 47)
    add_extension_call_result_379359 = invoke(stypy.reporting.localization.Localization(__file__, 47, 4), add_extension_379308, *[str_379309], **kwargs_379358)
    
    # Getting the type of 'config' (line 59)
    config_379360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type', config_379360)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_379361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_379361)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_379361

# Assigning a type to the variable 'configuration' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 62, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 62)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
    import_379362 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 62, 4), 'numpy.distutils.core')

    if (type(import_379362) is not StypyTypeError):

        if (import_379362 != 'pyd_module'):
            __import__(import_379362)
            sys_modules_379363 = sys.modules[import_379362]
            import_from_module(stypy.reporting.localization.Localization(__file__, 62, 4), 'numpy.distutils.core', sys_modules_379363.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 62, 4), __file__, sys_modules_379363, sys_modules_379363.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 62, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'numpy.distutils.core', import_379362)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
    
    
    # Call to setup(...): (line 63)
    # Processing the call keyword arguments (line 63)
    
    # Call to todict(...): (line 63)
    # Processing the call keyword arguments (line 63)
    kwargs_379371 = {}
    
    # Call to configuration(...): (line 63)
    # Processing the call keyword arguments (line 63)
    str_379366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 35), 'str', '')
    keyword_379367 = str_379366
    kwargs_379368 = {'top_path': keyword_379367}
    # Getting the type of 'configuration' (line 63)
    configuration_379365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 63)
    configuration_call_result_379369 = invoke(stypy.reporting.localization.Localization(__file__, 63, 12), configuration_379365, *[], **kwargs_379368)
    
    # Obtaining the member 'todict' of a type (line 63)
    todict_379370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 12), configuration_call_result_379369, 'todict')
    # Calling todict(args, kwargs) (line 63)
    todict_call_result_379372 = invoke(stypy.reporting.localization.Localization(__file__, 63, 12), todict_379370, *[], **kwargs_379371)
    
    kwargs_379373 = {'todict_call_result_379372': todict_call_result_379372}
    # Getting the type of 'setup' (line 63)
    setup_379364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 63)
    setup_call_result_379374 = invoke(stypy.reporting.localization.Localization(__file__, 63, 4), setup_379364, *[], **kwargs_379373)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
