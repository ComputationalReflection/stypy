
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import os
4: 
5: 
6: def configuration(parent_package='',top_path=None):
7:     from numpy.distutils.misc_util import Configuration
8: 
9:     config = Configuration('_lib', parent_package, top_path)
10:     config.add_data_files('tests/*.py')
11: 
12:     include_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
13:     depends = [os.path.join(include_dir, 'ccallback.h')]
14: 
15:     config.add_extension("_ccallback_c",
16:                          sources=["_ccallback_c.c"],
17:                          depends=depends,
18:                          include_dirs=[include_dir])
19: 
20:     config.add_extension("_test_ccallback",
21:                          sources=["src/_test_ccallback.c"],
22:                          depends=depends,
23:                          include_dirs=[include_dir])
24: 
25:     config.add_extension("_fpumode",
26:                          sources=["_fpumode.c"])
27: 
28:     def get_messagestream_config(ext, build_dir):
29:         # Generate a header file containing defines
30:         config_cmd = config.get_config_cmd()
31:         defines = []
32:         if config_cmd.check_func('open_memstream', decl=True, call=True):
33:             defines.append(('HAVE_OPEN_MEMSTREAM', '1'))
34:         target = os.path.join(os.path.dirname(__file__), 'src',
35:                               'messagestream_config.h')
36:         with open(target, 'w') as f:
37:             for name, value in defines:
38:                 f.write('#define {0} {1}\n'.format(name, value))
39: 
40:     depends = [os.path.join(include_dir, 'messagestream.h')]
41:     config.add_extension("messagestream",
42:                          sources=["messagestream.c"] + [get_messagestream_config],
43:                          depends=depends,
44:                          include_dirs=[include_dir])
45: 
46:     return config
47: 
48: 
49: if __name__ == '__main__':
50:     from numpy.distutils.core import setup
51: 
52:     setup(**configuration(top_path='').todict())
53: 

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


@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_707521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 33), 'str', '')
    # Getting the type of 'None' (line 6)
    None_707522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 45), 'None')
    defaults = [str_707521, None_707522]
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
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/')
    import_707523 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util')

    if (type(import_707523) is not StypyTypeError):

        if (import_707523 != 'pyd_module'):
            __import__(import_707523)
            sys_modules_707524 = sys.modules[import_707523]
            import_from_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util', sys_modules_707524.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 7, 4), __file__, sys_modules_707524, sys_modules_707524.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util', import_707523)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/')
    
    
    # Assigning a Call to a Name (line 9):
    
    # Call to Configuration(...): (line 9)
    # Processing the call arguments (line 9)
    str_707526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 27), 'str', '_lib')
    # Getting the type of 'parent_package' (line 9)
    parent_package_707527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 35), 'parent_package', False)
    # Getting the type of 'top_path' (line 9)
    top_path_707528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 51), 'top_path', False)
    # Processing the call keyword arguments (line 9)
    kwargs_707529 = {}
    # Getting the type of 'Configuration' (line 9)
    Configuration_707525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 9)
    Configuration_call_result_707530 = invoke(stypy.reporting.localization.Localization(__file__, 9, 13), Configuration_707525, *[str_707526, parent_package_707527, top_path_707528], **kwargs_707529)
    
    # Assigning a type to the variable 'config' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'config', Configuration_call_result_707530)
    
    # Call to add_data_files(...): (line 10)
    # Processing the call arguments (line 10)
    str_707533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 26), 'str', 'tests/*.py')
    # Processing the call keyword arguments (line 10)
    kwargs_707534 = {}
    # Getting the type of 'config' (line 10)
    config_707531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'config', False)
    # Obtaining the member 'add_data_files' of a type (line 10)
    add_data_files_707532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 4), config_707531, 'add_data_files')
    # Calling add_data_files(args, kwargs) (line 10)
    add_data_files_call_result_707535 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), add_data_files_707532, *[str_707533], **kwargs_707534)
    
    
    # Assigning a Call to a Name (line 12):
    
    # Call to abspath(...): (line 12)
    # Processing the call arguments (line 12)
    
    # Call to join(...): (line 12)
    # Processing the call arguments (line 12)
    
    # Call to dirname(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of '__file__' (line 12)
    file___707545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 63), '__file__', False)
    # Processing the call keyword arguments (line 12)
    kwargs_707546 = {}
    # Getting the type of 'os' (line 12)
    os_707542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 47), 'os', False)
    # Obtaining the member 'path' of a type (line 12)
    path_707543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 47), os_707542, 'path')
    # Obtaining the member 'dirname' of a type (line 12)
    dirname_707544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 47), path_707543, 'dirname')
    # Calling dirname(args, kwargs) (line 12)
    dirname_call_result_707547 = invoke(stypy.reporting.localization.Localization(__file__, 12, 47), dirname_707544, *[file___707545], **kwargs_707546)
    
    str_707548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 74), 'str', 'src')
    # Processing the call keyword arguments (line 12)
    kwargs_707549 = {}
    # Getting the type of 'os' (line 12)
    os_707539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 34), 'os', False)
    # Obtaining the member 'path' of a type (line 12)
    path_707540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 34), os_707539, 'path')
    # Obtaining the member 'join' of a type (line 12)
    join_707541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 34), path_707540, 'join')
    # Calling join(args, kwargs) (line 12)
    join_call_result_707550 = invoke(stypy.reporting.localization.Localization(__file__, 12, 34), join_707541, *[dirname_call_result_707547, str_707548], **kwargs_707549)
    
    # Processing the call keyword arguments (line 12)
    kwargs_707551 = {}
    # Getting the type of 'os' (line 12)
    os_707536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 18), 'os', False)
    # Obtaining the member 'path' of a type (line 12)
    path_707537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 18), os_707536, 'path')
    # Obtaining the member 'abspath' of a type (line 12)
    abspath_707538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 18), path_707537, 'abspath')
    # Calling abspath(args, kwargs) (line 12)
    abspath_call_result_707552 = invoke(stypy.reporting.localization.Localization(__file__, 12, 18), abspath_707538, *[join_call_result_707550], **kwargs_707551)
    
    # Assigning a type to the variable 'include_dir' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'include_dir', abspath_call_result_707552)
    
    # Assigning a List to a Name (line 13):
    
    # Obtaining an instance of the builtin type 'list' (line 13)
    list_707553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 13)
    # Adding element type (line 13)
    
    # Call to join(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'include_dir' (line 13)
    include_dir_707557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 28), 'include_dir', False)
    str_707558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 41), 'str', 'ccallback.h')
    # Processing the call keyword arguments (line 13)
    kwargs_707559 = {}
    # Getting the type of 'os' (line 13)
    os_707554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 13)
    path_707555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 15), os_707554, 'path')
    # Obtaining the member 'join' of a type (line 13)
    join_707556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 15), path_707555, 'join')
    # Calling join(args, kwargs) (line 13)
    join_call_result_707560 = invoke(stypy.reporting.localization.Localization(__file__, 13, 15), join_707556, *[include_dir_707557, str_707558], **kwargs_707559)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 14), list_707553, join_call_result_707560)
    
    # Assigning a type to the variable 'depends' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'depends', list_707553)
    
    # Call to add_extension(...): (line 15)
    # Processing the call arguments (line 15)
    str_707563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 25), 'str', '_ccallback_c')
    # Processing the call keyword arguments (line 15)
    
    # Obtaining an instance of the builtin type 'list' (line 16)
    list_707564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 16)
    # Adding element type (line 16)
    str_707565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 34), 'str', '_ccallback_c.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 33), list_707564, str_707565)
    
    keyword_707566 = list_707564
    # Getting the type of 'depends' (line 17)
    depends_707567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 33), 'depends', False)
    keyword_707568 = depends_707567
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_707569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    # Adding element type (line 18)
    # Getting the type of 'include_dir' (line 18)
    include_dir_707570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 39), 'include_dir', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 38), list_707569, include_dir_707570)
    
    keyword_707571 = list_707569
    kwargs_707572 = {'sources': keyword_707566, 'depends': keyword_707568, 'include_dirs': keyword_707571}
    # Getting the type of 'config' (line 15)
    config_707561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 15)
    add_extension_707562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 4), config_707561, 'add_extension')
    # Calling add_extension(args, kwargs) (line 15)
    add_extension_call_result_707573 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), add_extension_707562, *[str_707563], **kwargs_707572)
    
    
    # Call to add_extension(...): (line 20)
    # Processing the call arguments (line 20)
    str_707576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 25), 'str', '_test_ccallback')
    # Processing the call keyword arguments (line 20)
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_707577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    str_707578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 34), 'str', 'src/_test_ccallback.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 33), list_707577, str_707578)
    
    keyword_707579 = list_707577
    # Getting the type of 'depends' (line 22)
    depends_707580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 33), 'depends', False)
    keyword_707581 = depends_707580
    
    # Obtaining an instance of the builtin type 'list' (line 23)
    list_707582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 23)
    # Adding element type (line 23)
    # Getting the type of 'include_dir' (line 23)
    include_dir_707583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 39), 'include_dir', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 38), list_707582, include_dir_707583)
    
    keyword_707584 = list_707582
    kwargs_707585 = {'sources': keyword_707579, 'depends': keyword_707581, 'include_dirs': keyword_707584}
    # Getting the type of 'config' (line 20)
    config_707574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 20)
    add_extension_707575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 4), config_707574, 'add_extension')
    # Calling add_extension(args, kwargs) (line 20)
    add_extension_call_result_707586 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), add_extension_707575, *[str_707576], **kwargs_707585)
    
    
    # Call to add_extension(...): (line 25)
    # Processing the call arguments (line 25)
    str_707589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 25), 'str', '_fpumode')
    # Processing the call keyword arguments (line 25)
    
    # Obtaining an instance of the builtin type 'list' (line 26)
    list_707590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 26)
    # Adding element type (line 26)
    str_707591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 34), 'str', '_fpumode.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 33), list_707590, str_707591)
    
    keyword_707592 = list_707590
    kwargs_707593 = {'sources': keyword_707592}
    # Getting the type of 'config' (line 25)
    config_707587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 25)
    add_extension_707588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 4), config_707587, 'add_extension')
    # Calling add_extension(args, kwargs) (line 25)
    add_extension_call_result_707594 = invoke(stypy.reporting.localization.Localization(__file__, 25, 4), add_extension_707588, *[str_707589], **kwargs_707593)
    

    @norecursion
    def get_messagestream_config(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_messagestream_config'
        module_type_store = module_type_store.open_function_context('get_messagestream_config', 28, 4, False)
        
        # Passed parameters checking function
        get_messagestream_config.stypy_localization = localization
        get_messagestream_config.stypy_type_of_self = None
        get_messagestream_config.stypy_type_store = module_type_store
        get_messagestream_config.stypy_function_name = 'get_messagestream_config'
        get_messagestream_config.stypy_param_names_list = ['ext', 'build_dir']
        get_messagestream_config.stypy_varargs_param_name = None
        get_messagestream_config.stypy_kwargs_param_name = None
        get_messagestream_config.stypy_call_defaults = defaults
        get_messagestream_config.stypy_call_varargs = varargs
        get_messagestream_config.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'get_messagestream_config', ['ext', 'build_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_messagestream_config', localization, ['ext', 'build_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_messagestream_config(...)' code ##################

        
        # Assigning a Call to a Name (line 30):
        
        # Call to get_config_cmd(...): (line 30)
        # Processing the call keyword arguments (line 30)
        kwargs_707597 = {}
        # Getting the type of 'config' (line 30)
        config_707595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 21), 'config', False)
        # Obtaining the member 'get_config_cmd' of a type (line 30)
        get_config_cmd_707596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 21), config_707595, 'get_config_cmd')
        # Calling get_config_cmd(args, kwargs) (line 30)
        get_config_cmd_call_result_707598 = invoke(stypy.reporting.localization.Localization(__file__, 30, 21), get_config_cmd_707596, *[], **kwargs_707597)
        
        # Assigning a type to the variable 'config_cmd' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'config_cmd', get_config_cmd_call_result_707598)
        
        # Assigning a List to a Name (line 31):
        
        # Obtaining an instance of the builtin type 'list' (line 31)
        list_707599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 31)
        
        # Assigning a type to the variable 'defines' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'defines', list_707599)
        
        
        # Call to check_func(...): (line 32)
        # Processing the call arguments (line 32)
        str_707602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 33), 'str', 'open_memstream')
        # Processing the call keyword arguments (line 32)
        # Getting the type of 'True' (line 32)
        True_707603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 56), 'True', False)
        keyword_707604 = True_707603
        # Getting the type of 'True' (line 32)
        True_707605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 67), 'True', False)
        keyword_707606 = True_707605
        kwargs_707607 = {'decl': keyword_707604, 'call': keyword_707606}
        # Getting the type of 'config_cmd' (line 32)
        config_cmd_707600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), 'config_cmd', False)
        # Obtaining the member 'check_func' of a type (line 32)
        check_func_707601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 11), config_cmd_707600, 'check_func')
        # Calling check_func(args, kwargs) (line 32)
        check_func_call_result_707608 = invoke(stypy.reporting.localization.Localization(__file__, 32, 11), check_func_707601, *[str_707602], **kwargs_707607)
        
        # Testing the type of an if condition (line 32)
        if_condition_707609 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 8), check_func_call_result_707608)
        # Assigning a type to the variable 'if_condition_707609' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'if_condition_707609', if_condition_707609)
        # SSA begins for if statement (line 32)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 33)
        # Processing the call arguments (line 33)
        
        # Obtaining an instance of the builtin type 'tuple' (line 33)
        tuple_707612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 33)
        # Adding element type (line 33)
        str_707613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 28), 'str', 'HAVE_OPEN_MEMSTREAM')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 28), tuple_707612, str_707613)
        # Adding element type (line 33)
        str_707614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 51), 'str', '1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 28), tuple_707612, str_707614)
        
        # Processing the call keyword arguments (line 33)
        kwargs_707615 = {}
        # Getting the type of 'defines' (line 33)
        defines_707610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'defines', False)
        # Obtaining the member 'append' of a type (line 33)
        append_707611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), defines_707610, 'append')
        # Calling append(args, kwargs) (line 33)
        append_call_result_707616 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), append_707611, *[tuple_707612], **kwargs_707615)
        
        # SSA join for if statement (line 32)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 34):
        
        # Call to join(...): (line 34)
        # Processing the call arguments (line 34)
        
        # Call to dirname(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of '__file__' (line 34)
        file___707623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 46), '__file__', False)
        # Processing the call keyword arguments (line 34)
        kwargs_707624 = {}
        # Getting the type of 'os' (line 34)
        os_707620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 30), 'os', False)
        # Obtaining the member 'path' of a type (line 34)
        path_707621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 30), os_707620, 'path')
        # Obtaining the member 'dirname' of a type (line 34)
        dirname_707622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 30), path_707621, 'dirname')
        # Calling dirname(args, kwargs) (line 34)
        dirname_call_result_707625 = invoke(stypy.reporting.localization.Localization(__file__, 34, 30), dirname_707622, *[file___707623], **kwargs_707624)
        
        str_707626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 57), 'str', 'src')
        str_707627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 30), 'str', 'messagestream_config.h')
        # Processing the call keyword arguments (line 34)
        kwargs_707628 = {}
        # Getting the type of 'os' (line 34)
        os_707617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 34)
        path_707618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 17), os_707617, 'path')
        # Obtaining the member 'join' of a type (line 34)
        join_707619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 17), path_707618, 'join')
        # Calling join(args, kwargs) (line 34)
        join_call_result_707629 = invoke(stypy.reporting.localization.Localization(__file__, 34, 17), join_707619, *[dirname_call_result_707625, str_707626, str_707627], **kwargs_707628)
        
        # Assigning a type to the variable 'target' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'target', join_call_result_707629)
        
        # Call to open(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'target' (line 36)
        target_707631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 18), 'target', False)
        str_707632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 26), 'str', 'w')
        # Processing the call keyword arguments (line 36)
        kwargs_707633 = {}
        # Getting the type of 'open' (line 36)
        open_707630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 13), 'open', False)
        # Calling open(args, kwargs) (line 36)
        open_call_result_707634 = invoke(stypy.reporting.localization.Localization(__file__, 36, 13), open_707630, *[target_707631, str_707632], **kwargs_707633)
        
        with_707635 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 36, 13), open_call_result_707634, 'with parameter', '__enter__', '__exit__')

        if with_707635:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 36)
            enter___707636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 13), open_call_result_707634, '__enter__')
            with_enter_707637 = invoke(stypy.reporting.localization.Localization(__file__, 36, 13), enter___707636)
            # Assigning a type to the variable 'f' (line 36)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 13), 'f', with_enter_707637)
            
            # Getting the type of 'defines' (line 37)
            defines_707638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 31), 'defines')
            # Testing the type of a for loop iterable (line 37)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 37, 12), defines_707638)
            # Getting the type of the for loop variable (line 37)
            for_loop_var_707639 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 37, 12), defines_707638)
            # Assigning a type to the variable 'name' (line 37)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 12), for_loop_var_707639))
            # Assigning a type to the variable 'value' (line 37)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 12), for_loop_var_707639))
            # SSA begins for a for statement (line 37)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to write(...): (line 38)
            # Processing the call arguments (line 38)
            
            # Call to format(...): (line 38)
            # Processing the call arguments (line 38)
            # Getting the type of 'name' (line 38)
            name_707644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 51), 'name', False)
            # Getting the type of 'value' (line 38)
            value_707645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 57), 'value', False)
            # Processing the call keyword arguments (line 38)
            kwargs_707646 = {}
            str_707642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 24), 'str', '#define {0} {1}\n')
            # Obtaining the member 'format' of a type (line 38)
            format_707643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 24), str_707642, 'format')
            # Calling format(args, kwargs) (line 38)
            format_call_result_707647 = invoke(stypy.reporting.localization.Localization(__file__, 38, 24), format_707643, *[name_707644, value_707645], **kwargs_707646)
            
            # Processing the call keyword arguments (line 38)
            kwargs_707648 = {}
            # Getting the type of 'f' (line 38)
            f_707640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'f', False)
            # Obtaining the member 'write' of a type (line 38)
            write_707641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 16), f_707640, 'write')
            # Calling write(args, kwargs) (line 38)
            write_call_result_707649 = invoke(stypy.reporting.localization.Localization(__file__, 38, 16), write_707641, *[format_call_result_707647], **kwargs_707648)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 36)
            exit___707650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 13), open_call_result_707634, '__exit__')
            with_exit_707651 = invoke(stypy.reporting.localization.Localization(__file__, 36, 13), exit___707650, None, None, None)

        
        # ################# End of 'get_messagestream_config(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_messagestream_config' in the type store
        # Getting the type of 'stypy_return_type' (line 28)
        stypy_return_type_707652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_707652)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_messagestream_config'
        return stypy_return_type_707652

    # Assigning a type to the variable 'get_messagestream_config' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'get_messagestream_config', get_messagestream_config)
    
    # Assigning a List to a Name (line 40):
    
    # Obtaining an instance of the builtin type 'list' (line 40)
    list_707653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 40)
    # Adding element type (line 40)
    
    # Call to join(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'include_dir' (line 40)
    include_dir_707657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 28), 'include_dir', False)
    str_707658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 41), 'str', 'messagestream.h')
    # Processing the call keyword arguments (line 40)
    kwargs_707659 = {}
    # Getting the type of 'os' (line 40)
    os_707654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 40)
    path_707655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 15), os_707654, 'path')
    # Obtaining the member 'join' of a type (line 40)
    join_707656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 15), path_707655, 'join')
    # Calling join(args, kwargs) (line 40)
    join_call_result_707660 = invoke(stypy.reporting.localization.Localization(__file__, 40, 15), join_707656, *[include_dir_707657, str_707658], **kwargs_707659)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 14), list_707653, join_call_result_707660)
    
    # Assigning a type to the variable 'depends' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'depends', list_707653)
    
    # Call to add_extension(...): (line 41)
    # Processing the call arguments (line 41)
    str_707663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 25), 'str', 'messagestream')
    # Processing the call keyword arguments (line 41)
    
    # Obtaining an instance of the builtin type 'list' (line 42)
    list_707664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 42)
    # Adding element type (line 42)
    str_707665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 34), 'str', 'messagestream.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 33), list_707664, str_707665)
    
    
    # Obtaining an instance of the builtin type 'list' (line 42)
    list_707666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 55), 'list')
    # Adding type elements to the builtin type 'list' instance (line 42)
    # Adding element type (line 42)
    # Getting the type of 'get_messagestream_config' (line 42)
    get_messagestream_config_707667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 56), 'get_messagestream_config', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 55), list_707666, get_messagestream_config_707667)
    
    # Applying the binary operator '+' (line 42)
    result_add_707668 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 33), '+', list_707664, list_707666)
    
    keyword_707669 = result_add_707668
    # Getting the type of 'depends' (line 43)
    depends_707670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 33), 'depends', False)
    keyword_707671 = depends_707670
    
    # Obtaining an instance of the builtin type 'list' (line 44)
    list_707672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 44)
    # Adding element type (line 44)
    # Getting the type of 'include_dir' (line 44)
    include_dir_707673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 39), 'include_dir', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 38), list_707672, include_dir_707673)
    
    keyword_707674 = list_707672
    kwargs_707675 = {'sources': keyword_707669, 'depends': keyword_707671, 'include_dirs': keyword_707674}
    # Getting the type of 'config' (line 41)
    config_707661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 41)
    add_extension_707662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 4), config_707661, 'add_extension')
    # Calling add_extension(args, kwargs) (line 41)
    add_extension_call_result_707676 = invoke(stypy.reporting.localization.Localization(__file__, 41, 4), add_extension_707662, *[str_707663], **kwargs_707675)
    
    # Getting the type of 'config' (line 46)
    config_707677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type', config_707677)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_707678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_707678)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_707678

# Assigning a type to the variable 'configuration' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 50, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 50)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/')
    import_707679 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 50, 4), 'numpy.distutils.core')

    if (type(import_707679) is not StypyTypeError):

        if (import_707679 != 'pyd_module'):
            __import__(import_707679)
            sys_modules_707680 = sys.modules[import_707679]
            import_from_module(stypy.reporting.localization.Localization(__file__, 50, 4), 'numpy.distutils.core', sys_modules_707680.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 50, 4), __file__, sys_modules_707680, sys_modules_707680.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 50, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'numpy.distutils.core', import_707679)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/')
    
    
    # Call to setup(...): (line 52)
    # Processing the call keyword arguments (line 52)
    
    # Call to todict(...): (line 52)
    # Processing the call keyword arguments (line 52)
    kwargs_707688 = {}
    
    # Call to configuration(...): (line 52)
    # Processing the call keyword arguments (line 52)
    str_707683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 35), 'str', '')
    keyword_707684 = str_707683
    kwargs_707685 = {'top_path': keyword_707684}
    # Getting the type of 'configuration' (line 52)
    configuration_707682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 52)
    configuration_call_result_707686 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), configuration_707682, *[], **kwargs_707685)
    
    # Obtaining the member 'todict' of a type (line 52)
    todict_707687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 12), configuration_call_result_707686, 'todict')
    # Calling todict(args, kwargs) (line 52)
    todict_call_result_707689 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), todict_707687, *[], **kwargs_707688)
    
    kwargs_707690 = {'todict_call_result_707689': todict_call_result_707689}
    # Getting the type of 'setup' (line 52)
    setup_707681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 52)
    setup_call_result_707691 = invoke(stypy.reporting.localization.Localization(__file__, 52, 4), setup_707681, *[], **kwargs_707690)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
