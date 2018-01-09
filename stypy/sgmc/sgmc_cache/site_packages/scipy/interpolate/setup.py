
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
8:     from numpy.distutils.system_info import get_info
9: 
10:     lapack_opt = get_info('lapack_opt', notfound_action=2)
11: 
12:     config = Configuration('interpolate', parent_package, top_path)
13: 
14:     fitpack_src = [join('fitpack', '*.f')]
15:     config.add_library('fitpack', sources=fitpack_src)
16: 
17:     config.add_extension('interpnd',
18:                          sources=['interpnd.c'])
19: 
20:     config.add_extension('_ppoly',
21:                          sources=['_ppoly.c'],
22:                          **lapack_opt)
23: 
24:     config.add_extension('_bspl',
25:                          sources=['_bspl.c'],
26:                          libraries=['fitpack'],
27:                          depends=['src/__fitpack.h'] + fitpack_src)
28: 
29:     config.add_extension('_fitpack',
30:                          sources=['src/_fitpackmodule.c'],
31:                          libraries=['fitpack'],
32:                          depends=(['src/__fitpack.h','src/multipack.h']
33:                                   + fitpack_src)
34:                          )
35: 
36:     config.add_extension('dfitpack',
37:                          sources=['src/fitpack.pyf'],
38:                          libraries=['fitpack'],
39:                          depends=fitpack_src,
40:                          )
41: 
42:     config.add_extension('_interpolate',
43:                          sources=['src/_interpolate.cpp'],
44:                          include_dirs=['src'],
45:                          depends=['src/interpolate.h'])
46: 
47:     config.add_data_dir('tests')
48: 
49:     return config
50: 
51: if __name__ == '__main__':
52:     from numpy.distutils.core import setup
53:     setup(**configuration(top_path='').todict())
54: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from os.path import join' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_73541 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path')

if (type(import_73541) is not StypyTypeError):

    if (import_73541 != 'pyd_module'):
        __import__(import_73541)
        sys_modules_73542 = sys.modules[import_73541]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', sys_modules_73542.module_type_store, module_type_store, ['join'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_73542, sys_modules_73542.module_type_store, module_type_store)
    else:
        from os.path import join

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', None, module_type_store, ['join'], [join])

else:
    # Assigning a type to the variable 'os.path' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', import_73541)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')


@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_73543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 33), 'str', '')
    # Getting the type of 'None' (line 6)
    None_73544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 45), 'None')
    defaults = [str_73543, None_73544]
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
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
    import_73545 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util')

    if (type(import_73545) is not StypyTypeError):

        if (import_73545 != 'pyd_module'):
            __import__(import_73545)
            sys_modules_73546 = sys.modules[import_73545]
            import_from_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util', sys_modules_73546.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 7, 4), __file__, sys_modules_73546, sys_modules_73546.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util', import_73545)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 4))
    
    # 'from numpy.distutils.system_info import get_info' statement (line 8)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
    import_73547 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.system_info')

    if (type(import_73547) is not StypyTypeError):

        if (import_73547 != 'pyd_module'):
            __import__(import_73547)
            sys_modules_73548 = sys.modules[import_73547]
            import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.system_info', sys_modules_73548.module_type_store, module_type_store, ['get_info'])
            nest_module(stypy.reporting.localization.Localization(__file__, 8, 4), __file__, sys_modules_73548, sys_modules_73548.module_type_store, module_type_store)
        else:
            from numpy.distutils.system_info import get_info

            import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.system_info', None, module_type_store, ['get_info'], [get_info])

    else:
        # Assigning a type to the variable 'numpy.distutils.system_info' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.system_info', import_73547)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')
    
    
    # Assigning a Call to a Name (line 10):
    
    # Call to get_info(...): (line 10)
    # Processing the call arguments (line 10)
    str_73550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 26), 'str', 'lapack_opt')
    # Processing the call keyword arguments (line 10)
    int_73551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 56), 'int')
    keyword_73552 = int_73551
    kwargs_73553 = {'notfound_action': keyword_73552}
    # Getting the type of 'get_info' (line 10)
    get_info_73549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 17), 'get_info', False)
    # Calling get_info(args, kwargs) (line 10)
    get_info_call_result_73554 = invoke(stypy.reporting.localization.Localization(__file__, 10, 17), get_info_73549, *[str_73550], **kwargs_73553)
    
    # Assigning a type to the variable 'lapack_opt' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'lapack_opt', get_info_call_result_73554)
    
    # Assigning a Call to a Name (line 12):
    
    # Call to Configuration(...): (line 12)
    # Processing the call arguments (line 12)
    str_73556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 27), 'str', 'interpolate')
    # Getting the type of 'parent_package' (line 12)
    parent_package_73557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 42), 'parent_package', False)
    # Getting the type of 'top_path' (line 12)
    top_path_73558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 58), 'top_path', False)
    # Processing the call keyword arguments (line 12)
    kwargs_73559 = {}
    # Getting the type of 'Configuration' (line 12)
    Configuration_73555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 12)
    Configuration_call_result_73560 = invoke(stypy.reporting.localization.Localization(__file__, 12, 13), Configuration_73555, *[str_73556, parent_package_73557, top_path_73558], **kwargs_73559)
    
    # Assigning a type to the variable 'config' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'config', Configuration_call_result_73560)
    
    # Assigning a List to a Name (line 14):
    
    # Obtaining an instance of the builtin type 'list' (line 14)
    list_73561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 14)
    # Adding element type (line 14)
    
    # Call to join(...): (line 14)
    # Processing the call arguments (line 14)
    str_73563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 24), 'str', 'fitpack')
    str_73564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 35), 'str', '*.f')
    # Processing the call keyword arguments (line 14)
    kwargs_73565 = {}
    # Getting the type of 'join' (line 14)
    join_73562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 19), 'join', False)
    # Calling join(args, kwargs) (line 14)
    join_call_result_73566 = invoke(stypy.reporting.localization.Localization(__file__, 14, 19), join_73562, *[str_73563, str_73564], **kwargs_73565)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 18), list_73561, join_call_result_73566)
    
    # Assigning a type to the variable 'fitpack_src' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'fitpack_src', list_73561)
    
    # Call to add_library(...): (line 15)
    # Processing the call arguments (line 15)
    str_73569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 23), 'str', 'fitpack')
    # Processing the call keyword arguments (line 15)
    # Getting the type of 'fitpack_src' (line 15)
    fitpack_src_73570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 42), 'fitpack_src', False)
    keyword_73571 = fitpack_src_73570
    kwargs_73572 = {'sources': keyword_73571}
    # Getting the type of 'config' (line 15)
    config_73567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'config', False)
    # Obtaining the member 'add_library' of a type (line 15)
    add_library_73568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 4), config_73567, 'add_library')
    # Calling add_library(args, kwargs) (line 15)
    add_library_call_result_73573 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), add_library_73568, *[str_73569], **kwargs_73572)
    
    
    # Call to add_extension(...): (line 17)
    # Processing the call arguments (line 17)
    str_73576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 25), 'str', 'interpnd')
    # Processing the call keyword arguments (line 17)
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_73577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    # Adding element type (line 18)
    str_73578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 34), 'str', 'interpnd.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 33), list_73577, str_73578)
    
    keyword_73579 = list_73577
    kwargs_73580 = {'sources': keyword_73579}
    # Getting the type of 'config' (line 17)
    config_73574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 17)
    add_extension_73575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 4), config_73574, 'add_extension')
    # Calling add_extension(args, kwargs) (line 17)
    add_extension_call_result_73581 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), add_extension_73575, *[str_73576], **kwargs_73580)
    
    
    # Call to add_extension(...): (line 20)
    # Processing the call arguments (line 20)
    str_73584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 25), 'str', '_ppoly')
    # Processing the call keyword arguments (line 20)
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_73585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    str_73586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 34), 'str', '_ppoly.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 33), list_73585, str_73586)
    
    keyword_73587 = list_73585
    # Getting the type of 'lapack_opt' (line 22)
    lapack_opt_73588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 27), 'lapack_opt', False)
    kwargs_73589 = {'sources': keyword_73587, 'lapack_opt_73588': lapack_opt_73588}
    # Getting the type of 'config' (line 20)
    config_73582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 20)
    add_extension_73583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 4), config_73582, 'add_extension')
    # Calling add_extension(args, kwargs) (line 20)
    add_extension_call_result_73590 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), add_extension_73583, *[str_73584], **kwargs_73589)
    
    
    # Call to add_extension(...): (line 24)
    # Processing the call arguments (line 24)
    str_73593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'str', '_bspl')
    # Processing the call keyword arguments (line 24)
    
    # Obtaining an instance of the builtin type 'list' (line 25)
    list_73594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 25)
    # Adding element type (line 25)
    str_73595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 34), 'str', '_bspl.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 33), list_73594, str_73595)
    
    keyword_73596 = list_73594
    
    # Obtaining an instance of the builtin type 'list' (line 26)
    list_73597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 26)
    # Adding element type (line 26)
    str_73598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 36), 'str', 'fitpack')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 35), list_73597, str_73598)
    
    keyword_73599 = list_73597
    
    # Obtaining an instance of the builtin type 'list' (line 27)
    list_73600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 27)
    # Adding element type (line 27)
    str_73601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 34), 'str', 'src/__fitpack.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 33), list_73600, str_73601)
    
    # Getting the type of 'fitpack_src' (line 27)
    fitpack_src_73602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 55), 'fitpack_src', False)
    # Applying the binary operator '+' (line 27)
    result_add_73603 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 33), '+', list_73600, fitpack_src_73602)
    
    keyword_73604 = result_add_73603
    kwargs_73605 = {'libraries': keyword_73599, 'sources': keyword_73596, 'depends': keyword_73604}
    # Getting the type of 'config' (line 24)
    config_73591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 24)
    add_extension_73592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 4), config_73591, 'add_extension')
    # Calling add_extension(args, kwargs) (line 24)
    add_extension_call_result_73606 = invoke(stypy.reporting.localization.Localization(__file__, 24, 4), add_extension_73592, *[str_73593], **kwargs_73605)
    
    
    # Call to add_extension(...): (line 29)
    # Processing the call arguments (line 29)
    str_73609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 25), 'str', '_fitpack')
    # Processing the call keyword arguments (line 29)
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_73610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    str_73611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 34), 'str', 'src/_fitpackmodule.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 33), list_73610, str_73611)
    
    keyword_73612 = list_73610
    
    # Obtaining an instance of the builtin type 'list' (line 31)
    list_73613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 31)
    # Adding element type (line 31)
    str_73614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 36), 'str', 'fitpack')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 35), list_73613, str_73614)
    
    keyword_73615 = list_73613
    
    # Obtaining an instance of the builtin type 'list' (line 32)
    list_73616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 32)
    # Adding element type (line 32)
    str_73617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 35), 'str', 'src/__fitpack.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 34), list_73616, str_73617)
    # Adding element type (line 32)
    str_73618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 53), 'str', 'src/multipack.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 34), list_73616, str_73618)
    
    # Getting the type of 'fitpack_src' (line 33)
    fitpack_src_73619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 36), 'fitpack_src', False)
    # Applying the binary operator '+' (line 32)
    result_add_73620 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 34), '+', list_73616, fitpack_src_73619)
    
    keyword_73621 = result_add_73620
    kwargs_73622 = {'libraries': keyword_73615, 'sources': keyword_73612, 'depends': keyword_73621}
    # Getting the type of 'config' (line 29)
    config_73607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 29)
    add_extension_73608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 4), config_73607, 'add_extension')
    # Calling add_extension(args, kwargs) (line 29)
    add_extension_call_result_73623 = invoke(stypy.reporting.localization.Localization(__file__, 29, 4), add_extension_73608, *[str_73609], **kwargs_73622)
    
    
    # Call to add_extension(...): (line 36)
    # Processing the call arguments (line 36)
    str_73626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 25), 'str', 'dfitpack')
    # Processing the call keyword arguments (line 36)
    
    # Obtaining an instance of the builtin type 'list' (line 37)
    list_73627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 37)
    # Adding element type (line 37)
    str_73628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 34), 'str', 'src/fitpack.pyf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 33), list_73627, str_73628)
    
    keyword_73629 = list_73627
    
    # Obtaining an instance of the builtin type 'list' (line 38)
    list_73630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 38)
    # Adding element type (line 38)
    str_73631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 36), 'str', 'fitpack')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 35), list_73630, str_73631)
    
    keyword_73632 = list_73630
    # Getting the type of 'fitpack_src' (line 39)
    fitpack_src_73633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 33), 'fitpack_src', False)
    keyword_73634 = fitpack_src_73633
    kwargs_73635 = {'libraries': keyword_73632, 'sources': keyword_73629, 'depends': keyword_73634}
    # Getting the type of 'config' (line 36)
    config_73624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 36)
    add_extension_73625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 4), config_73624, 'add_extension')
    # Calling add_extension(args, kwargs) (line 36)
    add_extension_call_result_73636 = invoke(stypy.reporting.localization.Localization(__file__, 36, 4), add_extension_73625, *[str_73626], **kwargs_73635)
    
    
    # Call to add_extension(...): (line 42)
    # Processing the call arguments (line 42)
    str_73639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 25), 'str', '_interpolate')
    # Processing the call keyword arguments (line 42)
    
    # Obtaining an instance of the builtin type 'list' (line 43)
    list_73640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 43)
    # Adding element type (line 43)
    str_73641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 34), 'str', 'src/_interpolate.cpp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 33), list_73640, str_73641)
    
    keyword_73642 = list_73640
    
    # Obtaining an instance of the builtin type 'list' (line 44)
    list_73643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 44)
    # Adding element type (line 44)
    str_73644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 39), 'str', 'src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 38), list_73643, str_73644)
    
    keyword_73645 = list_73643
    
    # Obtaining an instance of the builtin type 'list' (line 45)
    list_73646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 45)
    # Adding element type (line 45)
    str_73647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 34), 'str', 'src/interpolate.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 33), list_73646, str_73647)
    
    keyword_73648 = list_73646
    kwargs_73649 = {'sources': keyword_73642, 'depends': keyword_73648, 'include_dirs': keyword_73645}
    # Getting the type of 'config' (line 42)
    config_73637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 42)
    add_extension_73638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 4), config_73637, 'add_extension')
    # Calling add_extension(args, kwargs) (line 42)
    add_extension_call_result_73650 = invoke(stypy.reporting.localization.Localization(__file__, 42, 4), add_extension_73638, *[str_73639], **kwargs_73649)
    
    
    # Call to add_data_dir(...): (line 47)
    # Processing the call arguments (line 47)
    str_73653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 47)
    kwargs_73654 = {}
    # Getting the type of 'config' (line 47)
    config_73651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 47)
    add_data_dir_73652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 4), config_73651, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 47)
    add_data_dir_call_result_73655 = invoke(stypy.reporting.localization.Localization(__file__, 47, 4), add_data_dir_73652, *[str_73653], **kwargs_73654)
    
    # Getting the type of 'config' (line 49)
    config_73656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type', config_73656)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_73657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_73657)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_73657

# Assigning a type to the variable 'configuration' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 52, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 52)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
    import_73658 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 52, 4), 'numpy.distutils.core')

    if (type(import_73658) is not StypyTypeError):

        if (import_73658 != 'pyd_module'):
            __import__(import_73658)
            sys_modules_73659 = sys.modules[import_73658]
            import_from_module(stypy.reporting.localization.Localization(__file__, 52, 4), 'numpy.distutils.core', sys_modules_73659.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 52, 4), __file__, sys_modules_73659, sys_modules_73659.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 52, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'numpy.distutils.core', import_73658)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')
    
    
    # Call to setup(...): (line 53)
    # Processing the call keyword arguments (line 53)
    
    # Call to todict(...): (line 53)
    # Processing the call keyword arguments (line 53)
    kwargs_73667 = {}
    
    # Call to configuration(...): (line 53)
    # Processing the call keyword arguments (line 53)
    str_73662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 35), 'str', '')
    keyword_73663 = str_73662
    kwargs_73664 = {'top_path': keyword_73663}
    # Getting the type of 'configuration' (line 53)
    configuration_73661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 53)
    configuration_call_result_73665 = invoke(stypy.reporting.localization.Localization(__file__, 53, 12), configuration_73661, *[], **kwargs_73664)
    
    # Obtaining the member 'todict' of a type (line 53)
    todict_73666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 12), configuration_call_result_73665, 'todict')
    # Calling todict(args, kwargs) (line 53)
    todict_call_result_73668 = invoke(stypy.reporting.localization.Localization(__file__, 53, 12), todict_73666, *[], **kwargs_73667)
    
    kwargs_73669 = {'todict_call_result_73668': todict_call_result_73668}
    # Getting the type of 'setup' (line 53)
    setup_73660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 53)
    setup_call_result_73670 = invoke(stypy.reporting.localization.Localization(__file__, 53, 4), setup_73660, *[], **kwargs_73669)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
