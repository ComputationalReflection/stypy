
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Created by Pearu Peterson, August 2002
2: from __future__ import division, print_function, absolute_import
3: 
4: 
5: from os.path import join
6: 
7: 
8: def configuration(parent_package='',top_path=None):
9:     from numpy.distutils.misc_util import Configuration
10: 
11:     config = Configuration('fftpack',parent_package, top_path)
12: 
13:     config.add_data_dir('tests')
14: 
15:     dfftpack_src = [join('src/dfftpack','*.f')]
16:     config.add_library('dfftpack', sources=dfftpack_src)
17: 
18:     fftpack_src = [join('src/fftpack','*.f')]
19:     config.add_library('fftpack', sources=fftpack_src)
20: 
21:     sources = ['fftpack.pyf','src/zfft.c','src/drfft.c','src/zrfft.c',
22:                'src/zfftnd.c', 'src/dct.c.src', 'src/dst.c.src']
23: 
24:     config.add_extension('_fftpack',
25:         sources=sources,
26:         libraries=['dfftpack', 'fftpack'],
27:         include_dirs=['src'],
28:         depends=(dfftpack_src + fftpack_src))
29: 
30:     config.add_extension('convolve',
31:         sources=['convolve.pyf','src/convolve.c'],
32:         libraries=['dfftpack'],
33:         depends=dfftpack_src,
34:     )
35:     return config
36: 
37: 
38: if __name__ == '__main__':
39:     from numpy.distutils.core import setup
40:     setup(**configuration(top_path='').todict())
41: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from os.path import join' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/')
import_18537 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os.path')

if (type(import_18537) is not StypyTypeError):

    if (import_18537 != 'pyd_module'):
        __import__(import_18537)
        sys_modules_18538 = sys.modules[import_18537]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os.path', sys_modules_18538.module_type_store, module_type_store, ['join'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_18538, sys_modules_18538.module_type_store, module_type_store)
    else:
        from os.path import join

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os.path', None, module_type_store, ['join'], [join])

else:
    # Assigning a type to the variable 'os.path' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'os.path', import_18537)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/')


@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_18539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 33), 'str', '')
    # Getting the type of 'None' (line 8)
    None_18540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 45), 'None')
    defaults = [str_18539, None_18540]
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
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/')
    import_18541 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.misc_util')

    if (type(import_18541) is not StypyTypeError):

        if (import_18541 != 'pyd_module'):
            __import__(import_18541)
            sys_modules_18542 = sys.modules[import_18541]
            import_from_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.misc_util', sys_modules_18542.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 9, 4), __file__, sys_modules_18542, sys_modules_18542.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.misc_util', import_18541)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/')
    
    
    # Assigning a Call to a Name (line 11):
    
    # Call to Configuration(...): (line 11)
    # Processing the call arguments (line 11)
    str_18544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 27), 'str', 'fftpack')
    # Getting the type of 'parent_package' (line 11)
    parent_package_18545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 37), 'parent_package', False)
    # Getting the type of 'top_path' (line 11)
    top_path_18546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 53), 'top_path', False)
    # Processing the call keyword arguments (line 11)
    kwargs_18547 = {}
    # Getting the type of 'Configuration' (line 11)
    Configuration_18543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 11)
    Configuration_call_result_18548 = invoke(stypy.reporting.localization.Localization(__file__, 11, 13), Configuration_18543, *[str_18544, parent_package_18545, top_path_18546], **kwargs_18547)
    
    # Assigning a type to the variable 'config' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'config', Configuration_call_result_18548)
    
    # Call to add_data_dir(...): (line 13)
    # Processing the call arguments (line 13)
    str_18551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 13)
    kwargs_18552 = {}
    # Getting the type of 'config' (line 13)
    config_18549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 13)
    add_data_dir_18550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 4), config_18549, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 13)
    add_data_dir_call_result_18553 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), add_data_dir_18550, *[str_18551], **kwargs_18552)
    
    
    # Assigning a List to a Name (line 15):
    
    # Obtaining an instance of the builtin type 'list' (line 15)
    list_18554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 15)
    # Adding element type (line 15)
    
    # Call to join(...): (line 15)
    # Processing the call arguments (line 15)
    str_18556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 25), 'str', 'src/dfftpack')
    str_18557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 40), 'str', '*.f')
    # Processing the call keyword arguments (line 15)
    kwargs_18558 = {}
    # Getting the type of 'join' (line 15)
    join_18555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 20), 'join', False)
    # Calling join(args, kwargs) (line 15)
    join_call_result_18559 = invoke(stypy.reporting.localization.Localization(__file__, 15, 20), join_18555, *[str_18556, str_18557], **kwargs_18558)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 19), list_18554, join_call_result_18559)
    
    # Assigning a type to the variable 'dfftpack_src' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'dfftpack_src', list_18554)
    
    # Call to add_library(...): (line 16)
    # Processing the call arguments (line 16)
    str_18562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 23), 'str', 'dfftpack')
    # Processing the call keyword arguments (line 16)
    # Getting the type of 'dfftpack_src' (line 16)
    dfftpack_src_18563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 43), 'dfftpack_src', False)
    keyword_18564 = dfftpack_src_18563
    kwargs_18565 = {'sources': keyword_18564}
    # Getting the type of 'config' (line 16)
    config_18560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'config', False)
    # Obtaining the member 'add_library' of a type (line 16)
    add_library_18561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), config_18560, 'add_library')
    # Calling add_library(args, kwargs) (line 16)
    add_library_call_result_18566 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), add_library_18561, *[str_18562], **kwargs_18565)
    
    
    # Assigning a List to a Name (line 18):
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_18567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    # Adding element type (line 18)
    
    # Call to join(...): (line 18)
    # Processing the call arguments (line 18)
    str_18569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 24), 'str', 'src/fftpack')
    str_18570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 38), 'str', '*.f')
    # Processing the call keyword arguments (line 18)
    kwargs_18571 = {}
    # Getting the type of 'join' (line 18)
    join_18568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 19), 'join', False)
    # Calling join(args, kwargs) (line 18)
    join_call_result_18572 = invoke(stypy.reporting.localization.Localization(__file__, 18, 19), join_18568, *[str_18569, str_18570], **kwargs_18571)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 18), list_18567, join_call_result_18572)
    
    # Assigning a type to the variable 'fftpack_src' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'fftpack_src', list_18567)
    
    # Call to add_library(...): (line 19)
    # Processing the call arguments (line 19)
    str_18575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 23), 'str', 'fftpack')
    # Processing the call keyword arguments (line 19)
    # Getting the type of 'fftpack_src' (line 19)
    fftpack_src_18576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 42), 'fftpack_src', False)
    keyword_18577 = fftpack_src_18576
    kwargs_18578 = {'sources': keyword_18577}
    # Getting the type of 'config' (line 19)
    config_18573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'config', False)
    # Obtaining the member 'add_library' of a type (line 19)
    add_library_18574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 4), config_18573, 'add_library')
    # Calling add_library(args, kwargs) (line 19)
    add_library_call_result_18579 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), add_library_18574, *[str_18575], **kwargs_18578)
    
    
    # Assigning a List to a Name (line 21):
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_18580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    str_18581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 15), 'str', 'fftpack.pyf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 14), list_18580, str_18581)
    # Adding element type (line 21)
    str_18582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 29), 'str', 'src/zfft.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 14), list_18580, str_18582)
    # Adding element type (line 21)
    str_18583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 42), 'str', 'src/drfft.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 14), list_18580, str_18583)
    # Adding element type (line 21)
    str_18584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 56), 'str', 'src/zrfft.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 14), list_18580, str_18584)
    # Adding element type (line 21)
    str_18585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 15), 'str', 'src/zfftnd.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 14), list_18580, str_18585)
    # Adding element type (line 21)
    str_18586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 31), 'str', 'src/dct.c.src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 14), list_18580, str_18586)
    # Adding element type (line 21)
    str_18587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 48), 'str', 'src/dst.c.src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 14), list_18580, str_18587)
    
    # Assigning a type to the variable 'sources' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'sources', list_18580)
    
    # Call to add_extension(...): (line 24)
    # Processing the call arguments (line 24)
    str_18590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'str', '_fftpack')
    # Processing the call keyword arguments (line 24)
    # Getting the type of 'sources' (line 25)
    sources_18591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'sources', False)
    keyword_18592 = sources_18591
    
    # Obtaining an instance of the builtin type 'list' (line 26)
    list_18593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 26)
    # Adding element type (line 26)
    str_18594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'str', 'dfftpack')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 18), list_18593, str_18594)
    # Adding element type (line 26)
    str_18595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 31), 'str', 'fftpack')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 18), list_18593, str_18595)
    
    keyword_18596 = list_18593
    
    # Obtaining an instance of the builtin type 'list' (line 27)
    list_18597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 27)
    # Adding element type (line 27)
    str_18598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 22), 'str', 'src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 21), list_18597, str_18598)
    
    keyword_18599 = list_18597
    # Getting the type of 'dfftpack_src' (line 28)
    dfftpack_src_18600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 17), 'dfftpack_src', False)
    # Getting the type of 'fftpack_src' (line 28)
    fftpack_src_18601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 32), 'fftpack_src', False)
    # Applying the binary operator '+' (line 28)
    result_add_18602 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 17), '+', dfftpack_src_18600, fftpack_src_18601)
    
    keyword_18603 = result_add_18602
    kwargs_18604 = {'libraries': keyword_18596, 'sources': keyword_18592, 'depends': keyword_18603, 'include_dirs': keyword_18599}
    # Getting the type of 'config' (line 24)
    config_18588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 24)
    add_extension_18589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 4), config_18588, 'add_extension')
    # Calling add_extension(args, kwargs) (line 24)
    add_extension_call_result_18605 = invoke(stypy.reporting.localization.Localization(__file__, 24, 4), add_extension_18589, *[str_18590], **kwargs_18604)
    
    
    # Call to add_extension(...): (line 30)
    # Processing the call arguments (line 30)
    str_18608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 25), 'str', 'convolve')
    # Processing the call keyword arguments (line 30)
    
    # Obtaining an instance of the builtin type 'list' (line 31)
    list_18609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 31)
    # Adding element type (line 31)
    str_18610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 17), 'str', 'convolve.pyf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 16), list_18609, str_18610)
    # Adding element type (line 31)
    str_18611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 32), 'str', 'src/convolve.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 16), list_18609, str_18611)
    
    keyword_18612 = list_18609
    
    # Obtaining an instance of the builtin type 'list' (line 32)
    list_18613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 32)
    # Adding element type (line 32)
    str_18614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 19), 'str', 'dfftpack')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 18), list_18613, str_18614)
    
    keyword_18615 = list_18613
    # Getting the type of 'dfftpack_src' (line 33)
    dfftpack_src_18616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), 'dfftpack_src', False)
    keyword_18617 = dfftpack_src_18616
    kwargs_18618 = {'libraries': keyword_18615, 'sources': keyword_18612, 'depends': keyword_18617}
    # Getting the type of 'config' (line 30)
    config_18606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 30)
    add_extension_18607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 4), config_18606, 'add_extension')
    # Calling add_extension(args, kwargs) (line 30)
    add_extension_call_result_18619 = invoke(stypy.reporting.localization.Localization(__file__, 30, 4), add_extension_18607, *[str_18608], **kwargs_18618)
    
    # Getting the type of 'config' (line 35)
    config_18620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type', config_18620)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_18621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18621)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_18621

# Assigning a type to the variable 'configuration' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 39, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 39)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/')
    import_18622 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 39, 4), 'numpy.distutils.core')

    if (type(import_18622) is not StypyTypeError):

        if (import_18622 != 'pyd_module'):
            __import__(import_18622)
            sys_modules_18623 = sys.modules[import_18622]
            import_from_module(stypy.reporting.localization.Localization(__file__, 39, 4), 'numpy.distutils.core', sys_modules_18623.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 39, 4), __file__, sys_modules_18623, sys_modules_18623.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 39, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'numpy.distutils.core', import_18622)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/')
    
    
    # Call to setup(...): (line 40)
    # Processing the call keyword arguments (line 40)
    
    # Call to todict(...): (line 40)
    # Processing the call keyword arguments (line 40)
    kwargs_18631 = {}
    
    # Call to configuration(...): (line 40)
    # Processing the call keyword arguments (line 40)
    str_18626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 35), 'str', '')
    keyword_18627 = str_18626
    kwargs_18628 = {'top_path': keyword_18627}
    # Getting the type of 'configuration' (line 40)
    configuration_18625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 40)
    configuration_call_result_18629 = invoke(stypy.reporting.localization.Localization(__file__, 40, 12), configuration_18625, *[], **kwargs_18628)
    
    # Obtaining the member 'todict' of a type (line 40)
    todict_18630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 12), configuration_call_result_18629, 'todict')
    # Calling todict(args, kwargs) (line 40)
    todict_call_result_18632 = invoke(stypy.reporting.localization.Localization(__file__, 40, 12), todict_18630, *[], **kwargs_18631)
    
    kwargs_18633 = {'todict_call_result_18632': todict_call_result_18632}
    # Getting the type of 'setup' (line 40)
    setup_18624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 40)
    setup_call_result_18634 = invoke(stypy.reporting.localization.Localization(__file__, 40, 4), setup_18624, *[], **kwargs_18633)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
