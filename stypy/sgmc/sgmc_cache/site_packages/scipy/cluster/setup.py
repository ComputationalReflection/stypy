
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import sys
4: 
5: if sys.version_info[0] >= 3:
6:     DEFINE_MACROS = [("SCIPY_PY3K", None)]
7: else:
8:     DEFINE_MACROS = []
9: 
10: 
11: def configuration(parent_package='', top_path=None):
12:     from numpy.distutils.system_info import get_info
13:     from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
14:     config = Configuration('cluster', parent_package, top_path)
15: 
16:     blas_opt = get_info('lapack_opt')
17: 
18:     config.add_data_dir('tests')
19: 
20:     config.add_extension('_vq',
21:         sources=[('_vq.c')],
22:         include_dirs=[get_numpy_include_dirs()],
23:         extra_info=blas_opt)
24: 
25:     config.add_extension('_hierarchy',
26:         sources=[('_hierarchy.c')],
27:         include_dirs=[get_numpy_include_dirs()])
28: 
29:     config.add_extension('_optimal_leaf_ordering',
30:         sources=[('_optimal_leaf_ordering.c')],
31:         include_dirs=[get_numpy_include_dirs()])
32: 
33:     return config
34: 
35: 
36: if __name__ == '__main__':
37:     from numpy.distutils.core import setup
38:     setup(**configuration(top_path='').todict())
39: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import sys' statement (line 3)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'sys', sys, module_type_store)




# Obtaining the type of the subscript
int_5646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 20), 'int')
# Getting the type of 'sys' (line 5)
sys_5647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 5)
version_info_5648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 3), sys_5647, 'version_info')
# Obtaining the member '__getitem__' of a type (line 5)
getitem___5649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 3), version_info_5648, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 5)
subscript_call_result_5650 = invoke(stypy.reporting.localization.Localization(__file__, 5, 3), getitem___5649, int_5646)

int_5651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 26), 'int')
# Applying the binary operator '>=' (line 5)
result_ge_5652 = python_operator(stypy.reporting.localization.Localization(__file__, 5, 3), '>=', subscript_call_result_5650, int_5651)

# Testing the type of an if condition (line 5)
if_condition_5653 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 5, 0), result_ge_5652)
# Assigning a type to the variable 'if_condition_5653' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'if_condition_5653', if_condition_5653)
# SSA begins for if statement (line 5)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a List to a Name (line 6):

# Obtaining an instance of the builtin type 'list' (line 6)
list_5654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)

# Obtaining an instance of the builtin type 'tuple' (line 6)
tuple_5655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6)
# Adding element type (line 6)
str_5656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 22), 'str', 'SCIPY_PY3K')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 22), tuple_5655, str_5656)
# Adding element type (line 6)
# Getting the type of 'None' (line 6)
None_5657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 36), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 22), tuple_5655, None_5657)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 20), list_5654, tuple_5655)

# Assigning a type to the variable 'DEFINE_MACROS' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'DEFINE_MACROS', list_5654)
# SSA branch for the else part of an if statement (line 5)
module_type_store.open_ssa_branch('else')

# Assigning a List to a Name (line 8):

# Obtaining an instance of the builtin type 'list' (line 8)
list_5658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)

# Assigning a type to the variable 'DEFINE_MACROS' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'DEFINE_MACROS', list_5658)
# SSA join for if statement (line 5)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_5659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 33), 'str', '')
    # Getting the type of 'None' (line 11)
    None_5660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 46), 'None')
    defaults = [str_5659, None_5660]
    # Create a new context for function 'configuration'
    module_type_store = module_type_store.open_function_context('configuration', 11, 0, False)
    
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

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 4))
    
    # 'from numpy.distutils.system_info import get_info' statement (line 12)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/cluster/')
    import_5661 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 4), 'numpy.distutils.system_info')

    if (type(import_5661) is not StypyTypeError):

        if (import_5661 != 'pyd_module'):
            __import__(import_5661)
            sys_modules_5662 = sys.modules[import_5661]
            import_from_module(stypy.reporting.localization.Localization(__file__, 12, 4), 'numpy.distutils.system_info', sys_modules_5662.module_type_store, module_type_store, ['get_info'])
            nest_module(stypy.reporting.localization.Localization(__file__, 12, 4), __file__, sys_modules_5662, sys_modules_5662.module_type_store, module_type_store)
        else:
            from numpy.distutils.system_info import get_info

            import_from_module(stypy.reporting.localization.Localization(__file__, 12, 4), 'numpy.distutils.system_info', None, module_type_store, ['get_info'], [get_info])

    else:
        # Assigning a type to the variable 'numpy.distutils.system_info' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'numpy.distutils.system_info', import_5661)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/cluster/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 4))
    
    # 'from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs' statement (line 13)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/cluster/')
    import_5663 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy.distutils.misc_util')

    if (type(import_5663) is not StypyTypeError):

        if (import_5663 != 'pyd_module'):
            __import__(import_5663)
            sys_modules_5664 = sys.modules[import_5663]
            import_from_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy.distutils.misc_util', sys_modules_5664.module_type_store, module_type_store, ['Configuration', 'get_numpy_include_dirs'])
            nest_module(stypy.reporting.localization.Localization(__file__, 13, 4), __file__, sys_modules_5664, sys_modules_5664.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

            import_from_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration', 'get_numpy_include_dirs'], [Configuration, get_numpy_include_dirs])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy.distutils.misc_util', import_5663)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/cluster/')
    
    
    # Assigning a Call to a Name (line 14):
    
    # Call to Configuration(...): (line 14)
    # Processing the call arguments (line 14)
    str_5666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 27), 'str', 'cluster')
    # Getting the type of 'parent_package' (line 14)
    parent_package_5667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 38), 'parent_package', False)
    # Getting the type of 'top_path' (line 14)
    top_path_5668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 54), 'top_path', False)
    # Processing the call keyword arguments (line 14)
    kwargs_5669 = {}
    # Getting the type of 'Configuration' (line 14)
    Configuration_5665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 14)
    Configuration_call_result_5670 = invoke(stypy.reporting.localization.Localization(__file__, 14, 13), Configuration_5665, *[str_5666, parent_package_5667, top_path_5668], **kwargs_5669)
    
    # Assigning a type to the variable 'config' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'config', Configuration_call_result_5670)
    
    # Assigning a Call to a Name (line 16):
    
    # Call to get_info(...): (line 16)
    # Processing the call arguments (line 16)
    str_5672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 24), 'str', 'lapack_opt')
    # Processing the call keyword arguments (line 16)
    kwargs_5673 = {}
    # Getting the type of 'get_info' (line 16)
    get_info_5671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 15), 'get_info', False)
    # Calling get_info(args, kwargs) (line 16)
    get_info_call_result_5674 = invoke(stypy.reporting.localization.Localization(__file__, 16, 15), get_info_5671, *[str_5672], **kwargs_5673)
    
    # Assigning a type to the variable 'blas_opt' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'blas_opt', get_info_call_result_5674)
    
    # Call to add_data_dir(...): (line 18)
    # Processing the call arguments (line 18)
    str_5677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 18)
    kwargs_5678 = {}
    # Getting the type of 'config' (line 18)
    config_5675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 18)
    add_data_dir_5676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 4), config_5675, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 18)
    add_data_dir_call_result_5679 = invoke(stypy.reporting.localization.Localization(__file__, 18, 4), add_data_dir_5676, *[str_5677], **kwargs_5678)
    
    
    # Call to add_extension(...): (line 20)
    # Processing the call arguments (line 20)
    str_5682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 25), 'str', '_vq')
    # Processing the call keyword arguments (line 20)
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_5683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    str_5684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 18), 'str', '_vq.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 16), list_5683, str_5684)
    
    keyword_5685 = list_5683
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_5686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    
    # Call to get_numpy_include_dirs(...): (line 22)
    # Processing the call keyword arguments (line 22)
    kwargs_5688 = {}
    # Getting the type of 'get_numpy_include_dirs' (line 22)
    get_numpy_include_dirs_5687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 22), 'get_numpy_include_dirs', False)
    # Calling get_numpy_include_dirs(args, kwargs) (line 22)
    get_numpy_include_dirs_call_result_5689 = invoke(stypy.reporting.localization.Localization(__file__, 22, 22), get_numpy_include_dirs_5687, *[], **kwargs_5688)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 21), list_5686, get_numpy_include_dirs_call_result_5689)
    
    keyword_5690 = list_5686
    # Getting the type of 'blas_opt' (line 23)
    blas_opt_5691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 19), 'blas_opt', False)
    keyword_5692 = blas_opt_5691
    kwargs_5693 = {'sources': keyword_5685, 'extra_info': keyword_5692, 'include_dirs': keyword_5690}
    # Getting the type of 'config' (line 20)
    config_5680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 20)
    add_extension_5681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 4), config_5680, 'add_extension')
    # Calling add_extension(args, kwargs) (line 20)
    add_extension_call_result_5694 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), add_extension_5681, *[str_5682], **kwargs_5693)
    
    
    # Call to add_extension(...): (line 25)
    # Processing the call arguments (line 25)
    str_5697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 25), 'str', '_hierarchy')
    # Processing the call keyword arguments (line 25)
    
    # Obtaining an instance of the builtin type 'list' (line 26)
    list_5698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 26)
    # Adding element type (line 26)
    str_5699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 18), 'str', '_hierarchy.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 16), list_5698, str_5699)
    
    keyword_5700 = list_5698
    
    # Obtaining an instance of the builtin type 'list' (line 27)
    list_5701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 27)
    # Adding element type (line 27)
    
    # Call to get_numpy_include_dirs(...): (line 27)
    # Processing the call keyword arguments (line 27)
    kwargs_5703 = {}
    # Getting the type of 'get_numpy_include_dirs' (line 27)
    get_numpy_include_dirs_5702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 22), 'get_numpy_include_dirs', False)
    # Calling get_numpy_include_dirs(args, kwargs) (line 27)
    get_numpy_include_dirs_call_result_5704 = invoke(stypy.reporting.localization.Localization(__file__, 27, 22), get_numpy_include_dirs_5702, *[], **kwargs_5703)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 21), list_5701, get_numpy_include_dirs_call_result_5704)
    
    keyword_5705 = list_5701
    kwargs_5706 = {'sources': keyword_5700, 'include_dirs': keyword_5705}
    # Getting the type of 'config' (line 25)
    config_5695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 25)
    add_extension_5696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 4), config_5695, 'add_extension')
    # Calling add_extension(args, kwargs) (line 25)
    add_extension_call_result_5707 = invoke(stypy.reporting.localization.Localization(__file__, 25, 4), add_extension_5696, *[str_5697], **kwargs_5706)
    
    
    # Call to add_extension(...): (line 29)
    # Processing the call arguments (line 29)
    str_5710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 25), 'str', '_optimal_leaf_ordering')
    # Processing the call keyword arguments (line 29)
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_5711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    str_5712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 18), 'str', '_optimal_leaf_ordering.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 16), list_5711, str_5712)
    
    keyword_5713 = list_5711
    
    # Obtaining an instance of the builtin type 'list' (line 31)
    list_5714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 31)
    # Adding element type (line 31)
    
    # Call to get_numpy_include_dirs(...): (line 31)
    # Processing the call keyword arguments (line 31)
    kwargs_5716 = {}
    # Getting the type of 'get_numpy_include_dirs' (line 31)
    get_numpy_include_dirs_5715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 22), 'get_numpy_include_dirs', False)
    # Calling get_numpy_include_dirs(args, kwargs) (line 31)
    get_numpy_include_dirs_call_result_5717 = invoke(stypy.reporting.localization.Localization(__file__, 31, 22), get_numpy_include_dirs_5715, *[], **kwargs_5716)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 21), list_5714, get_numpy_include_dirs_call_result_5717)
    
    keyword_5718 = list_5714
    kwargs_5719 = {'sources': keyword_5713, 'include_dirs': keyword_5718}
    # Getting the type of 'config' (line 29)
    config_5708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 29)
    add_extension_5709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 4), config_5708, 'add_extension')
    # Calling add_extension(args, kwargs) (line 29)
    add_extension_call_result_5720 = invoke(stypy.reporting.localization.Localization(__file__, 29, 4), add_extension_5709, *[str_5710], **kwargs_5719)
    
    # Getting the type of 'config' (line 33)
    config_5721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type', config_5721)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 11)
    stypy_return_type_5722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5722)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_5722

# Assigning a type to the variable 'configuration' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 37, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 37)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/cluster/')
    import_5723 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 37, 4), 'numpy.distutils.core')

    if (type(import_5723) is not StypyTypeError):

        if (import_5723 != 'pyd_module'):
            __import__(import_5723)
            sys_modules_5724 = sys.modules[import_5723]
            import_from_module(stypy.reporting.localization.Localization(__file__, 37, 4), 'numpy.distutils.core', sys_modules_5724.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 37, 4), __file__, sys_modules_5724, sys_modules_5724.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 37, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'numpy.distutils.core', import_5723)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/cluster/')
    
    
    # Call to setup(...): (line 38)
    # Processing the call keyword arguments (line 38)
    
    # Call to todict(...): (line 38)
    # Processing the call keyword arguments (line 38)
    kwargs_5732 = {}
    
    # Call to configuration(...): (line 38)
    # Processing the call keyword arguments (line 38)
    str_5727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 35), 'str', '')
    keyword_5728 = str_5727
    kwargs_5729 = {'top_path': keyword_5728}
    # Getting the type of 'configuration' (line 38)
    configuration_5726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 38)
    configuration_call_result_5730 = invoke(stypy.reporting.localization.Localization(__file__, 38, 12), configuration_5726, *[], **kwargs_5729)
    
    # Obtaining the member 'todict' of a type (line 38)
    todict_5731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 12), configuration_call_result_5730, 'todict')
    # Calling todict(args, kwargs) (line 38)
    todict_call_result_5733 = invoke(stypy.reporting.localization.Localization(__file__, 38, 12), todict_5731, *[], **kwargs_5732)
    
    kwargs_5734 = {'todict_call_result_5733': todict_call_result_5733}
    # Getting the type of 'setup' (line 38)
    setup_5725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 38)
    setup_call_result_5735 = invoke(stypy.reporting.localization.Localization(__file__, 38, 4), setup_5725, *[], **kwargs_5734)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
