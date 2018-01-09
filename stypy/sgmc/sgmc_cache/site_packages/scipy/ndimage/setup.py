
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import os
4: 
5: from numpy.distutils.core import setup
6: from numpy.distutils.misc_util import Configuration
7: from numpy import get_include
8: from scipy._build_utils import numpy_nodepr_api
9: 
10: 
11: def configuration(parent_package='', top_path=None):
12: 
13:     config = Configuration('ndimage', parent_package, top_path)
14: 
15:     include_dirs = ['src',
16:                     get_include(),
17:                     os.path.join(os.path.dirname(__file__), '..', '_lib', 'src')]
18: 
19:     config.add_extension("_nd_image",
20:         sources=["src/nd_image.c","src/ni_filters.c",
21:                  "src/ni_fourier.c","src/ni_interpolation.c",
22:                  "src/ni_measure.c",
23:                  "src/ni_morphology.c","src/ni_support.c"],
24:         include_dirs=include_dirs,
25:         **numpy_nodepr_api)
26: 
27:     # Cython wants the .c and .pyx to have the underscore.
28:     config.add_extension("_ni_label",
29:                          sources=["src/_ni_label.c",],
30:                          include_dirs=['src']+[get_include()])
31: 
32:     config.add_extension("_ctest",
33:                          sources=["src/_ctest.c"],
34:                          include_dirs=[get_include()],
35:                          **numpy_nodepr_api)
36: 
37:     _define_macros = [("OLDAPI", 1)]
38:     if 'define_macros' in numpy_nodepr_api:
39:         _define_macros.extend(numpy_nodepr_api['define_macros'])
40: 
41:     config.add_extension("_ctest_oldapi",
42:                          sources=["src/_ctest.c"],
43:                          include_dirs=[get_include()],
44:                          define_macros=_define_macros)
45: 
46:     config.add_extension("_cytest",
47:                          sources=["src/_cytest.c"])
48: 
49:     config.add_data_dir('tests')
50: 
51:     return config
52: 
53: if __name__ == '__main__':
54:     setup(**configuration(top_path='').todict())
55: 

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

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy.distutils.core import setup' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_126497 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.distutils.core')

if (type(import_126497) is not StypyTypeError):

    if (import_126497 != 'pyd_module'):
        __import__(import_126497)
        sys_modules_126498 = sys.modules[import_126497]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.distutils.core', sys_modules_126498.module_type_store, module_type_store, ['setup'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_126498, sys_modules_126498.module_type_store, module_type_store)
    else:
        from numpy.distutils.core import setup

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

else:
    # Assigning a type to the variable 'numpy.distutils.core' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.distutils.core', import_126497)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.distutils.misc_util import Configuration' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_126499 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.distutils.misc_util')

if (type(import_126499) is not StypyTypeError):

    if (import_126499 != 'pyd_module'):
        __import__(import_126499)
        sys_modules_126500 = sys.modules[import_126499]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.distutils.misc_util', sys_modules_126500.module_type_store, module_type_store, ['Configuration'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_126500, sys_modules_126500.module_type_store, module_type_store)
    else:
        from numpy.distutils.misc_util import Configuration

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

else:
    # Assigning a type to the variable 'numpy.distutils.misc_util' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.distutils.misc_util', import_126499)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy import get_include' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_126501 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy')

if (type(import_126501) is not StypyTypeError):

    if (import_126501 != 'pyd_module'):
        __import__(import_126501)
        sys_modules_126502 = sys.modules[import_126501]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', sys_modules_126502.module_type_store, module_type_store, ['get_include'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_126502, sys_modules_126502.module_type_store, module_type_store)
    else:
        from numpy import get_include

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', None, module_type_store, ['get_include'], [get_include])

else:
    # Assigning a type to the variable 'numpy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', import_126501)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy._build_utils import numpy_nodepr_api' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_126503 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._build_utils')

if (type(import_126503) is not StypyTypeError):

    if (import_126503 != 'pyd_module'):
        __import__(import_126503)
        sys_modules_126504 = sys.modules[import_126503]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._build_utils', sys_modules_126504.module_type_store, module_type_store, ['numpy_nodepr_api'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_126504, sys_modules_126504.module_type_store, module_type_store)
    else:
        from scipy._build_utils import numpy_nodepr_api

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._build_utils', None, module_type_store, ['numpy_nodepr_api'], [numpy_nodepr_api])

else:
    # Assigning a type to the variable 'scipy._build_utils' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._build_utils', import_126503)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')


@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_126505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 33), 'str', '')
    # Getting the type of 'None' (line 11)
    None_126506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 46), 'None')
    defaults = [str_126505, None_126506]
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

    
    # Assigning a Call to a Name (line 13):
    
    # Call to Configuration(...): (line 13)
    # Processing the call arguments (line 13)
    str_126508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 27), 'str', 'ndimage')
    # Getting the type of 'parent_package' (line 13)
    parent_package_126509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 38), 'parent_package', False)
    # Getting the type of 'top_path' (line 13)
    top_path_126510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 54), 'top_path', False)
    # Processing the call keyword arguments (line 13)
    kwargs_126511 = {}
    # Getting the type of 'Configuration' (line 13)
    Configuration_126507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 13)
    Configuration_call_result_126512 = invoke(stypy.reporting.localization.Localization(__file__, 13, 13), Configuration_126507, *[str_126508, parent_package_126509, top_path_126510], **kwargs_126511)
    
    # Assigning a type to the variable 'config' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'config', Configuration_call_result_126512)
    
    # Assigning a List to a Name (line 15):
    
    # Obtaining an instance of the builtin type 'list' (line 15)
    list_126513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 15)
    # Adding element type (line 15)
    str_126514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 20), 'str', 'src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 19), list_126513, str_126514)
    # Adding element type (line 15)
    
    # Call to get_include(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_126516 = {}
    # Getting the type of 'get_include' (line 16)
    get_include_126515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 20), 'get_include', False)
    # Calling get_include(args, kwargs) (line 16)
    get_include_call_result_126517 = invoke(stypy.reporting.localization.Localization(__file__, 16, 20), get_include_126515, *[], **kwargs_126516)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 19), list_126513, get_include_call_result_126517)
    # Adding element type (line 15)
    
    # Call to join(...): (line 17)
    # Processing the call arguments (line 17)
    
    # Call to dirname(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of '__file__' (line 17)
    file___126524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 49), '__file__', False)
    # Processing the call keyword arguments (line 17)
    kwargs_126525 = {}
    # Getting the type of 'os' (line 17)
    os_126521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 33), 'os', False)
    # Obtaining the member 'path' of a type (line 17)
    path_126522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 33), os_126521, 'path')
    # Obtaining the member 'dirname' of a type (line 17)
    dirname_126523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 33), path_126522, 'dirname')
    # Calling dirname(args, kwargs) (line 17)
    dirname_call_result_126526 = invoke(stypy.reporting.localization.Localization(__file__, 17, 33), dirname_126523, *[file___126524], **kwargs_126525)
    
    str_126527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 60), 'str', '..')
    str_126528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 66), 'str', '_lib')
    str_126529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 74), 'str', 'src')
    # Processing the call keyword arguments (line 17)
    kwargs_126530 = {}
    # Getting the type of 'os' (line 17)
    os_126518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 20), 'os', False)
    # Obtaining the member 'path' of a type (line 17)
    path_126519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 20), os_126518, 'path')
    # Obtaining the member 'join' of a type (line 17)
    join_126520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 20), path_126519, 'join')
    # Calling join(args, kwargs) (line 17)
    join_call_result_126531 = invoke(stypy.reporting.localization.Localization(__file__, 17, 20), join_126520, *[dirname_call_result_126526, str_126527, str_126528, str_126529], **kwargs_126530)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 19), list_126513, join_call_result_126531)
    
    # Assigning a type to the variable 'include_dirs' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'include_dirs', list_126513)
    
    # Call to add_extension(...): (line 19)
    # Processing the call arguments (line 19)
    str_126534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 25), 'str', '_nd_image')
    # Processing the call keyword arguments (line 19)
    
    # Obtaining an instance of the builtin type 'list' (line 20)
    list_126535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 20)
    # Adding element type (line 20)
    str_126536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 17), 'str', 'src/nd_image.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 16), list_126535, str_126536)
    # Adding element type (line 20)
    str_126537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 34), 'str', 'src/ni_filters.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 16), list_126535, str_126537)
    # Adding element type (line 20)
    str_126538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 17), 'str', 'src/ni_fourier.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 16), list_126535, str_126538)
    # Adding element type (line 20)
    str_126539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 36), 'str', 'src/ni_interpolation.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 16), list_126535, str_126539)
    # Adding element type (line 20)
    str_126540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 17), 'str', 'src/ni_measure.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 16), list_126535, str_126540)
    # Adding element type (line 20)
    str_126541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 17), 'str', 'src/ni_morphology.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 16), list_126535, str_126541)
    # Adding element type (line 20)
    str_126542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 39), 'str', 'src/ni_support.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 16), list_126535, str_126542)
    
    keyword_126543 = list_126535
    # Getting the type of 'include_dirs' (line 24)
    include_dirs_126544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 21), 'include_dirs', False)
    keyword_126545 = include_dirs_126544
    # Getting the type of 'numpy_nodepr_api' (line 25)
    numpy_nodepr_api_126546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 10), 'numpy_nodepr_api', False)
    kwargs_126547 = {'sources': keyword_126543, 'numpy_nodepr_api_126546': numpy_nodepr_api_126546, 'include_dirs': keyword_126545}
    # Getting the type of 'config' (line 19)
    config_126532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 19)
    add_extension_126533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 4), config_126532, 'add_extension')
    # Calling add_extension(args, kwargs) (line 19)
    add_extension_call_result_126548 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), add_extension_126533, *[str_126534], **kwargs_126547)
    
    
    # Call to add_extension(...): (line 28)
    # Processing the call arguments (line 28)
    str_126551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 25), 'str', '_ni_label')
    # Processing the call keyword arguments (line 28)
    
    # Obtaining an instance of the builtin type 'list' (line 29)
    list_126552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 29)
    # Adding element type (line 29)
    str_126553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 34), 'str', 'src/_ni_label.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 33), list_126552, str_126553)
    
    keyword_126554 = list_126552
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_126555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    str_126556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 39), 'str', 'src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 38), list_126555, str_126556)
    
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_126557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 46), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    
    # Call to get_include(...): (line 30)
    # Processing the call keyword arguments (line 30)
    kwargs_126559 = {}
    # Getting the type of 'get_include' (line 30)
    get_include_126558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 47), 'get_include', False)
    # Calling get_include(args, kwargs) (line 30)
    get_include_call_result_126560 = invoke(stypy.reporting.localization.Localization(__file__, 30, 47), get_include_126558, *[], **kwargs_126559)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 46), list_126557, get_include_call_result_126560)
    
    # Applying the binary operator '+' (line 30)
    result_add_126561 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 38), '+', list_126555, list_126557)
    
    keyword_126562 = result_add_126561
    kwargs_126563 = {'sources': keyword_126554, 'include_dirs': keyword_126562}
    # Getting the type of 'config' (line 28)
    config_126549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 28)
    add_extension_126550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 4), config_126549, 'add_extension')
    # Calling add_extension(args, kwargs) (line 28)
    add_extension_call_result_126564 = invoke(stypy.reporting.localization.Localization(__file__, 28, 4), add_extension_126550, *[str_126551], **kwargs_126563)
    
    
    # Call to add_extension(...): (line 32)
    # Processing the call arguments (line 32)
    str_126567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 25), 'str', '_ctest')
    # Processing the call keyword arguments (line 32)
    
    # Obtaining an instance of the builtin type 'list' (line 33)
    list_126568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 33)
    # Adding element type (line 33)
    str_126569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 34), 'str', 'src/_ctest.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 33), list_126568, str_126569)
    
    keyword_126570 = list_126568
    
    # Obtaining an instance of the builtin type 'list' (line 34)
    list_126571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 34)
    # Adding element type (line 34)
    
    # Call to get_include(...): (line 34)
    # Processing the call keyword arguments (line 34)
    kwargs_126573 = {}
    # Getting the type of 'get_include' (line 34)
    get_include_126572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 39), 'get_include', False)
    # Calling get_include(args, kwargs) (line 34)
    get_include_call_result_126574 = invoke(stypy.reporting.localization.Localization(__file__, 34, 39), get_include_126572, *[], **kwargs_126573)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 38), list_126571, get_include_call_result_126574)
    
    keyword_126575 = list_126571
    # Getting the type of 'numpy_nodepr_api' (line 35)
    numpy_nodepr_api_126576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 27), 'numpy_nodepr_api', False)
    kwargs_126577 = {'sources': keyword_126570, 'numpy_nodepr_api_126576': numpy_nodepr_api_126576, 'include_dirs': keyword_126575}
    # Getting the type of 'config' (line 32)
    config_126565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 32)
    add_extension_126566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 4), config_126565, 'add_extension')
    # Calling add_extension(args, kwargs) (line 32)
    add_extension_call_result_126578 = invoke(stypy.reporting.localization.Localization(__file__, 32, 4), add_extension_126566, *[str_126567], **kwargs_126577)
    
    
    # Assigning a List to a Name (line 37):
    
    # Obtaining an instance of the builtin type 'list' (line 37)
    list_126579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 37)
    # Adding element type (line 37)
    
    # Obtaining an instance of the builtin type 'tuple' (line 37)
    tuple_126580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 37)
    # Adding element type (line 37)
    str_126581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 23), 'str', 'OLDAPI')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 23), tuple_126580, str_126581)
    # Adding element type (line 37)
    int_126582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 23), tuple_126580, int_126582)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 21), list_126579, tuple_126580)
    
    # Assigning a type to the variable '_define_macros' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), '_define_macros', list_126579)
    
    
    str_126583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 7), 'str', 'define_macros')
    # Getting the type of 'numpy_nodepr_api' (line 38)
    numpy_nodepr_api_126584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 26), 'numpy_nodepr_api')
    # Applying the binary operator 'in' (line 38)
    result_contains_126585 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 7), 'in', str_126583, numpy_nodepr_api_126584)
    
    # Testing the type of an if condition (line 38)
    if_condition_126586 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 4), result_contains_126585)
    # Assigning a type to the variable 'if_condition_126586' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'if_condition_126586', if_condition_126586)
    # SSA begins for if statement (line 38)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to extend(...): (line 39)
    # Processing the call arguments (line 39)
    
    # Obtaining the type of the subscript
    str_126589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 47), 'str', 'define_macros')
    # Getting the type of 'numpy_nodepr_api' (line 39)
    numpy_nodepr_api_126590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 30), 'numpy_nodepr_api', False)
    # Obtaining the member '__getitem__' of a type (line 39)
    getitem___126591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 30), numpy_nodepr_api_126590, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 39)
    subscript_call_result_126592 = invoke(stypy.reporting.localization.Localization(__file__, 39, 30), getitem___126591, str_126589)
    
    # Processing the call keyword arguments (line 39)
    kwargs_126593 = {}
    # Getting the type of '_define_macros' (line 39)
    _define_macros_126587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), '_define_macros', False)
    # Obtaining the member 'extend' of a type (line 39)
    extend_126588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), _define_macros_126587, 'extend')
    # Calling extend(args, kwargs) (line 39)
    extend_call_result_126594 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), extend_126588, *[subscript_call_result_126592], **kwargs_126593)
    
    # SSA join for if statement (line 38)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to add_extension(...): (line 41)
    # Processing the call arguments (line 41)
    str_126597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 25), 'str', '_ctest_oldapi')
    # Processing the call keyword arguments (line 41)
    
    # Obtaining an instance of the builtin type 'list' (line 42)
    list_126598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 42)
    # Adding element type (line 42)
    str_126599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 34), 'str', 'src/_ctest.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 33), list_126598, str_126599)
    
    keyword_126600 = list_126598
    
    # Obtaining an instance of the builtin type 'list' (line 43)
    list_126601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 43)
    # Adding element type (line 43)
    
    # Call to get_include(...): (line 43)
    # Processing the call keyword arguments (line 43)
    kwargs_126603 = {}
    # Getting the type of 'get_include' (line 43)
    get_include_126602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 39), 'get_include', False)
    # Calling get_include(args, kwargs) (line 43)
    get_include_call_result_126604 = invoke(stypy.reporting.localization.Localization(__file__, 43, 39), get_include_126602, *[], **kwargs_126603)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 38), list_126601, get_include_call_result_126604)
    
    keyword_126605 = list_126601
    # Getting the type of '_define_macros' (line 44)
    _define_macros_126606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 39), '_define_macros', False)
    keyword_126607 = _define_macros_126606
    kwargs_126608 = {'sources': keyword_126600, 'define_macros': keyword_126607, 'include_dirs': keyword_126605}
    # Getting the type of 'config' (line 41)
    config_126595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 41)
    add_extension_126596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 4), config_126595, 'add_extension')
    # Calling add_extension(args, kwargs) (line 41)
    add_extension_call_result_126609 = invoke(stypy.reporting.localization.Localization(__file__, 41, 4), add_extension_126596, *[str_126597], **kwargs_126608)
    
    
    # Call to add_extension(...): (line 46)
    # Processing the call arguments (line 46)
    str_126612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 25), 'str', '_cytest')
    # Processing the call keyword arguments (line 46)
    
    # Obtaining an instance of the builtin type 'list' (line 47)
    list_126613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 47)
    # Adding element type (line 47)
    str_126614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 34), 'str', 'src/_cytest.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 33), list_126613, str_126614)
    
    keyword_126615 = list_126613
    kwargs_126616 = {'sources': keyword_126615}
    # Getting the type of 'config' (line 46)
    config_126610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 46)
    add_extension_126611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 4), config_126610, 'add_extension')
    # Calling add_extension(args, kwargs) (line 46)
    add_extension_call_result_126617 = invoke(stypy.reporting.localization.Localization(__file__, 46, 4), add_extension_126611, *[str_126612], **kwargs_126616)
    
    
    # Call to add_data_dir(...): (line 49)
    # Processing the call arguments (line 49)
    str_126620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 49)
    kwargs_126621 = {}
    # Getting the type of 'config' (line 49)
    config_126618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 49)
    add_data_dir_126619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 4), config_126618, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 49)
    add_data_dir_call_result_126622 = invoke(stypy.reporting.localization.Localization(__file__, 49, 4), add_data_dir_126619, *[str_126620], **kwargs_126621)
    
    # Getting the type of 'config' (line 51)
    config_126623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'stypy_return_type', config_126623)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 11)
    stypy_return_type_126624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126624)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_126624

# Assigning a type to the variable 'configuration' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    
    # Call to setup(...): (line 54)
    # Processing the call keyword arguments (line 54)
    
    # Call to todict(...): (line 54)
    # Processing the call keyword arguments (line 54)
    kwargs_126632 = {}
    
    # Call to configuration(...): (line 54)
    # Processing the call keyword arguments (line 54)
    str_126627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 35), 'str', '')
    keyword_126628 = str_126627
    kwargs_126629 = {'top_path': keyword_126628}
    # Getting the type of 'configuration' (line 54)
    configuration_126626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 54)
    configuration_call_result_126630 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), configuration_126626, *[], **kwargs_126629)
    
    # Obtaining the member 'todict' of a type (line 54)
    todict_126631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), configuration_call_result_126630, 'todict')
    # Calling todict(args, kwargs) (line 54)
    todict_call_result_126633 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), todict_126631, *[], **kwargs_126632)
    
    kwargs_126634 = {'todict_call_result_126633': todict_call_result_126633}
    # Getting the type of 'setup' (line 54)
    setup_126625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 54)
    setup_call_result_126635 = invoke(stypy.reporting.localization.Localization(__file__, 54, 4), setup_126625, *[], **kwargs_126634)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
