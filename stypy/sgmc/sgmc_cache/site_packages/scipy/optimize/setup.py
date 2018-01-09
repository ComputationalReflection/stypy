
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from os.path import join
4: 
5: from scipy._build_utils import numpy_nodepr_api
6: 
7: def configuration(parent_package='',top_path=None):
8:     from numpy.distutils.misc_util import Configuration
9:     from numpy.distutils.system_info import get_info
10:     config = Configuration('optimize',parent_package, top_path)
11: 
12:     minpack_src = [join('minpack','*f')]
13:     config.add_library('minpack',sources=minpack_src)
14:     config.add_extension('_minpack',
15:                          sources=['_minpackmodule.c'],
16:                          libraries=['minpack'],
17:                          depends=(["minpack.h","__minpack.h"]
18:                                   + minpack_src),
19:                          **numpy_nodepr_api)
20: 
21:     rootfind_src = [join('Zeros','*.c')]
22:     rootfind_hdr = [join('Zeros','zeros.h')]
23:     config.add_library('rootfind',
24:                        sources=rootfind_src,
25:                        headers=rootfind_hdr,
26:                          **numpy_nodepr_api)
27: 
28:     config.add_extension('_zeros',
29:                          sources=['zeros.c'],
30:                          libraries=['rootfind'],
31:                          depends=(rootfind_src + rootfind_hdr),
32:                          **numpy_nodepr_api)
33: 
34:     lapack = get_info('lapack_opt')
35:     if 'define_macros' in numpy_nodepr_api:
36:         if ('define_macros' in lapack) and (lapack['define_macros'] is not None):
37:             lapack['define_macros'] = (lapack['define_macros'] +
38:                                        numpy_nodepr_api['define_macros'])
39:         else:
40:             lapack['define_macros'] = numpy_nodepr_api['define_macros']
41:     sources = ['lbfgsb.pyf', 'lbfgsb.f', 'linpack.f', 'timer.f']
42:     config.add_extension('_lbfgsb',
43:                          sources=[join('lbfgsb',x) for x in sources],
44:                          **lapack)
45: 
46:     sources = ['moduleTNC.c','tnc.c']
47:     config.add_extension('moduleTNC',
48:                          sources=[join('tnc',x) for x in sources],
49:                          depends=[join('tnc','tnc.h')],
50:                          **numpy_nodepr_api)
51: 
52:     config.add_extension('_cobyla',
53:                          sources=[join('cobyla',x) for x in ['cobyla.pyf',
54:                                                              'cobyla2.f',
55:                                                              'trstlp.f']],
56:                          **numpy_nodepr_api)
57: 
58:     sources = ['minpack2.pyf', 'dcsrch.f', 'dcstep.f']
59:     config.add_extension('minpack2',
60:                          sources=[join('minpack2',x) for x in sources],
61:                          **numpy_nodepr_api)
62: 
63:     sources = ['slsqp.pyf', 'slsqp_optmz.f']
64:     config.add_extension('_slsqp', sources=[join('slsqp', x) for x in sources],
65:                          **numpy_nodepr_api)
66: 
67:     config.add_extension('_nnls', sources=[join('nnls', x)
68:                                           for x in ["nnls.f","nnls.pyf"]],
69:                          **numpy_nodepr_api)
70: 
71:     config.add_extension('_group_columns', sources=['_group_columns.c'],)
72: 
73:     config.add_subpackage('_lsq')
74:     
75:     config.add_subpackage('_trlib')
76: 
77:     config.add_data_dir('tests')
78: 
79:     # Add license files
80:     config.add_data_files('lbfgsb/README')
81: 
82:     return config
83: 
84: 
85: if __name__ == '__main__':
86:     from numpy.distutils.core import setup
87:     setup(**configuration(top_path='').todict())
88: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from os.path import join' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_184259 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path')

if (type(import_184259) is not StypyTypeError):

    if (import_184259 != 'pyd_module'):
        __import__(import_184259)
        sys_modules_184260 = sys.modules[import_184259]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', sys_modules_184260.module_type_store, module_type_store, ['join'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_184260, sys_modules_184260.module_type_store, module_type_store)
    else:
        from os.path import join

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', None, module_type_store, ['join'], [join])

else:
    # Assigning a type to the variable 'os.path' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', import_184259)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy._build_utils import numpy_nodepr_api' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_184261 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._build_utils')

if (type(import_184261) is not StypyTypeError):

    if (import_184261 != 'pyd_module'):
        __import__(import_184261)
        sys_modules_184262 = sys.modules[import_184261]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._build_utils', sys_modules_184262.module_type_store, module_type_store, ['numpy_nodepr_api'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_184262, sys_modules_184262.module_type_store, module_type_store)
    else:
        from scipy._build_utils import numpy_nodepr_api

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._build_utils', None, module_type_store, ['numpy_nodepr_api'], [numpy_nodepr_api])

else:
    # Assigning a type to the variable 'scipy._build_utils' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._build_utils', import_184261)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')


@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_184263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 33), 'str', '')
    # Getting the type of 'None' (line 7)
    None_184264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 45), 'None')
    defaults = [str_184263, None_184264]
    # Create a new context for function 'configuration'
    module_type_store = module_type_store.open_function_context('configuration', 7, 0, False)
    
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

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 4))
    
    # 'from numpy.distutils.misc_util import Configuration' statement (line 8)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
    import_184265 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.misc_util')

    if (type(import_184265) is not StypyTypeError):

        if (import_184265 != 'pyd_module'):
            __import__(import_184265)
            sys_modules_184266 = sys.modules[import_184265]
            import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.misc_util', sys_modules_184266.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 8, 4), __file__, sys_modules_184266, sys_modules_184266.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.misc_util', import_184265)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 4))
    
    # 'from numpy.distutils.system_info import get_info' statement (line 9)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
    import_184267 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.system_info')

    if (type(import_184267) is not StypyTypeError):

        if (import_184267 != 'pyd_module'):
            __import__(import_184267)
            sys_modules_184268 = sys.modules[import_184267]
            import_from_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.system_info', sys_modules_184268.module_type_store, module_type_store, ['get_info'])
            nest_module(stypy.reporting.localization.Localization(__file__, 9, 4), __file__, sys_modules_184268, sys_modules_184268.module_type_store, module_type_store)
        else:
            from numpy.distutils.system_info import get_info

            import_from_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.system_info', None, module_type_store, ['get_info'], [get_info])

    else:
        # Assigning a type to the variable 'numpy.distutils.system_info' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.system_info', import_184267)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')
    
    
    # Assigning a Call to a Name (line 10):
    
    # Call to Configuration(...): (line 10)
    # Processing the call arguments (line 10)
    str_184270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 27), 'str', 'optimize')
    # Getting the type of 'parent_package' (line 10)
    parent_package_184271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 38), 'parent_package', False)
    # Getting the type of 'top_path' (line 10)
    top_path_184272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 54), 'top_path', False)
    # Processing the call keyword arguments (line 10)
    kwargs_184273 = {}
    # Getting the type of 'Configuration' (line 10)
    Configuration_184269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 10)
    Configuration_call_result_184274 = invoke(stypy.reporting.localization.Localization(__file__, 10, 13), Configuration_184269, *[str_184270, parent_package_184271, top_path_184272], **kwargs_184273)
    
    # Assigning a type to the variable 'config' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'config', Configuration_call_result_184274)
    
    # Assigning a List to a Name (line 12):
    
    # Obtaining an instance of the builtin type 'list' (line 12)
    list_184275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 12)
    # Adding element type (line 12)
    
    # Call to join(...): (line 12)
    # Processing the call arguments (line 12)
    str_184277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 24), 'str', 'minpack')
    str_184278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 34), 'str', '*f')
    # Processing the call keyword arguments (line 12)
    kwargs_184279 = {}
    # Getting the type of 'join' (line 12)
    join_184276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 19), 'join', False)
    # Calling join(args, kwargs) (line 12)
    join_call_result_184280 = invoke(stypy.reporting.localization.Localization(__file__, 12, 19), join_184276, *[str_184277, str_184278], **kwargs_184279)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 18), list_184275, join_call_result_184280)
    
    # Assigning a type to the variable 'minpack_src' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'minpack_src', list_184275)
    
    # Call to add_library(...): (line 13)
    # Processing the call arguments (line 13)
    str_184283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 23), 'str', 'minpack')
    # Processing the call keyword arguments (line 13)
    # Getting the type of 'minpack_src' (line 13)
    minpack_src_184284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 41), 'minpack_src', False)
    keyword_184285 = minpack_src_184284
    kwargs_184286 = {'sources': keyword_184285}
    # Getting the type of 'config' (line 13)
    config_184281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'config', False)
    # Obtaining the member 'add_library' of a type (line 13)
    add_library_184282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 4), config_184281, 'add_library')
    # Calling add_library(args, kwargs) (line 13)
    add_library_call_result_184287 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), add_library_184282, *[str_184283], **kwargs_184286)
    
    
    # Call to add_extension(...): (line 14)
    # Processing the call arguments (line 14)
    str_184290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 25), 'str', '_minpack')
    # Processing the call keyword arguments (line 14)
    
    # Obtaining an instance of the builtin type 'list' (line 15)
    list_184291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 15)
    # Adding element type (line 15)
    str_184292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 34), 'str', '_minpackmodule.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 33), list_184291, str_184292)
    
    keyword_184293 = list_184291
    
    # Obtaining an instance of the builtin type 'list' (line 16)
    list_184294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 16)
    # Adding element type (line 16)
    str_184295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 36), 'str', 'minpack')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 35), list_184294, str_184295)
    
    keyword_184296 = list_184294
    
    # Obtaining an instance of the builtin type 'list' (line 17)
    list_184297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 17)
    # Adding element type (line 17)
    str_184298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 35), 'str', 'minpack.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 34), list_184297, str_184298)
    # Adding element type (line 17)
    str_184299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 47), 'str', '__minpack.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 34), list_184297, str_184299)
    
    # Getting the type of 'minpack_src' (line 18)
    minpack_src_184300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 36), 'minpack_src', False)
    # Applying the binary operator '+' (line 17)
    result_add_184301 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 34), '+', list_184297, minpack_src_184300)
    
    keyword_184302 = result_add_184301
    # Getting the type of 'numpy_nodepr_api' (line 19)
    numpy_nodepr_api_184303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 27), 'numpy_nodepr_api', False)
    kwargs_184304 = {'libraries': keyword_184296, 'sources': keyword_184293, 'depends': keyword_184302, 'numpy_nodepr_api_184303': numpy_nodepr_api_184303}
    # Getting the type of 'config' (line 14)
    config_184288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 14)
    add_extension_184289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 4), config_184288, 'add_extension')
    # Calling add_extension(args, kwargs) (line 14)
    add_extension_call_result_184305 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), add_extension_184289, *[str_184290], **kwargs_184304)
    
    
    # Assigning a List to a Name (line 21):
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_184306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    
    # Call to join(...): (line 21)
    # Processing the call arguments (line 21)
    str_184308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 25), 'str', 'Zeros')
    str_184309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 33), 'str', '*.c')
    # Processing the call keyword arguments (line 21)
    kwargs_184310 = {}
    # Getting the type of 'join' (line 21)
    join_184307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 20), 'join', False)
    # Calling join(args, kwargs) (line 21)
    join_call_result_184311 = invoke(stypy.reporting.localization.Localization(__file__, 21, 20), join_184307, *[str_184308, str_184309], **kwargs_184310)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 19), list_184306, join_call_result_184311)
    
    # Assigning a type to the variable 'rootfind_src' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'rootfind_src', list_184306)
    
    # Assigning a List to a Name (line 22):
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_184312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    
    # Call to join(...): (line 22)
    # Processing the call arguments (line 22)
    str_184314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 25), 'str', 'Zeros')
    str_184315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 33), 'str', 'zeros.h')
    # Processing the call keyword arguments (line 22)
    kwargs_184316 = {}
    # Getting the type of 'join' (line 22)
    join_184313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 20), 'join', False)
    # Calling join(args, kwargs) (line 22)
    join_call_result_184317 = invoke(stypy.reporting.localization.Localization(__file__, 22, 20), join_184313, *[str_184314, str_184315], **kwargs_184316)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 19), list_184312, join_call_result_184317)
    
    # Assigning a type to the variable 'rootfind_hdr' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'rootfind_hdr', list_184312)
    
    # Call to add_library(...): (line 23)
    # Processing the call arguments (line 23)
    str_184320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 23), 'str', 'rootfind')
    # Processing the call keyword arguments (line 23)
    # Getting the type of 'rootfind_src' (line 24)
    rootfind_src_184321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 31), 'rootfind_src', False)
    keyword_184322 = rootfind_src_184321
    # Getting the type of 'rootfind_hdr' (line 25)
    rootfind_hdr_184323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 31), 'rootfind_hdr', False)
    keyword_184324 = rootfind_hdr_184323
    # Getting the type of 'numpy_nodepr_api' (line 26)
    numpy_nodepr_api_184325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 27), 'numpy_nodepr_api', False)
    kwargs_184326 = {'sources': keyword_184322, 'numpy_nodepr_api_184325': numpy_nodepr_api_184325, 'headers': keyword_184324}
    # Getting the type of 'config' (line 23)
    config_184318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'config', False)
    # Obtaining the member 'add_library' of a type (line 23)
    add_library_184319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 4), config_184318, 'add_library')
    # Calling add_library(args, kwargs) (line 23)
    add_library_call_result_184327 = invoke(stypy.reporting.localization.Localization(__file__, 23, 4), add_library_184319, *[str_184320], **kwargs_184326)
    
    
    # Call to add_extension(...): (line 28)
    # Processing the call arguments (line 28)
    str_184330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 25), 'str', '_zeros')
    # Processing the call keyword arguments (line 28)
    
    # Obtaining an instance of the builtin type 'list' (line 29)
    list_184331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 29)
    # Adding element type (line 29)
    str_184332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 34), 'str', 'zeros.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 33), list_184331, str_184332)
    
    keyword_184333 = list_184331
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_184334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    str_184335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 36), 'str', 'rootfind')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 35), list_184334, str_184335)
    
    keyword_184336 = list_184334
    # Getting the type of 'rootfind_src' (line 31)
    rootfind_src_184337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 34), 'rootfind_src', False)
    # Getting the type of 'rootfind_hdr' (line 31)
    rootfind_hdr_184338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 49), 'rootfind_hdr', False)
    # Applying the binary operator '+' (line 31)
    result_add_184339 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 34), '+', rootfind_src_184337, rootfind_hdr_184338)
    
    keyword_184340 = result_add_184339
    # Getting the type of 'numpy_nodepr_api' (line 32)
    numpy_nodepr_api_184341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 27), 'numpy_nodepr_api', False)
    kwargs_184342 = {'libraries': keyword_184336, 'sources': keyword_184333, 'depends': keyword_184340, 'numpy_nodepr_api_184341': numpy_nodepr_api_184341}
    # Getting the type of 'config' (line 28)
    config_184328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 28)
    add_extension_184329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 4), config_184328, 'add_extension')
    # Calling add_extension(args, kwargs) (line 28)
    add_extension_call_result_184343 = invoke(stypy.reporting.localization.Localization(__file__, 28, 4), add_extension_184329, *[str_184330], **kwargs_184342)
    
    
    # Assigning a Call to a Name (line 34):
    
    # Call to get_info(...): (line 34)
    # Processing the call arguments (line 34)
    str_184345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 22), 'str', 'lapack_opt')
    # Processing the call keyword arguments (line 34)
    kwargs_184346 = {}
    # Getting the type of 'get_info' (line 34)
    get_info_184344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 13), 'get_info', False)
    # Calling get_info(args, kwargs) (line 34)
    get_info_call_result_184347 = invoke(stypy.reporting.localization.Localization(__file__, 34, 13), get_info_184344, *[str_184345], **kwargs_184346)
    
    # Assigning a type to the variable 'lapack' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'lapack', get_info_call_result_184347)
    
    
    str_184348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 7), 'str', 'define_macros')
    # Getting the type of 'numpy_nodepr_api' (line 35)
    numpy_nodepr_api_184349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 26), 'numpy_nodepr_api')
    # Applying the binary operator 'in' (line 35)
    result_contains_184350 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 7), 'in', str_184348, numpy_nodepr_api_184349)
    
    # Testing the type of an if condition (line 35)
    if_condition_184351 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 4), result_contains_184350)
    # Assigning a type to the variable 'if_condition_184351' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'if_condition_184351', if_condition_184351)
    # SSA begins for if statement (line 35)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    str_184352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 12), 'str', 'define_macros')
    # Getting the type of 'lapack' (line 36)
    lapack_184353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), 'lapack')
    # Applying the binary operator 'in' (line 36)
    result_contains_184354 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 12), 'in', str_184352, lapack_184353)
    
    
    
    # Obtaining the type of the subscript
    str_184355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 51), 'str', 'define_macros')
    # Getting the type of 'lapack' (line 36)
    lapack_184356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 44), 'lapack')
    # Obtaining the member '__getitem__' of a type (line 36)
    getitem___184357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 44), lapack_184356, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 36)
    subscript_call_result_184358 = invoke(stypy.reporting.localization.Localization(__file__, 36, 44), getitem___184357, str_184355)
    
    # Getting the type of 'None' (line 36)
    None_184359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 75), 'None')
    # Applying the binary operator 'isnot' (line 36)
    result_is_not_184360 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 44), 'isnot', subscript_call_result_184358, None_184359)
    
    # Applying the binary operator 'and' (line 36)
    result_and_keyword_184361 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 11), 'and', result_contains_184354, result_is_not_184360)
    
    # Testing the type of an if condition (line 36)
    if_condition_184362 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 8), result_and_keyword_184361)
    # Assigning a type to the variable 'if_condition_184362' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'if_condition_184362', if_condition_184362)
    # SSA begins for if statement (line 36)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 37):
    
    # Obtaining the type of the subscript
    str_184363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 46), 'str', 'define_macros')
    # Getting the type of 'lapack' (line 37)
    lapack_184364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 39), 'lapack')
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___184365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 39), lapack_184364, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_184366 = invoke(stypy.reporting.localization.Localization(__file__, 37, 39), getitem___184365, str_184363)
    
    
    # Obtaining the type of the subscript
    str_184367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 56), 'str', 'define_macros')
    # Getting the type of 'numpy_nodepr_api' (line 38)
    numpy_nodepr_api_184368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 39), 'numpy_nodepr_api')
    # Obtaining the member '__getitem__' of a type (line 38)
    getitem___184369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 39), numpy_nodepr_api_184368, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 38)
    subscript_call_result_184370 = invoke(stypy.reporting.localization.Localization(__file__, 38, 39), getitem___184369, str_184367)
    
    # Applying the binary operator '+' (line 37)
    result_add_184371 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 39), '+', subscript_call_result_184366, subscript_call_result_184370)
    
    # Getting the type of 'lapack' (line 37)
    lapack_184372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'lapack')
    str_184373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 19), 'str', 'define_macros')
    # Storing an element on a container (line 37)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 12), lapack_184372, (str_184373, result_add_184371))
    # SSA branch for the else part of an if statement (line 36)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Subscript (line 40):
    
    # Obtaining the type of the subscript
    str_184374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 55), 'str', 'define_macros')
    # Getting the type of 'numpy_nodepr_api' (line 40)
    numpy_nodepr_api_184375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 38), 'numpy_nodepr_api')
    # Obtaining the member '__getitem__' of a type (line 40)
    getitem___184376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 38), numpy_nodepr_api_184375, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
    subscript_call_result_184377 = invoke(stypy.reporting.localization.Localization(__file__, 40, 38), getitem___184376, str_184374)
    
    # Getting the type of 'lapack' (line 40)
    lapack_184378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'lapack')
    str_184379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 19), 'str', 'define_macros')
    # Storing an element on a container (line 40)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 12), lapack_184378, (str_184379, subscript_call_result_184377))
    # SSA join for if statement (line 36)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 35)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 41):
    
    # Obtaining an instance of the builtin type 'list' (line 41)
    list_184380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 41)
    # Adding element type (line 41)
    str_184381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 15), 'str', 'lbfgsb.pyf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 14), list_184380, str_184381)
    # Adding element type (line 41)
    str_184382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 29), 'str', 'lbfgsb.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 14), list_184380, str_184382)
    # Adding element type (line 41)
    str_184383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 41), 'str', 'linpack.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 14), list_184380, str_184383)
    # Adding element type (line 41)
    str_184384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 54), 'str', 'timer.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 14), list_184380, str_184384)
    
    # Assigning a type to the variable 'sources' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'sources', list_184380)
    
    # Call to add_extension(...): (line 42)
    # Processing the call arguments (line 42)
    str_184387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 25), 'str', '_lbfgsb')
    # Processing the call keyword arguments (line 42)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'sources' (line 43)
    sources_184393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 60), 'sources', False)
    comprehension_184394 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 34), sources_184393)
    # Assigning a type to the variable 'x' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 34), 'x', comprehension_184394)
    
    # Call to join(...): (line 43)
    # Processing the call arguments (line 43)
    str_184389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 39), 'str', 'lbfgsb')
    # Getting the type of 'x' (line 43)
    x_184390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 48), 'x', False)
    # Processing the call keyword arguments (line 43)
    kwargs_184391 = {}
    # Getting the type of 'join' (line 43)
    join_184388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 34), 'join', False)
    # Calling join(args, kwargs) (line 43)
    join_call_result_184392 = invoke(stypy.reporting.localization.Localization(__file__, 43, 34), join_184388, *[str_184389, x_184390], **kwargs_184391)
    
    list_184395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 34), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 34), list_184395, join_call_result_184392)
    keyword_184396 = list_184395
    # Getting the type of 'lapack' (line 44)
    lapack_184397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 27), 'lapack', False)
    kwargs_184398 = {'lapack_184397': lapack_184397, 'sources': keyword_184396}
    # Getting the type of 'config' (line 42)
    config_184385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 42)
    add_extension_184386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 4), config_184385, 'add_extension')
    # Calling add_extension(args, kwargs) (line 42)
    add_extension_call_result_184399 = invoke(stypy.reporting.localization.Localization(__file__, 42, 4), add_extension_184386, *[str_184387], **kwargs_184398)
    
    
    # Assigning a List to a Name (line 46):
    
    # Obtaining an instance of the builtin type 'list' (line 46)
    list_184400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 46)
    # Adding element type (line 46)
    str_184401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 15), 'str', 'moduleTNC.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 14), list_184400, str_184401)
    # Adding element type (line 46)
    str_184402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 29), 'str', 'tnc.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 14), list_184400, str_184402)
    
    # Assigning a type to the variable 'sources' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'sources', list_184400)
    
    # Call to add_extension(...): (line 47)
    # Processing the call arguments (line 47)
    str_184405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 25), 'str', 'moduleTNC')
    # Processing the call keyword arguments (line 47)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'sources' (line 48)
    sources_184411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 57), 'sources', False)
    comprehension_184412 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 34), sources_184411)
    # Assigning a type to the variable 'x' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 34), 'x', comprehension_184412)
    
    # Call to join(...): (line 48)
    # Processing the call arguments (line 48)
    str_184407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 39), 'str', 'tnc')
    # Getting the type of 'x' (line 48)
    x_184408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 45), 'x', False)
    # Processing the call keyword arguments (line 48)
    kwargs_184409 = {}
    # Getting the type of 'join' (line 48)
    join_184406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 34), 'join', False)
    # Calling join(args, kwargs) (line 48)
    join_call_result_184410 = invoke(stypy.reporting.localization.Localization(__file__, 48, 34), join_184406, *[str_184407, x_184408], **kwargs_184409)
    
    list_184413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 34), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 34), list_184413, join_call_result_184410)
    keyword_184414 = list_184413
    
    # Obtaining an instance of the builtin type 'list' (line 49)
    list_184415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 49)
    # Adding element type (line 49)
    
    # Call to join(...): (line 49)
    # Processing the call arguments (line 49)
    str_184417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 39), 'str', 'tnc')
    str_184418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 45), 'str', 'tnc.h')
    # Processing the call keyword arguments (line 49)
    kwargs_184419 = {}
    # Getting the type of 'join' (line 49)
    join_184416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 34), 'join', False)
    # Calling join(args, kwargs) (line 49)
    join_call_result_184420 = invoke(stypy.reporting.localization.Localization(__file__, 49, 34), join_184416, *[str_184417, str_184418], **kwargs_184419)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 33), list_184415, join_call_result_184420)
    
    keyword_184421 = list_184415
    # Getting the type of 'numpy_nodepr_api' (line 50)
    numpy_nodepr_api_184422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 27), 'numpy_nodepr_api', False)
    kwargs_184423 = {'sources': keyword_184414, 'depends': keyword_184421, 'numpy_nodepr_api_184422': numpy_nodepr_api_184422}
    # Getting the type of 'config' (line 47)
    config_184403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 47)
    add_extension_184404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 4), config_184403, 'add_extension')
    # Calling add_extension(args, kwargs) (line 47)
    add_extension_call_result_184424 = invoke(stypy.reporting.localization.Localization(__file__, 47, 4), add_extension_184404, *[str_184405], **kwargs_184423)
    
    
    # Call to add_extension(...): (line 52)
    # Processing the call arguments (line 52)
    str_184427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 25), 'str', '_cobyla')
    # Processing the call keyword arguments (line 52)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 53)
    list_184433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 60), 'list')
    # Adding type elements to the builtin type 'list' instance (line 53)
    # Adding element type (line 53)
    str_184434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 61), 'str', 'cobyla.pyf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 60), list_184433, str_184434)
    # Adding element type (line 53)
    str_184435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 61), 'str', 'cobyla2.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 60), list_184433, str_184435)
    # Adding element type (line 53)
    str_184436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 61), 'str', 'trstlp.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 60), list_184433, str_184436)
    
    comprehension_184437 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 34), list_184433)
    # Assigning a type to the variable 'x' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 34), 'x', comprehension_184437)
    
    # Call to join(...): (line 53)
    # Processing the call arguments (line 53)
    str_184429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 39), 'str', 'cobyla')
    # Getting the type of 'x' (line 53)
    x_184430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 48), 'x', False)
    # Processing the call keyword arguments (line 53)
    kwargs_184431 = {}
    # Getting the type of 'join' (line 53)
    join_184428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 34), 'join', False)
    # Calling join(args, kwargs) (line 53)
    join_call_result_184432 = invoke(stypy.reporting.localization.Localization(__file__, 53, 34), join_184428, *[str_184429, x_184430], **kwargs_184431)
    
    list_184438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 34), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 34), list_184438, join_call_result_184432)
    keyword_184439 = list_184438
    # Getting the type of 'numpy_nodepr_api' (line 56)
    numpy_nodepr_api_184440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 27), 'numpy_nodepr_api', False)
    kwargs_184441 = {'sources': keyword_184439, 'numpy_nodepr_api_184440': numpy_nodepr_api_184440}
    # Getting the type of 'config' (line 52)
    config_184425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 52)
    add_extension_184426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 4), config_184425, 'add_extension')
    # Calling add_extension(args, kwargs) (line 52)
    add_extension_call_result_184442 = invoke(stypy.reporting.localization.Localization(__file__, 52, 4), add_extension_184426, *[str_184427], **kwargs_184441)
    
    
    # Assigning a List to a Name (line 58):
    
    # Obtaining an instance of the builtin type 'list' (line 58)
    list_184443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 58)
    # Adding element type (line 58)
    str_184444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 15), 'str', 'minpack2.pyf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 14), list_184443, str_184444)
    # Adding element type (line 58)
    str_184445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 31), 'str', 'dcsrch.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 14), list_184443, str_184445)
    # Adding element type (line 58)
    str_184446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 43), 'str', 'dcstep.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 14), list_184443, str_184446)
    
    # Assigning a type to the variable 'sources' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'sources', list_184443)
    
    # Call to add_extension(...): (line 59)
    # Processing the call arguments (line 59)
    str_184449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 25), 'str', 'minpack2')
    # Processing the call keyword arguments (line 59)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'sources' (line 60)
    sources_184455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 62), 'sources', False)
    comprehension_184456 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 34), sources_184455)
    # Assigning a type to the variable 'x' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 34), 'x', comprehension_184456)
    
    # Call to join(...): (line 60)
    # Processing the call arguments (line 60)
    str_184451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 39), 'str', 'minpack2')
    # Getting the type of 'x' (line 60)
    x_184452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 50), 'x', False)
    # Processing the call keyword arguments (line 60)
    kwargs_184453 = {}
    # Getting the type of 'join' (line 60)
    join_184450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 34), 'join', False)
    # Calling join(args, kwargs) (line 60)
    join_call_result_184454 = invoke(stypy.reporting.localization.Localization(__file__, 60, 34), join_184450, *[str_184451, x_184452], **kwargs_184453)
    
    list_184457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 34), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 34), list_184457, join_call_result_184454)
    keyword_184458 = list_184457
    # Getting the type of 'numpy_nodepr_api' (line 61)
    numpy_nodepr_api_184459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 27), 'numpy_nodepr_api', False)
    kwargs_184460 = {'sources': keyword_184458, 'numpy_nodepr_api_184459': numpy_nodepr_api_184459}
    # Getting the type of 'config' (line 59)
    config_184447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 59)
    add_extension_184448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 4), config_184447, 'add_extension')
    # Calling add_extension(args, kwargs) (line 59)
    add_extension_call_result_184461 = invoke(stypy.reporting.localization.Localization(__file__, 59, 4), add_extension_184448, *[str_184449], **kwargs_184460)
    
    
    # Assigning a List to a Name (line 63):
    
    # Obtaining an instance of the builtin type 'list' (line 63)
    list_184462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 63)
    # Adding element type (line 63)
    str_184463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 15), 'str', 'slsqp.pyf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 14), list_184462, str_184463)
    # Adding element type (line 63)
    str_184464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 28), 'str', 'slsqp_optmz.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 14), list_184462, str_184464)
    
    # Assigning a type to the variable 'sources' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'sources', list_184462)
    
    # Call to add_extension(...): (line 64)
    # Processing the call arguments (line 64)
    str_184467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 25), 'str', '_slsqp')
    # Processing the call keyword arguments (line 64)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'sources' (line 64)
    sources_184473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 70), 'sources', False)
    comprehension_184474 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 44), sources_184473)
    # Assigning a type to the variable 'x' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 44), 'x', comprehension_184474)
    
    # Call to join(...): (line 64)
    # Processing the call arguments (line 64)
    str_184469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 49), 'str', 'slsqp')
    # Getting the type of 'x' (line 64)
    x_184470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 58), 'x', False)
    # Processing the call keyword arguments (line 64)
    kwargs_184471 = {}
    # Getting the type of 'join' (line 64)
    join_184468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 44), 'join', False)
    # Calling join(args, kwargs) (line 64)
    join_call_result_184472 = invoke(stypy.reporting.localization.Localization(__file__, 64, 44), join_184468, *[str_184469, x_184470], **kwargs_184471)
    
    list_184475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 44), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 44), list_184475, join_call_result_184472)
    keyword_184476 = list_184475
    # Getting the type of 'numpy_nodepr_api' (line 65)
    numpy_nodepr_api_184477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 27), 'numpy_nodepr_api', False)
    kwargs_184478 = {'sources': keyword_184476, 'numpy_nodepr_api_184477': numpy_nodepr_api_184477}
    # Getting the type of 'config' (line 64)
    config_184465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 64)
    add_extension_184466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 4), config_184465, 'add_extension')
    # Calling add_extension(args, kwargs) (line 64)
    add_extension_call_result_184479 = invoke(stypy.reporting.localization.Localization(__file__, 64, 4), add_extension_184466, *[str_184467], **kwargs_184478)
    
    
    # Call to add_extension(...): (line 67)
    # Processing the call arguments (line 67)
    str_184482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 25), 'str', '_nnls')
    # Processing the call keyword arguments (line 67)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 68)
    list_184488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 51), 'list')
    # Adding type elements to the builtin type 'list' instance (line 68)
    # Adding element type (line 68)
    str_184489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 52), 'str', 'nnls.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 51), list_184488, str_184489)
    # Adding element type (line 68)
    str_184490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 61), 'str', 'nnls.pyf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 51), list_184488, str_184490)
    
    comprehension_184491 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 43), list_184488)
    # Assigning a type to the variable 'x' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 43), 'x', comprehension_184491)
    
    # Call to join(...): (line 67)
    # Processing the call arguments (line 67)
    str_184484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 48), 'str', 'nnls')
    # Getting the type of 'x' (line 67)
    x_184485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 56), 'x', False)
    # Processing the call keyword arguments (line 67)
    kwargs_184486 = {}
    # Getting the type of 'join' (line 67)
    join_184483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 43), 'join', False)
    # Calling join(args, kwargs) (line 67)
    join_call_result_184487 = invoke(stypy.reporting.localization.Localization(__file__, 67, 43), join_184483, *[str_184484, x_184485], **kwargs_184486)
    
    list_184492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 43), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 43), list_184492, join_call_result_184487)
    keyword_184493 = list_184492
    # Getting the type of 'numpy_nodepr_api' (line 69)
    numpy_nodepr_api_184494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 27), 'numpy_nodepr_api', False)
    kwargs_184495 = {'sources': keyword_184493, 'numpy_nodepr_api_184494': numpy_nodepr_api_184494}
    # Getting the type of 'config' (line 67)
    config_184480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 67)
    add_extension_184481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 4), config_184480, 'add_extension')
    # Calling add_extension(args, kwargs) (line 67)
    add_extension_call_result_184496 = invoke(stypy.reporting.localization.Localization(__file__, 67, 4), add_extension_184481, *[str_184482], **kwargs_184495)
    
    
    # Call to add_extension(...): (line 71)
    # Processing the call arguments (line 71)
    str_184499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 25), 'str', '_group_columns')
    # Processing the call keyword arguments (line 71)
    
    # Obtaining an instance of the builtin type 'list' (line 71)
    list_184500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 51), 'list')
    # Adding type elements to the builtin type 'list' instance (line 71)
    # Adding element type (line 71)
    str_184501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 52), 'str', '_group_columns.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 51), list_184500, str_184501)
    
    keyword_184502 = list_184500
    kwargs_184503 = {'sources': keyword_184502}
    # Getting the type of 'config' (line 71)
    config_184497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 71)
    add_extension_184498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 4), config_184497, 'add_extension')
    # Calling add_extension(args, kwargs) (line 71)
    add_extension_call_result_184504 = invoke(stypy.reporting.localization.Localization(__file__, 71, 4), add_extension_184498, *[str_184499], **kwargs_184503)
    
    
    # Call to add_subpackage(...): (line 73)
    # Processing the call arguments (line 73)
    str_184507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 26), 'str', '_lsq')
    # Processing the call keyword arguments (line 73)
    kwargs_184508 = {}
    # Getting the type of 'config' (line 73)
    config_184505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 73)
    add_subpackage_184506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 4), config_184505, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 73)
    add_subpackage_call_result_184509 = invoke(stypy.reporting.localization.Localization(__file__, 73, 4), add_subpackage_184506, *[str_184507], **kwargs_184508)
    
    
    # Call to add_subpackage(...): (line 75)
    # Processing the call arguments (line 75)
    str_184512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 26), 'str', '_trlib')
    # Processing the call keyword arguments (line 75)
    kwargs_184513 = {}
    # Getting the type of 'config' (line 75)
    config_184510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 75)
    add_subpackage_184511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 4), config_184510, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 75)
    add_subpackage_call_result_184514 = invoke(stypy.reporting.localization.Localization(__file__, 75, 4), add_subpackage_184511, *[str_184512], **kwargs_184513)
    
    
    # Call to add_data_dir(...): (line 77)
    # Processing the call arguments (line 77)
    str_184517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 77)
    kwargs_184518 = {}
    # Getting the type of 'config' (line 77)
    config_184515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 77)
    add_data_dir_184516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 4), config_184515, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 77)
    add_data_dir_call_result_184519 = invoke(stypy.reporting.localization.Localization(__file__, 77, 4), add_data_dir_184516, *[str_184517], **kwargs_184518)
    
    
    # Call to add_data_files(...): (line 80)
    # Processing the call arguments (line 80)
    str_184522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 26), 'str', 'lbfgsb/README')
    # Processing the call keyword arguments (line 80)
    kwargs_184523 = {}
    # Getting the type of 'config' (line 80)
    config_184520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'config', False)
    # Obtaining the member 'add_data_files' of a type (line 80)
    add_data_files_184521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 4), config_184520, 'add_data_files')
    # Calling add_data_files(args, kwargs) (line 80)
    add_data_files_call_result_184524 = invoke(stypy.reporting.localization.Localization(__file__, 80, 4), add_data_files_184521, *[str_184522], **kwargs_184523)
    
    # Getting the type of 'config' (line 82)
    config_184525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type', config_184525)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 7)
    stypy_return_type_184526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_184526)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_184526

# Assigning a type to the variable 'configuration' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 86, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 86)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
    import_184527 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 86, 4), 'numpy.distutils.core')

    if (type(import_184527) is not StypyTypeError):

        if (import_184527 != 'pyd_module'):
            __import__(import_184527)
            sys_modules_184528 = sys.modules[import_184527]
            import_from_module(stypy.reporting.localization.Localization(__file__, 86, 4), 'numpy.distutils.core', sys_modules_184528.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 86, 4), __file__, sys_modules_184528, sys_modules_184528.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 86, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'numpy.distutils.core', import_184527)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')
    
    
    # Call to setup(...): (line 87)
    # Processing the call keyword arguments (line 87)
    
    # Call to todict(...): (line 87)
    # Processing the call keyword arguments (line 87)
    kwargs_184536 = {}
    
    # Call to configuration(...): (line 87)
    # Processing the call keyword arguments (line 87)
    str_184531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 35), 'str', '')
    keyword_184532 = str_184531
    kwargs_184533 = {'top_path': keyword_184532}
    # Getting the type of 'configuration' (line 87)
    configuration_184530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 87)
    configuration_call_result_184534 = invoke(stypy.reporting.localization.Localization(__file__, 87, 12), configuration_184530, *[], **kwargs_184533)
    
    # Obtaining the member 'todict' of a type (line 87)
    todict_184535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 12), configuration_call_result_184534, 'todict')
    # Calling todict(args, kwargs) (line 87)
    todict_call_result_184537 = invoke(stypy.reporting.localization.Localization(__file__, 87, 12), todict_184535, *[], **kwargs_184536)
    
    kwargs_184538 = {'todict_call_result_184537': todict_call_result_184537}
    # Getting the type of 'setup' (line 87)
    setup_184529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 87)
    setup_call_result_184539 = invoke(stypy.reporting.localization.Localization(__file__, 87, 4), setup_184529, *[], **kwargs_184538)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
