
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from os.path import join, dirname
4: import glob
5: 
6: 
7: def configuration(parent_package='', top_path=None):
8:     from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
9:     from numpy.distutils.misc_util import get_info as get_misc_info
10:     from numpy.distutils.system_info import get_info as get_sys_info
11:     from distutils.sysconfig import get_python_inc
12: 
13:     config = Configuration('spatial', parent_package, top_path)
14: 
15:     config.add_data_dir('tests')
16: 
17:     # qhull
18:     qhull_src = list(glob.glob(join(dirname(__file__), 'qhull',
19:                                     'src', '*.c')))
20: 
21:     inc_dirs = [get_python_inc()]
22:     if inc_dirs[0] != get_python_inc(plat_specific=1):
23:         inc_dirs.append(get_python_inc(plat_specific=1))
24:     inc_dirs.append(get_numpy_include_dirs())
25: 
26:     cfg = dict(get_sys_info('lapack_opt'))
27:     cfg.setdefault('include_dirs', []).extend(inc_dirs)
28:     config.add_extension('qhull',
29:                          sources=['qhull.c'] + qhull_src,
30:                          **cfg)
31: 
32:     # cKDTree
33:     ckdtree_src = ['query.cxx',
34:                    'build.cxx',
35:                    'globals.cxx',
36:                    'cpp_exc.cxx',
37:                    'query_pairs.cxx',
38:                    'count_neighbors.cxx',
39:                    'query_ball_point.cxx',
40:                    'query_ball_tree.cxx',
41:                    'sparse_distances.cxx',
42:                    'fmax.cxx']
43: 
44:     ckdtree_src = [join('ckdtree', 'src', x) for x in ckdtree_src]
45: 
46:     ckdtree_headers = ['ckdtree_decl.h',
47:                        'ckdtree_methods.h',
48:                        'coo_entries.h',
49:                        'cpp_exc.h',
50:                        'cpp_utils.h',
51:                        'distance_base.h',
52:                        'distance.h',
53:                        'fmax.h',
54:                        'ordered_pair.h',
55:                        'partial_sort.h',
56:                        'rectangle.h']
57: 
58:     ckdtree_headers = [join('ckdtree', 'src', x) for x in ckdtree_headers]
59: 
60:     ckdtree_dep = ['ckdtree.cxx'] + ckdtree_headers + ckdtree_src
61:     config.add_extension('ckdtree',
62:                          sources=['ckdtree.cxx'] + ckdtree_src,
63:                          depends=ckdtree_dep,
64:                          include_dirs=inc_dirs + [join('ckdtree', 'src')])
65:     # _distance_wrap
66:     config.add_extension('_distance_wrap',
67:                          sources=[join('src', 'distance_wrap.c')],
68:                          depends=[join('src', 'distance_impl.h')],
69:                          include_dirs=[get_numpy_include_dirs()],
70:                          extra_info=get_misc_info("npymath"))
71: 
72:     config.add_extension('_voronoi',
73:                          sources=['_voronoi.c'])
74: 
75:     config.add_extension('_hausdorff',
76:                          sources=['_hausdorff.c'])
77: 
78:     # Add license files
79:     config.add_data_files('qhull/COPYING.txt')
80: 
81:     return config
82: 
83: 
84: if __name__ == '__main__':
85:     from numpy.distutils.core import setup
86:     setup(**configuration(top_path='').todict())
87: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from os.path import join, dirname' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
import_470063 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path')

if (type(import_470063) is not StypyTypeError):

    if (import_470063 != 'pyd_module'):
        __import__(import_470063)
        sys_modules_470064 = sys.modules[import_470063]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', sys_modules_470064.module_type_store, module_type_store, ['join', 'dirname'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_470064, sys_modules_470064.module_type_store, module_type_store)
    else:
        from os.path import join, dirname

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', None, module_type_store, ['join', 'dirname'], [join, dirname])

else:
    # Assigning a type to the variable 'os.path' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', import_470063)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import glob' statement (line 4)
import glob

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'glob', glob, module_type_store)


@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_470065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 33), 'str', '')
    # Getting the type of 'None' (line 7)
    None_470066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 46), 'None')
    defaults = [str_470065, None_470066]
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
    
    # 'from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs' statement (line 8)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
    import_470067 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.misc_util')

    if (type(import_470067) is not StypyTypeError):

        if (import_470067 != 'pyd_module'):
            __import__(import_470067)
            sys_modules_470068 = sys.modules[import_470067]
            import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.misc_util', sys_modules_470068.module_type_store, module_type_store, ['Configuration', 'get_numpy_include_dirs'])
            nest_module(stypy.reporting.localization.Localization(__file__, 8, 4), __file__, sys_modules_470068, sys_modules_470068.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

            import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration', 'get_numpy_include_dirs'], [Configuration, get_numpy_include_dirs])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.misc_util', import_470067)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 4))
    
    # 'from numpy.distutils.misc_util import get_misc_info' statement (line 9)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
    import_470069 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.misc_util')

    if (type(import_470069) is not StypyTypeError):

        if (import_470069 != 'pyd_module'):
            __import__(import_470069)
            sys_modules_470070 = sys.modules[import_470069]
            import_from_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.misc_util', sys_modules_470070.module_type_store, module_type_store, ['get_info'])
            nest_module(stypy.reporting.localization.Localization(__file__, 9, 4), __file__, sys_modules_470070, sys_modules_470070.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import get_info as get_misc_info

            import_from_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.misc_util', None, module_type_store, ['get_info'], [get_misc_info])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.misc_util', import_470069)

    # Adding an alias
    module_type_store.add_alias('get_misc_info', 'get_info')
    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 4))
    
    # 'from numpy.distutils.system_info import get_sys_info' statement (line 10)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
    import_470071 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.distutils.system_info')

    if (type(import_470071) is not StypyTypeError):

        if (import_470071 != 'pyd_module'):
            __import__(import_470071)
            sys_modules_470072 = sys.modules[import_470071]
            import_from_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.distutils.system_info', sys_modules_470072.module_type_store, module_type_store, ['get_info'])
            nest_module(stypy.reporting.localization.Localization(__file__, 10, 4), __file__, sys_modules_470072, sys_modules_470072.module_type_store, module_type_store)
        else:
            from numpy.distutils.system_info import get_info as get_sys_info

            import_from_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.distutils.system_info', None, module_type_store, ['get_info'], [get_sys_info])

    else:
        # Assigning a type to the variable 'numpy.distutils.system_info' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.distutils.system_info', import_470071)

    # Adding an alias
    module_type_store.add_alias('get_sys_info', 'get_info')
    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 4))
    
    # 'from distutils.sysconfig import get_python_inc' statement (line 11)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
    import_470073 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'distutils.sysconfig')

    if (type(import_470073) is not StypyTypeError):

        if (import_470073 != 'pyd_module'):
            __import__(import_470073)
            sys_modules_470074 = sys.modules[import_470073]
            import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'distutils.sysconfig', sys_modules_470074.module_type_store, module_type_store, ['get_python_inc'])
            nest_module(stypy.reporting.localization.Localization(__file__, 11, 4), __file__, sys_modules_470074, sys_modules_470074.module_type_store, module_type_store)
        else:
            from distutils.sysconfig import get_python_inc

            import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'distutils.sysconfig', None, module_type_store, ['get_python_inc'], [get_python_inc])

    else:
        # Assigning a type to the variable 'distutils.sysconfig' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'distutils.sysconfig', import_470073)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')
    
    
    # Assigning a Call to a Name (line 13):
    
    # Call to Configuration(...): (line 13)
    # Processing the call arguments (line 13)
    str_470076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 27), 'str', 'spatial')
    # Getting the type of 'parent_package' (line 13)
    parent_package_470077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 38), 'parent_package', False)
    # Getting the type of 'top_path' (line 13)
    top_path_470078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 54), 'top_path', False)
    # Processing the call keyword arguments (line 13)
    kwargs_470079 = {}
    # Getting the type of 'Configuration' (line 13)
    Configuration_470075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 13)
    Configuration_call_result_470080 = invoke(stypy.reporting.localization.Localization(__file__, 13, 13), Configuration_470075, *[str_470076, parent_package_470077, top_path_470078], **kwargs_470079)
    
    # Assigning a type to the variable 'config' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'config', Configuration_call_result_470080)
    
    # Call to add_data_dir(...): (line 15)
    # Processing the call arguments (line 15)
    str_470083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 15)
    kwargs_470084 = {}
    # Getting the type of 'config' (line 15)
    config_470081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 15)
    add_data_dir_470082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 4), config_470081, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 15)
    add_data_dir_call_result_470085 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), add_data_dir_470082, *[str_470083], **kwargs_470084)
    
    
    # Assigning a Call to a Name (line 18):
    
    # Call to list(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Call to glob(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Call to join(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Call to dirname(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of '__file__' (line 18)
    file___470091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 44), '__file__', False)
    # Processing the call keyword arguments (line 18)
    kwargs_470092 = {}
    # Getting the type of 'dirname' (line 18)
    dirname_470090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 36), 'dirname', False)
    # Calling dirname(args, kwargs) (line 18)
    dirname_call_result_470093 = invoke(stypy.reporting.localization.Localization(__file__, 18, 36), dirname_470090, *[file___470091], **kwargs_470092)
    
    str_470094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 55), 'str', 'qhull')
    str_470095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 36), 'str', 'src')
    str_470096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 43), 'str', '*.c')
    # Processing the call keyword arguments (line 18)
    kwargs_470097 = {}
    # Getting the type of 'join' (line 18)
    join_470089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 31), 'join', False)
    # Calling join(args, kwargs) (line 18)
    join_call_result_470098 = invoke(stypy.reporting.localization.Localization(__file__, 18, 31), join_470089, *[dirname_call_result_470093, str_470094, str_470095, str_470096], **kwargs_470097)
    
    # Processing the call keyword arguments (line 18)
    kwargs_470099 = {}
    # Getting the type of 'glob' (line 18)
    glob_470087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 21), 'glob', False)
    # Obtaining the member 'glob' of a type (line 18)
    glob_470088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 21), glob_470087, 'glob')
    # Calling glob(args, kwargs) (line 18)
    glob_call_result_470100 = invoke(stypy.reporting.localization.Localization(__file__, 18, 21), glob_470088, *[join_call_result_470098], **kwargs_470099)
    
    # Processing the call keyword arguments (line 18)
    kwargs_470101 = {}
    # Getting the type of 'list' (line 18)
    list_470086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 16), 'list', False)
    # Calling list(args, kwargs) (line 18)
    list_call_result_470102 = invoke(stypy.reporting.localization.Localization(__file__, 18, 16), list_470086, *[glob_call_result_470100], **kwargs_470101)
    
    # Assigning a type to the variable 'qhull_src' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'qhull_src', list_call_result_470102)
    
    # Assigning a List to a Name (line 21):
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_470103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    
    # Call to get_python_inc(...): (line 21)
    # Processing the call keyword arguments (line 21)
    kwargs_470105 = {}
    # Getting the type of 'get_python_inc' (line 21)
    get_python_inc_470104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'get_python_inc', False)
    # Calling get_python_inc(args, kwargs) (line 21)
    get_python_inc_call_result_470106 = invoke(stypy.reporting.localization.Localization(__file__, 21, 16), get_python_inc_470104, *[], **kwargs_470105)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 15), list_470103, get_python_inc_call_result_470106)
    
    # Assigning a type to the variable 'inc_dirs' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'inc_dirs', list_470103)
    
    
    
    # Obtaining the type of the subscript
    int_470107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'int')
    # Getting the type of 'inc_dirs' (line 22)
    inc_dirs_470108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 7), 'inc_dirs')
    # Obtaining the member '__getitem__' of a type (line 22)
    getitem___470109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 7), inc_dirs_470108, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 22)
    subscript_call_result_470110 = invoke(stypy.reporting.localization.Localization(__file__, 22, 7), getitem___470109, int_470107)
    
    
    # Call to get_python_inc(...): (line 22)
    # Processing the call keyword arguments (line 22)
    int_470112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 51), 'int')
    keyword_470113 = int_470112
    kwargs_470114 = {'plat_specific': keyword_470113}
    # Getting the type of 'get_python_inc' (line 22)
    get_python_inc_470111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 22), 'get_python_inc', False)
    # Calling get_python_inc(args, kwargs) (line 22)
    get_python_inc_call_result_470115 = invoke(stypy.reporting.localization.Localization(__file__, 22, 22), get_python_inc_470111, *[], **kwargs_470114)
    
    # Applying the binary operator '!=' (line 22)
    result_ne_470116 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 7), '!=', subscript_call_result_470110, get_python_inc_call_result_470115)
    
    # Testing the type of an if condition (line 22)
    if_condition_470117 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 22, 4), result_ne_470116)
    # Assigning a type to the variable 'if_condition_470117' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'if_condition_470117', if_condition_470117)
    # SSA begins for if statement (line 22)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Call to get_python_inc(...): (line 23)
    # Processing the call keyword arguments (line 23)
    int_470121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 53), 'int')
    keyword_470122 = int_470121
    kwargs_470123 = {'plat_specific': keyword_470122}
    # Getting the type of 'get_python_inc' (line 23)
    get_python_inc_470120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 24), 'get_python_inc', False)
    # Calling get_python_inc(args, kwargs) (line 23)
    get_python_inc_call_result_470124 = invoke(stypy.reporting.localization.Localization(__file__, 23, 24), get_python_inc_470120, *[], **kwargs_470123)
    
    # Processing the call keyword arguments (line 23)
    kwargs_470125 = {}
    # Getting the type of 'inc_dirs' (line 23)
    inc_dirs_470118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'inc_dirs', False)
    # Obtaining the member 'append' of a type (line 23)
    append_470119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), inc_dirs_470118, 'append')
    # Calling append(args, kwargs) (line 23)
    append_call_result_470126 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), append_470119, *[get_python_inc_call_result_470124], **kwargs_470125)
    
    # SSA join for if statement (line 22)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 24)
    # Processing the call arguments (line 24)
    
    # Call to get_numpy_include_dirs(...): (line 24)
    # Processing the call keyword arguments (line 24)
    kwargs_470130 = {}
    # Getting the type of 'get_numpy_include_dirs' (line 24)
    get_numpy_include_dirs_470129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 20), 'get_numpy_include_dirs', False)
    # Calling get_numpy_include_dirs(args, kwargs) (line 24)
    get_numpy_include_dirs_call_result_470131 = invoke(stypy.reporting.localization.Localization(__file__, 24, 20), get_numpy_include_dirs_470129, *[], **kwargs_470130)
    
    # Processing the call keyword arguments (line 24)
    kwargs_470132 = {}
    # Getting the type of 'inc_dirs' (line 24)
    inc_dirs_470127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'inc_dirs', False)
    # Obtaining the member 'append' of a type (line 24)
    append_470128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 4), inc_dirs_470127, 'append')
    # Calling append(args, kwargs) (line 24)
    append_call_result_470133 = invoke(stypy.reporting.localization.Localization(__file__, 24, 4), append_470128, *[get_numpy_include_dirs_call_result_470131], **kwargs_470132)
    
    
    # Assigning a Call to a Name (line 26):
    
    # Call to dict(...): (line 26)
    # Processing the call arguments (line 26)
    
    # Call to get_sys_info(...): (line 26)
    # Processing the call arguments (line 26)
    str_470136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 28), 'str', 'lapack_opt')
    # Processing the call keyword arguments (line 26)
    kwargs_470137 = {}
    # Getting the type of 'get_sys_info' (line 26)
    get_sys_info_470135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'get_sys_info', False)
    # Calling get_sys_info(args, kwargs) (line 26)
    get_sys_info_call_result_470138 = invoke(stypy.reporting.localization.Localization(__file__, 26, 15), get_sys_info_470135, *[str_470136], **kwargs_470137)
    
    # Processing the call keyword arguments (line 26)
    kwargs_470139 = {}
    # Getting the type of 'dict' (line 26)
    dict_470134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 10), 'dict', False)
    # Calling dict(args, kwargs) (line 26)
    dict_call_result_470140 = invoke(stypy.reporting.localization.Localization(__file__, 26, 10), dict_470134, *[get_sys_info_call_result_470138], **kwargs_470139)
    
    # Assigning a type to the variable 'cfg' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'cfg', dict_call_result_470140)
    
    # Call to extend(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'inc_dirs' (line 27)
    inc_dirs_470148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 46), 'inc_dirs', False)
    # Processing the call keyword arguments (line 27)
    kwargs_470149 = {}
    
    # Call to setdefault(...): (line 27)
    # Processing the call arguments (line 27)
    str_470143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 19), 'str', 'include_dirs')
    
    # Obtaining an instance of the builtin type 'list' (line 27)
    list_470144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 27)
    
    # Processing the call keyword arguments (line 27)
    kwargs_470145 = {}
    # Getting the type of 'cfg' (line 27)
    cfg_470141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'cfg', False)
    # Obtaining the member 'setdefault' of a type (line 27)
    setdefault_470142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 4), cfg_470141, 'setdefault')
    # Calling setdefault(args, kwargs) (line 27)
    setdefault_call_result_470146 = invoke(stypy.reporting.localization.Localization(__file__, 27, 4), setdefault_470142, *[str_470143, list_470144], **kwargs_470145)
    
    # Obtaining the member 'extend' of a type (line 27)
    extend_470147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 4), setdefault_call_result_470146, 'extend')
    # Calling extend(args, kwargs) (line 27)
    extend_call_result_470150 = invoke(stypy.reporting.localization.Localization(__file__, 27, 4), extend_470147, *[inc_dirs_470148], **kwargs_470149)
    
    
    # Call to add_extension(...): (line 28)
    # Processing the call arguments (line 28)
    str_470153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 25), 'str', 'qhull')
    # Processing the call keyword arguments (line 28)
    
    # Obtaining an instance of the builtin type 'list' (line 29)
    list_470154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 29)
    # Adding element type (line 29)
    str_470155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 34), 'str', 'qhull.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 33), list_470154, str_470155)
    
    # Getting the type of 'qhull_src' (line 29)
    qhull_src_470156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 47), 'qhull_src', False)
    # Applying the binary operator '+' (line 29)
    result_add_470157 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 33), '+', list_470154, qhull_src_470156)
    
    keyword_470158 = result_add_470157
    # Getting the type of 'cfg' (line 30)
    cfg_470159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 27), 'cfg', False)
    kwargs_470160 = {'sources': keyword_470158, 'cfg_470159': cfg_470159}
    # Getting the type of 'config' (line 28)
    config_470151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 28)
    add_extension_470152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 4), config_470151, 'add_extension')
    # Calling add_extension(args, kwargs) (line 28)
    add_extension_call_result_470161 = invoke(stypy.reporting.localization.Localization(__file__, 28, 4), add_extension_470152, *[str_470153], **kwargs_470160)
    
    
    # Assigning a List to a Name (line 33):
    
    # Obtaining an instance of the builtin type 'list' (line 33)
    list_470162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 33)
    # Adding element type (line 33)
    str_470163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 19), 'str', 'query.cxx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 18), list_470162, str_470163)
    # Adding element type (line 33)
    str_470164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 19), 'str', 'build.cxx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 18), list_470162, str_470164)
    # Adding element type (line 33)
    str_470165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 19), 'str', 'globals.cxx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 18), list_470162, str_470165)
    # Adding element type (line 33)
    str_470166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 19), 'str', 'cpp_exc.cxx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 18), list_470162, str_470166)
    # Adding element type (line 33)
    str_470167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 19), 'str', 'query_pairs.cxx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 18), list_470162, str_470167)
    # Adding element type (line 33)
    str_470168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 19), 'str', 'count_neighbors.cxx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 18), list_470162, str_470168)
    # Adding element type (line 33)
    str_470169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 19), 'str', 'query_ball_point.cxx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 18), list_470162, str_470169)
    # Adding element type (line 33)
    str_470170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 19), 'str', 'query_ball_tree.cxx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 18), list_470162, str_470170)
    # Adding element type (line 33)
    str_470171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 19), 'str', 'sparse_distances.cxx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 18), list_470162, str_470171)
    # Adding element type (line 33)
    str_470172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 19), 'str', 'fmax.cxx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 18), list_470162, str_470172)
    
    # Assigning a type to the variable 'ckdtree_src' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'ckdtree_src', list_470162)
    
    # Assigning a ListComp to a Name (line 44):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'ckdtree_src' (line 44)
    ckdtree_src_470179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 54), 'ckdtree_src')
    comprehension_470180 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 19), ckdtree_src_470179)
    # Assigning a type to the variable 'x' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 19), 'x', comprehension_470180)
    
    # Call to join(...): (line 44)
    # Processing the call arguments (line 44)
    str_470174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 24), 'str', 'ckdtree')
    str_470175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 35), 'str', 'src')
    # Getting the type of 'x' (line 44)
    x_470176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 42), 'x', False)
    # Processing the call keyword arguments (line 44)
    kwargs_470177 = {}
    # Getting the type of 'join' (line 44)
    join_470173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 19), 'join', False)
    # Calling join(args, kwargs) (line 44)
    join_call_result_470178 = invoke(stypy.reporting.localization.Localization(__file__, 44, 19), join_470173, *[str_470174, str_470175, x_470176], **kwargs_470177)
    
    list_470181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 19), list_470181, join_call_result_470178)
    # Assigning a type to the variable 'ckdtree_src' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'ckdtree_src', list_470181)
    
    # Assigning a List to a Name (line 46):
    
    # Obtaining an instance of the builtin type 'list' (line 46)
    list_470182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 46)
    # Adding element type (line 46)
    str_470183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 23), 'str', 'ckdtree_decl.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_470182, str_470183)
    # Adding element type (line 46)
    str_470184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 23), 'str', 'ckdtree_methods.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_470182, str_470184)
    # Adding element type (line 46)
    str_470185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 23), 'str', 'coo_entries.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_470182, str_470185)
    # Adding element type (line 46)
    str_470186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 23), 'str', 'cpp_exc.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_470182, str_470186)
    # Adding element type (line 46)
    str_470187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 23), 'str', 'cpp_utils.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_470182, str_470187)
    # Adding element type (line 46)
    str_470188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 23), 'str', 'distance_base.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_470182, str_470188)
    # Adding element type (line 46)
    str_470189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 23), 'str', 'distance.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_470182, str_470189)
    # Adding element type (line 46)
    str_470190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 23), 'str', 'fmax.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_470182, str_470190)
    # Adding element type (line 46)
    str_470191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 23), 'str', 'ordered_pair.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_470182, str_470191)
    # Adding element type (line 46)
    str_470192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 23), 'str', 'partial_sort.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_470182, str_470192)
    # Adding element type (line 46)
    str_470193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 23), 'str', 'rectangle.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_470182, str_470193)
    
    # Assigning a type to the variable 'ckdtree_headers' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'ckdtree_headers', list_470182)
    
    # Assigning a ListComp to a Name (line 58):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'ckdtree_headers' (line 58)
    ckdtree_headers_470200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 58), 'ckdtree_headers')
    comprehension_470201 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 23), ckdtree_headers_470200)
    # Assigning a type to the variable 'x' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 23), 'x', comprehension_470201)
    
    # Call to join(...): (line 58)
    # Processing the call arguments (line 58)
    str_470195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 28), 'str', 'ckdtree')
    str_470196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 39), 'str', 'src')
    # Getting the type of 'x' (line 58)
    x_470197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 46), 'x', False)
    # Processing the call keyword arguments (line 58)
    kwargs_470198 = {}
    # Getting the type of 'join' (line 58)
    join_470194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 23), 'join', False)
    # Calling join(args, kwargs) (line 58)
    join_call_result_470199 = invoke(stypy.reporting.localization.Localization(__file__, 58, 23), join_470194, *[str_470195, str_470196, x_470197], **kwargs_470198)
    
    list_470202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 23), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 23), list_470202, join_call_result_470199)
    # Assigning a type to the variable 'ckdtree_headers' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'ckdtree_headers', list_470202)
    
    # Assigning a BinOp to a Name (line 60):
    
    # Obtaining an instance of the builtin type 'list' (line 60)
    list_470203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 60)
    # Adding element type (line 60)
    str_470204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 19), 'str', 'ckdtree.cxx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 18), list_470203, str_470204)
    
    # Getting the type of 'ckdtree_headers' (line 60)
    ckdtree_headers_470205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 36), 'ckdtree_headers')
    # Applying the binary operator '+' (line 60)
    result_add_470206 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 18), '+', list_470203, ckdtree_headers_470205)
    
    # Getting the type of 'ckdtree_src' (line 60)
    ckdtree_src_470207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 54), 'ckdtree_src')
    # Applying the binary operator '+' (line 60)
    result_add_470208 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 52), '+', result_add_470206, ckdtree_src_470207)
    
    # Assigning a type to the variable 'ckdtree_dep' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'ckdtree_dep', result_add_470208)
    
    # Call to add_extension(...): (line 61)
    # Processing the call arguments (line 61)
    str_470211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 25), 'str', 'ckdtree')
    # Processing the call keyword arguments (line 61)
    
    # Obtaining an instance of the builtin type 'list' (line 62)
    list_470212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 62)
    # Adding element type (line 62)
    str_470213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 34), 'str', 'ckdtree.cxx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 33), list_470212, str_470213)
    
    # Getting the type of 'ckdtree_src' (line 62)
    ckdtree_src_470214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 51), 'ckdtree_src', False)
    # Applying the binary operator '+' (line 62)
    result_add_470215 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 33), '+', list_470212, ckdtree_src_470214)
    
    keyword_470216 = result_add_470215
    # Getting the type of 'ckdtree_dep' (line 63)
    ckdtree_dep_470217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 33), 'ckdtree_dep', False)
    keyword_470218 = ckdtree_dep_470217
    # Getting the type of 'inc_dirs' (line 64)
    inc_dirs_470219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 38), 'inc_dirs', False)
    
    # Obtaining an instance of the builtin type 'list' (line 64)
    list_470220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 49), 'list')
    # Adding type elements to the builtin type 'list' instance (line 64)
    # Adding element type (line 64)
    
    # Call to join(...): (line 64)
    # Processing the call arguments (line 64)
    str_470222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 55), 'str', 'ckdtree')
    str_470223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 66), 'str', 'src')
    # Processing the call keyword arguments (line 64)
    kwargs_470224 = {}
    # Getting the type of 'join' (line 64)
    join_470221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 50), 'join', False)
    # Calling join(args, kwargs) (line 64)
    join_call_result_470225 = invoke(stypy.reporting.localization.Localization(__file__, 64, 50), join_470221, *[str_470222, str_470223], **kwargs_470224)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 49), list_470220, join_call_result_470225)
    
    # Applying the binary operator '+' (line 64)
    result_add_470226 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 38), '+', inc_dirs_470219, list_470220)
    
    keyword_470227 = result_add_470226
    kwargs_470228 = {'sources': keyword_470216, 'depends': keyword_470218, 'include_dirs': keyword_470227}
    # Getting the type of 'config' (line 61)
    config_470209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 61)
    add_extension_470210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 4), config_470209, 'add_extension')
    # Calling add_extension(args, kwargs) (line 61)
    add_extension_call_result_470229 = invoke(stypy.reporting.localization.Localization(__file__, 61, 4), add_extension_470210, *[str_470211], **kwargs_470228)
    
    
    # Call to add_extension(...): (line 66)
    # Processing the call arguments (line 66)
    str_470232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 25), 'str', '_distance_wrap')
    # Processing the call keyword arguments (line 66)
    
    # Obtaining an instance of the builtin type 'list' (line 67)
    list_470233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 67)
    # Adding element type (line 67)
    
    # Call to join(...): (line 67)
    # Processing the call arguments (line 67)
    str_470235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 39), 'str', 'src')
    str_470236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 46), 'str', 'distance_wrap.c')
    # Processing the call keyword arguments (line 67)
    kwargs_470237 = {}
    # Getting the type of 'join' (line 67)
    join_470234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 34), 'join', False)
    # Calling join(args, kwargs) (line 67)
    join_call_result_470238 = invoke(stypy.reporting.localization.Localization(__file__, 67, 34), join_470234, *[str_470235, str_470236], **kwargs_470237)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 33), list_470233, join_call_result_470238)
    
    keyword_470239 = list_470233
    
    # Obtaining an instance of the builtin type 'list' (line 68)
    list_470240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 68)
    # Adding element type (line 68)
    
    # Call to join(...): (line 68)
    # Processing the call arguments (line 68)
    str_470242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 39), 'str', 'src')
    str_470243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 46), 'str', 'distance_impl.h')
    # Processing the call keyword arguments (line 68)
    kwargs_470244 = {}
    # Getting the type of 'join' (line 68)
    join_470241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 34), 'join', False)
    # Calling join(args, kwargs) (line 68)
    join_call_result_470245 = invoke(stypy.reporting.localization.Localization(__file__, 68, 34), join_470241, *[str_470242, str_470243], **kwargs_470244)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 33), list_470240, join_call_result_470245)
    
    keyword_470246 = list_470240
    
    # Obtaining an instance of the builtin type 'list' (line 69)
    list_470247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 69)
    # Adding element type (line 69)
    
    # Call to get_numpy_include_dirs(...): (line 69)
    # Processing the call keyword arguments (line 69)
    kwargs_470249 = {}
    # Getting the type of 'get_numpy_include_dirs' (line 69)
    get_numpy_include_dirs_470248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 39), 'get_numpy_include_dirs', False)
    # Calling get_numpy_include_dirs(args, kwargs) (line 69)
    get_numpy_include_dirs_call_result_470250 = invoke(stypy.reporting.localization.Localization(__file__, 69, 39), get_numpy_include_dirs_470248, *[], **kwargs_470249)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 38), list_470247, get_numpy_include_dirs_call_result_470250)
    
    keyword_470251 = list_470247
    
    # Call to get_misc_info(...): (line 70)
    # Processing the call arguments (line 70)
    str_470253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 50), 'str', 'npymath')
    # Processing the call keyword arguments (line 70)
    kwargs_470254 = {}
    # Getting the type of 'get_misc_info' (line 70)
    get_misc_info_470252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 36), 'get_misc_info', False)
    # Calling get_misc_info(args, kwargs) (line 70)
    get_misc_info_call_result_470255 = invoke(stypy.reporting.localization.Localization(__file__, 70, 36), get_misc_info_470252, *[str_470253], **kwargs_470254)
    
    keyword_470256 = get_misc_info_call_result_470255
    kwargs_470257 = {'sources': keyword_470239, 'depends': keyword_470246, 'extra_info': keyword_470256, 'include_dirs': keyword_470251}
    # Getting the type of 'config' (line 66)
    config_470230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 66)
    add_extension_470231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 4), config_470230, 'add_extension')
    # Calling add_extension(args, kwargs) (line 66)
    add_extension_call_result_470258 = invoke(stypy.reporting.localization.Localization(__file__, 66, 4), add_extension_470231, *[str_470232], **kwargs_470257)
    
    
    # Call to add_extension(...): (line 72)
    # Processing the call arguments (line 72)
    str_470261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 25), 'str', '_voronoi')
    # Processing the call keyword arguments (line 72)
    
    # Obtaining an instance of the builtin type 'list' (line 73)
    list_470262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 73)
    # Adding element type (line 73)
    str_470263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 34), 'str', '_voronoi.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 33), list_470262, str_470263)
    
    keyword_470264 = list_470262
    kwargs_470265 = {'sources': keyword_470264}
    # Getting the type of 'config' (line 72)
    config_470259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 72)
    add_extension_470260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 4), config_470259, 'add_extension')
    # Calling add_extension(args, kwargs) (line 72)
    add_extension_call_result_470266 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), add_extension_470260, *[str_470261], **kwargs_470265)
    
    
    # Call to add_extension(...): (line 75)
    # Processing the call arguments (line 75)
    str_470269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'str', '_hausdorff')
    # Processing the call keyword arguments (line 75)
    
    # Obtaining an instance of the builtin type 'list' (line 76)
    list_470270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 76)
    # Adding element type (line 76)
    str_470271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 34), 'str', '_hausdorff.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 33), list_470270, str_470271)
    
    keyword_470272 = list_470270
    kwargs_470273 = {'sources': keyword_470272}
    # Getting the type of 'config' (line 75)
    config_470267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 75)
    add_extension_470268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 4), config_470267, 'add_extension')
    # Calling add_extension(args, kwargs) (line 75)
    add_extension_call_result_470274 = invoke(stypy.reporting.localization.Localization(__file__, 75, 4), add_extension_470268, *[str_470269], **kwargs_470273)
    
    
    # Call to add_data_files(...): (line 79)
    # Processing the call arguments (line 79)
    str_470277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 26), 'str', 'qhull/COPYING.txt')
    # Processing the call keyword arguments (line 79)
    kwargs_470278 = {}
    # Getting the type of 'config' (line 79)
    config_470275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'config', False)
    # Obtaining the member 'add_data_files' of a type (line 79)
    add_data_files_470276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 4), config_470275, 'add_data_files')
    # Calling add_data_files(args, kwargs) (line 79)
    add_data_files_call_result_470279 = invoke(stypy.reporting.localization.Localization(__file__, 79, 4), add_data_files_470276, *[str_470277], **kwargs_470278)
    
    # Getting the type of 'config' (line 81)
    config_470280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type', config_470280)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 7)
    stypy_return_type_470281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_470281)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_470281

# Assigning a type to the variable 'configuration' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 85, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 85)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
    import_470282 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 85, 4), 'numpy.distutils.core')

    if (type(import_470282) is not StypyTypeError):

        if (import_470282 != 'pyd_module'):
            __import__(import_470282)
            sys_modules_470283 = sys.modules[import_470282]
            import_from_module(stypy.reporting.localization.Localization(__file__, 85, 4), 'numpy.distutils.core', sys_modules_470283.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 85, 4), __file__, sys_modules_470283, sys_modules_470283.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 85, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'numpy.distutils.core', import_470282)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')
    
    
    # Call to setup(...): (line 86)
    # Processing the call keyword arguments (line 86)
    
    # Call to todict(...): (line 86)
    # Processing the call keyword arguments (line 86)
    kwargs_470291 = {}
    
    # Call to configuration(...): (line 86)
    # Processing the call keyword arguments (line 86)
    str_470286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 35), 'str', '')
    keyword_470287 = str_470286
    kwargs_470288 = {'top_path': keyword_470287}
    # Getting the type of 'configuration' (line 86)
    configuration_470285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 86)
    configuration_call_result_470289 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), configuration_470285, *[], **kwargs_470288)
    
    # Obtaining the member 'todict' of a type (line 86)
    todict_470290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), configuration_call_result_470289, 'todict')
    # Calling todict(args, kwargs) (line 86)
    todict_call_result_470292 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), todict_470290, *[], **kwargs_470291)
    
    kwargs_470293 = {'todict_call_result_470292': todict_call_result_470292}
    # Getting the type of 'setup' (line 86)
    setup_470284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 86)
    setup_call_result_470294 = invoke(stypy.reporting.localization.Localization(__file__, 86, 4), setup_470284, *[], **kwargs_470293)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
