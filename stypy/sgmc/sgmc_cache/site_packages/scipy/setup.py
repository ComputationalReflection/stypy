
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import sys
4: 
5: 
6: def configuration(parent_package='',top_path=None):
7:     from numpy.distutils.misc_util import Configuration
8:     config = Configuration('scipy',parent_package,top_path)
9:     config.add_subpackage('cluster')
10:     config.add_subpackage('constants')
11:     config.add_subpackage('fftpack')
12:     config.add_subpackage('integrate')
13:     config.add_subpackage('interpolate')
14:     config.add_subpackage('io')
15:     config.add_subpackage('linalg')
16:     config.add_data_files('*.pxd')
17:     config.add_subpackage('misc')
18:     config.add_subpackage('odr')
19:     config.add_subpackage('optimize')
20:     config.add_subpackage('signal')
21:     config.add_subpackage('sparse')
22:     config.add_subpackage('spatial')
23:     config.add_subpackage('special')
24:     config.add_subpackage('stats')
25:     config.add_subpackage('ndimage')
26:     config.add_subpackage('_build_utils')
27:     config.add_subpackage('_lib')
28:     config.make_config_py()
29:     return config
30: 
31: if __name__ == '__main__':
32:     from numpy.distutils.core import setup
33:     setup(**configuration(top_path='').todict())
34: 

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


@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_63 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 33), 'str', '')
    # Getting the type of 'None' (line 6)
    None_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 45), 'None')
    defaults = [str_63, None_64]
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
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/')
    import_65 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util')

    if (type(import_65) is not StypyTypeError):

        if (import_65 != 'pyd_module'):
            __import__(import_65)
            sys_modules_66 = sys.modules[import_65]
            import_from_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util', sys_modules_66.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 7, 4), __file__, sys_modules_66, sys_modules_66.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util', import_65)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/')
    
    
    # Assigning a Call to a Name (line 8):
    
    # Call to Configuration(...): (line 8)
    # Processing the call arguments (line 8)
    str_68 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 27), 'str', 'scipy')
    # Getting the type of 'parent_package' (line 8)
    parent_package_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 35), 'parent_package', False)
    # Getting the type of 'top_path' (line 8)
    top_path_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 50), 'top_path', False)
    # Processing the call keyword arguments (line 8)
    kwargs_71 = {}
    # Getting the type of 'Configuration' (line 8)
    Configuration_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 8)
    Configuration_call_result_72 = invoke(stypy.reporting.localization.Localization(__file__, 8, 13), Configuration_67, *[str_68, parent_package_69, top_path_70], **kwargs_71)
    
    # Assigning a type to the variable 'config' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'config', Configuration_call_result_72)
    
    # Call to add_subpackage(...): (line 9)
    # Processing the call arguments (line 9)
    str_75 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 26), 'str', 'cluster')
    # Processing the call keyword arguments (line 9)
    kwargs_76 = {}
    # Getting the type of 'config' (line 9)
    config_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 9)
    add_subpackage_74 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), config_73, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 9)
    add_subpackage_call_result_77 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), add_subpackage_74, *[str_75], **kwargs_76)
    
    
    # Call to add_subpackage(...): (line 10)
    # Processing the call arguments (line 10)
    str_80 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 26), 'str', 'constants')
    # Processing the call keyword arguments (line 10)
    kwargs_81 = {}
    # Getting the type of 'config' (line 10)
    config_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 10)
    add_subpackage_79 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 4), config_78, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 10)
    add_subpackage_call_result_82 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), add_subpackage_79, *[str_80], **kwargs_81)
    
    
    # Call to add_subpackage(...): (line 11)
    # Processing the call arguments (line 11)
    str_85 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 26), 'str', 'fftpack')
    # Processing the call keyword arguments (line 11)
    kwargs_86 = {}
    # Getting the type of 'config' (line 11)
    config_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 11)
    add_subpackage_84 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), config_83, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 11)
    add_subpackage_call_result_87 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), add_subpackage_84, *[str_85], **kwargs_86)
    
    
    # Call to add_subpackage(...): (line 12)
    # Processing the call arguments (line 12)
    str_90 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 26), 'str', 'integrate')
    # Processing the call keyword arguments (line 12)
    kwargs_91 = {}
    # Getting the type of 'config' (line 12)
    config_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 12)
    add_subpackage_89 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), config_88, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 12)
    add_subpackage_call_result_92 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), add_subpackage_89, *[str_90], **kwargs_91)
    
    
    # Call to add_subpackage(...): (line 13)
    # Processing the call arguments (line 13)
    str_95 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 26), 'str', 'interpolate')
    # Processing the call keyword arguments (line 13)
    kwargs_96 = {}
    # Getting the type of 'config' (line 13)
    config_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 13)
    add_subpackage_94 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 4), config_93, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 13)
    add_subpackage_call_result_97 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), add_subpackage_94, *[str_95], **kwargs_96)
    
    
    # Call to add_subpackage(...): (line 14)
    # Processing the call arguments (line 14)
    str_100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 26), 'str', 'io')
    # Processing the call keyword arguments (line 14)
    kwargs_101 = {}
    # Getting the type of 'config' (line 14)
    config_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 14)
    add_subpackage_99 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 4), config_98, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 14)
    add_subpackage_call_result_102 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), add_subpackage_99, *[str_100], **kwargs_101)
    
    
    # Call to add_subpackage(...): (line 15)
    # Processing the call arguments (line 15)
    str_105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 26), 'str', 'linalg')
    # Processing the call keyword arguments (line 15)
    kwargs_106 = {}
    # Getting the type of 'config' (line 15)
    config_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 15)
    add_subpackage_104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 4), config_103, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 15)
    add_subpackage_call_result_107 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), add_subpackage_104, *[str_105], **kwargs_106)
    
    
    # Call to add_data_files(...): (line 16)
    # Processing the call arguments (line 16)
    str_110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 26), 'str', '*.pxd')
    # Processing the call keyword arguments (line 16)
    kwargs_111 = {}
    # Getting the type of 'config' (line 16)
    config_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'config', False)
    # Obtaining the member 'add_data_files' of a type (line 16)
    add_data_files_109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), config_108, 'add_data_files')
    # Calling add_data_files(args, kwargs) (line 16)
    add_data_files_call_result_112 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), add_data_files_109, *[str_110], **kwargs_111)
    
    
    # Call to add_subpackage(...): (line 17)
    # Processing the call arguments (line 17)
    str_115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 26), 'str', 'misc')
    # Processing the call keyword arguments (line 17)
    kwargs_116 = {}
    # Getting the type of 'config' (line 17)
    config_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 17)
    add_subpackage_114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 4), config_113, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 17)
    add_subpackage_call_result_117 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), add_subpackage_114, *[str_115], **kwargs_116)
    
    
    # Call to add_subpackage(...): (line 18)
    # Processing the call arguments (line 18)
    str_120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 26), 'str', 'odr')
    # Processing the call keyword arguments (line 18)
    kwargs_121 = {}
    # Getting the type of 'config' (line 18)
    config_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 18)
    add_subpackage_119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 4), config_118, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 18)
    add_subpackage_call_result_122 = invoke(stypy.reporting.localization.Localization(__file__, 18, 4), add_subpackage_119, *[str_120], **kwargs_121)
    
    
    # Call to add_subpackage(...): (line 19)
    # Processing the call arguments (line 19)
    str_125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 26), 'str', 'optimize')
    # Processing the call keyword arguments (line 19)
    kwargs_126 = {}
    # Getting the type of 'config' (line 19)
    config_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 19)
    add_subpackage_124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 4), config_123, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 19)
    add_subpackage_call_result_127 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), add_subpackage_124, *[str_125], **kwargs_126)
    
    
    # Call to add_subpackage(...): (line 20)
    # Processing the call arguments (line 20)
    str_130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 26), 'str', 'signal')
    # Processing the call keyword arguments (line 20)
    kwargs_131 = {}
    # Getting the type of 'config' (line 20)
    config_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 20)
    add_subpackage_129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 4), config_128, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 20)
    add_subpackage_call_result_132 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), add_subpackage_129, *[str_130], **kwargs_131)
    
    
    # Call to add_subpackage(...): (line 21)
    # Processing the call arguments (line 21)
    str_135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 26), 'str', 'sparse')
    # Processing the call keyword arguments (line 21)
    kwargs_136 = {}
    # Getting the type of 'config' (line 21)
    config_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 21)
    add_subpackage_134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 4), config_133, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 21)
    add_subpackage_call_result_137 = invoke(stypy.reporting.localization.Localization(__file__, 21, 4), add_subpackage_134, *[str_135], **kwargs_136)
    
    
    # Call to add_subpackage(...): (line 22)
    # Processing the call arguments (line 22)
    str_140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 26), 'str', 'spatial')
    # Processing the call keyword arguments (line 22)
    kwargs_141 = {}
    # Getting the type of 'config' (line 22)
    config_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 22)
    add_subpackage_139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 4), config_138, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 22)
    add_subpackage_call_result_142 = invoke(stypy.reporting.localization.Localization(__file__, 22, 4), add_subpackage_139, *[str_140], **kwargs_141)
    
    
    # Call to add_subpackage(...): (line 23)
    # Processing the call arguments (line 23)
    str_145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 26), 'str', 'special')
    # Processing the call keyword arguments (line 23)
    kwargs_146 = {}
    # Getting the type of 'config' (line 23)
    config_143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 23)
    add_subpackage_144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 4), config_143, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 23)
    add_subpackage_call_result_147 = invoke(stypy.reporting.localization.Localization(__file__, 23, 4), add_subpackage_144, *[str_145], **kwargs_146)
    
    
    # Call to add_subpackage(...): (line 24)
    # Processing the call arguments (line 24)
    str_150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 26), 'str', 'stats')
    # Processing the call keyword arguments (line 24)
    kwargs_151 = {}
    # Getting the type of 'config' (line 24)
    config_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 24)
    add_subpackage_149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 4), config_148, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 24)
    add_subpackage_call_result_152 = invoke(stypy.reporting.localization.Localization(__file__, 24, 4), add_subpackage_149, *[str_150], **kwargs_151)
    
    
    # Call to add_subpackage(...): (line 25)
    # Processing the call arguments (line 25)
    str_155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 26), 'str', 'ndimage')
    # Processing the call keyword arguments (line 25)
    kwargs_156 = {}
    # Getting the type of 'config' (line 25)
    config_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 25)
    add_subpackage_154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 4), config_153, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 25)
    add_subpackage_call_result_157 = invoke(stypy.reporting.localization.Localization(__file__, 25, 4), add_subpackage_154, *[str_155], **kwargs_156)
    
    
    # Call to add_subpackage(...): (line 26)
    # Processing the call arguments (line 26)
    str_160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 26), 'str', '_build_utils')
    # Processing the call keyword arguments (line 26)
    kwargs_161 = {}
    # Getting the type of 'config' (line 26)
    config_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 26)
    add_subpackage_159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 4), config_158, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 26)
    add_subpackage_call_result_162 = invoke(stypy.reporting.localization.Localization(__file__, 26, 4), add_subpackage_159, *[str_160], **kwargs_161)
    
    
    # Call to add_subpackage(...): (line 27)
    # Processing the call arguments (line 27)
    str_165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 26), 'str', '_lib')
    # Processing the call keyword arguments (line 27)
    kwargs_166 = {}
    # Getting the type of 'config' (line 27)
    config_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 27)
    add_subpackage_164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 4), config_163, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 27)
    add_subpackage_call_result_167 = invoke(stypy.reporting.localization.Localization(__file__, 27, 4), add_subpackage_164, *[str_165], **kwargs_166)
    
    
    # Call to make_config_py(...): (line 28)
    # Processing the call keyword arguments (line 28)
    kwargs_170 = {}
    # Getting the type of 'config' (line 28)
    config_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'config', False)
    # Obtaining the member 'make_config_py' of a type (line 28)
    make_config_py_169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 4), config_168, 'make_config_py')
    # Calling make_config_py(args, kwargs) (line 28)
    make_config_py_call_result_171 = invoke(stypy.reporting.localization.Localization(__file__, 28, 4), make_config_py_169, *[], **kwargs_170)
    
    # Getting the type of 'config' (line 29)
    config_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type', config_172)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_173)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_173

# Assigning a type to the variable 'configuration' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 32)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/')
    import_174 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 32, 4), 'numpy.distutils.core')

    if (type(import_174) is not StypyTypeError):

        if (import_174 != 'pyd_module'):
            __import__(import_174)
            sys_modules_175 = sys.modules[import_174]
            import_from_module(stypy.reporting.localization.Localization(__file__, 32, 4), 'numpy.distutils.core', sys_modules_175.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 32, 4), __file__, sys_modules_175, sys_modules_175.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 32, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'numpy.distutils.core', import_174)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/')
    
    
    # Call to setup(...): (line 33)
    # Processing the call keyword arguments (line 33)
    
    # Call to todict(...): (line 33)
    # Processing the call keyword arguments (line 33)
    kwargs_183 = {}
    
    # Call to configuration(...): (line 33)
    # Processing the call keyword arguments (line 33)
    str_178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 35), 'str', '')
    keyword_179 = str_178
    kwargs_180 = {'top_path': keyword_179}
    # Getting the type of 'configuration' (line 33)
    configuration_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 33)
    configuration_call_result_181 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), configuration_177, *[], **kwargs_180)
    
    # Obtaining the member 'todict' of a type (line 33)
    todict_182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), configuration_call_result_181, 'todict')
    # Calling todict(args, kwargs) (line 33)
    todict_call_result_184 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), todict_182, *[], **kwargs_183)
    
    kwargs_185 = {'todict_call_result_184': todict_call_result_184}
    # Getting the type of 'setup' (line 33)
    setup_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 33)
    setup_call_result_186 = invoke(stypy.reporting.localization.Localization(__file__, 33, 4), setup_176, *[], **kwargs_185)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
