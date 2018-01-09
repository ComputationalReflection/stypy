
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import platform
4: 
5: from distutils.unixccompiler import UnixCCompiler
6: from numpy.distutils.exec_command import find_executable
7: from numpy.distutils.ccompiler import simple_version_match
8: if platform.system() == 'Windows':
9:     from numpy.distutils.msvc9compiler import MSVCCompiler
10: 
11: 
12: class IntelCCompiler(UnixCCompiler):
13:     '''A modified Intel compiler compatible with a GCC-built Python.'''
14:     compiler_type = 'intel'
15:     cc_exe = 'icc'
16:     cc_args = 'fPIC'
17: 
18:     def __init__(self, verbose=0, dry_run=0, force=0):
19:         UnixCCompiler.__init__(self, verbose, dry_run, force)
20:         self.cc_exe = ('icc -fPIC -fp-model strict -O3 '
21:                        '-fomit-frame-pointer -openmp')
22:         compiler = self.cc_exe
23:         if platform.system() == 'Darwin':
24:             shared_flag = '-Wl,-undefined,dynamic_lookup'
25:         else:
26:             shared_flag = '-shared'
27:         self.set_executables(compiler=compiler,
28:                              compiler_so=compiler,
29:                              compiler_cxx=compiler,
30:                              archiver='xiar' + ' cru',
31:                              linker_exe=compiler + ' -shared-intel',
32:                              linker_so=compiler + ' ' + shared_flag +
33:                              ' -shared-intel')
34: 
35: 
36: class IntelItaniumCCompiler(IntelCCompiler):
37:     compiler_type = 'intele'
38: 
39:     # On Itanium, the Intel Compiler used to be called ecc, let's search for
40:     # it (now it's also icc, so ecc is last in the search).
41:     for cc_exe in map(find_executable, ['icc', 'ecc']):
42:         if cc_exe:
43:             break
44: 
45: 
46: class IntelEM64TCCompiler(UnixCCompiler):
47:     '''
48:     A modified Intel x86_64 compiler compatible with a 64bit GCC-built Python.
49:     '''
50:     compiler_type = 'intelem'
51:     cc_exe = 'icc -m64'
52:     cc_args = '-fPIC'
53: 
54:     def __init__(self, verbose=0, dry_run=0, force=0):
55:         UnixCCompiler.__init__(self, verbose, dry_run, force)
56:         self.cc_exe = ('icc -m64 -fPIC -fp-model strict -O3 '
57:                        '-fomit-frame-pointer -openmp -xSSE4.2')
58:         compiler = self.cc_exe
59:         if platform.system() == 'Darwin':
60:             shared_flag = '-Wl,-undefined,dynamic_lookup'
61:         else:
62:             shared_flag = '-shared'
63:         self.set_executables(compiler=compiler,
64:                              compiler_so=compiler,
65:                              compiler_cxx=compiler,
66:                              archiver='xiar' + ' cru',
67:                              linker_exe=compiler + ' -shared-intel',
68:                              linker_so=compiler + ' ' + shared_flag +
69:                              ' -shared-intel')
70: 
71: 
72: if platform.system() == 'Windows':
73:     class IntelCCompilerW(MSVCCompiler):
74:         '''
75:         A modified Intel compiler compatible with an MSVC-built Python.
76:         '''
77:         compiler_type = 'intelw'
78:         compiler_cxx = 'icl'
79: 
80:         def __init__(self, verbose=0, dry_run=0, force=0):
81:             MSVCCompiler.__init__(self, verbose, dry_run, force)
82:             version_match = simple_version_match(start='Intel\(R\).*?32,')
83:             self.__version = version_match
84: 
85:         def initialize(self, plat_name=None):
86:             MSVCCompiler.initialize(self, plat_name)
87:             self.cc = self.find_exe('icl.exe')
88:             self.lib = self.find_exe('xilib')
89:             self.linker = self.find_exe('xilink')
90:             self.compile_options = ['/nologo', '/O3', '/MD', '/W3',
91:                                     '/Qstd=c99', '/QaxSSE4.2']
92:             self.compile_options_debug = ['/nologo', '/Od', '/MDd', '/W3',
93:                                           '/Qstd=c99', '/Z7', '/D_DEBUG']
94: 
95:     class IntelEM64TCCompilerW(IntelCCompilerW):
96:         '''
97:         A modified Intel x86_64 compiler compatible with
98:         a 64bit MSVC-built Python.
99:         '''
100:         compiler_type = 'intelemw'
101: 
102:         def __init__(self, verbose=0, dry_run=0, force=0):
103:             MSVCCompiler.__init__(self, verbose, dry_run, force)
104:             version_match = simple_version_match(start='Intel\(R\).*?64,')
105:             self.__version = version_match
106: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import platform' statement (line 3)
import platform

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'platform', platform, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from distutils.unixccompiler import UnixCCompiler' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_35806 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.unixccompiler')

if (type(import_35806) is not StypyTypeError):

    if (import_35806 != 'pyd_module'):
        __import__(import_35806)
        sys_modules_35807 = sys.modules[import_35806]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.unixccompiler', sys_modules_35807.module_type_store, module_type_store, ['UnixCCompiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_35807, sys_modules_35807.module_type_store, module_type_store)
    else:
        from distutils.unixccompiler import UnixCCompiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.unixccompiler', None, module_type_store, ['UnixCCompiler'], [UnixCCompiler])

else:
    # Assigning a type to the variable 'distutils.unixccompiler' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.unixccompiler', import_35806)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.distutils.exec_command import find_executable' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_35808 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.distutils.exec_command')

if (type(import_35808) is not StypyTypeError):

    if (import_35808 != 'pyd_module'):
        __import__(import_35808)
        sys_modules_35809 = sys.modules[import_35808]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.distutils.exec_command', sys_modules_35809.module_type_store, module_type_store, ['find_executable'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_35809, sys_modules_35809.module_type_store, module_type_store)
    else:
        from numpy.distutils.exec_command import find_executable

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.distutils.exec_command', None, module_type_store, ['find_executable'], [find_executable])

else:
    # Assigning a type to the variable 'numpy.distutils.exec_command' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.distutils.exec_command', import_35808)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.distutils.ccompiler import simple_version_match' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_35810 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.distutils.ccompiler')

if (type(import_35810) is not StypyTypeError):

    if (import_35810 != 'pyd_module'):
        __import__(import_35810)
        sys_modules_35811 = sys.modules[import_35810]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.distutils.ccompiler', sys_modules_35811.module_type_store, module_type_store, ['simple_version_match'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_35811, sys_modules_35811.module_type_store, module_type_store)
    else:
        from numpy.distutils.ccompiler import simple_version_match

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.distutils.ccompiler', None, module_type_store, ['simple_version_match'], [simple_version_match])

else:
    # Assigning a type to the variable 'numpy.distutils.ccompiler' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.distutils.ccompiler', import_35810)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')




# Call to system(...): (line 8)
# Processing the call keyword arguments (line 8)
kwargs_35814 = {}
# Getting the type of 'platform' (line 8)
platform_35812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 3), 'platform', False)
# Obtaining the member 'system' of a type (line 8)
system_35813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 3), platform_35812, 'system')
# Calling system(args, kwargs) (line 8)
system_call_result_35815 = invoke(stypy.reporting.localization.Localization(__file__, 8, 3), system_35813, *[], **kwargs_35814)

str_35816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 24), 'str', 'Windows')
# Applying the binary operator '==' (line 8)
result_eq_35817 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 3), '==', system_call_result_35815, str_35816)

# Testing the type of an if condition (line 8)
if_condition_35818 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 8, 0), result_eq_35817)
# Assigning a type to the variable 'if_condition_35818' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'if_condition_35818', if_condition_35818)
# SSA begins for if statement (line 8)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 4))

# 'from numpy.distutils.msvc9compiler import MSVCCompiler' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_35819 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.msvc9compiler')

if (type(import_35819) is not StypyTypeError):

    if (import_35819 != 'pyd_module'):
        __import__(import_35819)
        sys_modules_35820 = sys.modules[import_35819]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.msvc9compiler', sys_modules_35820.module_type_store, module_type_store, ['MSVCCompiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 4), __file__, sys_modules_35820, sys_modules_35820.module_type_store, module_type_store)
    else:
        from numpy.distutils.msvc9compiler import MSVCCompiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.msvc9compiler', None, module_type_store, ['MSVCCompiler'], [MSVCCompiler])

else:
    # Assigning a type to the variable 'numpy.distutils.msvc9compiler' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.msvc9compiler', import_35819)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

# SSA join for if statement (line 8)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'IntelCCompiler' class
# Getting the type of 'UnixCCompiler' (line 12)
UnixCCompiler_35821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 21), 'UnixCCompiler')

class IntelCCompiler(UnixCCompiler_35821, ):
    str_35822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 4), 'str', 'A modified Intel compiler compatible with a GCC-built Python.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_35823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 31), 'int')
        int_35824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 42), 'int')
        int_35825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 51), 'int')
        defaults = [int_35823, int_35824, int_35825]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 18, 4, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelCCompiler.__init__', ['verbose', 'dry_run', 'force'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['verbose', 'dry_run', 'force'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'self' (line 19)
        self_35828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 31), 'self', False)
        # Getting the type of 'verbose' (line 19)
        verbose_35829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 37), 'verbose', False)
        # Getting the type of 'dry_run' (line 19)
        dry_run_35830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 46), 'dry_run', False)
        # Getting the type of 'force' (line 19)
        force_35831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 55), 'force', False)
        # Processing the call keyword arguments (line 19)
        kwargs_35832 = {}
        # Getting the type of 'UnixCCompiler' (line 19)
        UnixCCompiler_35826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'UnixCCompiler', False)
        # Obtaining the member '__init__' of a type (line 19)
        init___35827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 8), UnixCCompiler_35826, '__init__')
        # Calling __init__(args, kwargs) (line 19)
        init___call_result_35833 = invoke(stypy.reporting.localization.Localization(__file__, 19, 8), init___35827, *[self_35828, verbose_35829, dry_run_35830, force_35831], **kwargs_35832)
        
        
        # Assigning a Str to a Attribute (line 20):
        str_35834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 23), 'str', 'icc -fPIC -fp-model strict -O3 -fomit-frame-pointer -openmp')
        # Getting the type of 'self' (line 20)
        self_35835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'self')
        # Setting the type of the member 'cc_exe' of a type (line 20)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), self_35835, 'cc_exe', str_35834)
        
        # Assigning a Attribute to a Name (line 22):
        # Getting the type of 'self' (line 22)
        self_35836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 19), 'self')
        # Obtaining the member 'cc_exe' of a type (line 22)
        cc_exe_35837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 19), self_35836, 'cc_exe')
        # Assigning a type to the variable 'compiler' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'compiler', cc_exe_35837)
        
        
        
        # Call to system(...): (line 23)
        # Processing the call keyword arguments (line 23)
        kwargs_35840 = {}
        # Getting the type of 'platform' (line 23)
        platform_35838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), 'platform', False)
        # Obtaining the member 'system' of a type (line 23)
        system_35839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 11), platform_35838, 'system')
        # Calling system(args, kwargs) (line 23)
        system_call_result_35841 = invoke(stypy.reporting.localization.Localization(__file__, 23, 11), system_35839, *[], **kwargs_35840)
        
        str_35842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 32), 'str', 'Darwin')
        # Applying the binary operator '==' (line 23)
        result_eq_35843 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 11), '==', system_call_result_35841, str_35842)
        
        # Testing the type of an if condition (line 23)
        if_condition_35844 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 23, 8), result_eq_35843)
        # Assigning a type to the variable 'if_condition_35844' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'if_condition_35844', if_condition_35844)
        # SSA begins for if statement (line 23)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 24):
        str_35845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 26), 'str', '-Wl,-undefined,dynamic_lookup')
        # Assigning a type to the variable 'shared_flag' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'shared_flag', str_35845)
        # SSA branch for the else part of an if statement (line 23)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 26):
        str_35846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 26), 'str', '-shared')
        # Assigning a type to the variable 'shared_flag' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'shared_flag', str_35846)
        # SSA join for if statement (line 23)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_executables(...): (line 27)
        # Processing the call keyword arguments (line 27)
        # Getting the type of 'compiler' (line 27)
        compiler_35849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 38), 'compiler', False)
        keyword_35850 = compiler_35849
        # Getting the type of 'compiler' (line 28)
        compiler_35851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 41), 'compiler', False)
        keyword_35852 = compiler_35851
        # Getting the type of 'compiler' (line 29)
        compiler_35853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 42), 'compiler', False)
        keyword_35854 = compiler_35853
        str_35855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 38), 'str', 'xiar')
        str_35856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 47), 'str', ' cru')
        # Applying the binary operator '+' (line 30)
        result_add_35857 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 38), '+', str_35855, str_35856)
        
        keyword_35858 = result_add_35857
        # Getting the type of 'compiler' (line 31)
        compiler_35859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 40), 'compiler', False)
        str_35860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 51), 'str', ' -shared-intel')
        # Applying the binary operator '+' (line 31)
        result_add_35861 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 40), '+', compiler_35859, str_35860)
        
        keyword_35862 = result_add_35861
        # Getting the type of 'compiler' (line 32)
        compiler_35863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 39), 'compiler', False)
        str_35864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 50), 'str', ' ')
        # Applying the binary operator '+' (line 32)
        result_add_35865 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 39), '+', compiler_35863, str_35864)
        
        # Getting the type of 'shared_flag' (line 32)
        shared_flag_35866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 56), 'shared_flag', False)
        # Applying the binary operator '+' (line 32)
        result_add_35867 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 54), '+', result_add_35865, shared_flag_35866)
        
        str_35868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 29), 'str', ' -shared-intel')
        # Applying the binary operator '+' (line 32)
        result_add_35869 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 68), '+', result_add_35867, str_35868)
        
        keyword_35870 = result_add_35869
        kwargs_35871 = {'compiler_cxx': keyword_35854, 'linker_exe': keyword_35862, 'compiler_so': keyword_35852, 'archiver': keyword_35858, 'linker_so': keyword_35870, 'compiler': keyword_35850}
        # Getting the type of 'self' (line 27)
        self_35847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'self', False)
        # Obtaining the member 'set_executables' of a type (line 27)
        set_executables_35848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), self_35847, 'set_executables')
        # Calling set_executables(args, kwargs) (line 27)
        set_executables_call_result_35872 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), set_executables_35848, *[], **kwargs_35871)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'IntelCCompiler' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'IntelCCompiler', IntelCCompiler)

# Assigning a Str to a Name (line 14):
str_35873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 20), 'str', 'intel')
# Getting the type of 'IntelCCompiler'
IntelCCompiler_35874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelCCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelCCompiler_35874, 'compiler_type', str_35873)

# Assigning a Str to a Name (line 15):
str_35875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 13), 'str', 'icc')
# Getting the type of 'IntelCCompiler'
IntelCCompiler_35876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelCCompiler')
# Setting the type of the member 'cc_exe' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelCCompiler_35876, 'cc_exe', str_35875)

# Assigning a Str to a Name (line 16):
str_35877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 14), 'str', 'fPIC')
# Getting the type of 'IntelCCompiler'
IntelCCompiler_35878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelCCompiler')
# Setting the type of the member 'cc_args' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelCCompiler_35878, 'cc_args', str_35877)
# Declaration of the 'IntelItaniumCCompiler' class
# Getting the type of 'IntelCCompiler' (line 36)
IntelCCompiler_35879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 28), 'IntelCCompiler')

class IntelItaniumCCompiler(IntelCCompiler_35879, ):
    
    
    # Call to map(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'find_executable' (line 41)
    find_executable_35881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 22), 'find_executable', False)
    
    # Obtaining an instance of the builtin type 'list' (line 41)
    list_35882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 41)
    # Adding element type (line 41)
    str_35883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 40), 'str', 'icc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 39), list_35882, str_35883)
    # Adding element type (line 41)
    str_35884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 47), 'str', 'ecc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 39), list_35882, str_35884)
    
    # Processing the call keyword arguments (line 41)
    kwargs_35885 = {}
    # Getting the type of 'map' (line 41)
    map_35880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 18), 'map', False)
    # Calling map(args, kwargs) (line 41)
    map_call_result_35886 = invoke(stypy.reporting.localization.Localization(__file__, 41, 18), map_35880, *[find_executable_35881, list_35882], **kwargs_35885)
    
    # Testing the type of a for loop iterable (line 41)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 41, 4), map_call_result_35886)
    # Getting the type of the for loop variable (line 41)
    for_loop_var_35887 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 41, 4), map_call_result_35886)
    # Assigning a type to the variable 'cc_exe' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'cc_exe', for_loop_var_35887)
    # SSA begins for a for statement (line 41)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'cc_exe' (line 42)
    cc_exe_35888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'cc_exe')
    # Testing the type of an if condition (line 42)
    if_condition_35889 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 8), cc_exe_35888)
    # Assigning a type to the variable 'if_condition_35889' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'if_condition_35889', if_condition_35889)
    # SSA begins for if statement (line 42)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 42)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 36, 0, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelItaniumCCompiler.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'IntelItaniumCCompiler' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'IntelItaniumCCompiler', IntelItaniumCCompiler)

# Assigning a Str to a Name (line 37):
str_35890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 20), 'str', 'intele')
# Getting the type of 'IntelItaniumCCompiler'
IntelItaniumCCompiler_35891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelItaniumCCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelItaniumCCompiler_35891, 'compiler_type', str_35890)
# Declaration of the 'IntelEM64TCCompiler' class
# Getting the type of 'UnixCCompiler' (line 46)
UnixCCompiler_35892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 26), 'UnixCCompiler')

class IntelEM64TCCompiler(UnixCCompiler_35892, ):
    str_35893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, (-1)), 'str', '\n    A modified Intel x86_64 compiler compatible with a 64bit GCC-built Python.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_35894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 31), 'int')
        int_35895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 42), 'int')
        int_35896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 51), 'int')
        defaults = [int_35894, int_35895, int_35896]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 54, 4, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelEM64TCCompiler.__init__', ['verbose', 'dry_run', 'force'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['verbose', 'dry_run', 'force'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'self' (line 55)
        self_35899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 31), 'self', False)
        # Getting the type of 'verbose' (line 55)
        verbose_35900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 37), 'verbose', False)
        # Getting the type of 'dry_run' (line 55)
        dry_run_35901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 46), 'dry_run', False)
        # Getting the type of 'force' (line 55)
        force_35902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 55), 'force', False)
        # Processing the call keyword arguments (line 55)
        kwargs_35903 = {}
        # Getting the type of 'UnixCCompiler' (line 55)
        UnixCCompiler_35897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'UnixCCompiler', False)
        # Obtaining the member '__init__' of a type (line 55)
        init___35898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), UnixCCompiler_35897, '__init__')
        # Calling __init__(args, kwargs) (line 55)
        init___call_result_35904 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), init___35898, *[self_35899, verbose_35900, dry_run_35901, force_35902], **kwargs_35903)
        
        
        # Assigning a Str to a Attribute (line 56):
        str_35905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 23), 'str', 'icc -m64 -fPIC -fp-model strict -O3 -fomit-frame-pointer -openmp -xSSE4.2')
        # Getting the type of 'self' (line 56)
        self_35906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self')
        # Setting the type of the member 'cc_exe' of a type (line 56)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), self_35906, 'cc_exe', str_35905)
        
        # Assigning a Attribute to a Name (line 58):
        # Getting the type of 'self' (line 58)
        self_35907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 19), 'self')
        # Obtaining the member 'cc_exe' of a type (line 58)
        cc_exe_35908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 19), self_35907, 'cc_exe')
        # Assigning a type to the variable 'compiler' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'compiler', cc_exe_35908)
        
        
        
        # Call to system(...): (line 59)
        # Processing the call keyword arguments (line 59)
        kwargs_35911 = {}
        # Getting the type of 'platform' (line 59)
        platform_35909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 11), 'platform', False)
        # Obtaining the member 'system' of a type (line 59)
        system_35910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 11), platform_35909, 'system')
        # Calling system(args, kwargs) (line 59)
        system_call_result_35912 = invoke(stypy.reporting.localization.Localization(__file__, 59, 11), system_35910, *[], **kwargs_35911)
        
        str_35913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 32), 'str', 'Darwin')
        # Applying the binary operator '==' (line 59)
        result_eq_35914 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 11), '==', system_call_result_35912, str_35913)
        
        # Testing the type of an if condition (line 59)
        if_condition_35915 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 8), result_eq_35914)
        # Assigning a type to the variable 'if_condition_35915' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'if_condition_35915', if_condition_35915)
        # SSA begins for if statement (line 59)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 60):
        str_35916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 26), 'str', '-Wl,-undefined,dynamic_lookup')
        # Assigning a type to the variable 'shared_flag' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'shared_flag', str_35916)
        # SSA branch for the else part of an if statement (line 59)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 62):
        str_35917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 26), 'str', '-shared')
        # Assigning a type to the variable 'shared_flag' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'shared_flag', str_35917)
        # SSA join for if statement (line 59)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_executables(...): (line 63)
        # Processing the call keyword arguments (line 63)
        # Getting the type of 'compiler' (line 63)
        compiler_35920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 38), 'compiler', False)
        keyword_35921 = compiler_35920
        # Getting the type of 'compiler' (line 64)
        compiler_35922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 41), 'compiler', False)
        keyword_35923 = compiler_35922
        # Getting the type of 'compiler' (line 65)
        compiler_35924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 42), 'compiler', False)
        keyword_35925 = compiler_35924
        str_35926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 38), 'str', 'xiar')
        str_35927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 47), 'str', ' cru')
        # Applying the binary operator '+' (line 66)
        result_add_35928 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 38), '+', str_35926, str_35927)
        
        keyword_35929 = result_add_35928
        # Getting the type of 'compiler' (line 67)
        compiler_35930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 40), 'compiler', False)
        str_35931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 51), 'str', ' -shared-intel')
        # Applying the binary operator '+' (line 67)
        result_add_35932 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 40), '+', compiler_35930, str_35931)
        
        keyword_35933 = result_add_35932
        # Getting the type of 'compiler' (line 68)
        compiler_35934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 39), 'compiler', False)
        str_35935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 50), 'str', ' ')
        # Applying the binary operator '+' (line 68)
        result_add_35936 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 39), '+', compiler_35934, str_35935)
        
        # Getting the type of 'shared_flag' (line 68)
        shared_flag_35937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 56), 'shared_flag', False)
        # Applying the binary operator '+' (line 68)
        result_add_35938 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 54), '+', result_add_35936, shared_flag_35937)
        
        str_35939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 29), 'str', ' -shared-intel')
        # Applying the binary operator '+' (line 68)
        result_add_35940 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 68), '+', result_add_35938, str_35939)
        
        keyword_35941 = result_add_35940
        kwargs_35942 = {'compiler_cxx': keyword_35925, 'linker_exe': keyword_35933, 'compiler_so': keyword_35923, 'archiver': keyword_35929, 'linker_so': keyword_35941, 'compiler': keyword_35921}
        # Getting the type of 'self' (line 63)
        self_35918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self', False)
        # Obtaining the member 'set_executables' of a type (line 63)
        set_executables_35919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_35918, 'set_executables')
        # Calling set_executables(args, kwargs) (line 63)
        set_executables_call_result_35943 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), set_executables_35919, *[], **kwargs_35942)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'IntelEM64TCCompiler' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'IntelEM64TCCompiler', IntelEM64TCCompiler)

# Assigning a Str to a Name (line 50):
str_35944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 20), 'str', 'intelem')
# Getting the type of 'IntelEM64TCCompiler'
IntelEM64TCCompiler_35945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelEM64TCCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelEM64TCCompiler_35945, 'compiler_type', str_35944)

# Assigning a Str to a Name (line 51):
str_35946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 13), 'str', 'icc -m64')
# Getting the type of 'IntelEM64TCCompiler'
IntelEM64TCCompiler_35947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelEM64TCCompiler')
# Setting the type of the member 'cc_exe' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelEM64TCCompiler_35947, 'cc_exe', str_35946)

# Assigning a Str to a Name (line 52):
str_35948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 14), 'str', '-fPIC')
# Getting the type of 'IntelEM64TCCompiler'
IntelEM64TCCompiler_35949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelEM64TCCompiler')
# Setting the type of the member 'cc_args' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelEM64TCCompiler_35949, 'cc_args', str_35948)



# Call to system(...): (line 72)
# Processing the call keyword arguments (line 72)
kwargs_35952 = {}
# Getting the type of 'platform' (line 72)
platform_35950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 3), 'platform', False)
# Obtaining the member 'system' of a type (line 72)
system_35951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 3), platform_35950, 'system')
# Calling system(args, kwargs) (line 72)
system_call_result_35953 = invoke(stypy.reporting.localization.Localization(__file__, 72, 3), system_35951, *[], **kwargs_35952)

str_35954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 24), 'str', 'Windows')
# Applying the binary operator '==' (line 72)
result_eq_35955 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 3), '==', system_call_result_35953, str_35954)

# Testing the type of an if condition (line 72)
if_condition_35956 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 0), result_eq_35955)
# Assigning a type to the variable 'if_condition_35956' (line 72)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'if_condition_35956', if_condition_35956)
# SSA begins for if statement (line 72)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
# Declaration of the 'IntelCCompilerW' class
# Getting the type of 'MSVCCompiler' (line 73)
MSVCCompiler_35957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 26), 'MSVCCompiler')

class IntelCCompilerW(MSVCCompiler_35957, ):
    str_35958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, (-1)), 'str', '\n        A modified Intel compiler compatible with an MSVC-built Python.\n        ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_35959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 35), 'int')
        int_35960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 46), 'int')
        int_35961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 55), 'int')
        defaults = [int_35959, int_35960, int_35961]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 80, 8, False)
        # Assigning a type to the variable 'self' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelCCompilerW.__init__', ['verbose', 'dry_run', 'force'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['verbose', 'dry_run', 'force'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'self' (line 81)
        self_35964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 34), 'self', False)
        # Getting the type of 'verbose' (line 81)
        verbose_35965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 40), 'verbose', False)
        # Getting the type of 'dry_run' (line 81)
        dry_run_35966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 49), 'dry_run', False)
        # Getting the type of 'force' (line 81)
        force_35967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 58), 'force', False)
        # Processing the call keyword arguments (line 81)
        kwargs_35968 = {}
        # Getting the type of 'MSVCCompiler' (line 81)
        MSVCCompiler_35962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'MSVCCompiler', False)
        # Obtaining the member '__init__' of a type (line 81)
        init___35963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), MSVCCompiler_35962, '__init__')
        # Calling __init__(args, kwargs) (line 81)
        init___call_result_35969 = invoke(stypy.reporting.localization.Localization(__file__, 81, 12), init___35963, *[self_35964, verbose_35965, dry_run_35966, force_35967], **kwargs_35968)
        
        
        # Assigning a Call to a Name (line 82):
        
        # Call to simple_version_match(...): (line 82)
        # Processing the call keyword arguments (line 82)
        str_35971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 55), 'str', 'Intel\\(R\\).*?32,')
        keyword_35972 = str_35971
        kwargs_35973 = {'start': keyword_35972}
        # Getting the type of 'simple_version_match' (line 82)
        simple_version_match_35970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 28), 'simple_version_match', False)
        # Calling simple_version_match(args, kwargs) (line 82)
        simple_version_match_call_result_35974 = invoke(stypy.reporting.localization.Localization(__file__, 82, 28), simple_version_match_35970, *[], **kwargs_35973)
        
        # Assigning a type to the variable 'version_match' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'version_match', simple_version_match_call_result_35974)
        
        # Assigning a Name to a Attribute (line 83):
        # Getting the type of 'version_match' (line 83)
        version_match_35975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 29), 'version_match')
        # Getting the type of 'self' (line 83)
        self_35976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'self')
        # Setting the type of the member '__version' of a type (line 83)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 12), self_35976, '__version', version_match_35975)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def initialize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 85)
        None_35977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 39), 'None')
        defaults = [None_35977]
        # Create a new context for function 'initialize'
        module_type_store = module_type_store.open_function_context('initialize', 85, 8, False)
        # Assigning a type to the variable 'self' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        IntelCCompilerW.initialize.__dict__.__setitem__('stypy_localization', localization)
        IntelCCompilerW.initialize.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntelCCompilerW.initialize.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntelCCompilerW.initialize.__dict__.__setitem__('stypy_function_name', 'IntelCCompilerW.initialize')
        IntelCCompilerW.initialize.__dict__.__setitem__('stypy_param_names_list', ['plat_name'])
        IntelCCompilerW.initialize.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntelCCompilerW.initialize.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntelCCompilerW.initialize.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntelCCompilerW.initialize.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntelCCompilerW.initialize.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntelCCompilerW.initialize.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelCCompilerW.initialize', ['plat_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'initialize', localization, ['plat_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'initialize(...)' code ##################

        
        # Call to initialize(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'self' (line 86)
        self_35980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 36), 'self', False)
        # Getting the type of 'plat_name' (line 86)
        plat_name_35981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 42), 'plat_name', False)
        # Processing the call keyword arguments (line 86)
        kwargs_35982 = {}
        # Getting the type of 'MSVCCompiler' (line 86)
        MSVCCompiler_35978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'MSVCCompiler', False)
        # Obtaining the member 'initialize' of a type (line 86)
        initialize_35979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), MSVCCompiler_35978, 'initialize')
        # Calling initialize(args, kwargs) (line 86)
        initialize_call_result_35983 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), initialize_35979, *[self_35980, plat_name_35981], **kwargs_35982)
        
        
        # Assigning a Call to a Attribute (line 87):
        
        # Call to find_exe(...): (line 87)
        # Processing the call arguments (line 87)
        str_35986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 36), 'str', 'icl.exe')
        # Processing the call keyword arguments (line 87)
        kwargs_35987 = {}
        # Getting the type of 'self' (line 87)
        self_35984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 22), 'self', False)
        # Obtaining the member 'find_exe' of a type (line 87)
        find_exe_35985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 22), self_35984, 'find_exe')
        # Calling find_exe(args, kwargs) (line 87)
        find_exe_call_result_35988 = invoke(stypy.reporting.localization.Localization(__file__, 87, 22), find_exe_35985, *[str_35986], **kwargs_35987)
        
        # Getting the type of 'self' (line 87)
        self_35989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'self')
        # Setting the type of the member 'cc' of a type (line 87)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 12), self_35989, 'cc', find_exe_call_result_35988)
        
        # Assigning a Call to a Attribute (line 88):
        
        # Call to find_exe(...): (line 88)
        # Processing the call arguments (line 88)
        str_35992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 37), 'str', 'xilib')
        # Processing the call keyword arguments (line 88)
        kwargs_35993 = {}
        # Getting the type of 'self' (line 88)
        self_35990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 23), 'self', False)
        # Obtaining the member 'find_exe' of a type (line 88)
        find_exe_35991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 23), self_35990, 'find_exe')
        # Calling find_exe(args, kwargs) (line 88)
        find_exe_call_result_35994 = invoke(stypy.reporting.localization.Localization(__file__, 88, 23), find_exe_35991, *[str_35992], **kwargs_35993)
        
        # Getting the type of 'self' (line 88)
        self_35995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'self')
        # Setting the type of the member 'lib' of a type (line 88)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 12), self_35995, 'lib', find_exe_call_result_35994)
        
        # Assigning a Call to a Attribute (line 89):
        
        # Call to find_exe(...): (line 89)
        # Processing the call arguments (line 89)
        str_35998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 40), 'str', 'xilink')
        # Processing the call keyword arguments (line 89)
        kwargs_35999 = {}
        # Getting the type of 'self' (line 89)
        self_35996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'self', False)
        # Obtaining the member 'find_exe' of a type (line 89)
        find_exe_35997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 26), self_35996, 'find_exe')
        # Calling find_exe(args, kwargs) (line 89)
        find_exe_call_result_36000 = invoke(stypy.reporting.localization.Localization(__file__, 89, 26), find_exe_35997, *[str_35998], **kwargs_35999)
        
        # Getting the type of 'self' (line 89)
        self_36001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'self')
        # Setting the type of the member 'linker' of a type (line 89)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), self_36001, 'linker', find_exe_call_result_36000)
        
        # Assigning a List to a Attribute (line 90):
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_36002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        str_36003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 36), 'str', '/nologo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 35), list_36002, str_36003)
        # Adding element type (line 90)
        str_36004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 47), 'str', '/O3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 35), list_36002, str_36004)
        # Adding element type (line 90)
        str_36005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 54), 'str', '/MD')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 35), list_36002, str_36005)
        # Adding element type (line 90)
        str_36006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 61), 'str', '/W3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 35), list_36002, str_36006)
        # Adding element type (line 90)
        str_36007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 36), 'str', '/Qstd=c99')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 35), list_36002, str_36007)
        # Adding element type (line 90)
        str_36008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 49), 'str', '/QaxSSE4.2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 35), list_36002, str_36008)
        
        # Getting the type of 'self' (line 90)
        self_36009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'self')
        # Setting the type of the member 'compile_options' of a type (line 90)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), self_36009, 'compile_options', list_36002)
        
        # Assigning a List to a Attribute (line 92):
        
        # Obtaining an instance of the builtin type 'list' (line 92)
        list_36010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 92)
        # Adding element type (line 92)
        str_36011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 42), 'str', '/nologo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 41), list_36010, str_36011)
        # Adding element type (line 92)
        str_36012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 53), 'str', '/Od')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 41), list_36010, str_36012)
        # Adding element type (line 92)
        str_36013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 60), 'str', '/MDd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 41), list_36010, str_36013)
        # Adding element type (line 92)
        str_36014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 68), 'str', '/W3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 41), list_36010, str_36014)
        # Adding element type (line 92)
        str_36015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 42), 'str', '/Qstd=c99')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 41), list_36010, str_36015)
        # Adding element type (line 92)
        str_36016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 55), 'str', '/Z7')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 41), list_36010, str_36016)
        # Adding element type (line 92)
        str_36017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 62), 'str', '/D_DEBUG')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 41), list_36010, str_36017)
        
        # Getting the type of 'self' (line 92)
        self_36018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'self')
        # Setting the type of the member 'compile_options_debug' of a type (line 92)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), self_36018, 'compile_options_debug', list_36010)
        
        # ################# End of 'initialize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize' in the type store
        # Getting the type of 'stypy_return_type' (line 85)
        stypy_return_type_36019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36019)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize'
        return stypy_return_type_36019


# Assigning a type to the variable 'IntelCCompilerW' (line 73)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'IntelCCompilerW', IntelCCompilerW)

# Assigning a Str to a Name (line 77):
str_36020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 24), 'str', 'intelw')
# Getting the type of 'IntelCCompilerW'
IntelCCompilerW_36021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelCCompilerW')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelCCompilerW_36021, 'compiler_type', str_36020)

# Assigning a Str to a Name (line 78):
str_36022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 23), 'str', 'icl')
# Getting the type of 'IntelCCompilerW'
IntelCCompilerW_36023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelCCompilerW')
# Setting the type of the member 'compiler_cxx' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelCCompilerW_36023, 'compiler_cxx', str_36022)
# Declaration of the 'IntelEM64TCCompilerW' class
# Getting the type of 'IntelCCompilerW' (line 95)
IntelCCompilerW_36024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 31), 'IntelCCompilerW')

class IntelEM64TCCompilerW(IntelCCompilerW_36024, ):
    str_36025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, (-1)), 'str', '\n        A modified Intel x86_64 compiler compatible with\n        a 64bit MSVC-built Python.\n        ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_36026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 35), 'int')
        int_36027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 46), 'int')
        int_36028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 55), 'int')
        defaults = [int_36026, int_36027, int_36028]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 102, 8, False)
        # Assigning a type to the variable 'self' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelEM64TCCompilerW.__init__', ['verbose', 'dry_run', 'force'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['verbose', 'dry_run', 'force'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'self' (line 103)
        self_36031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 34), 'self', False)
        # Getting the type of 'verbose' (line 103)
        verbose_36032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 40), 'verbose', False)
        # Getting the type of 'dry_run' (line 103)
        dry_run_36033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 49), 'dry_run', False)
        # Getting the type of 'force' (line 103)
        force_36034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 58), 'force', False)
        # Processing the call keyword arguments (line 103)
        kwargs_36035 = {}
        # Getting the type of 'MSVCCompiler' (line 103)
        MSVCCompiler_36029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'MSVCCompiler', False)
        # Obtaining the member '__init__' of a type (line 103)
        init___36030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), MSVCCompiler_36029, '__init__')
        # Calling __init__(args, kwargs) (line 103)
        init___call_result_36036 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), init___36030, *[self_36031, verbose_36032, dry_run_36033, force_36034], **kwargs_36035)
        
        
        # Assigning a Call to a Name (line 104):
        
        # Call to simple_version_match(...): (line 104)
        # Processing the call keyword arguments (line 104)
        str_36038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 55), 'str', 'Intel\\(R\\).*?64,')
        keyword_36039 = str_36038
        kwargs_36040 = {'start': keyword_36039}
        # Getting the type of 'simple_version_match' (line 104)
        simple_version_match_36037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 28), 'simple_version_match', False)
        # Calling simple_version_match(args, kwargs) (line 104)
        simple_version_match_call_result_36041 = invoke(stypy.reporting.localization.Localization(__file__, 104, 28), simple_version_match_36037, *[], **kwargs_36040)
        
        # Assigning a type to the variable 'version_match' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'version_match', simple_version_match_call_result_36041)
        
        # Assigning a Name to a Attribute (line 105):
        # Getting the type of 'version_match' (line 105)
        version_match_36042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 29), 'version_match')
        # Getting the type of 'self' (line 105)
        self_36043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'self')
        # Setting the type of the member '__version' of a type (line 105)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), self_36043, '__version', version_match_36042)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'IntelEM64TCCompilerW' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'IntelEM64TCCompilerW', IntelEM64TCCompilerW)

# Assigning a Str to a Name (line 100):
str_36044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 24), 'str', 'intelemw')
# Getting the type of 'IntelEM64TCCompilerW'
IntelEM64TCCompilerW_36045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelEM64TCCompilerW')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelEM64TCCompilerW_36045, 'compiler_type', str_36044)
# SSA join for if statement (line 72)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
