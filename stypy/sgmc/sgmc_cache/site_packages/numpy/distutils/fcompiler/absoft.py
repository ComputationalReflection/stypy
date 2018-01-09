
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: # http://www.absoft.com/literature/osxuserguide.pdf
3: # http://www.absoft.com/documentation.html
4: 
5: # Notes:
6: # - when using -g77 then use -DUNDERSCORE_G77 to compile f2py
7: #   generated extension modules (works for f2py v2.45.241_1936 and up)
8: from __future__ import division, absolute_import, print_function
9: 
10: import os
11: 
12: from numpy.distutils.cpuinfo import cpu
13: from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file
14: from numpy.distutils.misc_util import cyg2win32
15: 
16: compilers = ['AbsoftFCompiler']
17: 
18: class AbsoftFCompiler(FCompiler):
19: 
20:     compiler_type = 'absoft'
21:     description = 'Absoft Corp Fortran Compiler'
22:     #version_pattern = r'FORTRAN 77 Compiler (?P<version>[^\s*,]*).*?Absoft Corp'
23:     version_pattern = r'(f90:.*?(Absoft Pro FORTRAN Version|FORTRAN 77 Compiler|Absoft Fortran Compiler Version|Copyright Absoft Corporation.*?Version))'+\
24:                        r' (?P<version>[^\s*,]*)(.*?Absoft Corp|)'
25: 
26:     # on windows: f90 -V -c dummy.f
27:     # f90: Copyright Absoft Corporation 1994-1998 mV2; Cray Research, Inc. 1994-1996 CF90 (2.x.x.x  f36t87) Version 2.3 Wed Apr 19, 2006  13:05:16
28: 
29:     # samt5735(8)$ f90 -V -c dummy.f
30:     # f90: Copyright Absoft Corporation 1994-2002; Absoft Pro FORTRAN Version 8.0
31:     # Note that fink installs g77 as f77, so need to use f90 for detection.
32: 
33:     executables = {
34:         'version_cmd'  : None,          # set by update_executables
35:         'compiler_f77' : ["f77"],
36:         'compiler_fix' : ["f90"],
37:         'compiler_f90' : ["f90"],
38:         'linker_so'    : ["<F90>"],
39:         'archiver'     : ["ar", "-cr"],
40:         'ranlib'       : ["ranlib"]
41:         }
42: 
43:     if os.name=='nt':
44:         library_switch = '/out:'      #No space after /out:!
45: 
46:     module_dir_switch = None
47:     module_include_switch = '-p'
48: 
49:     def update_executables(self):
50:         f = cyg2win32(dummy_fortran_file())
51:         self.executables['version_cmd'] = ['<F90>', '-V', '-c',
52:                                            f+'.f', '-o', f+'.o']
53: 
54:     def get_flags_linker_so(self):
55:         if os.name=='nt':
56:             opt = ['/dll']
57:         # The "-K shared" switches are being left in for pre-9.0 versions
58:         # of Absoft though I don't think versions earlier than 9 can
59:         # actually be used to build shared libraries.  In fact, version
60:         # 8 of Absoft doesn't recognize "-K shared" and will fail.
61:         elif self.get_version() >= '9.0':
62:             opt = ['-shared']
63:         else:
64:             opt = ["-K", "shared"]
65:         return opt
66: 
67:     def library_dir_option(self, dir):
68:         if os.name=='nt':
69:             return ['-link', '/PATH:"%s"' % (dir)]
70:         return "-L" + dir
71: 
72:     def library_option(self, lib):
73:         if os.name=='nt':
74:             return '%s.lib' % (lib)
75:         return "-l" + lib
76: 
77:     def get_library_dirs(self):
78:         opt = FCompiler.get_library_dirs(self)
79:         d = os.environ.get('ABSOFT')
80:         if d:
81:             if self.get_version() >= '10.0':
82:                 # use shared libraries, the static libraries were not compiled -fPIC
83:                 prefix = 'sh'
84:             else:
85:                 prefix = ''
86:             if cpu.is_64bit():
87:                 suffix = '64'
88:             else:
89:                 suffix = ''
90:             opt.append(os.path.join(d, '%slib%s' % (prefix, suffix)))
91:         return opt
92: 
93:     def get_libraries(self):
94:         opt = FCompiler.get_libraries(self)
95:         if self.get_version() >= '11.0':
96:             opt.extend(['af90math', 'afio', 'af77math', 'amisc'])
97:         elif self.get_version() >= '10.0':
98:             opt.extend(['af90math', 'afio', 'af77math', 'U77'])
99:         elif self.get_version() >= '8.0':
100:             opt.extend(['f90math', 'fio', 'f77math', 'U77'])
101:         else:
102:             opt.extend(['fio', 'f90math', 'fmath', 'U77'])
103:         if os.name =='nt':
104:             opt.append('COMDLG32')
105:         return opt
106: 
107:     def get_flags(self):
108:         opt = FCompiler.get_flags(self)
109:         if os.name != 'nt':
110:             opt.extend(['-s'])
111:             if self.get_version():
112:                 if self.get_version()>='8.2':
113:                     opt.append('-fpic')
114:         return opt
115: 
116:     def get_flags_f77(self):
117:         opt = FCompiler.get_flags_f77(self)
118:         opt.extend(['-N22', '-N90', '-N110'])
119:         v = self.get_version()
120:         if os.name == 'nt':
121:             if v and v>='8.0':
122:                 opt.extend(['-f', '-N15'])
123:         else:
124:             opt.append('-f')
125:             if v:
126:                 if v<='4.6':
127:                     opt.append('-B108')
128:                 else:
129:                     # Though -N15 is undocumented, it works with
130:                     # Absoft 8.0 on Linux
131:                     opt.append('-N15')
132:         return opt
133: 
134:     def get_flags_f90(self):
135:         opt = FCompiler.get_flags_f90(self)
136:         opt.extend(["-YCFRL=1", "-YCOM_NAMES=LCS", "-YCOM_PFX", "-YEXT_PFX",
137:                     "-YCOM_SFX=_", "-YEXT_SFX=_", "-YEXT_NAMES=LCS"])
138:         if self.get_version():
139:             if self.get_version()>'4.6':
140:                 opt.extend(["-YDEALLOC=ALL"])
141:         return opt
142: 
143:     def get_flags_fix(self):
144:         opt = FCompiler.get_flags_fix(self)
145:         opt.extend(["-YCFRL=1", "-YCOM_NAMES=LCS", "-YCOM_PFX", "-YEXT_PFX",
146:                     "-YCOM_SFX=_", "-YEXT_SFX=_", "-YEXT_NAMES=LCS"])
147:         opt.extend(["-f", "fixed"])
148:         return opt
149: 
150:     def get_flags_opt(self):
151:         opt = ['-O']
152:         return opt
153: 
154: if __name__ == '__main__':
155:     from distutils import log
156:     log.set_verbosity(2)
157:     from numpy.distutils.fcompiler import new_fcompiler
158:     compiler = new_fcompiler(compiler='absoft')
159:     compiler.customize()
160:     print(compiler.get_version())
161: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import os' statement (line 10)
import os

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from numpy.distutils.cpuinfo import cpu' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_59846 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.distutils.cpuinfo')

if (type(import_59846) is not StypyTypeError):

    if (import_59846 != 'pyd_module'):
        __import__(import_59846)
        sys_modules_59847 = sys.modules[import_59846]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.distutils.cpuinfo', sys_modules_59847.module_type_store, module_type_store, ['cpu'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_59847, sys_modules_59847.module_type_store, module_type_store)
    else:
        from numpy.distutils.cpuinfo import cpu

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.distutils.cpuinfo', None, module_type_store, ['cpu'], [cpu])

else:
    # Assigning a type to the variable 'numpy.distutils.cpuinfo' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.distutils.cpuinfo', import_59846)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_59848 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.distutils.fcompiler')

if (type(import_59848) is not StypyTypeError):

    if (import_59848 != 'pyd_module'):
        __import__(import_59848)
        sys_modules_59849 = sys.modules[import_59848]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.distutils.fcompiler', sys_modules_59849.module_type_store, module_type_store, ['FCompiler', 'dummy_fortran_file'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_59849, sys_modules_59849.module_type_store, module_type_store)
    else:
        from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.distutils.fcompiler', None, module_type_store, ['FCompiler', 'dummy_fortran_file'], [FCompiler, dummy_fortran_file])

else:
    # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.distutils.fcompiler', import_59848)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from numpy.distutils.misc_util import cyg2win32' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_59850 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.distutils.misc_util')

if (type(import_59850) is not StypyTypeError):

    if (import_59850 != 'pyd_module'):
        __import__(import_59850)
        sys_modules_59851 = sys.modules[import_59850]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.distutils.misc_util', sys_modules_59851.module_type_store, module_type_store, ['cyg2win32'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_59851, sys_modules_59851.module_type_store, module_type_store)
    else:
        from numpy.distutils.misc_util import cyg2win32

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.distutils.misc_util', None, module_type_store, ['cyg2win32'], [cyg2win32])

else:
    # Assigning a type to the variable 'numpy.distutils.misc_util' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.distutils.misc_util', import_59850)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')


# Assigning a List to a Name (line 16):

# Obtaining an instance of the builtin type 'list' (line 16)
list_59852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
str_59853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 13), 'str', 'AbsoftFCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 12), list_59852, str_59853)

# Assigning a type to the variable 'compilers' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'compilers', list_59852)
# Declaration of the 'AbsoftFCompiler' class
# Getting the type of 'FCompiler' (line 18)
FCompiler_59854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 22), 'FCompiler')

class AbsoftFCompiler(FCompiler_59854, ):

    @norecursion
    def update_executables(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update_executables'
        module_type_store = module_type_store.open_function_context('update_executables', 49, 4, False)
        # Assigning a type to the variable 'self' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AbsoftFCompiler.update_executables.__dict__.__setitem__('stypy_localization', localization)
        AbsoftFCompiler.update_executables.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AbsoftFCompiler.update_executables.__dict__.__setitem__('stypy_type_store', module_type_store)
        AbsoftFCompiler.update_executables.__dict__.__setitem__('stypy_function_name', 'AbsoftFCompiler.update_executables')
        AbsoftFCompiler.update_executables.__dict__.__setitem__('stypy_param_names_list', [])
        AbsoftFCompiler.update_executables.__dict__.__setitem__('stypy_varargs_param_name', None)
        AbsoftFCompiler.update_executables.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AbsoftFCompiler.update_executables.__dict__.__setitem__('stypy_call_defaults', defaults)
        AbsoftFCompiler.update_executables.__dict__.__setitem__('stypy_call_varargs', varargs)
        AbsoftFCompiler.update_executables.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AbsoftFCompiler.update_executables.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AbsoftFCompiler.update_executables', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update_executables', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update_executables(...)' code ##################

        
        # Assigning a Call to a Name (line 50):
        
        # Call to cyg2win32(...): (line 50)
        # Processing the call arguments (line 50)
        
        # Call to dummy_fortran_file(...): (line 50)
        # Processing the call keyword arguments (line 50)
        kwargs_59857 = {}
        # Getting the type of 'dummy_fortran_file' (line 50)
        dummy_fortran_file_59856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 22), 'dummy_fortran_file', False)
        # Calling dummy_fortran_file(args, kwargs) (line 50)
        dummy_fortran_file_call_result_59858 = invoke(stypy.reporting.localization.Localization(__file__, 50, 22), dummy_fortran_file_59856, *[], **kwargs_59857)
        
        # Processing the call keyword arguments (line 50)
        kwargs_59859 = {}
        # Getting the type of 'cyg2win32' (line 50)
        cyg2win32_59855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'cyg2win32', False)
        # Calling cyg2win32(args, kwargs) (line 50)
        cyg2win32_call_result_59860 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), cyg2win32_59855, *[dummy_fortran_file_call_result_59858], **kwargs_59859)
        
        # Assigning a type to the variable 'f' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'f', cyg2win32_call_result_59860)
        
        # Assigning a List to a Subscript (line 51):
        
        # Obtaining an instance of the builtin type 'list' (line 51)
        list_59861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 51)
        # Adding element type (line 51)
        str_59862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 43), 'str', '<F90>')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 42), list_59861, str_59862)
        # Adding element type (line 51)
        str_59863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 52), 'str', '-V')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 42), list_59861, str_59863)
        # Adding element type (line 51)
        str_59864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 58), 'str', '-c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 42), list_59861, str_59864)
        # Adding element type (line 51)
        # Getting the type of 'f' (line 52)
        f_59865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 43), 'f')
        str_59866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 45), 'str', '.f')
        # Applying the binary operator '+' (line 52)
        result_add_59867 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 43), '+', f_59865, str_59866)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 42), list_59861, result_add_59867)
        # Adding element type (line 51)
        str_59868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 51), 'str', '-o')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 42), list_59861, str_59868)
        # Adding element type (line 51)
        # Getting the type of 'f' (line 52)
        f_59869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 57), 'f')
        str_59870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 59), 'str', '.o')
        # Applying the binary operator '+' (line 52)
        result_add_59871 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 57), '+', f_59869, str_59870)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 42), list_59861, result_add_59871)
        
        # Getting the type of 'self' (line 51)
        self_59872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'self')
        # Obtaining the member 'executables' of a type (line 51)
        executables_59873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), self_59872, 'executables')
        str_59874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 25), 'str', 'version_cmd')
        # Storing an element on a container (line 51)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 8), executables_59873, (str_59874, list_59861))
        
        # ################# End of 'update_executables(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update_executables' in the type store
        # Getting the type of 'stypy_return_type' (line 49)
        stypy_return_type_59875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59875)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update_executables'
        return stypy_return_type_59875


    @norecursion
    def get_flags_linker_so(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_linker_so'
        module_type_store = module_type_store.open_function_context('get_flags_linker_so', 54, 4, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AbsoftFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_localization', localization)
        AbsoftFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AbsoftFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_type_store', module_type_store)
        AbsoftFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_function_name', 'AbsoftFCompiler.get_flags_linker_so')
        AbsoftFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_param_names_list', [])
        AbsoftFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_varargs_param_name', None)
        AbsoftFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AbsoftFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_call_defaults', defaults)
        AbsoftFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_call_varargs', varargs)
        AbsoftFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AbsoftFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AbsoftFCompiler.get_flags_linker_so', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_linker_so', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_linker_so(...)' code ##################

        
        
        # Getting the type of 'os' (line 55)
        os_59876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'os')
        # Obtaining the member 'name' of a type (line 55)
        name_59877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 11), os_59876, 'name')
        str_59878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 20), 'str', 'nt')
        # Applying the binary operator '==' (line 55)
        result_eq_59879 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 11), '==', name_59877, str_59878)
        
        # Testing the type of an if condition (line 55)
        if_condition_59880 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 55, 8), result_eq_59879)
        # Assigning a type to the variable 'if_condition_59880' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'if_condition_59880', if_condition_59880)
        # SSA begins for if statement (line 55)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 56):
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_59881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        # Adding element type (line 56)
        str_59882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 19), 'str', '/dll')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 18), list_59881, str_59882)
        
        # Assigning a type to the variable 'opt' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'opt', list_59881)
        # SSA branch for the else part of an if statement (line 55)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to get_version(...): (line 61)
        # Processing the call keyword arguments (line 61)
        kwargs_59885 = {}
        # Getting the type of 'self' (line 61)
        self_59883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 13), 'self', False)
        # Obtaining the member 'get_version' of a type (line 61)
        get_version_59884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 13), self_59883, 'get_version')
        # Calling get_version(args, kwargs) (line 61)
        get_version_call_result_59886 = invoke(stypy.reporting.localization.Localization(__file__, 61, 13), get_version_59884, *[], **kwargs_59885)
        
        str_59887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 35), 'str', '9.0')
        # Applying the binary operator '>=' (line 61)
        result_ge_59888 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 13), '>=', get_version_call_result_59886, str_59887)
        
        # Testing the type of an if condition (line 61)
        if_condition_59889 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 13), result_ge_59888)
        # Assigning a type to the variable 'if_condition_59889' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 13), 'if_condition_59889', if_condition_59889)
        # SSA begins for if statement (line 61)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 62):
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_59890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        # Adding element type (line 62)
        str_59891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 19), 'str', '-shared')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 18), list_59890, str_59891)
        
        # Assigning a type to the variable 'opt' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'opt', list_59890)
        # SSA branch for the else part of an if statement (line 61)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Name (line 64):
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_59892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        # Adding element type (line 64)
        str_59893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 19), 'str', '-K')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 18), list_59892, str_59893)
        # Adding element type (line 64)
        str_59894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 25), 'str', 'shared')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 18), list_59892, str_59894)
        
        # Assigning a type to the variable 'opt' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'opt', list_59892)
        # SSA join for if statement (line 61)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 55)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'opt' (line 65)
        opt_59895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'opt')
        # Assigning a type to the variable 'stypy_return_type' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'stypy_return_type', opt_59895)
        
        # ################# End of 'get_flags_linker_so(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_linker_so' in the type store
        # Getting the type of 'stypy_return_type' (line 54)
        stypy_return_type_59896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59896)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_linker_so'
        return stypy_return_type_59896


    @norecursion
    def library_dir_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'library_dir_option'
        module_type_store = module_type_store.open_function_context('library_dir_option', 67, 4, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AbsoftFCompiler.library_dir_option.__dict__.__setitem__('stypy_localization', localization)
        AbsoftFCompiler.library_dir_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AbsoftFCompiler.library_dir_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        AbsoftFCompiler.library_dir_option.__dict__.__setitem__('stypy_function_name', 'AbsoftFCompiler.library_dir_option')
        AbsoftFCompiler.library_dir_option.__dict__.__setitem__('stypy_param_names_list', ['dir'])
        AbsoftFCompiler.library_dir_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        AbsoftFCompiler.library_dir_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AbsoftFCompiler.library_dir_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        AbsoftFCompiler.library_dir_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        AbsoftFCompiler.library_dir_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AbsoftFCompiler.library_dir_option.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AbsoftFCompiler.library_dir_option', ['dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'library_dir_option', localization, ['dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'library_dir_option(...)' code ##################

        
        
        # Getting the type of 'os' (line 68)
        os_59897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'os')
        # Obtaining the member 'name' of a type (line 68)
        name_59898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 11), os_59897, 'name')
        str_59899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 20), 'str', 'nt')
        # Applying the binary operator '==' (line 68)
        result_eq_59900 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 11), '==', name_59898, str_59899)
        
        # Testing the type of an if condition (line 68)
        if_condition_59901 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 8), result_eq_59900)
        # Assigning a type to the variable 'if_condition_59901' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'if_condition_59901', if_condition_59901)
        # SSA begins for if statement (line 68)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 69)
        list_59902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 69)
        # Adding element type (line 69)
        str_59903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 20), 'str', '-link')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 19), list_59902, str_59903)
        # Adding element type (line 69)
        str_59904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 29), 'str', '/PATH:"%s"')
        # Getting the type of 'dir' (line 69)
        dir_59905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 45), 'dir')
        # Applying the binary operator '%' (line 69)
        result_mod_59906 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 29), '%', str_59904, dir_59905)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 19), list_59902, result_mod_59906)
        
        # Assigning a type to the variable 'stypy_return_type' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'stypy_return_type', list_59902)
        # SSA join for if statement (line 68)
        module_type_store = module_type_store.join_ssa_context()
        
        str_59907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 15), 'str', '-L')
        # Getting the type of 'dir' (line 70)
        dir_59908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 22), 'dir')
        # Applying the binary operator '+' (line 70)
        result_add_59909 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 15), '+', str_59907, dir_59908)
        
        # Assigning a type to the variable 'stypy_return_type' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'stypy_return_type', result_add_59909)
        
        # ################# End of 'library_dir_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'library_dir_option' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_59910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59910)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'library_dir_option'
        return stypy_return_type_59910


    @norecursion
    def library_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'library_option'
        module_type_store = module_type_store.open_function_context('library_option', 72, 4, False)
        # Assigning a type to the variable 'self' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AbsoftFCompiler.library_option.__dict__.__setitem__('stypy_localization', localization)
        AbsoftFCompiler.library_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AbsoftFCompiler.library_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        AbsoftFCompiler.library_option.__dict__.__setitem__('stypy_function_name', 'AbsoftFCompiler.library_option')
        AbsoftFCompiler.library_option.__dict__.__setitem__('stypy_param_names_list', ['lib'])
        AbsoftFCompiler.library_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        AbsoftFCompiler.library_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AbsoftFCompiler.library_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        AbsoftFCompiler.library_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        AbsoftFCompiler.library_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AbsoftFCompiler.library_option.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AbsoftFCompiler.library_option', ['lib'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'library_option', localization, ['lib'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'library_option(...)' code ##################

        
        
        # Getting the type of 'os' (line 73)
        os_59911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 11), 'os')
        # Obtaining the member 'name' of a type (line 73)
        name_59912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 11), os_59911, 'name')
        str_59913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 20), 'str', 'nt')
        # Applying the binary operator '==' (line 73)
        result_eq_59914 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 11), '==', name_59912, str_59913)
        
        # Testing the type of an if condition (line 73)
        if_condition_59915 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 8), result_eq_59914)
        # Assigning a type to the variable 'if_condition_59915' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'if_condition_59915', if_condition_59915)
        # SSA begins for if statement (line 73)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_59916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 19), 'str', '%s.lib')
        # Getting the type of 'lib' (line 74)
        lib_59917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 31), 'lib')
        # Applying the binary operator '%' (line 74)
        result_mod_59918 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 19), '%', str_59916, lib_59917)
        
        # Assigning a type to the variable 'stypy_return_type' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'stypy_return_type', result_mod_59918)
        # SSA join for if statement (line 73)
        module_type_store = module_type_store.join_ssa_context()
        
        str_59919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 15), 'str', '-l')
        # Getting the type of 'lib' (line 75)
        lib_59920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), 'lib')
        # Applying the binary operator '+' (line 75)
        result_add_59921 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 15), '+', str_59919, lib_59920)
        
        # Assigning a type to the variable 'stypy_return_type' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'stypy_return_type', result_add_59921)
        
        # ################# End of 'library_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'library_option' in the type store
        # Getting the type of 'stypy_return_type' (line 72)
        stypy_return_type_59922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59922)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'library_option'
        return stypy_return_type_59922


    @norecursion
    def get_library_dirs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_library_dirs'
        module_type_store = module_type_store.open_function_context('get_library_dirs', 77, 4, False)
        # Assigning a type to the variable 'self' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AbsoftFCompiler.get_library_dirs.__dict__.__setitem__('stypy_localization', localization)
        AbsoftFCompiler.get_library_dirs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AbsoftFCompiler.get_library_dirs.__dict__.__setitem__('stypy_type_store', module_type_store)
        AbsoftFCompiler.get_library_dirs.__dict__.__setitem__('stypy_function_name', 'AbsoftFCompiler.get_library_dirs')
        AbsoftFCompiler.get_library_dirs.__dict__.__setitem__('stypy_param_names_list', [])
        AbsoftFCompiler.get_library_dirs.__dict__.__setitem__('stypy_varargs_param_name', None)
        AbsoftFCompiler.get_library_dirs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AbsoftFCompiler.get_library_dirs.__dict__.__setitem__('stypy_call_defaults', defaults)
        AbsoftFCompiler.get_library_dirs.__dict__.__setitem__('stypy_call_varargs', varargs)
        AbsoftFCompiler.get_library_dirs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AbsoftFCompiler.get_library_dirs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AbsoftFCompiler.get_library_dirs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_library_dirs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_library_dirs(...)' code ##################

        
        # Assigning a Call to a Name (line 78):
        
        # Call to get_library_dirs(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'self' (line 78)
        self_59925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 41), 'self', False)
        # Processing the call keyword arguments (line 78)
        kwargs_59926 = {}
        # Getting the type of 'FCompiler' (line 78)
        FCompiler_59923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 14), 'FCompiler', False)
        # Obtaining the member 'get_library_dirs' of a type (line 78)
        get_library_dirs_59924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 14), FCompiler_59923, 'get_library_dirs')
        # Calling get_library_dirs(args, kwargs) (line 78)
        get_library_dirs_call_result_59927 = invoke(stypy.reporting.localization.Localization(__file__, 78, 14), get_library_dirs_59924, *[self_59925], **kwargs_59926)
        
        # Assigning a type to the variable 'opt' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'opt', get_library_dirs_call_result_59927)
        
        # Assigning a Call to a Name (line 79):
        
        # Call to get(...): (line 79)
        # Processing the call arguments (line 79)
        str_59931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 27), 'str', 'ABSOFT')
        # Processing the call keyword arguments (line 79)
        kwargs_59932 = {}
        # Getting the type of 'os' (line 79)
        os_59928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'os', False)
        # Obtaining the member 'environ' of a type (line 79)
        environ_59929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), os_59928, 'environ')
        # Obtaining the member 'get' of a type (line 79)
        get_59930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), environ_59929, 'get')
        # Calling get(args, kwargs) (line 79)
        get_call_result_59933 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), get_59930, *[str_59931], **kwargs_59932)
        
        # Assigning a type to the variable 'd' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'd', get_call_result_59933)
        
        # Getting the type of 'd' (line 80)
        d_59934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'd')
        # Testing the type of an if condition (line 80)
        if_condition_59935 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 8), d_59934)
        # Assigning a type to the variable 'if_condition_59935' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'if_condition_59935', if_condition_59935)
        # SSA begins for if statement (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to get_version(...): (line 81)
        # Processing the call keyword arguments (line 81)
        kwargs_59938 = {}
        # Getting the type of 'self' (line 81)
        self_59936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 15), 'self', False)
        # Obtaining the member 'get_version' of a type (line 81)
        get_version_59937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 15), self_59936, 'get_version')
        # Calling get_version(args, kwargs) (line 81)
        get_version_call_result_59939 = invoke(stypy.reporting.localization.Localization(__file__, 81, 15), get_version_59937, *[], **kwargs_59938)
        
        str_59940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 37), 'str', '10.0')
        # Applying the binary operator '>=' (line 81)
        result_ge_59941 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 15), '>=', get_version_call_result_59939, str_59940)
        
        # Testing the type of an if condition (line 81)
        if_condition_59942 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 12), result_ge_59941)
        # Assigning a type to the variable 'if_condition_59942' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'if_condition_59942', if_condition_59942)
        # SSA begins for if statement (line 81)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 83):
        str_59943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 25), 'str', 'sh')
        # Assigning a type to the variable 'prefix' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'prefix', str_59943)
        # SSA branch for the else part of an if statement (line 81)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 85):
        str_59944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 25), 'str', '')
        # Assigning a type to the variable 'prefix' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'prefix', str_59944)
        # SSA join for if statement (line 81)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to is_64bit(...): (line 86)
        # Processing the call keyword arguments (line 86)
        kwargs_59947 = {}
        # Getting the type of 'cpu' (line 86)
        cpu_59945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'cpu', False)
        # Obtaining the member 'is_64bit' of a type (line 86)
        is_64bit_59946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 15), cpu_59945, 'is_64bit')
        # Calling is_64bit(args, kwargs) (line 86)
        is_64bit_call_result_59948 = invoke(stypy.reporting.localization.Localization(__file__, 86, 15), is_64bit_59946, *[], **kwargs_59947)
        
        # Testing the type of an if condition (line 86)
        if_condition_59949 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 12), is_64bit_call_result_59948)
        # Assigning a type to the variable 'if_condition_59949' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'if_condition_59949', if_condition_59949)
        # SSA begins for if statement (line 86)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 87):
        str_59950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 25), 'str', '64')
        # Assigning a type to the variable 'suffix' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'suffix', str_59950)
        # SSA branch for the else part of an if statement (line 86)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 89):
        str_59951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 25), 'str', '')
        # Assigning a type to the variable 'suffix' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'suffix', str_59951)
        # SSA join for if statement (line 86)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 90)
        # Processing the call arguments (line 90)
        
        # Call to join(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'd' (line 90)
        d_59957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 36), 'd', False)
        str_59958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 39), 'str', '%slib%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 90)
        tuple_59959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 90)
        # Adding element type (line 90)
        # Getting the type of 'prefix' (line 90)
        prefix_59960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 52), 'prefix', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 52), tuple_59959, prefix_59960)
        # Adding element type (line 90)
        # Getting the type of 'suffix' (line 90)
        suffix_59961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 60), 'suffix', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 52), tuple_59959, suffix_59961)
        
        # Applying the binary operator '%' (line 90)
        result_mod_59962 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 39), '%', str_59958, tuple_59959)
        
        # Processing the call keyword arguments (line 90)
        kwargs_59963 = {}
        # Getting the type of 'os' (line 90)
        os_59954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 90)
        path_59955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 23), os_59954, 'path')
        # Obtaining the member 'join' of a type (line 90)
        join_59956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 23), path_59955, 'join')
        # Calling join(args, kwargs) (line 90)
        join_call_result_59964 = invoke(stypy.reporting.localization.Localization(__file__, 90, 23), join_59956, *[d_59957, result_mod_59962], **kwargs_59963)
        
        # Processing the call keyword arguments (line 90)
        kwargs_59965 = {}
        # Getting the type of 'opt' (line 90)
        opt_59952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'opt', False)
        # Obtaining the member 'append' of a type (line 90)
        append_59953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), opt_59952, 'append')
        # Calling append(args, kwargs) (line 90)
        append_call_result_59966 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), append_59953, *[join_call_result_59964], **kwargs_59965)
        
        # SSA join for if statement (line 80)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'opt' (line 91)
        opt_59967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 15), 'opt')
        # Assigning a type to the variable 'stypy_return_type' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'stypy_return_type', opt_59967)
        
        # ################# End of 'get_library_dirs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_library_dirs' in the type store
        # Getting the type of 'stypy_return_type' (line 77)
        stypy_return_type_59968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59968)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_library_dirs'
        return stypy_return_type_59968


    @norecursion
    def get_libraries(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_libraries'
        module_type_store = module_type_store.open_function_context('get_libraries', 93, 4, False)
        # Assigning a type to the variable 'self' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AbsoftFCompiler.get_libraries.__dict__.__setitem__('stypy_localization', localization)
        AbsoftFCompiler.get_libraries.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AbsoftFCompiler.get_libraries.__dict__.__setitem__('stypy_type_store', module_type_store)
        AbsoftFCompiler.get_libraries.__dict__.__setitem__('stypy_function_name', 'AbsoftFCompiler.get_libraries')
        AbsoftFCompiler.get_libraries.__dict__.__setitem__('stypy_param_names_list', [])
        AbsoftFCompiler.get_libraries.__dict__.__setitem__('stypy_varargs_param_name', None)
        AbsoftFCompiler.get_libraries.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AbsoftFCompiler.get_libraries.__dict__.__setitem__('stypy_call_defaults', defaults)
        AbsoftFCompiler.get_libraries.__dict__.__setitem__('stypy_call_varargs', varargs)
        AbsoftFCompiler.get_libraries.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AbsoftFCompiler.get_libraries.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AbsoftFCompiler.get_libraries', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_libraries', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_libraries(...)' code ##################

        
        # Assigning a Call to a Name (line 94):
        
        # Call to get_libraries(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'self' (line 94)
        self_59971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 38), 'self', False)
        # Processing the call keyword arguments (line 94)
        kwargs_59972 = {}
        # Getting the type of 'FCompiler' (line 94)
        FCompiler_59969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 14), 'FCompiler', False)
        # Obtaining the member 'get_libraries' of a type (line 94)
        get_libraries_59970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 14), FCompiler_59969, 'get_libraries')
        # Calling get_libraries(args, kwargs) (line 94)
        get_libraries_call_result_59973 = invoke(stypy.reporting.localization.Localization(__file__, 94, 14), get_libraries_59970, *[self_59971], **kwargs_59972)
        
        # Assigning a type to the variable 'opt' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'opt', get_libraries_call_result_59973)
        
        
        
        # Call to get_version(...): (line 95)
        # Processing the call keyword arguments (line 95)
        kwargs_59976 = {}
        # Getting the type of 'self' (line 95)
        self_59974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 11), 'self', False)
        # Obtaining the member 'get_version' of a type (line 95)
        get_version_59975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 11), self_59974, 'get_version')
        # Calling get_version(args, kwargs) (line 95)
        get_version_call_result_59977 = invoke(stypy.reporting.localization.Localization(__file__, 95, 11), get_version_59975, *[], **kwargs_59976)
        
        str_59978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 33), 'str', '11.0')
        # Applying the binary operator '>=' (line 95)
        result_ge_59979 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 11), '>=', get_version_call_result_59977, str_59978)
        
        # Testing the type of an if condition (line 95)
        if_condition_59980 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 8), result_ge_59979)
        # Assigning a type to the variable 'if_condition_59980' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'if_condition_59980', if_condition_59980)
        # SSA begins for if statement (line 95)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_59983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        str_59984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 24), 'str', 'af90math')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 23), list_59983, str_59984)
        # Adding element type (line 96)
        str_59985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 36), 'str', 'afio')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 23), list_59983, str_59985)
        # Adding element type (line 96)
        str_59986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 44), 'str', 'af77math')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 23), list_59983, str_59986)
        # Adding element type (line 96)
        str_59987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 56), 'str', 'amisc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 23), list_59983, str_59987)
        
        # Processing the call keyword arguments (line 96)
        kwargs_59988 = {}
        # Getting the type of 'opt' (line 96)
        opt_59981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'opt', False)
        # Obtaining the member 'extend' of a type (line 96)
        extend_59982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 12), opt_59981, 'extend')
        # Calling extend(args, kwargs) (line 96)
        extend_call_result_59989 = invoke(stypy.reporting.localization.Localization(__file__, 96, 12), extend_59982, *[list_59983], **kwargs_59988)
        
        # SSA branch for the else part of an if statement (line 95)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to get_version(...): (line 97)
        # Processing the call keyword arguments (line 97)
        kwargs_59992 = {}
        # Getting the type of 'self' (line 97)
        self_59990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 13), 'self', False)
        # Obtaining the member 'get_version' of a type (line 97)
        get_version_59991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 13), self_59990, 'get_version')
        # Calling get_version(args, kwargs) (line 97)
        get_version_call_result_59993 = invoke(stypy.reporting.localization.Localization(__file__, 97, 13), get_version_59991, *[], **kwargs_59992)
        
        str_59994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 35), 'str', '10.0')
        # Applying the binary operator '>=' (line 97)
        result_ge_59995 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 13), '>=', get_version_call_result_59993, str_59994)
        
        # Testing the type of an if condition (line 97)
        if_condition_59996 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 13), result_ge_59995)
        # Assigning a type to the variable 'if_condition_59996' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 13), 'if_condition_59996', if_condition_59996)
        # SSA begins for if statement (line 97)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 98)
        # Processing the call arguments (line 98)
        
        # Obtaining an instance of the builtin type 'list' (line 98)
        list_59999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 98)
        # Adding element type (line 98)
        str_60000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 24), 'str', 'af90math')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 23), list_59999, str_60000)
        # Adding element type (line 98)
        str_60001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 36), 'str', 'afio')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 23), list_59999, str_60001)
        # Adding element type (line 98)
        str_60002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 44), 'str', 'af77math')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 23), list_59999, str_60002)
        # Adding element type (line 98)
        str_60003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 56), 'str', 'U77')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 23), list_59999, str_60003)
        
        # Processing the call keyword arguments (line 98)
        kwargs_60004 = {}
        # Getting the type of 'opt' (line 98)
        opt_59997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'opt', False)
        # Obtaining the member 'extend' of a type (line 98)
        extend_59998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), opt_59997, 'extend')
        # Calling extend(args, kwargs) (line 98)
        extend_call_result_60005 = invoke(stypy.reporting.localization.Localization(__file__, 98, 12), extend_59998, *[list_59999], **kwargs_60004)
        
        # SSA branch for the else part of an if statement (line 97)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to get_version(...): (line 99)
        # Processing the call keyword arguments (line 99)
        kwargs_60008 = {}
        # Getting the type of 'self' (line 99)
        self_60006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 13), 'self', False)
        # Obtaining the member 'get_version' of a type (line 99)
        get_version_60007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 13), self_60006, 'get_version')
        # Calling get_version(args, kwargs) (line 99)
        get_version_call_result_60009 = invoke(stypy.reporting.localization.Localization(__file__, 99, 13), get_version_60007, *[], **kwargs_60008)
        
        str_60010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 35), 'str', '8.0')
        # Applying the binary operator '>=' (line 99)
        result_ge_60011 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 13), '>=', get_version_call_result_60009, str_60010)
        
        # Testing the type of an if condition (line 99)
        if_condition_60012 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 13), result_ge_60011)
        # Assigning a type to the variable 'if_condition_60012' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 13), 'if_condition_60012', if_condition_60012)
        # SSA begins for if statement (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 100)
        # Processing the call arguments (line 100)
        
        # Obtaining an instance of the builtin type 'list' (line 100)
        list_60015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 100)
        # Adding element type (line 100)
        str_60016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 24), 'str', 'f90math')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 23), list_60015, str_60016)
        # Adding element type (line 100)
        str_60017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 35), 'str', 'fio')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 23), list_60015, str_60017)
        # Adding element type (line 100)
        str_60018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 42), 'str', 'f77math')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 23), list_60015, str_60018)
        # Adding element type (line 100)
        str_60019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 53), 'str', 'U77')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 23), list_60015, str_60019)
        
        # Processing the call keyword arguments (line 100)
        kwargs_60020 = {}
        # Getting the type of 'opt' (line 100)
        opt_60013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'opt', False)
        # Obtaining the member 'extend' of a type (line 100)
        extend_60014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), opt_60013, 'extend')
        # Calling extend(args, kwargs) (line 100)
        extend_call_result_60021 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), extend_60014, *[list_60015], **kwargs_60020)
        
        # SSA branch for the else part of an if statement (line 99)
        module_type_store.open_ssa_branch('else')
        
        # Call to extend(...): (line 102)
        # Processing the call arguments (line 102)
        
        # Obtaining an instance of the builtin type 'list' (line 102)
        list_60024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 102)
        # Adding element type (line 102)
        str_60025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 24), 'str', 'fio')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 23), list_60024, str_60025)
        # Adding element type (line 102)
        str_60026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 31), 'str', 'f90math')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 23), list_60024, str_60026)
        # Adding element type (line 102)
        str_60027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 42), 'str', 'fmath')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 23), list_60024, str_60027)
        # Adding element type (line 102)
        str_60028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 51), 'str', 'U77')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 23), list_60024, str_60028)
        
        # Processing the call keyword arguments (line 102)
        kwargs_60029 = {}
        # Getting the type of 'opt' (line 102)
        opt_60022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'opt', False)
        # Obtaining the member 'extend' of a type (line 102)
        extend_60023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), opt_60022, 'extend')
        # Calling extend(args, kwargs) (line 102)
        extend_call_result_60030 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), extend_60023, *[list_60024], **kwargs_60029)
        
        # SSA join for if statement (line 99)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 97)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 95)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'os' (line 103)
        os_60031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 11), 'os')
        # Obtaining the member 'name' of a type (line 103)
        name_60032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 11), os_60031, 'name')
        str_60033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 21), 'str', 'nt')
        # Applying the binary operator '==' (line 103)
        result_eq_60034 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 11), '==', name_60032, str_60033)
        
        # Testing the type of an if condition (line 103)
        if_condition_60035 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 8), result_eq_60034)
        # Assigning a type to the variable 'if_condition_60035' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'if_condition_60035', if_condition_60035)
        # SSA begins for if statement (line 103)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 104)
        # Processing the call arguments (line 104)
        str_60038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 23), 'str', 'COMDLG32')
        # Processing the call keyword arguments (line 104)
        kwargs_60039 = {}
        # Getting the type of 'opt' (line 104)
        opt_60036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'opt', False)
        # Obtaining the member 'append' of a type (line 104)
        append_60037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 12), opt_60036, 'append')
        # Calling append(args, kwargs) (line 104)
        append_call_result_60040 = invoke(stypy.reporting.localization.Localization(__file__, 104, 12), append_60037, *[str_60038], **kwargs_60039)
        
        # SSA join for if statement (line 103)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'opt' (line 105)
        opt_60041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), 'opt')
        # Assigning a type to the variable 'stypy_return_type' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'stypy_return_type', opt_60041)
        
        # ################# End of 'get_libraries(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_libraries' in the type store
        # Getting the type of 'stypy_return_type' (line 93)
        stypy_return_type_60042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60042)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_libraries'
        return stypy_return_type_60042


    @norecursion
    def get_flags(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags'
        module_type_store = module_type_store.open_function_context('get_flags', 107, 4, False)
        # Assigning a type to the variable 'self' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AbsoftFCompiler.get_flags.__dict__.__setitem__('stypy_localization', localization)
        AbsoftFCompiler.get_flags.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AbsoftFCompiler.get_flags.__dict__.__setitem__('stypy_type_store', module_type_store)
        AbsoftFCompiler.get_flags.__dict__.__setitem__('stypy_function_name', 'AbsoftFCompiler.get_flags')
        AbsoftFCompiler.get_flags.__dict__.__setitem__('stypy_param_names_list', [])
        AbsoftFCompiler.get_flags.__dict__.__setitem__('stypy_varargs_param_name', None)
        AbsoftFCompiler.get_flags.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AbsoftFCompiler.get_flags.__dict__.__setitem__('stypy_call_defaults', defaults)
        AbsoftFCompiler.get_flags.__dict__.__setitem__('stypy_call_varargs', varargs)
        AbsoftFCompiler.get_flags.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AbsoftFCompiler.get_flags.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AbsoftFCompiler.get_flags', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags(...)' code ##################

        
        # Assigning a Call to a Name (line 108):
        
        # Call to get_flags(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'self' (line 108)
        self_60045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 34), 'self', False)
        # Processing the call keyword arguments (line 108)
        kwargs_60046 = {}
        # Getting the type of 'FCompiler' (line 108)
        FCompiler_60043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 'FCompiler', False)
        # Obtaining the member 'get_flags' of a type (line 108)
        get_flags_60044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), FCompiler_60043, 'get_flags')
        # Calling get_flags(args, kwargs) (line 108)
        get_flags_call_result_60047 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), get_flags_60044, *[self_60045], **kwargs_60046)
        
        # Assigning a type to the variable 'opt' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'opt', get_flags_call_result_60047)
        
        
        # Getting the type of 'os' (line 109)
        os_60048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'os')
        # Obtaining the member 'name' of a type (line 109)
        name_60049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 11), os_60048, 'name')
        str_60050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 22), 'str', 'nt')
        # Applying the binary operator '!=' (line 109)
        result_ne_60051 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 11), '!=', name_60049, str_60050)
        
        # Testing the type of an if condition (line 109)
        if_condition_60052 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 8), result_ne_60051)
        # Assigning a type to the variable 'if_condition_60052' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'if_condition_60052', if_condition_60052)
        # SSA begins for if statement (line 109)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 110)
        # Processing the call arguments (line 110)
        
        # Obtaining an instance of the builtin type 'list' (line 110)
        list_60055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 110)
        # Adding element type (line 110)
        str_60056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 24), 'str', '-s')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 23), list_60055, str_60056)
        
        # Processing the call keyword arguments (line 110)
        kwargs_60057 = {}
        # Getting the type of 'opt' (line 110)
        opt_60053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'opt', False)
        # Obtaining the member 'extend' of a type (line 110)
        extend_60054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), opt_60053, 'extend')
        # Calling extend(args, kwargs) (line 110)
        extend_call_result_60058 = invoke(stypy.reporting.localization.Localization(__file__, 110, 12), extend_60054, *[list_60055], **kwargs_60057)
        
        
        
        # Call to get_version(...): (line 111)
        # Processing the call keyword arguments (line 111)
        kwargs_60061 = {}
        # Getting the type of 'self' (line 111)
        self_60059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'self', False)
        # Obtaining the member 'get_version' of a type (line 111)
        get_version_60060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 15), self_60059, 'get_version')
        # Calling get_version(args, kwargs) (line 111)
        get_version_call_result_60062 = invoke(stypy.reporting.localization.Localization(__file__, 111, 15), get_version_60060, *[], **kwargs_60061)
        
        # Testing the type of an if condition (line 111)
        if_condition_60063 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 12), get_version_call_result_60062)
        # Assigning a type to the variable 'if_condition_60063' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'if_condition_60063', if_condition_60063)
        # SSA begins for if statement (line 111)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to get_version(...): (line 112)
        # Processing the call keyword arguments (line 112)
        kwargs_60066 = {}
        # Getting the type of 'self' (line 112)
        self_60064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 19), 'self', False)
        # Obtaining the member 'get_version' of a type (line 112)
        get_version_60065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 19), self_60064, 'get_version')
        # Calling get_version(args, kwargs) (line 112)
        get_version_call_result_60067 = invoke(stypy.reporting.localization.Localization(__file__, 112, 19), get_version_60065, *[], **kwargs_60066)
        
        str_60068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 39), 'str', '8.2')
        # Applying the binary operator '>=' (line 112)
        result_ge_60069 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 19), '>=', get_version_call_result_60067, str_60068)
        
        # Testing the type of an if condition (line 112)
        if_condition_60070 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 16), result_ge_60069)
        # Assigning a type to the variable 'if_condition_60070' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'if_condition_60070', if_condition_60070)
        # SSA begins for if statement (line 112)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 113)
        # Processing the call arguments (line 113)
        str_60073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 31), 'str', '-fpic')
        # Processing the call keyword arguments (line 113)
        kwargs_60074 = {}
        # Getting the type of 'opt' (line 113)
        opt_60071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'opt', False)
        # Obtaining the member 'append' of a type (line 113)
        append_60072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 20), opt_60071, 'append')
        # Calling append(args, kwargs) (line 113)
        append_call_result_60075 = invoke(stypy.reporting.localization.Localization(__file__, 113, 20), append_60072, *[str_60073], **kwargs_60074)
        
        # SSA join for if statement (line 112)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 111)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 109)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'opt' (line 114)
        opt_60076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 15), 'opt')
        # Assigning a type to the variable 'stypy_return_type' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'stypy_return_type', opt_60076)
        
        # ################# End of 'get_flags(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags' in the type store
        # Getting the type of 'stypy_return_type' (line 107)
        stypy_return_type_60077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60077)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags'
        return stypy_return_type_60077


    @norecursion
    def get_flags_f77(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_f77'
        module_type_store = module_type_store.open_function_context('get_flags_f77', 116, 4, False)
        # Assigning a type to the variable 'self' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AbsoftFCompiler.get_flags_f77.__dict__.__setitem__('stypy_localization', localization)
        AbsoftFCompiler.get_flags_f77.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AbsoftFCompiler.get_flags_f77.__dict__.__setitem__('stypy_type_store', module_type_store)
        AbsoftFCompiler.get_flags_f77.__dict__.__setitem__('stypy_function_name', 'AbsoftFCompiler.get_flags_f77')
        AbsoftFCompiler.get_flags_f77.__dict__.__setitem__('stypy_param_names_list', [])
        AbsoftFCompiler.get_flags_f77.__dict__.__setitem__('stypy_varargs_param_name', None)
        AbsoftFCompiler.get_flags_f77.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AbsoftFCompiler.get_flags_f77.__dict__.__setitem__('stypy_call_defaults', defaults)
        AbsoftFCompiler.get_flags_f77.__dict__.__setitem__('stypy_call_varargs', varargs)
        AbsoftFCompiler.get_flags_f77.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AbsoftFCompiler.get_flags_f77.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AbsoftFCompiler.get_flags_f77', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_f77', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_f77(...)' code ##################

        
        # Assigning a Call to a Name (line 117):
        
        # Call to get_flags_f77(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'self' (line 117)
        self_60080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 38), 'self', False)
        # Processing the call keyword arguments (line 117)
        kwargs_60081 = {}
        # Getting the type of 'FCompiler' (line 117)
        FCompiler_60078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 14), 'FCompiler', False)
        # Obtaining the member 'get_flags_f77' of a type (line 117)
        get_flags_f77_60079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 14), FCompiler_60078, 'get_flags_f77')
        # Calling get_flags_f77(args, kwargs) (line 117)
        get_flags_f77_call_result_60082 = invoke(stypy.reporting.localization.Localization(__file__, 117, 14), get_flags_f77_60079, *[self_60080], **kwargs_60081)
        
        # Assigning a type to the variable 'opt' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'opt', get_flags_f77_call_result_60082)
        
        # Call to extend(...): (line 118)
        # Processing the call arguments (line 118)
        
        # Obtaining an instance of the builtin type 'list' (line 118)
        list_60085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 118)
        # Adding element type (line 118)
        str_60086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 20), 'str', '-N22')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 19), list_60085, str_60086)
        # Adding element type (line 118)
        str_60087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 28), 'str', '-N90')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 19), list_60085, str_60087)
        # Adding element type (line 118)
        str_60088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 36), 'str', '-N110')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 19), list_60085, str_60088)
        
        # Processing the call keyword arguments (line 118)
        kwargs_60089 = {}
        # Getting the type of 'opt' (line 118)
        opt_60083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'opt', False)
        # Obtaining the member 'extend' of a type (line 118)
        extend_60084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), opt_60083, 'extend')
        # Calling extend(args, kwargs) (line 118)
        extend_call_result_60090 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), extend_60084, *[list_60085], **kwargs_60089)
        
        
        # Assigning a Call to a Name (line 119):
        
        # Call to get_version(...): (line 119)
        # Processing the call keyword arguments (line 119)
        kwargs_60093 = {}
        # Getting the type of 'self' (line 119)
        self_60091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'self', False)
        # Obtaining the member 'get_version' of a type (line 119)
        get_version_60092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 12), self_60091, 'get_version')
        # Calling get_version(args, kwargs) (line 119)
        get_version_call_result_60094 = invoke(stypy.reporting.localization.Localization(__file__, 119, 12), get_version_60092, *[], **kwargs_60093)
        
        # Assigning a type to the variable 'v' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'v', get_version_call_result_60094)
        
        
        # Getting the type of 'os' (line 120)
        os_60095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 11), 'os')
        # Obtaining the member 'name' of a type (line 120)
        name_60096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 11), os_60095, 'name')
        str_60097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 22), 'str', 'nt')
        # Applying the binary operator '==' (line 120)
        result_eq_60098 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 11), '==', name_60096, str_60097)
        
        # Testing the type of an if condition (line 120)
        if_condition_60099 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 120, 8), result_eq_60098)
        # Assigning a type to the variable 'if_condition_60099' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'if_condition_60099', if_condition_60099)
        # SSA begins for if statement (line 120)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Evaluating a boolean operation
        # Getting the type of 'v' (line 121)
        v_60100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'v')
        
        # Getting the type of 'v' (line 121)
        v_60101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 21), 'v')
        str_60102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 24), 'str', '8.0')
        # Applying the binary operator '>=' (line 121)
        result_ge_60103 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 21), '>=', v_60101, str_60102)
        
        # Applying the binary operator 'and' (line 121)
        result_and_keyword_60104 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 15), 'and', v_60100, result_ge_60103)
        
        # Testing the type of an if condition (line 121)
        if_condition_60105 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 12), result_and_keyword_60104)
        # Assigning a type to the variable 'if_condition_60105' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'if_condition_60105', if_condition_60105)
        # SSA begins for if statement (line 121)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Obtaining an instance of the builtin type 'list' (line 122)
        list_60108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 122)
        # Adding element type (line 122)
        str_60109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 28), 'str', '-f')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 27), list_60108, str_60109)
        # Adding element type (line 122)
        str_60110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 34), 'str', '-N15')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 27), list_60108, str_60110)
        
        # Processing the call keyword arguments (line 122)
        kwargs_60111 = {}
        # Getting the type of 'opt' (line 122)
        opt_60106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'opt', False)
        # Obtaining the member 'extend' of a type (line 122)
        extend_60107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 16), opt_60106, 'extend')
        # Calling extend(args, kwargs) (line 122)
        extend_call_result_60112 = invoke(stypy.reporting.localization.Localization(__file__, 122, 16), extend_60107, *[list_60108], **kwargs_60111)
        
        # SSA join for if statement (line 121)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 120)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 124)
        # Processing the call arguments (line 124)
        str_60115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 23), 'str', '-f')
        # Processing the call keyword arguments (line 124)
        kwargs_60116 = {}
        # Getting the type of 'opt' (line 124)
        opt_60113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'opt', False)
        # Obtaining the member 'append' of a type (line 124)
        append_60114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 12), opt_60113, 'append')
        # Calling append(args, kwargs) (line 124)
        append_call_result_60117 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), append_60114, *[str_60115], **kwargs_60116)
        
        
        # Getting the type of 'v' (line 125)
        v_60118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 15), 'v')
        # Testing the type of an if condition (line 125)
        if_condition_60119 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 12), v_60118)
        # Assigning a type to the variable 'if_condition_60119' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'if_condition_60119', if_condition_60119)
        # SSA begins for if statement (line 125)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'v' (line 126)
        v_60120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 19), 'v')
        str_60121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 22), 'str', '4.6')
        # Applying the binary operator '<=' (line 126)
        result_le_60122 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 19), '<=', v_60120, str_60121)
        
        # Testing the type of an if condition (line 126)
        if_condition_60123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 16), result_le_60122)
        # Assigning a type to the variable 'if_condition_60123' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'if_condition_60123', if_condition_60123)
        # SSA begins for if statement (line 126)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 127)
        # Processing the call arguments (line 127)
        str_60126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 31), 'str', '-B108')
        # Processing the call keyword arguments (line 127)
        kwargs_60127 = {}
        # Getting the type of 'opt' (line 127)
        opt_60124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 20), 'opt', False)
        # Obtaining the member 'append' of a type (line 127)
        append_60125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 20), opt_60124, 'append')
        # Calling append(args, kwargs) (line 127)
        append_call_result_60128 = invoke(stypy.reporting.localization.Localization(__file__, 127, 20), append_60125, *[str_60126], **kwargs_60127)
        
        # SSA branch for the else part of an if statement (line 126)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 131)
        # Processing the call arguments (line 131)
        str_60131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 31), 'str', '-N15')
        # Processing the call keyword arguments (line 131)
        kwargs_60132 = {}
        # Getting the type of 'opt' (line 131)
        opt_60129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 20), 'opt', False)
        # Obtaining the member 'append' of a type (line 131)
        append_60130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 20), opt_60129, 'append')
        # Calling append(args, kwargs) (line 131)
        append_call_result_60133 = invoke(stypy.reporting.localization.Localization(__file__, 131, 20), append_60130, *[str_60131], **kwargs_60132)
        
        # SSA join for if statement (line 126)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 125)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 120)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'opt' (line 132)
        opt_60134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 15), 'opt')
        # Assigning a type to the variable 'stypy_return_type' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'stypy_return_type', opt_60134)
        
        # ################# End of 'get_flags_f77(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_f77' in the type store
        # Getting the type of 'stypy_return_type' (line 116)
        stypy_return_type_60135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60135)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_f77'
        return stypy_return_type_60135


    @norecursion
    def get_flags_f90(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_f90'
        module_type_store = module_type_store.open_function_context('get_flags_f90', 134, 4, False)
        # Assigning a type to the variable 'self' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AbsoftFCompiler.get_flags_f90.__dict__.__setitem__('stypy_localization', localization)
        AbsoftFCompiler.get_flags_f90.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AbsoftFCompiler.get_flags_f90.__dict__.__setitem__('stypy_type_store', module_type_store)
        AbsoftFCompiler.get_flags_f90.__dict__.__setitem__('stypy_function_name', 'AbsoftFCompiler.get_flags_f90')
        AbsoftFCompiler.get_flags_f90.__dict__.__setitem__('stypy_param_names_list', [])
        AbsoftFCompiler.get_flags_f90.__dict__.__setitem__('stypy_varargs_param_name', None)
        AbsoftFCompiler.get_flags_f90.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AbsoftFCompiler.get_flags_f90.__dict__.__setitem__('stypy_call_defaults', defaults)
        AbsoftFCompiler.get_flags_f90.__dict__.__setitem__('stypy_call_varargs', varargs)
        AbsoftFCompiler.get_flags_f90.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AbsoftFCompiler.get_flags_f90.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AbsoftFCompiler.get_flags_f90', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_f90', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_f90(...)' code ##################

        
        # Assigning a Call to a Name (line 135):
        
        # Call to get_flags_f90(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'self' (line 135)
        self_60138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 38), 'self', False)
        # Processing the call keyword arguments (line 135)
        kwargs_60139 = {}
        # Getting the type of 'FCompiler' (line 135)
        FCompiler_60136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 14), 'FCompiler', False)
        # Obtaining the member 'get_flags_f90' of a type (line 135)
        get_flags_f90_60137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 14), FCompiler_60136, 'get_flags_f90')
        # Calling get_flags_f90(args, kwargs) (line 135)
        get_flags_f90_call_result_60140 = invoke(stypy.reporting.localization.Localization(__file__, 135, 14), get_flags_f90_60137, *[self_60138], **kwargs_60139)
        
        # Assigning a type to the variable 'opt' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'opt', get_flags_f90_call_result_60140)
        
        # Call to extend(...): (line 136)
        # Processing the call arguments (line 136)
        
        # Obtaining an instance of the builtin type 'list' (line 136)
        list_60143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 136)
        # Adding element type (line 136)
        str_60144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 20), 'str', '-YCFRL=1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 19), list_60143, str_60144)
        # Adding element type (line 136)
        str_60145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 32), 'str', '-YCOM_NAMES=LCS')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 19), list_60143, str_60145)
        # Adding element type (line 136)
        str_60146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 51), 'str', '-YCOM_PFX')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 19), list_60143, str_60146)
        # Adding element type (line 136)
        str_60147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 64), 'str', '-YEXT_PFX')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 19), list_60143, str_60147)
        # Adding element type (line 136)
        str_60148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 20), 'str', '-YCOM_SFX=_')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 19), list_60143, str_60148)
        # Adding element type (line 136)
        str_60149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 35), 'str', '-YEXT_SFX=_')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 19), list_60143, str_60149)
        # Adding element type (line 136)
        str_60150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 50), 'str', '-YEXT_NAMES=LCS')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 19), list_60143, str_60150)
        
        # Processing the call keyword arguments (line 136)
        kwargs_60151 = {}
        # Getting the type of 'opt' (line 136)
        opt_60141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'opt', False)
        # Obtaining the member 'extend' of a type (line 136)
        extend_60142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), opt_60141, 'extend')
        # Calling extend(args, kwargs) (line 136)
        extend_call_result_60152 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), extend_60142, *[list_60143], **kwargs_60151)
        
        
        
        # Call to get_version(...): (line 138)
        # Processing the call keyword arguments (line 138)
        kwargs_60155 = {}
        # Getting the type of 'self' (line 138)
        self_60153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 11), 'self', False)
        # Obtaining the member 'get_version' of a type (line 138)
        get_version_60154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 11), self_60153, 'get_version')
        # Calling get_version(args, kwargs) (line 138)
        get_version_call_result_60156 = invoke(stypy.reporting.localization.Localization(__file__, 138, 11), get_version_60154, *[], **kwargs_60155)
        
        # Testing the type of an if condition (line 138)
        if_condition_60157 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 138, 8), get_version_call_result_60156)
        # Assigning a type to the variable 'if_condition_60157' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'if_condition_60157', if_condition_60157)
        # SSA begins for if statement (line 138)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to get_version(...): (line 139)
        # Processing the call keyword arguments (line 139)
        kwargs_60160 = {}
        # Getting the type of 'self' (line 139)
        self_60158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 15), 'self', False)
        # Obtaining the member 'get_version' of a type (line 139)
        get_version_60159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 15), self_60158, 'get_version')
        # Calling get_version(args, kwargs) (line 139)
        get_version_call_result_60161 = invoke(stypy.reporting.localization.Localization(__file__, 139, 15), get_version_60159, *[], **kwargs_60160)
        
        str_60162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 34), 'str', '4.6')
        # Applying the binary operator '>' (line 139)
        result_gt_60163 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 15), '>', get_version_call_result_60161, str_60162)
        
        # Testing the type of an if condition (line 139)
        if_condition_60164 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 12), result_gt_60163)
        # Assigning a type to the variable 'if_condition_60164' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'if_condition_60164', if_condition_60164)
        # SSA begins for if statement (line 139)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 140)
        # Processing the call arguments (line 140)
        
        # Obtaining an instance of the builtin type 'list' (line 140)
        list_60167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 140)
        # Adding element type (line 140)
        str_60168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 28), 'str', '-YDEALLOC=ALL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 27), list_60167, str_60168)
        
        # Processing the call keyword arguments (line 140)
        kwargs_60169 = {}
        # Getting the type of 'opt' (line 140)
        opt_60165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), 'opt', False)
        # Obtaining the member 'extend' of a type (line 140)
        extend_60166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 16), opt_60165, 'extend')
        # Calling extend(args, kwargs) (line 140)
        extend_call_result_60170 = invoke(stypy.reporting.localization.Localization(__file__, 140, 16), extend_60166, *[list_60167], **kwargs_60169)
        
        # SSA join for if statement (line 139)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 138)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'opt' (line 141)
        opt_60171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'opt')
        # Assigning a type to the variable 'stypy_return_type' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'stypy_return_type', opt_60171)
        
        # ################# End of 'get_flags_f90(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_f90' in the type store
        # Getting the type of 'stypy_return_type' (line 134)
        stypy_return_type_60172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60172)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_f90'
        return stypy_return_type_60172


    @norecursion
    def get_flags_fix(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_fix'
        module_type_store = module_type_store.open_function_context('get_flags_fix', 143, 4, False)
        # Assigning a type to the variable 'self' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AbsoftFCompiler.get_flags_fix.__dict__.__setitem__('stypy_localization', localization)
        AbsoftFCompiler.get_flags_fix.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AbsoftFCompiler.get_flags_fix.__dict__.__setitem__('stypy_type_store', module_type_store)
        AbsoftFCompiler.get_flags_fix.__dict__.__setitem__('stypy_function_name', 'AbsoftFCompiler.get_flags_fix')
        AbsoftFCompiler.get_flags_fix.__dict__.__setitem__('stypy_param_names_list', [])
        AbsoftFCompiler.get_flags_fix.__dict__.__setitem__('stypy_varargs_param_name', None)
        AbsoftFCompiler.get_flags_fix.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AbsoftFCompiler.get_flags_fix.__dict__.__setitem__('stypy_call_defaults', defaults)
        AbsoftFCompiler.get_flags_fix.__dict__.__setitem__('stypy_call_varargs', varargs)
        AbsoftFCompiler.get_flags_fix.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AbsoftFCompiler.get_flags_fix.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AbsoftFCompiler.get_flags_fix', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_fix', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_fix(...)' code ##################

        
        # Assigning a Call to a Name (line 144):
        
        # Call to get_flags_fix(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'self' (line 144)
        self_60175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 38), 'self', False)
        # Processing the call keyword arguments (line 144)
        kwargs_60176 = {}
        # Getting the type of 'FCompiler' (line 144)
        FCompiler_60173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 14), 'FCompiler', False)
        # Obtaining the member 'get_flags_fix' of a type (line 144)
        get_flags_fix_60174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 14), FCompiler_60173, 'get_flags_fix')
        # Calling get_flags_fix(args, kwargs) (line 144)
        get_flags_fix_call_result_60177 = invoke(stypy.reporting.localization.Localization(__file__, 144, 14), get_flags_fix_60174, *[self_60175], **kwargs_60176)
        
        # Assigning a type to the variable 'opt' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'opt', get_flags_fix_call_result_60177)
        
        # Call to extend(...): (line 145)
        # Processing the call arguments (line 145)
        
        # Obtaining an instance of the builtin type 'list' (line 145)
        list_60180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 145)
        # Adding element type (line 145)
        str_60181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 20), 'str', '-YCFRL=1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 19), list_60180, str_60181)
        # Adding element type (line 145)
        str_60182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 32), 'str', '-YCOM_NAMES=LCS')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 19), list_60180, str_60182)
        # Adding element type (line 145)
        str_60183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 51), 'str', '-YCOM_PFX')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 19), list_60180, str_60183)
        # Adding element type (line 145)
        str_60184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 64), 'str', '-YEXT_PFX')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 19), list_60180, str_60184)
        # Adding element type (line 145)
        str_60185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 20), 'str', '-YCOM_SFX=_')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 19), list_60180, str_60185)
        # Adding element type (line 145)
        str_60186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 35), 'str', '-YEXT_SFX=_')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 19), list_60180, str_60186)
        # Adding element type (line 145)
        str_60187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 50), 'str', '-YEXT_NAMES=LCS')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 19), list_60180, str_60187)
        
        # Processing the call keyword arguments (line 145)
        kwargs_60188 = {}
        # Getting the type of 'opt' (line 145)
        opt_60178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'opt', False)
        # Obtaining the member 'extend' of a type (line 145)
        extend_60179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), opt_60178, 'extend')
        # Calling extend(args, kwargs) (line 145)
        extend_call_result_60189 = invoke(stypy.reporting.localization.Localization(__file__, 145, 8), extend_60179, *[list_60180], **kwargs_60188)
        
        
        # Call to extend(...): (line 147)
        # Processing the call arguments (line 147)
        
        # Obtaining an instance of the builtin type 'list' (line 147)
        list_60192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 147)
        # Adding element type (line 147)
        str_60193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 20), 'str', '-f')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 19), list_60192, str_60193)
        # Adding element type (line 147)
        str_60194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 26), 'str', 'fixed')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 19), list_60192, str_60194)
        
        # Processing the call keyword arguments (line 147)
        kwargs_60195 = {}
        # Getting the type of 'opt' (line 147)
        opt_60190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'opt', False)
        # Obtaining the member 'extend' of a type (line 147)
        extend_60191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), opt_60190, 'extend')
        # Calling extend(args, kwargs) (line 147)
        extend_call_result_60196 = invoke(stypy.reporting.localization.Localization(__file__, 147, 8), extend_60191, *[list_60192], **kwargs_60195)
        
        # Getting the type of 'opt' (line 148)
        opt_60197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 15), 'opt')
        # Assigning a type to the variable 'stypy_return_type' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'stypy_return_type', opt_60197)
        
        # ################# End of 'get_flags_fix(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_fix' in the type store
        # Getting the type of 'stypy_return_type' (line 143)
        stypy_return_type_60198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60198)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_fix'
        return stypy_return_type_60198


    @norecursion
    def get_flags_opt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_opt'
        module_type_store = module_type_store.open_function_context('get_flags_opt', 150, 4, False)
        # Assigning a type to the variable 'self' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AbsoftFCompiler.get_flags_opt.__dict__.__setitem__('stypy_localization', localization)
        AbsoftFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AbsoftFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_store', module_type_store)
        AbsoftFCompiler.get_flags_opt.__dict__.__setitem__('stypy_function_name', 'AbsoftFCompiler.get_flags_opt')
        AbsoftFCompiler.get_flags_opt.__dict__.__setitem__('stypy_param_names_list', [])
        AbsoftFCompiler.get_flags_opt.__dict__.__setitem__('stypy_varargs_param_name', None)
        AbsoftFCompiler.get_flags_opt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AbsoftFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_defaults', defaults)
        AbsoftFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_varargs', varargs)
        AbsoftFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AbsoftFCompiler.get_flags_opt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AbsoftFCompiler.get_flags_opt', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_opt', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_opt(...)' code ##################

        
        # Assigning a List to a Name (line 151):
        
        # Obtaining an instance of the builtin type 'list' (line 151)
        list_60199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 151)
        # Adding element type (line 151)
        str_60200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 15), 'str', '-O')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 14), list_60199, str_60200)
        
        # Assigning a type to the variable 'opt' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'opt', list_60199)
        # Getting the type of 'opt' (line 152)
        opt_60201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 15), 'opt')
        # Assigning a type to the variable 'stypy_return_type' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'stypy_return_type', opt_60201)
        
        # ################# End of 'get_flags_opt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_opt' in the type store
        # Getting the type of 'stypy_return_type' (line 150)
        stypy_return_type_60202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60202)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_opt'
        return stypy_return_type_60202


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 18, 0, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AbsoftFCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'AbsoftFCompiler' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'AbsoftFCompiler', AbsoftFCompiler)

# Assigning a Str to a Name (line 20):
str_60203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 20), 'str', 'absoft')
# Getting the type of 'AbsoftFCompiler'
AbsoftFCompiler_60204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'AbsoftFCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), AbsoftFCompiler_60204, 'compiler_type', str_60203)

# Assigning a Str to a Name (line 21):
str_60205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 18), 'str', 'Absoft Corp Fortran Compiler')
# Getting the type of 'AbsoftFCompiler'
AbsoftFCompiler_60206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'AbsoftFCompiler')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), AbsoftFCompiler_60206, 'description', str_60205)

# Assigning a BinOp to a Name (line 23):
str_60207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 22), 'str', '(f90:.*?(Absoft Pro FORTRAN Version|FORTRAN 77 Compiler|Absoft Fortran Compiler Version|Copyright Absoft Corporation.*?Version))')
str_60208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 23), 'str', ' (?P<version>[^\\s*,]*)(.*?Absoft Corp|)')
# Applying the binary operator '+' (line 23)
result_add_60209 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 22), '+', str_60207, str_60208)

# Getting the type of 'AbsoftFCompiler'
AbsoftFCompiler_60210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'AbsoftFCompiler')
# Setting the type of the member 'version_pattern' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), AbsoftFCompiler_60210, 'version_pattern', result_add_60209)

# Assigning a Dict to a Name (line 33):

# Obtaining an instance of the builtin type 'dict' (line 33)
dict_60211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 33)
# Adding element type (key, value) (line 33)
str_60212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 8), 'str', 'version_cmd')
# Getting the type of 'None' (line 34)
None_60213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 25), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 18), dict_60211, (str_60212, None_60213))
# Adding element type (key, value) (line 33)
str_60214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 8), 'str', 'compiler_f77')

# Obtaining an instance of the builtin type 'list' (line 35)
list_60215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 35)
# Adding element type (line 35)
str_60216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 26), 'str', 'f77')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 25), list_60215, str_60216)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 18), dict_60211, (str_60214, list_60215))
# Adding element type (key, value) (line 33)
str_60217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 8), 'str', 'compiler_fix')

# Obtaining an instance of the builtin type 'list' (line 36)
list_60218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 36)
# Adding element type (line 36)
str_60219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 26), 'str', 'f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 25), list_60218, str_60219)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 18), dict_60211, (str_60217, list_60218))
# Adding element type (key, value) (line 33)
str_60220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 8), 'str', 'compiler_f90')

# Obtaining an instance of the builtin type 'list' (line 37)
list_60221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 37)
# Adding element type (line 37)
str_60222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 26), 'str', 'f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 25), list_60221, str_60222)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 18), dict_60211, (str_60220, list_60221))
# Adding element type (key, value) (line 33)
str_60223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 8), 'str', 'linker_so')

# Obtaining an instance of the builtin type 'list' (line 38)
list_60224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 38)
# Adding element type (line 38)
str_60225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 26), 'str', '<F90>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 25), list_60224, str_60225)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 18), dict_60211, (str_60223, list_60224))
# Adding element type (key, value) (line 33)
str_60226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 8), 'str', 'archiver')

# Obtaining an instance of the builtin type 'list' (line 39)
list_60227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 39)
# Adding element type (line 39)
str_60228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 26), 'str', 'ar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 25), list_60227, str_60228)
# Adding element type (line 39)
str_60229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 32), 'str', '-cr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 25), list_60227, str_60229)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 18), dict_60211, (str_60226, list_60227))
# Adding element type (key, value) (line 33)
str_60230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 8), 'str', 'ranlib')

# Obtaining an instance of the builtin type 'list' (line 40)
list_60231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 40)
# Adding element type (line 40)
str_60232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 26), 'str', 'ranlib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 25), list_60231, str_60232)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 18), dict_60211, (str_60230, list_60231))

# Getting the type of 'AbsoftFCompiler'
AbsoftFCompiler_60233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'AbsoftFCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), AbsoftFCompiler_60233, 'executables', dict_60211)


# Getting the type of 'os' (line 43)
os_60234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 7), 'os')
# Obtaining the member 'name' of a type (line 43)
name_60235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 7), os_60234, 'name')
str_60236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 16), 'str', 'nt')
# Applying the binary operator '==' (line 43)
result_eq_60237 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 7), '==', name_60235, str_60236)

# Testing the type of an if condition (line 43)
if_condition_60238 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 4), result_eq_60237)
# Assigning a type to the variable 'if_condition_60238' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'if_condition_60238', if_condition_60238)
# SSA begins for if statement (line 43)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Str to a Name (line 44):
str_60239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 25), 'str', '/out:')
# Assigning a type to the variable 'library_switch' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'library_switch', str_60239)
# SSA join for if statement (line 43)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Name to a Name (line 46):
# Getting the type of 'None' (line 46)
None_60240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 24), 'None')
# Getting the type of 'AbsoftFCompiler'
AbsoftFCompiler_60241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'AbsoftFCompiler')
# Setting the type of the member 'module_dir_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), AbsoftFCompiler_60241, 'module_dir_switch', None_60240)

# Assigning a Str to a Name (line 47):
str_60242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 28), 'str', '-p')
# Getting the type of 'AbsoftFCompiler'
AbsoftFCompiler_60243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'AbsoftFCompiler')
# Setting the type of the member 'module_include_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), AbsoftFCompiler_60243, 'module_include_switch', str_60242)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 155, 4))
    
    # 'from distutils import log' statement (line 155)
    from distutils import log

    import_from_module(stypy.reporting.localization.Localization(__file__, 155, 4), 'distutils', None, module_type_store, ['log'], [log])
    
    
    # Call to set_verbosity(...): (line 156)
    # Processing the call arguments (line 156)
    int_60246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 22), 'int')
    # Processing the call keyword arguments (line 156)
    kwargs_60247 = {}
    # Getting the type of 'log' (line 156)
    log_60244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'log', False)
    # Obtaining the member 'set_verbosity' of a type (line 156)
    set_verbosity_60245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 4), log_60244, 'set_verbosity')
    # Calling set_verbosity(args, kwargs) (line 156)
    set_verbosity_call_result_60248 = invoke(stypy.reporting.localization.Localization(__file__, 156, 4), set_verbosity_60245, *[int_60246], **kwargs_60247)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 157, 4))
    
    # 'from numpy.distutils.fcompiler import new_fcompiler' statement (line 157)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
    import_60249 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 157, 4), 'numpy.distutils.fcompiler')

    if (type(import_60249) is not StypyTypeError):

        if (import_60249 != 'pyd_module'):
            __import__(import_60249)
            sys_modules_60250 = sys.modules[import_60249]
            import_from_module(stypy.reporting.localization.Localization(__file__, 157, 4), 'numpy.distutils.fcompiler', sys_modules_60250.module_type_store, module_type_store, ['new_fcompiler'])
            nest_module(stypy.reporting.localization.Localization(__file__, 157, 4), __file__, sys_modules_60250, sys_modules_60250.module_type_store, module_type_store)
        else:
            from numpy.distutils.fcompiler import new_fcompiler

            import_from_module(stypy.reporting.localization.Localization(__file__, 157, 4), 'numpy.distutils.fcompiler', None, module_type_store, ['new_fcompiler'], [new_fcompiler])

    else:
        # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'numpy.distutils.fcompiler', import_60249)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
    
    
    # Assigning a Call to a Name (line 158):
    
    # Call to new_fcompiler(...): (line 158)
    # Processing the call keyword arguments (line 158)
    str_60252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 38), 'str', 'absoft')
    keyword_60253 = str_60252
    kwargs_60254 = {'compiler': keyword_60253}
    # Getting the type of 'new_fcompiler' (line 158)
    new_fcompiler_60251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 15), 'new_fcompiler', False)
    # Calling new_fcompiler(args, kwargs) (line 158)
    new_fcompiler_call_result_60255 = invoke(stypy.reporting.localization.Localization(__file__, 158, 15), new_fcompiler_60251, *[], **kwargs_60254)
    
    # Assigning a type to the variable 'compiler' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'compiler', new_fcompiler_call_result_60255)
    
    # Call to customize(...): (line 159)
    # Processing the call keyword arguments (line 159)
    kwargs_60258 = {}
    # Getting the type of 'compiler' (line 159)
    compiler_60256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'compiler', False)
    # Obtaining the member 'customize' of a type (line 159)
    customize_60257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 4), compiler_60256, 'customize')
    # Calling customize(args, kwargs) (line 159)
    customize_call_result_60259 = invoke(stypy.reporting.localization.Localization(__file__, 159, 4), customize_60257, *[], **kwargs_60258)
    
    
    # Call to print(...): (line 160)
    # Processing the call arguments (line 160)
    
    # Call to get_version(...): (line 160)
    # Processing the call keyword arguments (line 160)
    kwargs_60263 = {}
    # Getting the type of 'compiler' (line 160)
    compiler_60261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 10), 'compiler', False)
    # Obtaining the member 'get_version' of a type (line 160)
    get_version_60262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 10), compiler_60261, 'get_version')
    # Calling get_version(args, kwargs) (line 160)
    get_version_call_result_60264 = invoke(stypy.reporting.localization.Localization(__file__, 160, 10), get_version_60262, *[], **kwargs_60263)
    
    # Processing the call keyword arguments (line 160)
    kwargs_60265 = {}
    # Getting the type of 'print' (line 160)
    print_60260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'print', False)
    # Calling print(args, kwargs) (line 160)
    print_call_result_60266 = invoke(stypy.reporting.localization.Localization(__file__, 160, 4), print_60260, *[get_version_call_result_60264], **kwargs_60265)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
