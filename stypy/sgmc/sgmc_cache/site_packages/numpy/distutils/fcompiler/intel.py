
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://developer.intel.com/software/products/compilers/flin/
2: from __future__ import division, absolute_import, print_function
3: 
4: import sys
5: 
6: from numpy.distutils.ccompiler import simple_version_match
7: from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file
8: 
9: compilers = ['IntelFCompiler', 'IntelVisualFCompiler',
10:              'IntelItaniumFCompiler', 'IntelItaniumVisualFCompiler',
11:              'IntelEM64VisualFCompiler', 'IntelEM64TFCompiler']
12: 
13: 
14: def intel_version_match(type):
15:     # Match against the important stuff in the version string
16:     return simple_version_match(start=r'Intel.*?Fortran.*?(?:%s).*?Version' % (type,))
17: 
18: 
19: class BaseIntelFCompiler(FCompiler):
20:     def update_executables(self):
21:         f = dummy_fortran_file()
22:         self.executables['version_cmd'] = ['<F77>', '-FI', '-V', '-c',
23:                                            f + '.f', '-o', f + '.o']
24: 
25:     def runtime_library_dir_option(self, dir):
26:         return '-Wl,-rpath="%s"' % dir
27: 
28: 
29: class IntelFCompiler(BaseIntelFCompiler):
30: 
31:     compiler_type = 'intel'
32:     compiler_aliases = ('ifort',)
33:     description = 'Intel Fortran Compiler for 32-bit apps'
34:     version_match = intel_version_match('32-bit|IA-32')
35: 
36:     possible_executables = ['ifort', 'ifc']
37: 
38:     executables = {
39:         'version_cmd'  : None,          # set by update_executables
40:         'compiler_f77' : [None, "-72", "-w90", "-w95"],
41:         'compiler_f90' : [None],
42:         'compiler_fix' : [None, "-FI"],
43:         'linker_so'    : ["<F90>", "-shared"],
44:         'archiver'     : ["ar", "-cr"],
45:         'ranlib'       : ["ranlib"]
46:         }
47: 
48:     pic_flags = ['-fPIC']
49:     module_dir_switch = '-module '  # Don't remove ending space!
50:     module_include_switch = '-I'
51: 
52:     def get_flags_free(self):
53:         return ['-FR']
54: 
55:     def get_flags(self):
56:         return ['-fPIC']
57: 
58:     def get_flags_opt(self):  # Scipy test failures with -O2
59:         return ['-xhost -openmp -fp-model strict -O1']
60: 
61:     def get_flags_arch(self):
62:         return []
63: 
64:     def get_flags_linker_so(self):
65:         opt = FCompiler.get_flags_linker_so(self)
66:         v = self.get_version()
67:         if v and v >= '8.0':
68:             opt.append('-nofor_main')
69:         if sys.platform == 'darwin':
70:             # Here, it's -dynamiclib
71:             try:
72:                 idx = opt.index('-shared')
73:                 opt.remove('-shared')
74:             except ValueError:
75:                 idx = 0
76:             opt[idx:idx] = ['-dynamiclib', '-Wl,-undefined,dynamic_lookup']
77:         return opt
78: 
79: 
80: class IntelItaniumFCompiler(IntelFCompiler):
81:     compiler_type = 'intele'
82:     compiler_aliases = ()
83:     description = 'Intel Fortran Compiler for Itanium apps'
84: 
85:     version_match = intel_version_match('Itanium|IA-64')
86: 
87:     possible_executables = ['ifort', 'efort', 'efc']
88: 
89:     executables = {
90:         'version_cmd'  : None,
91:         'compiler_f77' : [None, "-FI", "-w90", "-w95"],
92:         'compiler_fix' : [None, "-FI"],
93:         'compiler_f90' : [None],
94:         'linker_so'    : ['<F90>', "-shared"],
95:         'archiver'     : ["ar", "-cr"],
96:         'ranlib'       : ["ranlib"]
97:         }
98: 
99: 
100: class IntelEM64TFCompiler(IntelFCompiler):
101:     compiler_type = 'intelem'
102:     compiler_aliases = ()
103:     description = 'Intel Fortran Compiler for 64-bit apps'
104: 
105:     version_match = intel_version_match('EM64T-based|Intel\\(R\\) 64|64|IA-64|64-bit')
106: 
107:     possible_executables = ['ifort', 'efort', 'efc']
108: 
109:     executables = {
110:         'version_cmd'  : None,
111:         'compiler_f77' : [None, "-FI"],
112:         'compiler_fix' : [None, "-FI"],
113:         'compiler_f90' : [None],
114:         'linker_so'    : ['<F90>', "-shared"],
115:         'archiver'     : ["ar", "-cr"],
116:         'ranlib'       : ["ranlib"]
117:         }
118: 
119:     def get_flags(self):
120:         return ['-fPIC']
121: 
122:     def get_flags_opt(self):  # Scipy test failures with -O2
123:         return ['-openmp -fp-model strict -O1']
124: 
125:     def get_flags_arch(self):
126:         return ['-xSSE4.2']
127: 
128: # Is there no difference in the version string between the above compilers
129: # and the Visual compilers?
130: 
131: 
132: class IntelVisualFCompiler(BaseIntelFCompiler):
133:     compiler_type = 'intelv'
134:     description = 'Intel Visual Fortran Compiler for 32-bit apps'
135:     version_match = intel_version_match('32-bit|IA-32')
136: 
137:     def update_executables(self):
138:         f = dummy_fortran_file()
139:         self.executables['version_cmd'] = ['<F77>', '/FI', '/c',
140:                                            f + '.f', '/o', f + '.o']
141: 
142:     ar_exe = 'lib.exe'
143:     possible_executables = ['ifort', 'ifl']
144: 
145:     executables = {
146:         'version_cmd'  : None,
147:         'compiler_f77' : [None],
148:         'compiler_fix' : [None],
149:         'compiler_f90' : [None],
150:         'linker_so'    : [None],
151:         'archiver'     : [ar_exe, "/verbose", "/OUT:"],
152:         'ranlib'       : None
153:         }
154: 
155:     compile_switch = '/c '
156:     object_switch = '/Fo'     # No space after /Fo!
157:     library_switch = '/OUT:'  # No space after /OUT:!
158:     module_dir_switch = '/module:'  # No space after /module:
159:     module_include_switch = '/I'
160: 
161:     def get_flags(self):
162:         opt = ['/nologo', '/MD', '/nbs', '/names:lowercase', '/assume:underscore']
163:         return opt
164: 
165:     def get_flags_free(self):
166:         return []
167: 
168:     def get_flags_debug(self):
169:         return ['/4Yb', '/d2']
170: 
171:     def get_flags_opt(self):
172:         return ['/O1']  # Scipy test failures with /O2
173: 
174:     def get_flags_arch(self):
175:         return ["/arch:IA32", "/QaxSSE3"]
176: 
177:     def runtime_library_dir_option(self, dir):
178:         raise NotImplementedError
179: 
180: 
181: class IntelItaniumVisualFCompiler(IntelVisualFCompiler):
182:     compiler_type = 'intelev'
183:     description = 'Intel Visual Fortran Compiler for Itanium apps'
184: 
185:     version_match = intel_version_match('Itanium')
186: 
187:     possible_executables = ['efl']  # XXX this is a wild guess
188:     ar_exe = IntelVisualFCompiler.ar_exe
189: 
190:     executables = {
191:         'version_cmd'  : None,
192:         'compiler_f77' : [None, "-FI", "-w90", "-w95"],
193:         'compiler_fix' : [None, "-FI", "-4L72", "-w"],
194:         'compiler_f90' : [None],
195:         'linker_so'    : ['<F90>', "-shared"],
196:         'archiver'     : [ar_exe, "/verbose", "/OUT:"],
197:         'ranlib'       : None
198:         }
199: 
200: 
201: class IntelEM64VisualFCompiler(IntelVisualFCompiler):
202:     compiler_type = 'intelvem'
203:     description = 'Intel Visual Fortran Compiler for 64-bit apps'
204: 
205:     version_match = simple_version_match(start='Intel\(R\).*?64,')
206: 
207:     def get_flags_arch(self):
208:         return ['/QaxSSE4.2']
209: 
210: 
211: if __name__ == '__main__':
212:     from distutils import log
213:     log.set_verbosity(2)
214:     from numpy.distutils.fcompiler import new_fcompiler
215:     compiler = new_fcompiler(compiler='intel')
216:     compiler.customize()
217:     print(compiler.get_version())
218: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.distutils.ccompiler import simple_version_match' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_62194 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.distutils.ccompiler')

if (type(import_62194) is not StypyTypeError):

    if (import_62194 != 'pyd_module'):
        __import__(import_62194)
        sys_modules_62195 = sys.modules[import_62194]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.distutils.ccompiler', sys_modules_62195.module_type_store, module_type_store, ['simple_version_match'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_62195, sys_modules_62195.module_type_store, module_type_store)
    else:
        from numpy.distutils.ccompiler import simple_version_match

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.distutils.ccompiler', None, module_type_store, ['simple_version_match'], [simple_version_match])

else:
    # Assigning a type to the variable 'numpy.distutils.ccompiler' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.distutils.ccompiler', import_62194)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_62196 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.distutils.fcompiler')

if (type(import_62196) is not StypyTypeError):

    if (import_62196 != 'pyd_module'):
        __import__(import_62196)
        sys_modules_62197 = sys.modules[import_62196]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.distutils.fcompiler', sys_modules_62197.module_type_store, module_type_store, ['FCompiler', 'dummy_fortran_file'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_62197, sys_modules_62197.module_type_store, module_type_store)
    else:
        from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.distutils.fcompiler', None, module_type_store, ['FCompiler', 'dummy_fortran_file'], [FCompiler, dummy_fortran_file])

else:
    # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.distutils.fcompiler', import_62196)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')


# Assigning a List to a Name (line 9):

# Obtaining an instance of the builtin type 'list' (line 9)
list_62198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
str_62199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 13), 'str', 'IntelFCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 12), list_62198, str_62199)
# Adding element type (line 9)
str_62200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 31), 'str', 'IntelVisualFCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 12), list_62198, str_62200)
# Adding element type (line 9)
str_62201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 13), 'str', 'IntelItaniumFCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 12), list_62198, str_62201)
# Adding element type (line 9)
str_62202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 38), 'str', 'IntelItaniumVisualFCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 12), list_62198, str_62202)
# Adding element type (line 9)
str_62203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 13), 'str', 'IntelEM64VisualFCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 12), list_62198, str_62203)
# Adding element type (line 9)
str_62204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 41), 'str', 'IntelEM64TFCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 12), list_62198, str_62204)

# Assigning a type to the variable 'compilers' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'compilers', list_62198)

@norecursion
def intel_version_match(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'intel_version_match'
    module_type_store = module_type_store.open_function_context('intel_version_match', 14, 0, False)
    
    # Passed parameters checking function
    intel_version_match.stypy_localization = localization
    intel_version_match.stypy_type_of_self = None
    intel_version_match.stypy_type_store = module_type_store
    intel_version_match.stypy_function_name = 'intel_version_match'
    intel_version_match.stypy_param_names_list = ['type']
    intel_version_match.stypy_varargs_param_name = None
    intel_version_match.stypy_kwargs_param_name = None
    intel_version_match.stypy_call_defaults = defaults
    intel_version_match.stypy_call_varargs = varargs
    intel_version_match.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'intel_version_match', ['type'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'intel_version_match', localization, ['type'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'intel_version_match(...)' code ##################

    
    # Call to simple_version_match(...): (line 16)
    # Processing the call keyword arguments (line 16)
    str_62206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 38), 'str', 'Intel.*?Fortran.*?(?:%s).*?Version')
    
    # Obtaining an instance of the builtin type 'tuple' (line 16)
    tuple_62207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 79), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 16)
    # Adding element type (line 16)
    # Getting the type of 'type' (line 16)
    type_62208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 79), 'type', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 79), tuple_62207, type_62208)
    
    # Applying the binary operator '%' (line 16)
    result_mod_62209 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 38), '%', str_62206, tuple_62207)
    
    keyword_62210 = result_mod_62209
    kwargs_62211 = {'start': keyword_62210}
    # Getting the type of 'simple_version_match' (line 16)
    simple_version_match_62205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'simple_version_match', False)
    # Calling simple_version_match(args, kwargs) (line 16)
    simple_version_match_call_result_62212 = invoke(stypy.reporting.localization.Localization(__file__, 16, 11), simple_version_match_62205, *[], **kwargs_62211)
    
    # Assigning a type to the variable 'stypy_return_type' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type', simple_version_match_call_result_62212)
    
    # ################# End of 'intel_version_match(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'intel_version_match' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_62213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_62213)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'intel_version_match'
    return stypy_return_type_62213

# Assigning a type to the variable 'intel_version_match' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'intel_version_match', intel_version_match)
# Declaration of the 'BaseIntelFCompiler' class
# Getting the type of 'FCompiler' (line 19)
FCompiler_62214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 25), 'FCompiler')

class BaseIntelFCompiler(FCompiler_62214, ):

    @norecursion
    def update_executables(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update_executables'
        module_type_store = module_type_store.open_function_context('update_executables', 20, 4, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseIntelFCompiler.update_executables.__dict__.__setitem__('stypy_localization', localization)
        BaseIntelFCompiler.update_executables.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseIntelFCompiler.update_executables.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseIntelFCompiler.update_executables.__dict__.__setitem__('stypy_function_name', 'BaseIntelFCompiler.update_executables')
        BaseIntelFCompiler.update_executables.__dict__.__setitem__('stypy_param_names_list', [])
        BaseIntelFCompiler.update_executables.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseIntelFCompiler.update_executables.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseIntelFCompiler.update_executables.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseIntelFCompiler.update_executables.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseIntelFCompiler.update_executables.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseIntelFCompiler.update_executables.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseIntelFCompiler.update_executables', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 21):
        
        # Call to dummy_fortran_file(...): (line 21)
        # Processing the call keyword arguments (line 21)
        kwargs_62216 = {}
        # Getting the type of 'dummy_fortran_file' (line 21)
        dummy_fortran_file_62215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'dummy_fortran_file', False)
        # Calling dummy_fortran_file(args, kwargs) (line 21)
        dummy_fortran_file_call_result_62217 = invoke(stypy.reporting.localization.Localization(__file__, 21, 12), dummy_fortran_file_62215, *[], **kwargs_62216)
        
        # Assigning a type to the variable 'f' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'f', dummy_fortran_file_call_result_62217)
        
        # Assigning a List to a Subscript (line 22):
        
        # Obtaining an instance of the builtin type 'list' (line 22)
        list_62218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 22)
        # Adding element type (line 22)
        str_62219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 43), 'str', '<F77>')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 42), list_62218, str_62219)
        # Adding element type (line 22)
        str_62220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 52), 'str', '-FI')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 42), list_62218, str_62220)
        # Adding element type (line 22)
        str_62221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 59), 'str', '-V')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 42), list_62218, str_62221)
        # Adding element type (line 22)
        str_62222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 65), 'str', '-c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 42), list_62218, str_62222)
        # Adding element type (line 22)
        # Getting the type of 'f' (line 23)
        f_62223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 43), 'f')
        str_62224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 47), 'str', '.f')
        # Applying the binary operator '+' (line 23)
        result_add_62225 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 43), '+', f_62223, str_62224)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 42), list_62218, result_add_62225)
        # Adding element type (line 22)
        str_62226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 53), 'str', '-o')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 42), list_62218, str_62226)
        # Adding element type (line 22)
        # Getting the type of 'f' (line 23)
        f_62227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 59), 'f')
        str_62228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 63), 'str', '.o')
        # Applying the binary operator '+' (line 23)
        result_add_62229 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 59), '+', f_62227, str_62228)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 42), list_62218, result_add_62229)
        
        # Getting the type of 'self' (line 22)
        self_62230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self')
        # Obtaining the member 'executables' of a type (line 22)
        executables_62231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), self_62230, 'executables')
        str_62232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 25), 'str', 'version_cmd')
        # Storing an element on a container (line 22)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 8), executables_62231, (str_62232, list_62218))
        
        # ################# End of 'update_executables(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update_executables' in the type store
        # Getting the type of 'stypy_return_type' (line 20)
        stypy_return_type_62233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62233)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update_executables'
        return stypy_return_type_62233


    @norecursion
    def runtime_library_dir_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'runtime_library_dir_option'
        module_type_store = module_type_store.open_function_context('runtime_library_dir_option', 25, 4, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseIntelFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_localization', localization)
        BaseIntelFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseIntelFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseIntelFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_function_name', 'BaseIntelFCompiler.runtime_library_dir_option')
        BaseIntelFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_param_names_list', ['dir'])
        BaseIntelFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseIntelFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseIntelFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseIntelFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseIntelFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseIntelFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseIntelFCompiler.runtime_library_dir_option', ['dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'runtime_library_dir_option', localization, ['dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'runtime_library_dir_option(...)' code ##################

        str_62234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 15), 'str', '-Wl,-rpath="%s"')
        # Getting the type of 'dir' (line 26)
        dir_62235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 35), 'dir')
        # Applying the binary operator '%' (line 26)
        result_mod_62236 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 15), '%', str_62234, dir_62235)
        
        # Assigning a type to the variable 'stypy_return_type' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'stypy_return_type', result_mod_62236)
        
        # ################# End of 'runtime_library_dir_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'runtime_library_dir_option' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_62237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62237)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'runtime_library_dir_option'
        return stypy_return_type_62237


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 19, 0, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseIntelFCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'BaseIntelFCompiler' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'BaseIntelFCompiler', BaseIntelFCompiler)
# Declaration of the 'IntelFCompiler' class
# Getting the type of 'BaseIntelFCompiler' (line 29)
BaseIntelFCompiler_62238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 21), 'BaseIntelFCompiler')

class IntelFCompiler(BaseIntelFCompiler_62238, ):

    @norecursion
    def get_flags_free(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_free'
        module_type_store = module_type_store.open_function_context('get_flags_free', 52, 4, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntelFCompiler.get_flags_free.__dict__.__setitem__('stypy_localization', localization)
        IntelFCompiler.get_flags_free.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntelFCompiler.get_flags_free.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntelFCompiler.get_flags_free.__dict__.__setitem__('stypy_function_name', 'IntelFCompiler.get_flags_free')
        IntelFCompiler.get_flags_free.__dict__.__setitem__('stypy_param_names_list', [])
        IntelFCompiler.get_flags_free.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntelFCompiler.get_flags_free.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntelFCompiler.get_flags_free.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntelFCompiler.get_flags_free.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntelFCompiler.get_flags_free.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntelFCompiler.get_flags_free.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelFCompiler.get_flags_free', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_free', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_free(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 53)
        list_62239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 53)
        # Adding element type (line 53)
        str_62240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 16), 'str', '-FR')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 15), list_62239, str_62240)
        
        # Assigning a type to the variable 'stypy_return_type' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'stypy_return_type', list_62239)
        
        # ################# End of 'get_flags_free(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_free' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_62241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62241)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_free'
        return stypy_return_type_62241


    @norecursion
    def get_flags(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags'
        module_type_store = module_type_store.open_function_context('get_flags', 55, 4, False)
        # Assigning a type to the variable 'self' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntelFCompiler.get_flags.__dict__.__setitem__('stypy_localization', localization)
        IntelFCompiler.get_flags.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntelFCompiler.get_flags.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntelFCompiler.get_flags.__dict__.__setitem__('stypy_function_name', 'IntelFCompiler.get_flags')
        IntelFCompiler.get_flags.__dict__.__setitem__('stypy_param_names_list', [])
        IntelFCompiler.get_flags.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntelFCompiler.get_flags.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntelFCompiler.get_flags.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntelFCompiler.get_flags.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntelFCompiler.get_flags.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntelFCompiler.get_flags.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelFCompiler.get_flags', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_62242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        # Adding element type (line 56)
        str_62243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 16), 'str', '-fPIC')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 15), list_62242, str_62243)
        
        # Assigning a type to the variable 'stypy_return_type' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'stypy_return_type', list_62242)
        
        # ################# End of 'get_flags(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags' in the type store
        # Getting the type of 'stypy_return_type' (line 55)
        stypy_return_type_62244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62244)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags'
        return stypy_return_type_62244


    @norecursion
    def get_flags_opt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_opt'
        module_type_store = module_type_store.open_function_context('get_flags_opt', 58, 4, False)
        # Assigning a type to the variable 'self' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntelFCompiler.get_flags_opt.__dict__.__setitem__('stypy_localization', localization)
        IntelFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntelFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntelFCompiler.get_flags_opt.__dict__.__setitem__('stypy_function_name', 'IntelFCompiler.get_flags_opt')
        IntelFCompiler.get_flags_opt.__dict__.__setitem__('stypy_param_names_list', [])
        IntelFCompiler.get_flags_opt.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntelFCompiler.get_flags_opt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntelFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntelFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntelFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntelFCompiler.get_flags_opt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelFCompiler.get_flags_opt', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'list' (line 59)
        list_62245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 59)
        # Adding element type (line 59)
        str_62246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 16), 'str', '-xhost -openmp -fp-model strict -O1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 15), list_62245, str_62246)
        
        # Assigning a type to the variable 'stypy_return_type' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'stypy_return_type', list_62245)
        
        # ################# End of 'get_flags_opt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_opt' in the type store
        # Getting the type of 'stypy_return_type' (line 58)
        stypy_return_type_62247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62247)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_opt'
        return stypy_return_type_62247


    @norecursion
    def get_flags_arch(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_arch'
        module_type_store = module_type_store.open_function_context('get_flags_arch', 61, 4, False)
        # Assigning a type to the variable 'self' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntelFCompiler.get_flags_arch.__dict__.__setitem__('stypy_localization', localization)
        IntelFCompiler.get_flags_arch.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntelFCompiler.get_flags_arch.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntelFCompiler.get_flags_arch.__dict__.__setitem__('stypy_function_name', 'IntelFCompiler.get_flags_arch')
        IntelFCompiler.get_flags_arch.__dict__.__setitem__('stypy_param_names_list', [])
        IntelFCompiler.get_flags_arch.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntelFCompiler.get_flags_arch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntelFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntelFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntelFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntelFCompiler.get_flags_arch.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelFCompiler.get_flags_arch', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_arch', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_arch(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_62248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        
        # Assigning a type to the variable 'stypy_return_type' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'stypy_return_type', list_62248)
        
        # ################# End of 'get_flags_arch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_arch' in the type store
        # Getting the type of 'stypy_return_type' (line 61)
        stypy_return_type_62249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62249)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_arch'
        return stypy_return_type_62249


    @norecursion
    def get_flags_linker_so(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_linker_so'
        module_type_store = module_type_store.open_function_context('get_flags_linker_so', 64, 4, False)
        # Assigning a type to the variable 'self' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntelFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_localization', localization)
        IntelFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntelFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntelFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_function_name', 'IntelFCompiler.get_flags_linker_so')
        IntelFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_param_names_list', [])
        IntelFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntelFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntelFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntelFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntelFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntelFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelFCompiler.get_flags_linker_so', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 65):
        
        # Call to get_flags_linker_so(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'self' (line 65)
        self_62252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 44), 'self', False)
        # Processing the call keyword arguments (line 65)
        kwargs_62253 = {}
        # Getting the type of 'FCompiler' (line 65)
        FCompiler_62250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 14), 'FCompiler', False)
        # Obtaining the member 'get_flags_linker_so' of a type (line 65)
        get_flags_linker_so_62251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 14), FCompiler_62250, 'get_flags_linker_so')
        # Calling get_flags_linker_so(args, kwargs) (line 65)
        get_flags_linker_so_call_result_62254 = invoke(stypy.reporting.localization.Localization(__file__, 65, 14), get_flags_linker_so_62251, *[self_62252], **kwargs_62253)
        
        # Assigning a type to the variable 'opt' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'opt', get_flags_linker_so_call_result_62254)
        
        # Assigning a Call to a Name (line 66):
        
        # Call to get_version(...): (line 66)
        # Processing the call keyword arguments (line 66)
        kwargs_62257 = {}
        # Getting the type of 'self' (line 66)
        self_62255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'self', False)
        # Obtaining the member 'get_version' of a type (line 66)
        get_version_62256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), self_62255, 'get_version')
        # Calling get_version(args, kwargs) (line 66)
        get_version_call_result_62258 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), get_version_62256, *[], **kwargs_62257)
        
        # Assigning a type to the variable 'v' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'v', get_version_call_result_62258)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'v' (line 67)
        v_62259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'v')
        
        # Getting the type of 'v' (line 67)
        v_62260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'v')
        str_62261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 22), 'str', '8.0')
        # Applying the binary operator '>=' (line 67)
        result_ge_62262 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 17), '>=', v_62260, str_62261)
        
        # Applying the binary operator 'and' (line 67)
        result_and_keyword_62263 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 11), 'and', v_62259, result_ge_62262)
        
        # Testing the type of an if condition (line 67)
        if_condition_62264 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 8), result_and_keyword_62263)
        # Assigning a type to the variable 'if_condition_62264' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'if_condition_62264', if_condition_62264)
        # SSA begins for if statement (line 67)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 68)
        # Processing the call arguments (line 68)
        str_62267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 23), 'str', '-nofor_main')
        # Processing the call keyword arguments (line 68)
        kwargs_62268 = {}
        # Getting the type of 'opt' (line 68)
        opt_62265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'opt', False)
        # Obtaining the member 'append' of a type (line 68)
        append_62266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), opt_62265, 'append')
        # Calling append(args, kwargs) (line 68)
        append_call_result_62269 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), append_62266, *[str_62267], **kwargs_62268)
        
        # SSA join for if statement (line 67)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'sys' (line 69)
        sys_62270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 'sys')
        # Obtaining the member 'platform' of a type (line 69)
        platform_62271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 11), sys_62270, 'platform')
        str_62272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 27), 'str', 'darwin')
        # Applying the binary operator '==' (line 69)
        result_eq_62273 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 11), '==', platform_62271, str_62272)
        
        # Testing the type of an if condition (line 69)
        if_condition_62274 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 8), result_eq_62273)
        # Assigning a type to the variable 'if_condition_62274' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'if_condition_62274', if_condition_62274)
        # SSA begins for if statement (line 69)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 71)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 72):
        
        # Call to index(...): (line 72)
        # Processing the call arguments (line 72)
        str_62277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 32), 'str', '-shared')
        # Processing the call keyword arguments (line 72)
        kwargs_62278 = {}
        # Getting the type of 'opt' (line 72)
        opt_62275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 22), 'opt', False)
        # Obtaining the member 'index' of a type (line 72)
        index_62276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 22), opt_62275, 'index')
        # Calling index(args, kwargs) (line 72)
        index_call_result_62279 = invoke(stypy.reporting.localization.Localization(__file__, 72, 22), index_62276, *[str_62277], **kwargs_62278)
        
        # Assigning a type to the variable 'idx' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'idx', index_call_result_62279)
        
        # Call to remove(...): (line 73)
        # Processing the call arguments (line 73)
        str_62282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 27), 'str', '-shared')
        # Processing the call keyword arguments (line 73)
        kwargs_62283 = {}
        # Getting the type of 'opt' (line 73)
        opt_62280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'opt', False)
        # Obtaining the member 'remove' of a type (line 73)
        remove_62281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 16), opt_62280, 'remove')
        # Calling remove(args, kwargs) (line 73)
        remove_call_result_62284 = invoke(stypy.reporting.localization.Localization(__file__, 73, 16), remove_62281, *[str_62282], **kwargs_62283)
        
        # SSA branch for the except part of a try statement (line 71)
        # SSA branch for the except 'ValueError' branch of a try statement (line 71)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Num to a Name (line 75):
        int_62285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 22), 'int')
        # Assigning a type to the variable 'idx' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'idx', int_62285)
        # SSA join for try-except statement (line 71)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Subscript (line 76):
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_62286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        # Adding element type (line 76)
        str_62287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 28), 'str', '-dynamiclib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 27), list_62286, str_62287)
        # Adding element type (line 76)
        str_62288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 43), 'str', '-Wl,-undefined,dynamic_lookup')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 27), list_62286, str_62288)
        
        # Getting the type of 'opt' (line 76)
        opt_62289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'opt')
        # Getting the type of 'idx' (line 76)
        idx_62290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'idx')
        # Getting the type of 'idx' (line 76)
        idx_62291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 20), 'idx')
        slice_62292 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 76, 12), idx_62290, idx_62291, None)
        # Storing an element on a container (line 76)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 12), opt_62289, (slice_62292, list_62286))
        # SSA join for if statement (line 69)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'opt' (line 77)
        opt_62293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 15), 'opt')
        # Assigning a type to the variable 'stypy_return_type' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'stypy_return_type', opt_62293)
        
        # ################# End of 'get_flags_linker_so(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_linker_so' in the type store
        # Getting the type of 'stypy_return_type' (line 64)
        stypy_return_type_62294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62294)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_linker_so'
        return stypy_return_type_62294


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 29, 0, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelFCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'IntelFCompiler' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'IntelFCompiler', IntelFCompiler)

# Assigning a Str to a Name (line 31):
str_62295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 20), 'str', 'intel')
# Getting the type of 'IntelFCompiler'
IntelFCompiler_62296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelFCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelFCompiler_62296, 'compiler_type', str_62295)

# Assigning a Tuple to a Name (line 32):

# Obtaining an instance of the builtin type 'tuple' (line 32)
tuple_62297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 32)
# Adding element type (line 32)
str_62298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 24), 'str', 'ifort')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 24), tuple_62297, str_62298)

# Getting the type of 'IntelFCompiler'
IntelFCompiler_62299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelFCompiler')
# Setting the type of the member 'compiler_aliases' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelFCompiler_62299, 'compiler_aliases', tuple_62297)

# Assigning a Str to a Name (line 33):
str_62300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 18), 'str', 'Intel Fortran Compiler for 32-bit apps')
# Getting the type of 'IntelFCompiler'
IntelFCompiler_62301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelFCompiler')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelFCompiler_62301, 'description', str_62300)

# Assigning a Call to a Name (line 34):

# Call to intel_version_match(...): (line 34)
# Processing the call arguments (line 34)
str_62303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 40), 'str', '32-bit|IA-32')
# Processing the call keyword arguments (line 34)
kwargs_62304 = {}
# Getting the type of 'intel_version_match' (line 34)
intel_version_match_62302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 20), 'intel_version_match', False)
# Calling intel_version_match(args, kwargs) (line 34)
intel_version_match_call_result_62305 = invoke(stypy.reporting.localization.Localization(__file__, 34, 20), intel_version_match_62302, *[str_62303], **kwargs_62304)

# Getting the type of 'IntelFCompiler'
IntelFCompiler_62306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelFCompiler')
# Setting the type of the member 'version_match' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelFCompiler_62306, 'version_match', intel_version_match_call_result_62305)

# Assigning a List to a Name (line 36):

# Obtaining an instance of the builtin type 'list' (line 36)
list_62307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 36)
# Adding element type (line 36)
str_62308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 28), 'str', 'ifort')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 27), list_62307, str_62308)
# Adding element type (line 36)
str_62309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 37), 'str', 'ifc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 27), list_62307, str_62309)

# Getting the type of 'IntelFCompiler'
IntelFCompiler_62310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelFCompiler')
# Setting the type of the member 'possible_executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelFCompiler_62310, 'possible_executables', list_62307)

# Assigning a Dict to a Name (line 38):

# Obtaining an instance of the builtin type 'dict' (line 38)
dict_62311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 38)
# Adding element type (key, value) (line 38)
str_62312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 8), 'str', 'version_cmd')
# Getting the type of 'None' (line 39)
None_62313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 25), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 18), dict_62311, (str_62312, None_62313))
# Adding element type (key, value) (line 38)
str_62314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 8), 'str', 'compiler_f77')

# Obtaining an instance of the builtin type 'list' (line 40)
list_62315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 40)
# Adding element type (line 40)
# Getting the type of 'None' (line 40)
None_62316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 25), list_62315, None_62316)
# Adding element type (line 40)
str_62317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 32), 'str', '-72')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 25), list_62315, str_62317)
# Adding element type (line 40)
str_62318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 39), 'str', '-w90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 25), list_62315, str_62318)
# Adding element type (line 40)
str_62319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 47), 'str', '-w95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 25), list_62315, str_62319)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 18), dict_62311, (str_62314, list_62315))
# Adding element type (key, value) (line 38)
str_62320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 8), 'str', 'compiler_f90')

# Obtaining an instance of the builtin type 'list' (line 41)
list_62321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 41)
# Adding element type (line 41)
# Getting the type of 'None' (line 41)
None_62322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 25), list_62321, None_62322)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 18), dict_62311, (str_62320, list_62321))
# Adding element type (key, value) (line 38)
str_62323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 8), 'str', 'compiler_fix')

# Obtaining an instance of the builtin type 'list' (line 42)
list_62324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 42)
# Adding element type (line 42)
# Getting the type of 'None' (line 42)
None_62325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 25), list_62324, None_62325)
# Adding element type (line 42)
str_62326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 32), 'str', '-FI')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 25), list_62324, str_62326)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 18), dict_62311, (str_62323, list_62324))
# Adding element type (key, value) (line 38)
str_62327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 8), 'str', 'linker_so')

# Obtaining an instance of the builtin type 'list' (line 43)
list_62328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 43)
# Adding element type (line 43)
str_62329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 26), 'str', '<F90>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 25), list_62328, str_62329)
# Adding element type (line 43)
str_62330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 35), 'str', '-shared')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 25), list_62328, str_62330)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 18), dict_62311, (str_62327, list_62328))
# Adding element type (key, value) (line 38)
str_62331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 8), 'str', 'archiver')

# Obtaining an instance of the builtin type 'list' (line 44)
list_62332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 44)
# Adding element type (line 44)
str_62333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 26), 'str', 'ar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 25), list_62332, str_62333)
# Adding element type (line 44)
str_62334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 32), 'str', '-cr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 25), list_62332, str_62334)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 18), dict_62311, (str_62331, list_62332))
# Adding element type (key, value) (line 38)
str_62335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 8), 'str', 'ranlib')

# Obtaining an instance of the builtin type 'list' (line 45)
list_62336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 45)
# Adding element type (line 45)
str_62337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 26), 'str', 'ranlib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 25), list_62336, str_62337)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 18), dict_62311, (str_62335, list_62336))

# Getting the type of 'IntelFCompiler'
IntelFCompiler_62338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelFCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelFCompiler_62338, 'executables', dict_62311)

# Assigning a List to a Name (line 48):

# Obtaining an instance of the builtin type 'list' (line 48)
list_62339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 48)
# Adding element type (line 48)
str_62340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 17), 'str', '-fPIC')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 16), list_62339, str_62340)

# Getting the type of 'IntelFCompiler'
IntelFCompiler_62341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelFCompiler')
# Setting the type of the member 'pic_flags' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelFCompiler_62341, 'pic_flags', list_62339)

# Assigning a Str to a Name (line 49):
str_62342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 24), 'str', '-module ')
# Getting the type of 'IntelFCompiler'
IntelFCompiler_62343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelFCompiler')
# Setting the type of the member 'module_dir_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelFCompiler_62343, 'module_dir_switch', str_62342)

# Assigning a Str to a Name (line 50):
str_62344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 28), 'str', '-I')
# Getting the type of 'IntelFCompiler'
IntelFCompiler_62345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelFCompiler')
# Setting the type of the member 'module_include_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelFCompiler_62345, 'module_include_switch', str_62344)
# Declaration of the 'IntelItaniumFCompiler' class
# Getting the type of 'IntelFCompiler' (line 80)
IntelFCompiler_62346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 28), 'IntelFCompiler')

class IntelItaniumFCompiler(IntelFCompiler_62346, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 80, 0, False)
        # Assigning a type to the variable 'self' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelItaniumFCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'IntelItaniumFCompiler' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'IntelItaniumFCompiler', IntelItaniumFCompiler)

# Assigning a Str to a Name (line 81):
str_62347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 20), 'str', 'intele')
# Getting the type of 'IntelItaniumFCompiler'
IntelItaniumFCompiler_62348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelItaniumFCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelItaniumFCompiler_62348, 'compiler_type', str_62347)

# Assigning a Tuple to a Name (line 82):

# Obtaining an instance of the builtin type 'tuple' (line 82)
tuple_62349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 82)

# Getting the type of 'IntelItaniumFCompiler'
IntelItaniumFCompiler_62350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelItaniumFCompiler')
# Setting the type of the member 'compiler_aliases' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelItaniumFCompiler_62350, 'compiler_aliases', tuple_62349)

# Assigning a Str to a Name (line 83):
str_62351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 18), 'str', 'Intel Fortran Compiler for Itanium apps')
# Getting the type of 'IntelItaniumFCompiler'
IntelItaniumFCompiler_62352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelItaniumFCompiler')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelItaniumFCompiler_62352, 'description', str_62351)

# Assigning a Call to a Name (line 85):

# Call to intel_version_match(...): (line 85)
# Processing the call arguments (line 85)
str_62354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 40), 'str', 'Itanium|IA-64')
# Processing the call keyword arguments (line 85)
kwargs_62355 = {}
# Getting the type of 'intel_version_match' (line 85)
intel_version_match_62353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 20), 'intel_version_match', False)
# Calling intel_version_match(args, kwargs) (line 85)
intel_version_match_call_result_62356 = invoke(stypy.reporting.localization.Localization(__file__, 85, 20), intel_version_match_62353, *[str_62354], **kwargs_62355)

# Getting the type of 'IntelItaniumFCompiler'
IntelItaniumFCompiler_62357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelItaniumFCompiler')
# Setting the type of the member 'version_match' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelItaniumFCompiler_62357, 'version_match', intel_version_match_call_result_62356)

# Assigning a List to a Name (line 87):

# Obtaining an instance of the builtin type 'list' (line 87)
list_62358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 87)
# Adding element type (line 87)
str_62359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 28), 'str', 'ifort')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 27), list_62358, str_62359)
# Adding element type (line 87)
str_62360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 37), 'str', 'efort')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 27), list_62358, str_62360)
# Adding element type (line 87)
str_62361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 46), 'str', 'efc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 27), list_62358, str_62361)

# Getting the type of 'IntelItaniumFCompiler'
IntelItaniumFCompiler_62362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelItaniumFCompiler')
# Setting the type of the member 'possible_executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelItaniumFCompiler_62362, 'possible_executables', list_62358)

# Assigning a Dict to a Name (line 89):

# Obtaining an instance of the builtin type 'dict' (line 89)
dict_62363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 89)
# Adding element type (key, value) (line 89)
str_62364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 8), 'str', 'version_cmd')
# Getting the type of 'None' (line 90)
None_62365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 18), dict_62363, (str_62364, None_62365))
# Adding element type (key, value) (line 89)
str_62366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'str', 'compiler_f77')

# Obtaining an instance of the builtin type 'list' (line 91)
list_62367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 91)
# Adding element type (line 91)
# Getting the type of 'None' (line 91)
None_62368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 25), list_62367, None_62368)
# Adding element type (line 91)
str_62369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 32), 'str', '-FI')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 25), list_62367, str_62369)
# Adding element type (line 91)
str_62370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 39), 'str', '-w90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 25), list_62367, str_62370)
# Adding element type (line 91)
str_62371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 47), 'str', '-w95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 25), list_62367, str_62371)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 18), dict_62363, (str_62366, list_62367))
# Adding element type (key, value) (line 89)
str_62372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 8), 'str', 'compiler_fix')

# Obtaining an instance of the builtin type 'list' (line 92)
list_62373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 92)
# Adding element type (line 92)
# Getting the type of 'None' (line 92)
None_62374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 25), list_62373, None_62374)
# Adding element type (line 92)
str_62375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 32), 'str', '-FI')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 25), list_62373, str_62375)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 18), dict_62363, (str_62372, list_62373))
# Adding element type (key, value) (line 89)
str_62376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 8), 'str', 'compiler_f90')

# Obtaining an instance of the builtin type 'list' (line 93)
list_62377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 93)
# Adding element type (line 93)
# Getting the type of 'None' (line 93)
None_62378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 25), list_62377, None_62378)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 18), dict_62363, (str_62376, list_62377))
# Adding element type (key, value) (line 89)
str_62379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 8), 'str', 'linker_so')

# Obtaining an instance of the builtin type 'list' (line 94)
list_62380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 94)
# Adding element type (line 94)
str_62381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 26), 'str', '<F90>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 25), list_62380, str_62381)
# Adding element type (line 94)
str_62382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 35), 'str', '-shared')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 25), list_62380, str_62382)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 18), dict_62363, (str_62379, list_62380))
# Adding element type (key, value) (line 89)
str_62383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 8), 'str', 'archiver')

# Obtaining an instance of the builtin type 'list' (line 95)
list_62384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 95)
# Adding element type (line 95)
str_62385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 26), 'str', 'ar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 25), list_62384, str_62385)
# Adding element type (line 95)
str_62386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 32), 'str', '-cr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 25), list_62384, str_62386)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 18), dict_62363, (str_62383, list_62384))
# Adding element type (key, value) (line 89)
str_62387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 8), 'str', 'ranlib')

# Obtaining an instance of the builtin type 'list' (line 96)
list_62388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 96)
# Adding element type (line 96)
str_62389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 26), 'str', 'ranlib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 25), list_62388, str_62389)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 18), dict_62363, (str_62387, list_62388))

# Getting the type of 'IntelItaniumFCompiler'
IntelItaniumFCompiler_62390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelItaniumFCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelItaniumFCompiler_62390, 'executables', dict_62363)
# Declaration of the 'IntelEM64TFCompiler' class
# Getting the type of 'IntelFCompiler' (line 100)
IntelFCompiler_62391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 26), 'IntelFCompiler')

class IntelEM64TFCompiler(IntelFCompiler_62391, ):

    @norecursion
    def get_flags(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags'
        module_type_store = module_type_store.open_function_context('get_flags', 119, 4, False)
        # Assigning a type to the variable 'self' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntelEM64TFCompiler.get_flags.__dict__.__setitem__('stypy_localization', localization)
        IntelEM64TFCompiler.get_flags.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntelEM64TFCompiler.get_flags.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntelEM64TFCompiler.get_flags.__dict__.__setitem__('stypy_function_name', 'IntelEM64TFCompiler.get_flags')
        IntelEM64TFCompiler.get_flags.__dict__.__setitem__('stypy_param_names_list', [])
        IntelEM64TFCompiler.get_flags.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntelEM64TFCompiler.get_flags.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntelEM64TFCompiler.get_flags.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntelEM64TFCompiler.get_flags.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntelEM64TFCompiler.get_flags.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntelEM64TFCompiler.get_flags.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelEM64TFCompiler.get_flags', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'list' (line 120)
        list_62392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 120)
        # Adding element type (line 120)
        str_62393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 16), 'str', '-fPIC')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 15), list_62392, str_62393)
        
        # Assigning a type to the variable 'stypy_return_type' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'stypy_return_type', list_62392)
        
        # ################# End of 'get_flags(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags' in the type store
        # Getting the type of 'stypy_return_type' (line 119)
        stypy_return_type_62394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62394)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags'
        return stypy_return_type_62394


    @norecursion
    def get_flags_opt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_opt'
        module_type_store = module_type_store.open_function_context('get_flags_opt', 122, 4, False)
        # Assigning a type to the variable 'self' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntelEM64TFCompiler.get_flags_opt.__dict__.__setitem__('stypy_localization', localization)
        IntelEM64TFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntelEM64TFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntelEM64TFCompiler.get_flags_opt.__dict__.__setitem__('stypy_function_name', 'IntelEM64TFCompiler.get_flags_opt')
        IntelEM64TFCompiler.get_flags_opt.__dict__.__setitem__('stypy_param_names_list', [])
        IntelEM64TFCompiler.get_flags_opt.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntelEM64TFCompiler.get_flags_opt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntelEM64TFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntelEM64TFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntelEM64TFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntelEM64TFCompiler.get_flags_opt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelEM64TFCompiler.get_flags_opt', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'list' (line 123)
        list_62395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 123)
        # Adding element type (line 123)
        str_62396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 16), 'str', '-openmp -fp-model strict -O1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 15), list_62395, str_62396)
        
        # Assigning a type to the variable 'stypy_return_type' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'stypy_return_type', list_62395)
        
        # ################# End of 'get_flags_opt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_opt' in the type store
        # Getting the type of 'stypy_return_type' (line 122)
        stypy_return_type_62397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62397)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_opt'
        return stypy_return_type_62397


    @norecursion
    def get_flags_arch(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_arch'
        module_type_store = module_type_store.open_function_context('get_flags_arch', 125, 4, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntelEM64TFCompiler.get_flags_arch.__dict__.__setitem__('stypy_localization', localization)
        IntelEM64TFCompiler.get_flags_arch.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntelEM64TFCompiler.get_flags_arch.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntelEM64TFCompiler.get_flags_arch.__dict__.__setitem__('stypy_function_name', 'IntelEM64TFCompiler.get_flags_arch')
        IntelEM64TFCompiler.get_flags_arch.__dict__.__setitem__('stypy_param_names_list', [])
        IntelEM64TFCompiler.get_flags_arch.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntelEM64TFCompiler.get_flags_arch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntelEM64TFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntelEM64TFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntelEM64TFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntelEM64TFCompiler.get_flags_arch.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelEM64TFCompiler.get_flags_arch', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_arch', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_arch(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 126)
        list_62398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 126)
        # Adding element type (line 126)
        str_62399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 16), 'str', '-xSSE4.2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 15), list_62398, str_62399)
        
        # Assigning a type to the variable 'stypy_return_type' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'stypy_return_type', list_62398)
        
        # ################# End of 'get_flags_arch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_arch' in the type store
        # Getting the type of 'stypy_return_type' (line 125)
        stypy_return_type_62400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62400)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_arch'
        return stypy_return_type_62400


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 100, 0, False)
        # Assigning a type to the variable 'self' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelEM64TFCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'IntelEM64TFCompiler' (line 100)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'IntelEM64TFCompiler', IntelEM64TFCompiler)

# Assigning a Str to a Name (line 101):
str_62401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 20), 'str', 'intelem')
# Getting the type of 'IntelEM64TFCompiler'
IntelEM64TFCompiler_62402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelEM64TFCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelEM64TFCompiler_62402, 'compiler_type', str_62401)

# Assigning a Tuple to a Name (line 102):

# Obtaining an instance of the builtin type 'tuple' (line 102)
tuple_62403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 102)

# Getting the type of 'IntelEM64TFCompiler'
IntelEM64TFCompiler_62404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelEM64TFCompiler')
# Setting the type of the member 'compiler_aliases' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelEM64TFCompiler_62404, 'compiler_aliases', tuple_62403)

# Assigning a Str to a Name (line 103):
str_62405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 18), 'str', 'Intel Fortran Compiler for 64-bit apps')
# Getting the type of 'IntelEM64TFCompiler'
IntelEM64TFCompiler_62406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelEM64TFCompiler')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelEM64TFCompiler_62406, 'description', str_62405)

# Assigning a Call to a Name (line 105):

# Call to intel_version_match(...): (line 105)
# Processing the call arguments (line 105)
str_62408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 40), 'str', 'EM64T-based|Intel\\(R\\) 64|64|IA-64|64-bit')
# Processing the call keyword arguments (line 105)
kwargs_62409 = {}
# Getting the type of 'intel_version_match' (line 105)
intel_version_match_62407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 20), 'intel_version_match', False)
# Calling intel_version_match(args, kwargs) (line 105)
intel_version_match_call_result_62410 = invoke(stypy.reporting.localization.Localization(__file__, 105, 20), intel_version_match_62407, *[str_62408], **kwargs_62409)

# Getting the type of 'IntelEM64TFCompiler'
IntelEM64TFCompiler_62411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelEM64TFCompiler')
# Setting the type of the member 'version_match' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelEM64TFCompiler_62411, 'version_match', intel_version_match_call_result_62410)

# Assigning a List to a Name (line 107):

# Obtaining an instance of the builtin type 'list' (line 107)
list_62412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 107)
# Adding element type (line 107)
str_62413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 28), 'str', 'ifort')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 27), list_62412, str_62413)
# Adding element type (line 107)
str_62414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 37), 'str', 'efort')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 27), list_62412, str_62414)
# Adding element type (line 107)
str_62415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 46), 'str', 'efc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 27), list_62412, str_62415)

# Getting the type of 'IntelEM64TFCompiler'
IntelEM64TFCompiler_62416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelEM64TFCompiler')
# Setting the type of the member 'possible_executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelEM64TFCompiler_62416, 'possible_executables', list_62412)

# Assigning a Dict to a Name (line 109):

# Obtaining an instance of the builtin type 'dict' (line 109)
dict_62417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 109)
# Adding element type (key, value) (line 109)
str_62418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 8), 'str', 'version_cmd')
# Getting the type of 'None' (line 110)
None_62419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 25), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 18), dict_62417, (str_62418, None_62419))
# Adding element type (key, value) (line 109)
str_62420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 8), 'str', 'compiler_f77')

# Obtaining an instance of the builtin type 'list' (line 111)
list_62421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 111)
# Adding element type (line 111)
# Getting the type of 'None' (line 111)
None_62422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 25), list_62421, None_62422)
# Adding element type (line 111)
str_62423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 32), 'str', '-FI')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 25), list_62421, str_62423)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 18), dict_62417, (str_62420, list_62421))
# Adding element type (key, value) (line 109)
str_62424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 8), 'str', 'compiler_fix')

# Obtaining an instance of the builtin type 'list' (line 112)
list_62425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 112)
# Adding element type (line 112)
# Getting the type of 'None' (line 112)
None_62426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 25), list_62425, None_62426)
# Adding element type (line 112)
str_62427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 32), 'str', '-FI')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 25), list_62425, str_62427)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 18), dict_62417, (str_62424, list_62425))
# Adding element type (key, value) (line 109)
str_62428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 8), 'str', 'compiler_f90')

# Obtaining an instance of the builtin type 'list' (line 113)
list_62429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 113)
# Adding element type (line 113)
# Getting the type of 'None' (line 113)
None_62430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 25), list_62429, None_62430)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 18), dict_62417, (str_62428, list_62429))
# Adding element type (key, value) (line 109)
str_62431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 8), 'str', 'linker_so')

# Obtaining an instance of the builtin type 'list' (line 114)
list_62432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 114)
# Adding element type (line 114)
str_62433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 26), 'str', '<F90>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 25), list_62432, str_62433)
# Adding element type (line 114)
str_62434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 35), 'str', '-shared')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 25), list_62432, str_62434)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 18), dict_62417, (str_62431, list_62432))
# Adding element type (key, value) (line 109)
str_62435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 8), 'str', 'archiver')

# Obtaining an instance of the builtin type 'list' (line 115)
list_62436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 115)
# Adding element type (line 115)
str_62437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 26), 'str', 'ar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 25), list_62436, str_62437)
# Adding element type (line 115)
str_62438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 32), 'str', '-cr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 25), list_62436, str_62438)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 18), dict_62417, (str_62435, list_62436))
# Adding element type (key, value) (line 109)
str_62439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 8), 'str', 'ranlib')

# Obtaining an instance of the builtin type 'list' (line 116)
list_62440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 116)
# Adding element type (line 116)
str_62441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 26), 'str', 'ranlib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 25), list_62440, str_62441)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 18), dict_62417, (str_62439, list_62440))

# Getting the type of 'IntelEM64TFCompiler'
IntelEM64TFCompiler_62442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelEM64TFCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelEM64TFCompiler_62442, 'executables', dict_62417)
# Declaration of the 'IntelVisualFCompiler' class
# Getting the type of 'BaseIntelFCompiler' (line 132)
BaseIntelFCompiler_62443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 27), 'BaseIntelFCompiler')

class IntelVisualFCompiler(BaseIntelFCompiler_62443, ):

    @norecursion
    def update_executables(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update_executables'
        module_type_store = module_type_store.open_function_context('update_executables', 137, 4, False)
        # Assigning a type to the variable 'self' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntelVisualFCompiler.update_executables.__dict__.__setitem__('stypy_localization', localization)
        IntelVisualFCompiler.update_executables.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntelVisualFCompiler.update_executables.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntelVisualFCompiler.update_executables.__dict__.__setitem__('stypy_function_name', 'IntelVisualFCompiler.update_executables')
        IntelVisualFCompiler.update_executables.__dict__.__setitem__('stypy_param_names_list', [])
        IntelVisualFCompiler.update_executables.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntelVisualFCompiler.update_executables.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntelVisualFCompiler.update_executables.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntelVisualFCompiler.update_executables.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntelVisualFCompiler.update_executables.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntelVisualFCompiler.update_executables.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelVisualFCompiler.update_executables', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 138):
        
        # Call to dummy_fortran_file(...): (line 138)
        # Processing the call keyword arguments (line 138)
        kwargs_62445 = {}
        # Getting the type of 'dummy_fortran_file' (line 138)
        dummy_fortran_file_62444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'dummy_fortran_file', False)
        # Calling dummy_fortran_file(args, kwargs) (line 138)
        dummy_fortran_file_call_result_62446 = invoke(stypy.reporting.localization.Localization(__file__, 138, 12), dummy_fortran_file_62444, *[], **kwargs_62445)
        
        # Assigning a type to the variable 'f' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'f', dummy_fortran_file_call_result_62446)
        
        # Assigning a List to a Subscript (line 139):
        
        # Obtaining an instance of the builtin type 'list' (line 139)
        list_62447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 139)
        # Adding element type (line 139)
        str_62448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 43), 'str', '<F77>')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 42), list_62447, str_62448)
        # Adding element type (line 139)
        str_62449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 52), 'str', '/FI')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 42), list_62447, str_62449)
        # Adding element type (line 139)
        str_62450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 59), 'str', '/c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 42), list_62447, str_62450)
        # Adding element type (line 139)
        # Getting the type of 'f' (line 140)
        f_62451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 43), 'f')
        str_62452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 47), 'str', '.f')
        # Applying the binary operator '+' (line 140)
        result_add_62453 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 43), '+', f_62451, str_62452)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 42), list_62447, result_add_62453)
        # Adding element type (line 139)
        str_62454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 53), 'str', '/o')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 42), list_62447, str_62454)
        # Adding element type (line 139)
        # Getting the type of 'f' (line 140)
        f_62455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 59), 'f')
        str_62456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 63), 'str', '.o')
        # Applying the binary operator '+' (line 140)
        result_add_62457 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 59), '+', f_62455, str_62456)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 42), list_62447, result_add_62457)
        
        # Getting the type of 'self' (line 139)
        self_62458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'self')
        # Obtaining the member 'executables' of a type (line 139)
        executables_62459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), self_62458, 'executables')
        str_62460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 25), 'str', 'version_cmd')
        # Storing an element on a container (line 139)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 8), executables_62459, (str_62460, list_62447))
        
        # ################# End of 'update_executables(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update_executables' in the type store
        # Getting the type of 'stypy_return_type' (line 137)
        stypy_return_type_62461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62461)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update_executables'
        return stypy_return_type_62461


    @norecursion
    def get_flags(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags'
        module_type_store = module_type_store.open_function_context('get_flags', 161, 4, False)
        # Assigning a type to the variable 'self' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntelVisualFCompiler.get_flags.__dict__.__setitem__('stypy_localization', localization)
        IntelVisualFCompiler.get_flags.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntelVisualFCompiler.get_flags.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntelVisualFCompiler.get_flags.__dict__.__setitem__('stypy_function_name', 'IntelVisualFCompiler.get_flags')
        IntelVisualFCompiler.get_flags.__dict__.__setitem__('stypy_param_names_list', [])
        IntelVisualFCompiler.get_flags.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntelVisualFCompiler.get_flags.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntelVisualFCompiler.get_flags.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntelVisualFCompiler.get_flags.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntelVisualFCompiler.get_flags.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntelVisualFCompiler.get_flags.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelVisualFCompiler.get_flags', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Name (line 162):
        
        # Obtaining an instance of the builtin type 'list' (line 162)
        list_62462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 162)
        # Adding element type (line 162)
        str_62463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 15), 'str', '/nologo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 14), list_62462, str_62463)
        # Adding element type (line 162)
        str_62464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 26), 'str', '/MD')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 14), list_62462, str_62464)
        # Adding element type (line 162)
        str_62465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 33), 'str', '/nbs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 14), list_62462, str_62465)
        # Adding element type (line 162)
        str_62466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 41), 'str', '/names:lowercase')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 14), list_62462, str_62466)
        # Adding element type (line 162)
        str_62467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 61), 'str', '/assume:underscore')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 14), list_62462, str_62467)
        
        # Assigning a type to the variable 'opt' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'opt', list_62462)
        # Getting the type of 'opt' (line 163)
        opt_62468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 15), 'opt')
        # Assigning a type to the variable 'stypy_return_type' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'stypy_return_type', opt_62468)
        
        # ################# End of 'get_flags(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags' in the type store
        # Getting the type of 'stypy_return_type' (line 161)
        stypy_return_type_62469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62469)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags'
        return stypy_return_type_62469


    @norecursion
    def get_flags_free(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_free'
        module_type_store = module_type_store.open_function_context('get_flags_free', 165, 4, False)
        # Assigning a type to the variable 'self' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntelVisualFCompiler.get_flags_free.__dict__.__setitem__('stypy_localization', localization)
        IntelVisualFCompiler.get_flags_free.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntelVisualFCompiler.get_flags_free.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntelVisualFCompiler.get_flags_free.__dict__.__setitem__('stypy_function_name', 'IntelVisualFCompiler.get_flags_free')
        IntelVisualFCompiler.get_flags_free.__dict__.__setitem__('stypy_param_names_list', [])
        IntelVisualFCompiler.get_flags_free.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntelVisualFCompiler.get_flags_free.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntelVisualFCompiler.get_flags_free.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntelVisualFCompiler.get_flags_free.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntelVisualFCompiler.get_flags_free.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntelVisualFCompiler.get_flags_free.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelVisualFCompiler.get_flags_free', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_free', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_free(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 166)
        list_62470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 166)
        
        # Assigning a type to the variable 'stypy_return_type' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'stypy_return_type', list_62470)
        
        # ################# End of 'get_flags_free(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_free' in the type store
        # Getting the type of 'stypy_return_type' (line 165)
        stypy_return_type_62471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62471)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_free'
        return stypy_return_type_62471


    @norecursion
    def get_flags_debug(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_debug'
        module_type_store = module_type_store.open_function_context('get_flags_debug', 168, 4, False)
        # Assigning a type to the variable 'self' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntelVisualFCompiler.get_flags_debug.__dict__.__setitem__('stypy_localization', localization)
        IntelVisualFCompiler.get_flags_debug.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntelVisualFCompiler.get_flags_debug.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntelVisualFCompiler.get_flags_debug.__dict__.__setitem__('stypy_function_name', 'IntelVisualFCompiler.get_flags_debug')
        IntelVisualFCompiler.get_flags_debug.__dict__.__setitem__('stypy_param_names_list', [])
        IntelVisualFCompiler.get_flags_debug.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntelVisualFCompiler.get_flags_debug.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntelVisualFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntelVisualFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntelVisualFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntelVisualFCompiler.get_flags_debug.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelVisualFCompiler.get_flags_debug', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_debug', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_debug(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 169)
        list_62472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 169)
        # Adding element type (line 169)
        str_62473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 16), 'str', '/4Yb')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 15), list_62472, str_62473)
        # Adding element type (line 169)
        str_62474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 24), 'str', '/d2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 15), list_62472, str_62474)
        
        # Assigning a type to the variable 'stypy_return_type' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'stypy_return_type', list_62472)
        
        # ################# End of 'get_flags_debug(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_debug' in the type store
        # Getting the type of 'stypy_return_type' (line 168)
        stypy_return_type_62475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62475)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_debug'
        return stypy_return_type_62475


    @norecursion
    def get_flags_opt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_opt'
        module_type_store = module_type_store.open_function_context('get_flags_opt', 171, 4, False)
        # Assigning a type to the variable 'self' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntelVisualFCompiler.get_flags_opt.__dict__.__setitem__('stypy_localization', localization)
        IntelVisualFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntelVisualFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntelVisualFCompiler.get_flags_opt.__dict__.__setitem__('stypy_function_name', 'IntelVisualFCompiler.get_flags_opt')
        IntelVisualFCompiler.get_flags_opt.__dict__.__setitem__('stypy_param_names_list', [])
        IntelVisualFCompiler.get_flags_opt.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntelVisualFCompiler.get_flags_opt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntelVisualFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntelVisualFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntelVisualFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntelVisualFCompiler.get_flags_opt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelVisualFCompiler.get_flags_opt', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'list' (line 172)
        list_62476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 172)
        # Adding element type (line 172)
        str_62477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 16), 'str', '/O1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 15), list_62476, str_62477)
        
        # Assigning a type to the variable 'stypy_return_type' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'stypy_return_type', list_62476)
        
        # ################# End of 'get_flags_opt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_opt' in the type store
        # Getting the type of 'stypy_return_type' (line 171)
        stypy_return_type_62478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62478)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_opt'
        return stypy_return_type_62478


    @norecursion
    def get_flags_arch(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_arch'
        module_type_store = module_type_store.open_function_context('get_flags_arch', 174, 4, False)
        # Assigning a type to the variable 'self' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntelVisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_localization', localization)
        IntelVisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntelVisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntelVisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_function_name', 'IntelVisualFCompiler.get_flags_arch')
        IntelVisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_param_names_list', [])
        IntelVisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntelVisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntelVisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntelVisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntelVisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntelVisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelVisualFCompiler.get_flags_arch', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_arch', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_arch(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 175)
        list_62479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 175)
        # Adding element type (line 175)
        str_62480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 16), 'str', '/arch:IA32')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 15), list_62479, str_62480)
        # Adding element type (line 175)
        str_62481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 30), 'str', '/QaxSSE3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 15), list_62479, str_62481)
        
        # Assigning a type to the variable 'stypy_return_type' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'stypy_return_type', list_62479)
        
        # ################# End of 'get_flags_arch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_arch' in the type store
        # Getting the type of 'stypy_return_type' (line 174)
        stypy_return_type_62482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62482)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_arch'
        return stypy_return_type_62482


    @norecursion
    def runtime_library_dir_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'runtime_library_dir_option'
        module_type_store = module_type_store.open_function_context('runtime_library_dir_option', 177, 4, False)
        # Assigning a type to the variable 'self' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntelVisualFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_localization', localization)
        IntelVisualFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntelVisualFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntelVisualFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_function_name', 'IntelVisualFCompiler.runtime_library_dir_option')
        IntelVisualFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_param_names_list', ['dir'])
        IntelVisualFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntelVisualFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntelVisualFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntelVisualFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntelVisualFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntelVisualFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelVisualFCompiler.runtime_library_dir_option', ['dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'runtime_library_dir_option', localization, ['dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'runtime_library_dir_option(...)' code ##################

        # Getting the type of 'NotImplementedError' (line 178)
        NotImplementedError_62483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 14), 'NotImplementedError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 178, 8), NotImplementedError_62483, 'raise parameter', BaseException)
        
        # ################# End of 'runtime_library_dir_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'runtime_library_dir_option' in the type store
        # Getting the type of 'stypy_return_type' (line 177)
        stypy_return_type_62484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62484)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'runtime_library_dir_option'
        return stypy_return_type_62484


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 132, 0, False)
        # Assigning a type to the variable 'self' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelVisualFCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'IntelVisualFCompiler' (line 132)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 0), 'IntelVisualFCompiler', IntelVisualFCompiler)

# Assigning a Str to a Name (line 133):
str_62485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 20), 'str', 'intelv')
# Getting the type of 'IntelVisualFCompiler'
IntelVisualFCompiler_62486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelVisualFCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelVisualFCompiler_62486, 'compiler_type', str_62485)

# Assigning a Str to a Name (line 134):
str_62487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 18), 'str', 'Intel Visual Fortran Compiler for 32-bit apps')
# Getting the type of 'IntelVisualFCompiler'
IntelVisualFCompiler_62488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelVisualFCompiler')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelVisualFCompiler_62488, 'description', str_62487)

# Assigning a Call to a Name (line 135):

# Call to intel_version_match(...): (line 135)
# Processing the call arguments (line 135)
str_62490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 40), 'str', '32-bit|IA-32')
# Processing the call keyword arguments (line 135)
kwargs_62491 = {}
# Getting the type of 'intel_version_match' (line 135)
intel_version_match_62489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 20), 'intel_version_match', False)
# Calling intel_version_match(args, kwargs) (line 135)
intel_version_match_call_result_62492 = invoke(stypy.reporting.localization.Localization(__file__, 135, 20), intel_version_match_62489, *[str_62490], **kwargs_62491)

# Getting the type of 'IntelVisualFCompiler'
IntelVisualFCompiler_62493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelVisualFCompiler')
# Setting the type of the member 'version_match' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelVisualFCompiler_62493, 'version_match', intel_version_match_call_result_62492)

# Assigning a Str to a Name (line 142):
str_62494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 13), 'str', 'lib.exe')
# Getting the type of 'IntelVisualFCompiler'
IntelVisualFCompiler_62495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelVisualFCompiler')
# Setting the type of the member 'ar_exe' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelVisualFCompiler_62495, 'ar_exe', str_62494)

# Assigning a List to a Name (line 143):

# Obtaining an instance of the builtin type 'list' (line 143)
list_62496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 143)
# Adding element type (line 143)
str_62497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 28), 'str', 'ifort')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 27), list_62496, str_62497)
# Adding element type (line 143)
str_62498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 37), 'str', 'ifl')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 27), list_62496, str_62498)

# Getting the type of 'IntelVisualFCompiler'
IntelVisualFCompiler_62499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelVisualFCompiler')
# Setting the type of the member 'possible_executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelVisualFCompiler_62499, 'possible_executables', list_62496)

# Assigning a Dict to a Name (line 145):

# Obtaining an instance of the builtin type 'dict' (line 145)
dict_62500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 145)
# Adding element type (key, value) (line 145)
str_62501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 8), 'str', 'version_cmd')
# Getting the type of 'None' (line 146)
None_62502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 25), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 18), dict_62500, (str_62501, None_62502))
# Adding element type (key, value) (line 145)
str_62503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 8), 'str', 'compiler_f77')

# Obtaining an instance of the builtin type 'list' (line 147)
list_62504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 147)
# Adding element type (line 147)
# Getting the type of 'None' (line 147)
None_62505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 25), list_62504, None_62505)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 18), dict_62500, (str_62503, list_62504))
# Adding element type (key, value) (line 145)
str_62506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 8), 'str', 'compiler_fix')

# Obtaining an instance of the builtin type 'list' (line 148)
list_62507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 148)
# Adding element type (line 148)
# Getting the type of 'None' (line 148)
None_62508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 25), list_62507, None_62508)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 18), dict_62500, (str_62506, list_62507))
# Adding element type (key, value) (line 145)
str_62509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 8), 'str', 'compiler_f90')

# Obtaining an instance of the builtin type 'list' (line 149)
list_62510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 149)
# Adding element type (line 149)
# Getting the type of 'None' (line 149)
None_62511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 25), list_62510, None_62511)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 18), dict_62500, (str_62509, list_62510))
# Adding element type (key, value) (line 145)
str_62512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 8), 'str', 'linker_so')

# Obtaining an instance of the builtin type 'list' (line 150)
list_62513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 150)
# Adding element type (line 150)
# Getting the type of 'None' (line 150)
None_62514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 25), list_62513, None_62514)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 18), dict_62500, (str_62512, list_62513))
# Adding element type (key, value) (line 145)
str_62515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 8), 'str', 'archiver')

# Obtaining an instance of the builtin type 'list' (line 151)
list_62516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 151)
# Adding element type (line 151)
# Getting the type of 'IntelVisualFCompiler'
IntelVisualFCompiler_62517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelVisualFCompiler')
# Obtaining the member 'ar_exe' of a type
ar_exe_62518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelVisualFCompiler_62517, 'ar_exe')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 25), list_62516, ar_exe_62518)
# Adding element type (line 151)
str_62519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 34), 'str', '/verbose')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 25), list_62516, str_62519)
# Adding element type (line 151)
str_62520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 46), 'str', '/OUT:')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 25), list_62516, str_62520)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 18), dict_62500, (str_62515, list_62516))
# Adding element type (key, value) (line 145)
str_62521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 8), 'str', 'ranlib')
# Getting the type of 'None' (line 152)
None_62522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 25), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 18), dict_62500, (str_62521, None_62522))

# Getting the type of 'IntelVisualFCompiler'
IntelVisualFCompiler_62523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelVisualFCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelVisualFCompiler_62523, 'executables', dict_62500)

# Assigning a Str to a Name (line 155):
str_62524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 21), 'str', '/c ')
# Getting the type of 'IntelVisualFCompiler'
IntelVisualFCompiler_62525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelVisualFCompiler')
# Setting the type of the member 'compile_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelVisualFCompiler_62525, 'compile_switch', str_62524)

# Assigning a Str to a Name (line 156):
str_62526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 20), 'str', '/Fo')
# Getting the type of 'IntelVisualFCompiler'
IntelVisualFCompiler_62527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelVisualFCompiler')
# Setting the type of the member 'object_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelVisualFCompiler_62527, 'object_switch', str_62526)

# Assigning a Str to a Name (line 157):
str_62528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 21), 'str', '/OUT:')
# Getting the type of 'IntelVisualFCompiler'
IntelVisualFCompiler_62529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelVisualFCompiler')
# Setting the type of the member 'library_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelVisualFCompiler_62529, 'library_switch', str_62528)

# Assigning a Str to a Name (line 158):
str_62530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 24), 'str', '/module:')
# Getting the type of 'IntelVisualFCompiler'
IntelVisualFCompiler_62531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelVisualFCompiler')
# Setting the type of the member 'module_dir_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelVisualFCompiler_62531, 'module_dir_switch', str_62530)

# Assigning a Str to a Name (line 159):
str_62532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 28), 'str', '/I')
# Getting the type of 'IntelVisualFCompiler'
IntelVisualFCompiler_62533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelVisualFCompiler')
# Setting the type of the member 'module_include_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelVisualFCompiler_62533, 'module_include_switch', str_62532)
# Declaration of the 'IntelItaniumVisualFCompiler' class
# Getting the type of 'IntelVisualFCompiler' (line 181)
IntelVisualFCompiler_62534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 34), 'IntelVisualFCompiler')

class IntelItaniumVisualFCompiler(IntelVisualFCompiler_62534, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 181, 0, False)
        # Assigning a type to the variable 'self' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelItaniumVisualFCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'IntelItaniumVisualFCompiler' (line 181)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 0), 'IntelItaniumVisualFCompiler', IntelItaniumVisualFCompiler)

# Assigning a Str to a Name (line 182):
str_62535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 20), 'str', 'intelev')
# Getting the type of 'IntelItaniumVisualFCompiler'
IntelItaniumVisualFCompiler_62536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelItaniumVisualFCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelItaniumVisualFCompiler_62536, 'compiler_type', str_62535)

# Assigning a Str to a Name (line 183):
str_62537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 18), 'str', 'Intel Visual Fortran Compiler for Itanium apps')
# Getting the type of 'IntelItaniumVisualFCompiler'
IntelItaniumVisualFCompiler_62538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelItaniumVisualFCompiler')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelItaniumVisualFCompiler_62538, 'description', str_62537)

# Assigning a Call to a Name (line 185):

# Call to intel_version_match(...): (line 185)
# Processing the call arguments (line 185)
str_62540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 40), 'str', 'Itanium')
# Processing the call keyword arguments (line 185)
kwargs_62541 = {}
# Getting the type of 'intel_version_match' (line 185)
intel_version_match_62539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 20), 'intel_version_match', False)
# Calling intel_version_match(args, kwargs) (line 185)
intel_version_match_call_result_62542 = invoke(stypy.reporting.localization.Localization(__file__, 185, 20), intel_version_match_62539, *[str_62540], **kwargs_62541)

# Getting the type of 'IntelItaniumVisualFCompiler'
IntelItaniumVisualFCompiler_62543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelItaniumVisualFCompiler')
# Setting the type of the member 'version_match' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelItaniumVisualFCompiler_62543, 'version_match', intel_version_match_call_result_62542)

# Assigning a List to a Name (line 187):

# Obtaining an instance of the builtin type 'list' (line 187)
list_62544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 187)
# Adding element type (line 187)
str_62545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 28), 'str', 'efl')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 27), list_62544, str_62545)

# Getting the type of 'IntelItaniumVisualFCompiler'
IntelItaniumVisualFCompiler_62546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelItaniumVisualFCompiler')
# Setting the type of the member 'possible_executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelItaniumVisualFCompiler_62546, 'possible_executables', list_62544)

# Assigning a Attribute to a Name (line 188):
# Getting the type of 'IntelVisualFCompiler' (line 188)
IntelVisualFCompiler_62547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 13), 'IntelVisualFCompiler')
# Obtaining the member 'ar_exe' of a type (line 188)
ar_exe_62548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 13), IntelVisualFCompiler_62547, 'ar_exe')
# Getting the type of 'IntelItaniumVisualFCompiler'
IntelItaniumVisualFCompiler_62549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelItaniumVisualFCompiler')
# Setting the type of the member 'ar_exe' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelItaniumVisualFCompiler_62549, 'ar_exe', ar_exe_62548)

# Assigning a Dict to a Name (line 190):

# Obtaining an instance of the builtin type 'dict' (line 190)
dict_62550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 190)
# Adding element type (key, value) (line 190)
str_62551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 8), 'str', 'version_cmd')
# Getting the type of 'None' (line 191)
None_62552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 25), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 18), dict_62550, (str_62551, None_62552))
# Adding element type (key, value) (line 190)
str_62553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 8), 'str', 'compiler_f77')

# Obtaining an instance of the builtin type 'list' (line 192)
list_62554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 192)
# Adding element type (line 192)
# Getting the type of 'None' (line 192)
None_62555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 25), list_62554, None_62555)
# Adding element type (line 192)
str_62556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 32), 'str', '-FI')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 25), list_62554, str_62556)
# Adding element type (line 192)
str_62557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 39), 'str', '-w90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 25), list_62554, str_62557)
# Adding element type (line 192)
str_62558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 47), 'str', '-w95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 25), list_62554, str_62558)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 18), dict_62550, (str_62553, list_62554))
# Adding element type (key, value) (line 190)
str_62559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 8), 'str', 'compiler_fix')

# Obtaining an instance of the builtin type 'list' (line 193)
list_62560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 193)
# Adding element type (line 193)
# Getting the type of 'None' (line 193)
None_62561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 25), list_62560, None_62561)
# Adding element type (line 193)
str_62562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 32), 'str', '-FI')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 25), list_62560, str_62562)
# Adding element type (line 193)
str_62563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 39), 'str', '-4L72')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 25), list_62560, str_62563)
# Adding element type (line 193)
str_62564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 48), 'str', '-w')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 25), list_62560, str_62564)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 18), dict_62550, (str_62559, list_62560))
# Adding element type (key, value) (line 190)
str_62565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 8), 'str', 'compiler_f90')

# Obtaining an instance of the builtin type 'list' (line 194)
list_62566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 194)
# Adding element type (line 194)
# Getting the type of 'None' (line 194)
None_62567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 25), list_62566, None_62567)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 18), dict_62550, (str_62565, list_62566))
# Adding element type (key, value) (line 190)
str_62568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 8), 'str', 'linker_so')

# Obtaining an instance of the builtin type 'list' (line 195)
list_62569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 195)
# Adding element type (line 195)
str_62570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 26), 'str', '<F90>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 25), list_62569, str_62570)
# Adding element type (line 195)
str_62571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 35), 'str', '-shared')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 25), list_62569, str_62571)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 18), dict_62550, (str_62568, list_62569))
# Adding element type (key, value) (line 190)
str_62572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 8), 'str', 'archiver')

# Obtaining an instance of the builtin type 'list' (line 196)
list_62573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 196)
# Adding element type (line 196)
# Getting the type of 'IntelItaniumVisualFCompiler'
IntelItaniumVisualFCompiler_62574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelItaniumVisualFCompiler')
# Obtaining the member 'ar_exe' of a type
ar_exe_62575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelItaniumVisualFCompiler_62574, 'ar_exe')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 25), list_62573, ar_exe_62575)
# Adding element type (line 196)
str_62576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 34), 'str', '/verbose')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 25), list_62573, str_62576)
# Adding element type (line 196)
str_62577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 46), 'str', '/OUT:')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 25), list_62573, str_62577)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 18), dict_62550, (str_62572, list_62573))
# Adding element type (key, value) (line 190)
str_62578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 8), 'str', 'ranlib')
# Getting the type of 'None' (line 197)
None_62579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 25), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 18), dict_62550, (str_62578, None_62579))

# Getting the type of 'IntelItaniumVisualFCompiler'
IntelItaniumVisualFCompiler_62580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelItaniumVisualFCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelItaniumVisualFCompiler_62580, 'executables', dict_62550)
# Declaration of the 'IntelEM64VisualFCompiler' class
# Getting the type of 'IntelVisualFCompiler' (line 201)
IntelVisualFCompiler_62581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 31), 'IntelVisualFCompiler')

class IntelEM64VisualFCompiler(IntelVisualFCompiler_62581, ):

    @norecursion
    def get_flags_arch(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_arch'
        module_type_store = module_type_store.open_function_context('get_flags_arch', 207, 4, False)
        # Assigning a type to the variable 'self' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntelEM64VisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_localization', localization)
        IntelEM64VisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntelEM64VisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntelEM64VisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_function_name', 'IntelEM64VisualFCompiler.get_flags_arch')
        IntelEM64VisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_param_names_list', [])
        IntelEM64VisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntelEM64VisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntelEM64VisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntelEM64VisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntelEM64VisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntelEM64VisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelEM64VisualFCompiler.get_flags_arch', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_arch', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_arch(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 208)
        list_62582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 208)
        # Adding element type (line 208)
        str_62583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 16), 'str', '/QaxSSE4.2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 15), list_62582, str_62583)
        
        # Assigning a type to the variable 'stypy_return_type' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'stypy_return_type', list_62582)
        
        # ################# End of 'get_flags_arch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_arch' in the type store
        # Getting the type of 'stypy_return_type' (line 207)
        stypy_return_type_62584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62584)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_arch'
        return stypy_return_type_62584


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 201, 0, False)
        # Assigning a type to the variable 'self' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntelEM64VisualFCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'IntelEM64VisualFCompiler' (line 201)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 0), 'IntelEM64VisualFCompiler', IntelEM64VisualFCompiler)

# Assigning a Str to a Name (line 202):
str_62585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 20), 'str', 'intelvem')
# Getting the type of 'IntelEM64VisualFCompiler'
IntelEM64VisualFCompiler_62586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelEM64VisualFCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelEM64VisualFCompiler_62586, 'compiler_type', str_62585)

# Assigning a Str to a Name (line 203):
str_62587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 18), 'str', 'Intel Visual Fortran Compiler for 64-bit apps')
# Getting the type of 'IntelEM64VisualFCompiler'
IntelEM64VisualFCompiler_62588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelEM64VisualFCompiler')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelEM64VisualFCompiler_62588, 'description', str_62587)

# Assigning a Call to a Name (line 205):

# Call to simple_version_match(...): (line 205)
# Processing the call keyword arguments (line 205)
str_62590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 47), 'str', 'Intel\\(R\\).*?64,')
keyword_62591 = str_62590
kwargs_62592 = {'start': keyword_62591}
# Getting the type of 'simple_version_match' (line 205)
simple_version_match_62589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'simple_version_match', False)
# Calling simple_version_match(args, kwargs) (line 205)
simple_version_match_call_result_62593 = invoke(stypy.reporting.localization.Localization(__file__, 205, 20), simple_version_match_62589, *[], **kwargs_62592)

# Getting the type of 'IntelEM64VisualFCompiler'
IntelEM64VisualFCompiler_62594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IntelEM64VisualFCompiler')
# Setting the type of the member 'version_match' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IntelEM64VisualFCompiler_62594, 'version_match', simple_version_match_call_result_62593)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 212, 4))
    
    # 'from distutils import log' statement (line 212)
    from distutils import log

    import_from_module(stypy.reporting.localization.Localization(__file__, 212, 4), 'distutils', None, module_type_store, ['log'], [log])
    
    
    # Call to set_verbosity(...): (line 213)
    # Processing the call arguments (line 213)
    int_62597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 22), 'int')
    # Processing the call keyword arguments (line 213)
    kwargs_62598 = {}
    # Getting the type of 'log' (line 213)
    log_62595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'log', False)
    # Obtaining the member 'set_verbosity' of a type (line 213)
    set_verbosity_62596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 4), log_62595, 'set_verbosity')
    # Calling set_verbosity(args, kwargs) (line 213)
    set_verbosity_call_result_62599 = invoke(stypy.reporting.localization.Localization(__file__, 213, 4), set_verbosity_62596, *[int_62597], **kwargs_62598)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 214, 4))
    
    # 'from numpy.distutils.fcompiler import new_fcompiler' statement (line 214)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
    import_62600 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 214, 4), 'numpy.distutils.fcompiler')

    if (type(import_62600) is not StypyTypeError):

        if (import_62600 != 'pyd_module'):
            __import__(import_62600)
            sys_modules_62601 = sys.modules[import_62600]
            import_from_module(stypy.reporting.localization.Localization(__file__, 214, 4), 'numpy.distutils.fcompiler', sys_modules_62601.module_type_store, module_type_store, ['new_fcompiler'])
            nest_module(stypy.reporting.localization.Localization(__file__, 214, 4), __file__, sys_modules_62601, sys_modules_62601.module_type_store, module_type_store)
        else:
            from numpy.distutils.fcompiler import new_fcompiler

            import_from_module(stypy.reporting.localization.Localization(__file__, 214, 4), 'numpy.distutils.fcompiler', None, module_type_store, ['new_fcompiler'], [new_fcompiler])

    else:
        # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'numpy.distutils.fcompiler', import_62600)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
    
    
    # Assigning a Call to a Name (line 215):
    
    # Call to new_fcompiler(...): (line 215)
    # Processing the call keyword arguments (line 215)
    str_62603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 38), 'str', 'intel')
    keyword_62604 = str_62603
    kwargs_62605 = {'compiler': keyword_62604}
    # Getting the type of 'new_fcompiler' (line 215)
    new_fcompiler_62602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 15), 'new_fcompiler', False)
    # Calling new_fcompiler(args, kwargs) (line 215)
    new_fcompiler_call_result_62606 = invoke(stypy.reporting.localization.Localization(__file__, 215, 15), new_fcompiler_62602, *[], **kwargs_62605)
    
    # Assigning a type to the variable 'compiler' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'compiler', new_fcompiler_call_result_62606)
    
    # Call to customize(...): (line 216)
    # Processing the call keyword arguments (line 216)
    kwargs_62609 = {}
    # Getting the type of 'compiler' (line 216)
    compiler_62607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'compiler', False)
    # Obtaining the member 'customize' of a type (line 216)
    customize_62608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 4), compiler_62607, 'customize')
    # Calling customize(args, kwargs) (line 216)
    customize_call_result_62610 = invoke(stypy.reporting.localization.Localization(__file__, 216, 4), customize_62608, *[], **kwargs_62609)
    
    
    # Call to print(...): (line 217)
    # Processing the call arguments (line 217)
    
    # Call to get_version(...): (line 217)
    # Processing the call keyword arguments (line 217)
    kwargs_62614 = {}
    # Getting the type of 'compiler' (line 217)
    compiler_62612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 10), 'compiler', False)
    # Obtaining the member 'get_version' of a type (line 217)
    get_version_62613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 10), compiler_62612, 'get_version')
    # Calling get_version(args, kwargs) (line 217)
    get_version_call_result_62615 = invoke(stypy.reporting.localization.Localization(__file__, 217, 10), get_version_62613, *[], **kwargs_62614)
    
    # Processing the call keyword arguments (line 217)
    kwargs_62616 = {}
    # Getting the type of 'print' (line 217)
    print_62611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'print', False)
    # Calling print(args, kwargs) (line 217)
    print_call_result_62617 = invoke(stypy.reporting.localization.Localization(__file__, 217, 4), print_62611, *[get_version_call_result_62615], **kwargs_62616)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
