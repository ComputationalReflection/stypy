
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import re
2: import sys
3: import os
4: import glob
5: from distutils.dep_util import newer
6: 
7: 
8: __all__ = ['needs_g77_abi_wrapper', 'split_fortran_files',
9:            'get_g77_abi_wrappers',
10:            'needs_sgemv_fix', 'get_sgemv_fix']
11: 
12: 
13: def uses_veclib(info):
14:     if sys.platform != "darwin":
15:         return False
16:     r_accelerate = re.compile("vecLib")
17:     extra_link_args = info.get('extra_link_args', '')
18:     for arg in extra_link_args:
19:         if r_accelerate.search(arg):
20:             return True
21:     return False
22: 
23: 
24: def uses_accelerate(info):
25:     if sys.platform != "darwin":
26:         return False
27:     r_accelerate = re.compile("Accelerate")
28:     extra_link_args = info.get('extra_link_args', '')
29:     for arg in extra_link_args:
30:         if r_accelerate.search(arg):
31:             return True
32:     return False
33: 
34: 
35: def uses_mkl(info):
36:     r_mkl = re.compile("mkl")
37:     libraries = info.get('libraries', '')
38:     for library in libraries:
39:         if r_mkl.search(library):
40:             return True
41: 
42:     return False
43: 
44: 
45: def needs_g77_abi_wrapper(info):
46:     '''Returns True if g77 ABI wrapper must be used.'''
47:     if uses_accelerate(info) or uses_veclib(info):
48:         return True
49:     elif uses_mkl(info):
50:         return True
51:     else:
52:         return False
53: 
54: 
55: def get_g77_abi_wrappers(info):
56:     '''
57:     Returns file names of source files containing Fortran ABI wrapper
58:     routines.
59:     '''
60:     wrapper_sources = []
61: 
62:     path = os.path.abspath(os.path.dirname(__file__))
63:     if needs_g77_abi_wrapper(info):
64:         wrapper_sources += [
65:             os.path.join(path, 'src', 'wrap_g77_abi_f.f'),
66:             os.path.join(path, 'src', 'wrap_g77_abi_c.c'),
67:         ]
68:         if uses_accelerate(info):
69:             wrapper_sources += [
70:                     os.path.join(path, 'src', 'wrap_accelerate_c.c'),
71:                     os.path.join(path, 'src', 'wrap_accelerate_f.f'),
72:             ]
73:         elif uses_mkl(info):
74:             wrapper_sources += [
75:                     os.path.join(path, 'src', 'wrap_dummy_accelerate.f'),
76:             ]
77:         else:
78:             raise NotImplementedError("Do not know how to handle LAPACK %s on mac os x" % (info,))
79:     else:
80:         wrapper_sources += [
81:             os.path.join(path, 'src', 'wrap_dummy_g77_abi.f'),
82:             os.path.join(path, 'src', 'wrap_dummy_accelerate.f'),
83:         ]
84:     return wrapper_sources
85: 
86: 
87: def needs_sgemv_fix(info):
88:     '''Returns True if SGEMV must be fixed.'''
89:     if uses_accelerate(info):
90:         return True
91:     else:
92:         return False
93: 
94: 
95: def get_sgemv_fix(info):
96:     ''' Returns source file needed to correct SGEMV '''
97:     path = os.path.abspath(os.path.dirname(__file__))
98:     if needs_sgemv_fix(info):
99:         return [os.path.join(path, 'src', 'apple_sgemv_fix.c')]
100:     else:
101:         return []
102: 
103: 
104: def split_fortran_files(source_dir, subroutines=None):
105:     '''Split each file in `source_dir` into separate files per subroutine.
106: 
107:     Parameters
108:     ----------
109:     source_dir : str
110:         Full path to directory in which sources to be split are located.
111:     subroutines : list of str, optional
112:         Subroutines to split. (Default: all)
113: 
114:     Returns
115:     -------
116:     fnames : list of str
117:         List of file names (not including any path) that were created
118:         in `source_dir`.
119: 
120:     Notes
121:     -----
122:     This function is useful for code that can't be compiled with g77 because of
123:     type casting errors which do work with gfortran.
124: 
125:     Created files are named: ``original_name + '_subr_i' + '.f'``, with ``i``
126:     starting at zero and ending at ``num_subroutines_in_file - 1``.
127: 
128:     '''
129: 
130:     if subroutines is not None:
131:         subroutines = [x.lower() for x in subroutines]
132: 
133:     def split_file(fname):
134:         with open(fname, 'rb') as f:
135:             lines = f.readlines()
136:             subs = []
137:             need_split_next = True
138: 
139:             # find lines with SUBROUTINE statements
140:             for ix, line in enumerate(lines):
141:                 m = re.match(b'^\\s+subroutine\\s+([a-z0-9_]+)\\s*\\(', line, re.I)
142:                 if m and line[0] not in b'Cc!*':
143:                     if subroutines is not None:
144:                         subr_name = m.group(1).decode('ascii').lower()
145:                         subr_wanted = (subr_name in subroutines)
146:                     else:
147:                         subr_wanted = True
148:                     if subr_wanted or need_split_next:
149:                         need_split_next = subr_wanted
150:                         subs.append(ix)
151: 
152:             # check if no split needed
153:             if len(subs) <= 1:
154:                 return [fname]
155: 
156:             # write out one file per subroutine
157:             new_fnames = []
158:             num_files = len(subs)
159:             for nfile in range(num_files):
160:                 new_fname = fname[:-2] + '_subr_' + str(nfile) + '.f'
161:                 new_fnames.append(new_fname)
162:                 if not newer(fname, new_fname):
163:                     continue
164:                 with open(new_fname, 'wb') as fn:
165:                     if nfile + 1 == num_files:
166:                         fn.writelines(lines[subs[nfile]:])
167:                     else:
168:                         fn.writelines(lines[subs[nfile]:subs[nfile+1]])
169: 
170:         return new_fnames
171: 
172:     exclude_pattern = re.compile('_subr_[0-9]')
173:     source_fnames = [f for f in glob.glob(os.path.join(source_dir, '*.f'))
174:                              if not exclude_pattern.search(os.path.basename(f))]
175:     fnames = []
176:     for source_fname in source_fnames:
177:         created_files = split_file(source_fname)
178:         if created_files is not None:
179:             for cfile in created_files:
180:                 fnames.append(os.path.basename(cfile))
181: 
182:     return fnames
183: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import re' statement (line 1)
import re

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import sys' statement (line 2)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import glob' statement (line 4)
import glob

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'glob', glob, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from distutils.dep_util import newer' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_build_utils/')
import_705852 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.dep_util')

if (type(import_705852) is not StypyTypeError):

    if (import_705852 != 'pyd_module'):
        __import__(import_705852)
        sys_modules_705853 = sys.modules[import_705852]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.dep_util', sys_modules_705853.module_type_store, module_type_store, ['newer'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_705853, sys_modules_705853.module_type_store, module_type_store)
    else:
        from distutils.dep_util import newer

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.dep_util', None, module_type_store, ['newer'], [newer])

else:
    # Assigning a type to the variable 'distutils.dep_util' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.dep_util', import_705852)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_build_utils/')


# Assigning a List to a Name (line 8):
__all__ = ['needs_g77_abi_wrapper', 'split_fortran_files', 'get_g77_abi_wrappers', 'needs_sgemv_fix', 'get_sgemv_fix']
module_type_store.set_exportable_members(['needs_g77_abi_wrapper', 'split_fortran_files', 'get_g77_abi_wrappers', 'needs_sgemv_fix', 'get_sgemv_fix'])

# Obtaining an instance of the builtin type 'list' (line 8)
list_705854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)
str_705855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 11), 'str', 'needs_g77_abi_wrapper')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_705854, str_705855)
# Adding element type (line 8)
str_705856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 36), 'str', 'split_fortran_files')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_705854, str_705856)
# Adding element type (line 8)
str_705857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'str', 'get_g77_abi_wrappers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_705854, str_705857)
# Adding element type (line 8)
str_705858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 11), 'str', 'needs_sgemv_fix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_705854, str_705858)
# Adding element type (line 8)
str_705859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 30), 'str', 'get_sgemv_fix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_705854, str_705859)

# Assigning a type to the variable '__all__' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__all__', list_705854)

@norecursion
def uses_veclib(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'uses_veclib'
    module_type_store = module_type_store.open_function_context('uses_veclib', 13, 0, False)
    
    # Passed parameters checking function
    uses_veclib.stypy_localization = localization
    uses_veclib.stypy_type_of_self = None
    uses_veclib.stypy_type_store = module_type_store
    uses_veclib.stypy_function_name = 'uses_veclib'
    uses_veclib.stypy_param_names_list = ['info']
    uses_veclib.stypy_varargs_param_name = None
    uses_veclib.stypy_kwargs_param_name = None
    uses_veclib.stypy_call_defaults = defaults
    uses_veclib.stypy_call_varargs = varargs
    uses_veclib.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'uses_veclib', ['info'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'uses_veclib', localization, ['info'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'uses_veclib(...)' code ##################

    
    
    # Getting the type of 'sys' (line 14)
    sys_705860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 7), 'sys')
    # Obtaining the member 'platform' of a type (line 14)
    platform_705861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 7), sys_705860, 'platform')
    str_705862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 23), 'str', 'darwin')
    # Applying the binary operator '!=' (line 14)
    result_ne_705863 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 7), '!=', platform_705861, str_705862)
    
    # Testing the type of an if condition (line 14)
    if_condition_705864 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 4), result_ne_705863)
    # Assigning a type to the variable 'if_condition_705864' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'if_condition_705864', if_condition_705864)
    # SSA begins for if statement (line 14)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'False' (line 15)
    False_705865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', False_705865)
    # SSA join for if statement (line 14)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 16):
    
    # Call to compile(...): (line 16)
    # Processing the call arguments (line 16)
    str_705868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 30), 'str', 'vecLib')
    # Processing the call keyword arguments (line 16)
    kwargs_705869 = {}
    # Getting the type of 're' (line 16)
    re_705866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 19), 're', False)
    # Obtaining the member 'compile' of a type (line 16)
    compile_705867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 19), re_705866, 'compile')
    # Calling compile(args, kwargs) (line 16)
    compile_call_result_705870 = invoke(stypy.reporting.localization.Localization(__file__, 16, 19), compile_705867, *[str_705868], **kwargs_705869)
    
    # Assigning a type to the variable 'r_accelerate' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'r_accelerate', compile_call_result_705870)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to get(...): (line 17)
    # Processing the call arguments (line 17)
    str_705873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 31), 'str', 'extra_link_args')
    str_705874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 50), 'str', '')
    # Processing the call keyword arguments (line 17)
    kwargs_705875 = {}
    # Getting the type of 'info' (line 17)
    info_705871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 22), 'info', False)
    # Obtaining the member 'get' of a type (line 17)
    get_705872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 22), info_705871, 'get')
    # Calling get(args, kwargs) (line 17)
    get_call_result_705876 = invoke(stypy.reporting.localization.Localization(__file__, 17, 22), get_705872, *[str_705873, str_705874], **kwargs_705875)
    
    # Assigning a type to the variable 'extra_link_args' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'extra_link_args', get_call_result_705876)
    
    # Getting the type of 'extra_link_args' (line 18)
    extra_link_args_705877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 15), 'extra_link_args')
    # Testing the type of a for loop iterable (line 18)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 18, 4), extra_link_args_705877)
    # Getting the type of the for loop variable (line 18)
    for_loop_var_705878 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 18, 4), extra_link_args_705877)
    # Assigning a type to the variable 'arg' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'arg', for_loop_var_705878)
    # SSA begins for a for statement (line 18)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to search(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'arg' (line 19)
    arg_705881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 31), 'arg', False)
    # Processing the call keyword arguments (line 19)
    kwargs_705882 = {}
    # Getting the type of 'r_accelerate' (line 19)
    r_accelerate_705879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 11), 'r_accelerate', False)
    # Obtaining the member 'search' of a type (line 19)
    search_705880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 11), r_accelerate_705879, 'search')
    # Calling search(args, kwargs) (line 19)
    search_call_result_705883 = invoke(stypy.reporting.localization.Localization(__file__, 19, 11), search_705880, *[arg_705881], **kwargs_705882)
    
    # Testing the type of an if condition (line 19)
    if_condition_705884 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 19, 8), search_call_result_705883)
    # Assigning a type to the variable 'if_condition_705884' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'if_condition_705884', if_condition_705884)
    # SSA begins for if statement (line 19)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'True' (line 20)
    True_705885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 19), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'stypy_return_type', True_705885)
    # SSA join for if statement (line 19)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'False' (line 21)
    False_705886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type', False_705886)
    
    # ################# End of 'uses_veclib(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'uses_veclib' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_705887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_705887)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'uses_veclib'
    return stypy_return_type_705887

# Assigning a type to the variable 'uses_veclib' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'uses_veclib', uses_veclib)

@norecursion
def uses_accelerate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'uses_accelerate'
    module_type_store = module_type_store.open_function_context('uses_accelerate', 24, 0, False)
    
    # Passed parameters checking function
    uses_accelerate.stypy_localization = localization
    uses_accelerate.stypy_type_of_self = None
    uses_accelerate.stypy_type_store = module_type_store
    uses_accelerate.stypy_function_name = 'uses_accelerate'
    uses_accelerate.stypy_param_names_list = ['info']
    uses_accelerate.stypy_varargs_param_name = None
    uses_accelerate.stypy_kwargs_param_name = None
    uses_accelerate.stypy_call_defaults = defaults
    uses_accelerate.stypy_call_varargs = varargs
    uses_accelerate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'uses_accelerate', ['info'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'uses_accelerate', localization, ['info'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'uses_accelerate(...)' code ##################

    
    
    # Getting the type of 'sys' (line 25)
    sys_705888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 7), 'sys')
    # Obtaining the member 'platform' of a type (line 25)
    platform_705889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 7), sys_705888, 'platform')
    str_705890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 23), 'str', 'darwin')
    # Applying the binary operator '!=' (line 25)
    result_ne_705891 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 7), '!=', platform_705889, str_705890)
    
    # Testing the type of an if condition (line 25)
    if_condition_705892 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 25, 4), result_ne_705891)
    # Assigning a type to the variable 'if_condition_705892' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'if_condition_705892', if_condition_705892)
    # SSA begins for if statement (line 25)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'False' (line 26)
    False_705893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'stypy_return_type', False_705893)
    # SSA join for if statement (line 25)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 27):
    
    # Call to compile(...): (line 27)
    # Processing the call arguments (line 27)
    str_705896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 30), 'str', 'Accelerate')
    # Processing the call keyword arguments (line 27)
    kwargs_705897 = {}
    # Getting the type of 're' (line 27)
    re_705894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 19), 're', False)
    # Obtaining the member 'compile' of a type (line 27)
    compile_705895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 19), re_705894, 'compile')
    # Calling compile(args, kwargs) (line 27)
    compile_call_result_705898 = invoke(stypy.reporting.localization.Localization(__file__, 27, 19), compile_705895, *[str_705896], **kwargs_705897)
    
    # Assigning a type to the variable 'r_accelerate' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'r_accelerate', compile_call_result_705898)
    
    # Assigning a Call to a Name (line 28):
    
    # Call to get(...): (line 28)
    # Processing the call arguments (line 28)
    str_705901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 31), 'str', 'extra_link_args')
    str_705902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 50), 'str', '')
    # Processing the call keyword arguments (line 28)
    kwargs_705903 = {}
    # Getting the type of 'info' (line 28)
    info_705899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 22), 'info', False)
    # Obtaining the member 'get' of a type (line 28)
    get_705900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 22), info_705899, 'get')
    # Calling get(args, kwargs) (line 28)
    get_call_result_705904 = invoke(stypy.reporting.localization.Localization(__file__, 28, 22), get_705900, *[str_705901, str_705902], **kwargs_705903)
    
    # Assigning a type to the variable 'extra_link_args' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'extra_link_args', get_call_result_705904)
    
    # Getting the type of 'extra_link_args' (line 29)
    extra_link_args_705905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'extra_link_args')
    # Testing the type of a for loop iterable (line 29)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 29, 4), extra_link_args_705905)
    # Getting the type of the for loop variable (line 29)
    for_loop_var_705906 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 29, 4), extra_link_args_705905)
    # Assigning a type to the variable 'arg' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'arg', for_loop_var_705906)
    # SSA begins for a for statement (line 29)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to search(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'arg' (line 30)
    arg_705909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 31), 'arg', False)
    # Processing the call keyword arguments (line 30)
    kwargs_705910 = {}
    # Getting the type of 'r_accelerate' (line 30)
    r_accelerate_705907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'r_accelerate', False)
    # Obtaining the member 'search' of a type (line 30)
    search_705908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 11), r_accelerate_705907, 'search')
    # Calling search(args, kwargs) (line 30)
    search_call_result_705911 = invoke(stypy.reporting.localization.Localization(__file__, 30, 11), search_705908, *[arg_705909], **kwargs_705910)
    
    # Testing the type of an if condition (line 30)
    if_condition_705912 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 30, 8), search_call_result_705911)
    # Assigning a type to the variable 'if_condition_705912' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'if_condition_705912', if_condition_705912)
    # SSA begins for if statement (line 30)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'True' (line 31)
    True_705913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 19), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'stypy_return_type', True_705913)
    # SSA join for if statement (line 30)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'False' (line 32)
    False_705914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type', False_705914)
    
    # ################# End of 'uses_accelerate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'uses_accelerate' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_705915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_705915)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'uses_accelerate'
    return stypy_return_type_705915

# Assigning a type to the variable 'uses_accelerate' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'uses_accelerate', uses_accelerate)

@norecursion
def uses_mkl(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'uses_mkl'
    module_type_store = module_type_store.open_function_context('uses_mkl', 35, 0, False)
    
    # Passed parameters checking function
    uses_mkl.stypy_localization = localization
    uses_mkl.stypy_type_of_self = None
    uses_mkl.stypy_type_store = module_type_store
    uses_mkl.stypy_function_name = 'uses_mkl'
    uses_mkl.stypy_param_names_list = ['info']
    uses_mkl.stypy_varargs_param_name = None
    uses_mkl.stypy_kwargs_param_name = None
    uses_mkl.stypy_call_defaults = defaults
    uses_mkl.stypy_call_varargs = varargs
    uses_mkl.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'uses_mkl', ['info'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'uses_mkl', localization, ['info'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'uses_mkl(...)' code ##################

    
    # Assigning a Call to a Name (line 36):
    
    # Call to compile(...): (line 36)
    # Processing the call arguments (line 36)
    str_705918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 23), 'str', 'mkl')
    # Processing the call keyword arguments (line 36)
    kwargs_705919 = {}
    # Getting the type of 're' (line 36)
    re_705916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 're', False)
    # Obtaining the member 'compile' of a type (line 36)
    compile_705917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), re_705916, 'compile')
    # Calling compile(args, kwargs) (line 36)
    compile_call_result_705920 = invoke(stypy.reporting.localization.Localization(__file__, 36, 12), compile_705917, *[str_705918], **kwargs_705919)
    
    # Assigning a type to the variable 'r_mkl' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'r_mkl', compile_call_result_705920)
    
    # Assigning a Call to a Name (line 37):
    
    # Call to get(...): (line 37)
    # Processing the call arguments (line 37)
    str_705923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 25), 'str', 'libraries')
    str_705924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 38), 'str', '')
    # Processing the call keyword arguments (line 37)
    kwargs_705925 = {}
    # Getting the type of 'info' (line 37)
    info_705921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 16), 'info', False)
    # Obtaining the member 'get' of a type (line 37)
    get_705922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 16), info_705921, 'get')
    # Calling get(args, kwargs) (line 37)
    get_call_result_705926 = invoke(stypy.reporting.localization.Localization(__file__, 37, 16), get_705922, *[str_705923, str_705924], **kwargs_705925)
    
    # Assigning a type to the variable 'libraries' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'libraries', get_call_result_705926)
    
    # Getting the type of 'libraries' (line 38)
    libraries_705927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 19), 'libraries')
    # Testing the type of a for loop iterable (line 38)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 38, 4), libraries_705927)
    # Getting the type of the for loop variable (line 38)
    for_loop_var_705928 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 38, 4), libraries_705927)
    # Assigning a type to the variable 'library' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'library', for_loop_var_705928)
    # SSA begins for a for statement (line 38)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to search(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'library' (line 39)
    library_705931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 24), 'library', False)
    # Processing the call keyword arguments (line 39)
    kwargs_705932 = {}
    # Getting the type of 'r_mkl' (line 39)
    r_mkl_705929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 11), 'r_mkl', False)
    # Obtaining the member 'search' of a type (line 39)
    search_705930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 11), r_mkl_705929, 'search')
    # Calling search(args, kwargs) (line 39)
    search_call_result_705933 = invoke(stypy.reporting.localization.Localization(__file__, 39, 11), search_705930, *[library_705931], **kwargs_705932)
    
    # Testing the type of an if condition (line 39)
    if_condition_705934 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 8), search_call_result_705933)
    # Assigning a type to the variable 'if_condition_705934' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'if_condition_705934', if_condition_705934)
    # SSA begins for if statement (line 39)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'True' (line 40)
    True_705935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'stypy_return_type', True_705935)
    # SSA join for if statement (line 39)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'False' (line 42)
    False_705936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type', False_705936)
    
    # ################# End of 'uses_mkl(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'uses_mkl' in the type store
    # Getting the type of 'stypy_return_type' (line 35)
    stypy_return_type_705937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_705937)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'uses_mkl'
    return stypy_return_type_705937

# Assigning a type to the variable 'uses_mkl' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'uses_mkl', uses_mkl)

@norecursion
def needs_g77_abi_wrapper(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'needs_g77_abi_wrapper'
    module_type_store = module_type_store.open_function_context('needs_g77_abi_wrapper', 45, 0, False)
    
    # Passed parameters checking function
    needs_g77_abi_wrapper.stypy_localization = localization
    needs_g77_abi_wrapper.stypy_type_of_self = None
    needs_g77_abi_wrapper.stypy_type_store = module_type_store
    needs_g77_abi_wrapper.stypy_function_name = 'needs_g77_abi_wrapper'
    needs_g77_abi_wrapper.stypy_param_names_list = ['info']
    needs_g77_abi_wrapper.stypy_varargs_param_name = None
    needs_g77_abi_wrapper.stypy_kwargs_param_name = None
    needs_g77_abi_wrapper.stypy_call_defaults = defaults
    needs_g77_abi_wrapper.stypy_call_varargs = varargs
    needs_g77_abi_wrapper.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'needs_g77_abi_wrapper', ['info'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'needs_g77_abi_wrapper', localization, ['info'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'needs_g77_abi_wrapper(...)' code ##################

    str_705938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 4), 'str', 'Returns True if g77 ABI wrapper must be used.')
    
    
    # Evaluating a boolean operation
    
    # Call to uses_accelerate(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'info' (line 47)
    info_705940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 23), 'info', False)
    # Processing the call keyword arguments (line 47)
    kwargs_705941 = {}
    # Getting the type of 'uses_accelerate' (line 47)
    uses_accelerate_705939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 7), 'uses_accelerate', False)
    # Calling uses_accelerate(args, kwargs) (line 47)
    uses_accelerate_call_result_705942 = invoke(stypy.reporting.localization.Localization(__file__, 47, 7), uses_accelerate_705939, *[info_705940], **kwargs_705941)
    
    
    # Call to uses_veclib(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'info' (line 47)
    info_705944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 44), 'info', False)
    # Processing the call keyword arguments (line 47)
    kwargs_705945 = {}
    # Getting the type of 'uses_veclib' (line 47)
    uses_veclib_705943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 32), 'uses_veclib', False)
    # Calling uses_veclib(args, kwargs) (line 47)
    uses_veclib_call_result_705946 = invoke(stypy.reporting.localization.Localization(__file__, 47, 32), uses_veclib_705943, *[info_705944], **kwargs_705945)
    
    # Applying the binary operator 'or' (line 47)
    result_or_keyword_705947 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 7), 'or', uses_accelerate_call_result_705942, uses_veclib_call_result_705946)
    
    # Testing the type of an if condition (line 47)
    if_condition_705948 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 4), result_or_keyword_705947)
    # Assigning a type to the variable 'if_condition_705948' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'if_condition_705948', if_condition_705948)
    # SSA begins for if statement (line 47)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'True' (line 48)
    True_705949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'stypy_return_type', True_705949)
    # SSA branch for the else part of an if statement (line 47)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to uses_mkl(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'info' (line 49)
    info_705951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 18), 'info', False)
    # Processing the call keyword arguments (line 49)
    kwargs_705952 = {}
    # Getting the type of 'uses_mkl' (line 49)
    uses_mkl_705950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 9), 'uses_mkl', False)
    # Calling uses_mkl(args, kwargs) (line 49)
    uses_mkl_call_result_705953 = invoke(stypy.reporting.localization.Localization(__file__, 49, 9), uses_mkl_705950, *[info_705951], **kwargs_705952)
    
    # Testing the type of an if condition (line 49)
    if_condition_705954 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 9), uses_mkl_call_result_705953)
    # Assigning a type to the variable 'if_condition_705954' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 9), 'if_condition_705954', if_condition_705954)
    # SSA begins for if statement (line 49)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'True' (line 50)
    True_705955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'stypy_return_type', True_705955)
    # SSA branch for the else part of an if statement (line 49)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'False' (line 52)
    False_705956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'stypy_return_type', False_705956)
    # SSA join for if statement (line 49)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 47)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'needs_g77_abi_wrapper(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'needs_g77_abi_wrapper' in the type store
    # Getting the type of 'stypy_return_type' (line 45)
    stypy_return_type_705957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_705957)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'needs_g77_abi_wrapper'
    return stypy_return_type_705957

# Assigning a type to the variable 'needs_g77_abi_wrapper' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'needs_g77_abi_wrapper', needs_g77_abi_wrapper)

@norecursion
def get_g77_abi_wrappers(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_g77_abi_wrappers'
    module_type_store = module_type_store.open_function_context('get_g77_abi_wrappers', 55, 0, False)
    
    # Passed parameters checking function
    get_g77_abi_wrappers.stypy_localization = localization
    get_g77_abi_wrappers.stypy_type_of_self = None
    get_g77_abi_wrappers.stypy_type_store = module_type_store
    get_g77_abi_wrappers.stypy_function_name = 'get_g77_abi_wrappers'
    get_g77_abi_wrappers.stypy_param_names_list = ['info']
    get_g77_abi_wrappers.stypy_varargs_param_name = None
    get_g77_abi_wrappers.stypy_kwargs_param_name = None
    get_g77_abi_wrappers.stypy_call_defaults = defaults
    get_g77_abi_wrappers.stypy_call_varargs = varargs
    get_g77_abi_wrappers.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_g77_abi_wrappers', ['info'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_g77_abi_wrappers', localization, ['info'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_g77_abi_wrappers(...)' code ##################

    str_705958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, (-1)), 'str', '\n    Returns file names of source files containing Fortran ABI wrapper\n    routines.\n    ')
    
    # Assigning a List to a Name (line 60):
    
    # Obtaining an instance of the builtin type 'list' (line 60)
    list_705959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 60)
    
    # Assigning a type to the variable 'wrapper_sources' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'wrapper_sources', list_705959)
    
    # Assigning a Call to a Name (line 62):
    
    # Call to abspath(...): (line 62)
    # Processing the call arguments (line 62)
    
    # Call to dirname(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of '__file__' (line 62)
    file___705966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 43), '__file__', False)
    # Processing the call keyword arguments (line 62)
    kwargs_705967 = {}
    # Getting the type of 'os' (line 62)
    os_705963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 27), 'os', False)
    # Obtaining the member 'path' of a type (line 62)
    path_705964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 27), os_705963, 'path')
    # Obtaining the member 'dirname' of a type (line 62)
    dirname_705965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 27), path_705964, 'dirname')
    # Calling dirname(args, kwargs) (line 62)
    dirname_call_result_705968 = invoke(stypy.reporting.localization.Localization(__file__, 62, 27), dirname_705965, *[file___705966], **kwargs_705967)
    
    # Processing the call keyword arguments (line 62)
    kwargs_705969 = {}
    # Getting the type of 'os' (line 62)
    os_705960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 62)
    path_705961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 11), os_705960, 'path')
    # Obtaining the member 'abspath' of a type (line 62)
    abspath_705962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 11), path_705961, 'abspath')
    # Calling abspath(args, kwargs) (line 62)
    abspath_call_result_705970 = invoke(stypy.reporting.localization.Localization(__file__, 62, 11), abspath_705962, *[dirname_call_result_705968], **kwargs_705969)
    
    # Assigning a type to the variable 'path' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'path', abspath_call_result_705970)
    
    
    # Call to needs_g77_abi_wrapper(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'info' (line 63)
    info_705972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 29), 'info', False)
    # Processing the call keyword arguments (line 63)
    kwargs_705973 = {}
    # Getting the type of 'needs_g77_abi_wrapper' (line 63)
    needs_g77_abi_wrapper_705971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 7), 'needs_g77_abi_wrapper', False)
    # Calling needs_g77_abi_wrapper(args, kwargs) (line 63)
    needs_g77_abi_wrapper_call_result_705974 = invoke(stypy.reporting.localization.Localization(__file__, 63, 7), needs_g77_abi_wrapper_705971, *[info_705972], **kwargs_705973)
    
    # Testing the type of an if condition (line 63)
    if_condition_705975 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 63, 4), needs_g77_abi_wrapper_call_result_705974)
    # Assigning a type to the variable 'if_condition_705975' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'if_condition_705975', if_condition_705975)
    # SSA begins for if statement (line 63)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'wrapper_sources' (line 64)
    wrapper_sources_705976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'wrapper_sources')
    
    # Obtaining an instance of the builtin type 'list' (line 64)
    list_705977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 64)
    # Adding element type (line 64)
    
    # Call to join(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'path' (line 65)
    path_705981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 25), 'path', False)
    str_705982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 31), 'str', 'src')
    str_705983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 38), 'str', 'wrap_g77_abi_f.f')
    # Processing the call keyword arguments (line 65)
    kwargs_705984 = {}
    # Getting the type of 'os' (line 65)
    os_705978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'os', False)
    # Obtaining the member 'path' of a type (line 65)
    path_705979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), os_705978, 'path')
    # Obtaining the member 'join' of a type (line 65)
    join_705980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), path_705979, 'join')
    # Calling join(args, kwargs) (line 65)
    join_call_result_705985 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), join_705980, *[path_705981, str_705982, str_705983], **kwargs_705984)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 27), list_705977, join_call_result_705985)
    # Adding element type (line 64)
    
    # Call to join(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'path' (line 66)
    path_705989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'path', False)
    str_705990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 31), 'str', 'src')
    str_705991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 38), 'str', 'wrap_g77_abi_c.c')
    # Processing the call keyword arguments (line 66)
    kwargs_705992 = {}
    # Getting the type of 'os' (line 66)
    os_705986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'os', False)
    # Obtaining the member 'path' of a type (line 66)
    path_705987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), os_705986, 'path')
    # Obtaining the member 'join' of a type (line 66)
    join_705988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), path_705987, 'join')
    # Calling join(args, kwargs) (line 66)
    join_call_result_705993 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), join_705988, *[path_705989, str_705990, str_705991], **kwargs_705992)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 27), list_705977, join_call_result_705993)
    
    # Applying the binary operator '+=' (line 64)
    result_iadd_705994 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 8), '+=', wrapper_sources_705976, list_705977)
    # Assigning a type to the variable 'wrapper_sources' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'wrapper_sources', result_iadd_705994)
    
    
    
    # Call to uses_accelerate(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'info' (line 68)
    info_705996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 27), 'info', False)
    # Processing the call keyword arguments (line 68)
    kwargs_705997 = {}
    # Getting the type of 'uses_accelerate' (line 68)
    uses_accelerate_705995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'uses_accelerate', False)
    # Calling uses_accelerate(args, kwargs) (line 68)
    uses_accelerate_call_result_705998 = invoke(stypy.reporting.localization.Localization(__file__, 68, 11), uses_accelerate_705995, *[info_705996], **kwargs_705997)
    
    # Testing the type of an if condition (line 68)
    if_condition_705999 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 8), uses_accelerate_call_result_705998)
    # Assigning a type to the variable 'if_condition_705999' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'if_condition_705999', if_condition_705999)
    # SSA begins for if statement (line 68)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'wrapper_sources' (line 69)
    wrapper_sources_706000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'wrapper_sources')
    
    # Obtaining an instance of the builtin type 'list' (line 69)
    list_706001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 69)
    # Adding element type (line 69)
    
    # Call to join(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'path' (line 70)
    path_706005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 33), 'path', False)
    str_706006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 39), 'str', 'src')
    str_706007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 46), 'str', 'wrap_accelerate_c.c')
    # Processing the call keyword arguments (line 70)
    kwargs_706008 = {}
    # Getting the type of 'os' (line 70)
    os_706002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 20), 'os', False)
    # Obtaining the member 'path' of a type (line 70)
    path_706003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 20), os_706002, 'path')
    # Obtaining the member 'join' of a type (line 70)
    join_706004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 20), path_706003, 'join')
    # Calling join(args, kwargs) (line 70)
    join_call_result_706009 = invoke(stypy.reporting.localization.Localization(__file__, 70, 20), join_706004, *[path_706005, str_706006, str_706007], **kwargs_706008)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 31), list_706001, join_call_result_706009)
    # Adding element type (line 69)
    
    # Call to join(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'path' (line 71)
    path_706013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 33), 'path', False)
    str_706014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 39), 'str', 'src')
    str_706015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 46), 'str', 'wrap_accelerate_f.f')
    # Processing the call keyword arguments (line 71)
    kwargs_706016 = {}
    # Getting the type of 'os' (line 71)
    os_706010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 20), 'os', False)
    # Obtaining the member 'path' of a type (line 71)
    path_706011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 20), os_706010, 'path')
    # Obtaining the member 'join' of a type (line 71)
    join_706012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 20), path_706011, 'join')
    # Calling join(args, kwargs) (line 71)
    join_call_result_706017 = invoke(stypy.reporting.localization.Localization(__file__, 71, 20), join_706012, *[path_706013, str_706014, str_706015], **kwargs_706016)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 31), list_706001, join_call_result_706017)
    
    # Applying the binary operator '+=' (line 69)
    result_iadd_706018 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 12), '+=', wrapper_sources_706000, list_706001)
    # Assigning a type to the variable 'wrapper_sources' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'wrapper_sources', result_iadd_706018)
    
    # SSA branch for the else part of an if statement (line 68)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to uses_mkl(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'info' (line 73)
    info_706020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 22), 'info', False)
    # Processing the call keyword arguments (line 73)
    kwargs_706021 = {}
    # Getting the type of 'uses_mkl' (line 73)
    uses_mkl_706019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 13), 'uses_mkl', False)
    # Calling uses_mkl(args, kwargs) (line 73)
    uses_mkl_call_result_706022 = invoke(stypy.reporting.localization.Localization(__file__, 73, 13), uses_mkl_706019, *[info_706020], **kwargs_706021)
    
    # Testing the type of an if condition (line 73)
    if_condition_706023 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 13), uses_mkl_call_result_706022)
    # Assigning a type to the variable 'if_condition_706023' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 13), 'if_condition_706023', if_condition_706023)
    # SSA begins for if statement (line 73)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'wrapper_sources' (line 74)
    wrapper_sources_706024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'wrapper_sources')
    
    # Obtaining an instance of the builtin type 'list' (line 74)
    list_706025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 74)
    # Adding element type (line 74)
    
    # Call to join(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'path' (line 75)
    path_706029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 33), 'path', False)
    str_706030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 39), 'str', 'src')
    str_706031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 46), 'str', 'wrap_dummy_accelerate.f')
    # Processing the call keyword arguments (line 75)
    kwargs_706032 = {}
    # Getting the type of 'os' (line 75)
    os_706026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), 'os', False)
    # Obtaining the member 'path' of a type (line 75)
    path_706027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 20), os_706026, 'path')
    # Obtaining the member 'join' of a type (line 75)
    join_706028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 20), path_706027, 'join')
    # Calling join(args, kwargs) (line 75)
    join_call_result_706033 = invoke(stypy.reporting.localization.Localization(__file__, 75, 20), join_706028, *[path_706029, str_706030, str_706031], **kwargs_706032)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 31), list_706025, join_call_result_706033)
    
    # Applying the binary operator '+=' (line 74)
    result_iadd_706034 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 12), '+=', wrapper_sources_706024, list_706025)
    # Assigning a type to the variable 'wrapper_sources' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'wrapper_sources', result_iadd_706034)
    
    # SSA branch for the else part of an if statement (line 73)
    module_type_store.open_ssa_branch('else')
    
    # Call to NotImplementedError(...): (line 78)
    # Processing the call arguments (line 78)
    str_706036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 38), 'str', 'Do not know how to handle LAPACK %s on mac os x')
    
    # Obtaining an instance of the builtin type 'tuple' (line 78)
    tuple_706037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 91), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 78)
    # Adding element type (line 78)
    # Getting the type of 'info' (line 78)
    info_706038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 91), 'info', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 91), tuple_706037, info_706038)
    
    # Applying the binary operator '%' (line 78)
    result_mod_706039 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 38), '%', str_706036, tuple_706037)
    
    # Processing the call keyword arguments (line 78)
    kwargs_706040 = {}
    # Getting the type of 'NotImplementedError' (line 78)
    NotImplementedError_706035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 18), 'NotImplementedError', False)
    # Calling NotImplementedError(args, kwargs) (line 78)
    NotImplementedError_call_result_706041 = invoke(stypy.reporting.localization.Localization(__file__, 78, 18), NotImplementedError_706035, *[result_mod_706039], **kwargs_706040)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 78, 12), NotImplementedError_call_result_706041, 'raise parameter', BaseException)
    # SSA join for if statement (line 73)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 68)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 63)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'wrapper_sources' (line 80)
    wrapper_sources_706042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'wrapper_sources')
    
    # Obtaining an instance of the builtin type 'list' (line 80)
    list_706043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 80)
    # Adding element type (line 80)
    
    # Call to join(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'path' (line 81)
    path_706047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 25), 'path', False)
    str_706048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 31), 'str', 'src')
    str_706049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 38), 'str', 'wrap_dummy_g77_abi.f')
    # Processing the call keyword arguments (line 81)
    kwargs_706050 = {}
    # Getting the type of 'os' (line 81)
    os_706044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'os', False)
    # Obtaining the member 'path' of a type (line 81)
    path_706045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), os_706044, 'path')
    # Obtaining the member 'join' of a type (line 81)
    join_706046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), path_706045, 'join')
    # Calling join(args, kwargs) (line 81)
    join_call_result_706051 = invoke(stypy.reporting.localization.Localization(__file__, 81, 12), join_706046, *[path_706047, str_706048, str_706049], **kwargs_706050)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 27), list_706043, join_call_result_706051)
    # Adding element type (line 80)
    
    # Call to join(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'path' (line 82)
    path_706055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 25), 'path', False)
    str_706056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 31), 'str', 'src')
    str_706057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 38), 'str', 'wrap_dummy_accelerate.f')
    # Processing the call keyword arguments (line 82)
    kwargs_706058 = {}
    # Getting the type of 'os' (line 82)
    os_706052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'os', False)
    # Obtaining the member 'path' of a type (line 82)
    path_706053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 12), os_706052, 'path')
    # Obtaining the member 'join' of a type (line 82)
    join_706054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 12), path_706053, 'join')
    # Calling join(args, kwargs) (line 82)
    join_call_result_706059 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), join_706054, *[path_706055, str_706056, str_706057], **kwargs_706058)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 27), list_706043, join_call_result_706059)
    
    # Applying the binary operator '+=' (line 80)
    result_iadd_706060 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 8), '+=', wrapper_sources_706042, list_706043)
    # Assigning a type to the variable 'wrapper_sources' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'wrapper_sources', result_iadd_706060)
    
    # SSA join for if statement (line 63)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'wrapper_sources' (line 84)
    wrapper_sources_706061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 11), 'wrapper_sources')
    # Assigning a type to the variable 'stypy_return_type' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type', wrapper_sources_706061)
    
    # ################# End of 'get_g77_abi_wrappers(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_g77_abi_wrappers' in the type store
    # Getting the type of 'stypy_return_type' (line 55)
    stypy_return_type_706062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_706062)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_g77_abi_wrappers'
    return stypy_return_type_706062

# Assigning a type to the variable 'get_g77_abi_wrappers' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'get_g77_abi_wrappers', get_g77_abi_wrappers)

@norecursion
def needs_sgemv_fix(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'needs_sgemv_fix'
    module_type_store = module_type_store.open_function_context('needs_sgemv_fix', 87, 0, False)
    
    # Passed parameters checking function
    needs_sgemv_fix.stypy_localization = localization
    needs_sgemv_fix.stypy_type_of_self = None
    needs_sgemv_fix.stypy_type_store = module_type_store
    needs_sgemv_fix.stypy_function_name = 'needs_sgemv_fix'
    needs_sgemv_fix.stypy_param_names_list = ['info']
    needs_sgemv_fix.stypy_varargs_param_name = None
    needs_sgemv_fix.stypy_kwargs_param_name = None
    needs_sgemv_fix.stypy_call_defaults = defaults
    needs_sgemv_fix.stypy_call_varargs = varargs
    needs_sgemv_fix.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'needs_sgemv_fix', ['info'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'needs_sgemv_fix', localization, ['info'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'needs_sgemv_fix(...)' code ##################

    str_706063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 4), 'str', 'Returns True if SGEMV must be fixed.')
    
    
    # Call to uses_accelerate(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'info' (line 89)
    info_706065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 23), 'info', False)
    # Processing the call keyword arguments (line 89)
    kwargs_706066 = {}
    # Getting the type of 'uses_accelerate' (line 89)
    uses_accelerate_706064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 7), 'uses_accelerate', False)
    # Calling uses_accelerate(args, kwargs) (line 89)
    uses_accelerate_call_result_706067 = invoke(stypy.reporting.localization.Localization(__file__, 89, 7), uses_accelerate_706064, *[info_706065], **kwargs_706066)
    
    # Testing the type of an if condition (line 89)
    if_condition_706068 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 4), uses_accelerate_call_result_706067)
    # Assigning a type to the variable 'if_condition_706068' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'if_condition_706068', if_condition_706068)
    # SSA begins for if statement (line 89)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'True' (line 90)
    True_706069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 15), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'stypy_return_type', True_706069)
    # SSA branch for the else part of an if statement (line 89)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'False' (line 92)
    False_706070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'stypy_return_type', False_706070)
    # SSA join for if statement (line 89)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'needs_sgemv_fix(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'needs_sgemv_fix' in the type store
    # Getting the type of 'stypy_return_type' (line 87)
    stypy_return_type_706071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_706071)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'needs_sgemv_fix'
    return stypy_return_type_706071

# Assigning a type to the variable 'needs_sgemv_fix' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'needs_sgemv_fix', needs_sgemv_fix)

@norecursion
def get_sgemv_fix(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_sgemv_fix'
    module_type_store = module_type_store.open_function_context('get_sgemv_fix', 95, 0, False)
    
    # Passed parameters checking function
    get_sgemv_fix.stypy_localization = localization
    get_sgemv_fix.stypy_type_of_self = None
    get_sgemv_fix.stypy_type_store = module_type_store
    get_sgemv_fix.stypy_function_name = 'get_sgemv_fix'
    get_sgemv_fix.stypy_param_names_list = ['info']
    get_sgemv_fix.stypy_varargs_param_name = None
    get_sgemv_fix.stypy_kwargs_param_name = None
    get_sgemv_fix.stypy_call_defaults = defaults
    get_sgemv_fix.stypy_call_varargs = varargs
    get_sgemv_fix.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_sgemv_fix', ['info'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_sgemv_fix', localization, ['info'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_sgemv_fix(...)' code ##################

    str_706072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 4), 'str', ' Returns source file needed to correct SGEMV ')
    
    # Assigning a Call to a Name (line 97):
    
    # Call to abspath(...): (line 97)
    # Processing the call arguments (line 97)
    
    # Call to dirname(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of '__file__' (line 97)
    file___706079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 43), '__file__', False)
    # Processing the call keyword arguments (line 97)
    kwargs_706080 = {}
    # Getting the type of 'os' (line 97)
    os_706076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 27), 'os', False)
    # Obtaining the member 'path' of a type (line 97)
    path_706077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 27), os_706076, 'path')
    # Obtaining the member 'dirname' of a type (line 97)
    dirname_706078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 27), path_706077, 'dirname')
    # Calling dirname(args, kwargs) (line 97)
    dirname_call_result_706081 = invoke(stypy.reporting.localization.Localization(__file__, 97, 27), dirname_706078, *[file___706079], **kwargs_706080)
    
    # Processing the call keyword arguments (line 97)
    kwargs_706082 = {}
    # Getting the type of 'os' (line 97)
    os_706073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 97)
    path_706074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 11), os_706073, 'path')
    # Obtaining the member 'abspath' of a type (line 97)
    abspath_706075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 11), path_706074, 'abspath')
    # Calling abspath(args, kwargs) (line 97)
    abspath_call_result_706083 = invoke(stypy.reporting.localization.Localization(__file__, 97, 11), abspath_706075, *[dirname_call_result_706081], **kwargs_706082)
    
    # Assigning a type to the variable 'path' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'path', abspath_call_result_706083)
    
    
    # Call to needs_sgemv_fix(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'info' (line 98)
    info_706085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 23), 'info', False)
    # Processing the call keyword arguments (line 98)
    kwargs_706086 = {}
    # Getting the type of 'needs_sgemv_fix' (line 98)
    needs_sgemv_fix_706084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 7), 'needs_sgemv_fix', False)
    # Calling needs_sgemv_fix(args, kwargs) (line 98)
    needs_sgemv_fix_call_result_706087 = invoke(stypy.reporting.localization.Localization(__file__, 98, 7), needs_sgemv_fix_706084, *[info_706085], **kwargs_706086)
    
    # Testing the type of an if condition (line 98)
    if_condition_706088 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 4), needs_sgemv_fix_call_result_706087)
    # Assigning a type to the variable 'if_condition_706088' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'if_condition_706088', if_condition_706088)
    # SSA begins for if statement (line 98)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'list' (line 99)
    list_706089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 99)
    # Adding element type (line 99)
    
    # Call to join(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'path' (line 99)
    path_706093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 29), 'path', False)
    str_706094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 35), 'str', 'src')
    str_706095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 42), 'str', 'apple_sgemv_fix.c')
    # Processing the call keyword arguments (line 99)
    kwargs_706096 = {}
    # Getting the type of 'os' (line 99)
    os_706090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'os', False)
    # Obtaining the member 'path' of a type (line 99)
    path_706091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), os_706090, 'path')
    # Obtaining the member 'join' of a type (line 99)
    join_706092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), path_706091, 'join')
    # Calling join(args, kwargs) (line 99)
    join_call_result_706097 = invoke(stypy.reporting.localization.Localization(__file__, 99, 16), join_706092, *[path_706093, str_706094, str_706095], **kwargs_706096)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 15), list_706089, join_call_result_706097)
    
    # Assigning a type to the variable 'stypy_return_type' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'stypy_return_type', list_706089)
    # SSA branch for the else part of an if statement (line 98)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'list' (line 101)
    list_706098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 101)
    
    # Assigning a type to the variable 'stypy_return_type' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'stypy_return_type', list_706098)
    # SSA join for if statement (line 98)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'get_sgemv_fix(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_sgemv_fix' in the type store
    # Getting the type of 'stypy_return_type' (line 95)
    stypy_return_type_706099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_706099)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_sgemv_fix'
    return stypy_return_type_706099

# Assigning a type to the variable 'get_sgemv_fix' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'get_sgemv_fix', get_sgemv_fix)

@norecursion
def split_fortran_files(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 104)
    None_706100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 48), 'None')
    defaults = [None_706100]
    # Create a new context for function 'split_fortran_files'
    module_type_store = module_type_store.open_function_context('split_fortran_files', 104, 0, False)
    
    # Passed parameters checking function
    split_fortran_files.stypy_localization = localization
    split_fortran_files.stypy_type_of_self = None
    split_fortran_files.stypy_type_store = module_type_store
    split_fortran_files.stypy_function_name = 'split_fortran_files'
    split_fortran_files.stypy_param_names_list = ['source_dir', 'subroutines']
    split_fortran_files.stypy_varargs_param_name = None
    split_fortran_files.stypy_kwargs_param_name = None
    split_fortran_files.stypy_call_defaults = defaults
    split_fortran_files.stypy_call_varargs = varargs
    split_fortran_files.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'split_fortran_files', ['source_dir', 'subroutines'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'split_fortran_files', localization, ['source_dir', 'subroutines'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'split_fortran_files(...)' code ##################

    str_706101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, (-1)), 'str', "Split each file in `source_dir` into separate files per subroutine.\n\n    Parameters\n    ----------\n    source_dir : str\n        Full path to directory in which sources to be split are located.\n    subroutines : list of str, optional\n        Subroutines to split. (Default: all)\n\n    Returns\n    -------\n    fnames : list of str\n        List of file names (not including any path) that were created\n        in `source_dir`.\n\n    Notes\n    -----\n    This function is useful for code that can't be compiled with g77 because of\n    type casting errors which do work with gfortran.\n\n    Created files are named: ``original_name + '_subr_i' + '.f'``, with ``i``\n    starting at zero and ending at ``num_subroutines_in_file - 1``.\n\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 130)
    # Getting the type of 'subroutines' (line 130)
    subroutines_706102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'subroutines')
    # Getting the type of 'None' (line 130)
    None_706103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 26), 'None')
    
    (may_be_706104, more_types_in_union_706105) = may_not_be_none(subroutines_706102, None_706103)

    if may_be_706104:

        if more_types_in_union_706105:
            # Runtime conditional SSA (line 130)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a ListComp to a Name (line 131):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'subroutines' (line 131)
        subroutines_706110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 42), 'subroutines')
        comprehension_706111 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), subroutines_706110)
        # Assigning a type to the variable 'x' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 23), 'x', comprehension_706111)
        
        # Call to lower(...): (line 131)
        # Processing the call keyword arguments (line 131)
        kwargs_706108 = {}
        # Getting the type of 'x' (line 131)
        x_706106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 23), 'x', False)
        # Obtaining the member 'lower' of a type (line 131)
        lower_706107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 23), x_706106, 'lower')
        # Calling lower(args, kwargs) (line 131)
        lower_call_result_706109 = invoke(stypy.reporting.localization.Localization(__file__, 131, 23), lower_706107, *[], **kwargs_706108)
        
        list_706112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 23), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_706112, lower_call_result_706109)
        # Assigning a type to the variable 'subroutines' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'subroutines', list_706112)

        if more_types_in_union_706105:
            # SSA join for if statement (line 130)
            module_type_store = module_type_store.join_ssa_context()


    

    @norecursion
    def split_file(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'split_file'
        module_type_store = module_type_store.open_function_context('split_file', 133, 4, False)
        
        # Passed parameters checking function
        split_file.stypy_localization = localization
        split_file.stypy_type_of_self = None
        split_file.stypy_type_store = module_type_store
        split_file.stypy_function_name = 'split_file'
        split_file.stypy_param_names_list = ['fname']
        split_file.stypy_varargs_param_name = None
        split_file.stypy_kwargs_param_name = None
        split_file.stypy_call_defaults = defaults
        split_file.stypy_call_varargs = varargs
        split_file.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'split_file', ['fname'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'split_file', localization, ['fname'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'split_file(...)' code ##################

        
        # Call to open(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'fname' (line 134)
        fname_706114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 18), 'fname', False)
        str_706115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 25), 'str', 'rb')
        # Processing the call keyword arguments (line 134)
        kwargs_706116 = {}
        # Getting the type of 'open' (line 134)
        open_706113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 13), 'open', False)
        # Calling open(args, kwargs) (line 134)
        open_call_result_706117 = invoke(stypy.reporting.localization.Localization(__file__, 134, 13), open_706113, *[fname_706114, str_706115], **kwargs_706116)
        
        with_706118 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 134, 13), open_call_result_706117, 'with parameter', '__enter__', '__exit__')

        if with_706118:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 134)
            enter___706119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 13), open_call_result_706117, '__enter__')
            with_enter_706120 = invoke(stypy.reporting.localization.Localization(__file__, 134, 13), enter___706119)
            # Assigning a type to the variable 'f' (line 134)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 13), 'f', with_enter_706120)
            
            # Assigning a Call to a Name (line 135):
            
            # Call to readlines(...): (line 135)
            # Processing the call keyword arguments (line 135)
            kwargs_706123 = {}
            # Getting the type of 'f' (line 135)
            f_706121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 20), 'f', False)
            # Obtaining the member 'readlines' of a type (line 135)
            readlines_706122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 20), f_706121, 'readlines')
            # Calling readlines(args, kwargs) (line 135)
            readlines_call_result_706124 = invoke(stypy.reporting.localization.Localization(__file__, 135, 20), readlines_706122, *[], **kwargs_706123)
            
            # Assigning a type to the variable 'lines' (line 135)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'lines', readlines_call_result_706124)
            
            # Assigning a List to a Name (line 136):
            
            # Obtaining an instance of the builtin type 'list' (line 136)
            list_706125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 19), 'list')
            # Adding type elements to the builtin type 'list' instance (line 136)
            
            # Assigning a type to the variable 'subs' (line 136)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'subs', list_706125)
            
            # Assigning a Name to a Name (line 137):
            # Getting the type of 'True' (line 137)
            True_706126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 30), 'True')
            # Assigning a type to the variable 'need_split_next' (line 137)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'need_split_next', True_706126)
            
            
            # Call to enumerate(...): (line 140)
            # Processing the call arguments (line 140)
            # Getting the type of 'lines' (line 140)
            lines_706128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 38), 'lines', False)
            # Processing the call keyword arguments (line 140)
            kwargs_706129 = {}
            # Getting the type of 'enumerate' (line 140)
            enumerate_706127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 28), 'enumerate', False)
            # Calling enumerate(args, kwargs) (line 140)
            enumerate_call_result_706130 = invoke(stypy.reporting.localization.Localization(__file__, 140, 28), enumerate_706127, *[lines_706128], **kwargs_706129)
            
            # Testing the type of a for loop iterable (line 140)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 140, 12), enumerate_call_result_706130)
            # Getting the type of the for loop variable (line 140)
            for_loop_var_706131 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 140, 12), enumerate_call_result_706130)
            # Assigning a type to the variable 'ix' (line 140)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'ix', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 12), for_loop_var_706131))
            # Assigning a type to the variable 'line' (line 140)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'line', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 12), for_loop_var_706131))
            # SSA begins for a for statement (line 140)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 141):
            
            # Call to match(...): (line 141)
            # Processing the call arguments (line 141)
            str_706134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 29), 'str', '^\\s+subroutine\\s+([a-z0-9_]+)\\s*\\(')
            # Getting the type of 'line' (line 141)
            line_706135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 72), 'line', False)
            # Getting the type of 're' (line 141)
            re_706136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 78), 're', False)
            # Obtaining the member 'I' of a type (line 141)
            I_706137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 78), re_706136, 'I')
            # Processing the call keyword arguments (line 141)
            kwargs_706138 = {}
            # Getting the type of 're' (line 141)
            re_706132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 20), 're', False)
            # Obtaining the member 'match' of a type (line 141)
            match_706133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 20), re_706132, 'match')
            # Calling match(args, kwargs) (line 141)
            match_call_result_706139 = invoke(stypy.reporting.localization.Localization(__file__, 141, 20), match_706133, *[str_706134, line_706135, I_706137], **kwargs_706138)
            
            # Assigning a type to the variable 'm' (line 141)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'm', match_call_result_706139)
            
            
            # Evaluating a boolean operation
            # Getting the type of 'm' (line 142)
            m_706140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 'm')
            
            
            # Obtaining the type of the subscript
            int_706141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 30), 'int')
            # Getting the type of 'line' (line 142)
            line_706142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 25), 'line')
            # Obtaining the member '__getitem__' of a type (line 142)
            getitem___706143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 25), line_706142, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 142)
            subscript_call_result_706144 = invoke(stypy.reporting.localization.Localization(__file__, 142, 25), getitem___706143, int_706141)
            
            str_706145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 40), 'str', 'Cc!*')
            # Applying the binary operator 'notin' (line 142)
            result_contains_706146 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 25), 'notin', subscript_call_result_706144, str_706145)
            
            # Applying the binary operator 'and' (line 142)
            result_and_keyword_706147 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 19), 'and', m_706140, result_contains_706146)
            
            # Testing the type of an if condition (line 142)
            if_condition_706148 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 16), result_and_keyword_706147)
            # Assigning a type to the variable 'if_condition_706148' (line 142)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 16), 'if_condition_706148', if_condition_706148)
            # SSA begins for if statement (line 142)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Type idiom detected: calculating its left and rigth part (line 143)
            # Getting the type of 'subroutines' (line 143)
            subroutines_706149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 20), 'subroutines')
            # Getting the type of 'None' (line 143)
            None_706150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 42), 'None')
            
            (may_be_706151, more_types_in_union_706152) = may_not_be_none(subroutines_706149, None_706150)

            if may_be_706151:

                if more_types_in_union_706152:
                    # Runtime conditional SSA (line 143)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Call to a Name (line 144):
                
                # Call to lower(...): (line 144)
                # Processing the call keyword arguments (line 144)
                kwargs_706163 = {}
                
                # Call to decode(...): (line 144)
                # Processing the call arguments (line 144)
                str_706159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 54), 'str', 'ascii')
                # Processing the call keyword arguments (line 144)
                kwargs_706160 = {}
                
                # Call to group(...): (line 144)
                # Processing the call arguments (line 144)
                int_706155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 44), 'int')
                # Processing the call keyword arguments (line 144)
                kwargs_706156 = {}
                # Getting the type of 'm' (line 144)
                m_706153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 36), 'm', False)
                # Obtaining the member 'group' of a type (line 144)
                group_706154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 36), m_706153, 'group')
                # Calling group(args, kwargs) (line 144)
                group_call_result_706157 = invoke(stypy.reporting.localization.Localization(__file__, 144, 36), group_706154, *[int_706155], **kwargs_706156)
                
                # Obtaining the member 'decode' of a type (line 144)
                decode_706158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 36), group_call_result_706157, 'decode')
                # Calling decode(args, kwargs) (line 144)
                decode_call_result_706161 = invoke(stypy.reporting.localization.Localization(__file__, 144, 36), decode_706158, *[str_706159], **kwargs_706160)
                
                # Obtaining the member 'lower' of a type (line 144)
                lower_706162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 36), decode_call_result_706161, 'lower')
                # Calling lower(args, kwargs) (line 144)
                lower_call_result_706164 = invoke(stypy.reporting.localization.Localization(__file__, 144, 36), lower_706162, *[], **kwargs_706163)
                
                # Assigning a type to the variable 'subr_name' (line 144)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 24), 'subr_name', lower_call_result_706164)
                
                # Assigning a Compare to a Name (line 145):
                
                # Getting the type of 'subr_name' (line 145)
                subr_name_706165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 39), 'subr_name')
                # Getting the type of 'subroutines' (line 145)
                subroutines_706166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 52), 'subroutines')
                # Applying the binary operator 'in' (line 145)
                result_contains_706167 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 39), 'in', subr_name_706165, subroutines_706166)
                
                # Assigning a type to the variable 'subr_wanted' (line 145)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 24), 'subr_wanted', result_contains_706167)

                if more_types_in_union_706152:
                    # Runtime conditional SSA for else branch (line 143)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_706151) or more_types_in_union_706152):
                
                # Assigning a Name to a Name (line 147):
                # Getting the type of 'True' (line 147)
                True_706168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 38), 'True')
                # Assigning a type to the variable 'subr_wanted' (line 147)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 24), 'subr_wanted', True_706168)

                if (may_be_706151 and more_types_in_union_706152):
                    # SSA join for if statement (line 143)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            
            # Evaluating a boolean operation
            # Getting the type of 'subr_wanted' (line 148)
            subr_wanted_706169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 23), 'subr_wanted')
            # Getting the type of 'need_split_next' (line 148)
            need_split_next_706170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 38), 'need_split_next')
            # Applying the binary operator 'or' (line 148)
            result_or_keyword_706171 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 23), 'or', subr_wanted_706169, need_split_next_706170)
            
            # Testing the type of an if condition (line 148)
            if_condition_706172 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 20), result_or_keyword_706171)
            # Assigning a type to the variable 'if_condition_706172' (line 148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'if_condition_706172', if_condition_706172)
            # SSA begins for if statement (line 148)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 149):
            # Getting the type of 'subr_wanted' (line 149)
            subr_wanted_706173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 42), 'subr_wanted')
            # Assigning a type to the variable 'need_split_next' (line 149)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), 'need_split_next', subr_wanted_706173)
            
            # Call to append(...): (line 150)
            # Processing the call arguments (line 150)
            # Getting the type of 'ix' (line 150)
            ix_706176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 36), 'ix', False)
            # Processing the call keyword arguments (line 150)
            kwargs_706177 = {}
            # Getting the type of 'subs' (line 150)
            subs_706174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 24), 'subs', False)
            # Obtaining the member 'append' of a type (line 150)
            append_706175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 24), subs_706174, 'append')
            # Calling append(args, kwargs) (line 150)
            append_call_result_706178 = invoke(stypy.reporting.localization.Localization(__file__, 150, 24), append_706175, *[ix_706176], **kwargs_706177)
            
            # SSA join for if statement (line 148)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 142)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            
            # Call to len(...): (line 153)
            # Processing the call arguments (line 153)
            # Getting the type of 'subs' (line 153)
            subs_706180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 19), 'subs', False)
            # Processing the call keyword arguments (line 153)
            kwargs_706181 = {}
            # Getting the type of 'len' (line 153)
            len_706179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 15), 'len', False)
            # Calling len(args, kwargs) (line 153)
            len_call_result_706182 = invoke(stypy.reporting.localization.Localization(__file__, 153, 15), len_706179, *[subs_706180], **kwargs_706181)
            
            int_706183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 28), 'int')
            # Applying the binary operator '<=' (line 153)
            result_le_706184 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 15), '<=', len_call_result_706182, int_706183)
            
            # Testing the type of an if condition (line 153)
            if_condition_706185 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 12), result_le_706184)
            # Assigning a type to the variable 'if_condition_706185' (line 153)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'if_condition_706185', if_condition_706185)
            # SSA begins for if statement (line 153)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining an instance of the builtin type 'list' (line 154)
            list_706186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 23), 'list')
            # Adding type elements to the builtin type 'list' instance (line 154)
            # Adding element type (line 154)
            # Getting the type of 'fname' (line 154)
            fname_706187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 24), 'fname')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 23), list_706186, fname_706187)
            
            # Assigning a type to the variable 'stypy_return_type' (line 154)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'stypy_return_type', list_706186)
            # SSA join for if statement (line 153)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a List to a Name (line 157):
            
            # Obtaining an instance of the builtin type 'list' (line 157)
            list_706188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 25), 'list')
            # Adding type elements to the builtin type 'list' instance (line 157)
            
            # Assigning a type to the variable 'new_fnames' (line 157)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'new_fnames', list_706188)
            
            # Assigning a Call to a Name (line 158):
            
            # Call to len(...): (line 158)
            # Processing the call arguments (line 158)
            # Getting the type of 'subs' (line 158)
            subs_706190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 28), 'subs', False)
            # Processing the call keyword arguments (line 158)
            kwargs_706191 = {}
            # Getting the type of 'len' (line 158)
            len_706189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 24), 'len', False)
            # Calling len(args, kwargs) (line 158)
            len_call_result_706192 = invoke(stypy.reporting.localization.Localization(__file__, 158, 24), len_706189, *[subs_706190], **kwargs_706191)
            
            # Assigning a type to the variable 'num_files' (line 158)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'num_files', len_call_result_706192)
            
            
            # Call to range(...): (line 159)
            # Processing the call arguments (line 159)
            # Getting the type of 'num_files' (line 159)
            num_files_706194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 31), 'num_files', False)
            # Processing the call keyword arguments (line 159)
            kwargs_706195 = {}
            # Getting the type of 'range' (line 159)
            range_706193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 25), 'range', False)
            # Calling range(args, kwargs) (line 159)
            range_call_result_706196 = invoke(stypy.reporting.localization.Localization(__file__, 159, 25), range_706193, *[num_files_706194], **kwargs_706195)
            
            # Testing the type of a for loop iterable (line 159)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 159, 12), range_call_result_706196)
            # Getting the type of the for loop variable (line 159)
            for_loop_var_706197 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 159, 12), range_call_result_706196)
            # Assigning a type to the variable 'nfile' (line 159)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'nfile', for_loop_var_706197)
            # SSA begins for a for statement (line 159)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a BinOp to a Name (line 160):
            
            # Obtaining the type of the subscript
            int_706198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 35), 'int')
            slice_706199 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 160, 28), None, int_706198, None)
            # Getting the type of 'fname' (line 160)
            fname_706200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 28), 'fname')
            # Obtaining the member '__getitem__' of a type (line 160)
            getitem___706201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 28), fname_706200, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 160)
            subscript_call_result_706202 = invoke(stypy.reporting.localization.Localization(__file__, 160, 28), getitem___706201, slice_706199)
            
            str_706203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 41), 'str', '_subr_')
            # Applying the binary operator '+' (line 160)
            result_add_706204 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 28), '+', subscript_call_result_706202, str_706203)
            
            
            # Call to str(...): (line 160)
            # Processing the call arguments (line 160)
            # Getting the type of 'nfile' (line 160)
            nfile_706206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 56), 'nfile', False)
            # Processing the call keyword arguments (line 160)
            kwargs_706207 = {}
            # Getting the type of 'str' (line 160)
            str_706205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 52), 'str', False)
            # Calling str(args, kwargs) (line 160)
            str_call_result_706208 = invoke(stypy.reporting.localization.Localization(__file__, 160, 52), str_706205, *[nfile_706206], **kwargs_706207)
            
            # Applying the binary operator '+' (line 160)
            result_add_706209 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 50), '+', result_add_706204, str_call_result_706208)
            
            str_706210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 65), 'str', '.f')
            # Applying the binary operator '+' (line 160)
            result_add_706211 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 63), '+', result_add_706209, str_706210)
            
            # Assigning a type to the variable 'new_fname' (line 160)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 16), 'new_fname', result_add_706211)
            
            # Call to append(...): (line 161)
            # Processing the call arguments (line 161)
            # Getting the type of 'new_fname' (line 161)
            new_fname_706214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 34), 'new_fname', False)
            # Processing the call keyword arguments (line 161)
            kwargs_706215 = {}
            # Getting the type of 'new_fnames' (line 161)
            new_fnames_706212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'new_fnames', False)
            # Obtaining the member 'append' of a type (line 161)
            append_706213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 16), new_fnames_706212, 'append')
            # Calling append(args, kwargs) (line 161)
            append_call_result_706216 = invoke(stypy.reporting.localization.Localization(__file__, 161, 16), append_706213, *[new_fname_706214], **kwargs_706215)
            
            
            
            
            # Call to newer(...): (line 162)
            # Processing the call arguments (line 162)
            # Getting the type of 'fname' (line 162)
            fname_706218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 29), 'fname', False)
            # Getting the type of 'new_fname' (line 162)
            new_fname_706219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 36), 'new_fname', False)
            # Processing the call keyword arguments (line 162)
            kwargs_706220 = {}
            # Getting the type of 'newer' (line 162)
            newer_706217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 23), 'newer', False)
            # Calling newer(args, kwargs) (line 162)
            newer_call_result_706221 = invoke(stypy.reporting.localization.Localization(__file__, 162, 23), newer_706217, *[fname_706218, new_fname_706219], **kwargs_706220)
            
            # Applying the 'not' unary operator (line 162)
            result_not__706222 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 19), 'not', newer_call_result_706221)
            
            # Testing the type of an if condition (line 162)
            if_condition_706223 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 16), result_not__706222)
            # Assigning a type to the variable 'if_condition_706223' (line 162)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'if_condition_706223', if_condition_706223)
            # SSA begins for if statement (line 162)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 162)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to open(...): (line 164)
            # Processing the call arguments (line 164)
            # Getting the type of 'new_fname' (line 164)
            new_fname_706225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 26), 'new_fname', False)
            str_706226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 37), 'str', 'wb')
            # Processing the call keyword arguments (line 164)
            kwargs_706227 = {}
            # Getting the type of 'open' (line 164)
            open_706224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 21), 'open', False)
            # Calling open(args, kwargs) (line 164)
            open_call_result_706228 = invoke(stypy.reporting.localization.Localization(__file__, 164, 21), open_706224, *[new_fname_706225, str_706226], **kwargs_706227)
            
            with_706229 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 164, 21), open_call_result_706228, 'with parameter', '__enter__', '__exit__')

            if with_706229:
                # Calling the __enter__ method to initiate a with section
                # Obtaining the member '__enter__' of a type (line 164)
                enter___706230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 21), open_call_result_706228, '__enter__')
                with_enter_706231 = invoke(stypy.reporting.localization.Localization(__file__, 164, 21), enter___706230)
                # Assigning a type to the variable 'fn' (line 164)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 21), 'fn', with_enter_706231)
                
                
                # Getting the type of 'nfile' (line 165)
                nfile_706232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 23), 'nfile')
                int_706233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 31), 'int')
                # Applying the binary operator '+' (line 165)
                result_add_706234 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 23), '+', nfile_706232, int_706233)
                
                # Getting the type of 'num_files' (line 165)
                num_files_706235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 36), 'num_files')
                # Applying the binary operator '==' (line 165)
                result_eq_706236 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 23), '==', result_add_706234, num_files_706235)
                
                # Testing the type of an if condition (line 165)
                if_condition_706237 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 165, 20), result_eq_706236)
                # Assigning a type to the variable 'if_condition_706237' (line 165)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 20), 'if_condition_706237', if_condition_706237)
                # SSA begins for if statement (line 165)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to writelines(...): (line 166)
                # Processing the call arguments (line 166)
                
                # Obtaining the type of the subscript
                
                # Obtaining the type of the subscript
                # Getting the type of 'nfile' (line 166)
                nfile_706240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 49), 'nfile', False)
                # Getting the type of 'subs' (line 166)
                subs_706241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 44), 'subs', False)
                # Obtaining the member '__getitem__' of a type (line 166)
                getitem___706242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 44), subs_706241, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 166)
                subscript_call_result_706243 = invoke(stypy.reporting.localization.Localization(__file__, 166, 44), getitem___706242, nfile_706240)
                
                slice_706244 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 166, 38), subscript_call_result_706243, None, None)
                # Getting the type of 'lines' (line 166)
                lines_706245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 38), 'lines', False)
                # Obtaining the member '__getitem__' of a type (line 166)
                getitem___706246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 38), lines_706245, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 166)
                subscript_call_result_706247 = invoke(stypy.reporting.localization.Localization(__file__, 166, 38), getitem___706246, slice_706244)
                
                # Processing the call keyword arguments (line 166)
                kwargs_706248 = {}
                # Getting the type of 'fn' (line 166)
                fn_706238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 24), 'fn', False)
                # Obtaining the member 'writelines' of a type (line 166)
                writelines_706239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 24), fn_706238, 'writelines')
                # Calling writelines(args, kwargs) (line 166)
                writelines_call_result_706249 = invoke(stypy.reporting.localization.Localization(__file__, 166, 24), writelines_706239, *[subscript_call_result_706247], **kwargs_706248)
                
                # SSA branch for the else part of an if statement (line 165)
                module_type_store.open_ssa_branch('else')
                
                # Call to writelines(...): (line 168)
                # Processing the call arguments (line 168)
                
                # Obtaining the type of the subscript
                
                # Obtaining the type of the subscript
                # Getting the type of 'nfile' (line 168)
                nfile_706252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 49), 'nfile', False)
                # Getting the type of 'subs' (line 168)
                subs_706253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 44), 'subs', False)
                # Obtaining the member '__getitem__' of a type (line 168)
                getitem___706254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 44), subs_706253, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 168)
                subscript_call_result_706255 = invoke(stypy.reporting.localization.Localization(__file__, 168, 44), getitem___706254, nfile_706252)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'nfile' (line 168)
                nfile_706256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 61), 'nfile', False)
                int_706257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 67), 'int')
                # Applying the binary operator '+' (line 168)
                result_add_706258 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 61), '+', nfile_706256, int_706257)
                
                # Getting the type of 'subs' (line 168)
                subs_706259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 56), 'subs', False)
                # Obtaining the member '__getitem__' of a type (line 168)
                getitem___706260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 56), subs_706259, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 168)
                subscript_call_result_706261 = invoke(stypy.reporting.localization.Localization(__file__, 168, 56), getitem___706260, result_add_706258)
                
                slice_706262 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 168, 38), subscript_call_result_706255, subscript_call_result_706261, None)
                # Getting the type of 'lines' (line 168)
                lines_706263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 38), 'lines', False)
                # Obtaining the member '__getitem__' of a type (line 168)
                getitem___706264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 38), lines_706263, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 168)
                subscript_call_result_706265 = invoke(stypy.reporting.localization.Localization(__file__, 168, 38), getitem___706264, slice_706262)
                
                # Processing the call keyword arguments (line 168)
                kwargs_706266 = {}
                # Getting the type of 'fn' (line 168)
                fn_706250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 24), 'fn', False)
                # Obtaining the member 'writelines' of a type (line 168)
                writelines_706251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 24), fn_706250, 'writelines')
                # Calling writelines(args, kwargs) (line 168)
                writelines_call_result_706267 = invoke(stypy.reporting.localization.Localization(__file__, 168, 24), writelines_706251, *[subscript_call_result_706265], **kwargs_706266)
                
                # SSA join for if statement (line 165)
                module_type_store = module_type_store.join_ssa_context()
                
                # Calling the __exit__ method to finish a with section
                # Obtaining the member '__exit__' of a type (line 164)
                exit___706268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 21), open_call_result_706228, '__exit__')
                with_exit_706269 = invoke(stypy.reporting.localization.Localization(__file__, 164, 21), exit___706268, None, None, None)

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 134)
            exit___706270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 13), open_call_result_706117, '__exit__')
            with_exit_706271 = invoke(stypy.reporting.localization.Localization(__file__, 134, 13), exit___706270, None, None, None)

        # Getting the type of 'new_fnames' (line 170)
        new_fnames_706272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 15), 'new_fnames')
        # Assigning a type to the variable 'stypy_return_type' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'stypy_return_type', new_fnames_706272)
        
        # ################# End of 'split_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'split_file' in the type store
        # Getting the type of 'stypy_return_type' (line 133)
        stypy_return_type_706273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_706273)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'split_file'
        return stypy_return_type_706273

    # Assigning a type to the variable 'split_file' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'split_file', split_file)
    
    # Assigning a Call to a Name (line 172):
    
    # Call to compile(...): (line 172)
    # Processing the call arguments (line 172)
    str_706276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 33), 'str', '_subr_[0-9]')
    # Processing the call keyword arguments (line 172)
    kwargs_706277 = {}
    # Getting the type of 're' (line 172)
    re_706274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 22), 're', False)
    # Obtaining the member 'compile' of a type (line 172)
    compile_706275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 22), re_706274, 'compile')
    # Calling compile(args, kwargs) (line 172)
    compile_call_result_706278 = invoke(stypy.reporting.localization.Localization(__file__, 172, 22), compile_706275, *[str_706276], **kwargs_706277)
    
    # Assigning a type to the variable 'exclude_pattern' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'exclude_pattern', compile_call_result_706278)
    
    # Assigning a ListComp to a Name (line 173):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to glob(...): (line 173)
    # Processing the call arguments (line 173)
    
    # Call to join(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'source_dir' (line 173)
    source_dir_706296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 55), 'source_dir', False)
    str_706297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 67), 'str', '*.f')
    # Processing the call keyword arguments (line 173)
    kwargs_706298 = {}
    # Getting the type of 'os' (line 173)
    os_706293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 42), 'os', False)
    # Obtaining the member 'path' of a type (line 173)
    path_706294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 42), os_706293, 'path')
    # Obtaining the member 'join' of a type (line 173)
    join_706295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 42), path_706294, 'join')
    # Calling join(args, kwargs) (line 173)
    join_call_result_706299 = invoke(stypy.reporting.localization.Localization(__file__, 173, 42), join_706295, *[source_dir_706296, str_706297], **kwargs_706298)
    
    # Processing the call keyword arguments (line 173)
    kwargs_706300 = {}
    # Getting the type of 'glob' (line 173)
    glob_706291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 32), 'glob', False)
    # Obtaining the member 'glob' of a type (line 173)
    glob_706292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 32), glob_706291, 'glob')
    # Calling glob(args, kwargs) (line 173)
    glob_call_result_706301 = invoke(stypy.reporting.localization.Localization(__file__, 173, 32), glob_706292, *[join_call_result_706299], **kwargs_706300)
    
    comprehension_706302 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 21), glob_call_result_706301)
    # Assigning a type to the variable 'f' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 21), 'f', comprehension_706302)
    
    
    # Call to search(...): (line 174)
    # Processing the call arguments (line 174)
    
    # Call to basename(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'f' (line 174)
    f_706285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 76), 'f', False)
    # Processing the call keyword arguments (line 174)
    kwargs_706286 = {}
    # Getting the type of 'os' (line 174)
    os_706282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 59), 'os', False)
    # Obtaining the member 'path' of a type (line 174)
    path_706283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 59), os_706282, 'path')
    # Obtaining the member 'basename' of a type (line 174)
    basename_706284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 59), path_706283, 'basename')
    # Calling basename(args, kwargs) (line 174)
    basename_call_result_706287 = invoke(stypy.reporting.localization.Localization(__file__, 174, 59), basename_706284, *[f_706285], **kwargs_706286)
    
    # Processing the call keyword arguments (line 174)
    kwargs_706288 = {}
    # Getting the type of 'exclude_pattern' (line 174)
    exclude_pattern_706280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 36), 'exclude_pattern', False)
    # Obtaining the member 'search' of a type (line 174)
    search_706281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 36), exclude_pattern_706280, 'search')
    # Calling search(args, kwargs) (line 174)
    search_call_result_706289 = invoke(stypy.reporting.localization.Localization(__file__, 174, 36), search_706281, *[basename_call_result_706287], **kwargs_706288)
    
    # Applying the 'not' unary operator (line 174)
    result_not__706290 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 32), 'not', search_call_result_706289)
    
    # Getting the type of 'f' (line 173)
    f_706279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 21), 'f')
    list_706303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 21), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 21), list_706303, f_706279)
    # Assigning a type to the variable 'source_fnames' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'source_fnames', list_706303)
    
    # Assigning a List to a Name (line 175):
    
    # Obtaining an instance of the builtin type 'list' (line 175)
    list_706304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 175)
    
    # Assigning a type to the variable 'fnames' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'fnames', list_706304)
    
    # Getting the type of 'source_fnames' (line 176)
    source_fnames_706305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 24), 'source_fnames')
    # Testing the type of a for loop iterable (line 176)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 176, 4), source_fnames_706305)
    # Getting the type of the for loop variable (line 176)
    for_loop_var_706306 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 176, 4), source_fnames_706305)
    # Assigning a type to the variable 'source_fname' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'source_fname', for_loop_var_706306)
    # SSA begins for a for statement (line 176)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 177):
    
    # Call to split_file(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'source_fname' (line 177)
    source_fname_706308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 35), 'source_fname', False)
    # Processing the call keyword arguments (line 177)
    kwargs_706309 = {}
    # Getting the type of 'split_file' (line 177)
    split_file_706307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 24), 'split_file', False)
    # Calling split_file(args, kwargs) (line 177)
    split_file_call_result_706310 = invoke(stypy.reporting.localization.Localization(__file__, 177, 24), split_file_706307, *[source_fname_706308], **kwargs_706309)
    
    # Assigning a type to the variable 'created_files' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'created_files', split_file_call_result_706310)
    
    # Type idiom detected: calculating its left and rigth part (line 178)
    # Getting the type of 'created_files' (line 178)
    created_files_706311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'created_files')
    # Getting the type of 'None' (line 178)
    None_706312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 32), 'None')
    
    (may_be_706313, more_types_in_union_706314) = may_not_be_none(created_files_706311, None_706312)

    if may_be_706313:

        if more_types_in_union_706314:
            # Runtime conditional SSA (line 178)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Getting the type of 'created_files' (line 179)
        created_files_706315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 25), 'created_files')
        # Testing the type of a for loop iterable (line 179)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 179, 12), created_files_706315)
        # Getting the type of the for loop variable (line 179)
        for_loop_var_706316 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 179, 12), created_files_706315)
        # Assigning a type to the variable 'cfile' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'cfile', for_loop_var_706316)
        # SSA begins for a for statement (line 179)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 180)
        # Processing the call arguments (line 180)
        
        # Call to basename(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'cfile' (line 180)
        cfile_706322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 47), 'cfile', False)
        # Processing the call keyword arguments (line 180)
        kwargs_706323 = {}
        # Getting the type of 'os' (line 180)
        os_706319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 30), 'os', False)
        # Obtaining the member 'path' of a type (line 180)
        path_706320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 30), os_706319, 'path')
        # Obtaining the member 'basename' of a type (line 180)
        basename_706321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 30), path_706320, 'basename')
        # Calling basename(args, kwargs) (line 180)
        basename_call_result_706324 = invoke(stypy.reporting.localization.Localization(__file__, 180, 30), basename_706321, *[cfile_706322], **kwargs_706323)
        
        # Processing the call keyword arguments (line 180)
        kwargs_706325 = {}
        # Getting the type of 'fnames' (line 180)
        fnames_706317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 16), 'fnames', False)
        # Obtaining the member 'append' of a type (line 180)
        append_706318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 16), fnames_706317, 'append')
        # Calling append(args, kwargs) (line 180)
        append_call_result_706326 = invoke(stypy.reporting.localization.Localization(__file__, 180, 16), append_706318, *[basename_call_result_706324], **kwargs_706325)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_706314:
            # SSA join for if statement (line 178)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'fnames' (line 182)
    fnames_706327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 11), 'fnames')
    # Assigning a type to the variable 'stypy_return_type' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'stypy_return_type', fnames_706327)
    
    # ################# End of 'split_fortran_files(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'split_fortran_files' in the type store
    # Getting the type of 'stypy_return_type' (line 104)
    stypy_return_type_706328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_706328)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'split_fortran_files'
    return stypy_return_type_706328

# Assigning a type to the variable 'split_fortran_files' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'split_fortran_files', split_fortran_files)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
