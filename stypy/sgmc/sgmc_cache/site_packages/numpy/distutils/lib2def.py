
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import re
4: import sys
5: import os
6: import subprocess
7: 
8: __doc__ = '''This module generates a DEF file from the symbols in
9: an MSVC-compiled DLL import library.  It correctly discriminates between
10: data and functions.  The data is collected from the output of the program
11: nm(1).
12: 
13: Usage:
14:     python lib2def.py [libname.lib] [output.def]
15: or
16:     python lib2def.py [libname.lib] > output.def
17: 
18: libname.lib defaults to python<py_ver>.lib and output.def defaults to stdout
19: 
20: Author: Robert Kern <kernr@mail.ncifcrf.gov>
21: Last Update: April 30, 1999
22: '''
23: 
24: __version__ = '0.1a'
25: 
26: py_ver = "%d%d" % tuple(sys.version_info[:2])
27: 
28: DEFAULT_NM = 'nm -Cs'
29: 
30: DEF_HEADER = '''LIBRARY         python%s.dll
31: ;CODE           PRELOAD MOVEABLE DISCARDABLE
32: ;DATA           PRELOAD SINGLE
33: 
34: EXPORTS
35: ''' % py_ver
36: # the header of the DEF file
37: 
38: FUNC_RE = re.compile(r"^(.*) in python%s\.dll" % py_ver, re.MULTILINE)
39: DATA_RE = re.compile(r"^_imp__(.*) in python%s\.dll" % py_ver, re.MULTILINE)
40: 
41: def parse_cmd():
42:     '''Parses the command-line arguments.
43: 
44: libfile, deffile = parse_cmd()'''
45:     if len(sys.argv) == 3:
46:         if sys.argv[1][-4:] == '.lib' and sys.argv[2][-4:] == '.def':
47:             libfile, deffile = sys.argv[1:]
48:         elif sys.argv[1][-4:] == '.def' and sys.argv[2][-4:] == '.lib':
49:             deffile, libfile = sys.argv[1:]
50:         else:
51:             print("I'm assuming that your first argument is the library")
52:             print("and the second is the DEF file.")
53:     elif len(sys.argv) == 2:
54:         if sys.argv[1][-4:] == '.def':
55:             deffile = sys.argv[1]
56:             libfile = 'python%s.lib' % py_ver
57:         elif sys.argv[1][-4:] == '.lib':
58:             deffile = None
59:             libfile = sys.argv[1]
60:     else:
61:         libfile = 'python%s.lib' % py_ver
62:         deffile = None
63:     return libfile, deffile
64: 
65: def getnm(nm_cmd = ['nm', '-Cs', 'python%s.lib' % py_ver]):
66:     '''Returns the output of nm_cmd via a pipe.
67: 
68: nm_output = getnam(nm_cmd = 'nm -Cs py_lib')'''
69:     f = subprocess.Popen(nm_cmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
70:     nm_output = f.stdout.read()
71:     f.stdout.close()
72:     return nm_output
73: 
74: def parse_nm(nm_output):
75:     '''Returns a tuple of lists: dlist for the list of data
76: symbols and flist for the list of function symbols.
77: 
78: dlist, flist = parse_nm(nm_output)'''
79:     data = DATA_RE.findall(nm_output)
80:     func = FUNC_RE.findall(nm_output)
81: 
82:     flist = []
83:     for sym in data:
84:         if sym in func and (sym[:2] == 'Py' or sym[:3] == '_Py' or sym[:4] == 'init'):
85:             flist.append(sym)
86: 
87:     dlist = []
88:     for sym in data:
89:         if sym not in flist and (sym[:2] == 'Py' or sym[:3] == '_Py'):
90:             dlist.append(sym)
91: 
92:     dlist.sort()
93:     flist.sort()
94:     return dlist, flist
95: 
96: def output_def(dlist, flist, header, file = sys.stdout):
97:     '''Outputs the final DEF file to a file defaulting to stdout.
98: 
99: output_def(dlist, flist, header, file = sys.stdout)'''
100:     for data_sym in dlist:
101:         header = header + '\t%s DATA\n' % data_sym
102:     header = header + '\n' # blank line
103:     for func_sym in flist:
104:         header = header + '\t%s\n' % func_sym
105:     file.write(header)
106: 
107: if __name__ == '__main__':
108:     libfile, deffile = parse_cmd()
109:     if deffile is None:
110:         deffile = sys.stdout
111:     else:
112:         deffile = open(deffile, 'w')
113:     nm_cmd = [str(DEFAULT_NM), str(libfile)]
114:     nm_output = getnm(nm_cmd)
115:     dlist, flist = parse_nm(nm_output)
116:     output_def(dlist, flist, DEF_HEADER, deffile)
117: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import re' statement (line 3)
import re

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import os' statement (line 5)
import os

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import subprocess' statement (line 6)
import subprocess

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'subprocess', subprocess, module_type_store)


# Assigning a Str to a Name (line 8):

# Assigning a Str to a Name (line 8):
str_36056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, (-1)), 'str', 'This module generates a DEF file from the symbols in\nan MSVC-compiled DLL import library.  It correctly discriminates between\ndata and functions.  The data is collected from the output of the program\nnm(1).\n\nUsage:\n    python lib2def.py [libname.lib] [output.def]\nor\n    python lib2def.py [libname.lib] > output.def\n\nlibname.lib defaults to python<py_ver>.lib and output.def defaults to stdout\n\nAuthor: Robert Kern <kernr@mail.ncifcrf.gov>\nLast Update: April 30, 1999\n')
# Assigning a type to the variable '__doc__' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__doc__', str_36056)

# Assigning a Str to a Name (line 24):

# Assigning a Str to a Name (line 24):
str_36057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 14), 'str', '0.1a')
# Assigning a type to the variable '__version__' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), '__version__', str_36057)

# Assigning a BinOp to a Name (line 26):

# Assigning a BinOp to a Name (line 26):
str_36058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 9), 'str', '%d%d')

# Call to tuple(...): (line 26)
# Processing the call arguments (line 26)

# Obtaining the type of the subscript
int_36060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 42), 'int')
slice_36061 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 26, 24), None, int_36060, None)
# Getting the type of 'sys' (line 26)
sys_36062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 24), 'sys', False)
# Obtaining the member 'version_info' of a type (line 26)
version_info_36063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 24), sys_36062, 'version_info')
# Obtaining the member '__getitem__' of a type (line 26)
getitem___36064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 24), version_info_36063, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 26)
subscript_call_result_36065 = invoke(stypy.reporting.localization.Localization(__file__, 26, 24), getitem___36064, slice_36061)

# Processing the call keyword arguments (line 26)
kwargs_36066 = {}
# Getting the type of 'tuple' (line 26)
tuple_36059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), 'tuple', False)
# Calling tuple(args, kwargs) (line 26)
tuple_call_result_36067 = invoke(stypy.reporting.localization.Localization(__file__, 26, 18), tuple_36059, *[subscript_call_result_36065], **kwargs_36066)

# Applying the binary operator '%' (line 26)
result_mod_36068 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 9), '%', str_36058, tuple_call_result_36067)

# Assigning a type to the variable 'py_ver' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'py_ver', result_mod_36068)

# Assigning a Str to a Name (line 28):

# Assigning a Str to a Name (line 28):
str_36069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 13), 'str', 'nm -Cs')
# Assigning a type to the variable 'DEFAULT_NM' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'DEFAULT_NM', str_36069)

# Assigning a BinOp to a Name (line 30):

# Assigning a BinOp to a Name (line 30):
str_36070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, (-1)), 'str', 'LIBRARY         python%s.dll\n;CODE           PRELOAD MOVEABLE DISCARDABLE\n;DATA           PRELOAD SINGLE\n\nEXPORTS\n')
# Getting the type of 'py_ver' (line 35)
py_ver_36071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 6), 'py_ver')
# Applying the binary operator '%' (line 35)
result_mod_36072 = python_operator(stypy.reporting.localization.Localization(__file__, 35, (-1)), '%', str_36070, py_ver_36071)

# Assigning a type to the variable 'DEF_HEADER' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'DEF_HEADER', result_mod_36072)

# Assigning a Call to a Name (line 38):

# Assigning a Call to a Name (line 38):

# Call to compile(...): (line 38)
# Processing the call arguments (line 38)
str_36075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 21), 'str', '^(.*) in python%s\\.dll')
# Getting the type of 'py_ver' (line 38)
py_ver_36076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 49), 'py_ver', False)
# Applying the binary operator '%' (line 38)
result_mod_36077 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 21), '%', str_36075, py_ver_36076)

# Getting the type of 're' (line 38)
re_36078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 57), 're', False)
# Obtaining the member 'MULTILINE' of a type (line 38)
MULTILINE_36079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 57), re_36078, 'MULTILINE')
# Processing the call keyword arguments (line 38)
kwargs_36080 = {}
# Getting the type of 're' (line 38)
re_36073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 10), 're', False)
# Obtaining the member 'compile' of a type (line 38)
compile_36074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 10), re_36073, 'compile')
# Calling compile(args, kwargs) (line 38)
compile_call_result_36081 = invoke(stypy.reporting.localization.Localization(__file__, 38, 10), compile_36074, *[result_mod_36077, MULTILINE_36079], **kwargs_36080)

# Assigning a type to the variable 'FUNC_RE' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'FUNC_RE', compile_call_result_36081)

# Assigning a Call to a Name (line 39):

# Assigning a Call to a Name (line 39):

# Call to compile(...): (line 39)
# Processing the call arguments (line 39)
str_36084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 21), 'str', '^_imp__(.*) in python%s\\.dll')
# Getting the type of 'py_ver' (line 39)
py_ver_36085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 55), 'py_ver', False)
# Applying the binary operator '%' (line 39)
result_mod_36086 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 21), '%', str_36084, py_ver_36085)

# Getting the type of 're' (line 39)
re_36087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 63), 're', False)
# Obtaining the member 'MULTILINE' of a type (line 39)
MULTILINE_36088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 63), re_36087, 'MULTILINE')
# Processing the call keyword arguments (line 39)
kwargs_36089 = {}
# Getting the type of 're' (line 39)
re_36082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 10), 're', False)
# Obtaining the member 'compile' of a type (line 39)
compile_36083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 10), re_36082, 'compile')
# Calling compile(args, kwargs) (line 39)
compile_call_result_36090 = invoke(stypy.reporting.localization.Localization(__file__, 39, 10), compile_36083, *[result_mod_36086, MULTILINE_36088], **kwargs_36089)

# Assigning a type to the variable 'DATA_RE' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'DATA_RE', compile_call_result_36090)

@norecursion
def parse_cmd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'parse_cmd'
    module_type_store = module_type_store.open_function_context('parse_cmd', 41, 0, False)
    
    # Passed parameters checking function
    parse_cmd.stypy_localization = localization
    parse_cmd.stypy_type_of_self = None
    parse_cmd.stypy_type_store = module_type_store
    parse_cmd.stypy_function_name = 'parse_cmd'
    parse_cmd.stypy_param_names_list = []
    parse_cmd.stypy_varargs_param_name = None
    parse_cmd.stypy_kwargs_param_name = None
    parse_cmd.stypy_call_defaults = defaults
    parse_cmd.stypy_call_varargs = varargs
    parse_cmd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'parse_cmd', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'parse_cmd', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'parse_cmd(...)' code ##################

    str_36091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, (-1)), 'str', 'Parses the command-line arguments.\n\nlibfile, deffile = parse_cmd()')
    
    
    
    # Call to len(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'sys' (line 45)
    sys_36093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 11), 'sys', False)
    # Obtaining the member 'argv' of a type (line 45)
    argv_36094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 11), sys_36093, 'argv')
    # Processing the call keyword arguments (line 45)
    kwargs_36095 = {}
    # Getting the type of 'len' (line 45)
    len_36092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 7), 'len', False)
    # Calling len(args, kwargs) (line 45)
    len_call_result_36096 = invoke(stypy.reporting.localization.Localization(__file__, 45, 7), len_36092, *[argv_36094], **kwargs_36095)
    
    int_36097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 24), 'int')
    # Applying the binary operator '==' (line 45)
    result_eq_36098 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 7), '==', len_call_result_36096, int_36097)
    
    # Testing the type of an if condition (line 45)
    if_condition_36099 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 4), result_eq_36098)
    # Assigning a type to the variable 'if_condition_36099' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'if_condition_36099', if_condition_36099)
    # SSA begins for if statement (line 45)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_36100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 23), 'int')
    slice_36101 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 46, 11), int_36100, None, None)
    
    # Obtaining the type of the subscript
    int_36102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 20), 'int')
    # Getting the type of 'sys' (line 46)
    sys_36103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'sys')
    # Obtaining the member 'argv' of a type (line 46)
    argv_36104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 11), sys_36103, 'argv')
    # Obtaining the member '__getitem__' of a type (line 46)
    getitem___36105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 11), argv_36104, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 46)
    subscript_call_result_36106 = invoke(stypy.reporting.localization.Localization(__file__, 46, 11), getitem___36105, int_36102)
    
    # Obtaining the member '__getitem__' of a type (line 46)
    getitem___36107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 11), subscript_call_result_36106, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 46)
    subscript_call_result_36108 = invoke(stypy.reporting.localization.Localization(__file__, 46, 11), getitem___36107, slice_36101)
    
    str_36109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 31), 'str', '.lib')
    # Applying the binary operator '==' (line 46)
    result_eq_36110 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 11), '==', subscript_call_result_36108, str_36109)
    
    
    
    # Obtaining the type of the subscript
    int_36111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 54), 'int')
    slice_36112 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 46, 42), int_36111, None, None)
    
    # Obtaining the type of the subscript
    int_36113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 51), 'int')
    # Getting the type of 'sys' (line 46)
    sys_36114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 42), 'sys')
    # Obtaining the member 'argv' of a type (line 46)
    argv_36115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 42), sys_36114, 'argv')
    # Obtaining the member '__getitem__' of a type (line 46)
    getitem___36116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 42), argv_36115, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 46)
    subscript_call_result_36117 = invoke(stypy.reporting.localization.Localization(__file__, 46, 42), getitem___36116, int_36113)
    
    # Obtaining the member '__getitem__' of a type (line 46)
    getitem___36118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 42), subscript_call_result_36117, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 46)
    subscript_call_result_36119 = invoke(stypy.reporting.localization.Localization(__file__, 46, 42), getitem___36118, slice_36112)
    
    str_36120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 62), 'str', '.def')
    # Applying the binary operator '==' (line 46)
    result_eq_36121 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 42), '==', subscript_call_result_36119, str_36120)
    
    # Applying the binary operator 'and' (line 46)
    result_and_keyword_36122 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 11), 'and', result_eq_36110, result_eq_36121)
    
    # Testing the type of an if condition (line 46)
    if_condition_36123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 8), result_and_keyword_36122)
    # Assigning a type to the variable 'if_condition_36123' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'if_condition_36123', if_condition_36123)
    # SSA begins for if statement (line 46)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Tuple (line 47):
    
    # Assigning a Subscript to a Name (line 47):
    
    # Obtaining the type of the subscript
    int_36124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 12), 'int')
    
    # Obtaining the type of the subscript
    int_36125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 40), 'int')
    slice_36126 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 47, 31), int_36125, None, None)
    # Getting the type of 'sys' (line 47)
    sys_36127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 31), 'sys')
    # Obtaining the member 'argv' of a type (line 47)
    argv_36128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 31), sys_36127, 'argv')
    # Obtaining the member '__getitem__' of a type (line 47)
    getitem___36129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 31), argv_36128, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 47)
    subscript_call_result_36130 = invoke(stypy.reporting.localization.Localization(__file__, 47, 31), getitem___36129, slice_36126)
    
    # Obtaining the member '__getitem__' of a type (line 47)
    getitem___36131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), subscript_call_result_36130, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 47)
    subscript_call_result_36132 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), getitem___36131, int_36124)
    
    # Assigning a type to the variable 'tuple_var_assignment_36046' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'tuple_var_assignment_36046', subscript_call_result_36132)
    
    # Assigning a Subscript to a Name (line 47):
    
    # Obtaining the type of the subscript
    int_36133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 12), 'int')
    
    # Obtaining the type of the subscript
    int_36134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 40), 'int')
    slice_36135 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 47, 31), int_36134, None, None)
    # Getting the type of 'sys' (line 47)
    sys_36136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 31), 'sys')
    # Obtaining the member 'argv' of a type (line 47)
    argv_36137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 31), sys_36136, 'argv')
    # Obtaining the member '__getitem__' of a type (line 47)
    getitem___36138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 31), argv_36137, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 47)
    subscript_call_result_36139 = invoke(stypy.reporting.localization.Localization(__file__, 47, 31), getitem___36138, slice_36135)
    
    # Obtaining the member '__getitem__' of a type (line 47)
    getitem___36140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), subscript_call_result_36139, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 47)
    subscript_call_result_36141 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), getitem___36140, int_36133)
    
    # Assigning a type to the variable 'tuple_var_assignment_36047' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'tuple_var_assignment_36047', subscript_call_result_36141)
    
    # Assigning a Name to a Name (line 47):
    # Getting the type of 'tuple_var_assignment_36046' (line 47)
    tuple_var_assignment_36046_36142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'tuple_var_assignment_36046')
    # Assigning a type to the variable 'libfile' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'libfile', tuple_var_assignment_36046_36142)
    
    # Assigning a Name to a Name (line 47):
    # Getting the type of 'tuple_var_assignment_36047' (line 47)
    tuple_var_assignment_36047_36143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'tuple_var_assignment_36047')
    # Assigning a type to the variable 'deffile' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 21), 'deffile', tuple_var_assignment_36047_36143)
    # SSA branch for the else part of an if statement (line 46)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_36144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 25), 'int')
    slice_36145 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 48, 13), int_36144, None, None)
    
    # Obtaining the type of the subscript
    int_36146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 22), 'int')
    # Getting the type of 'sys' (line 48)
    sys_36147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 13), 'sys')
    # Obtaining the member 'argv' of a type (line 48)
    argv_36148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 13), sys_36147, 'argv')
    # Obtaining the member '__getitem__' of a type (line 48)
    getitem___36149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 13), argv_36148, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 48)
    subscript_call_result_36150 = invoke(stypy.reporting.localization.Localization(__file__, 48, 13), getitem___36149, int_36146)
    
    # Obtaining the member '__getitem__' of a type (line 48)
    getitem___36151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 13), subscript_call_result_36150, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 48)
    subscript_call_result_36152 = invoke(stypy.reporting.localization.Localization(__file__, 48, 13), getitem___36151, slice_36145)
    
    str_36153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 33), 'str', '.def')
    # Applying the binary operator '==' (line 48)
    result_eq_36154 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 13), '==', subscript_call_result_36152, str_36153)
    
    
    
    # Obtaining the type of the subscript
    int_36155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 56), 'int')
    slice_36156 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 48, 44), int_36155, None, None)
    
    # Obtaining the type of the subscript
    int_36157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 53), 'int')
    # Getting the type of 'sys' (line 48)
    sys_36158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 44), 'sys')
    # Obtaining the member 'argv' of a type (line 48)
    argv_36159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 44), sys_36158, 'argv')
    # Obtaining the member '__getitem__' of a type (line 48)
    getitem___36160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 44), argv_36159, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 48)
    subscript_call_result_36161 = invoke(stypy.reporting.localization.Localization(__file__, 48, 44), getitem___36160, int_36157)
    
    # Obtaining the member '__getitem__' of a type (line 48)
    getitem___36162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 44), subscript_call_result_36161, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 48)
    subscript_call_result_36163 = invoke(stypy.reporting.localization.Localization(__file__, 48, 44), getitem___36162, slice_36156)
    
    str_36164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 64), 'str', '.lib')
    # Applying the binary operator '==' (line 48)
    result_eq_36165 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 44), '==', subscript_call_result_36163, str_36164)
    
    # Applying the binary operator 'and' (line 48)
    result_and_keyword_36166 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 13), 'and', result_eq_36154, result_eq_36165)
    
    # Testing the type of an if condition (line 48)
    if_condition_36167 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 13), result_and_keyword_36166)
    # Assigning a type to the variable 'if_condition_36167' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 13), 'if_condition_36167', if_condition_36167)
    # SSA begins for if statement (line 48)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Tuple (line 49):
    
    # Assigning a Subscript to a Name (line 49):
    
    # Obtaining the type of the subscript
    int_36168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 12), 'int')
    
    # Obtaining the type of the subscript
    int_36169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 40), 'int')
    slice_36170 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 49, 31), int_36169, None, None)
    # Getting the type of 'sys' (line 49)
    sys_36171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 31), 'sys')
    # Obtaining the member 'argv' of a type (line 49)
    argv_36172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 31), sys_36171, 'argv')
    # Obtaining the member '__getitem__' of a type (line 49)
    getitem___36173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 31), argv_36172, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 49)
    subscript_call_result_36174 = invoke(stypy.reporting.localization.Localization(__file__, 49, 31), getitem___36173, slice_36170)
    
    # Obtaining the member '__getitem__' of a type (line 49)
    getitem___36175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), subscript_call_result_36174, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 49)
    subscript_call_result_36176 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), getitem___36175, int_36168)
    
    # Assigning a type to the variable 'tuple_var_assignment_36048' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'tuple_var_assignment_36048', subscript_call_result_36176)
    
    # Assigning a Subscript to a Name (line 49):
    
    # Obtaining the type of the subscript
    int_36177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 12), 'int')
    
    # Obtaining the type of the subscript
    int_36178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 40), 'int')
    slice_36179 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 49, 31), int_36178, None, None)
    # Getting the type of 'sys' (line 49)
    sys_36180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 31), 'sys')
    # Obtaining the member 'argv' of a type (line 49)
    argv_36181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 31), sys_36180, 'argv')
    # Obtaining the member '__getitem__' of a type (line 49)
    getitem___36182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 31), argv_36181, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 49)
    subscript_call_result_36183 = invoke(stypy.reporting.localization.Localization(__file__, 49, 31), getitem___36182, slice_36179)
    
    # Obtaining the member '__getitem__' of a type (line 49)
    getitem___36184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), subscript_call_result_36183, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 49)
    subscript_call_result_36185 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), getitem___36184, int_36177)
    
    # Assigning a type to the variable 'tuple_var_assignment_36049' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'tuple_var_assignment_36049', subscript_call_result_36185)
    
    # Assigning a Name to a Name (line 49):
    # Getting the type of 'tuple_var_assignment_36048' (line 49)
    tuple_var_assignment_36048_36186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'tuple_var_assignment_36048')
    # Assigning a type to the variable 'deffile' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'deffile', tuple_var_assignment_36048_36186)
    
    # Assigning a Name to a Name (line 49):
    # Getting the type of 'tuple_var_assignment_36049' (line 49)
    tuple_var_assignment_36049_36187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'tuple_var_assignment_36049')
    # Assigning a type to the variable 'libfile' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 21), 'libfile', tuple_var_assignment_36049_36187)
    # SSA branch for the else part of an if statement (line 48)
    module_type_store.open_ssa_branch('else')
    
    # Call to print(...): (line 51)
    # Processing the call arguments (line 51)
    str_36189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 18), 'str', "I'm assuming that your first argument is the library")
    # Processing the call keyword arguments (line 51)
    kwargs_36190 = {}
    # Getting the type of 'print' (line 51)
    print_36188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'print', False)
    # Calling print(args, kwargs) (line 51)
    print_call_result_36191 = invoke(stypy.reporting.localization.Localization(__file__, 51, 12), print_36188, *[str_36189], **kwargs_36190)
    
    
    # Call to print(...): (line 52)
    # Processing the call arguments (line 52)
    str_36193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 18), 'str', 'and the second is the DEF file.')
    # Processing the call keyword arguments (line 52)
    kwargs_36194 = {}
    # Getting the type of 'print' (line 52)
    print_36192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'print', False)
    # Calling print(args, kwargs) (line 52)
    print_call_result_36195 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), print_36192, *[str_36193], **kwargs_36194)
    
    # SSA join for if statement (line 48)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 46)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 45)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'sys' (line 53)
    sys_36197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 13), 'sys', False)
    # Obtaining the member 'argv' of a type (line 53)
    argv_36198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 13), sys_36197, 'argv')
    # Processing the call keyword arguments (line 53)
    kwargs_36199 = {}
    # Getting the type of 'len' (line 53)
    len_36196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 9), 'len', False)
    # Calling len(args, kwargs) (line 53)
    len_call_result_36200 = invoke(stypy.reporting.localization.Localization(__file__, 53, 9), len_36196, *[argv_36198], **kwargs_36199)
    
    int_36201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 26), 'int')
    # Applying the binary operator '==' (line 53)
    result_eq_36202 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 9), '==', len_call_result_36200, int_36201)
    
    # Testing the type of an if condition (line 53)
    if_condition_36203 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 9), result_eq_36202)
    # Assigning a type to the variable 'if_condition_36203' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 9), 'if_condition_36203', if_condition_36203)
    # SSA begins for if statement (line 53)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    int_36204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 23), 'int')
    slice_36205 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 54, 11), int_36204, None, None)
    
    # Obtaining the type of the subscript
    int_36206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 20), 'int')
    # Getting the type of 'sys' (line 54)
    sys_36207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 11), 'sys')
    # Obtaining the member 'argv' of a type (line 54)
    argv_36208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 11), sys_36207, 'argv')
    # Obtaining the member '__getitem__' of a type (line 54)
    getitem___36209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 11), argv_36208, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 54)
    subscript_call_result_36210 = invoke(stypy.reporting.localization.Localization(__file__, 54, 11), getitem___36209, int_36206)
    
    # Obtaining the member '__getitem__' of a type (line 54)
    getitem___36211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 11), subscript_call_result_36210, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 54)
    subscript_call_result_36212 = invoke(stypy.reporting.localization.Localization(__file__, 54, 11), getitem___36211, slice_36205)
    
    str_36213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 31), 'str', '.def')
    # Applying the binary operator '==' (line 54)
    result_eq_36214 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 11), '==', subscript_call_result_36212, str_36213)
    
    # Testing the type of an if condition (line 54)
    if_condition_36215 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 8), result_eq_36214)
    # Assigning a type to the variable 'if_condition_36215' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'if_condition_36215', if_condition_36215)
    # SSA begins for if statement (line 54)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 55):
    
    # Assigning a Subscript to a Name (line 55):
    
    # Obtaining the type of the subscript
    int_36216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 31), 'int')
    # Getting the type of 'sys' (line 55)
    sys_36217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 22), 'sys')
    # Obtaining the member 'argv' of a type (line 55)
    argv_36218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 22), sys_36217, 'argv')
    # Obtaining the member '__getitem__' of a type (line 55)
    getitem___36219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 22), argv_36218, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 55)
    subscript_call_result_36220 = invoke(stypy.reporting.localization.Localization(__file__, 55, 22), getitem___36219, int_36216)
    
    # Assigning a type to the variable 'deffile' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'deffile', subscript_call_result_36220)
    
    # Assigning a BinOp to a Name (line 56):
    
    # Assigning a BinOp to a Name (line 56):
    str_36221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 22), 'str', 'python%s.lib')
    # Getting the type of 'py_ver' (line 56)
    py_ver_36222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 39), 'py_ver')
    # Applying the binary operator '%' (line 56)
    result_mod_36223 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 22), '%', str_36221, py_ver_36222)
    
    # Assigning a type to the variable 'libfile' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'libfile', result_mod_36223)
    # SSA branch for the else part of an if statement (line 54)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_36224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 25), 'int')
    slice_36225 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 57, 13), int_36224, None, None)
    
    # Obtaining the type of the subscript
    int_36226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 22), 'int')
    # Getting the type of 'sys' (line 57)
    sys_36227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 13), 'sys')
    # Obtaining the member 'argv' of a type (line 57)
    argv_36228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 13), sys_36227, 'argv')
    # Obtaining the member '__getitem__' of a type (line 57)
    getitem___36229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 13), argv_36228, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 57)
    subscript_call_result_36230 = invoke(stypy.reporting.localization.Localization(__file__, 57, 13), getitem___36229, int_36226)
    
    # Obtaining the member '__getitem__' of a type (line 57)
    getitem___36231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 13), subscript_call_result_36230, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 57)
    subscript_call_result_36232 = invoke(stypy.reporting.localization.Localization(__file__, 57, 13), getitem___36231, slice_36225)
    
    str_36233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 33), 'str', '.lib')
    # Applying the binary operator '==' (line 57)
    result_eq_36234 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 13), '==', subscript_call_result_36232, str_36233)
    
    # Testing the type of an if condition (line 57)
    if_condition_36235 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 13), result_eq_36234)
    # Assigning a type to the variable 'if_condition_36235' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 13), 'if_condition_36235', if_condition_36235)
    # SSA begins for if statement (line 57)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 58):
    
    # Assigning a Name to a Name (line 58):
    # Getting the type of 'None' (line 58)
    None_36236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 22), 'None')
    # Assigning a type to the variable 'deffile' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'deffile', None_36236)
    
    # Assigning a Subscript to a Name (line 59):
    
    # Assigning a Subscript to a Name (line 59):
    
    # Obtaining the type of the subscript
    int_36237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 31), 'int')
    # Getting the type of 'sys' (line 59)
    sys_36238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 22), 'sys')
    # Obtaining the member 'argv' of a type (line 59)
    argv_36239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 22), sys_36238, 'argv')
    # Obtaining the member '__getitem__' of a type (line 59)
    getitem___36240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 22), argv_36239, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 59)
    subscript_call_result_36241 = invoke(stypy.reporting.localization.Localization(__file__, 59, 22), getitem___36240, int_36237)
    
    # Assigning a type to the variable 'libfile' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'libfile', subscript_call_result_36241)
    # SSA join for if statement (line 57)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 54)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 53)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 61):
    
    # Assigning a BinOp to a Name (line 61):
    str_36242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 18), 'str', 'python%s.lib')
    # Getting the type of 'py_ver' (line 61)
    py_ver_36243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 35), 'py_ver')
    # Applying the binary operator '%' (line 61)
    result_mod_36244 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 18), '%', str_36242, py_ver_36243)
    
    # Assigning a type to the variable 'libfile' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'libfile', result_mod_36244)
    
    # Assigning a Name to a Name (line 62):
    
    # Assigning a Name to a Name (line 62):
    # Getting the type of 'None' (line 62)
    None_36245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 18), 'None')
    # Assigning a type to the variable 'deffile' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'deffile', None_36245)
    # SSA join for if statement (line 53)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 45)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 63)
    tuple_36246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 63)
    # Adding element type (line 63)
    # Getting the type of 'libfile' (line 63)
    libfile_36247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 11), 'libfile')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 11), tuple_36246, libfile_36247)
    # Adding element type (line 63)
    # Getting the type of 'deffile' (line 63)
    deffile_36248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'deffile')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 11), tuple_36246, deffile_36248)
    
    # Assigning a type to the variable 'stypy_return_type' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type', tuple_36246)
    
    # ################# End of 'parse_cmd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'parse_cmd' in the type store
    # Getting the type of 'stypy_return_type' (line 41)
    stypy_return_type_36249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_36249)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'parse_cmd'
    return stypy_return_type_36249

# Assigning a type to the variable 'parse_cmd' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'parse_cmd', parse_cmd)

@norecursion
def getnm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'list' (line 65)
    list_36250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 65)
    # Adding element type (line 65)
    str_36251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 20), 'str', 'nm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 19), list_36250, str_36251)
    # Adding element type (line 65)
    str_36252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 26), 'str', '-Cs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 19), list_36250, str_36252)
    # Adding element type (line 65)
    str_36253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 33), 'str', 'python%s.lib')
    # Getting the type of 'py_ver' (line 65)
    py_ver_36254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 50), 'py_ver')
    # Applying the binary operator '%' (line 65)
    result_mod_36255 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 33), '%', str_36253, py_ver_36254)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 19), list_36250, result_mod_36255)
    
    defaults = [list_36250]
    # Create a new context for function 'getnm'
    module_type_store = module_type_store.open_function_context('getnm', 65, 0, False)
    
    # Passed parameters checking function
    getnm.stypy_localization = localization
    getnm.stypy_type_of_self = None
    getnm.stypy_type_store = module_type_store
    getnm.stypy_function_name = 'getnm'
    getnm.stypy_param_names_list = ['nm_cmd']
    getnm.stypy_varargs_param_name = None
    getnm.stypy_kwargs_param_name = None
    getnm.stypy_call_defaults = defaults
    getnm.stypy_call_varargs = varargs
    getnm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getnm', ['nm_cmd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getnm', localization, ['nm_cmd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getnm(...)' code ##################

    str_36256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, (-1)), 'str', "Returns the output of nm_cmd via a pipe.\n\nnm_output = getnam(nm_cmd = 'nm -Cs py_lib')")
    
    # Assigning a Call to a Name (line 69):
    
    # Assigning a Call to a Name (line 69):
    
    # Call to Popen(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'nm_cmd' (line 69)
    nm_cmd_36259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 25), 'nm_cmd', False)
    # Processing the call keyword arguments (line 69)
    # Getting the type of 'True' (line 69)
    True_36260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 39), 'True', False)
    keyword_36261 = True_36260
    # Getting the type of 'subprocess' (line 69)
    subprocess_36262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 52), 'subprocess', False)
    # Obtaining the member 'PIPE' of a type (line 69)
    PIPE_36263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 52), subprocess_36262, 'PIPE')
    keyword_36264 = PIPE_36263
    # Getting the type of 'True' (line 69)
    True_36265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 88), 'True', False)
    keyword_36266 = True_36265
    kwargs_36267 = {'shell': keyword_36261, 'universal_newlines': keyword_36266, 'stdout': keyword_36264}
    # Getting the type of 'subprocess' (line 69)
    subprocess_36257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'subprocess', False)
    # Obtaining the member 'Popen' of a type (line 69)
    Popen_36258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), subprocess_36257, 'Popen')
    # Calling Popen(args, kwargs) (line 69)
    Popen_call_result_36268 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), Popen_36258, *[nm_cmd_36259], **kwargs_36267)
    
    # Assigning a type to the variable 'f' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'f', Popen_call_result_36268)
    
    # Assigning a Call to a Name (line 70):
    
    # Assigning a Call to a Name (line 70):
    
    # Call to read(...): (line 70)
    # Processing the call keyword arguments (line 70)
    kwargs_36272 = {}
    # Getting the type of 'f' (line 70)
    f_36269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'f', False)
    # Obtaining the member 'stdout' of a type (line 70)
    stdout_36270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), f_36269, 'stdout')
    # Obtaining the member 'read' of a type (line 70)
    read_36271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), stdout_36270, 'read')
    # Calling read(args, kwargs) (line 70)
    read_call_result_36273 = invoke(stypy.reporting.localization.Localization(__file__, 70, 16), read_36271, *[], **kwargs_36272)
    
    # Assigning a type to the variable 'nm_output' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'nm_output', read_call_result_36273)
    
    # Call to close(...): (line 71)
    # Processing the call keyword arguments (line 71)
    kwargs_36277 = {}
    # Getting the type of 'f' (line 71)
    f_36274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'f', False)
    # Obtaining the member 'stdout' of a type (line 71)
    stdout_36275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 4), f_36274, 'stdout')
    # Obtaining the member 'close' of a type (line 71)
    close_36276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 4), stdout_36275, 'close')
    # Calling close(args, kwargs) (line 71)
    close_call_result_36278 = invoke(stypy.reporting.localization.Localization(__file__, 71, 4), close_36276, *[], **kwargs_36277)
    
    # Getting the type of 'nm_output' (line 72)
    nm_output_36279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 11), 'nm_output')
    # Assigning a type to the variable 'stypy_return_type' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stypy_return_type', nm_output_36279)
    
    # ################# End of 'getnm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getnm' in the type store
    # Getting the type of 'stypy_return_type' (line 65)
    stypy_return_type_36280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_36280)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getnm'
    return stypy_return_type_36280

# Assigning a type to the variable 'getnm' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'getnm', getnm)

@norecursion
def parse_nm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'parse_nm'
    module_type_store = module_type_store.open_function_context('parse_nm', 74, 0, False)
    
    # Passed parameters checking function
    parse_nm.stypy_localization = localization
    parse_nm.stypy_type_of_self = None
    parse_nm.stypy_type_store = module_type_store
    parse_nm.stypy_function_name = 'parse_nm'
    parse_nm.stypy_param_names_list = ['nm_output']
    parse_nm.stypy_varargs_param_name = None
    parse_nm.stypy_kwargs_param_name = None
    parse_nm.stypy_call_defaults = defaults
    parse_nm.stypy_call_varargs = varargs
    parse_nm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'parse_nm', ['nm_output'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'parse_nm', localization, ['nm_output'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'parse_nm(...)' code ##################

    str_36281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, (-1)), 'str', 'Returns a tuple of lists: dlist for the list of data\nsymbols and flist for the list of function symbols.\n\ndlist, flist = parse_nm(nm_output)')
    
    # Assigning a Call to a Name (line 79):
    
    # Assigning a Call to a Name (line 79):
    
    # Call to findall(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'nm_output' (line 79)
    nm_output_36284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 27), 'nm_output', False)
    # Processing the call keyword arguments (line 79)
    kwargs_36285 = {}
    # Getting the type of 'DATA_RE' (line 79)
    DATA_RE_36282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), 'DATA_RE', False)
    # Obtaining the member 'findall' of a type (line 79)
    findall_36283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 11), DATA_RE_36282, 'findall')
    # Calling findall(args, kwargs) (line 79)
    findall_call_result_36286 = invoke(stypy.reporting.localization.Localization(__file__, 79, 11), findall_36283, *[nm_output_36284], **kwargs_36285)
    
    # Assigning a type to the variable 'data' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'data', findall_call_result_36286)
    
    # Assigning a Call to a Name (line 80):
    
    # Assigning a Call to a Name (line 80):
    
    # Call to findall(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'nm_output' (line 80)
    nm_output_36289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 27), 'nm_output', False)
    # Processing the call keyword arguments (line 80)
    kwargs_36290 = {}
    # Getting the type of 'FUNC_RE' (line 80)
    FUNC_RE_36287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'FUNC_RE', False)
    # Obtaining the member 'findall' of a type (line 80)
    findall_36288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 11), FUNC_RE_36287, 'findall')
    # Calling findall(args, kwargs) (line 80)
    findall_call_result_36291 = invoke(stypy.reporting.localization.Localization(__file__, 80, 11), findall_36288, *[nm_output_36289], **kwargs_36290)
    
    # Assigning a type to the variable 'func' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'func', findall_call_result_36291)
    
    # Assigning a List to a Name (line 82):
    
    # Assigning a List to a Name (line 82):
    
    # Obtaining an instance of the builtin type 'list' (line 82)
    list_36292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 82)
    
    # Assigning a type to the variable 'flist' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'flist', list_36292)
    
    # Getting the type of 'data' (line 83)
    data_36293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 15), 'data')
    # Testing the type of a for loop iterable (line 83)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 83, 4), data_36293)
    # Getting the type of the for loop variable (line 83)
    for_loop_var_36294 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 83, 4), data_36293)
    # Assigning a type to the variable 'sym' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'sym', for_loop_var_36294)
    # SSA begins for a for statement (line 83)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'sym' (line 84)
    sym_36295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 11), 'sym')
    # Getting the type of 'func' (line 84)
    func_36296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 18), 'func')
    # Applying the binary operator 'in' (line 84)
    result_contains_36297 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 11), 'in', sym_36295, func_36296)
    
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_36298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 33), 'int')
    slice_36299 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 84, 28), None, int_36298, None)
    # Getting the type of 'sym' (line 84)
    sym_36300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 28), 'sym')
    # Obtaining the member '__getitem__' of a type (line 84)
    getitem___36301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 28), sym_36300, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 84)
    subscript_call_result_36302 = invoke(stypy.reporting.localization.Localization(__file__, 84, 28), getitem___36301, slice_36299)
    
    str_36303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 39), 'str', 'Py')
    # Applying the binary operator '==' (line 84)
    result_eq_36304 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 28), '==', subscript_call_result_36302, str_36303)
    
    
    
    # Obtaining the type of the subscript
    int_36305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 52), 'int')
    slice_36306 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 84, 47), None, int_36305, None)
    # Getting the type of 'sym' (line 84)
    sym_36307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 47), 'sym')
    # Obtaining the member '__getitem__' of a type (line 84)
    getitem___36308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 47), sym_36307, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 84)
    subscript_call_result_36309 = invoke(stypy.reporting.localization.Localization(__file__, 84, 47), getitem___36308, slice_36306)
    
    str_36310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 58), 'str', '_Py')
    # Applying the binary operator '==' (line 84)
    result_eq_36311 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 47), '==', subscript_call_result_36309, str_36310)
    
    # Applying the binary operator 'or' (line 84)
    result_or_keyword_36312 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 28), 'or', result_eq_36304, result_eq_36311)
    
    
    # Obtaining the type of the subscript
    int_36313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 72), 'int')
    slice_36314 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 84, 67), None, int_36313, None)
    # Getting the type of 'sym' (line 84)
    sym_36315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 67), 'sym')
    # Obtaining the member '__getitem__' of a type (line 84)
    getitem___36316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 67), sym_36315, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 84)
    subscript_call_result_36317 = invoke(stypy.reporting.localization.Localization(__file__, 84, 67), getitem___36316, slice_36314)
    
    str_36318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 78), 'str', 'init')
    # Applying the binary operator '==' (line 84)
    result_eq_36319 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 67), '==', subscript_call_result_36317, str_36318)
    
    # Applying the binary operator 'or' (line 84)
    result_or_keyword_36320 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 28), 'or', result_or_keyword_36312, result_eq_36319)
    
    # Applying the binary operator 'and' (line 84)
    result_and_keyword_36321 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 11), 'and', result_contains_36297, result_or_keyword_36320)
    
    # Testing the type of an if condition (line 84)
    if_condition_36322 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 8), result_and_keyword_36321)
    # Assigning a type to the variable 'if_condition_36322' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'if_condition_36322', if_condition_36322)
    # SSA begins for if statement (line 84)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'sym' (line 85)
    sym_36325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 25), 'sym', False)
    # Processing the call keyword arguments (line 85)
    kwargs_36326 = {}
    # Getting the type of 'flist' (line 85)
    flist_36323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'flist', False)
    # Obtaining the member 'append' of a type (line 85)
    append_36324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 12), flist_36323, 'append')
    # Calling append(args, kwargs) (line 85)
    append_call_result_36327 = invoke(stypy.reporting.localization.Localization(__file__, 85, 12), append_36324, *[sym_36325], **kwargs_36326)
    
    # SSA join for if statement (line 84)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 87):
    
    # Assigning a List to a Name (line 87):
    
    # Obtaining an instance of the builtin type 'list' (line 87)
    list_36328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 87)
    
    # Assigning a type to the variable 'dlist' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'dlist', list_36328)
    
    # Getting the type of 'data' (line 88)
    data_36329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'data')
    # Testing the type of a for loop iterable (line 88)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 88, 4), data_36329)
    # Getting the type of the for loop variable (line 88)
    for_loop_var_36330 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 88, 4), data_36329)
    # Assigning a type to the variable 'sym' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'sym', for_loop_var_36330)
    # SSA begins for a for statement (line 88)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'sym' (line 89)
    sym_36331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'sym')
    # Getting the type of 'flist' (line 89)
    flist_36332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 22), 'flist')
    # Applying the binary operator 'notin' (line 89)
    result_contains_36333 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 11), 'notin', sym_36331, flist_36332)
    
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_36334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 38), 'int')
    slice_36335 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 89, 33), None, int_36334, None)
    # Getting the type of 'sym' (line 89)
    sym_36336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 33), 'sym')
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___36337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 33), sym_36336, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_36338 = invoke(stypy.reporting.localization.Localization(__file__, 89, 33), getitem___36337, slice_36335)
    
    str_36339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 44), 'str', 'Py')
    # Applying the binary operator '==' (line 89)
    result_eq_36340 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 33), '==', subscript_call_result_36338, str_36339)
    
    
    
    # Obtaining the type of the subscript
    int_36341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 57), 'int')
    slice_36342 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 89, 52), None, int_36341, None)
    # Getting the type of 'sym' (line 89)
    sym_36343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 52), 'sym')
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___36344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 52), sym_36343, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_36345 = invoke(stypy.reporting.localization.Localization(__file__, 89, 52), getitem___36344, slice_36342)
    
    str_36346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 63), 'str', '_Py')
    # Applying the binary operator '==' (line 89)
    result_eq_36347 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 52), '==', subscript_call_result_36345, str_36346)
    
    # Applying the binary operator 'or' (line 89)
    result_or_keyword_36348 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 33), 'or', result_eq_36340, result_eq_36347)
    
    # Applying the binary operator 'and' (line 89)
    result_and_keyword_36349 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 11), 'and', result_contains_36333, result_or_keyword_36348)
    
    # Testing the type of an if condition (line 89)
    if_condition_36350 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 8), result_and_keyword_36349)
    # Assigning a type to the variable 'if_condition_36350' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'if_condition_36350', if_condition_36350)
    # SSA begins for if statement (line 89)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'sym' (line 90)
    sym_36353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'sym', False)
    # Processing the call keyword arguments (line 90)
    kwargs_36354 = {}
    # Getting the type of 'dlist' (line 90)
    dlist_36351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'dlist', False)
    # Obtaining the member 'append' of a type (line 90)
    append_36352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), dlist_36351, 'append')
    # Calling append(args, kwargs) (line 90)
    append_call_result_36355 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), append_36352, *[sym_36353], **kwargs_36354)
    
    # SSA join for if statement (line 89)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to sort(...): (line 92)
    # Processing the call keyword arguments (line 92)
    kwargs_36358 = {}
    # Getting the type of 'dlist' (line 92)
    dlist_36356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'dlist', False)
    # Obtaining the member 'sort' of a type (line 92)
    sort_36357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 4), dlist_36356, 'sort')
    # Calling sort(args, kwargs) (line 92)
    sort_call_result_36359 = invoke(stypy.reporting.localization.Localization(__file__, 92, 4), sort_36357, *[], **kwargs_36358)
    
    
    # Call to sort(...): (line 93)
    # Processing the call keyword arguments (line 93)
    kwargs_36362 = {}
    # Getting the type of 'flist' (line 93)
    flist_36360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'flist', False)
    # Obtaining the member 'sort' of a type (line 93)
    sort_36361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 4), flist_36360, 'sort')
    # Calling sort(args, kwargs) (line 93)
    sort_call_result_36363 = invoke(stypy.reporting.localization.Localization(__file__, 93, 4), sort_36361, *[], **kwargs_36362)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 94)
    tuple_36364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 94)
    # Adding element type (line 94)
    # Getting the type of 'dlist' (line 94)
    dlist_36365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 11), 'dlist')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 11), tuple_36364, dlist_36365)
    # Adding element type (line 94)
    # Getting the type of 'flist' (line 94)
    flist_36366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 18), 'flist')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 11), tuple_36364, flist_36366)
    
    # Assigning a type to the variable 'stypy_return_type' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'stypy_return_type', tuple_36364)
    
    # ################# End of 'parse_nm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'parse_nm' in the type store
    # Getting the type of 'stypy_return_type' (line 74)
    stypy_return_type_36367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_36367)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'parse_nm'
    return stypy_return_type_36367

# Assigning a type to the variable 'parse_nm' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'parse_nm', parse_nm)

@norecursion
def output_def(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'sys' (line 96)
    sys_36368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 44), 'sys')
    # Obtaining the member 'stdout' of a type (line 96)
    stdout_36369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 44), sys_36368, 'stdout')
    defaults = [stdout_36369]
    # Create a new context for function 'output_def'
    module_type_store = module_type_store.open_function_context('output_def', 96, 0, False)
    
    # Passed parameters checking function
    output_def.stypy_localization = localization
    output_def.stypy_type_of_self = None
    output_def.stypy_type_store = module_type_store
    output_def.stypy_function_name = 'output_def'
    output_def.stypy_param_names_list = ['dlist', 'flist', 'header', 'file']
    output_def.stypy_varargs_param_name = None
    output_def.stypy_kwargs_param_name = None
    output_def.stypy_call_defaults = defaults
    output_def.stypy_call_varargs = varargs
    output_def.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'output_def', ['dlist', 'flist', 'header', 'file'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'output_def', localization, ['dlist', 'flist', 'header', 'file'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'output_def(...)' code ##################

    str_36370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, (-1)), 'str', 'Outputs the final DEF file to a file defaulting to stdout.\n\noutput_def(dlist, flist, header, file = sys.stdout)')
    
    # Getting the type of 'dlist' (line 100)
    dlist_36371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), 'dlist')
    # Testing the type of a for loop iterable (line 100)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 100, 4), dlist_36371)
    # Getting the type of the for loop variable (line 100)
    for_loop_var_36372 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 100, 4), dlist_36371)
    # Assigning a type to the variable 'data_sym' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'data_sym', for_loop_var_36372)
    # SSA begins for a for statement (line 100)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 101):
    
    # Assigning a BinOp to a Name (line 101):
    # Getting the type of 'header' (line 101)
    header_36373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 'header')
    str_36374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 26), 'str', '\t%s DATA\n')
    # Getting the type of 'data_sym' (line 101)
    data_sym_36375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 42), 'data_sym')
    # Applying the binary operator '%' (line 101)
    result_mod_36376 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 26), '%', str_36374, data_sym_36375)
    
    # Applying the binary operator '+' (line 101)
    result_add_36377 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 17), '+', header_36373, result_mod_36376)
    
    # Assigning a type to the variable 'header' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'header', result_add_36377)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 102):
    
    # Assigning a BinOp to a Name (line 102):
    # Getting the type of 'header' (line 102)
    header_36378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 13), 'header')
    str_36379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 22), 'str', '\n')
    # Applying the binary operator '+' (line 102)
    result_add_36380 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 13), '+', header_36378, str_36379)
    
    # Assigning a type to the variable 'header' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'header', result_add_36380)
    
    # Getting the type of 'flist' (line 103)
    flist_36381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'flist')
    # Testing the type of a for loop iterable (line 103)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 103, 4), flist_36381)
    # Getting the type of the for loop variable (line 103)
    for_loop_var_36382 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 103, 4), flist_36381)
    # Assigning a type to the variable 'func_sym' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'func_sym', for_loop_var_36382)
    # SSA begins for a for statement (line 103)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 104):
    
    # Assigning a BinOp to a Name (line 104):
    # Getting the type of 'header' (line 104)
    header_36383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 17), 'header')
    str_36384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 26), 'str', '\t%s\n')
    # Getting the type of 'func_sym' (line 104)
    func_sym_36385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 37), 'func_sym')
    # Applying the binary operator '%' (line 104)
    result_mod_36386 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 26), '%', str_36384, func_sym_36385)
    
    # Applying the binary operator '+' (line 104)
    result_add_36387 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 17), '+', header_36383, result_mod_36386)
    
    # Assigning a type to the variable 'header' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'header', result_add_36387)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to write(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'header' (line 105)
    header_36390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), 'header', False)
    # Processing the call keyword arguments (line 105)
    kwargs_36391 = {}
    # Getting the type of 'file' (line 105)
    file_36388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'file', False)
    # Obtaining the member 'write' of a type (line 105)
    write_36389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 4), file_36388, 'write')
    # Calling write(args, kwargs) (line 105)
    write_call_result_36392 = invoke(stypy.reporting.localization.Localization(__file__, 105, 4), write_36389, *[header_36390], **kwargs_36391)
    
    
    # ################# End of 'output_def(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'output_def' in the type store
    # Getting the type of 'stypy_return_type' (line 96)
    stypy_return_type_36393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_36393)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'output_def'
    return stypy_return_type_36393

# Assigning a type to the variable 'output_def' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'output_def', output_def)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Tuple (line 108):
    
    # Assigning a Call to a Name:
    
    # Call to parse_cmd(...): (line 108)
    # Processing the call keyword arguments (line 108)
    kwargs_36395 = {}
    # Getting the type of 'parse_cmd' (line 108)
    parse_cmd_36394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 23), 'parse_cmd', False)
    # Calling parse_cmd(args, kwargs) (line 108)
    parse_cmd_call_result_36396 = invoke(stypy.reporting.localization.Localization(__file__, 108, 23), parse_cmd_36394, *[], **kwargs_36395)
    
    # Assigning a type to the variable 'call_assignment_36050' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'call_assignment_36050', parse_cmd_call_result_36396)
    
    # Assigning a Call to a Name (line 108):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_36399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 4), 'int')
    # Processing the call keyword arguments
    kwargs_36400 = {}
    # Getting the type of 'call_assignment_36050' (line 108)
    call_assignment_36050_36397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'call_assignment_36050', False)
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___36398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 4), call_assignment_36050_36397, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_36401 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___36398, *[int_36399], **kwargs_36400)
    
    # Assigning a type to the variable 'call_assignment_36051' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'call_assignment_36051', getitem___call_result_36401)
    
    # Assigning a Name to a Name (line 108):
    # Getting the type of 'call_assignment_36051' (line 108)
    call_assignment_36051_36402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'call_assignment_36051')
    # Assigning a type to the variable 'libfile' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'libfile', call_assignment_36051_36402)
    
    # Assigning a Call to a Name (line 108):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_36405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 4), 'int')
    # Processing the call keyword arguments
    kwargs_36406 = {}
    # Getting the type of 'call_assignment_36050' (line 108)
    call_assignment_36050_36403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'call_assignment_36050', False)
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___36404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 4), call_assignment_36050_36403, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_36407 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___36404, *[int_36405], **kwargs_36406)
    
    # Assigning a type to the variable 'call_assignment_36052' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'call_assignment_36052', getitem___call_result_36407)
    
    # Assigning a Name to a Name (line 108):
    # Getting the type of 'call_assignment_36052' (line 108)
    call_assignment_36052_36408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'call_assignment_36052')
    # Assigning a type to the variable 'deffile' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'deffile', call_assignment_36052_36408)
    
    # Type idiom detected: calculating its left and rigth part (line 109)
    # Getting the type of 'deffile' (line 109)
    deffile_36409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 7), 'deffile')
    # Getting the type of 'None' (line 109)
    None_36410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 18), 'None')
    
    (may_be_36411, more_types_in_union_36412) = may_be_none(deffile_36409, None_36410)

    if may_be_36411:

        if more_types_in_union_36412:
            # Runtime conditional SSA (line 109)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 110):
        
        # Assigning a Attribute to a Name (line 110):
        # Getting the type of 'sys' (line 110)
        sys_36413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 18), 'sys')
        # Obtaining the member 'stdout' of a type (line 110)
        stdout_36414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 18), sys_36413, 'stdout')
        # Assigning a type to the variable 'deffile' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'deffile', stdout_36414)

        if more_types_in_union_36412:
            # Runtime conditional SSA for else branch (line 109)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_36411) or more_types_in_union_36412):
        
        # Assigning a Call to a Name (line 112):
        
        # Assigning a Call to a Name (line 112):
        
        # Call to open(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'deffile' (line 112)
        deffile_36416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 23), 'deffile', False)
        str_36417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 32), 'str', 'w')
        # Processing the call keyword arguments (line 112)
        kwargs_36418 = {}
        # Getting the type of 'open' (line 112)
        open_36415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 18), 'open', False)
        # Calling open(args, kwargs) (line 112)
        open_call_result_36419 = invoke(stypy.reporting.localization.Localization(__file__, 112, 18), open_36415, *[deffile_36416, str_36417], **kwargs_36418)
        
        # Assigning a type to the variable 'deffile' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'deffile', open_call_result_36419)

        if (may_be_36411 and more_types_in_union_36412):
            # SSA join for if statement (line 109)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a List to a Name (line 113):
    
    # Assigning a List to a Name (line 113):
    
    # Obtaining an instance of the builtin type 'list' (line 113)
    list_36420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 113)
    # Adding element type (line 113)
    
    # Call to str(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'DEFAULT_NM' (line 113)
    DEFAULT_NM_36422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 18), 'DEFAULT_NM', False)
    # Processing the call keyword arguments (line 113)
    kwargs_36423 = {}
    # Getting the type of 'str' (line 113)
    str_36421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 14), 'str', False)
    # Calling str(args, kwargs) (line 113)
    str_call_result_36424 = invoke(stypy.reporting.localization.Localization(__file__, 113, 14), str_36421, *[DEFAULT_NM_36422], **kwargs_36423)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 13), list_36420, str_call_result_36424)
    # Adding element type (line 113)
    
    # Call to str(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'libfile' (line 113)
    libfile_36426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 35), 'libfile', False)
    # Processing the call keyword arguments (line 113)
    kwargs_36427 = {}
    # Getting the type of 'str' (line 113)
    str_36425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 31), 'str', False)
    # Calling str(args, kwargs) (line 113)
    str_call_result_36428 = invoke(stypy.reporting.localization.Localization(__file__, 113, 31), str_36425, *[libfile_36426], **kwargs_36427)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 13), list_36420, str_call_result_36428)
    
    # Assigning a type to the variable 'nm_cmd' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'nm_cmd', list_36420)
    
    # Assigning a Call to a Name (line 114):
    
    # Assigning a Call to a Name (line 114):
    
    # Call to getnm(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'nm_cmd' (line 114)
    nm_cmd_36430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 22), 'nm_cmd', False)
    # Processing the call keyword arguments (line 114)
    kwargs_36431 = {}
    # Getting the type of 'getnm' (line 114)
    getnm_36429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'getnm', False)
    # Calling getnm(args, kwargs) (line 114)
    getnm_call_result_36432 = invoke(stypy.reporting.localization.Localization(__file__, 114, 16), getnm_36429, *[nm_cmd_36430], **kwargs_36431)
    
    # Assigning a type to the variable 'nm_output' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'nm_output', getnm_call_result_36432)
    
    # Assigning a Call to a Tuple (line 115):
    
    # Assigning a Call to a Name:
    
    # Call to parse_nm(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'nm_output' (line 115)
    nm_output_36434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 28), 'nm_output', False)
    # Processing the call keyword arguments (line 115)
    kwargs_36435 = {}
    # Getting the type of 'parse_nm' (line 115)
    parse_nm_36433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), 'parse_nm', False)
    # Calling parse_nm(args, kwargs) (line 115)
    parse_nm_call_result_36436 = invoke(stypy.reporting.localization.Localization(__file__, 115, 19), parse_nm_36433, *[nm_output_36434], **kwargs_36435)
    
    # Assigning a type to the variable 'call_assignment_36053' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'call_assignment_36053', parse_nm_call_result_36436)
    
    # Assigning a Call to a Name (line 115):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_36439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 4), 'int')
    # Processing the call keyword arguments
    kwargs_36440 = {}
    # Getting the type of 'call_assignment_36053' (line 115)
    call_assignment_36053_36437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'call_assignment_36053', False)
    # Obtaining the member '__getitem__' of a type (line 115)
    getitem___36438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 4), call_assignment_36053_36437, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_36441 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___36438, *[int_36439], **kwargs_36440)
    
    # Assigning a type to the variable 'call_assignment_36054' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'call_assignment_36054', getitem___call_result_36441)
    
    # Assigning a Name to a Name (line 115):
    # Getting the type of 'call_assignment_36054' (line 115)
    call_assignment_36054_36442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'call_assignment_36054')
    # Assigning a type to the variable 'dlist' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'dlist', call_assignment_36054_36442)
    
    # Assigning a Call to a Name (line 115):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_36445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 4), 'int')
    # Processing the call keyword arguments
    kwargs_36446 = {}
    # Getting the type of 'call_assignment_36053' (line 115)
    call_assignment_36053_36443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'call_assignment_36053', False)
    # Obtaining the member '__getitem__' of a type (line 115)
    getitem___36444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 4), call_assignment_36053_36443, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_36447 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___36444, *[int_36445], **kwargs_36446)
    
    # Assigning a type to the variable 'call_assignment_36055' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'call_assignment_36055', getitem___call_result_36447)
    
    # Assigning a Name to a Name (line 115):
    # Getting the type of 'call_assignment_36055' (line 115)
    call_assignment_36055_36448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'call_assignment_36055')
    # Assigning a type to the variable 'flist' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'flist', call_assignment_36055_36448)
    
    # Call to output_def(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'dlist' (line 116)
    dlist_36450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'dlist', False)
    # Getting the type of 'flist' (line 116)
    flist_36451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 22), 'flist', False)
    # Getting the type of 'DEF_HEADER' (line 116)
    DEF_HEADER_36452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 29), 'DEF_HEADER', False)
    # Getting the type of 'deffile' (line 116)
    deffile_36453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 41), 'deffile', False)
    # Processing the call keyword arguments (line 116)
    kwargs_36454 = {}
    # Getting the type of 'output_def' (line 116)
    output_def_36449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'output_def', False)
    # Calling output_def(args, kwargs) (line 116)
    output_def_call_result_36455 = invoke(stypy.reporting.localization.Localization(__file__, 116, 4), output_def_36449, *[dlist_36450, flist_36451, DEF_HEADER_36452, deffile_36453], **kwargs_36454)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
