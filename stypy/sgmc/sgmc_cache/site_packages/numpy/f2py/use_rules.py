
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: '''
3: 
4: Build 'use others module data' mechanism for f2py2e.
5: 
6: Unfinished.
7: 
8: Copyright 2000 Pearu Peterson all rights reserved,
9: Pearu Peterson <pearu@ioc.ee>
10: Permission to use, modify, and distribute this software is given under the
11: terms of the NumPy License.
12: 
13: NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
14: $Date: 2000/09/10 12:35:43 $
15: Pearu Peterson
16: 
17: '''
18: from __future__ import division, absolute_import, print_function
19: 
20: __version__ = "$Revision: 1.3 $"[10:-1]
21: 
22: f2py_version = 'See `f2py -v`'
23: 
24: 
25: from .auxfuncs import (
26:     applyrules, dictappend, gentitle, hasnote, outmess
27: )
28: 
29: 
30: usemodule_rules = {
31:     'body': '''
32: #begintitle#
33: static char doc_#apiname#[] = \"\\\nVariable wrapper signature:\\n\\
34: \t #name# = get_#name#()\\n\\
35: Arguments:\\n\\
36: #docstr#\";
37: extern F_MODFUNC(#usemodulename#,#USEMODULENAME#,#realname#,#REALNAME#);
38: static PyObject *#apiname#(PyObject *capi_self, PyObject *capi_args) {
39: /*#decl#*/
40: \tif (!PyArg_ParseTuple(capi_args, \"\")) goto capi_fail;
41: printf(\"c: %d\\n\",F_MODFUNC(#usemodulename#,#USEMODULENAME#,#realname#,#REALNAME#));
42: \treturn Py_BuildValue(\"\");
43: capi_fail:
44: \treturn NULL;
45: }
46: ''',
47:     'method': '\t{\"get_#name#\",#apiname#,METH_VARARGS|METH_KEYWORDS,doc_#apiname#},',
48:     'need': ['F_MODFUNC']
49: }
50: 
51: ################
52: 
53: 
54: def buildusevars(m, r):
55:     ret = {}
56:     outmess(
57:         '\t\tBuilding use variable hooks for module "%s" (feature only for F90/F95)...\n' % (m['name']))
58:     varsmap = {}
59:     revmap = {}
60:     if 'map' in r:
61:         for k in r['map'].keys():
62:             if r['map'][k] in revmap:
63:                 outmess('\t\t\tVariable "%s<=%s" is already mapped by "%s". Skipping.\n' % (
64:                     r['map'][k], k, revmap[r['map'][k]]))
65:             else:
66:                 revmap[r['map'][k]] = k
67:     if 'only' in r and r['only']:
68:         for v in r['map'].keys():
69:             if r['map'][v] in m['vars']:
70: 
71:                 if revmap[r['map'][v]] == v:
72:                     varsmap[v] = r['map'][v]
73:                 else:
74:                     outmess('\t\t\tIgnoring map "%s=>%s". See above.\n' %
75:                             (v, r['map'][v]))
76:             else:
77:                 outmess(
78:                     '\t\t\tNo definition for variable "%s=>%s". Skipping.\n' % (v, r['map'][v]))
79:     else:
80:         for v in m['vars'].keys():
81:             if v in revmap:
82:                 varsmap[v] = revmap[v]
83:             else:
84:                 varsmap[v] = v
85:     for v in varsmap.keys():
86:         ret = dictappend(ret, buildusevar(v, varsmap[v], m['vars'], m['name']))
87:     return ret
88: 
89: 
90: def buildusevar(name, realname, vars, usemodulename):
91:     outmess('\t\t\tConstructing wrapper function for variable "%s=>%s"...\n' % (
92:         name, realname))
93:     ret = {}
94:     vrd = {'name': name,
95:            'realname': realname,
96:            'REALNAME': realname.upper(),
97:            'usemodulename': usemodulename,
98:            'USEMODULENAME': usemodulename.upper(),
99:            'texname': name.replace('_', '\\_'),
100:            'begintitle': gentitle('%s=>%s' % (name, realname)),
101:            'endtitle': gentitle('end of %s=>%s' % (name, realname)),
102:            'apiname': '#modulename#_use_%s_from_%s' % (realname, usemodulename)
103:            }
104:     nummap = {0: 'Ro', 1: 'Ri', 2: 'Rii', 3: 'Riii', 4: 'Riv',
105:               5: 'Rv', 6: 'Rvi', 7: 'Rvii', 8: 'Rviii', 9: 'Rix'}
106:     vrd['texnamename'] = name
107:     for i in nummap.keys():
108:         vrd['texnamename'] = vrd['texnamename'].replace(repr(i), nummap[i])
109:     if hasnote(vars[realname]):
110:         vrd['note'] = vars[realname]['note']
111:     rd = dictappend({}, vrd)
112: 
113:     print(name, realname, vars[realname])
114:     ret = applyrules(usemodule_rules, rd)
115:     return ret
116: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_99442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', "\n\nBuild 'use others module data' mechanism for f2py2e.\n\nUnfinished.\n\nCopyright 2000 Pearu Peterson all rights reserved,\nPearu Peterson <pearu@ioc.ee>\nPermission to use, modify, and distribute this software is given under the\nterms of the NumPy License.\n\nNO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.\n$Date: 2000/09/10 12:35:43 $\nPearu Peterson\n\n")

# Assigning a Subscript to a Name (line 20):

# Obtaining the type of the subscript
int_99443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 33), 'int')
int_99444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 36), 'int')
slice_99445 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 20, 14), int_99443, int_99444, None)
str_99446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 14), 'str', '$Revision: 1.3 $')
# Obtaining the member '__getitem__' of a type (line 20)
getitem___99447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 14), str_99446, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 20)
subscript_call_result_99448 = invoke(stypy.reporting.localization.Localization(__file__, 20, 14), getitem___99447, slice_99445)

# Assigning a type to the variable '__version__' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), '__version__', subscript_call_result_99448)

# Assigning a Str to a Name (line 22):
str_99449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 15), 'str', 'See `f2py -v`')
# Assigning a type to the variable 'f2py_version' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'f2py_version', str_99449)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from numpy.f2py.auxfuncs import applyrules, dictappend, gentitle, hasnote, outmess' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_99450 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.f2py.auxfuncs')

if (type(import_99450) is not StypyTypeError):

    if (import_99450 != 'pyd_module'):
        __import__(import_99450)
        sys_modules_99451 = sys.modules[import_99450]
        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.f2py.auxfuncs', sys_modules_99451.module_type_store, module_type_store, ['applyrules', 'dictappend', 'gentitle', 'hasnote', 'outmess'])
        nest_module(stypy.reporting.localization.Localization(__file__, 25, 0), __file__, sys_modules_99451, sys_modules_99451.module_type_store, module_type_store)
    else:
        from numpy.f2py.auxfuncs import applyrules, dictappend, gentitle, hasnote, outmess

        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.f2py.auxfuncs', None, module_type_store, ['applyrules', 'dictappend', 'gentitle', 'hasnote', 'outmess'], [applyrules, dictappend, gentitle, hasnote, outmess])

else:
    # Assigning a type to the variable 'numpy.f2py.auxfuncs' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.f2py.auxfuncs', import_99450)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')


# Assigning a Dict to a Name (line 30):

# Obtaining an instance of the builtin type 'dict' (line 30)
dict_99452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 30)
# Adding element type (key, value) (line 30)
str_99453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 4), 'str', 'body')
str_99454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, (-1)), 'str', '\n#begintitle#\nstatic char doc_#apiname#[] = "\\\nVariable wrapper signature:\\n\\\n\t #name# = get_#name#()\\n\\\nArguments:\\n\\\n#docstr#";\nextern F_MODFUNC(#usemodulename#,#USEMODULENAME#,#realname#,#REALNAME#);\nstatic PyObject *#apiname#(PyObject *capi_self, PyObject *capi_args) {\n/*#decl#*/\n\tif (!PyArg_ParseTuple(capi_args, "")) goto capi_fail;\nprintf("c: %d\\n",F_MODFUNC(#usemodulename#,#USEMODULENAME#,#realname#,#REALNAME#));\n\treturn Py_BuildValue("");\ncapi_fail:\n\treturn NULL;\n}\n')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 18), dict_99452, (str_99453, str_99454))
# Adding element type (key, value) (line 30)
str_99455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 4), 'str', 'method')
str_99456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 14), 'str', '\t{"get_#name#",#apiname#,METH_VARARGS|METH_KEYWORDS,doc_#apiname#},')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 18), dict_99452, (str_99455, str_99456))
# Adding element type (key, value) (line 30)
str_99457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 4), 'str', 'need')

# Obtaining an instance of the builtin type 'list' (line 48)
list_99458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 48)
# Adding element type (line 48)
str_99459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 13), 'str', 'F_MODFUNC')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 12), list_99458, str_99459)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 18), dict_99452, (str_99457, list_99458))

# Assigning a type to the variable 'usemodule_rules' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'usemodule_rules', dict_99452)

@norecursion
def buildusevars(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'buildusevars'
    module_type_store = module_type_store.open_function_context('buildusevars', 54, 0, False)
    
    # Passed parameters checking function
    buildusevars.stypy_localization = localization
    buildusevars.stypy_type_of_self = None
    buildusevars.stypy_type_store = module_type_store
    buildusevars.stypy_function_name = 'buildusevars'
    buildusevars.stypy_param_names_list = ['m', 'r']
    buildusevars.stypy_varargs_param_name = None
    buildusevars.stypy_kwargs_param_name = None
    buildusevars.stypy_call_defaults = defaults
    buildusevars.stypy_call_varargs = varargs
    buildusevars.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'buildusevars', ['m', 'r'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'buildusevars', localization, ['m', 'r'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'buildusevars(...)' code ##################

    
    # Assigning a Dict to a Name (line 55):
    
    # Obtaining an instance of the builtin type 'dict' (line 55)
    dict_99460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 10), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 55)
    
    # Assigning a type to the variable 'ret' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'ret', dict_99460)
    
    # Call to outmess(...): (line 56)
    # Processing the call arguments (line 56)
    str_99462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 8), 'str', '\t\tBuilding use variable hooks for module "%s" (feature only for F90/F95)...\n')
    
    # Obtaining the type of the subscript
    str_99463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 95), 'str', 'name')
    # Getting the type of 'm' (line 57)
    m_99464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 93), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 57)
    getitem___99465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 93), m_99464, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 57)
    subscript_call_result_99466 = invoke(stypy.reporting.localization.Localization(__file__, 57, 93), getitem___99465, str_99463)
    
    # Applying the binary operator '%' (line 57)
    result_mod_99467 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 8), '%', str_99462, subscript_call_result_99466)
    
    # Processing the call keyword arguments (line 56)
    kwargs_99468 = {}
    # Getting the type of 'outmess' (line 56)
    outmess_99461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'outmess', False)
    # Calling outmess(args, kwargs) (line 56)
    outmess_call_result_99469 = invoke(stypy.reporting.localization.Localization(__file__, 56, 4), outmess_99461, *[result_mod_99467], **kwargs_99468)
    
    
    # Assigning a Dict to a Name (line 58):
    
    # Obtaining an instance of the builtin type 'dict' (line 58)
    dict_99470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 14), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 58)
    
    # Assigning a type to the variable 'varsmap' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'varsmap', dict_99470)
    
    # Assigning a Dict to a Name (line 59):
    
    # Obtaining an instance of the builtin type 'dict' (line 59)
    dict_99471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 59)
    
    # Assigning a type to the variable 'revmap' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'revmap', dict_99471)
    
    
    str_99472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 7), 'str', 'map')
    # Getting the type of 'r' (line 60)
    r_99473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'r')
    # Applying the binary operator 'in' (line 60)
    result_contains_99474 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 7), 'in', str_99472, r_99473)
    
    # Testing the type of an if condition (line 60)
    if_condition_99475 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 4), result_contains_99474)
    # Assigning a type to the variable 'if_condition_99475' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'if_condition_99475', if_condition_99475)
    # SSA begins for if statement (line 60)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to keys(...): (line 61)
    # Processing the call keyword arguments (line 61)
    kwargs_99481 = {}
    
    # Obtaining the type of the subscript
    str_99476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 19), 'str', 'map')
    # Getting the type of 'r' (line 61)
    r_99477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 17), 'r', False)
    # Obtaining the member '__getitem__' of a type (line 61)
    getitem___99478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 17), r_99477, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 61)
    subscript_call_result_99479 = invoke(stypy.reporting.localization.Localization(__file__, 61, 17), getitem___99478, str_99476)
    
    # Obtaining the member 'keys' of a type (line 61)
    keys_99480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 17), subscript_call_result_99479, 'keys')
    # Calling keys(args, kwargs) (line 61)
    keys_call_result_99482 = invoke(stypy.reporting.localization.Localization(__file__, 61, 17), keys_99480, *[], **kwargs_99481)
    
    # Testing the type of a for loop iterable (line 61)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 61, 8), keys_call_result_99482)
    # Getting the type of the for loop variable (line 61)
    for_loop_var_99483 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 61, 8), keys_call_result_99482)
    # Assigning a type to the variable 'k' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'k', for_loop_var_99483)
    # SSA begins for a for statement (line 61)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 62)
    k_99484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 24), 'k')
    
    # Obtaining the type of the subscript
    str_99485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 17), 'str', 'map')
    # Getting the type of 'r' (line 62)
    r_99486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'r')
    # Obtaining the member '__getitem__' of a type (line 62)
    getitem___99487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 15), r_99486, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 62)
    subscript_call_result_99488 = invoke(stypy.reporting.localization.Localization(__file__, 62, 15), getitem___99487, str_99485)
    
    # Obtaining the member '__getitem__' of a type (line 62)
    getitem___99489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 15), subscript_call_result_99488, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 62)
    subscript_call_result_99490 = invoke(stypy.reporting.localization.Localization(__file__, 62, 15), getitem___99489, k_99484)
    
    # Getting the type of 'revmap' (line 62)
    revmap_99491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 30), 'revmap')
    # Applying the binary operator 'in' (line 62)
    result_contains_99492 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 15), 'in', subscript_call_result_99490, revmap_99491)
    
    # Testing the type of an if condition (line 62)
    if_condition_99493 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 12), result_contains_99492)
    # Assigning a type to the variable 'if_condition_99493' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'if_condition_99493', if_condition_99493)
    # SSA begins for if statement (line 62)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to outmess(...): (line 63)
    # Processing the call arguments (line 63)
    str_99495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 24), 'str', '\t\t\tVariable "%s<=%s" is already mapped by "%s". Skipping.\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 64)
    tuple_99496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 64)
    # Adding element type (line 64)
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 64)
    k_99497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 29), 'k', False)
    
    # Obtaining the type of the subscript
    str_99498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 22), 'str', 'map')
    # Getting the type of 'r' (line 64)
    r_99499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 20), 'r', False)
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___99500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 20), r_99499, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_99501 = invoke(stypy.reporting.localization.Localization(__file__, 64, 20), getitem___99500, str_99498)
    
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___99502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 20), subscript_call_result_99501, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_99503 = invoke(stypy.reporting.localization.Localization(__file__, 64, 20), getitem___99502, k_99497)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 20), tuple_99496, subscript_call_result_99503)
    # Adding element type (line 64)
    # Getting the type of 'k' (line 64)
    k_99504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 33), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 20), tuple_99496, k_99504)
    # Adding element type (line 64)
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 64)
    k_99505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 52), 'k', False)
    
    # Obtaining the type of the subscript
    str_99506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 45), 'str', 'map')
    # Getting the type of 'r' (line 64)
    r_99507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 43), 'r', False)
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___99508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 43), r_99507, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_99509 = invoke(stypy.reporting.localization.Localization(__file__, 64, 43), getitem___99508, str_99506)
    
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___99510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 43), subscript_call_result_99509, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_99511 = invoke(stypy.reporting.localization.Localization(__file__, 64, 43), getitem___99510, k_99505)
    
    # Getting the type of 'revmap' (line 64)
    revmap_99512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 36), 'revmap', False)
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___99513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 36), revmap_99512, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_99514 = invoke(stypy.reporting.localization.Localization(__file__, 64, 36), getitem___99513, subscript_call_result_99511)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 20), tuple_99496, subscript_call_result_99514)
    
    # Applying the binary operator '%' (line 63)
    result_mod_99515 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 24), '%', str_99495, tuple_99496)
    
    # Processing the call keyword arguments (line 63)
    kwargs_99516 = {}
    # Getting the type of 'outmess' (line 63)
    outmess_99494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 'outmess', False)
    # Calling outmess(args, kwargs) (line 63)
    outmess_call_result_99517 = invoke(stypy.reporting.localization.Localization(__file__, 63, 16), outmess_99494, *[result_mod_99515], **kwargs_99516)
    
    # SSA branch for the else part of an if statement (line 62)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Subscript (line 66):
    # Getting the type of 'k' (line 66)
    k_99518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 38), 'k')
    # Getting the type of 'revmap' (line 66)
    revmap_99519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'revmap')
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 66)
    k_99520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 32), 'k')
    
    # Obtaining the type of the subscript
    str_99521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 25), 'str', 'map')
    # Getting the type of 'r' (line 66)
    r_99522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 23), 'r')
    # Obtaining the member '__getitem__' of a type (line 66)
    getitem___99523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 23), r_99522, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 66)
    subscript_call_result_99524 = invoke(stypy.reporting.localization.Localization(__file__, 66, 23), getitem___99523, str_99521)
    
    # Obtaining the member '__getitem__' of a type (line 66)
    getitem___99525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 23), subscript_call_result_99524, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 66)
    subscript_call_result_99526 = invoke(stypy.reporting.localization.Localization(__file__, 66, 23), getitem___99525, k_99520)
    
    # Storing an element on a container (line 66)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 16), revmap_99519, (subscript_call_result_99526, k_99518))
    # SSA join for if statement (line 62)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 60)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    str_99527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 7), 'str', 'only')
    # Getting the type of 'r' (line 67)
    r_99528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'r')
    # Applying the binary operator 'in' (line 67)
    result_contains_99529 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 7), 'in', str_99527, r_99528)
    
    
    # Obtaining the type of the subscript
    str_99530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 25), 'str', 'only')
    # Getting the type of 'r' (line 67)
    r_99531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 23), 'r')
    # Obtaining the member '__getitem__' of a type (line 67)
    getitem___99532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 23), r_99531, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 67)
    subscript_call_result_99533 = invoke(stypy.reporting.localization.Localization(__file__, 67, 23), getitem___99532, str_99530)
    
    # Applying the binary operator 'and' (line 67)
    result_and_keyword_99534 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 7), 'and', result_contains_99529, subscript_call_result_99533)
    
    # Testing the type of an if condition (line 67)
    if_condition_99535 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 4), result_and_keyword_99534)
    # Assigning a type to the variable 'if_condition_99535' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'if_condition_99535', if_condition_99535)
    # SSA begins for if statement (line 67)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to keys(...): (line 68)
    # Processing the call keyword arguments (line 68)
    kwargs_99541 = {}
    
    # Obtaining the type of the subscript
    str_99536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 19), 'str', 'map')
    # Getting the type of 'r' (line 68)
    r_99537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 17), 'r', False)
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___99538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 17), r_99537, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_99539 = invoke(stypy.reporting.localization.Localization(__file__, 68, 17), getitem___99538, str_99536)
    
    # Obtaining the member 'keys' of a type (line 68)
    keys_99540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 17), subscript_call_result_99539, 'keys')
    # Calling keys(args, kwargs) (line 68)
    keys_call_result_99542 = invoke(stypy.reporting.localization.Localization(__file__, 68, 17), keys_99540, *[], **kwargs_99541)
    
    # Testing the type of a for loop iterable (line 68)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 68, 8), keys_call_result_99542)
    # Getting the type of the for loop variable (line 68)
    for_loop_var_99543 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 68, 8), keys_call_result_99542)
    # Assigning a type to the variable 'v' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'v', for_loop_var_99543)
    # SSA begins for a for statement (line 68)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'v' (line 69)
    v_99544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 24), 'v')
    
    # Obtaining the type of the subscript
    str_99545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 17), 'str', 'map')
    # Getting the type of 'r' (line 69)
    r_99546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'r')
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___99547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 15), r_99546, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_99548 = invoke(stypy.reporting.localization.Localization(__file__, 69, 15), getitem___99547, str_99545)
    
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___99549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 15), subscript_call_result_99548, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_99550 = invoke(stypy.reporting.localization.Localization(__file__, 69, 15), getitem___99549, v_99544)
    
    
    # Obtaining the type of the subscript
    str_99551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 32), 'str', 'vars')
    # Getting the type of 'm' (line 69)
    m_99552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 30), 'm')
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___99553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 30), m_99552, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_99554 = invoke(stypy.reporting.localization.Localization(__file__, 69, 30), getitem___99553, str_99551)
    
    # Applying the binary operator 'in' (line 69)
    result_contains_99555 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 15), 'in', subscript_call_result_99550, subscript_call_result_99554)
    
    # Testing the type of an if condition (line 69)
    if_condition_99556 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 12), result_contains_99555)
    # Assigning a type to the variable 'if_condition_99556' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'if_condition_99556', if_condition_99556)
    # SSA begins for if statement (line 69)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    # Getting the type of 'v' (line 71)
    v_99557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 35), 'v')
    
    # Obtaining the type of the subscript
    str_99558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 28), 'str', 'map')
    # Getting the type of 'r' (line 71)
    r_99559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 26), 'r')
    # Obtaining the member '__getitem__' of a type (line 71)
    getitem___99560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 26), r_99559, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
    subscript_call_result_99561 = invoke(stypy.reporting.localization.Localization(__file__, 71, 26), getitem___99560, str_99558)
    
    # Obtaining the member '__getitem__' of a type (line 71)
    getitem___99562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 26), subscript_call_result_99561, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
    subscript_call_result_99563 = invoke(stypy.reporting.localization.Localization(__file__, 71, 26), getitem___99562, v_99557)
    
    # Getting the type of 'revmap' (line 71)
    revmap_99564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'revmap')
    # Obtaining the member '__getitem__' of a type (line 71)
    getitem___99565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 19), revmap_99564, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
    subscript_call_result_99566 = invoke(stypy.reporting.localization.Localization(__file__, 71, 19), getitem___99565, subscript_call_result_99563)
    
    # Getting the type of 'v' (line 71)
    v_99567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 42), 'v')
    # Applying the binary operator '==' (line 71)
    result_eq_99568 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 19), '==', subscript_call_result_99566, v_99567)
    
    # Testing the type of an if condition (line 71)
    if_condition_99569 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 16), result_eq_99568)
    # Assigning a type to the variable 'if_condition_99569' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'if_condition_99569', if_condition_99569)
    # SSA begins for if statement (line 71)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 72):
    
    # Obtaining the type of the subscript
    # Getting the type of 'v' (line 72)
    v_99570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 42), 'v')
    
    # Obtaining the type of the subscript
    str_99571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 35), 'str', 'map')
    # Getting the type of 'r' (line 72)
    r_99572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 33), 'r')
    # Obtaining the member '__getitem__' of a type (line 72)
    getitem___99573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 33), r_99572, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
    subscript_call_result_99574 = invoke(stypy.reporting.localization.Localization(__file__, 72, 33), getitem___99573, str_99571)
    
    # Obtaining the member '__getitem__' of a type (line 72)
    getitem___99575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 33), subscript_call_result_99574, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
    subscript_call_result_99576 = invoke(stypy.reporting.localization.Localization(__file__, 72, 33), getitem___99575, v_99570)
    
    # Getting the type of 'varsmap' (line 72)
    varsmap_99577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 20), 'varsmap')
    # Getting the type of 'v' (line 72)
    v_99578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 28), 'v')
    # Storing an element on a container (line 72)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 20), varsmap_99577, (v_99578, subscript_call_result_99576))
    # SSA branch for the else part of an if statement (line 71)
    module_type_store.open_ssa_branch('else')
    
    # Call to outmess(...): (line 74)
    # Processing the call arguments (line 74)
    str_99580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 28), 'str', '\t\t\tIgnoring map "%s=>%s". See above.\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 75)
    tuple_99581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 75)
    # Adding element type (line 75)
    # Getting the type of 'v' (line 75)
    v_99582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 29), 'v', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 29), tuple_99581, v_99582)
    # Adding element type (line 75)
    
    # Obtaining the type of the subscript
    # Getting the type of 'v' (line 75)
    v_99583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 41), 'v', False)
    
    # Obtaining the type of the subscript
    str_99584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 34), 'str', 'map')
    # Getting the type of 'r' (line 75)
    r_99585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 32), 'r', False)
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___99586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 32), r_99585, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_99587 = invoke(stypy.reporting.localization.Localization(__file__, 75, 32), getitem___99586, str_99584)
    
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___99588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 32), subscript_call_result_99587, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_99589 = invoke(stypy.reporting.localization.Localization(__file__, 75, 32), getitem___99588, v_99583)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 29), tuple_99581, subscript_call_result_99589)
    
    # Applying the binary operator '%' (line 74)
    result_mod_99590 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 28), '%', str_99580, tuple_99581)
    
    # Processing the call keyword arguments (line 74)
    kwargs_99591 = {}
    # Getting the type of 'outmess' (line 74)
    outmess_99579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'outmess', False)
    # Calling outmess(args, kwargs) (line 74)
    outmess_call_result_99592 = invoke(stypy.reporting.localization.Localization(__file__, 74, 20), outmess_99579, *[result_mod_99590], **kwargs_99591)
    
    # SSA join for if statement (line 71)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 69)
    module_type_store.open_ssa_branch('else')
    
    # Call to outmess(...): (line 77)
    # Processing the call arguments (line 77)
    str_99594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 20), 'str', '\t\t\tNo definition for variable "%s=>%s". Skipping.\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 78)
    tuple_99595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 80), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 78)
    # Adding element type (line 78)
    # Getting the type of 'v' (line 78)
    v_99596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 80), 'v', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 80), tuple_99595, v_99596)
    # Adding element type (line 78)
    
    # Obtaining the type of the subscript
    # Getting the type of 'v' (line 78)
    v_99597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 92), 'v', False)
    
    # Obtaining the type of the subscript
    str_99598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 85), 'str', 'map')
    # Getting the type of 'r' (line 78)
    r_99599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 83), 'r', False)
    # Obtaining the member '__getitem__' of a type (line 78)
    getitem___99600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 83), r_99599, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 78)
    subscript_call_result_99601 = invoke(stypy.reporting.localization.Localization(__file__, 78, 83), getitem___99600, str_99598)
    
    # Obtaining the member '__getitem__' of a type (line 78)
    getitem___99602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 83), subscript_call_result_99601, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 78)
    subscript_call_result_99603 = invoke(stypy.reporting.localization.Localization(__file__, 78, 83), getitem___99602, v_99597)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 80), tuple_99595, subscript_call_result_99603)
    
    # Applying the binary operator '%' (line 78)
    result_mod_99604 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 20), '%', str_99594, tuple_99595)
    
    # Processing the call keyword arguments (line 77)
    kwargs_99605 = {}
    # Getting the type of 'outmess' (line 77)
    outmess_99593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 'outmess', False)
    # Calling outmess(args, kwargs) (line 77)
    outmess_call_result_99606 = invoke(stypy.reporting.localization.Localization(__file__, 77, 16), outmess_99593, *[result_mod_99604], **kwargs_99605)
    
    # SSA join for if statement (line 69)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 67)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to keys(...): (line 80)
    # Processing the call keyword arguments (line 80)
    kwargs_99612 = {}
    
    # Obtaining the type of the subscript
    str_99607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 19), 'str', 'vars')
    # Getting the type of 'm' (line 80)
    m_99608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 17), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 80)
    getitem___99609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 17), m_99608, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 80)
    subscript_call_result_99610 = invoke(stypy.reporting.localization.Localization(__file__, 80, 17), getitem___99609, str_99607)
    
    # Obtaining the member 'keys' of a type (line 80)
    keys_99611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 17), subscript_call_result_99610, 'keys')
    # Calling keys(args, kwargs) (line 80)
    keys_call_result_99613 = invoke(stypy.reporting.localization.Localization(__file__, 80, 17), keys_99611, *[], **kwargs_99612)
    
    # Testing the type of a for loop iterable (line 80)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 80, 8), keys_call_result_99613)
    # Getting the type of the for loop variable (line 80)
    for_loop_var_99614 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 80, 8), keys_call_result_99613)
    # Assigning a type to the variable 'v' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'v', for_loop_var_99614)
    # SSA begins for a for statement (line 80)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'v' (line 81)
    v_99615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 15), 'v')
    # Getting the type of 'revmap' (line 81)
    revmap_99616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 20), 'revmap')
    # Applying the binary operator 'in' (line 81)
    result_contains_99617 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 15), 'in', v_99615, revmap_99616)
    
    # Testing the type of an if condition (line 81)
    if_condition_99618 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 12), result_contains_99617)
    # Assigning a type to the variable 'if_condition_99618' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'if_condition_99618', if_condition_99618)
    # SSA begins for if statement (line 81)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 82):
    
    # Obtaining the type of the subscript
    # Getting the type of 'v' (line 82)
    v_99619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 36), 'v')
    # Getting the type of 'revmap' (line 82)
    revmap_99620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 29), 'revmap')
    # Obtaining the member '__getitem__' of a type (line 82)
    getitem___99621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 29), revmap_99620, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 82)
    subscript_call_result_99622 = invoke(stypy.reporting.localization.Localization(__file__, 82, 29), getitem___99621, v_99619)
    
    # Getting the type of 'varsmap' (line 82)
    varsmap_99623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'varsmap')
    # Getting the type of 'v' (line 82)
    v_99624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 24), 'v')
    # Storing an element on a container (line 82)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 16), varsmap_99623, (v_99624, subscript_call_result_99622))
    # SSA branch for the else part of an if statement (line 81)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Subscript (line 84):
    # Getting the type of 'v' (line 84)
    v_99625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 29), 'v')
    # Getting the type of 'varsmap' (line 84)
    varsmap_99626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'varsmap')
    # Getting the type of 'v' (line 84)
    v_99627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 24), 'v')
    # Storing an element on a container (line 84)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 16), varsmap_99626, (v_99627, v_99625))
    # SSA join for if statement (line 81)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 67)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to keys(...): (line 85)
    # Processing the call keyword arguments (line 85)
    kwargs_99630 = {}
    # Getting the type of 'varsmap' (line 85)
    varsmap_99628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 13), 'varsmap', False)
    # Obtaining the member 'keys' of a type (line 85)
    keys_99629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 13), varsmap_99628, 'keys')
    # Calling keys(args, kwargs) (line 85)
    keys_call_result_99631 = invoke(stypy.reporting.localization.Localization(__file__, 85, 13), keys_99629, *[], **kwargs_99630)
    
    # Testing the type of a for loop iterable (line 85)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 85, 4), keys_call_result_99631)
    # Getting the type of the for loop variable (line 85)
    for_loop_var_99632 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 85, 4), keys_call_result_99631)
    # Assigning a type to the variable 'v' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'v', for_loop_var_99632)
    # SSA begins for a for statement (line 85)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 86):
    
    # Call to dictappend(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'ret' (line 86)
    ret_99634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 25), 'ret', False)
    
    # Call to buildusevar(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'v' (line 86)
    v_99636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 42), 'v', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'v' (line 86)
    v_99637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 53), 'v', False)
    # Getting the type of 'varsmap' (line 86)
    varsmap_99638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 45), 'varsmap', False)
    # Obtaining the member '__getitem__' of a type (line 86)
    getitem___99639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 45), varsmap_99638, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 86)
    subscript_call_result_99640 = invoke(stypy.reporting.localization.Localization(__file__, 86, 45), getitem___99639, v_99637)
    
    
    # Obtaining the type of the subscript
    str_99641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 59), 'str', 'vars')
    # Getting the type of 'm' (line 86)
    m_99642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 57), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 86)
    getitem___99643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 57), m_99642, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 86)
    subscript_call_result_99644 = invoke(stypy.reporting.localization.Localization(__file__, 86, 57), getitem___99643, str_99641)
    
    
    # Obtaining the type of the subscript
    str_99645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 70), 'str', 'name')
    # Getting the type of 'm' (line 86)
    m_99646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 68), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 86)
    getitem___99647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 68), m_99646, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 86)
    subscript_call_result_99648 = invoke(stypy.reporting.localization.Localization(__file__, 86, 68), getitem___99647, str_99645)
    
    # Processing the call keyword arguments (line 86)
    kwargs_99649 = {}
    # Getting the type of 'buildusevar' (line 86)
    buildusevar_99635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 30), 'buildusevar', False)
    # Calling buildusevar(args, kwargs) (line 86)
    buildusevar_call_result_99650 = invoke(stypy.reporting.localization.Localization(__file__, 86, 30), buildusevar_99635, *[v_99636, subscript_call_result_99640, subscript_call_result_99644, subscript_call_result_99648], **kwargs_99649)
    
    # Processing the call keyword arguments (line 86)
    kwargs_99651 = {}
    # Getting the type of 'dictappend' (line 86)
    dictappend_99633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 14), 'dictappend', False)
    # Calling dictappend(args, kwargs) (line 86)
    dictappend_call_result_99652 = invoke(stypy.reporting.localization.Localization(__file__, 86, 14), dictappend_99633, *[ret_99634, buildusevar_call_result_99650], **kwargs_99651)
    
    # Assigning a type to the variable 'ret' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'ret', dictappend_call_result_99652)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ret' (line 87)
    ret_99653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 11), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'stypy_return_type', ret_99653)
    
    # ################# End of 'buildusevars(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'buildusevars' in the type store
    # Getting the type of 'stypy_return_type' (line 54)
    stypy_return_type_99654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_99654)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'buildusevars'
    return stypy_return_type_99654

# Assigning a type to the variable 'buildusevars' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'buildusevars', buildusevars)

@norecursion
def buildusevar(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'buildusevar'
    module_type_store = module_type_store.open_function_context('buildusevar', 90, 0, False)
    
    # Passed parameters checking function
    buildusevar.stypy_localization = localization
    buildusevar.stypy_type_of_self = None
    buildusevar.stypy_type_store = module_type_store
    buildusevar.stypy_function_name = 'buildusevar'
    buildusevar.stypy_param_names_list = ['name', 'realname', 'vars', 'usemodulename']
    buildusevar.stypy_varargs_param_name = None
    buildusevar.stypy_kwargs_param_name = None
    buildusevar.stypy_call_defaults = defaults
    buildusevar.stypy_call_varargs = varargs
    buildusevar.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'buildusevar', ['name', 'realname', 'vars', 'usemodulename'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'buildusevar', localization, ['name', 'realname', 'vars', 'usemodulename'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'buildusevar(...)' code ##################

    
    # Call to outmess(...): (line 91)
    # Processing the call arguments (line 91)
    str_99656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 12), 'str', '\t\t\tConstructing wrapper function for variable "%s=>%s"...\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 92)
    tuple_99657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 8), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 92)
    # Adding element type (line 92)
    # Getting the type of 'name' (line 92)
    name_99658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 8), tuple_99657, name_99658)
    # Adding element type (line 92)
    # Getting the type of 'realname' (line 92)
    realname_99659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 14), 'realname', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 8), tuple_99657, realname_99659)
    
    # Applying the binary operator '%' (line 91)
    result_mod_99660 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 12), '%', str_99656, tuple_99657)
    
    # Processing the call keyword arguments (line 91)
    kwargs_99661 = {}
    # Getting the type of 'outmess' (line 91)
    outmess_99655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'outmess', False)
    # Calling outmess(args, kwargs) (line 91)
    outmess_call_result_99662 = invoke(stypy.reporting.localization.Localization(__file__, 91, 4), outmess_99655, *[result_mod_99660], **kwargs_99661)
    
    
    # Assigning a Dict to a Name (line 93):
    
    # Obtaining an instance of the builtin type 'dict' (line 93)
    dict_99663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 10), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 93)
    
    # Assigning a type to the variable 'ret' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'ret', dict_99663)
    
    # Assigning a Dict to a Name (line 94):
    
    # Obtaining an instance of the builtin type 'dict' (line 94)
    dict_99664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 10), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 94)
    # Adding element type (key, value) (line 94)
    str_99665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 11), 'str', 'name')
    # Getting the type of 'name' (line 94)
    name_99666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 19), 'name')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 10), dict_99664, (str_99665, name_99666))
    # Adding element type (key, value) (line 94)
    str_99667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 11), 'str', 'realname')
    # Getting the type of 'realname' (line 95)
    realname_99668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 23), 'realname')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 10), dict_99664, (str_99667, realname_99668))
    # Adding element type (key, value) (line 94)
    str_99669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 11), 'str', 'REALNAME')
    
    # Call to upper(...): (line 96)
    # Processing the call keyword arguments (line 96)
    kwargs_99672 = {}
    # Getting the type of 'realname' (line 96)
    realname_99670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 23), 'realname', False)
    # Obtaining the member 'upper' of a type (line 96)
    upper_99671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 23), realname_99670, 'upper')
    # Calling upper(args, kwargs) (line 96)
    upper_call_result_99673 = invoke(stypy.reporting.localization.Localization(__file__, 96, 23), upper_99671, *[], **kwargs_99672)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 10), dict_99664, (str_99669, upper_call_result_99673))
    # Adding element type (key, value) (line 94)
    str_99674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 11), 'str', 'usemodulename')
    # Getting the type of 'usemodulename' (line 97)
    usemodulename_99675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 28), 'usemodulename')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 10), dict_99664, (str_99674, usemodulename_99675))
    # Adding element type (key, value) (line 94)
    str_99676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 11), 'str', 'USEMODULENAME')
    
    # Call to upper(...): (line 98)
    # Processing the call keyword arguments (line 98)
    kwargs_99679 = {}
    # Getting the type of 'usemodulename' (line 98)
    usemodulename_99677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 28), 'usemodulename', False)
    # Obtaining the member 'upper' of a type (line 98)
    upper_99678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 28), usemodulename_99677, 'upper')
    # Calling upper(args, kwargs) (line 98)
    upper_call_result_99680 = invoke(stypy.reporting.localization.Localization(__file__, 98, 28), upper_99678, *[], **kwargs_99679)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 10), dict_99664, (str_99676, upper_call_result_99680))
    # Adding element type (key, value) (line 94)
    str_99681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 11), 'str', 'texname')
    
    # Call to replace(...): (line 99)
    # Processing the call arguments (line 99)
    str_99684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 35), 'str', '_')
    str_99685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 40), 'str', '\\_')
    # Processing the call keyword arguments (line 99)
    kwargs_99686 = {}
    # Getting the type of 'name' (line 99)
    name_99682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 22), 'name', False)
    # Obtaining the member 'replace' of a type (line 99)
    replace_99683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 22), name_99682, 'replace')
    # Calling replace(args, kwargs) (line 99)
    replace_call_result_99687 = invoke(stypy.reporting.localization.Localization(__file__, 99, 22), replace_99683, *[str_99684, str_99685], **kwargs_99686)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 10), dict_99664, (str_99681, replace_call_result_99687))
    # Adding element type (key, value) (line 94)
    str_99688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 11), 'str', 'begintitle')
    
    # Call to gentitle(...): (line 100)
    # Processing the call arguments (line 100)
    str_99690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 34), 'str', '%s=>%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 100)
    tuple_99691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 100)
    # Adding element type (line 100)
    # Getting the type of 'name' (line 100)
    name_99692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 46), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 46), tuple_99691, name_99692)
    # Adding element type (line 100)
    # Getting the type of 'realname' (line 100)
    realname_99693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 52), 'realname', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 46), tuple_99691, realname_99693)
    
    # Applying the binary operator '%' (line 100)
    result_mod_99694 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 34), '%', str_99690, tuple_99691)
    
    # Processing the call keyword arguments (line 100)
    kwargs_99695 = {}
    # Getting the type of 'gentitle' (line 100)
    gentitle_99689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 25), 'gentitle', False)
    # Calling gentitle(args, kwargs) (line 100)
    gentitle_call_result_99696 = invoke(stypy.reporting.localization.Localization(__file__, 100, 25), gentitle_99689, *[result_mod_99694], **kwargs_99695)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 10), dict_99664, (str_99688, gentitle_call_result_99696))
    # Adding element type (key, value) (line 94)
    str_99697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 11), 'str', 'endtitle')
    
    # Call to gentitle(...): (line 101)
    # Processing the call arguments (line 101)
    str_99699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 32), 'str', 'end of %s=>%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 101)
    tuple_99700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 51), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 101)
    # Adding element type (line 101)
    # Getting the type of 'name' (line 101)
    name_99701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 51), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 51), tuple_99700, name_99701)
    # Adding element type (line 101)
    # Getting the type of 'realname' (line 101)
    realname_99702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 57), 'realname', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 51), tuple_99700, realname_99702)
    
    # Applying the binary operator '%' (line 101)
    result_mod_99703 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 32), '%', str_99699, tuple_99700)
    
    # Processing the call keyword arguments (line 101)
    kwargs_99704 = {}
    # Getting the type of 'gentitle' (line 101)
    gentitle_99698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 23), 'gentitle', False)
    # Calling gentitle(args, kwargs) (line 101)
    gentitle_call_result_99705 = invoke(stypy.reporting.localization.Localization(__file__, 101, 23), gentitle_99698, *[result_mod_99703], **kwargs_99704)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 10), dict_99664, (str_99697, gentitle_call_result_99705))
    # Adding element type (key, value) (line 94)
    str_99706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 11), 'str', 'apiname')
    str_99707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 22), 'str', '#modulename#_use_%s_from_%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 102)
    tuple_99708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 55), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 102)
    # Adding element type (line 102)
    # Getting the type of 'realname' (line 102)
    realname_99709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 55), 'realname')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 55), tuple_99708, realname_99709)
    # Adding element type (line 102)
    # Getting the type of 'usemodulename' (line 102)
    usemodulename_99710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 65), 'usemodulename')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 55), tuple_99708, usemodulename_99710)
    
    # Applying the binary operator '%' (line 102)
    result_mod_99711 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 22), '%', str_99707, tuple_99708)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 10), dict_99664, (str_99706, result_mod_99711))
    
    # Assigning a type to the variable 'vrd' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'vrd', dict_99664)
    
    # Assigning a Dict to a Name (line 104):
    
    # Obtaining an instance of the builtin type 'dict' (line 104)
    dict_99712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 104)
    # Adding element type (key, value) (line 104)
    int_99713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 14), 'int')
    str_99714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 17), 'str', 'Ro')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 13), dict_99712, (int_99713, str_99714))
    # Adding element type (key, value) (line 104)
    int_99715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 23), 'int')
    str_99716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 26), 'str', 'Ri')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 13), dict_99712, (int_99715, str_99716))
    # Adding element type (key, value) (line 104)
    int_99717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 32), 'int')
    str_99718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 35), 'str', 'Rii')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 13), dict_99712, (int_99717, str_99718))
    # Adding element type (key, value) (line 104)
    int_99719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 42), 'int')
    str_99720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 45), 'str', 'Riii')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 13), dict_99712, (int_99719, str_99720))
    # Adding element type (key, value) (line 104)
    int_99721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 53), 'int')
    str_99722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 56), 'str', 'Riv')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 13), dict_99712, (int_99721, str_99722))
    # Adding element type (key, value) (line 104)
    int_99723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 14), 'int')
    str_99724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 17), 'str', 'Rv')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 13), dict_99712, (int_99723, str_99724))
    # Adding element type (key, value) (line 104)
    int_99725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 23), 'int')
    str_99726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 26), 'str', 'Rvi')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 13), dict_99712, (int_99725, str_99726))
    # Adding element type (key, value) (line 104)
    int_99727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 33), 'int')
    str_99728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 36), 'str', 'Rvii')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 13), dict_99712, (int_99727, str_99728))
    # Adding element type (key, value) (line 104)
    int_99729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 44), 'int')
    str_99730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 47), 'str', 'Rviii')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 13), dict_99712, (int_99729, str_99730))
    # Adding element type (key, value) (line 104)
    int_99731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 56), 'int')
    str_99732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 59), 'str', 'Rix')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 13), dict_99712, (int_99731, str_99732))
    
    # Assigning a type to the variable 'nummap' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'nummap', dict_99712)
    
    # Assigning a Name to a Subscript (line 106):
    # Getting the type of 'name' (line 106)
    name_99733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 25), 'name')
    # Getting the type of 'vrd' (line 106)
    vrd_99734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'vrd')
    str_99735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'str', 'texnamename')
    # Storing an element on a container (line 106)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 4), vrd_99734, (str_99735, name_99733))
    
    
    # Call to keys(...): (line 107)
    # Processing the call keyword arguments (line 107)
    kwargs_99738 = {}
    # Getting the type of 'nummap' (line 107)
    nummap_99736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 13), 'nummap', False)
    # Obtaining the member 'keys' of a type (line 107)
    keys_99737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 13), nummap_99736, 'keys')
    # Calling keys(args, kwargs) (line 107)
    keys_call_result_99739 = invoke(stypy.reporting.localization.Localization(__file__, 107, 13), keys_99737, *[], **kwargs_99738)
    
    # Testing the type of a for loop iterable (line 107)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 107, 4), keys_call_result_99739)
    # Getting the type of the for loop variable (line 107)
    for_loop_var_99740 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 107, 4), keys_call_result_99739)
    # Assigning a type to the variable 'i' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'i', for_loop_var_99740)
    # SSA begins for a for statement (line 107)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 108):
    
    # Call to replace(...): (line 108)
    # Processing the call arguments (line 108)
    
    # Call to repr(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'i' (line 108)
    i_99747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 61), 'i', False)
    # Processing the call keyword arguments (line 108)
    kwargs_99748 = {}
    # Getting the type of 'repr' (line 108)
    repr_99746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 56), 'repr', False)
    # Calling repr(args, kwargs) (line 108)
    repr_call_result_99749 = invoke(stypy.reporting.localization.Localization(__file__, 108, 56), repr_99746, *[i_99747], **kwargs_99748)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 108)
    i_99750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 72), 'i', False)
    # Getting the type of 'nummap' (line 108)
    nummap_99751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 65), 'nummap', False)
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___99752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 65), nummap_99751, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_99753 = invoke(stypy.reporting.localization.Localization(__file__, 108, 65), getitem___99752, i_99750)
    
    # Processing the call keyword arguments (line 108)
    kwargs_99754 = {}
    
    # Obtaining the type of the subscript
    str_99741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 33), 'str', 'texnamename')
    # Getting the type of 'vrd' (line 108)
    vrd_99742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 'vrd', False)
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___99743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 29), vrd_99742, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_99744 = invoke(stypy.reporting.localization.Localization(__file__, 108, 29), getitem___99743, str_99741)
    
    # Obtaining the member 'replace' of a type (line 108)
    replace_99745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 29), subscript_call_result_99744, 'replace')
    # Calling replace(args, kwargs) (line 108)
    replace_call_result_99755 = invoke(stypy.reporting.localization.Localization(__file__, 108, 29), replace_99745, *[repr_call_result_99749, subscript_call_result_99753], **kwargs_99754)
    
    # Getting the type of 'vrd' (line 108)
    vrd_99756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'vrd')
    str_99757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 12), 'str', 'texnamename')
    # Storing an element on a container (line 108)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 8), vrd_99756, (str_99757, replace_call_result_99755))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to hasnote(...): (line 109)
    # Processing the call arguments (line 109)
    
    # Obtaining the type of the subscript
    # Getting the type of 'realname' (line 109)
    realname_99759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 20), 'realname', False)
    # Getting the type of 'vars' (line 109)
    vars_99760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 15), 'vars', False)
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___99761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 15), vars_99760, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_99762 = invoke(stypy.reporting.localization.Localization(__file__, 109, 15), getitem___99761, realname_99759)
    
    # Processing the call keyword arguments (line 109)
    kwargs_99763 = {}
    # Getting the type of 'hasnote' (line 109)
    hasnote_99758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 7), 'hasnote', False)
    # Calling hasnote(args, kwargs) (line 109)
    hasnote_call_result_99764 = invoke(stypy.reporting.localization.Localization(__file__, 109, 7), hasnote_99758, *[subscript_call_result_99762], **kwargs_99763)
    
    # Testing the type of an if condition (line 109)
    if_condition_99765 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 4), hasnote_call_result_99764)
    # Assigning a type to the variable 'if_condition_99765' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'if_condition_99765', if_condition_99765)
    # SSA begins for if statement (line 109)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 110):
    
    # Obtaining the type of the subscript
    str_99766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 37), 'str', 'note')
    
    # Obtaining the type of the subscript
    # Getting the type of 'realname' (line 110)
    realname_99767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 27), 'realname')
    # Getting the type of 'vars' (line 110)
    vars_99768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 22), 'vars')
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___99769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 22), vars_99768, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_99770 = invoke(stypy.reporting.localization.Localization(__file__, 110, 22), getitem___99769, realname_99767)
    
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___99771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 22), subscript_call_result_99770, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_99772 = invoke(stypy.reporting.localization.Localization(__file__, 110, 22), getitem___99771, str_99766)
    
    # Getting the type of 'vrd' (line 110)
    vrd_99773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'vrd')
    str_99774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 12), 'str', 'note')
    # Storing an element on a container (line 110)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 8), vrd_99773, (str_99774, subscript_call_result_99772))
    # SSA join for if statement (line 109)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 111):
    
    # Call to dictappend(...): (line 111)
    # Processing the call arguments (line 111)
    
    # Obtaining an instance of the builtin type 'dict' (line 111)
    dict_99776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 20), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 111)
    
    # Getting the type of 'vrd' (line 111)
    vrd_99777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'vrd', False)
    # Processing the call keyword arguments (line 111)
    kwargs_99778 = {}
    # Getting the type of 'dictappend' (line 111)
    dictappend_99775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 9), 'dictappend', False)
    # Calling dictappend(args, kwargs) (line 111)
    dictappend_call_result_99779 = invoke(stypy.reporting.localization.Localization(__file__, 111, 9), dictappend_99775, *[dict_99776, vrd_99777], **kwargs_99778)
    
    # Assigning a type to the variable 'rd' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'rd', dictappend_call_result_99779)
    
    # Call to print(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'name' (line 113)
    name_99781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 10), 'name', False)
    # Getting the type of 'realname' (line 113)
    realname_99782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'realname', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'realname' (line 113)
    realname_99783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 31), 'realname', False)
    # Getting the type of 'vars' (line 113)
    vars_99784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 26), 'vars', False)
    # Obtaining the member '__getitem__' of a type (line 113)
    getitem___99785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 26), vars_99784, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
    subscript_call_result_99786 = invoke(stypy.reporting.localization.Localization(__file__, 113, 26), getitem___99785, realname_99783)
    
    # Processing the call keyword arguments (line 113)
    kwargs_99787 = {}
    # Getting the type of 'print' (line 113)
    print_99780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'print', False)
    # Calling print(args, kwargs) (line 113)
    print_call_result_99788 = invoke(stypy.reporting.localization.Localization(__file__, 113, 4), print_99780, *[name_99781, realname_99782, subscript_call_result_99786], **kwargs_99787)
    
    
    # Assigning a Call to a Name (line 114):
    
    # Call to applyrules(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'usemodule_rules' (line 114)
    usemodule_rules_99790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 21), 'usemodule_rules', False)
    # Getting the type of 'rd' (line 114)
    rd_99791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 38), 'rd', False)
    # Processing the call keyword arguments (line 114)
    kwargs_99792 = {}
    # Getting the type of 'applyrules' (line 114)
    applyrules_99789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 10), 'applyrules', False)
    # Calling applyrules(args, kwargs) (line 114)
    applyrules_call_result_99793 = invoke(stypy.reporting.localization.Localization(__file__, 114, 10), applyrules_99789, *[usemodule_rules_99790, rd_99791], **kwargs_99792)
    
    # Assigning a type to the variable 'ret' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'ret', applyrules_call_result_99793)
    # Getting the type of 'ret' (line 115)
    ret_99794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type', ret_99794)
    
    # ################# End of 'buildusevar(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'buildusevar' in the type store
    # Getting the type of 'stypy_return_type' (line 90)
    stypy_return_type_99795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_99795)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'buildusevar'
    return stypy_return_type_99795

# Assigning a type to the variable 'buildusevar' (line 90)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'buildusevar', buildusevar)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
