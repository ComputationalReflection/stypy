
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: '''
3: 
4: Build common block mechanism for f2py2e.
5: 
6: Copyright 2000 Pearu Peterson all rights reserved,
7: Pearu Peterson <pearu@ioc.ee>
8: Permission to use, modify, and distribute this software is given under the
9: terms of the NumPy License
10: 
11: NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
12: $Date: 2005/05/06 10:57:33 $
13: Pearu Peterson
14: 
15: '''
16: from __future__ import division, absolute_import, print_function
17: 
18: __version__ = "$Revision: 1.19 $"[10:-1]
19: 
20: from . import __version__
21: f2py_version = __version__.version
22: 
23: from .auxfuncs import (
24:     hasbody, hascommon, hasnote, isintent_hide, outmess
25: )
26: from . import capi_maps
27: from . import func2subr
28: from .crackfortran import rmbadname
29: 
30: 
31: def findcommonblocks(block, top=1):
32:     ret = []
33:     if hascommon(block):
34:         for n in block['common'].keys():
35:             vars = {}
36:             for v in block['common'][n]:
37:                 vars[v] = block['vars'][v]
38:             ret.append((n, block['common'][n], vars))
39:     elif hasbody(block):
40:         for b in block['body']:
41:             ret = ret + findcommonblocks(b, 0)
42:     if top:
43:         tret = []
44:         names = []
45:         for t in ret:
46:             if t[0] not in names:
47:                 names.append(t[0])
48:                 tret.append(t)
49:         return tret
50:     return ret
51: 
52: 
53: def buildhooks(m):
54:     ret = {'commonhooks': [], 'initcommonhooks': [],
55:            'docs': ['"COMMON blocks:\\n"']}
56:     fwrap = ['']
57: 
58:     def fadd(line, s=fwrap):
59:         s[0] = '%s\n      %s' % (s[0], line)
60:     chooks = ['']
61: 
62:     def cadd(line, s=chooks):
63:         s[0] = '%s\n%s' % (s[0], line)
64:     ihooks = ['']
65: 
66:     def iadd(line, s=ihooks):
67:         s[0] = '%s\n%s' % (s[0], line)
68:     doc = ['']
69: 
70:     def dadd(line, s=doc):
71:         s[0] = '%s\n%s' % (s[0], line)
72:     for (name, vnames, vars) in findcommonblocks(m):
73:         lower_name = name.lower()
74:         hnames, inames = [], []
75:         for n in vnames:
76:             if isintent_hide(vars[n]):
77:                 hnames.append(n)
78:             else:
79:                 inames.append(n)
80:         if hnames:
81:             outmess('\t\tConstructing COMMON block support for "%s"...\n\t\t  %s\n\t\t  Hidden: %s\n' % (
82:                 name, ','.join(inames), ','.join(hnames)))
83:         else:
84:             outmess('\t\tConstructing COMMON block support for "%s"...\n\t\t  %s\n' % (
85:                 name, ','.join(inames)))
86:         fadd('subroutine f2pyinit%s(setupfunc)' % name)
87:         fadd('external setupfunc')
88:         for n in vnames:
89:             fadd(func2subr.var2fixfortran(vars, n))
90:         if name == '_BLNK_':
91:             fadd('common %s' % (','.join(vnames)))
92:         else:
93:             fadd('common /%s/ %s' % (name, ','.join(vnames)))
94:         fadd('call setupfunc(%s)' % (','.join(inames)))
95:         fadd('end\n')
96:         cadd('static FortranDataDef f2py_%s_def[] = {' % (name))
97:         idims = []
98:         for n in inames:
99:             ct = capi_maps.getctype(vars[n])
100:             at = capi_maps.c2capi_map[ct]
101:             dm = capi_maps.getarrdims(n, vars[n])
102:             if dm['dims']:
103:                 idims.append('(%s)' % (dm['dims']))
104:             else:
105:                 idims.append('')
106:             dms = dm['dims'].strip()
107:             if not dms:
108:                 dms = '-1'
109:             cadd('\t{\"%s\",%s,{{%s}},%s},' % (n, dm['rank'], dms, at))
110:         cadd('\t{NULL}\n};')
111:         inames1 = rmbadname(inames)
112:         inames1_tps = ','.join(['char *' + s for s in inames1])
113:         cadd('static void f2py_setup_%s(%s) {' % (name, inames1_tps))
114:         cadd('\tint i_f2py=0;')
115:         for n in inames1:
116:             cadd('\tf2py_%s_def[i_f2py++].data = %s;' % (name, n))
117:         cadd('}')
118:         if '_' in lower_name:
119:             F_FUNC = 'F_FUNC_US'
120:         else:
121:             F_FUNC = 'F_FUNC'
122:         cadd('extern void %s(f2pyinit%s,F2PYINIT%s)(void(*)(%s));'
123:              % (F_FUNC, lower_name, name.upper(),
124:                 ','.join(['char*'] * len(inames1))))
125:         cadd('static void f2py_init_%s(void) {' % name)
126:         cadd('\t%s(f2pyinit%s,F2PYINIT%s)(f2py_setup_%s);'
127:              % (F_FUNC, lower_name, name.upper(), name))
128:         cadd('}\n')
129:         iadd('\tF2PyDict_SetItemString(d, \"%s\", PyFortranObject_New(f2py_%s_def,f2py_init_%s));' % (
130:             name, name, name))
131:         tname = name.replace('_', '\\_')
132:         dadd('\\subsection{Common block \\texttt{%s}}\n' % (tname))
133:         dadd('\\begin{description}')
134:         for n in inames:
135:             dadd('\\item[]{{}\\verb@%s@{}}' %
136:                  (capi_maps.getarrdocsign(n, vars[n])))
137:             if hasnote(vars[n]):
138:                 note = vars[n]['note']
139:                 if isinstance(note, list):
140:                     note = '\n'.join(note)
141:                 dadd('--- %s' % (note))
142:         dadd('\\end{description}')
143:         ret['docs'].append(
144:             '"\t/%s/ %s\\n"' % (name, ','.join(map(lambda v, d: v + d, inames, idims))))
145:     ret['commonhooks'] = chooks
146:     ret['initcommonhooks'] = ihooks
147:     ret['latexdoc'] = doc[0]
148:     if len(ret['docs']) <= 1:
149:         ret['docs'] = ''
150:     return ret, fwrap[0]
151: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_75155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, (-1)), 'str', '\n\nBuild common block mechanism for f2py2e.\n\nCopyright 2000 Pearu Peterson all rights reserved,\nPearu Peterson <pearu@ioc.ee>\nPermission to use, modify, and distribute this software is given under the\nterms of the NumPy License\n\nNO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.\n$Date: 2005/05/06 10:57:33 $\nPearu Peterson\n\n')

# Assigning a Subscript to a Name (line 18):

# Assigning a Subscript to a Name (line 18):

# Obtaining the type of the subscript
int_75156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 34), 'int')
int_75157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 37), 'int')
slice_75158 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 18, 14), int_75156, int_75157, None)
str_75159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 14), 'str', '$Revision: 1.19 $')
# Obtaining the member '__getitem__' of a type (line 18)
getitem___75160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 14), str_75159, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 18)
subscript_call_result_75161 = invoke(stypy.reporting.localization.Localization(__file__, 18, 14), getitem___75160, slice_75158)

# Assigning a type to the variable '__version__' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), '__version__', subscript_call_result_75161)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from numpy.f2py import __version__' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_75162 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.f2py')

if (type(import_75162) is not StypyTypeError):

    if (import_75162 != 'pyd_module'):
        __import__(import_75162)
        sys_modules_75163 = sys.modules[import_75162]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.f2py', sys_modules_75163.module_type_store, module_type_store, ['__version__'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_75163, sys_modules_75163.module_type_store, module_type_store)
    else:
        from numpy.f2py import __version__

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.f2py', None, module_type_store, ['__version__'], [__version__])

else:
    # Assigning a type to the variable 'numpy.f2py' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.f2py', import_75162)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')


# Assigning a Attribute to a Name (line 21):

# Assigning a Attribute to a Name (line 21):
# Getting the type of '__version__' (line 21)
version___75164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), '__version__')
# Obtaining the member 'version' of a type (line 21)
version_75165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 15), version___75164, 'version')
# Assigning a type to the variable 'f2py_version' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'f2py_version', version_75165)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from numpy.f2py.auxfuncs import hasbody, hascommon, hasnote, isintent_hide, outmess' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_75166 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.f2py.auxfuncs')

if (type(import_75166) is not StypyTypeError):

    if (import_75166 != 'pyd_module'):
        __import__(import_75166)
        sys_modules_75167 = sys.modules[import_75166]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.f2py.auxfuncs', sys_modules_75167.module_type_store, module_type_store, ['hasbody', 'hascommon', 'hasnote', 'isintent_hide', 'outmess'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 0), __file__, sys_modules_75167, sys_modules_75167.module_type_store, module_type_store)
    else:
        from numpy.f2py.auxfuncs import hasbody, hascommon, hasnote, isintent_hide, outmess

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.f2py.auxfuncs', None, module_type_store, ['hasbody', 'hascommon', 'hasnote', 'isintent_hide', 'outmess'], [hasbody, hascommon, hasnote, isintent_hide, outmess])

else:
    # Assigning a type to the variable 'numpy.f2py.auxfuncs' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.f2py.auxfuncs', import_75166)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from numpy.f2py import capi_maps' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_75168 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.f2py')

if (type(import_75168) is not StypyTypeError):

    if (import_75168 != 'pyd_module'):
        __import__(import_75168)
        sys_modules_75169 = sys.modules[import_75168]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.f2py', sys_modules_75169.module_type_store, module_type_store, ['capi_maps'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_75169, sys_modules_75169.module_type_store, module_type_store)
    else:
        from numpy.f2py import capi_maps

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.f2py', None, module_type_store, ['capi_maps'], [capi_maps])

else:
    # Assigning a type to the variable 'numpy.f2py' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.f2py', import_75168)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'from numpy.f2py import func2subr' statement (line 27)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_75170 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.f2py')

if (type(import_75170) is not StypyTypeError):

    if (import_75170 != 'pyd_module'):
        __import__(import_75170)
        sys_modules_75171 = sys.modules[import_75170]
        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.f2py', sys_modules_75171.module_type_store, module_type_store, ['func2subr'])
        nest_module(stypy.reporting.localization.Localization(__file__, 27, 0), __file__, sys_modules_75171, sys_modules_75171.module_type_store, module_type_store)
    else:
        from numpy.f2py import func2subr

        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.f2py', None, module_type_store, ['func2subr'], [func2subr])

else:
    # Assigning a type to the variable 'numpy.f2py' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.f2py', import_75170)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'from numpy.f2py.crackfortran import rmbadname' statement (line 28)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_75172 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'numpy.f2py.crackfortran')

if (type(import_75172) is not StypyTypeError):

    if (import_75172 != 'pyd_module'):
        __import__(import_75172)
        sys_modules_75173 = sys.modules[import_75172]
        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'numpy.f2py.crackfortran', sys_modules_75173.module_type_store, module_type_store, ['rmbadname'])
        nest_module(stypy.reporting.localization.Localization(__file__, 28, 0), __file__, sys_modules_75173, sys_modules_75173.module_type_store, module_type_store)
    else:
        from numpy.f2py.crackfortran import rmbadname

        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'numpy.f2py.crackfortran', None, module_type_store, ['rmbadname'], [rmbadname])

else:
    # Assigning a type to the variable 'numpy.f2py.crackfortran' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'numpy.f2py.crackfortran', import_75172)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')


@norecursion
def findcommonblocks(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_75174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 32), 'int')
    defaults = [int_75174]
    # Create a new context for function 'findcommonblocks'
    module_type_store = module_type_store.open_function_context('findcommonblocks', 31, 0, False)
    
    # Passed parameters checking function
    findcommonblocks.stypy_localization = localization
    findcommonblocks.stypy_type_of_self = None
    findcommonblocks.stypy_type_store = module_type_store
    findcommonblocks.stypy_function_name = 'findcommonblocks'
    findcommonblocks.stypy_param_names_list = ['block', 'top']
    findcommonblocks.stypy_varargs_param_name = None
    findcommonblocks.stypy_kwargs_param_name = None
    findcommonblocks.stypy_call_defaults = defaults
    findcommonblocks.stypy_call_varargs = varargs
    findcommonblocks.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'findcommonblocks', ['block', 'top'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'findcommonblocks', localization, ['block', 'top'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'findcommonblocks(...)' code ##################

    
    # Assigning a List to a Name (line 32):
    
    # Assigning a List to a Name (line 32):
    
    # Obtaining an instance of the builtin type 'list' (line 32)
    list_75175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 32)
    
    # Assigning a type to the variable 'ret' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'ret', list_75175)
    
    
    # Call to hascommon(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'block' (line 33)
    block_75177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 17), 'block', False)
    # Processing the call keyword arguments (line 33)
    kwargs_75178 = {}
    # Getting the type of 'hascommon' (line 33)
    hascommon_75176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 7), 'hascommon', False)
    # Calling hascommon(args, kwargs) (line 33)
    hascommon_call_result_75179 = invoke(stypy.reporting.localization.Localization(__file__, 33, 7), hascommon_75176, *[block_75177], **kwargs_75178)
    
    # Testing the type of an if condition (line 33)
    if_condition_75180 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 4), hascommon_call_result_75179)
    # Assigning a type to the variable 'if_condition_75180' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'if_condition_75180', if_condition_75180)
    # SSA begins for if statement (line 33)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to keys(...): (line 34)
    # Processing the call keyword arguments (line 34)
    kwargs_75186 = {}
    
    # Obtaining the type of the subscript
    str_75181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 23), 'str', 'common')
    # Getting the type of 'block' (line 34)
    block_75182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 17), 'block', False)
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___75183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 17), block_75182, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_75184 = invoke(stypy.reporting.localization.Localization(__file__, 34, 17), getitem___75183, str_75181)
    
    # Obtaining the member 'keys' of a type (line 34)
    keys_75185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 17), subscript_call_result_75184, 'keys')
    # Calling keys(args, kwargs) (line 34)
    keys_call_result_75187 = invoke(stypy.reporting.localization.Localization(__file__, 34, 17), keys_75185, *[], **kwargs_75186)
    
    # Testing the type of a for loop iterable (line 34)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 34, 8), keys_call_result_75187)
    # Getting the type of the for loop variable (line 34)
    for_loop_var_75188 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 34, 8), keys_call_result_75187)
    # Assigning a type to the variable 'n' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'n', for_loop_var_75188)
    # SSA begins for a for statement (line 34)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Dict to a Name (line 35):
    
    # Assigning a Dict to a Name (line 35):
    
    # Obtaining an instance of the builtin type 'dict' (line 35)
    dict_75189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 19), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 35)
    
    # Assigning a type to the variable 'vars' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'vars', dict_75189)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 36)
    n_75190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 37), 'n')
    
    # Obtaining the type of the subscript
    str_75191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 27), 'str', 'common')
    # Getting the type of 'block' (line 36)
    block_75192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 21), 'block')
    # Obtaining the member '__getitem__' of a type (line 36)
    getitem___75193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 21), block_75192, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 36)
    subscript_call_result_75194 = invoke(stypy.reporting.localization.Localization(__file__, 36, 21), getitem___75193, str_75191)
    
    # Obtaining the member '__getitem__' of a type (line 36)
    getitem___75195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 21), subscript_call_result_75194, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 36)
    subscript_call_result_75196 = invoke(stypy.reporting.localization.Localization(__file__, 36, 21), getitem___75195, n_75190)
    
    # Testing the type of a for loop iterable (line 36)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 36, 12), subscript_call_result_75196)
    # Getting the type of the for loop variable (line 36)
    for_loop_var_75197 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 36, 12), subscript_call_result_75196)
    # Assigning a type to the variable 'v' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'v', for_loop_var_75197)
    # SSA begins for a for statement (line 36)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Subscript (line 37):
    
    # Assigning a Subscript to a Subscript (line 37):
    
    # Obtaining the type of the subscript
    # Getting the type of 'v' (line 37)
    v_75198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 40), 'v')
    
    # Obtaining the type of the subscript
    str_75199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 32), 'str', 'vars')
    # Getting the type of 'block' (line 37)
    block_75200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 26), 'block')
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___75201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 26), block_75200, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_75202 = invoke(stypy.reporting.localization.Localization(__file__, 37, 26), getitem___75201, str_75199)
    
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___75203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 26), subscript_call_result_75202, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_75204 = invoke(stypy.reporting.localization.Localization(__file__, 37, 26), getitem___75203, v_75198)
    
    # Getting the type of 'vars' (line 37)
    vars_75205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 16), 'vars')
    # Getting the type of 'v' (line 37)
    v_75206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 21), 'v')
    # Storing an element on a container (line 37)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 16), vars_75205, (v_75206, subscript_call_result_75204))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 38)
    # Processing the call arguments (line 38)
    
    # Obtaining an instance of the builtin type 'tuple' (line 38)
    tuple_75209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 38)
    # Adding element type (line 38)
    # Getting the type of 'n' (line 38)
    n_75210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 24), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 24), tuple_75209, n_75210)
    # Adding element type (line 38)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 38)
    n_75211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 43), 'n', False)
    
    # Obtaining the type of the subscript
    str_75212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 33), 'str', 'common')
    # Getting the type of 'block' (line 38)
    block_75213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 27), 'block', False)
    # Obtaining the member '__getitem__' of a type (line 38)
    getitem___75214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 27), block_75213, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 38)
    subscript_call_result_75215 = invoke(stypy.reporting.localization.Localization(__file__, 38, 27), getitem___75214, str_75212)
    
    # Obtaining the member '__getitem__' of a type (line 38)
    getitem___75216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 27), subscript_call_result_75215, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 38)
    subscript_call_result_75217 = invoke(stypy.reporting.localization.Localization(__file__, 38, 27), getitem___75216, n_75211)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 24), tuple_75209, subscript_call_result_75217)
    # Adding element type (line 38)
    # Getting the type of 'vars' (line 38)
    vars_75218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 47), 'vars', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 24), tuple_75209, vars_75218)
    
    # Processing the call keyword arguments (line 38)
    kwargs_75219 = {}
    # Getting the type of 'ret' (line 38)
    ret_75207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'ret', False)
    # Obtaining the member 'append' of a type (line 38)
    append_75208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 12), ret_75207, 'append')
    # Calling append(args, kwargs) (line 38)
    append_call_result_75220 = invoke(stypy.reporting.localization.Localization(__file__, 38, 12), append_75208, *[tuple_75209], **kwargs_75219)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 33)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to hasbody(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'block' (line 39)
    block_75222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 17), 'block', False)
    # Processing the call keyword arguments (line 39)
    kwargs_75223 = {}
    # Getting the type of 'hasbody' (line 39)
    hasbody_75221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 9), 'hasbody', False)
    # Calling hasbody(args, kwargs) (line 39)
    hasbody_call_result_75224 = invoke(stypy.reporting.localization.Localization(__file__, 39, 9), hasbody_75221, *[block_75222], **kwargs_75223)
    
    # Testing the type of an if condition (line 39)
    if_condition_75225 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 9), hasbody_call_result_75224)
    # Assigning a type to the variable 'if_condition_75225' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 9), 'if_condition_75225', if_condition_75225)
    # SSA begins for if statement (line 39)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Obtaining the type of the subscript
    str_75226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 23), 'str', 'body')
    # Getting the type of 'block' (line 40)
    block_75227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 17), 'block')
    # Obtaining the member '__getitem__' of a type (line 40)
    getitem___75228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 17), block_75227, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
    subscript_call_result_75229 = invoke(stypy.reporting.localization.Localization(__file__, 40, 17), getitem___75228, str_75226)
    
    # Testing the type of a for loop iterable (line 40)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 40, 8), subscript_call_result_75229)
    # Getting the type of the for loop variable (line 40)
    for_loop_var_75230 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 40, 8), subscript_call_result_75229)
    # Assigning a type to the variable 'b' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'b', for_loop_var_75230)
    # SSA begins for a for statement (line 40)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 41):
    
    # Assigning a BinOp to a Name (line 41):
    # Getting the type of 'ret' (line 41)
    ret_75231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 18), 'ret')
    
    # Call to findcommonblocks(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'b' (line 41)
    b_75233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 41), 'b', False)
    int_75234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 44), 'int')
    # Processing the call keyword arguments (line 41)
    kwargs_75235 = {}
    # Getting the type of 'findcommonblocks' (line 41)
    findcommonblocks_75232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 24), 'findcommonblocks', False)
    # Calling findcommonblocks(args, kwargs) (line 41)
    findcommonblocks_call_result_75236 = invoke(stypy.reporting.localization.Localization(__file__, 41, 24), findcommonblocks_75232, *[b_75233, int_75234], **kwargs_75235)
    
    # Applying the binary operator '+' (line 41)
    result_add_75237 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 18), '+', ret_75231, findcommonblocks_call_result_75236)
    
    # Assigning a type to the variable 'ret' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'ret', result_add_75237)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 39)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 33)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'top' (line 42)
    top_75238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 7), 'top')
    # Testing the type of an if condition (line 42)
    if_condition_75239 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 4), top_75238)
    # Assigning a type to the variable 'if_condition_75239' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'if_condition_75239', if_condition_75239)
    # SSA begins for if statement (line 42)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 43):
    
    # Assigning a List to a Name (line 43):
    
    # Obtaining an instance of the builtin type 'list' (line 43)
    list_75240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 43)
    
    # Assigning a type to the variable 'tret' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'tret', list_75240)
    
    # Assigning a List to a Name (line 44):
    
    # Assigning a List to a Name (line 44):
    
    # Obtaining an instance of the builtin type 'list' (line 44)
    list_75241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 44)
    
    # Assigning a type to the variable 'names' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'names', list_75241)
    
    # Getting the type of 'ret' (line 45)
    ret_75242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 17), 'ret')
    # Testing the type of a for loop iterable (line 45)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 45, 8), ret_75242)
    # Getting the type of the for loop variable (line 45)
    for_loop_var_75243 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 45, 8), ret_75242)
    # Assigning a type to the variable 't' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 't', for_loop_var_75243)
    # SSA begins for a for statement (line 45)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    int_75244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 17), 'int')
    # Getting the type of 't' (line 46)
    t_75245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 15), 't')
    # Obtaining the member '__getitem__' of a type (line 46)
    getitem___75246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 15), t_75245, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 46)
    subscript_call_result_75247 = invoke(stypy.reporting.localization.Localization(__file__, 46, 15), getitem___75246, int_75244)
    
    # Getting the type of 'names' (line 46)
    names_75248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 27), 'names')
    # Applying the binary operator 'notin' (line 46)
    result_contains_75249 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 15), 'notin', subscript_call_result_75247, names_75248)
    
    # Testing the type of an if condition (line 46)
    if_condition_75250 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 12), result_contains_75249)
    # Assigning a type to the variable 'if_condition_75250' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'if_condition_75250', if_condition_75250)
    # SSA begins for if statement (line 46)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 47)
    # Processing the call arguments (line 47)
    
    # Obtaining the type of the subscript
    int_75253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 31), 'int')
    # Getting the type of 't' (line 47)
    t_75254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 29), 't', False)
    # Obtaining the member '__getitem__' of a type (line 47)
    getitem___75255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 29), t_75254, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 47)
    subscript_call_result_75256 = invoke(stypy.reporting.localization.Localization(__file__, 47, 29), getitem___75255, int_75253)
    
    # Processing the call keyword arguments (line 47)
    kwargs_75257 = {}
    # Getting the type of 'names' (line 47)
    names_75251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'names', False)
    # Obtaining the member 'append' of a type (line 47)
    append_75252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 16), names_75251, 'append')
    # Calling append(args, kwargs) (line 47)
    append_call_result_75258 = invoke(stypy.reporting.localization.Localization(__file__, 47, 16), append_75252, *[subscript_call_result_75256], **kwargs_75257)
    
    
    # Call to append(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 't' (line 48)
    t_75261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 28), 't', False)
    # Processing the call keyword arguments (line 48)
    kwargs_75262 = {}
    # Getting the type of 'tret' (line 48)
    tret_75259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'tret', False)
    # Obtaining the member 'append' of a type (line 48)
    append_75260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 16), tret_75259, 'append')
    # Calling append(args, kwargs) (line 48)
    append_call_result_75263 = invoke(stypy.reporting.localization.Localization(__file__, 48, 16), append_75260, *[t_75261], **kwargs_75262)
    
    # SSA join for if statement (line 46)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'tret' (line 49)
    tret_75264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'tret')
    # Assigning a type to the variable 'stypy_return_type' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'stypy_return_type', tret_75264)
    # SSA join for if statement (line 42)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ret' (line 50)
    ret_75265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type', ret_75265)
    
    # ################# End of 'findcommonblocks(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'findcommonblocks' in the type store
    # Getting the type of 'stypy_return_type' (line 31)
    stypy_return_type_75266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_75266)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'findcommonblocks'
    return stypy_return_type_75266

# Assigning a type to the variable 'findcommonblocks' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'findcommonblocks', findcommonblocks)

@norecursion
def buildhooks(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'buildhooks'
    module_type_store = module_type_store.open_function_context('buildhooks', 53, 0, False)
    
    # Passed parameters checking function
    buildhooks.stypy_localization = localization
    buildhooks.stypy_type_of_self = None
    buildhooks.stypy_type_store = module_type_store
    buildhooks.stypy_function_name = 'buildhooks'
    buildhooks.stypy_param_names_list = ['m']
    buildhooks.stypy_varargs_param_name = None
    buildhooks.stypy_kwargs_param_name = None
    buildhooks.stypy_call_defaults = defaults
    buildhooks.stypy_call_varargs = varargs
    buildhooks.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'buildhooks', ['m'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'buildhooks', localization, ['m'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'buildhooks(...)' code ##################

    
    # Assigning a Dict to a Name (line 54):
    
    # Assigning a Dict to a Name (line 54):
    
    # Obtaining an instance of the builtin type 'dict' (line 54)
    dict_75267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 10), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 54)
    # Adding element type (key, value) (line 54)
    str_75268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 11), 'str', 'commonhooks')
    
    # Obtaining an instance of the builtin type 'list' (line 54)
    list_75269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 54)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 10), dict_75267, (str_75268, list_75269))
    # Adding element type (key, value) (line 54)
    str_75270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 30), 'str', 'initcommonhooks')
    
    # Obtaining an instance of the builtin type 'list' (line 54)
    list_75271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 49), 'list')
    # Adding type elements to the builtin type 'list' instance (line 54)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 10), dict_75267, (str_75270, list_75271))
    # Adding element type (key, value) (line 54)
    str_75272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 11), 'str', 'docs')
    
    # Obtaining an instance of the builtin type 'list' (line 55)
    list_75273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 55)
    # Adding element type (line 55)
    str_75274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 20), 'str', '"COMMON blocks:\\n"')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 19), list_75273, str_75274)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 10), dict_75267, (str_75272, list_75273))
    
    # Assigning a type to the variable 'ret' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'ret', dict_75267)
    
    # Assigning a List to a Name (line 56):
    
    # Assigning a List to a Name (line 56):
    
    # Obtaining an instance of the builtin type 'list' (line 56)
    list_75275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 56)
    # Adding element type (line 56)
    str_75276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 13), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 12), list_75275, str_75276)
    
    # Assigning a type to the variable 'fwrap' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'fwrap', list_75275)

    @norecursion
    def fadd(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'fwrap' (line 58)
        fwrap_75277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 21), 'fwrap')
        defaults = [fwrap_75277]
        # Create a new context for function 'fadd'
        module_type_store = module_type_store.open_function_context('fadd', 58, 4, False)
        
        # Passed parameters checking function
        fadd.stypy_localization = localization
        fadd.stypy_type_of_self = None
        fadd.stypy_type_store = module_type_store
        fadd.stypy_function_name = 'fadd'
        fadd.stypy_param_names_list = ['line', 's']
        fadd.stypy_varargs_param_name = None
        fadd.stypy_kwargs_param_name = None
        fadd.stypy_call_defaults = defaults
        fadd.stypy_call_varargs = varargs
        fadd.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'fadd', ['line', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fadd', localization, ['line', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fadd(...)' code ##################

        
        # Assigning a BinOp to a Subscript (line 59):
        
        # Assigning a BinOp to a Subscript (line 59):
        str_75278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 15), 'str', '%s\n      %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 59)
        tuple_75279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 59)
        # Adding element type (line 59)
        
        # Obtaining the type of the subscript
        int_75280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 35), 'int')
        # Getting the type of 's' (line 59)
        s_75281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 33), 's')
        # Obtaining the member '__getitem__' of a type (line 59)
        getitem___75282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 33), s_75281, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 59)
        subscript_call_result_75283 = invoke(stypy.reporting.localization.Localization(__file__, 59, 33), getitem___75282, int_75280)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 33), tuple_75279, subscript_call_result_75283)
        # Adding element type (line 59)
        # Getting the type of 'line' (line 59)
        line_75284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 39), 'line')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 33), tuple_75279, line_75284)
        
        # Applying the binary operator '%' (line 59)
        result_mod_75285 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 15), '%', str_75278, tuple_75279)
        
        # Getting the type of 's' (line 59)
        s_75286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 's')
        int_75287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 10), 'int')
        # Storing an element on a container (line 59)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 8), s_75286, (int_75287, result_mod_75285))
        
        # ################# End of 'fadd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fadd' in the type store
        # Getting the type of 'stypy_return_type' (line 58)
        stypy_return_type_75288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_75288)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fadd'
        return stypy_return_type_75288

    # Assigning a type to the variable 'fadd' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'fadd', fadd)
    
    # Assigning a List to a Name (line 60):
    
    # Assigning a List to a Name (line 60):
    
    # Obtaining an instance of the builtin type 'list' (line 60)
    list_75289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 60)
    # Adding element type (line 60)
    str_75290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 14), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 13), list_75289, str_75290)
    
    # Assigning a type to the variable 'chooks' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'chooks', list_75289)

    @norecursion
    def cadd(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'chooks' (line 62)
        chooks_75291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 21), 'chooks')
        defaults = [chooks_75291]
        # Create a new context for function 'cadd'
        module_type_store = module_type_store.open_function_context('cadd', 62, 4, False)
        
        # Passed parameters checking function
        cadd.stypy_localization = localization
        cadd.stypy_type_of_self = None
        cadd.stypy_type_store = module_type_store
        cadd.stypy_function_name = 'cadd'
        cadd.stypy_param_names_list = ['line', 's']
        cadd.stypy_varargs_param_name = None
        cadd.stypy_kwargs_param_name = None
        cadd.stypy_call_defaults = defaults
        cadd.stypy_call_varargs = varargs
        cadd.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'cadd', ['line', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cadd', localization, ['line', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cadd(...)' code ##################

        
        # Assigning a BinOp to a Subscript (line 63):
        
        # Assigning a BinOp to a Subscript (line 63):
        str_75292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 15), 'str', '%s\n%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 63)
        tuple_75293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 63)
        # Adding element type (line 63)
        
        # Obtaining the type of the subscript
        int_75294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 29), 'int')
        # Getting the type of 's' (line 63)
        s_75295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 27), 's')
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___75296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 27), s_75295, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_75297 = invoke(stypy.reporting.localization.Localization(__file__, 63, 27), getitem___75296, int_75294)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 27), tuple_75293, subscript_call_result_75297)
        # Adding element type (line 63)
        # Getting the type of 'line' (line 63)
        line_75298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 33), 'line')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 27), tuple_75293, line_75298)
        
        # Applying the binary operator '%' (line 63)
        result_mod_75299 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 15), '%', str_75292, tuple_75293)
        
        # Getting the type of 's' (line 63)
        s_75300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 's')
        int_75301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 10), 'int')
        # Storing an element on a container (line 63)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 8), s_75300, (int_75301, result_mod_75299))
        
        # ################# End of 'cadd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cadd' in the type store
        # Getting the type of 'stypy_return_type' (line 62)
        stypy_return_type_75302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_75302)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cadd'
        return stypy_return_type_75302

    # Assigning a type to the variable 'cadd' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'cadd', cadd)
    
    # Assigning a List to a Name (line 64):
    
    # Assigning a List to a Name (line 64):
    
    # Obtaining an instance of the builtin type 'list' (line 64)
    list_75303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 64)
    # Adding element type (line 64)
    str_75304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 14), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 13), list_75303, str_75304)
    
    # Assigning a type to the variable 'ihooks' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'ihooks', list_75303)

    @norecursion
    def iadd(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'ihooks' (line 66)
        ihooks_75305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 21), 'ihooks')
        defaults = [ihooks_75305]
        # Create a new context for function 'iadd'
        module_type_store = module_type_store.open_function_context('iadd', 66, 4, False)
        
        # Passed parameters checking function
        iadd.stypy_localization = localization
        iadd.stypy_type_of_self = None
        iadd.stypy_type_store = module_type_store
        iadd.stypy_function_name = 'iadd'
        iadd.stypy_param_names_list = ['line', 's']
        iadd.stypy_varargs_param_name = None
        iadd.stypy_kwargs_param_name = None
        iadd.stypy_call_defaults = defaults
        iadd.stypy_call_varargs = varargs
        iadd.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'iadd', ['line', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'iadd', localization, ['line', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'iadd(...)' code ##################

        
        # Assigning a BinOp to a Subscript (line 67):
        
        # Assigning a BinOp to a Subscript (line 67):
        str_75306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 15), 'str', '%s\n%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 67)
        tuple_75307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 67)
        # Adding element type (line 67)
        
        # Obtaining the type of the subscript
        int_75308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 29), 'int')
        # Getting the type of 's' (line 67)
        s_75309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 27), 's')
        # Obtaining the member '__getitem__' of a type (line 67)
        getitem___75310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 27), s_75309, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 67)
        subscript_call_result_75311 = invoke(stypy.reporting.localization.Localization(__file__, 67, 27), getitem___75310, int_75308)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 27), tuple_75307, subscript_call_result_75311)
        # Adding element type (line 67)
        # Getting the type of 'line' (line 67)
        line_75312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 33), 'line')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 27), tuple_75307, line_75312)
        
        # Applying the binary operator '%' (line 67)
        result_mod_75313 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 15), '%', str_75306, tuple_75307)
        
        # Getting the type of 's' (line 67)
        s_75314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 's')
        int_75315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 10), 'int')
        # Storing an element on a container (line 67)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 8), s_75314, (int_75315, result_mod_75313))
        
        # ################# End of 'iadd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'iadd' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_75316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_75316)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'iadd'
        return stypy_return_type_75316

    # Assigning a type to the variable 'iadd' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'iadd', iadd)
    
    # Assigning a List to a Name (line 68):
    
    # Assigning a List to a Name (line 68):
    
    # Obtaining an instance of the builtin type 'list' (line 68)
    list_75317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 68)
    # Adding element type (line 68)
    str_75318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 11), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 10), list_75317, str_75318)
    
    # Assigning a type to the variable 'doc' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'doc', list_75317)

    @norecursion
    def dadd(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'doc' (line 70)
        doc_75319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 21), 'doc')
        defaults = [doc_75319]
        # Create a new context for function 'dadd'
        module_type_store = module_type_store.open_function_context('dadd', 70, 4, False)
        
        # Passed parameters checking function
        dadd.stypy_localization = localization
        dadd.stypy_type_of_self = None
        dadd.stypy_type_store = module_type_store
        dadd.stypy_function_name = 'dadd'
        dadd.stypy_param_names_list = ['line', 's']
        dadd.stypy_varargs_param_name = None
        dadd.stypy_kwargs_param_name = None
        dadd.stypy_call_defaults = defaults
        dadd.stypy_call_varargs = varargs
        dadd.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'dadd', ['line', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dadd', localization, ['line', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dadd(...)' code ##################

        
        # Assigning a BinOp to a Subscript (line 71):
        
        # Assigning a BinOp to a Subscript (line 71):
        str_75320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 15), 'str', '%s\n%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 71)
        tuple_75321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 71)
        # Adding element type (line 71)
        
        # Obtaining the type of the subscript
        int_75322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 29), 'int')
        # Getting the type of 's' (line 71)
        s_75323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 27), 's')
        # Obtaining the member '__getitem__' of a type (line 71)
        getitem___75324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 27), s_75323, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 71)
        subscript_call_result_75325 = invoke(stypy.reporting.localization.Localization(__file__, 71, 27), getitem___75324, int_75322)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 27), tuple_75321, subscript_call_result_75325)
        # Adding element type (line 71)
        # Getting the type of 'line' (line 71)
        line_75326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 33), 'line')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 27), tuple_75321, line_75326)
        
        # Applying the binary operator '%' (line 71)
        result_mod_75327 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 15), '%', str_75320, tuple_75321)
        
        # Getting the type of 's' (line 71)
        s_75328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 's')
        int_75329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 10), 'int')
        # Storing an element on a container (line 71)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 8), s_75328, (int_75329, result_mod_75327))
        
        # ################# End of 'dadd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dadd' in the type store
        # Getting the type of 'stypy_return_type' (line 70)
        stypy_return_type_75330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_75330)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dadd'
        return stypy_return_type_75330

    # Assigning a type to the variable 'dadd' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'dadd', dadd)
    
    
    # Call to findcommonblocks(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'm' (line 72)
    m_75332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 49), 'm', False)
    # Processing the call keyword arguments (line 72)
    kwargs_75333 = {}
    # Getting the type of 'findcommonblocks' (line 72)
    findcommonblocks_75331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 32), 'findcommonblocks', False)
    # Calling findcommonblocks(args, kwargs) (line 72)
    findcommonblocks_call_result_75334 = invoke(stypy.reporting.localization.Localization(__file__, 72, 32), findcommonblocks_75331, *[m_75332], **kwargs_75333)
    
    # Testing the type of a for loop iterable (line 72)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 72, 4), findcommonblocks_call_result_75334)
    # Getting the type of the for loop variable (line 72)
    for_loop_var_75335 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 72, 4), findcommonblocks_call_result_75334)
    # Assigning a type to the variable 'name' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 4), for_loop_var_75335))
    # Assigning a type to the variable 'vnames' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'vnames', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 4), for_loop_var_75335))
    # Assigning a type to the variable 'vars' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'vars', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 4), for_loop_var_75335))
    # SSA begins for a for statement (line 72)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 73):
    
    # Assigning a Call to a Name (line 73):
    
    # Call to lower(...): (line 73)
    # Processing the call keyword arguments (line 73)
    kwargs_75338 = {}
    # Getting the type of 'name' (line 73)
    name_75336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 21), 'name', False)
    # Obtaining the member 'lower' of a type (line 73)
    lower_75337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 21), name_75336, 'lower')
    # Calling lower(args, kwargs) (line 73)
    lower_call_result_75339 = invoke(stypy.reporting.localization.Localization(__file__, 73, 21), lower_75337, *[], **kwargs_75338)
    
    # Assigning a type to the variable 'lower_name' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'lower_name', lower_call_result_75339)
    
    # Assigning a Tuple to a Tuple (line 74):
    
    # Assigning a List to a Name (line 74):
    
    # Obtaining an instance of the builtin type 'list' (line 74)
    list_75340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 74)
    
    # Assigning a type to the variable 'tuple_assignment_75153' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'tuple_assignment_75153', list_75340)
    
    # Assigning a List to a Name (line 74):
    
    # Obtaining an instance of the builtin type 'list' (line 74)
    list_75341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 74)
    
    # Assigning a type to the variable 'tuple_assignment_75154' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'tuple_assignment_75154', list_75341)
    
    # Assigning a Name to a Name (line 74):
    # Getting the type of 'tuple_assignment_75153' (line 74)
    tuple_assignment_75153_75342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'tuple_assignment_75153')
    # Assigning a type to the variable 'hnames' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'hnames', tuple_assignment_75153_75342)
    
    # Assigning a Name to a Name (line 74):
    # Getting the type of 'tuple_assignment_75154' (line 74)
    tuple_assignment_75154_75343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'tuple_assignment_75154')
    # Assigning a type to the variable 'inames' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'inames', tuple_assignment_75154_75343)
    
    # Getting the type of 'vnames' (line 75)
    vnames_75344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 17), 'vnames')
    # Testing the type of a for loop iterable (line 75)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 75, 8), vnames_75344)
    # Getting the type of the for loop variable (line 75)
    for_loop_var_75345 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 75, 8), vnames_75344)
    # Assigning a type to the variable 'n' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'n', for_loop_var_75345)
    # SSA begins for a for statement (line 75)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to isintent_hide(...): (line 76)
    # Processing the call arguments (line 76)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 76)
    n_75347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 34), 'n', False)
    # Getting the type of 'vars' (line 76)
    vars_75348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 29), 'vars', False)
    # Obtaining the member '__getitem__' of a type (line 76)
    getitem___75349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 29), vars_75348, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 76)
    subscript_call_result_75350 = invoke(stypy.reporting.localization.Localization(__file__, 76, 29), getitem___75349, n_75347)
    
    # Processing the call keyword arguments (line 76)
    kwargs_75351 = {}
    # Getting the type of 'isintent_hide' (line 76)
    isintent_hide_75346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'isintent_hide', False)
    # Calling isintent_hide(args, kwargs) (line 76)
    isintent_hide_call_result_75352 = invoke(stypy.reporting.localization.Localization(__file__, 76, 15), isintent_hide_75346, *[subscript_call_result_75350], **kwargs_75351)
    
    # Testing the type of an if condition (line 76)
    if_condition_75353 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 12), isintent_hide_call_result_75352)
    # Assigning a type to the variable 'if_condition_75353' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'if_condition_75353', if_condition_75353)
    # SSA begins for if statement (line 76)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'n' (line 77)
    n_75356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 30), 'n', False)
    # Processing the call keyword arguments (line 77)
    kwargs_75357 = {}
    # Getting the type of 'hnames' (line 77)
    hnames_75354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 'hnames', False)
    # Obtaining the member 'append' of a type (line 77)
    append_75355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 16), hnames_75354, 'append')
    # Calling append(args, kwargs) (line 77)
    append_call_result_75358 = invoke(stypy.reporting.localization.Localization(__file__, 77, 16), append_75355, *[n_75356], **kwargs_75357)
    
    # SSA branch for the else part of an if statement (line 76)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'n' (line 79)
    n_75361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 30), 'n', False)
    # Processing the call keyword arguments (line 79)
    kwargs_75362 = {}
    # Getting the type of 'inames' (line 79)
    inames_75359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 16), 'inames', False)
    # Obtaining the member 'append' of a type (line 79)
    append_75360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 16), inames_75359, 'append')
    # Calling append(args, kwargs) (line 79)
    append_call_result_75363 = invoke(stypy.reporting.localization.Localization(__file__, 79, 16), append_75360, *[n_75361], **kwargs_75362)
    
    # SSA join for if statement (line 76)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'hnames' (line 80)
    hnames_75364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'hnames')
    # Testing the type of an if condition (line 80)
    if_condition_75365 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 8), hnames_75364)
    # Assigning a type to the variable 'if_condition_75365' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'if_condition_75365', if_condition_75365)
    # SSA begins for if statement (line 80)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to outmess(...): (line 81)
    # Processing the call arguments (line 81)
    str_75367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 20), 'str', '\t\tConstructing COMMON block support for "%s"...\n\t\t  %s\n\t\t  Hidden: %s\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 82)
    tuple_75368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 82)
    # Adding element type (line 82)
    # Getting the type of 'name' (line 82)
    name_75369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 16), tuple_75368, name_75369)
    # Adding element type (line 82)
    
    # Call to join(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'inames' (line 82)
    inames_75372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 31), 'inames', False)
    # Processing the call keyword arguments (line 82)
    kwargs_75373 = {}
    str_75370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 22), 'str', ',')
    # Obtaining the member 'join' of a type (line 82)
    join_75371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 22), str_75370, 'join')
    # Calling join(args, kwargs) (line 82)
    join_call_result_75374 = invoke(stypy.reporting.localization.Localization(__file__, 82, 22), join_75371, *[inames_75372], **kwargs_75373)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 16), tuple_75368, join_call_result_75374)
    # Adding element type (line 82)
    
    # Call to join(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'hnames' (line 82)
    hnames_75377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 49), 'hnames', False)
    # Processing the call keyword arguments (line 82)
    kwargs_75378 = {}
    str_75375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 40), 'str', ',')
    # Obtaining the member 'join' of a type (line 82)
    join_75376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 40), str_75375, 'join')
    # Calling join(args, kwargs) (line 82)
    join_call_result_75379 = invoke(stypy.reporting.localization.Localization(__file__, 82, 40), join_75376, *[hnames_75377], **kwargs_75378)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 16), tuple_75368, join_call_result_75379)
    
    # Applying the binary operator '%' (line 81)
    result_mod_75380 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 20), '%', str_75367, tuple_75368)
    
    # Processing the call keyword arguments (line 81)
    kwargs_75381 = {}
    # Getting the type of 'outmess' (line 81)
    outmess_75366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'outmess', False)
    # Calling outmess(args, kwargs) (line 81)
    outmess_call_result_75382 = invoke(stypy.reporting.localization.Localization(__file__, 81, 12), outmess_75366, *[result_mod_75380], **kwargs_75381)
    
    # SSA branch for the else part of an if statement (line 80)
    module_type_store.open_ssa_branch('else')
    
    # Call to outmess(...): (line 84)
    # Processing the call arguments (line 84)
    str_75384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 20), 'str', '\t\tConstructing COMMON block support for "%s"...\n\t\t  %s\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 85)
    tuple_75385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 85)
    # Adding element type (line 85)
    # Getting the type of 'name' (line 85)
    name_75386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 16), tuple_75385, name_75386)
    # Adding element type (line 85)
    
    # Call to join(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'inames' (line 85)
    inames_75389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 31), 'inames', False)
    # Processing the call keyword arguments (line 85)
    kwargs_75390 = {}
    str_75387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 22), 'str', ',')
    # Obtaining the member 'join' of a type (line 85)
    join_75388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 22), str_75387, 'join')
    # Calling join(args, kwargs) (line 85)
    join_call_result_75391 = invoke(stypy.reporting.localization.Localization(__file__, 85, 22), join_75388, *[inames_75389], **kwargs_75390)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 16), tuple_75385, join_call_result_75391)
    
    # Applying the binary operator '%' (line 84)
    result_mod_75392 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 20), '%', str_75384, tuple_75385)
    
    # Processing the call keyword arguments (line 84)
    kwargs_75393 = {}
    # Getting the type of 'outmess' (line 84)
    outmess_75383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'outmess', False)
    # Calling outmess(args, kwargs) (line 84)
    outmess_call_result_75394 = invoke(stypy.reporting.localization.Localization(__file__, 84, 12), outmess_75383, *[result_mod_75392], **kwargs_75393)
    
    # SSA join for if statement (line 80)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to fadd(...): (line 86)
    # Processing the call arguments (line 86)
    str_75396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 13), 'str', 'subroutine f2pyinit%s(setupfunc)')
    # Getting the type of 'name' (line 86)
    name_75397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 50), 'name', False)
    # Applying the binary operator '%' (line 86)
    result_mod_75398 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 13), '%', str_75396, name_75397)
    
    # Processing the call keyword arguments (line 86)
    kwargs_75399 = {}
    # Getting the type of 'fadd' (line 86)
    fadd_75395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'fadd', False)
    # Calling fadd(args, kwargs) (line 86)
    fadd_call_result_75400 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), fadd_75395, *[result_mod_75398], **kwargs_75399)
    
    
    # Call to fadd(...): (line 87)
    # Processing the call arguments (line 87)
    str_75402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 13), 'str', 'external setupfunc')
    # Processing the call keyword arguments (line 87)
    kwargs_75403 = {}
    # Getting the type of 'fadd' (line 87)
    fadd_75401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'fadd', False)
    # Calling fadd(args, kwargs) (line 87)
    fadd_call_result_75404 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), fadd_75401, *[str_75402], **kwargs_75403)
    
    
    # Getting the type of 'vnames' (line 88)
    vnames_75405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 17), 'vnames')
    # Testing the type of a for loop iterable (line 88)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 88, 8), vnames_75405)
    # Getting the type of the for loop variable (line 88)
    for_loop_var_75406 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 88, 8), vnames_75405)
    # Assigning a type to the variable 'n' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'n', for_loop_var_75406)
    # SSA begins for a for statement (line 88)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to fadd(...): (line 89)
    # Processing the call arguments (line 89)
    
    # Call to var2fixfortran(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'vars' (line 89)
    vars_75410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 42), 'vars', False)
    # Getting the type of 'n' (line 89)
    n_75411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 48), 'n', False)
    # Processing the call keyword arguments (line 89)
    kwargs_75412 = {}
    # Getting the type of 'func2subr' (line 89)
    func2subr_75408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 17), 'func2subr', False)
    # Obtaining the member 'var2fixfortran' of a type (line 89)
    var2fixfortran_75409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 17), func2subr_75408, 'var2fixfortran')
    # Calling var2fixfortran(args, kwargs) (line 89)
    var2fixfortran_call_result_75413 = invoke(stypy.reporting.localization.Localization(__file__, 89, 17), var2fixfortran_75409, *[vars_75410, n_75411], **kwargs_75412)
    
    # Processing the call keyword arguments (line 89)
    kwargs_75414 = {}
    # Getting the type of 'fadd' (line 89)
    fadd_75407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'fadd', False)
    # Calling fadd(args, kwargs) (line 89)
    fadd_call_result_75415 = invoke(stypy.reporting.localization.Localization(__file__, 89, 12), fadd_75407, *[var2fixfortran_call_result_75413], **kwargs_75414)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'name' (line 90)
    name_75416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 11), 'name')
    str_75417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 19), 'str', '_BLNK_')
    # Applying the binary operator '==' (line 90)
    result_eq_75418 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 11), '==', name_75416, str_75417)
    
    # Testing the type of an if condition (line 90)
    if_condition_75419 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 8), result_eq_75418)
    # Assigning a type to the variable 'if_condition_75419' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'if_condition_75419', if_condition_75419)
    # SSA begins for if statement (line 90)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to fadd(...): (line 91)
    # Processing the call arguments (line 91)
    str_75421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 17), 'str', 'common %s')
    
    # Call to join(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'vnames' (line 91)
    vnames_75424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 41), 'vnames', False)
    # Processing the call keyword arguments (line 91)
    kwargs_75425 = {}
    str_75422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 32), 'str', ',')
    # Obtaining the member 'join' of a type (line 91)
    join_75423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 32), str_75422, 'join')
    # Calling join(args, kwargs) (line 91)
    join_call_result_75426 = invoke(stypy.reporting.localization.Localization(__file__, 91, 32), join_75423, *[vnames_75424], **kwargs_75425)
    
    # Applying the binary operator '%' (line 91)
    result_mod_75427 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 17), '%', str_75421, join_call_result_75426)
    
    # Processing the call keyword arguments (line 91)
    kwargs_75428 = {}
    # Getting the type of 'fadd' (line 91)
    fadd_75420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'fadd', False)
    # Calling fadd(args, kwargs) (line 91)
    fadd_call_result_75429 = invoke(stypy.reporting.localization.Localization(__file__, 91, 12), fadd_75420, *[result_mod_75427], **kwargs_75428)
    
    # SSA branch for the else part of an if statement (line 90)
    module_type_store.open_ssa_branch('else')
    
    # Call to fadd(...): (line 93)
    # Processing the call arguments (line 93)
    str_75431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 17), 'str', 'common /%s/ %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 93)
    tuple_75432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 93)
    # Adding element type (line 93)
    # Getting the type of 'name' (line 93)
    name_75433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 37), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 37), tuple_75432, name_75433)
    # Adding element type (line 93)
    
    # Call to join(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'vnames' (line 93)
    vnames_75436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 52), 'vnames', False)
    # Processing the call keyword arguments (line 93)
    kwargs_75437 = {}
    str_75434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 43), 'str', ',')
    # Obtaining the member 'join' of a type (line 93)
    join_75435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 43), str_75434, 'join')
    # Calling join(args, kwargs) (line 93)
    join_call_result_75438 = invoke(stypy.reporting.localization.Localization(__file__, 93, 43), join_75435, *[vnames_75436], **kwargs_75437)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 37), tuple_75432, join_call_result_75438)
    
    # Applying the binary operator '%' (line 93)
    result_mod_75439 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 17), '%', str_75431, tuple_75432)
    
    # Processing the call keyword arguments (line 93)
    kwargs_75440 = {}
    # Getting the type of 'fadd' (line 93)
    fadd_75430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'fadd', False)
    # Calling fadd(args, kwargs) (line 93)
    fadd_call_result_75441 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), fadd_75430, *[result_mod_75439], **kwargs_75440)
    
    # SSA join for if statement (line 90)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to fadd(...): (line 94)
    # Processing the call arguments (line 94)
    str_75443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 13), 'str', 'call setupfunc(%s)')
    
    # Call to join(...): (line 94)
    # Processing the call arguments (line 94)
    # Getting the type of 'inames' (line 94)
    inames_75446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 46), 'inames', False)
    # Processing the call keyword arguments (line 94)
    kwargs_75447 = {}
    str_75444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 37), 'str', ',')
    # Obtaining the member 'join' of a type (line 94)
    join_75445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 37), str_75444, 'join')
    # Calling join(args, kwargs) (line 94)
    join_call_result_75448 = invoke(stypy.reporting.localization.Localization(__file__, 94, 37), join_75445, *[inames_75446], **kwargs_75447)
    
    # Applying the binary operator '%' (line 94)
    result_mod_75449 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 13), '%', str_75443, join_call_result_75448)
    
    # Processing the call keyword arguments (line 94)
    kwargs_75450 = {}
    # Getting the type of 'fadd' (line 94)
    fadd_75442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'fadd', False)
    # Calling fadd(args, kwargs) (line 94)
    fadd_call_result_75451 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), fadd_75442, *[result_mod_75449], **kwargs_75450)
    
    
    # Call to fadd(...): (line 95)
    # Processing the call arguments (line 95)
    str_75453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 13), 'str', 'end\n')
    # Processing the call keyword arguments (line 95)
    kwargs_75454 = {}
    # Getting the type of 'fadd' (line 95)
    fadd_75452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'fadd', False)
    # Calling fadd(args, kwargs) (line 95)
    fadd_call_result_75455 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), fadd_75452, *[str_75453], **kwargs_75454)
    
    
    # Call to cadd(...): (line 96)
    # Processing the call arguments (line 96)
    str_75457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 13), 'str', 'static FortranDataDef f2py_%s_def[] = {')
    # Getting the type of 'name' (line 96)
    name_75458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 58), 'name', False)
    # Applying the binary operator '%' (line 96)
    result_mod_75459 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 13), '%', str_75457, name_75458)
    
    # Processing the call keyword arguments (line 96)
    kwargs_75460 = {}
    # Getting the type of 'cadd' (line 96)
    cadd_75456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'cadd', False)
    # Calling cadd(args, kwargs) (line 96)
    cadd_call_result_75461 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), cadd_75456, *[result_mod_75459], **kwargs_75460)
    
    
    # Assigning a List to a Name (line 97):
    
    # Assigning a List to a Name (line 97):
    
    # Obtaining an instance of the builtin type 'list' (line 97)
    list_75462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 97)
    
    # Assigning a type to the variable 'idims' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'idims', list_75462)
    
    # Getting the type of 'inames' (line 98)
    inames_75463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 17), 'inames')
    # Testing the type of a for loop iterable (line 98)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 98, 8), inames_75463)
    # Getting the type of the for loop variable (line 98)
    for_loop_var_75464 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 98, 8), inames_75463)
    # Assigning a type to the variable 'n' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'n', for_loop_var_75464)
    # SSA begins for a for statement (line 98)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 99):
    
    # Assigning a Call to a Name (line 99):
    
    # Call to getctype(...): (line 99)
    # Processing the call arguments (line 99)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 99)
    n_75467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 41), 'n', False)
    # Getting the type of 'vars' (line 99)
    vars_75468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 36), 'vars', False)
    # Obtaining the member '__getitem__' of a type (line 99)
    getitem___75469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 36), vars_75468, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 99)
    subscript_call_result_75470 = invoke(stypy.reporting.localization.Localization(__file__, 99, 36), getitem___75469, n_75467)
    
    # Processing the call keyword arguments (line 99)
    kwargs_75471 = {}
    # Getting the type of 'capi_maps' (line 99)
    capi_maps_75465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 17), 'capi_maps', False)
    # Obtaining the member 'getctype' of a type (line 99)
    getctype_75466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 17), capi_maps_75465, 'getctype')
    # Calling getctype(args, kwargs) (line 99)
    getctype_call_result_75472 = invoke(stypy.reporting.localization.Localization(__file__, 99, 17), getctype_75466, *[subscript_call_result_75470], **kwargs_75471)
    
    # Assigning a type to the variable 'ct' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'ct', getctype_call_result_75472)
    
    # Assigning a Subscript to a Name (line 100):
    
    # Assigning a Subscript to a Name (line 100):
    
    # Obtaining the type of the subscript
    # Getting the type of 'ct' (line 100)
    ct_75473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 38), 'ct')
    # Getting the type of 'capi_maps' (line 100)
    capi_maps_75474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 17), 'capi_maps')
    # Obtaining the member 'c2capi_map' of a type (line 100)
    c2capi_map_75475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 17), capi_maps_75474, 'c2capi_map')
    # Obtaining the member '__getitem__' of a type (line 100)
    getitem___75476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 17), c2capi_map_75475, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 100)
    subscript_call_result_75477 = invoke(stypy.reporting.localization.Localization(__file__, 100, 17), getitem___75476, ct_75473)
    
    # Assigning a type to the variable 'at' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'at', subscript_call_result_75477)
    
    # Assigning a Call to a Name (line 101):
    
    # Assigning a Call to a Name (line 101):
    
    # Call to getarrdims(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'n' (line 101)
    n_75480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 38), 'n', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 101)
    n_75481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 46), 'n', False)
    # Getting the type of 'vars' (line 101)
    vars_75482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 41), 'vars', False)
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___75483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 41), vars_75482, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_75484 = invoke(stypy.reporting.localization.Localization(__file__, 101, 41), getitem___75483, n_75481)
    
    # Processing the call keyword arguments (line 101)
    kwargs_75485 = {}
    # Getting the type of 'capi_maps' (line 101)
    capi_maps_75478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 'capi_maps', False)
    # Obtaining the member 'getarrdims' of a type (line 101)
    getarrdims_75479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 17), capi_maps_75478, 'getarrdims')
    # Calling getarrdims(args, kwargs) (line 101)
    getarrdims_call_result_75486 = invoke(stypy.reporting.localization.Localization(__file__, 101, 17), getarrdims_75479, *[n_75480, subscript_call_result_75484], **kwargs_75485)
    
    # Assigning a type to the variable 'dm' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'dm', getarrdims_call_result_75486)
    
    
    # Obtaining the type of the subscript
    str_75487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 18), 'str', 'dims')
    # Getting the type of 'dm' (line 102)
    dm_75488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 15), 'dm')
    # Obtaining the member '__getitem__' of a type (line 102)
    getitem___75489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 15), dm_75488, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 102)
    subscript_call_result_75490 = invoke(stypy.reporting.localization.Localization(__file__, 102, 15), getitem___75489, str_75487)
    
    # Testing the type of an if condition (line 102)
    if_condition_75491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 12), subscript_call_result_75490)
    # Assigning a type to the variable 'if_condition_75491' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'if_condition_75491', if_condition_75491)
    # SSA begins for if statement (line 102)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 103)
    # Processing the call arguments (line 103)
    str_75494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 29), 'str', '(%s)')
    
    # Obtaining the type of the subscript
    str_75495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 42), 'str', 'dims')
    # Getting the type of 'dm' (line 103)
    dm_75496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 39), 'dm', False)
    # Obtaining the member '__getitem__' of a type (line 103)
    getitem___75497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 39), dm_75496, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 103)
    subscript_call_result_75498 = invoke(stypy.reporting.localization.Localization(__file__, 103, 39), getitem___75497, str_75495)
    
    # Applying the binary operator '%' (line 103)
    result_mod_75499 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 29), '%', str_75494, subscript_call_result_75498)
    
    # Processing the call keyword arguments (line 103)
    kwargs_75500 = {}
    # Getting the type of 'idims' (line 103)
    idims_75492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'idims', False)
    # Obtaining the member 'append' of a type (line 103)
    append_75493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 16), idims_75492, 'append')
    # Calling append(args, kwargs) (line 103)
    append_call_result_75501 = invoke(stypy.reporting.localization.Localization(__file__, 103, 16), append_75493, *[result_mod_75499], **kwargs_75500)
    
    # SSA branch for the else part of an if statement (line 102)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 105)
    # Processing the call arguments (line 105)
    str_75504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 29), 'str', '')
    # Processing the call keyword arguments (line 105)
    kwargs_75505 = {}
    # Getting the type of 'idims' (line 105)
    idims_75502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'idims', False)
    # Obtaining the member 'append' of a type (line 105)
    append_75503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 16), idims_75502, 'append')
    # Calling append(args, kwargs) (line 105)
    append_call_result_75506 = invoke(stypy.reporting.localization.Localization(__file__, 105, 16), append_75503, *[str_75504], **kwargs_75505)
    
    # SSA join for if statement (line 102)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 106):
    
    # Assigning a Call to a Name (line 106):
    
    # Call to strip(...): (line 106)
    # Processing the call keyword arguments (line 106)
    kwargs_75512 = {}
    
    # Obtaining the type of the subscript
    str_75507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 21), 'str', 'dims')
    # Getting the type of 'dm' (line 106)
    dm_75508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), 'dm', False)
    # Obtaining the member '__getitem__' of a type (line 106)
    getitem___75509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 18), dm_75508, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
    subscript_call_result_75510 = invoke(stypy.reporting.localization.Localization(__file__, 106, 18), getitem___75509, str_75507)
    
    # Obtaining the member 'strip' of a type (line 106)
    strip_75511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 18), subscript_call_result_75510, 'strip')
    # Calling strip(args, kwargs) (line 106)
    strip_call_result_75513 = invoke(stypy.reporting.localization.Localization(__file__, 106, 18), strip_75511, *[], **kwargs_75512)
    
    # Assigning a type to the variable 'dms' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'dms', strip_call_result_75513)
    
    
    # Getting the type of 'dms' (line 107)
    dms_75514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'dms')
    # Applying the 'not' unary operator (line 107)
    result_not__75515 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 15), 'not', dms_75514)
    
    # Testing the type of an if condition (line 107)
    if_condition_75516 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 12), result_not__75515)
    # Assigning a type to the variable 'if_condition_75516' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'if_condition_75516', if_condition_75516)
    # SSA begins for if statement (line 107)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 108):
    
    # Assigning a Str to a Name (line 108):
    str_75517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 22), 'str', '-1')
    # Assigning a type to the variable 'dms' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'dms', str_75517)
    # SSA join for if statement (line 107)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to cadd(...): (line 109)
    # Processing the call arguments (line 109)
    str_75519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 17), 'str', '\t{"%s",%s,{{%s}},%s},')
    
    # Obtaining an instance of the builtin type 'tuple' (line 109)
    tuple_75520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 109)
    # Adding element type (line 109)
    # Getting the type of 'n' (line 109)
    n_75521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 47), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 47), tuple_75520, n_75521)
    # Adding element type (line 109)
    
    # Obtaining the type of the subscript
    str_75522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 53), 'str', 'rank')
    # Getting the type of 'dm' (line 109)
    dm_75523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 50), 'dm', False)
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___75524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 50), dm_75523, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_75525 = invoke(stypy.reporting.localization.Localization(__file__, 109, 50), getitem___75524, str_75522)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 47), tuple_75520, subscript_call_result_75525)
    # Adding element type (line 109)
    # Getting the type of 'dms' (line 109)
    dms_75526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 62), 'dms', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 47), tuple_75520, dms_75526)
    # Adding element type (line 109)
    # Getting the type of 'at' (line 109)
    at_75527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 67), 'at', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 47), tuple_75520, at_75527)
    
    # Applying the binary operator '%' (line 109)
    result_mod_75528 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 17), '%', str_75519, tuple_75520)
    
    # Processing the call keyword arguments (line 109)
    kwargs_75529 = {}
    # Getting the type of 'cadd' (line 109)
    cadd_75518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'cadd', False)
    # Calling cadd(args, kwargs) (line 109)
    cadd_call_result_75530 = invoke(stypy.reporting.localization.Localization(__file__, 109, 12), cadd_75518, *[result_mod_75528], **kwargs_75529)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to cadd(...): (line 110)
    # Processing the call arguments (line 110)
    str_75532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 13), 'str', '\t{NULL}\n};')
    # Processing the call keyword arguments (line 110)
    kwargs_75533 = {}
    # Getting the type of 'cadd' (line 110)
    cadd_75531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'cadd', False)
    # Calling cadd(args, kwargs) (line 110)
    cadd_call_result_75534 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), cadd_75531, *[str_75532], **kwargs_75533)
    
    
    # Assigning a Call to a Name (line 111):
    
    # Assigning a Call to a Name (line 111):
    
    # Call to rmbadname(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'inames' (line 111)
    inames_75536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 28), 'inames', False)
    # Processing the call keyword arguments (line 111)
    kwargs_75537 = {}
    # Getting the type of 'rmbadname' (line 111)
    rmbadname_75535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 18), 'rmbadname', False)
    # Calling rmbadname(args, kwargs) (line 111)
    rmbadname_call_result_75538 = invoke(stypy.reporting.localization.Localization(__file__, 111, 18), rmbadname_75535, *[inames_75536], **kwargs_75537)
    
    # Assigning a type to the variable 'inames1' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'inames1', rmbadname_call_result_75538)
    
    # Assigning a Call to a Name (line 112):
    
    # Assigning a Call to a Name (line 112):
    
    # Call to join(...): (line 112)
    # Processing the call arguments (line 112)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'inames1' (line 112)
    inames1_75544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 54), 'inames1', False)
    comprehension_75545 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 32), inames1_75544)
    # Assigning a type to the variable 's' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 32), 's', comprehension_75545)
    str_75541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 32), 'str', 'char *')
    # Getting the type of 's' (line 112)
    s_75542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 43), 's', False)
    # Applying the binary operator '+' (line 112)
    result_add_75543 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 32), '+', str_75541, s_75542)
    
    list_75546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 32), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 32), list_75546, result_add_75543)
    # Processing the call keyword arguments (line 112)
    kwargs_75547 = {}
    str_75539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 22), 'str', ',')
    # Obtaining the member 'join' of a type (line 112)
    join_75540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 22), str_75539, 'join')
    # Calling join(args, kwargs) (line 112)
    join_call_result_75548 = invoke(stypy.reporting.localization.Localization(__file__, 112, 22), join_75540, *[list_75546], **kwargs_75547)
    
    # Assigning a type to the variable 'inames1_tps' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'inames1_tps', join_call_result_75548)
    
    # Call to cadd(...): (line 113)
    # Processing the call arguments (line 113)
    str_75550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 13), 'str', 'static void f2py_setup_%s(%s) {')
    
    # Obtaining an instance of the builtin type 'tuple' (line 113)
    tuple_75551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 113)
    # Adding element type (line 113)
    # Getting the type of 'name' (line 113)
    name_75552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 50), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 50), tuple_75551, name_75552)
    # Adding element type (line 113)
    # Getting the type of 'inames1_tps' (line 113)
    inames1_tps_75553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 56), 'inames1_tps', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 50), tuple_75551, inames1_tps_75553)
    
    # Applying the binary operator '%' (line 113)
    result_mod_75554 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 13), '%', str_75550, tuple_75551)
    
    # Processing the call keyword arguments (line 113)
    kwargs_75555 = {}
    # Getting the type of 'cadd' (line 113)
    cadd_75549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'cadd', False)
    # Calling cadd(args, kwargs) (line 113)
    cadd_call_result_75556 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), cadd_75549, *[result_mod_75554], **kwargs_75555)
    
    
    # Call to cadd(...): (line 114)
    # Processing the call arguments (line 114)
    str_75558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 13), 'str', '\tint i_f2py=0;')
    # Processing the call keyword arguments (line 114)
    kwargs_75559 = {}
    # Getting the type of 'cadd' (line 114)
    cadd_75557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'cadd', False)
    # Calling cadd(args, kwargs) (line 114)
    cadd_call_result_75560 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), cadd_75557, *[str_75558], **kwargs_75559)
    
    
    # Getting the type of 'inames1' (line 115)
    inames1_75561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 17), 'inames1')
    # Testing the type of a for loop iterable (line 115)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 115, 8), inames1_75561)
    # Getting the type of the for loop variable (line 115)
    for_loop_var_75562 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 115, 8), inames1_75561)
    # Assigning a type to the variable 'n' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'n', for_loop_var_75562)
    # SSA begins for a for statement (line 115)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to cadd(...): (line 116)
    # Processing the call arguments (line 116)
    str_75564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 17), 'str', '\tf2py_%s_def[i_f2py++].data = %s;')
    
    # Obtaining an instance of the builtin type 'tuple' (line 116)
    tuple_75565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 57), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 116)
    # Adding element type (line 116)
    # Getting the type of 'name' (line 116)
    name_75566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 57), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 57), tuple_75565, name_75566)
    # Adding element type (line 116)
    # Getting the type of 'n' (line 116)
    n_75567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 63), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 57), tuple_75565, n_75567)
    
    # Applying the binary operator '%' (line 116)
    result_mod_75568 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 17), '%', str_75564, tuple_75565)
    
    # Processing the call keyword arguments (line 116)
    kwargs_75569 = {}
    # Getting the type of 'cadd' (line 116)
    cadd_75563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'cadd', False)
    # Calling cadd(args, kwargs) (line 116)
    cadd_call_result_75570 = invoke(stypy.reporting.localization.Localization(__file__, 116, 12), cadd_75563, *[result_mod_75568], **kwargs_75569)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to cadd(...): (line 117)
    # Processing the call arguments (line 117)
    str_75572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 13), 'str', '}')
    # Processing the call keyword arguments (line 117)
    kwargs_75573 = {}
    # Getting the type of 'cadd' (line 117)
    cadd_75571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'cadd', False)
    # Calling cadd(args, kwargs) (line 117)
    cadd_call_result_75574 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), cadd_75571, *[str_75572], **kwargs_75573)
    
    
    
    str_75575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 11), 'str', '_')
    # Getting the type of 'lower_name' (line 118)
    lower_name_75576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 18), 'lower_name')
    # Applying the binary operator 'in' (line 118)
    result_contains_75577 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 11), 'in', str_75575, lower_name_75576)
    
    # Testing the type of an if condition (line 118)
    if_condition_75578 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 8), result_contains_75577)
    # Assigning a type to the variable 'if_condition_75578' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'if_condition_75578', if_condition_75578)
    # SSA begins for if statement (line 118)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 119):
    
    # Assigning a Str to a Name (line 119):
    str_75579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 21), 'str', 'F_FUNC_US')
    # Assigning a type to the variable 'F_FUNC' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'F_FUNC', str_75579)
    # SSA branch for the else part of an if statement (line 118)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 121):
    
    # Assigning a Str to a Name (line 121):
    str_75580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 21), 'str', 'F_FUNC')
    # Assigning a type to the variable 'F_FUNC' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'F_FUNC', str_75580)
    # SSA join for if statement (line 118)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to cadd(...): (line 122)
    # Processing the call arguments (line 122)
    str_75582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 13), 'str', 'extern void %s(f2pyinit%s,F2PYINIT%s)(void(*)(%s));')
    
    # Obtaining an instance of the builtin type 'tuple' (line 123)
    tuple_75583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 123)
    # Adding element type (line 123)
    # Getting the type of 'F_FUNC' (line 123)
    F_FUNC_75584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'F_FUNC', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 16), tuple_75583, F_FUNC_75584)
    # Adding element type (line 123)
    # Getting the type of 'lower_name' (line 123)
    lower_name_75585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 24), 'lower_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 16), tuple_75583, lower_name_75585)
    # Adding element type (line 123)
    
    # Call to upper(...): (line 123)
    # Processing the call keyword arguments (line 123)
    kwargs_75588 = {}
    # Getting the type of 'name' (line 123)
    name_75586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 36), 'name', False)
    # Obtaining the member 'upper' of a type (line 123)
    upper_75587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 36), name_75586, 'upper')
    # Calling upper(args, kwargs) (line 123)
    upper_call_result_75589 = invoke(stypy.reporting.localization.Localization(__file__, 123, 36), upper_75587, *[], **kwargs_75588)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 16), tuple_75583, upper_call_result_75589)
    # Adding element type (line 123)
    
    # Call to join(...): (line 124)
    # Processing the call arguments (line 124)
    
    # Obtaining an instance of the builtin type 'list' (line 124)
    list_75592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 124)
    # Adding element type (line 124)
    str_75593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 26), 'str', 'char*')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 25), list_75592, str_75593)
    
    
    # Call to len(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'inames1' (line 124)
    inames1_75595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 41), 'inames1', False)
    # Processing the call keyword arguments (line 124)
    kwargs_75596 = {}
    # Getting the type of 'len' (line 124)
    len_75594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 37), 'len', False)
    # Calling len(args, kwargs) (line 124)
    len_call_result_75597 = invoke(stypy.reporting.localization.Localization(__file__, 124, 37), len_75594, *[inames1_75595], **kwargs_75596)
    
    # Applying the binary operator '*' (line 124)
    result_mul_75598 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 25), '*', list_75592, len_call_result_75597)
    
    # Processing the call keyword arguments (line 124)
    kwargs_75599 = {}
    str_75590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 16), 'str', ',')
    # Obtaining the member 'join' of a type (line 124)
    join_75591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 16), str_75590, 'join')
    # Calling join(args, kwargs) (line 124)
    join_call_result_75600 = invoke(stypy.reporting.localization.Localization(__file__, 124, 16), join_75591, *[result_mul_75598], **kwargs_75599)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 16), tuple_75583, join_call_result_75600)
    
    # Applying the binary operator '%' (line 122)
    result_mod_75601 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 13), '%', str_75582, tuple_75583)
    
    # Processing the call keyword arguments (line 122)
    kwargs_75602 = {}
    # Getting the type of 'cadd' (line 122)
    cadd_75581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'cadd', False)
    # Calling cadd(args, kwargs) (line 122)
    cadd_call_result_75603 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), cadd_75581, *[result_mod_75601], **kwargs_75602)
    
    
    # Call to cadd(...): (line 125)
    # Processing the call arguments (line 125)
    str_75605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 13), 'str', 'static void f2py_init_%s(void) {')
    # Getting the type of 'name' (line 125)
    name_75606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 50), 'name', False)
    # Applying the binary operator '%' (line 125)
    result_mod_75607 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 13), '%', str_75605, name_75606)
    
    # Processing the call keyword arguments (line 125)
    kwargs_75608 = {}
    # Getting the type of 'cadd' (line 125)
    cadd_75604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'cadd', False)
    # Calling cadd(args, kwargs) (line 125)
    cadd_call_result_75609 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), cadd_75604, *[result_mod_75607], **kwargs_75608)
    
    
    # Call to cadd(...): (line 126)
    # Processing the call arguments (line 126)
    str_75611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 13), 'str', '\t%s(f2pyinit%s,F2PYINIT%s)(f2py_setup_%s);')
    
    # Obtaining an instance of the builtin type 'tuple' (line 127)
    tuple_75612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 127)
    # Adding element type (line 127)
    # Getting the type of 'F_FUNC' (line 127)
    F_FUNC_75613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'F_FUNC', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 16), tuple_75612, F_FUNC_75613)
    # Adding element type (line 127)
    # Getting the type of 'lower_name' (line 127)
    lower_name_75614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 24), 'lower_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 16), tuple_75612, lower_name_75614)
    # Adding element type (line 127)
    
    # Call to upper(...): (line 127)
    # Processing the call keyword arguments (line 127)
    kwargs_75617 = {}
    # Getting the type of 'name' (line 127)
    name_75615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 36), 'name', False)
    # Obtaining the member 'upper' of a type (line 127)
    upper_75616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 36), name_75615, 'upper')
    # Calling upper(args, kwargs) (line 127)
    upper_call_result_75618 = invoke(stypy.reporting.localization.Localization(__file__, 127, 36), upper_75616, *[], **kwargs_75617)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 16), tuple_75612, upper_call_result_75618)
    # Adding element type (line 127)
    # Getting the type of 'name' (line 127)
    name_75619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 50), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 16), tuple_75612, name_75619)
    
    # Applying the binary operator '%' (line 126)
    result_mod_75620 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 13), '%', str_75611, tuple_75612)
    
    # Processing the call keyword arguments (line 126)
    kwargs_75621 = {}
    # Getting the type of 'cadd' (line 126)
    cadd_75610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'cadd', False)
    # Calling cadd(args, kwargs) (line 126)
    cadd_call_result_75622 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), cadd_75610, *[result_mod_75620], **kwargs_75621)
    
    
    # Call to cadd(...): (line 128)
    # Processing the call arguments (line 128)
    str_75624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 13), 'str', '}\n')
    # Processing the call keyword arguments (line 128)
    kwargs_75625 = {}
    # Getting the type of 'cadd' (line 128)
    cadd_75623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'cadd', False)
    # Calling cadd(args, kwargs) (line 128)
    cadd_call_result_75626 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), cadd_75623, *[str_75624], **kwargs_75625)
    
    
    # Call to iadd(...): (line 129)
    # Processing the call arguments (line 129)
    str_75628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 13), 'str', '\tF2PyDict_SetItemString(d, "%s", PyFortranObject_New(f2py_%s_def,f2py_init_%s));')
    
    # Obtaining an instance of the builtin type 'tuple' (line 130)
    tuple_75629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 130)
    # Adding element type (line 130)
    # Getting the type of 'name' (line 130)
    name_75630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 12), tuple_75629, name_75630)
    # Adding element type (line 130)
    # Getting the type of 'name' (line 130)
    name_75631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 18), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 12), tuple_75629, name_75631)
    # Adding element type (line 130)
    # Getting the type of 'name' (line 130)
    name_75632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 12), tuple_75629, name_75632)
    
    # Applying the binary operator '%' (line 129)
    result_mod_75633 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 13), '%', str_75628, tuple_75629)
    
    # Processing the call keyword arguments (line 129)
    kwargs_75634 = {}
    # Getting the type of 'iadd' (line 129)
    iadd_75627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'iadd', False)
    # Calling iadd(args, kwargs) (line 129)
    iadd_call_result_75635 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), iadd_75627, *[result_mod_75633], **kwargs_75634)
    
    
    # Assigning a Call to a Name (line 131):
    
    # Assigning a Call to a Name (line 131):
    
    # Call to replace(...): (line 131)
    # Processing the call arguments (line 131)
    str_75638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 29), 'str', '_')
    str_75639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 34), 'str', '\\_')
    # Processing the call keyword arguments (line 131)
    kwargs_75640 = {}
    # Getting the type of 'name' (line 131)
    name_75636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'name', False)
    # Obtaining the member 'replace' of a type (line 131)
    replace_75637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 16), name_75636, 'replace')
    # Calling replace(args, kwargs) (line 131)
    replace_call_result_75641 = invoke(stypy.reporting.localization.Localization(__file__, 131, 16), replace_75637, *[str_75638, str_75639], **kwargs_75640)
    
    # Assigning a type to the variable 'tname' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'tname', replace_call_result_75641)
    
    # Call to dadd(...): (line 132)
    # Processing the call arguments (line 132)
    str_75643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 13), 'str', '\\subsection{Common block \\texttt{%s}}\n')
    # Getting the type of 'tname' (line 132)
    tname_75644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 60), 'tname', False)
    # Applying the binary operator '%' (line 132)
    result_mod_75645 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 13), '%', str_75643, tname_75644)
    
    # Processing the call keyword arguments (line 132)
    kwargs_75646 = {}
    # Getting the type of 'dadd' (line 132)
    dadd_75642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'dadd', False)
    # Calling dadd(args, kwargs) (line 132)
    dadd_call_result_75647 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), dadd_75642, *[result_mod_75645], **kwargs_75646)
    
    
    # Call to dadd(...): (line 133)
    # Processing the call arguments (line 133)
    str_75649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 13), 'str', '\\begin{description}')
    # Processing the call keyword arguments (line 133)
    kwargs_75650 = {}
    # Getting the type of 'dadd' (line 133)
    dadd_75648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'dadd', False)
    # Calling dadd(args, kwargs) (line 133)
    dadd_call_result_75651 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), dadd_75648, *[str_75649], **kwargs_75650)
    
    
    # Getting the type of 'inames' (line 134)
    inames_75652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 17), 'inames')
    # Testing the type of a for loop iterable (line 134)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 134, 8), inames_75652)
    # Getting the type of the for loop variable (line 134)
    for_loop_var_75653 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 134, 8), inames_75652)
    # Assigning a type to the variable 'n' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'n', for_loop_var_75653)
    # SSA begins for a for statement (line 134)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to dadd(...): (line 135)
    # Processing the call arguments (line 135)
    str_75655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 17), 'str', '\\item[]{{}\\verb@%s@{}}')
    
    # Call to getarrdocsign(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'n' (line 136)
    n_75658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 42), 'n', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 136)
    n_75659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 50), 'n', False)
    # Getting the type of 'vars' (line 136)
    vars_75660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 45), 'vars', False)
    # Obtaining the member '__getitem__' of a type (line 136)
    getitem___75661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 45), vars_75660, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 136)
    subscript_call_result_75662 = invoke(stypy.reporting.localization.Localization(__file__, 136, 45), getitem___75661, n_75659)
    
    # Processing the call keyword arguments (line 136)
    kwargs_75663 = {}
    # Getting the type of 'capi_maps' (line 136)
    capi_maps_75656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 18), 'capi_maps', False)
    # Obtaining the member 'getarrdocsign' of a type (line 136)
    getarrdocsign_75657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 18), capi_maps_75656, 'getarrdocsign')
    # Calling getarrdocsign(args, kwargs) (line 136)
    getarrdocsign_call_result_75664 = invoke(stypy.reporting.localization.Localization(__file__, 136, 18), getarrdocsign_75657, *[n_75658, subscript_call_result_75662], **kwargs_75663)
    
    # Applying the binary operator '%' (line 135)
    result_mod_75665 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 17), '%', str_75655, getarrdocsign_call_result_75664)
    
    # Processing the call keyword arguments (line 135)
    kwargs_75666 = {}
    # Getting the type of 'dadd' (line 135)
    dadd_75654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'dadd', False)
    # Calling dadd(args, kwargs) (line 135)
    dadd_call_result_75667 = invoke(stypy.reporting.localization.Localization(__file__, 135, 12), dadd_75654, *[result_mod_75665], **kwargs_75666)
    
    
    
    # Call to hasnote(...): (line 137)
    # Processing the call arguments (line 137)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 137)
    n_75669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 28), 'n', False)
    # Getting the type of 'vars' (line 137)
    vars_75670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 23), 'vars', False)
    # Obtaining the member '__getitem__' of a type (line 137)
    getitem___75671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 23), vars_75670, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 137)
    subscript_call_result_75672 = invoke(stypy.reporting.localization.Localization(__file__, 137, 23), getitem___75671, n_75669)
    
    # Processing the call keyword arguments (line 137)
    kwargs_75673 = {}
    # Getting the type of 'hasnote' (line 137)
    hasnote_75668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 'hasnote', False)
    # Calling hasnote(args, kwargs) (line 137)
    hasnote_call_result_75674 = invoke(stypy.reporting.localization.Localization(__file__, 137, 15), hasnote_75668, *[subscript_call_result_75672], **kwargs_75673)
    
    # Testing the type of an if condition (line 137)
    if_condition_75675 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 12), hasnote_call_result_75674)
    # Assigning a type to the variable 'if_condition_75675' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'if_condition_75675', if_condition_75675)
    # SSA begins for if statement (line 137)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 138):
    
    # Assigning a Subscript to a Name (line 138):
    
    # Obtaining the type of the subscript
    str_75676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 31), 'str', 'note')
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 138)
    n_75677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 28), 'n')
    # Getting the type of 'vars' (line 138)
    vars_75678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 23), 'vars')
    # Obtaining the member '__getitem__' of a type (line 138)
    getitem___75679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 23), vars_75678, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 138)
    subscript_call_result_75680 = invoke(stypy.reporting.localization.Localization(__file__, 138, 23), getitem___75679, n_75677)
    
    # Obtaining the member '__getitem__' of a type (line 138)
    getitem___75681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 23), subscript_call_result_75680, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 138)
    subscript_call_result_75682 = invoke(stypy.reporting.localization.Localization(__file__, 138, 23), getitem___75681, str_75676)
    
    # Assigning a type to the variable 'note' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'note', subscript_call_result_75682)
    
    # Type idiom detected: calculating its left and rigth part (line 139)
    # Getting the type of 'list' (line 139)
    list_75683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 36), 'list')
    # Getting the type of 'note' (line 139)
    note_75684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 30), 'note')
    
    (may_be_75685, more_types_in_union_75686) = may_be_subtype(list_75683, note_75684)

    if may_be_75685:

        if more_types_in_union_75686:
            # Runtime conditional SSA (line 139)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'note' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'note', remove_not_subtype_from_union(note_75684, list))
        
        # Assigning a Call to a Name (line 140):
        
        # Assigning a Call to a Name (line 140):
        
        # Call to join(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'note' (line 140)
        note_75689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 37), 'note', False)
        # Processing the call keyword arguments (line 140)
        kwargs_75690 = {}
        str_75687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 27), 'str', '\n')
        # Obtaining the member 'join' of a type (line 140)
        join_75688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 27), str_75687, 'join')
        # Calling join(args, kwargs) (line 140)
        join_call_result_75691 = invoke(stypy.reporting.localization.Localization(__file__, 140, 27), join_75688, *[note_75689], **kwargs_75690)
        
        # Assigning a type to the variable 'note' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 20), 'note', join_call_result_75691)

        if more_types_in_union_75686:
            # SSA join for if statement (line 139)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to dadd(...): (line 141)
    # Processing the call arguments (line 141)
    str_75693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 21), 'str', '--- %s')
    # Getting the type of 'note' (line 141)
    note_75694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 33), 'note', False)
    # Applying the binary operator '%' (line 141)
    result_mod_75695 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 21), '%', str_75693, note_75694)
    
    # Processing the call keyword arguments (line 141)
    kwargs_75696 = {}
    # Getting the type of 'dadd' (line 141)
    dadd_75692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'dadd', False)
    # Calling dadd(args, kwargs) (line 141)
    dadd_call_result_75697 = invoke(stypy.reporting.localization.Localization(__file__, 141, 16), dadd_75692, *[result_mod_75695], **kwargs_75696)
    
    # SSA join for if statement (line 137)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to dadd(...): (line 142)
    # Processing the call arguments (line 142)
    str_75699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 13), 'str', '\\end{description}')
    # Processing the call keyword arguments (line 142)
    kwargs_75700 = {}
    # Getting the type of 'dadd' (line 142)
    dadd_75698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'dadd', False)
    # Calling dadd(args, kwargs) (line 142)
    dadd_call_result_75701 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), dadd_75698, *[str_75699], **kwargs_75700)
    
    
    # Call to append(...): (line 143)
    # Processing the call arguments (line 143)
    str_75707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 12), 'str', '"\t/%s/ %s\\n"')
    
    # Obtaining an instance of the builtin type 'tuple' (line 144)
    tuple_75708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 144)
    # Adding element type (line 144)
    # Getting the type of 'name' (line 144)
    name_75709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 32), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 32), tuple_75708, name_75709)
    # Adding element type (line 144)
    
    # Call to join(...): (line 144)
    # Processing the call arguments (line 144)
    
    # Call to map(...): (line 144)
    # Processing the call arguments (line 144)

    @norecursion
    def _stypy_temp_lambda_26(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_26'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_26', 144, 51, True)
        # Passed parameters checking function
        _stypy_temp_lambda_26.stypy_localization = localization
        _stypy_temp_lambda_26.stypy_type_of_self = None
        _stypy_temp_lambda_26.stypy_type_store = module_type_store
        _stypy_temp_lambda_26.stypy_function_name = '_stypy_temp_lambda_26'
        _stypy_temp_lambda_26.stypy_param_names_list = ['v', 'd']
        _stypy_temp_lambda_26.stypy_varargs_param_name = None
        _stypy_temp_lambda_26.stypy_kwargs_param_name = None
        _stypy_temp_lambda_26.stypy_call_defaults = defaults
        _stypy_temp_lambda_26.stypy_call_varargs = varargs
        _stypy_temp_lambda_26.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_26', ['v', 'd'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_26', ['v', 'd'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        # Getting the type of 'v' (line 144)
        v_75713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 64), 'v', False)
        # Getting the type of 'd' (line 144)
        d_75714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 68), 'd', False)
        # Applying the binary operator '+' (line 144)
        result_add_75715 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 64), '+', v_75713, d_75714)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 51), 'stypy_return_type', result_add_75715)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_26' in the type store
        # Getting the type of 'stypy_return_type' (line 144)
        stypy_return_type_75716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 51), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_75716)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_26'
        return stypy_return_type_75716

    # Assigning a type to the variable '_stypy_temp_lambda_26' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 51), '_stypy_temp_lambda_26', _stypy_temp_lambda_26)
    # Getting the type of '_stypy_temp_lambda_26' (line 144)
    _stypy_temp_lambda_26_75717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 51), '_stypy_temp_lambda_26')
    # Getting the type of 'inames' (line 144)
    inames_75718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 71), 'inames', False)
    # Getting the type of 'idims' (line 144)
    idims_75719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 79), 'idims', False)
    # Processing the call keyword arguments (line 144)
    kwargs_75720 = {}
    # Getting the type of 'map' (line 144)
    map_75712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 47), 'map', False)
    # Calling map(args, kwargs) (line 144)
    map_call_result_75721 = invoke(stypy.reporting.localization.Localization(__file__, 144, 47), map_75712, *[_stypy_temp_lambda_26_75717, inames_75718, idims_75719], **kwargs_75720)
    
    # Processing the call keyword arguments (line 144)
    kwargs_75722 = {}
    str_75710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 38), 'str', ',')
    # Obtaining the member 'join' of a type (line 144)
    join_75711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 38), str_75710, 'join')
    # Calling join(args, kwargs) (line 144)
    join_call_result_75723 = invoke(stypy.reporting.localization.Localization(__file__, 144, 38), join_75711, *[map_call_result_75721], **kwargs_75722)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 32), tuple_75708, join_call_result_75723)
    
    # Applying the binary operator '%' (line 144)
    result_mod_75724 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 12), '%', str_75707, tuple_75708)
    
    # Processing the call keyword arguments (line 143)
    kwargs_75725 = {}
    
    # Obtaining the type of the subscript
    str_75702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 12), 'str', 'docs')
    # Getting the type of 'ret' (line 143)
    ret_75703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'ret', False)
    # Obtaining the member '__getitem__' of a type (line 143)
    getitem___75704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), ret_75703, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 143)
    subscript_call_result_75705 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), getitem___75704, str_75702)
    
    # Obtaining the member 'append' of a type (line 143)
    append_75706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), subscript_call_result_75705, 'append')
    # Calling append(args, kwargs) (line 143)
    append_call_result_75726 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), append_75706, *[result_mod_75724], **kwargs_75725)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 145):
    
    # Assigning a Name to a Subscript (line 145):
    # Getting the type of 'chooks' (line 145)
    chooks_75727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 25), 'chooks')
    # Getting the type of 'ret' (line 145)
    ret_75728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'ret')
    str_75729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 8), 'str', 'commonhooks')
    # Storing an element on a container (line 145)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 4), ret_75728, (str_75729, chooks_75727))
    
    # Assigning a Name to a Subscript (line 146):
    
    # Assigning a Name to a Subscript (line 146):
    # Getting the type of 'ihooks' (line 146)
    ihooks_75730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 29), 'ihooks')
    # Getting the type of 'ret' (line 146)
    ret_75731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'ret')
    str_75732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 8), 'str', 'initcommonhooks')
    # Storing an element on a container (line 146)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 4), ret_75731, (str_75732, ihooks_75730))
    
    # Assigning a Subscript to a Subscript (line 147):
    
    # Assigning a Subscript to a Subscript (line 147):
    
    # Obtaining the type of the subscript
    int_75733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 26), 'int')
    # Getting the type of 'doc' (line 147)
    doc_75734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 22), 'doc')
    # Obtaining the member '__getitem__' of a type (line 147)
    getitem___75735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 22), doc_75734, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 147)
    subscript_call_result_75736 = invoke(stypy.reporting.localization.Localization(__file__, 147, 22), getitem___75735, int_75733)
    
    # Getting the type of 'ret' (line 147)
    ret_75737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'ret')
    str_75738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 8), 'str', 'latexdoc')
    # Storing an element on a container (line 147)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 4), ret_75737, (str_75738, subscript_call_result_75736))
    
    
    
    # Call to len(...): (line 148)
    # Processing the call arguments (line 148)
    
    # Obtaining the type of the subscript
    str_75740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 15), 'str', 'docs')
    # Getting the type of 'ret' (line 148)
    ret_75741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 11), 'ret', False)
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___75742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 11), ret_75741, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_75743 = invoke(stypy.reporting.localization.Localization(__file__, 148, 11), getitem___75742, str_75740)
    
    # Processing the call keyword arguments (line 148)
    kwargs_75744 = {}
    # Getting the type of 'len' (line 148)
    len_75739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 7), 'len', False)
    # Calling len(args, kwargs) (line 148)
    len_call_result_75745 = invoke(stypy.reporting.localization.Localization(__file__, 148, 7), len_75739, *[subscript_call_result_75743], **kwargs_75744)
    
    int_75746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 27), 'int')
    # Applying the binary operator '<=' (line 148)
    result_le_75747 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 7), '<=', len_call_result_75745, int_75746)
    
    # Testing the type of an if condition (line 148)
    if_condition_75748 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 4), result_le_75747)
    # Assigning a type to the variable 'if_condition_75748' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'if_condition_75748', if_condition_75748)
    # SSA begins for if statement (line 148)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Subscript (line 149):
    
    # Assigning a Str to a Subscript (line 149):
    str_75749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 22), 'str', '')
    # Getting the type of 'ret' (line 149)
    ret_75750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'ret')
    str_75751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 12), 'str', 'docs')
    # Storing an element on a container (line 149)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 8), ret_75750, (str_75751, str_75749))
    # SSA join for if statement (line 148)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 150)
    tuple_75752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 150)
    # Adding element type (line 150)
    # Getting the type of 'ret' (line 150)
    ret_75753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'ret')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 11), tuple_75752, ret_75753)
    # Adding element type (line 150)
    
    # Obtaining the type of the subscript
    int_75754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 22), 'int')
    # Getting the type of 'fwrap' (line 150)
    fwrap_75755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'fwrap')
    # Obtaining the member '__getitem__' of a type (line 150)
    getitem___75756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 16), fwrap_75755, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 150)
    subscript_call_result_75757 = invoke(stypy.reporting.localization.Localization(__file__, 150, 16), getitem___75756, int_75754)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 11), tuple_75752, subscript_call_result_75757)
    
    # Assigning a type to the variable 'stypy_return_type' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'stypy_return_type', tuple_75752)
    
    # ################# End of 'buildhooks(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'buildhooks' in the type store
    # Getting the type of 'stypy_return_type' (line 53)
    stypy_return_type_75758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_75758)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'buildhooks'
    return stypy_return_type_75758

# Assigning a type to the variable 'buildhooks' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'buildhooks', buildhooks)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
