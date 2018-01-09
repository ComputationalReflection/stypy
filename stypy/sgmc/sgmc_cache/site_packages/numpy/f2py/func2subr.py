
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: '''
3: 
4: Rules for building C/API module with f2py2e.
5: 
6: Copyright 1999,2000 Pearu Peterson all rights reserved,
7: Pearu Peterson <pearu@ioc.ee>
8: Permission to use, modify, and distribute this software is given under the
9: terms of the NumPy License.
10: 
11: NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
12: $Date: 2004/11/26 11:13:06 $
13: Pearu Peterson
14: 
15: '''
16: from __future__ import division, absolute_import, print_function
17: 
18: __version__ = "$Revision: 1.16 $"[10:-1]
19: 
20: f2py_version = 'See `f2py -v`'
21: 
22: import copy
23: 
24: from .auxfuncs import (
25:     getfortranname, isexternal, isfunction, isfunction_wrap, isintent_in,
26:     isintent_out, islogicalfunction, ismoduleroutine, isscalar,
27:     issubroutine, issubroutine_wrap, outmess, show
28: )
29: 
30: 
31: def var2fixfortran(vars, a, fa=None, f90mode=None):
32:     if fa is None:
33:         fa = a
34:     if a not in vars:
35:         show(vars)
36:         outmess('var2fixfortran: No definition for argument "%s".\n' % a)
37:         return ''
38:     if 'typespec' not in vars[a]:
39:         show(vars[a])
40:         outmess('var2fixfortran: No typespec for argument "%s".\n' % a)
41:         return ''
42:     vardef = vars[a]['typespec']
43:     if vardef == 'type' and 'typename' in vars[a]:
44:         vardef = '%s(%s)' % (vardef, vars[a]['typename'])
45:     selector = {}
46:     lk = ''
47:     if 'kindselector' in vars[a]:
48:         selector = vars[a]['kindselector']
49:         lk = 'kind'
50:     elif 'charselector' in vars[a]:
51:         selector = vars[a]['charselector']
52:         lk = 'len'
53:     if '*' in selector:
54:         if f90mode:
55:             if selector['*'] in ['*', ':', '(*)']:
56:                 vardef = '%s(len=*)' % (vardef)
57:             else:
58:                 vardef = '%s(%s=%s)' % (vardef, lk, selector['*'])
59:         else:
60:             if selector['*'] in ['*', ':']:
61:                 vardef = '%s*(%s)' % (vardef, selector['*'])
62:             else:
63:                 vardef = '%s*%s' % (vardef, selector['*'])
64:     else:
65:         if 'len' in selector:
66:             vardef = '%s(len=%s' % (vardef, selector['len'])
67:             if 'kind' in selector:
68:                 vardef = '%s,kind=%s)' % (vardef, selector['kind'])
69:             else:
70:                 vardef = '%s)' % (vardef)
71:         elif 'kind' in selector:
72:             vardef = '%s(kind=%s)' % (vardef, selector['kind'])
73: 
74:     vardef = '%s %s' % (vardef, fa)
75:     if 'dimension' in vars[a]:
76:         vardef = '%s(%s)' % (vardef, ','.join(vars[a]['dimension']))
77:     return vardef
78: 
79: 
80: def createfuncwrapper(rout, signature=0):
81:     assert isfunction(rout)
82: 
83:     extra_args = []
84:     vars = rout['vars']
85:     for a in rout['args']:
86:         v = rout['vars'][a]
87:         for i, d in enumerate(v.get('dimension', [])):
88:             if d == ':':
89:                 dn = 'f2py_%s_d%s' % (a, i)
90:                 dv = dict(typespec='integer', intent=['hide'])
91:                 dv['='] = 'shape(%s, %s)' % (a, i)
92:                 extra_args.append(dn)
93:                 vars[dn] = dv
94:                 v['dimension'][i] = dn
95:     rout['args'].extend(extra_args)
96:     need_interface = bool(extra_args)
97: 
98:     ret = ['']
99: 
100:     def add(line, ret=ret):
101:         ret[0] = '%s\n      %s' % (ret[0], line)
102:     name = rout['name']
103:     fortranname = getfortranname(rout)
104:     f90mode = ismoduleroutine(rout)
105:     newname = '%sf2pywrap' % (name)
106: 
107:     if newname not in vars:
108:         vars[newname] = vars[name]
109:         args = [newname] + rout['args'][1:]
110:     else:
111:         args = [newname] + rout['args']
112: 
113:     l = var2fixfortran(vars, name, newname, f90mode)
114:     if l[:13] == 'character*(*)':
115:         if f90mode:
116:             l = 'character(len=10)' + l[13:]
117:         else:
118:             l = 'character*10' + l[13:]
119:         charselect = vars[name]['charselector']
120:         if charselect.get('*', '') == '(*)':
121:             charselect['*'] = '10'
122:     sargs = ', '.join(args)
123:     if f90mode:
124:         add('subroutine f2pywrap_%s_%s (%s)' %
125:             (rout['modulename'], name, sargs))
126:         if not signature:
127:             add('use %s, only : %s' % (rout['modulename'], fortranname))
128:     else:
129:         add('subroutine f2pywrap%s (%s)' % (name, sargs))
130:         if not need_interface:
131:             add('external %s' % (fortranname))
132:             l = l + ', ' + fortranname
133:     if need_interface:
134:         for line in rout['saved_interface'].split('\n'):
135:             if line.lstrip().startswith('use '):
136:                 add(line)
137: 
138:     args = args[1:]
139:     dumped_args = []
140:     for a in args:
141:         if isexternal(vars[a]):
142:             add('external %s' % (a))
143:             dumped_args.append(a)
144:     for a in args:
145:         if a in dumped_args:
146:             continue
147:         if isscalar(vars[a]):
148:             add(var2fixfortran(vars, a, f90mode=f90mode))
149:             dumped_args.append(a)
150:     for a in args:
151:         if a in dumped_args:
152:             continue
153:         if isintent_in(vars[a]):
154:             add(var2fixfortran(vars, a, f90mode=f90mode))
155:             dumped_args.append(a)
156:     for a in args:
157:         if a in dumped_args:
158:             continue
159:         add(var2fixfortran(vars, a, f90mode=f90mode))
160: 
161:     add(l)
162: 
163:     if need_interface:
164:         if f90mode:
165:             # f90 module already defines needed interface
166:             pass
167:         else:
168:             add('interface')
169:             add(rout['saved_interface'].lstrip())
170:             add('end interface')
171: 
172:     sargs = ', '.join([a for a in args if a not in extra_args])
173: 
174:     if not signature:
175:         if islogicalfunction(rout):
176:             add('%s = .not.(.not.%s(%s))' % (newname, fortranname, sargs))
177:         else:
178:             add('%s = %s(%s)' % (newname, fortranname, sargs))
179:     if f90mode:
180:         add('end subroutine f2pywrap_%s_%s' % (rout['modulename'], name))
181:     else:
182:         add('end')
183:     return ret[0]
184: 
185: 
186: def createsubrwrapper(rout, signature=0):
187:     assert issubroutine(rout)
188: 
189:     extra_args = []
190:     vars = rout['vars']
191:     for a in rout['args']:
192:         v = rout['vars'][a]
193:         for i, d in enumerate(v.get('dimension', [])):
194:             if d == ':':
195:                 dn = 'f2py_%s_d%s' % (a, i)
196:                 dv = dict(typespec='integer', intent=['hide'])
197:                 dv['='] = 'shape(%s, %s)' % (a, i)
198:                 extra_args.append(dn)
199:                 vars[dn] = dv
200:                 v['dimension'][i] = dn
201:     rout['args'].extend(extra_args)
202:     need_interface = bool(extra_args)
203: 
204:     ret = ['']
205: 
206:     def add(line, ret=ret):
207:         ret[0] = '%s\n      %s' % (ret[0], line)
208:     name = rout['name']
209:     fortranname = getfortranname(rout)
210:     f90mode = ismoduleroutine(rout)
211: 
212:     args = rout['args']
213: 
214:     sargs = ', '.join(args)
215:     if f90mode:
216:         add('subroutine f2pywrap_%s_%s (%s)' %
217:             (rout['modulename'], name, sargs))
218:         if not signature:
219:             add('use %s, only : %s' % (rout['modulename'], fortranname))
220:     else:
221:         add('subroutine f2pywrap%s (%s)' % (name, sargs))
222:         if not need_interface:
223:             add('external %s' % (fortranname))
224: 
225:     if need_interface:
226:         for line in rout['saved_interface'].split('\n'):
227:             if line.lstrip().startswith('use '):
228:                 add(line)
229: 
230:     dumped_args = []
231:     for a in args:
232:         if isexternal(vars[a]):
233:             add('external %s' % (a))
234:             dumped_args.append(a)
235:     for a in args:
236:         if a in dumped_args:
237:             continue
238:         if isscalar(vars[a]):
239:             add(var2fixfortran(vars, a, f90mode=f90mode))
240:             dumped_args.append(a)
241:     for a in args:
242:         if a in dumped_args:
243:             continue
244:         add(var2fixfortran(vars, a, f90mode=f90mode))
245: 
246:     if need_interface:
247:         if f90mode:
248:             # f90 module already defines needed interface
249:             pass
250:         else:
251:             add('interface')
252:             add(rout['saved_interface'].lstrip())
253:             add('end interface')
254: 
255:     sargs = ', '.join([a for a in args if a not in extra_args])
256: 
257:     if not signature:
258:         add('call %s(%s)' % (fortranname, sargs))
259:     if f90mode:
260:         add('end subroutine f2pywrap_%s_%s' % (rout['modulename'], name))
261:     else:
262:         add('end')
263:     return ret[0]
264: 
265: 
266: def assubr(rout):
267:     if isfunction_wrap(rout):
268:         fortranname = getfortranname(rout)
269:         name = rout['name']
270:         outmess('\t\tCreating wrapper for Fortran function "%s"("%s")...\n' % (
271:             name, fortranname))
272:         rout = copy.copy(rout)
273:         fname = name
274:         rname = fname
275:         if 'result' in rout:
276:             rname = rout['result']
277:             rout['vars'][fname] = rout['vars'][rname]
278:         fvar = rout['vars'][fname]
279:         if not isintent_out(fvar):
280:             if 'intent' not in fvar:
281:                 fvar['intent'] = []
282:             fvar['intent'].append('out')
283:             flag = 1
284:             for i in fvar['intent']:
285:                 if i.startswith('out='):
286:                     flag = 0
287:                     break
288:             if flag:
289:                 fvar['intent'].append('out=%s' % (rname))
290:         rout['args'][:] = [fname] + rout['args']
291:         return rout, createfuncwrapper(rout)
292:     if issubroutine_wrap(rout):
293:         fortranname = getfortranname(rout)
294:         name = rout['name']
295:         outmess('\t\tCreating wrapper for Fortran subroutine "%s"("%s")...\n' % (
296:             name, fortranname))
297:         rout = copy.copy(rout)
298:         return rout, createsubrwrapper(rout)
299:     return rout, ''
300: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_94304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, (-1)), 'str', '\n\nRules for building C/API module with f2py2e.\n\nCopyright 1999,2000 Pearu Peterson all rights reserved,\nPearu Peterson <pearu@ioc.ee>\nPermission to use, modify, and distribute this software is given under the\nterms of the NumPy License.\n\nNO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.\n$Date: 2004/11/26 11:13:06 $\nPearu Peterson\n\n')

# Assigning a Subscript to a Name (line 18):

# Obtaining the type of the subscript
int_94305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 34), 'int')
int_94306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 37), 'int')
slice_94307 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 18, 14), int_94305, int_94306, None)
str_94308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 14), 'str', '$Revision: 1.16 $')
# Obtaining the member '__getitem__' of a type (line 18)
getitem___94309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 14), str_94308, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 18)
subscript_call_result_94310 = invoke(stypy.reporting.localization.Localization(__file__, 18, 14), getitem___94309, slice_94307)

# Assigning a type to the variable '__version__' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), '__version__', subscript_call_result_94310)

# Assigning a Str to a Name (line 20):
str_94311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 15), 'str', 'See `f2py -v`')
# Assigning a type to the variable 'f2py_version' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'f2py_version', str_94311)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'import copy' statement (line 22)
import copy

import_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'copy', copy, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from numpy.f2py.auxfuncs import getfortranname, isexternal, isfunction, isfunction_wrap, isintent_in, isintent_out, islogicalfunction, ismoduleroutine, isscalar, issubroutine, issubroutine_wrap, outmess, show' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_94312 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.f2py.auxfuncs')

if (type(import_94312) is not StypyTypeError):

    if (import_94312 != 'pyd_module'):
        __import__(import_94312)
        sys_modules_94313 = sys.modules[import_94312]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.f2py.auxfuncs', sys_modules_94313.module_type_store, module_type_store, ['getfortranname', 'isexternal', 'isfunction', 'isfunction_wrap', 'isintent_in', 'isintent_out', 'islogicalfunction', 'ismoduleroutine', 'isscalar', 'issubroutine', 'issubroutine_wrap', 'outmess', 'show'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_94313, sys_modules_94313.module_type_store, module_type_store)
    else:
        from numpy.f2py.auxfuncs import getfortranname, isexternal, isfunction, isfunction_wrap, isintent_in, isintent_out, islogicalfunction, ismoduleroutine, isscalar, issubroutine, issubroutine_wrap, outmess, show

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.f2py.auxfuncs', None, module_type_store, ['getfortranname', 'isexternal', 'isfunction', 'isfunction_wrap', 'isintent_in', 'isintent_out', 'islogicalfunction', 'ismoduleroutine', 'isscalar', 'issubroutine', 'issubroutine_wrap', 'outmess', 'show'], [getfortranname, isexternal, isfunction, isfunction_wrap, isintent_in, isintent_out, islogicalfunction, ismoduleroutine, isscalar, issubroutine, issubroutine_wrap, outmess, show])

else:
    # Assigning a type to the variable 'numpy.f2py.auxfuncs' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.f2py.auxfuncs', import_94312)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')


@norecursion
def var2fixfortran(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 31)
    None_94314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 31), 'None')
    # Getting the type of 'None' (line 31)
    None_94315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 45), 'None')
    defaults = [None_94314, None_94315]
    # Create a new context for function 'var2fixfortran'
    module_type_store = module_type_store.open_function_context('var2fixfortran', 31, 0, False)
    
    # Passed parameters checking function
    var2fixfortran.stypy_localization = localization
    var2fixfortran.stypy_type_of_self = None
    var2fixfortran.stypy_type_store = module_type_store
    var2fixfortran.stypy_function_name = 'var2fixfortran'
    var2fixfortran.stypy_param_names_list = ['vars', 'a', 'fa', 'f90mode']
    var2fixfortran.stypy_varargs_param_name = None
    var2fixfortran.stypy_kwargs_param_name = None
    var2fixfortran.stypy_call_defaults = defaults
    var2fixfortran.stypy_call_varargs = varargs
    var2fixfortran.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'var2fixfortran', ['vars', 'a', 'fa', 'f90mode'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'var2fixfortran', localization, ['vars', 'a', 'fa', 'f90mode'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'var2fixfortran(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 32)
    # Getting the type of 'fa' (line 32)
    fa_94316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 7), 'fa')
    # Getting the type of 'None' (line 32)
    None_94317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 13), 'None')
    
    (may_be_94318, more_types_in_union_94319) = may_be_none(fa_94316, None_94317)

    if may_be_94318:

        if more_types_in_union_94319:
            # Runtime conditional SSA (line 32)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 33):
        # Getting the type of 'a' (line 33)
        a_94320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'a')
        # Assigning a type to the variable 'fa' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'fa', a_94320)

        if more_types_in_union_94319:
            # SSA join for if statement (line 32)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'a' (line 34)
    a_94321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 7), 'a')
    # Getting the type of 'vars' (line 34)
    vars_94322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'vars')
    # Applying the binary operator 'notin' (line 34)
    result_contains_94323 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 7), 'notin', a_94321, vars_94322)
    
    # Testing the type of an if condition (line 34)
    if_condition_94324 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 4), result_contains_94323)
    # Assigning a type to the variable 'if_condition_94324' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'if_condition_94324', if_condition_94324)
    # SSA begins for if statement (line 34)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to show(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'vars' (line 35)
    vars_94326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 13), 'vars', False)
    # Processing the call keyword arguments (line 35)
    kwargs_94327 = {}
    # Getting the type of 'show' (line 35)
    show_94325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'show', False)
    # Calling show(args, kwargs) (line 35)
    show_call_result_94328 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), show_94325, *[vars_94326], **kwargs_94327)
    
    
    # Call to outmess(...): (line 36)
    # Processing the call arguments (line 36)
    str_94330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 16), 'str', 'var2fixfortran: No definition for argument "%s".\n')
    # Getting the type of 'a' (line 36)
    a_94331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 71), 'a', False)
    # Applying the binary operator '%' (line 36)
    result_mod_94332 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 16), '%', str_94330, a_94331)
    
    # Processing the call keyword arguments (line 36)
    kwargs_94333 = {}
    # Getting the type of 'outmess' (line 36)
    outmess_94329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'outmess', False)
    # Calling outmess(args, kwargs) (line 36)
    outmess_call_result_94334 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), outmess_94329, *[result_mod_94332], **kwargs_94333)
    
    str_94335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 15), 'str', '')
    # Assigning a type to the variable 'stypy_return_type' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'stypy_return_type', str_94335)
    # SSA join for if statement (line 34)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_94336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 7), 'str', 'typespec')
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 38)
    a_94337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 30), 'a')
    # Getting the type of 'vars' (line 38)
    vars_94338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 25), 'vars')
    # Obtaining the member '__getitem__' of a type (line 38)
    getitem___94339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 25), vars_94338, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 38)
    subscript_call_result_94340 = invoke(stypy.reporting.localization.Localization(__file__, 38, 25), getitem___94339, a_94337)
    
    # Applying the binary operator 'notin' (line 38)
    result_contains_94341 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 7), 'notin', str_94336, subscript_call_result_94340)
    
    # Testing the type of an if condition (line 38)
    if_condition_94342 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 4), result_contains_94341)
    # Assigning a type to the variable 'if_condition_94342' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'if_condition_94342', if_condition_94342)
    # SSA begins for if statement (line 38)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to show(...): (line 39)
    # Processing the call arguments (line 39)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 39)
    a_94344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 18), 'a', False)
    # Getting the type of 'vars' (line 39)
    vars_94345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 13), 'vars', False)
    # Obtaining the member '__getitem__' of a type (line 39)
    getitem___94346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 13), vars_94345, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 39)
    subscript_call_result_94347 = invoke(stypy.reporting.localization.Localization(__file__, 39, 13), getitem___94346, a_94344)
    
    # Processing the call keyword arguments (line 39)
    kwargs_94348 = {}
    # Getting the type of 'show' (line 39)
    show_94343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'show', False)
    # Calling show(args, kwargs) (line 39)
    show_call_result_94349 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), show_94343, *[subscript_call_result_94347], **kwargs_94348)
    
    
    # Call to outmess(...): (line 40)
    # Processing the call arguments (line 40)
    str_94351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 16), 'str', 'var2fixfortran: No typespec for argument "%s".\n')
    # Getting the type of 'a' (line 40)
    a_94352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 69), 'a', False)
    # Applying the binary operator '%' (line 40)
    result_mod_94353 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 16), '%', str_94351, a_94352)
    
    # Processing the call keyword arguments (line 40)
    kwargs_94354 = {}
    # Getting the type of 'outmess' (line 40)
    outmess_94350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'outmess', False)
    # Calling outmess(args, kwargs) (line 40)
    outmess_call_result_94355 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), outmess_94350, *[result_mod_94353], **kwargs_94354)
    
    str_94356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 15), 'str', '')
    # Assigning a type to the variable 'stypy_return_type' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'stypy_return_type', str_94356)
    # SSA join for if statement (line 38)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 42):
    
    # Obtaining the type of the subscript
    str_94357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 21), 'str', 'typespec')
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 42)
    a_94358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 18), 'a')
    # Getting the type of 'vars' (line 42)
    vars_94359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'vars')
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___94360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 13), vars_94359, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_94361 = invoke(stypy.reporting.localization.Localization(__file__, 42, 13), getitem___94360, a_94358)
    
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___94362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 13), subscript_call_result_94361, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_94363 = invoke(stypy.reporting.localization.Localization(__file__, 42, 13), getitem___94362, str_94357)
    
    # Assigning a type to the variable 'vardef' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'vardef', subscript_call_result_94363)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'vardef' (line 43)
    vardef_94364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 7), 'vardef')
    str_94365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 17), 'str', 'type')
    # Applying the binary operator '==' (line 43)
    result_eq_94366 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 7), '==', vardef_94364, str_94365)
    
    
    str_94367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 28), 'str', 'typename')
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 43)
    a_94368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 47), 'a')
    # Getting the type of 'vars' (line 43)
    vars_94369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 42), 'vars')
    # Obtaining the member '__getitem__' of a type (line 43)
    getitem___94370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 42), vars_94369, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
    subscript_call_result_94371 = invoke(stypy.reporting.localization.Localization(__file__, 43, 42), getitem___94370, a_94368)
    
    # Applying the binary operator 'in' (line 43)
    result_contains_94372 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 28), 'in', str_94367, subscript_call_result_94371)
    
    # Applying the binary operator 'and' (line 43)
    result_and_keyword_94373 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 7), 'and', result_eq_94366, result_contains_94372)
    
    # Testing the type of an if condition (line 43)
    if_condition_94374 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 4), result_and_keyword_94373)
    # Assigning a type to the variable 'if_condition_94374' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'if_condition_94374', if_condition_94374)
    # SSA begins for if statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 44):
    str_94375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 17), 'str', '%s(%s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 44)
    tuple_94376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 44)
    # Adding element type (line 44)
    # Getting the type of 'vardef' (line 44)
    vardef_94377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 29), 'vardef')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 29), tuple_94376, vardef_94377)
    # Adding element type (line 44)
    
    # Obtaining the type of the subscript
    str_94378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 45), 'str', 'typename')
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 44)
    a_94379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 42), 'a')
    # Getting the type of 'vars' (line 44)
    vars_94380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 37), 'vars')
    # Obtaining the member '__getitem__' of a type (line 44)
    getitem___94381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 37), vars_94380, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
    subscript_call_result_94382 = invoke(stypy.reporting.localization.Localization(__file__, 44, 37), getitem___94381, a_94379)
    
    # Obtaining the member '__getitem__' of a type (line 44)
    getitem___94383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 37), subscript_call_result_94382, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
    subscript_call_result_94384 = invoke(stypy.reporting.localization.Localization(__file__, 44, 37), getitem___94383, str_94378)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 29), tuple_94376, subscript_call_result_94384)
    
    # Applying the binary operator '%' (line 44)
    result_mod_94385 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 17), '%', str_94375, tuple_94376)
    
    # Assigning a type to the variable 'vardef' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'vardef', result_mod_94385)
    # SSA join for if statement (line 43)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Name (line 45):
    
    # Obtaining an instance of the builtin type 'dict' (line 45)
    dict_94386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 45)
    
    # Assigning a type to the variable 'selector' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'selector', dict_94386)
    
    # Assigning a Str to a Name (line 46):
    str_94387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 9), 'str', '')
    # Assigning a type to the variable 'lk' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'lk', str_94387)
    
    
    str_94388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 7), 'str', 'kindselector')
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 47)
    a_94389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 30), 'a')
    # Getting the type of 'vars' (line 47)
    vars_94390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 25), 'vars')
    # Obtaining the member '__getitem__' of a type (line 47)
    getitem___94391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 25), vars_94390, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 47)
    subscript_call_result_94392 = invoke(stypy.reporting.localization.Localization(__file__, 47, 25), getitem___94391, a_94389)
    
    # Applying the binary operator 'in' (line 47)
    result_contains_94393 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 7), 'in', str_94388, subscript_call_result_94392)
    
    # Testing the type of an if condition (line 47)
    if_condition_94394 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 4), result_contains_94393)
    # Assigning a type to the variable 'if_condition_94394' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'if_condition_94394', if_condition_94394)
    # SSA begins for if statement (line 47)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 48):
    
    # Obtaining the type of the subscript
    str_94395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 27), 'str', 'kindselector')
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 48)
    a_94396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 24), 'a')
    # Getting the type of 'vars' (line 48)
    vars_94397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'vars')
    # Obtaining the member '__getitem__' of a type (line 48)
    getitem___94398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 19), vars_94397, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 48)
    subscript_call_result_94399 = invoke(stypy.reporting.localization.Localization(__file__, 48, 19), getitem___94398, a_94396)
    
    # Obtaining the member '__getitem__' of a type (line 48)
    getitem___94400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 19), subscript_call_result_94399, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 48)
    subscript_call_result_94401 = invoke(stypy.reporting.localization.Localization(__file__, 48, 19), getitem___94400, str_94395)
    
    # Assigning a type to the variable 'selector' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'selector', subscript_call_result_94401)
    
    # Assigning a Str to a Name (line 49):
    str_94402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 13), 'str', 'kind')
    # Assigning a type to the variable 'lk' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'lk', str_94402)
    # SSA branch for the else part of an if statement (line 47)
    module_type_store.open_ssa_branch('else')
    
    
    str_94403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 9), 'str', 'charselector')
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 50)
    a_94404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 32), 'a')
    # Getting the type of 'vars' (line 50)
    vars_94405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 27), 'vars')
    # Obtaining the member '__getitem__' of a type (line 50)
    getitem___94406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 27), vars_94405, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 50)
    subscript_call_result_94407 = invoke(stypy.reporting.localization.Localization(__file__, 50, 27), getitem___94406, a_94404)
    
    # Applying the binary operator 'in' (line 50)
    result_contains_94408 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 9), 'in', str_94403, subscript_call_result_94407)
    
    # Testing the type of an if condition (line 50)
    if_condition_94409 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 9), result_contains_94408)
    # Assigning a type to the variable 'if_condition_94409' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 9), 'if_condition_94409', if_condition_94409)
    # SSA begins for if statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 51):
    
    # Obtaining the type of the subscript
    str_94410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 27), 'str', 'charselector')
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 51)
    a_94411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 24), 'a')
    # Getting the type of 'vars' (line 51)
    vars_94412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), 'vars')
    # Obtaining the member '__getitem__' of a type (line 51)
    getitem___94413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 19), vars_94412, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 51)
    subscript_call_result_94414 = invoke(stypy.reporting.localization.Localization(__file__, 51, 19), getitem___94413, a_94411)
    
    # Obtaining the member '__getitem__' of a type (line 51)
    getitem___94415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 19), subscript_call_result_94414, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 51)
    subscript_call_result_94416 = invoke(stypy.reporting.localization.Localization(__file__, 51, 19), getitem___94415, str_94410)
    
    # Assigning a type to the variable 'selector' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'selector', subscript_call_result_94416)
    
    # Assigning a Str to a Name (line 52):
    str_94417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 13), 'str', 'len')
    # Assigning a type to the variable 'lk' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'lk', str_94417)
    # SSA join for if statement (line 50)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 47)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_94418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 7), 'str', '*')
    # Getting the type of 'selector' (line 53)
    selector_94419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 14), 'selector')
    # Applying the binary operator 'in' (line 53)
    result_contains_94420 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 7), 'in', str_94418, selector_94419)
    
    # Testing the type of an if condition (line 53)
    if_condition_94421 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 4), result_contains_94420)
    # Assigning a type to the variable 'if_condition_94421' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'if_condition_94421', if_condition_94421)
    # SSA begins for if statement (line 53)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'f90mode' (line 54)
    f90mode_94422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 11), 'f90mode')
    # Testing the type of an if condition (line 54)
    if_condition_94423 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 8), f90mode_94422)
    # Assigning a type to the variable 'if_condition_94423' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'if_condition_94423', if_condition_94423)
    # SSA begins for if statement (line 54)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    str_94424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 24), 'str', '*')
    # Getting the type of 'selector' (line 55)
    selector_94425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 15), 'selector')
    # Obtaining the member '__getitem__' of a type (line 55)
    getitem___94426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 15), selector_94425, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 55)
    subscript_call_result_94427 = invoke(stypy.reporting.localization.Localization(__file__, 55, 15), getitem___94426, str_94424)
    
    
    # Obtaining an instance of the builtin type 'list' (line 55)
    list_94428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 55)
    # Adding element type (line 55)
    str_94429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 33), 'str', '*')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 32), list_94428, str_94429)
    # Adding element type (line 55)
    str_94430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 38), 'str', ':')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 32), list_94428, str_94430)
    # Adding element type (line 55)
    str_94431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 43), 'str', '(*)')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 32), list_94428, str_94431)
    
    # Applying the binary operator 'in' (line 55)
    result_contains_94432 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 15), 'in', subscript_call_result_94427, list_94428)
    
    # Testing the type of an if condition (line 55)
    if_condition_94433 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 55, 12), result_contains_94432)
    # Assigning a type to the variable 'if_condition_94433' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'if_condition_94433', if_condition_94433)
    # SSA begins for if statement (line 55)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 56):
    str_94434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 25), 'str', '%s(len=*)')
    # Getting the type of 'vardef' (line 56)
    vardef_94435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 40), 'vardef')
    # Applying the binary operator '%' (line 56)
    result_mod_94436 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 25), '%', str_94434, vardef_94435)
    
    # Assigning a type to the variable 'vardef' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'vardef', result_mod_94436)
    # SSA branch for the else part of an if statement (line 55)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 58):
    str_94437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 25), 'str', '%s(%s=%s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 58)
    tuple_94438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 58)
    # Adding element type (line 58)
    # Getting the type of 'vardef' (line 58)
    vardef_94439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 40), 'vardef')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 40), tuple_94438, vardef_94439)
    # Adding element type (line 58)
    # Getting the type of 'lk' (line 58)
    lk_94440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 48), 'lk')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 40), tuple_94438, lk_94440)
    # Adding element type (line 58)
    
    # Obtaining the type of the subscript
    str_94441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 61), 'str', '*')
    # Getting the type of 'selector' (line 58)
    selector_94442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 52), 'selector')
    # Obtaining the member '__getitem__' of a type (line 58)
    getitem___94443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 52), selector_94442, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 58)
    subscript_call_result_94444 = invoke(stypy.reporting.localization.Localization(__file__, 58, 52), getitem___94443, str_94441)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 40), tuple_94438, subscript_call_result_94444)
    
    # Applying the binary operator '%' (line 58)
    result_mod_94445 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 25), '%', str_94437, tuple_94438)
    
    # Assigning a type to the variable 'vardef' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'vardef', result_mod_94445)
    # SSA join for if statement (line 55)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 54)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    str_94446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 24), 'str', '*')
    # Getting the type of 'selector' (line 60)
    selector_94447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'selector')
    # Obtaining the member '__getitem__' of a type (line 60)
    getitem___94448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 15), selector_94447, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 60)
    subscript_call_result_94449 = invoke(stypy.reporting.localization.Localization(__file__, 60, 15), getitem___94448, str_94446)
    
    
    # Obtaining an instance of the builtin type 'list' (line 60)
    list_94450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 60)
    # Adding element type (line 60)
    str_94451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 33), 'str', '*')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 32), list_94450, str_94451)
    # Adding element type (line 60)
    str_94452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 38), 'str', ':')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 32), list_94450, str_94452)
    
    # Applying the binary operator 'in' (line 60)
    result_contains_94453 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 15), 'in', subscript_call_result_94449, list_94450)
    
    # Testing the type of an if condition (line 60)
    if_condition_94454 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 12), result_contains_94453)
    # Assigning a type to the variable 'if_condition_94454' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'if_condition_94454', if_condition_94454)
    # SSA begins for if statement (line 60)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 61):
    str_94455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 25), 'str', '%s*(%s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 61)
    tuple_94456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 61)
    # Adding element type (line 61)
    # Getting the type of 'vardef' (line 61)
    vardef_94457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 38), 'vardef')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 38), tuple_94456, vardef_94457)
    # Adding element type (line 61)
    
    # Obtaining the type of the subscript
    str_94458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 55), 'str', '*')
    # Getting the type of 'selector' (line 61)
    selector_94459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 46), 'selector')
    # Obtaining the member '__getitem__' of a type (line 61)
    getitem___94460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 46), selector_94459, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 61)
    subscript_call_result_94461 = invoke(stypy.reporting.localization.Localization(__file__, 61, 46), getitem___94460, str_94458)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 38), tuple_94456, subscript_call_result_94461)
    
    # Applying the binary operator '%' (line 61)
    result_mod_94462 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 25), '%', str_94455, tuple_94456)
    
    # Assigning a type to the variable 'vardef' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'vardef', result_mod_94462)
    # SSA branch for the else part of an if statement (line 60)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 63):
    str_94463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 25), 'str', '%s*%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 63)
    tuple_94464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 63)
    # Adding element type (line 63)
    # Getting the type of 'vardef' (line 63)
    vardef_94465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 36), 'vardef')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 36), tuple_94464, vardef_94465)
    # Adding element type (line 63)
    
    # Obtaining the type of the subscript
    str_94466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 53), 'str', '*')
    # Getting the type of 'selector' (line 63)
    selector_94467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 44), 'selector')
    # Obtaining the member '__getitem__' of a type (line 63)
    getitem___94468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 44), selector_94467, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 63)
    subscript_call_result_94469 = invoke(stypy.reporting.localization.Localization(__file__, 63, 44), getitem___94468, str_94466)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 36), tuple_94464, subscript_call_result_94469)
    
    # Applying the binary operator '%' (line 63)
    result_mod_94470 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 25), '%', str_94463, tuple_94464)
    
    # Assigning a type to the variable 'vardef' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 'vardef', result_mod_94470)
    # SSA join for if statement (line 60)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 54)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 53)
    module_type_store.open_ssa_branch('else')
    
    
    str_94471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 11), 'str', 'len')
    # Getting the type of 'selector' (line 65)
    selector_94472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'selector')
    # Applying the binary operator 'in' (line 65)
    result_contains_94473 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 11), 'in', str_94471, selector_94472)
    
    # Testing the type of an if condition (line 65)
    if_condition_94474 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 8), result_contains_94473)
    # Assigning a type to the variable 'if_condition_94474' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'if_condition_94474', if_condition_94474)
    # SSA begins for if statement (line 65)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 66):
    str_94475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 21), 'str', '%s(len=%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 66)
    tuple_94476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 66)
    # Adding element type (line 66)
    # Getting the type of 'vardef' (line 66)
    vardef_94477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 36), 'vardef')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 36), tuple_94476, vardef_94477)
    # Adding element type (line 66)
    
    # Obtaining the type of the subscript
    str_94478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 53), 'str', 'len')
    # Getting the type of 'selector' (line 66)
    selector_94479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 44), 'selector')
    # Obtaining the member '__getitem__' of a type (line 66)
    getitem___94480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 44), selector_94479, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 66)
    subscript_call_result_94481 = invoke(stypy.reporting.localization.Localization(__file__, 66, 44), getitem___94480, str_94478)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 36), tuple_94476, subscript_call_result_94481)
    
    # Applying the binary operator '%' (line 66)
    result_mod_94482 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 21), '%', str_94475, tuple_94476)
    
    # Assigning a type to the variable 'vardef' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'vardef', result_mod_94482)
    
    
    str_94483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 15), 'str', 'kind')
    # Getting the type of 'selector' (line 67)
    selector_94484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 25), 'selector')
    # Applying the binary operator 'in' (line 67)
    result_contains_94485 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 15), 'in', str_94483, selector_94484)
    
    # Testing the type of an if condition (line 67)
    if_condition_94486 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 12), result_contains_94485)
    # Assigning a type to the variable 'if_condition_94486' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'if_condition_94486', if_condition_94486)
    # SSA begins for if statement (line 67)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 68):
    str_94487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 25), 'str', '%s,kind=%s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 68)
    tuple_94488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 68)
    # Adding element type (line 68)
    # Getting the type of 'vardef' (line 68)
    vardef_94489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 42), 'vardef')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 42), tuple_94488, vardef_94489)
    # Adding element type (line 68)
    
    # Obtaining the type of the subscript
    str_94490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 59), 'str', 'kind')
    # Getting the type of 'selector' (line 68)
    selector_94491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 50), 'selector')
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___94492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 50), selector_94491, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_94493 = invoke(stypy.reporting.localization.Localization(__file__, 68, 50), getitem___94492, str_94490)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 42), tuple_94488, subscript_call_result_94493)
    
    # Applying the binary operator '%' (line 68)
    result_mod_94494 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 25), '%', str_94487, tuple_94488)
    
    # Assigning a type to the variable 'vardef' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'vardef', result_mod_94494)
    # SSA branch for the else part of an if statement (line 67)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 70):
    str_94495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 25), 'str', '%s)')
    # Getting the type of 'vardef' (line 70)
    vardef_94496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 34), 'vardef')
    # Applying the binary operator '%' (line 70)
    result_mod_94497 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 25), '%', str_94495, vardef_94496)
    
    # Assigning a type to the variable 'vardef' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'vardef', result_mod_94497)
    # SSA join for if statement (line 67)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 65)
    module_type_store.open_ssa_branch('else')
    
    
    str_94498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 13), 'str', 'kind')
    # Getting the type of 'selector' (line 71)
    selector_94499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 23), 'selector')
    # Applying the binary operator 'in' (line 71)
    result_contains_94500 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 13), 'in', str_94498, selector_94499)
    
    # Testing the type of an if condition (line 71)
    if_condition_94501 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 13), result_contains_94500)
    # Assigning a type to the variable 'if_condition_94501' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 13), 'if_condition_94501', if_condition_94501)
    # SSA begins for if statement (line 71)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 72):
    str_94502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 21), 'str', '%s(kind=%s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 72)
    tuple_94503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 72)
    # Adding element type (line 72)
    # Getting the type of 'vardef' (line 72)
    vardef_94504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 38), 'vardef')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 38), tuple_94503, vardef_94504)
    # Adding element type (line 72)
    
    # Obtaining the type of the subscript
    str_94505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 55), 'str', 'kind')
    # Getting the type of 'selector' (line 72)
    selector_94506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 46), 'selector')
    # Obtaining the member '__getitem__' of a type (line 72)
    getitem___94507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 46), selector_94506, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
    subscript_call_result_94508 = invoke(stypy.reporting.localization.Localization(__file__, 72, 46), getitem___94507, str_94505)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 38), tuple_94503, subscript_call_result_94508)
    
    # Applying the binary operator '%' (line 72)
    result_mod_94509 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 21), '%', str_94502, tuple_94503)
    
    # Assigning a type to the variable 'vardef' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'vardef', result_mod_94509)
    # SSA join for if statement (line 71)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 65)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 53)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 74):
    str_94510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 13), 'str', '%s %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 74)
    tuple_94511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 74)
    # Adding element type (line 74)
    # Getting the type of 'vardef' (line 74)
    vardef_94512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 24), 'vardef')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 24), tuple_94511, vardef_94512)
    # Adding element type (line 74)
    # Getting the type of 'fa' (line 74)
    fa_94513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 32), 'fa')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 24), tuple_94511, fa_94513)
    
    # Applying the binary operator '%' (line 74)
    result_mod_94514 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 13), '%', str_94510, tuple_94511)
    
    # Assigning a type to the variable 'vardef' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'vardef', result_mod_94514)
    
    
    str_94515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 7), 'str', 'dimension')
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 75)
    a_94516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 27), 'a')
    # Getting the type of 'vars' (line 75)
    vars_94517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), 'vars')
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___94518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 22), vars_94517, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_94519 = invoke(stypy.reporting.localization.Localization(__file__, 75, 22), getitem___94518, a_94516)
    
    # Applying the binary operator 'in' (line 75)
    result_contains_94520 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 7), 'in', str_94515, subscript_call_result_94519)
    
    # Testing the type of an if condition (line 75)
    if_condition_94521 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 4), result_contains_94520)
    # Assigning a type to the variable 'if_condition_94521' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'if_condition_94521', if_condition_94521)
    # SSA begins for if statement (line 75)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 76):
    str_94522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 17), 'str', '%s(%s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 76)
    tuple_94523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 76)
    # Adding element type (line 76)
    # Getting the type of 'vardef' (line 76)
    vardef_94524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 29), 'vardef')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 29), tuple_94523, vardef_94524)
    # Adding element type (line 76)
    
    # Call to join(...): (line 76)
    # Processing the call arguments (line 76)
    
    # Obtaining the type of the subscript
    str_94527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 54), 'str', 'dimension')
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 76)
    a_94528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 51), 'a', False)
    # Getting the type of 'vars' (line 76)
    vars_94529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 46), 'vars', False)
    # Obtaining the member '__getitem__' of a type (line 76)
    getitem___94530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 46), vars_94529, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 76)
    subscript_call_result_94531 = invoke(stypy.reporting.localization.Localization(__file__, 76, 46), getitem___94530, a_94528)
    
    # Obtaining the member '__getitem__' of a type (line 76)
    getitem___94532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 46), subscript_call_result_94531, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 76)
    subscript_call_result_94533 = invoke(stypy.reporting.localization.Localization(__file__, 76, 46), getitem___94532, str_94527)
    
    # Processing the call keyword arguments (line 76)
    kwargs_94534 = {}
    str_94525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 37), 'str', ',')
    # Obtaining the member 'join' of a type (line 76)
    join_94526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 37), str_94525, 'join')
    # Calling join(args, kwargs) (line 76)
    join_call_result_94535 = invoke(stypy.reporting.localization.Localization(__file__, 76, 37), join_94526, *[subscript_call_result_94533], **kwargs_94534)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 29), tuple_94523, join_call_result_94535)
    
    # Applying the binary operator '%' (line 76)
    result_mod_94536 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 17), '%', str_94522, tuple_94523)
    
    # Assigning a type to the variable 'vardef' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'vardef', result_mod_94536)
    # SSA join for if statement (line 75)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'vardef' (line 77)
    vardef_94537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'vardef')
    # Assigning a type to the variable 'stypy_return_type' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type', vardef_94537)
    
    # ################# End of 'var2fixfortran(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'var2fixfortran' in the type store
    # Getting the type of 'stypy_return_type' (line 31)
    stypy_return_type_94538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_94538)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'var2fixfortran'
    return stypy_return_type_94538

# Assigning a type to the variable 'var2fixfortran' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'var2fixfortran', var2fixfortran)

@norecursion
def createfuncwrapper(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_94539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 38), 'int')
    defaults = [int_94539]
    # Create a new context for function 'createfuncwrapper'
    module_type_store = module_type_store.open_function_context('createfuncwrapper', 80, 0, False)
    
    # Passed parameters checking function
    createfuncwrapper.stypy_localization = localization
    createfuncwrapper.stypy_type_of_self = None
    createfuncwrapper.stypy_type_store = module_type_store
    createfuncwrapper.stypy_function_name = 'createfuncwrapper'
    createfuncwrapper.stypy_param_names_list = ['rout', 'signature']
    createfuncwrapper.stypy_varargs_param_name = None
    createfuncwrapper.stypy_kwargs_param_name = None
    createfuncwrapper.stypy_call_defaults = defaults
    createfuncwrapper.stypy_call_varargs = varargs
    createfuncwrapper.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'createfuncwrapper', ['rout', 'signature'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'createfuncwrapper', localization, ['rout', 'signature'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'createfuncwrapper(...)' code ##################

    # Evaluating assert statement condition
    
    # Call to isfunction(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'rout' (line 81)
    rout_94541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 22), 'rout', False)
    # Processing the call keyword arguments (line 81)
    kwargs_94542 = {}
    # Getting the type of 'isfunction' (line 81)
    isfunction_94540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'isfunction', False)
    # Calling isfunction(args, kwargs) (line 81)
    isfunction_call_result_94543 = invoke(stypy.reporting.localization.Localization(__file__, 81, 11), isfunction_94540, *[rout_94541], **kwargs_94542)
    
    
    # Assigning a List to a Name (line 83):
    
    # Obtaining an instance of the builtin type 'list' (line 83)
    list_94544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 83)
    
    # Assigning a type to the variable 'extra_args' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'extra_args', list_94544)
    
    # Assigning a Subscript to a Name (line 84):
    
    # Obtaining the type of the subscript
    str_94545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 16), 'str', 'vars')
    # Getting the type of 'rout' (line 84)
    rout_94546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 11), 'rout')
    # Obtaining the member '__getitem__' of a type (line 84)
    getitem___94547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 11), rout_94546, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 84)
    subscript_call_result_94548 = invoke(stypy.reporting.localization.Localization(__file__, 84, 11), getitem___94547, str_94545)
    
    # Assigning a type to the variable 'vars' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'vars', subscript_call_result_94548)
    
    
    # Obtaining the type of the subscript
    str_94549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 18), 'str', 'args')
    # Getting the type of 'rout' (line 85)
    rout_94550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 13), 'rout')
    # Obtaining the member '__getitem__' of a type (line 85)
    getitem___94551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 13), rout_94550, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 85)
    subscript_call_result_94552 = invoke(stypy.reporting.localization.Localization(__file__, 85, 13), getitem___94551, str_94549)
    
    # Testing the type of a for loop iterable (line 85)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 85, 4), subscript_call_result_94552)
    # Getting the type of the for loop variable (line 85)
    for_loop_var_94553 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 85, 4), subscript_call_result_94552)
    # Assigning a type to the variable 'a' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'a', for_loop_var_94553)
    # SSA begins for a for statement (line 85)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 86):
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 86)
    a_94554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 25), 'a')
    
    # Obtaining the type of the subscript
    str_94555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 17), 'str', 'vars')
    # Getting the type of 'rout' (line 86)
    rout_94556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'rout')
    # Obtaining the member '__getitem__' of a type (line 86)
    getitem___94557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), rout_94556, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 86)
    subscript_call_result_94558 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), getitem___94557, str_94555)
    
    # Obtaining the member '__getitem__' of a type (line 86)
    getitem___94559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), subscript_call_result_94558, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 86)
    subscript_call_result_94560 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), getitem___94559, a_94554)
    
    # Assigning a type to the variable 'v' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'v', subscript_call_result_94560)
    
    
    # Call to enumerate(...): (line 87)
    # Processing the call arguments (line 87)
    
    # Call to get(...): (line 87)
    # Processing the call arguments (line 87)
    str_94564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 36), 'str', 'dimension')
    
    # Obtaining an instance of the builtin type 'list' (line 87)
    list_94565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 49), 'list')
    # Adding type elements to the builtin type 'list' instance (line 87)
    
    # Processing the call keyword arguments (line 87)
    kwargs_94566 = {}
    # Getting the type of 'v' (line 87)
    v_94562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 30), 'v', False)
    # Obtaining the member 'get' of a type (line 87)
    get_94563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 30), v_94562, 'get')
    # Calling get(args, kwargs) (line 87)
    get_call_result_94567 = invoke(stypy.reporting.localization.Localization(__file__, 87, 30), get_94563, *[str_94564, list_94565], **kwargs_94566)
    
    # Processing the call keyword arguments (line 87)
    kwargs_94568 = {}
    # Getting the type of 'enumerate' (line 87)
    enumerate_94561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 87)
    enumerate_call_result_94569 = invoke(stypy.reporting.localization.Localization(__file__, 87, 20), enumerate_94561, *[get_call_result_94567], **kwargs_94568)
    
    # Testing the type of a for loop iterable (line 87)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 87, 8), enumerate_call_result_94569)
    # Getting the type of the for loop variable (line 87)
    for_loop_var_94570 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 87, 8), enumerate_call_result_94569)
    # Assigning a type to the variable 'i' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 8), for_loop_var_94570))
    # Assigning a type to the variable 'd' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 8), for_loop_var_94570))
    # SSA begins for a for statement (line 87)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'd' (line 88)
    d_94571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'd')
    str_94572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 20), 'str', ':')
    # Applying the binary operator '==' (line 88)
    result_eq_94573 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 15), '==', d_94571, str_94572)
    
    # Testing the type of an if condition (line 88)
    if_condition_94574 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 12), result_eq_94573)
    # Assigning a type to the variable 'if_condition_94574' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'if_condition_94574', if_condition_94574)
    # SSA begins for if statement (line 88)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 89):
    str_94575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 21), 'str', 'f2py_%s_d%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 89)
    tuple_94576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 89)
    # Adding element type (line 89)
    # Getting the type of 'a' (line 89)
    a_94577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 38), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 38), tuple_94576, a_94577)
    # Adding element type (line 89)
    # Getting the type of 'i' (line 89)
    i_94578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 41), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 38), tuple_94576, i_94578)
    
    # Applying the binary operator '%' (line 89)
    result_mod_94579 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 21), '%', str_94575, tuple_94576)
    
    # Assigning a type to the variable 'dn' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'dn', result_mod_94579)
    
    # Assigning a Call to a Name (line 90):
    
    # Call to dict(...): (line 90)
    # Processing the call keyword arguments (line 90)
    str_94581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 35), 'str', 'integer')
    keyword_94582 = str_94581
    
    # Obtaining an instance of the builtin type 'list' (line 90)
    list_94583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 53), 'list')
    # Adding type elements to the builtin type 'list' instance (line 90)
    # Adding element type (line 90)
    str_94584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 54), 'str', 'hide')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 53), list_94583, str_94584)
    
    keyword_94585 = list_94583
    kwargs_94586 = {'typespec': keyword_94582, 'intent': keyword_94585}
    # Getting the type of 'dict' (line 90)
    dict_94580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 21), 'dict', False)
    # Calling dict(args, kwargs) (line 90)
    dict_call_result_94587 = invoke(stypy.reporting.localization.Localization(__file__, 90, 21), dict_94580, *[], **kwargs_94586)
    
    # Assigning a type to the variable 'dv' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 16), 'dv', dict_call_result_94587)
    
    # Assigning a BinOp to a Subscript (line 91):
    str_94588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 26), 'str', 'shape(%s, %s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 91)
    tuple_94589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 45), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 91)
    # Adding element type (line 91)
    # Getting the type of 'a' (line 91)
    a_94590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 45), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 45), tuple_94589, a_94590)
    # Adding element type (line 91)
    # Getting the type of 'i' (line 91)
    i_94591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 48), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 45), tuple_94589, i_94591)
    
    # Applying the binary operator '%' (line 91)
    result_mod_94592 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 26), '%', str_94588, tuple_94589)
    
    # Getting the type of 'dv' (line 91)
    dv_94593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'dv')
    str_94594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 19), 'str', '=')
    # Storing an element on a container (line 91)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 16), dv_94593, (str_94594, result_mod_94592))
    
    # Call to append(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'dn' (line 92)
    dn_94597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 34), 'dn', False)
    # Processing the call keyword arguments (line 92)
    kwargs_94598 = {}
    # Getting the type of 'extra_args' (line 92)
    extra_args_94595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'extra_args', False)
    # Obtaining the member 'append' of a type (line 92)
    append_94596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 16), extra_args_94595, 'append')
    # Calling append(args, kwargs) (line 92)
    append_call_result_94599 = invoke(stypy.reporting.localization.Localization(__file__, 92, 16), append_94596, *[dn_94597], **kwargs_94598)
    
    
    # Assigning a Name to a Subscript (line 93):
    # Getting the type of 'dv' (line 93)
    dv_94600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 27), 'dv')
    # Getting the type of 'vars' (line 93)
    vars_94601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'vars')
    # Getting the type of 'dn' (line 93)
    dn_94602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 21), 'dn')
    # Storing an element on a container (line 93)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 16), vars_94601, (dn_94602, dv_94600))
    
    # Assigning a Name to a Subscript (line 94):
    # Getting the type of 'dn' (line 94)
    dn_94603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 36), 'dn')
    
    # Obtaining the type of the subscript
    str_94604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 18), 'str', 'dimension')
    # Getting the type of 'v' (line 94)
    v_94605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 16), 'v')
    # Obtaining the member '__getitem__' of a type (line 94)
    getitem___94606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 16), v_94605, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 94)
    subscript_call_result_94607 = invoke(stypy.reporting.localization.Localization(__file__, 94, 16), getitem___94606, str_94604)
    
    # Getting the type of 'i' (line 94)
    i_94608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 31), 'i')
    # Storing an element on a container (line 94)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 16), subscript_call_result_94607, (i_94608, dn_94603))
    # SSA join for if statement (line 88)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to extend(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'extra_args' (line 95)
    extra_args_94614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 24), 'extra_args', False)
    # Processing the call keyword arguments (line 95)
    kwargs_94615 = {}
    
    # Obtaining the type of the subscript
    str_94609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 9), 'str', 'args')
    # Getting the type of 'rout' (line 95)
    rout_94610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 95)
    getitem___94611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 4), rout_94610, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 95)
    subscript_call_result_94612 = invoke(stypy.reporting.localization.Localization(__file__, 95, 4), getitem___94611, str_94609)
    
    # Obtaining the member 'extend' of a type (line 95)
    extend_94613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 4), subscript_call_result_94612, 'extend')
    # Calling extend(args, kwargs) (line 95)
    extend_call_result_94616 = invoke(stypy.reporting.localization.Localization(__file__, 95, 4), extend_94613, *[extra_args_94614], **kwargs_94615)
    
    
    # Assigning a Call to a Name (line 96):
    
    # Call to bool(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'extra_args' (line 96)
    extra_args_94618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'extra_args', False)
    # Processing the call keyword arguments (line 96)
    kwargs_94619 = {}
    # Getting the type of 'bool' (line 96)
    bool_94617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 21), 'bool', False)
    # Calling bool(args, kwargs) (line 96)
    bool_call_result_94620 = invoke(stypy.reporting.localization.Localization(__file__, 96, 21), bool_94617, *[extra_args_94618], **kwargs_94619)
    
    # Assigning a type to the variable 'need_interface' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'need_interface', bool_call_result_94620)
    
    # Assigning a List to a Name (line 98):
    
    # Obtaining an instance of the builtin type 'list' (line 98)
    list_94621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 98)
    # Adding element type (line 98)
    str_94622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 11), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 10), list_94621, str_94622)
    
    # Assigning a type to the variable 'ret' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'ret', list_94621)

    @norecursion
    def add(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'ret' (line 100)
        ret_94623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 22), 'ret')
        defaults = [ret_94623]
        # Create a new context for function 'add'
        module_type_store = module_type_store.open_function_context('add', 100, 4, False)
        
        # Passed parameters checking function
        add.stypy_localization = localization
        add.stypy_type_of_self = None
        add.stypy_type_store = module_type_store
        add.stypy_function_name = 'add'
        add.stypy_param_names_list = ['line', 'ret']
        add.stypy_varargs_param_name = None
        add.stypy_kwargs_param_name = None
        add.stypy_call_defaults = defaults
        add.stypy_call_varargs = varargs
        add.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'add', ['line', 'ret'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add', localization, ['line', 'ret'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add(...)' code ##################

        
        # Assigning a BinOp to a Subscript (line 101):
        str_94624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 17), 'str', '%s\n      %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 101)
        tuple_94625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 101)
        # Adding element type (line 101)
        
        # Obtaining the type of the subscript
        int_94626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 39), 'int')
        # Getting the type of 'ret' (line 101)
        ret_94627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 35), 'ret')
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___94628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 35), ret_94627, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_94629 = invoke(stypy.reporting.localization.Localization(__file__, 101, 35), getitem___94628, int_94626)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 35), tuple_94625, subscript_call_result_94629)
        # Adding element type (line 101)
        # Getting the type of 'line' (line 101)
        line_94630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 43), 'line')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 35), tuple_94625, line_94630)
        
        # Applying the binary operator '%' (line 101)
        result_mod_94631 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 17), '%', str_94624, tuple_94625)
        
        # Getting the type of 'ret' (line 101)
        ret_94632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'ret')
        int_94633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 12), 'int')
        # Storing an element on a container (line 101)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 8), ret_94632, (int_94633, result_mod_94631))
        
        # ################# End of 'add(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add' in the type store
        # Getting the type of 'stypy_return_type' (line 100)
        stypy_return_type_94634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_94634)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add'
        return stypy_return_type_94634

    # Assigning a type to the variable 'add' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'add', add)
    
    # Assigning a Subscript to a Name (line 102):
    
    # Obtaining the type of the subscript
    str_94635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 16), 'str', 'name')
    # Getting the type of 'rout' (line 102)
    rout_94636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 11), 'rout')
    # Obtaining the member '__getitem__' of a type (line 102)
    getitem___94637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 11), rout_94636, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 102)
    subscript_call_result_94638 = invoke(stypy.reporting.localization.Localization(__file__, 102, 11), getitem___94637, str_94635)
    
    # Assigning a type to the variable 'name' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'name', subscript_call_result_94638)
    
    # Assigning a Call to a Name (line 103):
    
    # Call to getfortranname(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'rout' (line 103)
    rout_94640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 33), 'rout', False)
    # Processing the call keyword arguments (line 103)
    kwargs_94641 = {}
    # Getting the type of 'getfortranname' (line 103)
    getfortranname_94639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 18), 'getfortranname', False)
    # Calling getfortranname(args, kwargs) (line 103)
    getfortranname_call_result_94642 = invoke(stypy.reporting.localization.Localization(__file__, 103, 18), getfortranname_94639, *[rout_94640], **kwargs_94641)
    
    # Assigning a type to the variable 'fortranname' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'fortranname', getfortranname_call_result_94642)
    
    # Assigning a Call to a Name (line 104):
    
    # Call to ismoduleroutine(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'rout' (line 104)
    rout_94644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 30), 'rout', False)
    # Processing the call keyword arguments (line 104)
    kwargs_94645 = {}
    # Getting the type of 'ismoduleroutine' (line 104)
    ismoduleroutine_94643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 14), 'ismoduleroutine', False)
    # Calling ismoduleroutine(args, kwargs) (line 104)
    ismoduleroutine_call_result_94646 = invoke(stypy.reporting.localization.Localization(__file__, 104, 14), ismoduleroutine_94643, *[rout_94644], **kwargs_94645)
    
    # Assigning a type to the variable 'f90mode' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'f90mode', ismoduleroutine_call_result_94646)
    
    # Assigning a BinOp to a Name (line 105):
    str_94647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 14), 'str', '%sf2pywrap')
    # Getting the type of 'name' (line 105)
    name_94648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 30), 'name')
    # Applying the binary operator '%' (line 105)
    result_mod_94649 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 14), '%', str_94647, name_94648)
    
    # Assigning a type to the variable 'newname' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'newname', result_mod_94649)
    
    
    # Getting the type of 'newname' (line 107)
    newname_94650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 7), 'newname')
    # Getting the type of 'vars' (line 107)
    vars_94651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 22), 'vars')
    # Applying the binary operator 'notin' (line 107)
    result_contains_94652 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 7), 'notin', newname_94650, vars_94651)
    
    # Testing the type of an if condition (line 107)
    if_condition_94653 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 4), result_contains_94652)
    # Assigning a type to the variable 'if_condition_94653' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'if_condition_94653', if_condition_94653)
    # SSA begins for if statement (line 107)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 108):
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 108)
    name_94654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 'name')
    # Getting the type of 'vars' (line 108)
    vars_94655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 24), 'vars')
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___94656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 24), vars_94655, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_94657 = invoke(stypy.reporting.localization.Localization(__file__, 108, 24), getitem___94656, name_94654)
    
    # Getting the type of 'vars' (line 108)
    vars_94658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'vars')
    # Getting the type of 'newname' (line 108)
    newname_94659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'newname')
    # Storing an element on a container (line 108)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 8), vars_94658, (newname_94659, subscript_call_result_94657))
    
    # Assigning a BinOp to a Name (line 109):
    
    # Obtaining an instance of the builtin type 'list' (line 109)
    list_94660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 109)
    # Adding element type (line 109)
    # Getting the type of 'newname' (line 109)
    newname_94661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'newname')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 15), list_94660, newname_94661)
    
    
    # Obtaining the type of the subscript
    int_94662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 40), 'int')
    slice_94663 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 109, 27), int_94662, None, None)
    
    # Obtaining the type of the subscript
    str_94664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 32), 'str', 'args')
    # Getting the type of 'rout' (line 109)
    rout_94665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'rout')
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___94666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), rout_94665, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_94667 = invoke(stypy.reporting.localization.Localization(__file__, 109, 27), getitem___94666, str_94664)
    
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___94668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), subscript_call_result_94667, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_94669 = invoke(stypy.reporting.localization.Localization(__file__, 109, 27), getitem___94668, slice_94663)
    
    # Applying the binary operator '+' (line 109)
    result_add_94670 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 15), '+', list_94660, subscript_call_result_94669)
    
    # Assigning a type to the variable 'args' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'args', result_add_94670)
    # SSA branch for the else part of an if statement (line 107)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 111):
    
    # Obtaining an instance of the builtin type 'list' (line 111)
    list_94671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 111)
    # Adding element type (line 111)
    # Getting the type of 'newname' (line 111)
    newname_94672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'newname')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 15), list_94671, newname_94672)
    
    
    # Obtaining the type of the subscript
    str_94673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 32), 'str', 'args')
    # Getting the type of 'rout' (line 111)
    rout_94674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 27), 'rout')
    # Obtaining the member '__getitem__' of a type (line 111)
    getitem___94675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 27), rout_94674, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
    subscript_call_result_94676 = invoke(stypy.reporting.localization.Localization(__file__, 111, 27), getitem___94675, str_94673)
    
    # Applying the binary operator '+' (line 111)
    result_add_94677 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 15), '+', list_94671, subscript_call_result_94676)
    
    # Assigning a type to the variable 'args' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'args', result_add_94677)
    # SSA join for if statement (line 107)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 113):
    
    # Call to var2fixfortran(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'vars' (line 113)
    vars_94679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 23), 'vars', False)
    # Getting the type of 'name' (line 113)
    name_94680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 29), 'name', False)
    # Getting the type of 'newname' (line 113)
    newname_94681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 35), 'newname', False)
    # Getting the type of 'f90mode' (line 113)
    f90mode_94682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 44), 'f90mode', False)
    # Processing the call keyword arguments (line 113)
    kwargs_94683 = {}
    # Getting the type of 'var2fixfortran' (line 113)
    var2fixfortran_94678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'var2fixfortran', False)
    # Calling var2fixfortran(args, kwargs) (line 113)
    var2fixfortran_call_result_94684 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), var2fixfortran_94678, *[vars_94679, name_94680, newname_94681, f90mode_94682], **kwargs_94683)
    
    # Assigning a type to the variable 'l' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'l', var2fixfortran_call_result_94684)
    
    
    
    # Obtaining the type of the subscript
    int_94685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 10), 'int')
    slice_94686 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 114, 7), None, int_94685, None)
    # Getting the type of 'l' (line 114)
    l_94687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 7), 'l')
    # Obtaining the member '__getitem__' of a type (line 114)
    getitem___94688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 7), l_94687, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 114)
    subscript_call_result_94689 = invoke(stypy.reporting.localization.Localization(__file__, 114, 7), getitem___94688, slice_94686)
    
    str_94690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 17), 'str', 'character*(*)')
    # Applying the binary operator '==' (line 114)
    result_eq_94691 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 7), '==', subscript_call_result_94689, str_94690)
    
    # Testing the type of an if condition (line 114)
    if_condition_94692 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 114, 4), result_eq_94691)
    # Assigning a type to the variable 'if_condition_94692' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'if_condition_94692', if_condition_94692)
    # SSA begins for if statement (line 114)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'f90mode' (line 115)
    f90mode_94693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'f90mode')
    # Testing the type of an if condition (line 115)
    if_condition_94694 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 8), f90mode_94693)
    # Assigning a type to the variable 'if_condition_94694' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'if_condition_94694', if_condition_94694)
    # SSA begins for if statement (line 115)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 116):
    str_94695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 16), 'str', 'character(len=10)')
    
    # Obtaining the type of the subscript
    int_94696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 40), 'int')
    slice_94697 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 116, 38), int_94696, None, None)
    # Getting the type of 'l' (line 116)
    l_94698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 38), 'l')
    # Obtaining the member '__getitem__' of a type (line 116)
    getitem___94699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 38), l_94698, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 116)
    subscript_call_result_94700 = invoke(stypy.reporting.localization.Localization(__file__, 116, 38), getitem___94699, slice_94697)
    
    # Applying the binary operator '+' (line 116)
    result_add_94701 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 16), '+', str_94695, subscript_call_result_94700)
    
    # Assigning a type to the variable 'l' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'l', result_add_94701)
    # SSA branch for the else part of an if statement (line 115)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 118):
    str_94702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 16), 'str', 'character*10')
    
    # Obtaining the type of the subscript
    int_94703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 35), 'int')
    slice_94704 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 118, 33), int_94703, None, None)
    # Getting the type of 'l' (line 118)
    l_94705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 33), 'l')
    # Obtaining the member '__getitem__' of a type (line 118)
    getitem___94706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 33), l_94705, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 118)
    subscript_call_result_94707 = invoke(stypy.reporting.localization.Localization(__file__, 118, 33), getitem___94706, slice_94704)
    
    # Applying the binary operator '+' (line 118)
    result_add_94708 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 16), '+', str_94702, subscript_call_result_94707)
    
    # Assigning a type to the variable 'l' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'l', result_add_94708)
    # SSA join for if statement (line 115)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 119):
    
    # Obtaining the type of the subscript
    str_94709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 32), 'str', 'charselector')
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 119)
    name_94710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 26), 'name')
    # Getting the type of 'vars' (line 119)
    vars_94711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 21), 'vars')
    # Obtaining the member '__getitem__' of a type (line 119)
    getitem___94712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 21), vars_94711, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 119)
    subscript_call_result_94713 = invoke(stypy.reporting.localization.Localization(__file__, 119, 21), getitem___94712, name_94710)
    
    # Obtaining the member '__getitem__' of a type (line 119)
    getitem___94714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 21), subscript_call_result_94713, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 119)
    subscript_call_result_94715 = invoke(stypy.reporting.localization.Localization(__file__, 119, 21), getitem___94714, str_94709)
    
    # Assigning a type to the variable 'charselect' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'charselect', subscript_call_result_94715)
    
    
    
    # Call to get(...): (line 120)
    # Processing the call arguments (line 120)
    str_94718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 26), 'str', '*')
    str_94719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 31), 'str', '')
    # Processing the call keyword arguments (line 120)
    kwargs_94720 = {}
    # Getting the type of 'charselect' (line 120)
    charselect_94716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 11), 'charselect', False)
    # Obtaining the member 'get' of a type (line 120)
    get_94717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 11), charselect_94716, 'get')
    # Calling get(args, kwargs) (line 120)
    get_call_result_94721 = invoke(stypy.reporting.localization.Localization(__file__, 120, 11), get_94717, *[str_94718, str_94719], **kwargs_94720)
    
    str_94722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 38), 'str', '(*)')
    # Applying the binary operator '==' (line 120)
    result_eq_94723 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 11), '==', get_call_result_94721, str_94722)
    
    # Testing the type of an if condition (line 120)
    if_condition_94724 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 120, 8), result_eq_94723)
    # Assigning a type to the variable 'if_condition_94724' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'if_condition_94724', if_condition_94724)
    # SSA begins for if statement (line 120)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Subscript (line 121):
    str_94725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 30), 'str', '10')
    # Getting the type of 'charselect' (line 121)
    charselect_94726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'charselect')
    str_94727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 23), 'str', '*')
    # Storing an element on a container (line 121)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 12), charselect_94726, (str_94727, str_94725))
    # SSA join for if statement (line 120)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 114)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 122):
    
    # Call to join(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'args' (line 122)
    args_94730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 22), 'args', False)
    # Processing the call keyword arguments (line 122)
    kwargs_94731 = {}
    str_94728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 12), 'str', ', ')
    # Obtaining the member 'join' of a type (line 122)
    join_94729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 12), str_94728, 'join')
    # Calling join(args, kwargs) (line 122)
    join_call_result_94732 = invoke(stypy.reporting.localization.Localization(__file__, 122, 12), join_94729, *[args_94730], **kwargs_94731)
    
    # Assigning a type to the variable 'sargs' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'sargs', join_call_result_94732)
    
    # Getting the type of 'f90mode' (line 123)
    f90mode_94733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 7), 'f90mode')
    # Testing the type of an if condition (line 123)
    if_condition_94734 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 4), f90mode_94733)
    # Assigning a type to the variable 'if_condition_94734' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'if_condition_94734', if_condition_94734)
    # SSA begins for if statement (line 123)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to add(...): (line 124)
    # Processing the call arguments (line 124)
    str_94736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 12), 'str', 'subroutine f2pywrap_%s_%s (%s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 125)
    tuple_94737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 125)
    # Adding element type (line 125)
    
    # Obtaining the type of the subscript
    str_94738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 18), 'str', 'modulename')
    # Getting the type of 'rout' (line 125)
    rout_94739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 13), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 125)
    getitem___94740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 13), rout_94739, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 125)
    subscript_call_result_94741 = invoke(stypy.reporting.localization.Localization(__file__, 125, 13), getitem___94740, str_94738)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 13), tuple_94737, subscript_call_result_94741)
    # Adding element type (line 125)
    # Getting the type of 'name' (line 125)
    name_94742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 33), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 13), tuple_94737, name_94742)
    # Adding element type (line 125)
    # Getting the type of 'sargs' (line 125)
    sargs_94743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 39), 'sargs', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 13), tuple_94737, sargs_94743)
    
    # Applying the binary operator '%' (line 124)
    result_mod_94744 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 12), '%', str_94736, tuple_94737)
    
    # Processing the call keyword arguments (line 124)
    kwargs_94745 = {}
    # Getting the type of 'add' (line 124)
    add_94735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'add', False)
    # Calling add(args, kwargs) (line 124)
    add_call_result_94746 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), add_94735, *[result_mod_94744], **kwargs_94745)
    
    
    
    # Getting the type of 'signature' (line 126)
    signature_94747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 15), 'signature')
    # Applying the 'not' unary operator (line 126)
    result_not__94748 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 11), 'not', signature_94747)
    
    # Testing the type of an if condition (line 126)
    if_condition_94749 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 8), result_not__94748)
    # Assigning a type to the variable 'if_condition_94749' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'if_condition_94749', if_condition_94749)
    # SSA begins for if statement (line 126)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to add(...): (line 127)
    # Processing the call arguments (line 127)
    str_94751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 16), 'str', 'use %s, only : %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 127)
    tuple_94752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 127)
    # Adding element type (line 127)
    
    # Obtaining the type of the subscript
    str_94753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 44), 'str', 'modulename')
    # Getting the type of 'rout' (line 127)
    rout_94754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 39), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___94755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 39), rout_94754, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_94756 = invoke(stypy.reporting.localization.Localization(__file__, 127, 39), getitem___94755, str_94753)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 39), tuple_94752, subscript_call_result_94756)
    # Adding element type (line 127)
    # Getting the type of 'fortranname' (line 127)
    fortranname_94757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 59), 'fortranname', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 39), tuple_94752, fortranname_94757)
    
    # Applying the binary operator '%' (line 127)
    result_mod_94758 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 16), '%', str_94751, tuple_94752)
    
    # Processing the call keyword arguments (line 127)
    kwargs_94759 = {}
    # Getting the type of 'add' (line 127)
    add_94750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'add', False)
    # Calling add(args, kwargs) (line 127)
    add_call_result_94760 = invoke(stypy.reporting.localization.Localization(__file__, 127, 12), add_94750, *[result_mod_94758], **kwargs_94759)
    
    # SSA join for if statement (line 126)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 123)
    module_type_store.open_ssa_branch('else')
    
    # Call to add(...): (line 129)
    # Processing the call arguments (line 129)
    str_94762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 12), 'str', 'subroutine f2pywrap%s (%s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 129)
    tuple_94763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 129)
    # Adding element type (line 129)
    # Getting the type of 'name' (line 129)
    name_94764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 44), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 44), tuple_94763, name_94764)
    # Adding element type (line 129)
    # Getting the type of 'sargs' (line 129)
    sargs_94765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 50), 'sargs', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 44), tuple_94763, sargs_94765)
    
    # Applying the binary operator '%' (line 129)
    result_mod_94766 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 12), '%', str_94762, tuple_94763)
    
    # Processing the call keyword arguments (line 129)
    kwargs_94767 = {}
    # Getting the type of 'add' (line 129)
    add_94761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'add', False)
    # Calling add(args, kwargs) (line 129)
    add_call_result_94768 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), add_94761, *[result_mod_94766], **kwargs_94767)
    
    
    
    # Getting the type of 'need_interface' (line 130)
    need_interface_94769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 15), 'need_interface')
    # Applying the 'not' unary operator (line 130)
    result_not__94770 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 11), 'not', need_interface_94769)
    
    # Testing the type of an if condition (line 130)
    if_condition_94771 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 8), result_not__94770)
    # Assigning a type to the variable 'if_condition_94771' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'if_condition_94771', if_condition_94771)
    # SSA begins for if statement (line 130)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to add(...): (line 131)
    # Processing the call arguments (line 131)
    str_94773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 16), 'str', 'external %s')
    # Getting the type of 'fortranname' (line 131)
    fortranname_94774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 33), 'fortranname', False)
    # Applying the binary operator '%' (line 131)
    result_mod_94775 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 16), '%', str_94773, fortranname_94774)
    
    # Processing the call keyword arguments (line 131)
    kwargs_94776 = {}
    # Getting the type of 'add' (line 131)
    add_94772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'add', False)
    # Calling add(args, kwargs) (line 131)
    add_call_result_94777 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), add_94772, *[result_mod_94775], **kwargs_94776)
    
    
    # Assigning a BinOp to a Name (line 132):
    # Getting the type of 'l' (line 132)
    l_94778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'l')
    str_94779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 20), 'str', ', ')
    # Applying the binary operator '+' (line 132)
    result_add_94780 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 16), '+', l_94778, str_94779)
    
    # Getting the type of 'fortranname' (line 132)
    fortranname_94781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 27), 'fortranname')
    # Applying the binary operator '+' (line 132)
    result_add_94782 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 25), '+', result_add_94780, fortranname_94781)
    
    # Assigning a type to the variable 'l' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'l', result_add_94782)
    # SSA join for if statement (line 130)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 123)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'need_interface' (line 133)
    need_interface_94783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 7), 'need_interface')
    # Testing the type of an if condition (line 133)
    if_condition_94784 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 4), need_interface_94783)
    # Assigning a type to the variable 'if_condition_94784' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'if_condition_94784', if_condition_94784)
    # SSA begins for if statement (line 133)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to split(...): (line 134)
    # Processing the call arguments (line 134)
    str_94790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 50), 'str', '\n')
    # Processing the call keyword arguments (line 134)
    kwargs_94791 = {}
    
    # Obtaining the type of the subscript
    str_94785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 25), 'str', 'saved_interface')
    # Getting the type of 'rout' (line 134)
    rout_94786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 20), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 134)
    getitem___94787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 20), rout_94786, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 134)
    subscript_call_result_94788 = invoke(stypy.reporting.localization.Localization(__file__, 134, 20), getitem___94787, str_94785)
    
    # Obtaining the member 'split' of a type (line 134)
    split_94789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 20), subscript_call_result_94788, 'split')
    # Calling split(args, kwargs) (line 134)
    split_call_result_94792 = invoke(stypy.reporting.localization.Localization(__file__, 134, 20), split_94789, *[str_94790], **kwargs_94791)
    
    # Testing the type of a for loop iterable (line 134)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 134, 8), split_call_result_94792)
    # Getting the type of the for loop variable (line 134)
    for_loop_var_94793 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 134, 8), split_call_result_94792)
    # Assigning a type to the variable 'line' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'line', for_loop_var_94793)
    # SSA begins for a for statement (line 134)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to startswith(...): (line 135)
    # Processing the call arguments (line 135)
    str_94799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 40), 'str', 'use ')
    # Processing the call keyword arguments (line 135)
    kwargs_94800 = {}
    
    # Call to lstrip(...): (line 135)
    # Processing the call keyword arguments (line 135)
    kwargs_94796 = {}
    # Getting the type of 'line' (line 135)
    line_94794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), 'line', False)
    # Obtaining the member 'lstrip' of a type (line 135)
    lstrip_94795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 15), line_94794, 'lstrip')
    # Calling lstrip(args, kwargs) (line 135)
    lstrip_call_result_94797 = invoke(stypy.reporting.localization.Localization(__file__, 135, 15), lstrip_94795, *[], **kwargs_94796)
    
    # Obtaining the member 'startswith' of a type (line 135)
    startswith_94798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 15), lstrip_call_result_94797, 'startswith')
    # Calling startswith(args, kwargs) (line 135)
    startswith_call_result_94801 = invoke(stypy.reporting.localization.Localization(__file__, 135, 15), startswith_94798, *[str_94799], **kwargs_94800)
    
    # Testing the type of an if condition (line 135)
    if_condition_94802 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 12), startswith_call_result_94801)
    # Assigning a type to the variable 'if_condition_94802' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'if_condition_94802', if_condition_94802)
    # SSA begins for if statement (line 135)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to add(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'line' (line 136)
    line_94804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 20), 'line', False)
    # Processing the call keyword arguments (line 136)
    kwargs_94805 = {}
    # Getting the type of 'add' (line 136)
    add_94803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'add', False)
    # Calling add(args, kwargs) (line 136)
    add_call_result_94806 = invoke(stypy.reporting.localization.Localization(__file__, 136, 16), add_94803, *[line_94804], **kwargs_94805)
    
    # SSA join for if statement (line 135)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 133)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 138):
    
    # Obtaining the type of the subscript
    int_94807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 16), 'int')
    slice_94808 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 138, 11), int_94807, None, None)
    # Getting the type of 'args' (line 138)
    args_94809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 11), 'args')
    # Obtaining the member '__getitem__' of a type (line 138)
    getitem___94810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 11), args_94809, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 138)
    subscript_call_result_94811 = invoke(stypy.reporting.localization.Localization(__file__, 138, 11), getitem___94810, slice_94808)
    
    # Assigning a type to the variable 'args' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'args', subscript_call_result_94811)
    
    # Assigning a List to a Name (line 139):
    
    # Obtaining an instance of the builtin type 'list' (line 139)
    list_94812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 139)
    
    # Assigning a type to the variable 'dumped_args' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'dumped_args', list_94812)
    
    # Getting the type of 'args' (line 140)
    args_94813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 13), 'args')
    # Testing the type of a for loop iterable (line 140)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 140, 4), args_94813)
    # Getting the type of the for loop variable (line 140)
    for_loop_var_94814 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 140, 4), args_94813)
    # Assigning a type to the variable 'a' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'a', for_loop_var_94814)
    # SSA begins for a for statement (line 140)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to isexternal(...): (line 141)
    # Processing the call arguments (line 141)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 141)
    a_94816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 27), 'a', False)
    # Getting the type of 'vars' (line 141)
    vars_94817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 22), 'vars', False)
    # Obtaining the member '__getitem__' of a type (line 141)
    getitem___94818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 22), vars_94817, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 141)
    subscript_call_result_94819 = invoke(stypy.reporting.localization.Localization(__file__, 141, 22), getitem___94818, a_94816)
    
    # Processing the call keyword arguments (line 141)
    kwargs_94820 = {}
    # Getting the type of 'isexternal' (line 141)
    isexternal_94815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 11), 'isexternal', False)
    # Calling isexternal(args, kwargs) (line 141)
    isexternal_call_result_94821 = invoke(stypy.reporting.localization.Localization(__file__, 141, 11), isexternal_94815, *[subscript_call_result_94819], **kwargs_94820)
    
    # Testing the type of an if condition (line 141)
    if_condition_94822 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 8), isexternal_call_result_94821)
    # Assigning a type to the variable 'if_condition_94822' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'if_condition_94822', if_condition_94822)
    # SSA begins for if statement (line 141)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to add(...): (line 142)
    # Processing the call arguments (line 142)
    str_94824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 16), 'str', 'external %s')
    # Getting the type of 'a' (line 142)
    a_94825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 33), 'a', False)
    # Applying the binary operator '%' (line 142)
    result_mod_94826 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 16), '%', str_94824, a_94825)
    
    # Processing the call keyword arguments (line 142)
    kwargs_94827 = {}
    # Getting the type of 'add' (line 142)
    add_94823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'add', False)
    # Calling add(args, kwargs) (line 142)
    add_call_result_94828 = invoke(stypy.reporting.localization.Localization(__file__, 142, 12), add_94823, *[result_mod_94826], **kwargs_94827)
    
    
    # Call to append(...): (line 143)
    # Processing the call arguments (line 143)
    # Getting the type of 'a' (line 143)
    a_94831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 31), 'a', False)
    # Processing the call keyword arguments (line 143)
    kwargs_94832 = {}
    # Getting the type of 'dumped_args' (line 143)
    dumped_args_94829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'dumped_args', False)
    # Obtaining the member 'append' of a type (line 143)
    append_94830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 12), dumped_args_94829, 'append')
    # Calling append(args, kwargs) (line 143)
    append_call_result_94833 = invoke(stypy.reporting.localization.Localization(__file__, 143, 12), append_94830, *[a_94831], **kwargs_94832)
    
    # SSA join for if statement (line 141)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'args' (line 144)
    args_94834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 13), 'args')
    # Testing the type of a for loop iterable (line 144)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 144, 4), args_94834)
    # Getting the type of the for loop variable (line 144)
    for_loop_var_94835 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 144, 4), args_94834)
    # Assigning a type to the variable 'a' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'a', for_loop_var_94835)
    # SSA begins for a for statement (line 144)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'a' (line 145)
    a_94836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 11), 'a')
    # Getting the type of 'dumped_args' (line 145)
    dumped_args_94837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'dumped_args')
    # Applying the binary operator 'in' (line 145)
    result_contains_94838 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 11), 'in', a_94836, dumped_args_94837)
    
    # Testing the type of an if condition (line 145)
    if_condition_94839 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 8), result_contains_94838)
    # Assigning a type to the variable 'if_condition_94839' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'if_condition_94839', if_condition_94839)
    # SSA begins for if statement (line 145)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 145)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isscalar(...): (line 147)
    # Processing the call arguments (line 147)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 147)
    a_94841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 25), 'a', False)
    # Getting the type of 'vars' (line 147)
    vars_94842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 20), 'vars', False)
    # Obtaining the member '__getitem__' of a type (line 147)
    getitem___94843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 20), vars_94842, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 147)
    subscript_call_result_94844 = invoke(stypy.reporting.localization.Localization(__file__, 147, 20), getitem___94843, a_94841)
    
    # Processing the call keyword arguments (line 147)
    kwargs_94845 = {}
    # Getting the type of 'isscalar' (line 147)
    isscalar_94840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 11), 'isscalar', False)
    # Calling isscalar(args, kwargs) (line 147)
    isscalar_call_result_94846 = invoke(stypy.reporting.localization.Localization(__file__, 147, 11), isscalar_94840, *[subscript_call_result_94844], **kwargs_94845)
    
    # Testing the type of an if condition (line 147)
    if_condition_94847 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 8), isscalar_call_result_94846)
    # Assigning a type to the variable 'if_condition_94847' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'if_condition_94847', if_condition_94847)
    # SSA begins for if statement (line 147)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to add(...): (line 148)
    # Processing the call arguments (line 148)
    
    # Call to var2fixfortran(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'vars' (line 148)
    vars_94850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 31), 'vars', False)
    # Getting the type of 'a' (line 148)
    a_94851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 37), 'a', False)
    # Processing the call keyword arguments (line 148)
    # Getting the type of 'f90mode' (line 148)
    f90mode_94852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 48), 'f90mode', False)
    keyword_94853 = f90mode_94852
    kwargs_94854 = {'f90mode': keyword_94853}
    # Getting the type of 'var2fixfortran' (line 148)
    var2fixfortran_94849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 16), 'var2fixfortran', False)
    # Calling var2fixfortran(args, kwargs) (line 148)
    var2fixfortran_call_result_94855 = invoke(stypy.reporting.localization.Localization(__file__, 148, 16), var2fixfortran_94849, *[vars_94850, a_94851], **kwargs_94854)
    
    # Processing the call keyword arguments (line 148)
    kwargs_94856 = {}
    # Getting the type of 'add' (line 148)
    add_94848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'add', False)
    # Calling add(args, kwargs) (line 148)
    add_call_result_94857 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), add_94848, *[var2fixfortran_call_result_94855], **kwargs_94856)
    
    
    # Call to append(...): (line 149)
    # Processing the call arguments (line 149)
    # Getting the type of 'a' (line 149)
    a_94860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 31), 'a', False)
    # Processing the call keyword arguments (line 149)
    kwargs_94861 = {}
    # Getting the type of 'dumped_args' (line 149)
    dumped_args_94858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'dumped_args', False)
    # Obtaining the member 'append' of a type (line 149)
    append_94859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 12), dumped_args_94858, 'append')
    # Calling append(args, kwargs) (line 149)
    append_call_result_94862 = invoke(stypy.reporting.localization.Localization(__file__, 149, 12), append_94859, *[a_94860], **kwargs_94861)
    
    # SSA join for if statement (line 147)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'args' (line 150)
    args_94863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 13), 'args')
    # Testing the type of a for loop iterable (line 150)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 150, 4), args_94863)
    # Getting the type of the for loop variable (line 150)
    for_loop_var_94864 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 150, 4), args_94863)
    # Assigning a type to the variable 'a' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'a', for_loop_var_94864)
    # SSA begins for a for statement (line 150)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'a' (line 151)
    a_94865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 11), 'a')
    # Getting the type of 'dumped_args' (line 151)
    dumped_args_94866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'dumped_args')
    # Applying the binary operator 'in' (line 151)
    result_contains_94867 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 11), 'in', a_94865, dumped_args_94866)
    
    # Testing the type of an if condition (line 151)
    if_condition_94868 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 8), result_contains_94867)
    # Assigning a type to the variable 'if_condition_94868' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'if_condition_94868', if_condition_94868)
    # SSA begins for if statement (line 151)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 151)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isintent_in(...): (line 153)
    # Processing the call arguments (line 153)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 153)
    a_94870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 28), 'a', False)
    # Getting the type of 'vars' (line 153)
    vars_94871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 23), 'vars', False)
    # Obtaining the member '__getitem__' of a type (line 153)
    getitem___94872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 23), vars_94871, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 153)
    subscript_call_result_94873 = invoke(stypy.reporting.localization.Localization(__file__, 153, 23), getitem___94872, a_94870)
    
    # Processing the call keyword arguments (line 153)
    kwargs_94874 = {}
    # Getting the type of 'isintent_in' (line 153)
    isintent_in_94869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 11), 'isintent_in', False)
    # Calling isintent_in(args, kwargs) (line 153)
    isintent_in_call_result_94875 = invoke(stypy.reporting.localization.Localization(__file__, 153, 11), isintent_in_94869, *[subscript_call_result_94873], **kwargs_94874)
    
    # Testing the type of an if condition (line 153)
    if_condition_94876 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 8), isintent_in_call_result_94875)
    # Assigning a type to the variable 'if_condition_94876' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'if_condition_94876', if_condition_94876)
    # SSA begins for if statement (line 153)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to add(...): (line 154)
    # Processing the call arguments (line 154)
    
    # Call to var2fixfortran(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'vars' (line 154)
    vars_94879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 31), 'vars', False)
    # Getting the type of 'a' (line 154)
    a_94880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 37), 'a', False)
    # Processing the call keyword arguments (line 154)
    # Getting the type of 'f90mode' (line 154)
    f90mode_94881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 48), 'f90mode', False)
    keyword_94882 = f90mode_94881
    kwargs_94883 = {'f90mode': keyword_94882}
    # Getting the type of 'var2fixfortran' (line 154)
    var2fixfortran_94878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'var2fixfortran', False)
    # Calling var2fixfortran(args, kwargs) (line 154)
    var2fixfortran_call_result_94884 = invoke(stypy.reporting.localization.Localization(__file__, 154, 16), var2fixfortran_94878, *[vars_94879, a_94880], **kwargs_94883)
    
    # Processing the call keyword arguments (line 154)
    kwargs_94885 = {}
    # Getting the type of 'add' (line 154)
    add_94877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'add', False)
    # Calling add(args, kwargs) (line 154)
    add_call_result_94886 = invoke(stypy.reporting.localization.Localization(__file__, 154, 12), add_94877, *[var2fixfortran_call_result_94884], **kwargs_94885)
    
    
    # Call to append(...): (line 155)
    # Processing the call arguments (line 155)
    # Getting the type of 'a' (line 155)
    a_94889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 31), 'a', False)
    # Processing the call keyword arguments (line 155)
    kwargs_94890 = {}
    # Getting the type of 'dumped_args' (line 155)
    dumped_args_94887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'dumped_args', False)
    # Obtaining the member 'append' of a type (line 155)
    append_94888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 12), dumped_args_94887, 'append')
    # Calling append(args, kwargs) (line 155)
    append_call_result_94891 = invoke(stypy.reporting.localization.Localization(__file__, 155, 12), append_94888, *[a_94889], **kwargs_94890)
    
    # SSA join for if statement (line 153)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'args' (line 156)
    args_94892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 13), 'args')
    # Testing the type of a for loop iterable (line 156)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 156, 4), args_94892)
    # Getting the type of the for loop variable (line 156)
    for_loop_var_94893 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 156, 4), args_94892)
    # Assigning a type to the variable 'a' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'a', for_loop_var_94893)
    # SSA begins for a for statement (line 156)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'a' (line 157)
    a_94894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 11), 'a')
    # Getting the type of 'dumped_args' (line 157)
    dumped_args_94895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'dumped_args')
    # Applying the binary operator 'in' (line 157)
    result_contains_94896 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 11), 'in', a_94894, dumped_args_94895)
    
    # Testing the type of an if condition (line 157)
    if_condition_94897 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 8), result_contains_94896)
    # Assigning a type to the variable 'if_condition_94897' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'if_condition_94897', if_condition_94897)
    # SSA begins for if statement (line 157)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 157)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to add(...): (line 159)
    # Processing the call arguments (line 159)
    
    # Call to var2fixfortran(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'vars' (line 159)
    vars_94900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 27), 'vars', False)
    # Getting the type of 'a' (line 159)
    a_94901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 33), 'a', False)
    # Processing the call keyword arguments (line 159)
    # Getting the type of 'f90mode' (line 159)
    f90mode_94902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 44), 'f90mode', False)
    keyword_94903 = f90mode_94902
    kwargs_94904 = {'f90mode': keyword_94903}
    # Getting the type of 'var2fixfortran' (line 159)
    var2fixfortran_94899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'var2fixfortran', False)
    # Calling var2fixfortran(args, kwargs) (line 159)
    var2fixfortran_call_result_94905 = invoke(stypy.reporting.localization.Localization(__file__, 159, 12), var2fixfortran_94899, *[vars_94900, a_94901], **kwargs_94904)
    
    # Processing the call keyword arguments (line 159)
    kwargs_94906 = {}
    # Getting the type of 'add' (line 159)
    add_94898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'add', False)
    # Calling add(args, kwargs) (line 159)
    add_call_result_94907 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), add_94898, *[var2fixfortran_call_result_94905], **kwargs_94906)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to add(...): (line 161)
    # Processing the call arguments (line 161)
    # Getting the type of 'l' (line 161)
    l_94909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'l', False)
    # Processing the call keyword arguments (line 161)
    kwargs_94910 = {}
    # Getting the type of 'add' (line 161)
    add_94908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'add', False)
    # Calling add(args, kwargs) (line 161)
    add_call_result_94911 = invoke(stypy.reporting.localization.Localization(__file__, 161, 4), add_94908, *[l_94909], **kwargs_94910)
    
    
    # Getting the type of 'need_interface' (line 163)
    need_interface_94912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 7), 'need_interface')
    # Testing the type of an if condition (line 163)
    if_condition_94913 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 4), need_interface_94912)
    # Assigning a type to the variable 'if_condition_94913' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'if_condition_94913', if_condition_94913)
    # SSA begins for if statement (line 163)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'f90mode' (line 164)
    f90mode_94914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 11), 'f90mode')
    # Testing the type of an if condition (line 164)
    if_condition_94915 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 8), f90mode_94914)
    # Assigning a type to the variable 'if_condition_94915' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'if_condition_94915', if_condition_94915)
    # SSA begins for if statement (line 164)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    pass
    # SSA branch for the else part of an if statement (line 164)
    module_type_store.open_ssa_branch('else')
    
    # Call to add(...): (line 168)
    # Processing the call arguments (line 168)
    str_94917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 16), 'str', 'interface')
    # Processing the call keyword arguments (line 168)
    kwargs_94918 = {}
    # Getting the type of 'add' (line 168)
    add_94916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'add', False)
    # Calling add(args, kwargs) (line 168)
    add_call_result_94919 = invoke(stypy.reporting.localization.Localization(__file__, 168, 12), add_94916, *[str_94917], **kwargs_94918)
    
    
    # Call to add(...): (line 169)
    # Processing the call arguments (line 169)
    
    # Call to lstrip(...): (line 169)
    # Processing the call keyword arguments (line 169)
    kwargs_94926 = {}
    
    # Obtaining the type of the subscript
    str_94921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 21), 'str', 'saved_interface')
    # Getting the type of 'rout' (line 169)
    rout_94922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___94923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 16), rout_94922, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_94924 = invoke(stypy.reporting.localization.Localization(__file__, 169, 16), getitem___94923, str_94921)
    
    # Obtaining the member 'lstrip' of a type (line 169)
    lstrip_94925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 16), subscript_call_result_94924, 'lstrip')
    # Calling lstrip(args, kwargs) (line 169)
    lstrip_call_result_94927 = invoke(stypy.reporting.localization.Localization(__file__, 169, 16), lstrip_94925, *[], **kwargs_94926)
    
    # Processing the call keyword arguments (line 169)
    kwargs_94928 = {}
    # Getting the type of 'add' (line 169)
    add_94920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'add', False)
    # Calling add(args, kwargs) (line 169)
    add_call_result_94929 = invoke(stypy.reporting.localization.Localization(__file__, 169, 12), add_94920, *[lstrip_call_result_94927], **kwargs_94928)
    
    
    # Call to add(...): (line 170)
    # Processing the call arguments (line 170)
    str_94931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 16), 'str', 'end interface')
    # Processing the call keyword arguments (line 170)
    kwargs_94932 = {}
    # Getting the type of 'add' (line 170)
    add_94930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'add', False)
    # Calling add(args, kwargs) (line 170)
    add_call_result_94933 = invoke(stypy.reporting.localization.Localization(__file__, 170, 12), add_94930, *[str_94931], **kwargs_94932)
    
    # SSA join for if statement (line 164)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 163)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 172):
    
    # Call to join(...): (line 172)
    # Processing the call arguments (line 172)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'args' (line 172)
    args_94940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 34), 'args', False)
    comprehension_94941 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 23), args_94940)
    # Assigning a type to the variable 'a' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 23), 'a', comprehension_94941)
    
    # Getting the type of 'a' (line 172)
    a_94937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 42), 'a', False)
    # Getting the type of 'extra_args' (line 172)
    extra_args_94938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 51), 'extra_args', False)
    # Applying the binary operator 'notin' (line 172)
    result_contains_94939 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 42), 'notin', a_94937, extra_args_94938)
    
    # Getting the type of 'a' (line 172)
    a_94936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 23), 'a', False)
    list_94942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 23), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 23), list_94942, a_94936)
    # Processing the call keyword arguments (line 172)
    kwargs_94943 = {}
    str_94934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 12), 'str', ', ')
    # Obtaining the member 'join' of a type (line 172)
    join_94935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 12), str_94934, 'join')
    # Calling join(args, kwargs) (line 172)
    join_call_result_94944 = invoke(stypy.reporting.localization.Localization(__file__, 172, 12), join_94935, *[list_94942], **kwargs_94943)
    
    # Assigning a type to the variable 'sargs' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'sargs', join_call_result_94944)
    
    
    # Getting the type of 'signature' (line 174)
    signature_94945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 11), 'signature')
    # Applying the 'not' unary operator (line 174)
    result_not__94946 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 7), 'not', signature_94945)
    
    # Testing the type of an if condition (line 174)
    if_condition_94947 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 4), result_not__94946)
    # Assigning a type to the variable 'if_condition_94947' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'if_condition_94947', if_condition_94947)
    # SSA begins for if statement (line 174)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to islogicalfunction(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 'rout' (line 175)
    rout_94949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 29), 'rout', False)
    # Processing the call keyword arguments (line 175)
    kwargs_94950 = {}
    # Getting the type of 'islogicalfunction' (line 175)
    islogicalfunction_94948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 11), 'islogicalfunction', False)
    # Calling islogicalfunction(args, kwargs) (line 175)
    islogicalfunction_call_result_94951 = invoke(stypy.reporting.localization.Localization(__file__, 175, 11), islogicalfunction_94948, *[rout_94949], **kwargs_94950)
    
    # Testing the type of an if condition (line 175)
    if_condition_94952 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 175, 8), islogicalfunction_call_result_94951)
    # Assigning a type to the variable 'if_condition_94952' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'if_condition_94952', if_condition_94952)
    # SSA begins for if statement (line 175)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to add(...): (line 176)
    # Processing the call arguments (line 176)
    str_94954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 16), 'str', '%s = .not.(.not.%s(%s))')
    
    # Obtaining an instance of the builtin type 'tuple' (line 176)
    tuple_94955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 45), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 176)
    # Adding element type (line 176)
    # Getting the type of 'newname' (line 176)
    newname_94956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 45), 'newname', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 45), tuple_94955, newname_94956)
    # Adding element type (line 176)
    # Getting the type of 'fortranname' (line 176)
    fortranname_94957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 54), 'fortranname', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 45), tuple_94955, fortranname_94957)
    # Adding element type (line 176)
    # Getting the type of 'sargs' (line 176)
    sargs_94958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 67), 'sargs', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 45), tuple_94955, sargs_94958)
    
    # Applying the binary operator '%' (line 176)
    result_mod_94959 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 16), '%', str_94954, tuple_94955)
    
    # Processing the call keyword arguments (line 176)
    kwargs_94960 = {}
    # Getting the type of 'add' (line 176)
    add_94953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'add', False)
    # Calling add(args, kwargs) (line 176)
    add_call_result_94961 = invoke(stypy.reporting.localization.Localization(__file__, 176, 12), add_94953, *[result_mod_94959], **kwargs_94960)
    
    # SSA branch for the else part of an if statement (line 175)
    module_type_store.open_ssa_branch('else')
    
    # Call to add(...): (line 178)
    # Processing the call arguments (line 178)
    str_94963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 16), 'str', '%s = %s(%s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 178)
    tuple_94964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 178)
    # Adding element type (line 178)
    # Getting the type of 'newname' (line 178)
    newname_94965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 33), 'newname', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 33), tuple_94964, newname_94965)
    # Adding element type (line 178)
    # Getting the type of 'fortranname' (line 178)
    fortranname_94966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 42), 'fortranname', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 33), tuple_94964, fortranname_94966)
    # Adding element type (line 178)
    # Getting the type of 'sargs' (line 178)
    sargs_94967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 55), 'sargs', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 33), tuple_94964, sargs_94967)
    
    # Applying the binary operator '%' (line 178)
    result_mod_94968 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 16), '%', str_94963, tuple_94964)
    
    # Processing the call keyword arguments (line 178)
    kwargs_94969 = {}
    # Getting the type of 'add' (line 178)
    add_94962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'add', False)
    # Calling add(args, kwargs) (line 178)
    add_call_result_94970 = invoke(stypy.reporting.localization.Localization(__file__, 178, 12), add_94962, *[result_mod_94968], **kwargs_94969)
    
    # SSA join for if statement (line 175)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 174)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'f90mode' (line 179)
    f90mode_94971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 7), 'f90mode')
    # Testing the type of an if condition (line 179)
    if_condition_94972 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 179, 4), f90mode_94971)
    # Assigning a type to the variable 'if_condition_94972' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'if_condition_94972', if_condition_94972)
    # SSA begins for if statement (line 179)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to add(...): (line 180)
    # Processing the call arguments (line 180)
    str_94974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 12), 'str', 'end subroutine f2pywrap_%s_%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 180)
    tuple_94975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 180)
    # Adding element type (line 180)
    
    # Obtaining the type of the subscript
    str_94976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 52), 'str', 'modulename')
    # Getting the type of 'rout' (line 180)
    rout_94977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 47), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 180)
    getitem___94978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 47), rout_94977, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 180)
    subscript_call_result_94979 = invoke(stypy.reporting.localization.Localization(__file__, 180, 47), getitem___94978, str_94976)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 47), tuple_94975, subscript_call_result_94979)
    # Adding element type (line 180)
    # Getting the type of 'name' (line 180)
    name_94980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 67), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 47), tuple_94975, name_94980)
    
    # Applying the binary operator '%' (line 180)
    result_mod_94981 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 12), '%', str_94974, tuple_94975)
    
    # Processing the call keyword arguments (line 180)
    kwargs_94982 = {}
    # Getting the type of 'add' (line 180)
    add_94973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'add', False)
    # Calling add(args, kwargs) (line 180)
    add_call_result_94983 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), add_94973, *[result_mod_94981], **kwargs_94982)
    
    # SSA branch for the else part of an if statement (line 179)
    module_type_store.open_ssa_branch('else')
    
    # Call to add(...): (line 182)
    # Processing the call arguments (line 182)
    str_94985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 12), 'str', 'end')
    # Processing the call keyword arguments (line 182)
    kwargs_94986 = {}
    # Getting the type of 'add' (line 182)
    add_94984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'add', False)
    # Calling add(args, kwargs) (line 182)
    add_call_result_94987 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), add_94984, *[str_94985], **kwargs_94986)
    
    # SSA join for if statement (line 179)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining the type of the subscript
    int_94988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 15), 'int')
    # Getting the type of 'ret' (line 183)
    ret_94989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 11), 'ret')
    # Obtaining the member '__getitem__' of a type (line 183)
    getitem___94990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 11), ret_94989, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 183)
    subscript_call_result_94991 = invoke(stypy.reporting.localization.Localization(__file__, 183, 11), getitem___94990, int_94988)
    
    # Assigning a type to the variable 'stypy_return_type' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'stypy_return_type', subscript_call_result_94991)
    
    # ################# End of 'createfuncwrapper(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'createfuncwrapper' in the type store
    # Getting the type of 'stypy_return_type' (line 80)
    stypy_return_type_94992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_94992)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'createfuncwrapper'
    return stypy_return_type_94992

# Assigning a type to the variable 'createfuncwrapper' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'createfuncwrapper', createfuncwrapper)

@norecursion
def createsubrwrapper(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_94993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 38), 'int')
    defaults = [int_94993]
    # Create a new context for function 'createsubrwrapper'
    module_type_store = module_type_store.open_function_context('createsubrwrapper', 186, 0, False)
    
    # Passed parameters checking function
    createsubrwrapper.stypy_localization = localization
    createsubrwrapper.stypy_type_of_self = None
    createsubrwrapper.stypy_type_store = module_type_store
    createsubrwrapper.stypy_function_name = 'createsubrwrapper'
    createsubrwrapper.stypy_param_names_list = ['rout', 'signature']
    createsubrwrapper.stypy_varargs_param_name = None
    createsubrwrapper.stypy_kwargs_param_name = None
    createsubrwrapper.stypy_call_defaults = defaults
    createsubrwrapper.stypy_call_varargs = varargs
    createsubrwrapper.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'createsubrwrapper', ['rout', 'signature'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'createsubrwrapper', localization, ['rout', 'signature'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'createsubrwrapper(...)' code ##################

    # Evaluating assert statement condition
    
    # Call to issubroutine(...): (line 187)
    # Processing the call arguments (line 187)
    # Getting the type of 'rout' (line 187)
    rout_94995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 24), 'rout', False)
    # Processing the call keyword arguments (line 187)
    kwargs_94996 = {}
    # Getting the type of 'issubroutine' (line 187)
    issubroutine_94994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 11), 'issubroutine', False)
    # Calling issubroutine(args, kwargs) (line 187)
    issubroutine_call_result_94997 = invoke(stypy.reporting.localization.Localization(__file__, 187, 11), issubroutine_94994, *[rout_94995], **kwargs_94996)
    
    
    # Assigning a List to a Name (line 189):
    
    # Obtaining an instance of the builtin type 'list' (line 189)
    list_94998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 189)
    
    # Assigning a type to the variable 'extra_args' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'extra_args', list_94998)
    
    # Assigning a Subscript to a Name (line 190):
    
    # Obtaining the type of the subscript
    str_94999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 16), 'str', 'vars')
    # Getting the type of 'rout' (line 190)
    rout_95000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 11), 'rout')
    # Obtaining the member '__getitem__' of a type (line 190)
    getitem___95001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 11), rout_95000, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 190)
    subscript_call_result_95002 = invoke(stypy.reporting.localization.Localization(__file__, 190, 11), getitem___95001, str_94999)
    
    # Assigning a type to the variable 'vars' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'vars', subscript_call_result_95002)
    
    
    # Obtaining the type of the subscript
    str_95003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 18), 'str', 'args')
    # Getting the type of 'rout' (line 191)
    rout_95004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 13), 'rout')
    # Obtaining the member '__getitem__' of a type (line 191)
    getitem___95005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 13), rout_95004, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 191)
    subscript_call_result_95006 = invoke(stypy.reporting.localization.Localization(__file__, 191, 13), getitem___95005, str_95003)
    
    # Testing the type of a for loop iterable (line 191)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 191, 4), subscript_call_result_95006)
    # Getting the type of the for loop variable (line 191)
    for_loop_var_95007 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 191, 4), subscript_call_result_95006)
    # Assigning a type to the variable 'a' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'a', for_loop_var_95007)
    # SSA begins for a for statement (line 191)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 192):
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 192)
    a_95008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 25), 'a')
    
    # Obtaining the type of the subscript
    str_95009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 17), 'str', 'vars')
    # Getting the type of 'rout' (line 192)
    rout_95010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'rout')
    # Obtaining the member '__getitem__' of a type (line 192)
    getitem___95011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 12), rout_95010, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 192)
    subscript_call_result_95012 = invoke(stypy.reporting.localization.Localization(__file__, 192, 12), getitem___95011, str_95009)
    
    # Obtaining the member '__getitem__' of a type (line 192)
    getitem___95013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 12), subscript_call_result_95012, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 192)
    subscript_call_result_95014 = invoke(stypy.reporting.localization.Localization(__file__, 192, 12), getitem___95013, a_95008)
    
    # Assigning a type to the variable 'v' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'v', subscript_call_result_95014)
    
    
    # Call to enumerate(...): (line 193)
    # Processing the call arguments (line 193)
    
    # Call to get(...): (line 193)
    # Processing the call arguments (line 193)
    str_95018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 36), 'str', 'dimension')
    
    # Obtaining an instance of the builtin type 'list' (line 193)
    list_95019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 49), 'list')
    # Adding type elements to the builtin type 'list' instance (line 193)
    
    # Processing the call keyword arguments (line 193)
    kwargs_95020 = {}
    # Getting the type of 'v' (line 193)
    v_95016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 30), 'v', False)
    # Obtaining the member 'get' of a type (line 193)
    get_95017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 30), v_95016, 'get')
    # Calling get(args, kwargs) (line 193)
    get_call_result_95021 = invoke(stypy.reporting.localization.Localization(__file__, 193, 30), get_95017, *[str_95018, list_95019], **kwargs_95020)
    
    # Processing the call keyword arguments (line 193)
    kwargs_95022 = {}
    # Getting the type of 'enumerate' (line 193)
    enumerate_95015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 20), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 193)
    enumerate_call_result_95023 = invoke(stypy.reporting.localization.Localization(__file__, 193, 20), enumerate_95015, *[get_call_result_95021], **kwargs_95022)
    
    # Testing the type of a for loop iterable (line 193)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 193, 8), enumerate_call_result_95023)
    # Getting the type of the for loop variable (line 193)
    for_loop_var_95024 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 193, 8), enumerate_call_result_95023)
    # Assigning a type to the variable 'i' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 8), for_loop_var_95024))
    # Assigning a type to the variable 'd' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 8), for_loop_var_95024))
    # SSA begins for a for statement (line 193)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'd' (line 194)
    d_95025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 15), 'd')
    str_95026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 20), 'str', ':')
    # Applying the binary operator '==' (line 194)
    result_eq_95027 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 15), '==', d_95025, str_95026)
    
    # Testing the type of an if condition (line 194)
    if_condition_95028 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 12), result_eq_95027)
    # Assigning a type to the variable 'if_condition_95028' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'if_condition_95028', if_condition_95028)
    # SSA begins for if statement (line 194)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 195):
    str_95029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 21), 'str', 'f2py_%s_d%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 195)
    tuple_95030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 195)
    # Adding element type (line 195)
    # Getting the type of 'a' (line 195)
    a_95031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 38), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 38), tuple_95030, a_95031)
    # Adding element type (line 195)
    # Getting the type of 'i' (line 195)
    i_95032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 41), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 38), tuple_95030, i_95032)
    
    # Applying the binary operator '%' (line 195)
    result_mod_95033 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 21), '%', str_95029, tuple_95030)
    
    # Assigning a type to the variable 'dn' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 16), 'dn', result_mod_95033)
    
    # Assigning a Call to a Name (line 196):
    
    # Call to dict(...): (line 196)
    # Processing the call keyword arguments (line 196)
    str_95035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 35), 'str', 'integer')
    keyword_95036 = str_95035
    
    # Obtaining an instance of the builtin type 'list' (line 196)
    list_95037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 53), 'list')
    # Adding type elements to the builtin type 'list' instance (line 196)
    # Adding element type (line 196)
    str_95038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 54), 'str', 'hide')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 53), list_95037, str_95038)
    
    keyword_95039 = list_95037
    kwargs_95040 = {'typespec': keyword_95036, 'intent': keyword_95039}
    # Getting the type of 'dict' (line 196)
    dict_95034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 21), 'dict', False)
    # Calling dict(args, kwargs) (line 196)
    dict_call_result_95041 = invoke(stypy.reporting.localization.Localization(__file__, 196, 21), dict_95034, *[], **kwargs_95040)
    
    # Assigning a type to the variable 'dv' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), 'dv', dict_call_result_95041)
    
    # Assigning a BinOp to a Subscript (line 197):
    str_95042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 26), 'str', 'shape(%s, %s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 197)
    tuple_95043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 45), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 197)
    # Adding element type (line 197)
    # Getting the type of 'a' (line 197)
    a_95044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 45), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 45), tuple_95043, a_95044)
    # Adding element type (line 197)
    # Getting the type of 'i' (line 197)
    i_95045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 48), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 45), tuple_95043, i_95045)
    
    # Applying the binary operator '%' (line 197)
    result_mod_95046 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 26), '%', str_95042, tuple_95043)
    
    # Getting the type of 'dv' (line 197)
    dv_95047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 16), 'dv')
    str_95048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 19), 'str', '=')
    # Storing an element on a container (line 197)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 16), dv_95047, (str_95048, result_mod_95046))
    
    # Call to append(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'dn' (line 198)
    dn_95051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 34), 'dn', False)
    # Processing the call keyword arguments (line 198)
    kwargs_95052 = {}
    # Getting the type of 'extra_args' (line 198)
    extra_args_95049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'extra_args', False)
    # Obtaining the member 'append' of a type (line 198)
    append_95050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 16), extra_args_95049, 'append')
    # Calling append(args, kwargs) (line 198)
    append_call_result_95053 = invoke(stypy.reporting.localization.Localization(__file__, 198, 16), append_95050, *[dn_95051], **kwargs_95052)
    
    
    # Assigning a Name to a Subscript (line 199):
    # Getting the type of 'dv' (line 199)
    dv_95054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 27), 'dv')
    # Getting the type of 'vars' (line 199)
    vars_95055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'vars')
    # Getting the type of 'dn' (line 199)
    dn_95056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 21), 'dn')
    # Storing an element on a container (line 199)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 16), vars_95055, (dn_95056, dv_95054))
    
    # Assigning a Name to a Subscript (line 200):
    # Getting the type of 'dn' (line 200)
    dn_95057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 36), 'dn')
    
    # Obtaining the type of the subscript
    str_95058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 18), 'str', 'dimension')
    # Getting the type of 'v' (line 200)
    v_95059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 16), 'v')
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___95060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 16), v_95059, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 200)
    subscript_call_result_95061 = invoke(stypy.reporting.localization.Localization(__file__, 200, 16), getitem___95060, str_95058)
    
    # Getting the type of 'i' (line 200)
    i_95062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 31), 'i')
    # Storing an element on a container (line 200)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 16), subscript_call_result_95061, (i_95062, dn_95057))
    # SSA join for if statement (line 194)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to extend(...): (line 201)
    # Processing the call arguments (line 201)
    # Getting the type of 'extra_args' (line 201)
    extra_args_95068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 24), 'extra_args', False)
    # Processing the call keyword arguments (line 201)
    kwargs_95069 = {}
    
    # Obtaining the type of the subscript
    str_95063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 9), 'str', 'args')
    # Getting the type of 'rout' (line 201)
    rout_95064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 201)
    getitem___95065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 4), rout_95064, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 201)
    subscript_call_result_95066 = invoke(stypy.reporting.localization.Localization(__file__, 201, 4), getitem___95065, str_95063)
    
    # Obtaining the member 'extend' of a type (line 201)
    extend_95067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 4), subscript_call_result_95066, 'extend')
    # Calling extend(args, kwargs) (line 201)
    extend_call_result_95070 = invoke(stypy.reporting.localization.Localization(__file__, 201, 4), extend_95067, *[extra_args_95068], **kwargs_95069)
    
    
    # Assigning a Call to a Name (line 202):
    
    # Call to bool(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'extra_args' (line 202)
    extra_args_95072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 26), 'extra_args', False)
    # Processing the call keyword arguments (line 202)
    kwargs_95073 = {}
    # Getting the type of 'bool' (line 202)
    bool_95071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 21), 'bool', False)
    # Calling bool(args, kwargs) (line 202)
    bool_call_result_95074 = invoke(stypy.reporting.localization.Localization(__file__, 202, 21), bool_95071, *[extra_args_95072], **kwargs_95073)
    
    # Assigning a type to the variable 'need_interface' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'need_interface', bool_call_result_95074)
    
    # Assigning a List to a Name (line 204):
    
    # Obtaining an instance of the builtin type 'list' (line 204)
    list_95075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 204)
    # Adding element type (line 204)
    str_95076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 11), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 10), list_95075, str_95076)
    
    # Assigning a type to the variable 'ret' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'ret', list_95075)

    @norecursion
    def add(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'ret' (line 206)
        ret_95077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 22), 'ret')
        defaults = [ret_95077]
        # Create a new context for function 'add'
        module_type_store = module_type_store.open_function_context('add', 206, 4, False)
        
        # Passed parameters checking function
        add.stypy_localization = localization
        add.stypy_type_of_self = None
        add.stypy_type_store = module_type_store
        add.stypy_function_name = 'add'
        add.stypy_param_names_list = ['line', 'ret']
        add.stypy_varargs_param_name = None
        add.stypy_kwargs_param_name = None
        add.stypy_call_defaults = defaults
        add.stypy_call_varargs = varargs
        add.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'add', ['line', 'ret'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add', localization, ['line', 'ret'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add(...)' code ##################

        
        # Assigning a BinOp to a Subscript (line 207):
        str_95078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 17), 'str', '%s\n      %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 207)
        tuple_95079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 207)
        # Adding element type (line 207)
        
        # Obtaining the type of the subscript
        int_95080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 39), 'int')
        # Getting the type of 'ret' (line 207)
        ret_95081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 35), 'ret')
        # Obtaining the member '__getitem__' of a type (line 207)
        getitem___95082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 35), ret_95081, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 207)
        subscript_call_result_95083 = invoke(stypy.reporting.localization.Localization(__file__, 207, 35), getitem___95082, int_95080)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 35), tuple_95079, subscript_call_result_95083)
        # Adding element type (line 207)
        # Getting the type of 'line' (line 207)
        line_95084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 43), 'line')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 35), tuple_95079, line_95084)
        
        # Applying the binary operator '%' (line 207)
        result_mod_95085 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 17), '%', str_95078, tuple_95079)
        
        # Getting the type of 'ret' (line 207)
        ret_95086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'ret')
        int_95087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 12), 'int')
        # Storing an element on a container (line 207)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 8), ret_95086, (int_95087, result_mod_95085))
        
        # ################# End of 'add(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add' in the type store
        # Getting the type of 'stypy_return_type' (line 206)
        stypy_return_type_95088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_95088)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add'
        return stypy_return_type_95088

    # Assigning a type to the variable 'add' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'add', add)
    
    # Assigning a Subscript to a Name (line 208):
    
    # Obtaining the type of the subscript
    str_95089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 16), 'str', 'name')
    # Getting the type of 'rout' (line 208)
    rout_95090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 11), 'rout')
    # Obtaining the member '__getitem__' of a type (line 208)
    getitem___95091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 11), rout_95090, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 208)
    subscript_call_result_95092 = invoke(stypy.reporting.localization.Localization(__file__, 208, 11), getitem___95091, str_95089)
    
    # Assigning a type to the variable 'name' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'name', subscript_call_result_95092)
    
    # Assigning a Call to a Name (line 209):
    
    # Call to getfortranname(...): (line 209)
    # Processing the call arguments (line 209)
    # Getting the type of 'rout' (line 209)
    rout_95094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 33), 'rout', False)
    # Processing the call keyword arguments (line 209)
    kwargs_95095 = {}
    # Getting the type of 'getfortranname' (line 209)
    getfortranname_95093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 18), 'getfortranname', False)
    # Calling getfortranname(args, kwargs) (line 209)
    getfortranname_call_result_95096 = invoke(stypy.reporting.localization.Localization(__file__, 209, 18), getfortranname_95093, *[rout_95094], **kwargs_95095)
    
    # Assigning a type to the variable 'fortranname' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'fortranname', getfortranname_call_result_95096)
    
    # Assigning a Call to a Name (line 210):
    
    # Call to ismoduleroutine(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'rout' (line 210)
    rout_95098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 30), 'rout', False)
    # Processing the call keyword arguments (line 210)
    kwargs_95099 = {}
    # Getting the type of 'ismoduleroutine' (line 210)
    ismoduleroutine_95097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 14), 'ismoduleroutine', False)
    # Calling ismoduleroutine(args, kwargs) (line 210)
    ismoduleroutine_call_result_95100 = invoke(stypy.reporting.localization.Localization(__file__, 210, 14), ismoduleroutine_95097, *[rout_95098], **kwargs_95099)
    
    # Assigning a type to the variable 'f90mode' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'f90mode', ismoduleroutine_call_result_95100)
    
    # Assigning a Subscript to a Name (line 212):
    
    # Obtaining the type of the subscript
    str_95101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 16), 'str', 'args')
    # Getting the type of 'rout' (line 212)
    rout_95102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 11), 'rout')
    # Obtaining the member '__getitem__' of a type (line 212)
    getitem___95103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 11), rout_95102, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 212)
    subscript_call_result_95104 = invoke(stypy.reporting.localization.Localization(__file__, 212, 11), getitem___95103, str_95101)
    
    # Assigning a type to the variable 'args' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'args', subscript_call_result_95104)
    
    # Assigning a Call to a Name (line 214):
    
    # Call to join(...): (line 214)
    # Processing the call arguments (line 214)
    # Getting the type of 'args' (line 214)
    args_95107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 22), 'args', False)
    # Processing the call keyword arguments (line 214)
    kwargs_95108 = {}
    str_95105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 12), 'str', ', ')
    # Obtaining the member 'join' of a type (line 214)
    join_95106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 12), str_95105, 'join')
    # Calling join(args, kwargs) (line 214)
    join_call_result_95109 = invoke(stypy.reporting.localization.Localization(__file__, 214, 12), join_95106, *[args_95107], **kwargs_95108)
    
    # Assigning a type to the variable 'sargs' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'sargs', join_call_result_95109)
    
    # Getting the type of 'f90mode' (line 215)
    f90mode_95110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 7), 'f90mode')
    # Testing the type of an if condition (line 215)
    if_condition_95111 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 4), f90mode_95110)
    # Assigning a type to the variable 'if_condition_95111' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'if_condition_95111', if_condition_95111)
    # SSA begins for if statement (line 215)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to add(...): (line 216)
    # Processing the call arguments (line 216)
    str_95113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 12), 'str', 'subroutine f2pywrap_%s_%s (%s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 217)
    tuple_95114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 217)
    # Adding element type (line 217)
    
    # Obtaining the type of the subscript
    str_95115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 18), 'str', 'modulename')
    # Getting the type of 'rout' (line 217)
    rout_95116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 13), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 217)
    getitem___95117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 13), rout_95116, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 217)
    subscript_call_result_95118 = invoke(stypy.reporting.localization.Localization(__file__, 217, 13), getitem___95117, str_95115)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 13), tuple_95114, subscript_call_result_95118)
    # Adding element type (line 217)
    # Getting the type of 'name' (line 217)
    name_95119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 33), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 13), tuple_95114, name_95119)
    # Adding element type (line 217)
    # Getting the type of 'sargs' (line 217)
    sargs_95120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 39), 'sargs', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 13), tuple_95114, sargs_95120)
    
    # Applying the binary operator '%' (line 216)
    result_mod_95121 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 12), '%', str_95113, tuple_95114)
    
    # Processing the call keyword arguments (line 216)
    kwargs_95122 = {}
    # Getting the type of 'add' (line 216)
    add_95112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'add', False)
    # Calling add(args, kwargs) (line 216)
    add_call_result_95123 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), add_95112, *[result_mod_95121], **kwargs_95122)
    
    
    
    # Getting the type of 'signature' (line 218)
    signature_95124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 15), 'signature')
    # Applying the 'not' unary operator (line 218)
    result_not__95125 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 11), 'not', signature_95124)
    
    # Testing the type of an if condition (line 218)
    if_condition_95126 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 218, 8), result_not__95125)
    # Assigning a type to the variable 'if_condition_95126' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'if_condition_95126', if_condition_95126)
    # SSA begins for if statement (line 218)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to add(...): (line 219)
    # Processing the call arguments (line 219)
    str_95128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 16), 'str', 'use %s, only : %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 219)
    tuple_95129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 219)
    # Adding element type (line 219)
    
    # Obtaining the type of the subscript
    str_95130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 44), 'str', 'modulename')
    # Getting the type of 'rout' (line 219)
    rout_95131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 39), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 219)
    getitem___95132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 39), rout_95131, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 219)
    subscript_call_result_95133 = invoke(stypy.reporting.localization.Localization(__file__, 219, 39), getitem___95132, str_95130)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 39), tuple_95129, subscript_call_result_95133)
    # Adding element type (line 219)
    # Getting the type of 'fortranname' (line 219)
    fortranname_95134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 59), 'fortranname', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 39), tuple_95129, fortranname_95134)
    
    # Applying the binary operator '%' (line 219)
    result_mod_95135 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 16), '%', str_95128, tuple_95129)
    
    # Processing the call keyword arguments (line 219)
    kwargs_95136 = {}
    # Getting the type of 'add' (line 219)
    add_95127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'add', False)
    # Calling add(args, kwargs) (line 219)
    add_call_result_95137 = invoke(stypy.reporting.localization.Localization(__file__, 219, 12), add_95127, *[result_mod_95135], **kwargs_95136)
    
    # SSA join for if statement (line 218)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 215)
    module_type_store.open_ssa_branch('else')
    
    # Call to add(...): (line 221)
    # Processing the call arguments (line 221)
    str_95139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 12), 'str', 'subroutine f2pywrap%s (%s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 221)
    tuple_95140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 221)
    # Adding element type (line 221)
    # Getting the type of 'name' (line 221)
    name_95141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 44), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 44), tuple_95140, name_95141)
    # Adding element type (line 221)
    # Getting the type of 'sargs' (line 221)
    sargs_95142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 50), 'sargs', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 44), tuple_95140, sargs_95142)
    
    # Applying the binary operator '%' (line 221)
    result_mod_95143 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 12), '%', str_95139, tuple_95140)
    
    # Processing the call keyword arguments (line 221)
    kwargs_95144 = {}
    # Getting the type of 'add' (line 221)
    add_95138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'add', False)
    # Calling add(args, kwargs) (line 221)
    add_call_result_95145 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), add_95138, *[result_mod_95143], **kwargs_95144)
    
    
    
    # Getting the type of 'need_interface' (line 222)
    need_interface_95146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 15), 'need_interface')
    # Applying the 'not' unary operator (line 222)
    result_not__95147 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 11), 'not', need_interface_95146)
    
    # Testing the type of an if condition (line 222)
    if_condition_95148 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 222, 8), result_not__95147)
    # Assigning a type to the variable 'if_condition_95148' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'if_condition_95148', if_condition_95148)
    # SSA begins for if statement (line 222)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to add(...): (line 223)
    # Processing the call arguments (line 223)
    str_95150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 16), 'str', 'external %s')
    # Getting the type of 'fortranname' (line 223)
    fortranname_95151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 33), 'fortranname', False)
    # Applying the binary operator '%' (line 223)
    result_mod_95152 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 16), '%', str_95150, fortranname_95151)
    
    # Processing the call keyword arguments (line 223)
    kwargs_95153 = {}
    # Getting the type of 'add' (line 223)
    add_95149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'add', False)
    # Calling add(args, kwargs) (line 223)
    add_call_result_95154 = invoke(stypy.reporting.localization.Localization(__file__, 223, 12), add_95149, *[result_mod_95152], **kwargs_95153)
    
    # SSA join for if statement (line 222)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 215)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'need_interface' (line 225)
    need_interface_95155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 7), 'need_interface')
    # Testing the type of an if condition (line 225)
    if_condition_95156 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 225, 4), need_interface_95155)
    # Assigning a type to the variable 'if_condition_95156' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'if_condition_95156', if_condition_95156)
    # SSA begins for if statement (line 225)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to split(...): (line 226)
    # Processing the call arguments (line 226)
    str_95162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 50), 'str', '\n')
    # Processing the call keyword arguments (line 226)
    kwargs_95163 = {}
    
    # Obtaining the type of the subscript
    str_95157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 25), 'str', 'saved_interface')
    # Getting the type of 'rout' (line 226)
    rout_95158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 20), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 226)
    getitem___95159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 20), rout_95158, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 226)
    subscript_call_result_95160 = invoke(stypy.reporting.localization.Localization(__file__, 226, 20), getitem___95159, str_95157)
    
    # Obtaining the member 'split' of a type (line 226)
    split_95161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 20), subscript_call_result_95160, 'split')
    # Calling split(args, kwargs) (line 226)
    split_call_result_95164 = invoke(stypy.reporting.localization.Localization(__file__, 226, 20), split_95161, *[str_95162], **kwargs_95163)
    
    # Testing the type of a for loop iterable (line 226)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 226, 8), split_call_result_95164)
    # Getting the type of the for loop variable (line 226)
    for_loop_var_95165 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 226, 8), split_call_result_95164)
    # Assigning a type to the variable 'line' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'line', for_loop_var_95165)
    # SSA begins for a for statement (line 226)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to startswith(...): (line 227)
    # Processing the call arguments (line 227)
    str_95171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 40), 'str', 'use ')
    # Processing the call keyword arguments (line 227)
    kwargs_95172 = {}
    
    # Call to lstrip(...): (line 227)
    # Processing the call keyword arguments (line 227)
    kwargs_95168 = {}
    # Getting the type of 'line' (line 227)
    line_95166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 15), 'line', False)
    # Obtaining the member 'lstrip' of a type (line 227)
    lstrip_95167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 15), line_95166, 'lstrip')
    # Calling lstrip(args, kwargs) (line 227)
    lstrip_call_result_95169 = invoke(stypy.reporting.localization.Localization(__file__, 227, 15), lstrip_95167, *[], **kwargs_95168)
    
    # Obtaining the member 'startswith' of a type (line 227)
    startswith_95170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 15), lstrip_call_result_95169, 'startswith')
    # Calling startswith(args, kwargs) (line 227)
    startswith_call_result_95173 = invoke(stypy.reporting.localization.Localization(__file__, 227, 15), startswith_95170, *[str_95171], **kwargs_95172)
    
    # Testing the type of an if condition (line 227)
    if_condition_95174 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 12), startswith_call_result_95173)
    # Assigning a type to the variable 'if_condition_95174' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'if_condition_95174', if_condition_95174)
    # SSA begins for if statement (line 227)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to add(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'line' (line 228)
    line_95176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 20), 'line', False)
    # Processing the call keyword arguments (line 228)
    kwargs_95177 = {}
    # Getting the type of 'add' (line 228)
    add_95175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'add', False)
    # Calling add(args, kwargs) (line 228)
    add_call_result_95178 = invoke(stypy.reporting.localization.Localization(__file__, 228, 16), add_95175, *[line_95176], **kwargs_95177)
    
    # SSA join for if statement (line 227)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 225)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 230):
    
    # Obtaining an instance of the builtin type 'list' (line 230)
    list_95179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 230)
    
    # Assigning a type to the variable 'dumped_args' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'dumped_args', list_95179)
    
    # Getting the type of 'args' (line 231)
    args_95180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 13), 'args')
    # Testing the type of a for loop iterable (line 231)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 231, 4), args_95180)
    # Getting the type of the for loop variable (line 231)
    for_loop_var_95181 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 231, 4), args_95180)
    # Assigning a type to the variable 'a' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'a', for_loop_var_95181)
    # SSA begins for a for statement (line 231)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to isexternal(...): (line 232)
    # Processing the call arguments (line 232)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 232)
    a_95183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 27), 'a', False)
    # Getting the type of 'vars' (line 232)
    vars_95184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 22), 'vars', False)
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___95185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 22), vars_95184, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_95186 = invoke(stypy.reporting.localization.Localization(__file__, 232, 22), getitem___95185, a_95183)
    
    # Processing the call keyword arguments (line 232)
    kwargs_95187 = {}
    # Getting the type of 'isexternal' (line 232)
    isexternal_95182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 11), 'isexternal', False)
    # Calling isexternal(args, kwargs) (line 232)
    isexternal_call_result_95188 = invoke(stypy.reporting.localization.Localization(__file__, 232, 11), isexternal_95182, *[subscript_call_result_95186], **kwargs_95187)
    
    # Testing the type of an if condition (line 232)
    if_condition_95189 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 232, 8), isexternal_call_result_95188)
    # Assigning a type to the variable 'if_condition_95189' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'if_condition_95189', if_condition_95189)
    # SSA begins for if statement (line 232)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to add(...): (line 233)
    # Processing the call arguments (line 233)
    str_95191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 16), 'str', 'external %s')
    # Getting the type of 'a' (line 233)
    a_95192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 33), 'a', False)
    # Applying the binary operator '%' (line 233)
    result_mod_95193 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 16), '%', str_95191, a_95192)
    
    # Processing the call keyword arguments (line 233)
    kwargs_95194 = {}
    # Getting the type of 'add' (line 233)
    add_95190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'add', False)
    # Calling add(args, kwargs) (line 233)
    add_call_result_95195 = invoke(stypy.reporting.localization.Localization(__file__, 233, 12), add_95190, *[result_mod_95193], **kwargs_95194)
    
    
    # Call to append(...): (line 234)
    # Processing the call arguments (line 234)
    # Getting the type of 'a' (line 234)
    a_95198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 31), 'a', False)
    # Processing the call keyword arguments (line 234)
    kwargs_95199 = {}
    # Getting the type of 'dumped_args' (line 234)
    dumped_args_95196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'dumped_args', False)
    # Obtaining the member 'append' of a type (line 234)
    append_95197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 12), dumped_args_95196, 'append')
    # Calling append(args, kwargs) (line 234)
    append_call_result_95200 = invoke(stypy.reporting.localization.Localization(__file__, 234, 12), append_95197, *[a_95198], **kwargs_95199)
    
    # SSA join for if statement (line 232)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'args' (line 235)
    args_95201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 13), 'args')
    # Testing the type of a for loop iterable (line 235)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 235, 4), args_95201)
    # Getting the type of the for loop variable (line 235)
    for_loop_var_95202 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 235, 4), args_95201)
    # Assigning a type to the variable 'a' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'a', for_loop_var_95202)
    # SSA begins for a for statement (line 235)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'a' (line 236)
    a_95203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 11), 'a')
    # Getting the type of 'dumped_args' (line 236)
    dumped_args_95204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 16), 'dumped_args')
    # Applying the binary operator 'in' (line 236)
    result_contains_95205 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 11), 'in', a_95203, dumped_args_95204)
    
    # Testing the type of an if condition (line 236)
    if_condition_95206 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 8), result_contains_95205)
    # Assigning a type to the variable 'if_condition_95206' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'if_condition_95206', if_condition_95206)
    # SSA begins for if statement (line 236)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 236)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isscalar(...): (line 238)
    # Processing the call arguments (line 238)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 238)
    a_95208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 25), 'a', False)
    # Getting the type of 'vars' (line 238)
    vars_95209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 20), 'vars', False)
    # Obtaining the member '__getitem__' of a type (line 238)
    getitem___95210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 20), vars_95209, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 238)
    subscript_call_result_95211 = invoke(stypy.reporting.localization.Localization(__file__, 238, 20), getitem___95210, a_95208)
    
    # Processing the call keyword arguments (line 238)
    kwargs_95212 = {}
    # Getting the type of 'isscalar' (line 238)
    isscalar_95207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 11), 'isscalar', False)
    # Calling isscalar(args, kwargs) (line 238)
    isscalar_call_result_95213 = invoke(stypy.reporting.localization.Localization(__file__, 238, 11), isscalar_95207, *[subscript_call_result_95211], **kwargs_95212)
    
    # Testing the type of an if condition (line 238)
    if_condition_95214 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 238, 8), isscalar_call_result_95213)
    # Assigning a type to the variable 'if_condition_95214' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'if_condition_95214', if_condition_95214)
    # SSA begins for if statement (line 238)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to add(...): (line 239)
    # Processing the call arguments (line 239)
    
    # Call to var2fixfortran(...): (line 239)
    # Processing the call arguments (line 239)
    # Getting the type of 'vars' (line 239)
    vars_95217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 31), 'vars', False)
    # Getting the type of 'a' (line 239)
    a_95218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 37), 'a', False)
    # Processing the call keyword arguments (line 239)
    # Getting the type of 'f90mode' (line 239)
    f90mode_95219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 48), 'f90mode', False)
    keyword_95220 = f90mode_95219
    kwargs_95221 = {'f90mode': keyword_95220}
    # Getting the type of 'var2fixfortran' (line 239)
    var2fixfortran_95216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 16), 'var2fixfortran', False)
    # Calling var2fixfortran(args, kwargs) (line 239)
    var2fixfortran_call_result_95222 = invoke(stypy.reporting.localization.Localization(__file__, 239, 16), var2fixfortran_95216, *[vars_95217, a_95218], **kwargs_95221)
    
    # Processing the call keyword arguments (line 239)
    kwargs_95223 = {}
    # Getting the type of 'add' (line 239)
    add_95215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'add', False)
    # Calling add(args, kwargs) (line 239)
    add_call_result_95224 = invoke(stypy.reporting.localization.Localization(__file__, 239, 12), add_95215, *[var2fixfortran_call_result_95222], **kwargs_95223)
    
    
    # Call to append(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'a' (line 240)
    a_95227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 31), 'a', False)
    # Processing the call keyword arguments (line 240)
    kwargs_95228 = {}
    # Getting the type of 'dumped_args' (line 240)
    dumped_args_95225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'dumped_args', False)
    # Obtaining the member 'append' of a type (line 240)
    append_95226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 12), dumped_args_95225, 'append')
    # Calling append(args, kwargs) (line 240)
    append_call_result_95229 = invoke(stypy.reporting.localization.Localization(__file__, 240, 12), append_95226, *[a_95227], **kwargs_95228)
    
    # SSA join for if statement (line 238)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'args' (line 241)
    args_95230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 13), 'args')
    # Testing the type of a for loop iterable (line 241)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 241, 4), args_95230)
    # Getting the type of the for loop variable (line 241)
    for_loop_var_95231 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 241, 4), args_95230)
    # Assigning a type to the variable 'a' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'a', for_loop_var_95231)
    # SSA begins for a for statement (line 241)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'a' (line 242)
    a_95232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 11), 'a')
    # Getting the type of 'dumped_args' (line 242)
    dumped_args_95233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'dumped_args')
    # Applying the binary operator 'in' (line 242)
    result_contains_95234 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 11), 'in', a_95232, dumped_args_95233)
    
    # Testing the type of an if condition (line 242)
    if_condition_95235 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 8), result_contains_95234)
    # Assigning a type to the variable 'if_condition_95235' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'if_condition_95235', if_condition_95235)
    # SSA begins for if statement (line 242)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 242)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to add(...): (line 244)
    # Processing the call arguments (line 244)
    
    # Call to var2fixfortran(...): (line 244)
    # Processing the call arguments (line 244)
    # Getting the type of 'vars' (line 244)
    vars_95238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 27), 'vars', False)
    # Getting the type of 'a' (line 244)
    a_95239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 33), 'a', False)
    # Processing the call keyword arguments (line 244)
    # Getting the type of 'f90mode' (line 244)
    f90mode_95240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 44), 'f90mode', False)
    keyword_95241 = f90mode_95240
    kwargs_95242 = {'f90mode': keyword_95241}
    # Getting the type of 'var2fixfortran' (line 244)
    var2fixfortran_95237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'var2fixfortran', False)
    # Calling var2fixfortran(args, kwargs) (line 244)
    var2fixfortran_call_result_95243 = invoke(stypy.reporting.localization.Localization(__file__, 244, 12), var2fixfortran_95237, *[vars_95238, a_95239], **kwargs_95242)
    
    # Processing the call keyword arguments (line 244)
    kwargs_95244 = {}
    # Getting the type of 'add' (line 244)
    add_95236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'add', False)
    # Calling add(args, kwargs) (line 244)
    add_call_result_95245 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), add_95236, *[var2fixfortran_call_result_95243], **kwargs_95244)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'need_interface' (line 246)
    need_interface_95246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 7), 'need_interface')
    # Testing the type of an if condition (line 246)
    if_condition_95247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 4), need_interface_95246)
    # Assigning a type to the variable 'if_condition_95247' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'if_condition_95247', if_condition_95247)
    # SSA begins for if statement (line 246)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'f90mode' (line 247)
    f90mode_95248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 11), 'f90mode')
    # Testing the type of an if condition (line 247)
    if_condition_95249 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 247, 8), f90mode_95248)
    # Assigning a type to the variable 'if_condition_95249' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'if_condition_95249', if_condition_95249)
    # SSA begins for if statement (line 247)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    pass
    # SSA branch for the else part of an if statement (line 247)
    module_type_store.open_ssa_branch('else')
    
    # Call to add(...): (line 251)
    # Processing the call arguments (line 251)
    str_95251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 16), 'str', 'interface')
    # Processing the call keyword arguments (line 251)
    kwargs_95252 = {}
    # Getting the type of 'add' (line 251)
    add_95250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'add', False)
    # Calling add(args, kwargs) (line 251)
    add_call_result_95253 = invoke(stypy.reporting.localization.Localization(__file__, 251, 12), add_95250, *[str_95251], **kwargs_95252)
    
    
    # Call to add(...): (line 252)
    # Processing the call arguments (line 252)
    
    # Call to lstrip(...): (line 252)
    # Processing the call keyword arguments (line 252)
    kwargs_95260 = {}
    
    # Obtaining the type of the subscript
    str_95255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 21), 'str', 'saved_interface')
    # Getting the type of 'rout' (line 252)
    rout_95256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 252)
    getitem___95257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 16), rout_95256, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 252)
    subscript_call_result_95258 = invoke(stypy.reporting.localization.Localization(__file__, 252, 16), getitem___95257, str_95255)
    
    # Obtaining the member 'lstrip' of a type (line 252)
    lstrip_95259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 16), subscript_call_result_95258, 'lstrip')
    # Calling lstrip(args, kwargs) (line 252)
    lstrip_call_result_95261 = invoke(stypy.reporting.localization.Localization(__file__, 252, 16), lstrip_95259, *[], **kwargs_95260)
    
    # Processing the call keyword arguments (line 252)
    kwargs_95262 = {}
    # Getting the type of 'add' (line 252)
    add_95254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'add', False)
    # Calling add(args, kwargs) (line 252)
    add_call_result_95263 = invoke(stypy.reporting.localization.Localization(__file__, 252, 12), add_95254, *[lstrip_call_result_95261], **kwargs_95262)
    
    
    # Call to add(...): (line 253)
    # Processing the call arguments (line 253)
    str_95265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 16), 'str', 'end interface')
    # Processing the call keyword arguments (line 253)
    kwargs_95266 = {}
    # Getting the type of 'add' (line 253)
    add_95264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'add', False)
    # Calling add(args, kwargs) (line 253)
    add_call_result_95267 = invoke(stypy.reporting.localization.Localization(__file__, 253, 12), add_95264, *[str_95265], **kwargs_95266)
    
    # SSA join for if statement (line 247)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 246)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 255):
    
    # Call to join(...): (line 255)
    # Processing the call arguments (line 255)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'args' (line 255)
    args_95274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 34), 'args', False)
    comprehension_95275 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 23), args_95274)
    # Assigning a type to the variable 'a' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 23), 'a', comprehension_95275)
    
    # Getting the type of 'a' (line 255)
    a_95271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 42), 'a', False)
    # Getting the type of 'extra_args' (line 255)
    extra_args_95272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 51), 'extra_args', False)
    # Applying the binary operator 'notin' (line 255)
    result_contains_95273 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 42), 'notin', a_95271, extra_args_95272)
    
    # Getting the type of 'a' (line 255)
    a_95270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 23), 'a', False)
    list_95276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 23), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 23), list_95276, a_95270)
    # Processing the call keyword arguments (line 255)
    kwargs_95277 = {}
    str_95268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 12), 'str', ', ')
    # Obtaining the member 'join' of a type (line 255)
    join_95269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 12), str_95268, 'join')
    # Calling join(args, kwargs) (line 255)
    join_call_result_95278 = invoke(stypy.reporting.localization.Localization(__file__, 255, 12), join_95269, *[list_95276], **kwargs_95277)
    
    # Assigning a type to the variable 'sargs' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'sargs', join_call_result_95278)
    
    
    # Getting the type of 'signature' (line 257)
    signature_95279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 11), 'signature')
    # Applying the 'not' unary operator (line 257)
    result_not__95280 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 7), 'not', signature_95279)
    
    # Testing the type of an if condition (line 257)
    if_condition_95281 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 257, 4), result_not__95280)
    # Assigning a type to the variable 'if_condition_95281' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'if_condition_95281', if_condition_95281)
    # SSA begins for if statement (line 257)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to add(...): (line 258)
    # Processing the call arguments (line 258)
    str_95283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 12), 'str', 'call %s(%s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 258)
    tuple_95284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 258)
    # Adding element type (line 258)
    # Getting the type of 'fortranname' (line 258)
    fortranname_95285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 29), 'fortranname', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 29), tuple_95284, fortranname_95285)
    # Adding element type (line 258)
    # Getting the type of 'sargs' (line 258)
    sargs_95286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 42), 'sargs', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 29), tuple_95284, sargs_95286)
    
    # Applying the binary operator '%' (line 258)
    result_mod_95287 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 12), '%', str_95283, tuple_95284)
    
    # Processing the call keyword arguments (line 258)
    kwargs_95288 = {}
    # Getting the type of 'add' (line 258)
    add_95282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'add', False)
    # Calling add(args, kwargs) (line 258)
    add_call_result_95289 = invoke(stypy.reporting.localization.Localization(__file__, 258, 8), add_95282, *[result_mod_95287], **kwargs_95288)
    
    # SSA join for if statement (line 257)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'f90mode' (line 259)
    f90mode_95290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 7), 'f90mode')
    # Testing the type of an if condition (line 259)
    if_condition_95291 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 259, 4), f90mode_95290)
    # Assigning a type to the variable 'if_condition_95291' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'if_condition_95291', if_condition_95291)
    # SSA begins for if statement (line 259)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to add(...): (line 260)
    # Processing the call arguments (line 260)
    str_95293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 12), 'str', 'end subroutine f2pywrap_%s_%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 260)
    tuple_95294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 260)
    # Adding element type (line 260)
    
    # Obtaining the type of the subscript
    str_95295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 52), 'str', 'modulename')
    # Getting the type of 'rout' (line 260)
    rout_95296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 47), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 260)
    getitem___95297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 47), rout_95296, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 260)
    subscript_call_result_95298 = invoke(stypy.reporting.localization.Localization(__file__, 260, 47), getitem___95297, str_95295)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 47), tuple_95294, subscript_call_result_95298)
    # Adding element type (line 260)
    # Getting the type of 'name' (line 260)
    name_95299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 67), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 47), tuple_95294, name_95299)
    
    # Applying the binary operator '%' (line 260)
    result_mod_95300 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 12), '%', str_95293, tuple_95294)
    
    # Processing the call keyword arguments (line 260)
    kwargs_95301 = {}
    # Getting the type of 'add' (line 260)
    add_95292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'add', False)
    # Calling add(args, kwargs) (line 260)
    add_call_result_95302 = invoke(stypy.reporting.localization.Localization(__file__, 260, 8), add_95292, *[result_mod_95300], **kwargs_95301)
    
    # SSA branch for the else part of an if statement (line 259)
    module_type_store.open_ssa_branch('else')
    
    # Call to add(...): (line 262)
    # Processing the call arguments (line 262)
    str_95304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 12), 'str', 'end')
    # Processing the call keyword arguments (line 262)
    kwargs_95305 = {}
    # Getting the type of 'add' (line 262)
    add_95303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'add', False)
    # Calling add(args, kwargs) (line 262)
    add_call_result_95306 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), add_95303, *[str_95304], **kwargs_95305)
    
    # SSA join for if statement (line 259)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining the type of the subscript
    int_95307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 15), 'int')
    # Getting the type of 'ret' (line 263)
    ret_95308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 11), 'ret')
    # Obtaining the member '__getitem__' of a type (line 263)
    getitem___95309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 11), ret_95308, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 263)
    subscript_call_result_95310 = invoke(stypy.reporting.localization.Localization(__file__, 263, 11), getitem___95309, int_95307)
    
    # Assigning a type to the variable 'stypy_return_type' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'stypy_return_type', subscript_call_result_95310)
    
    # ################# End of 'createsubrwrapper(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'createsubrwrapper' in the type store
    # Getting the type of 'stypy_return_type' (line 186)
    stypy_return_type_95311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_95311)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'createsubrwrapper'
    return stypy_return_type_95311

# Assigning a type to the variable 'createsubrwrapper' (line 186)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 0), 'createsubrwrapper', createsubrwrapper)

@norecursion
def assubr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'assubr'
    module_type_store = module_type_store.open_function_context('assubr', 266, 0, False)
    
    # Passed parameters checking function
    assubr.stypy_localization = localization
    assubr.stypy_type_of_self = None
    assubr.stypy_type_store = module_type_store
    assubr.stypy_function_name = 'assubr'
    assubr.stypy_param_names_list = ['rout']
    assubr.stypy_varargs_param_name = None
    assubr.stypy_kwargs_param_name = None
    assubr.stypy_call_defaults = defaults
    assubr.stypy_call_varargs = varargs
    assubr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assubr', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assubr', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assubr(...)' code ##################

    
    
    # Call to isfunction_wrap(...): (line 267)
    # Processing the call arguments (line 267)
    # Getting the type of 'rout' (line 267)
    rout_95313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 23), 'rout', False)
    # Processing the call keyword arguments (line 267)
    kwargs_95314 = {}
    # Getting the type of 'isfunction_wrap' (line 267)
    isfunction_wrap_95312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 7), 'isfunction_wrap', False)
    # Calling isfunction_wrap(args, kwargs) (line 267)
    isfunction_wrap_call_result_95315 = invoke(stypy.reporting.localization.Localization(__file__, 267, 7), isfunction_wrap_95312, *[rout_95313], **kwargs_95314)
    
    # Testing the type of an if condition (line 267)
    if_condition_95316 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 267, 4), isfunction_wrap_call_result_95315)
    # Assigning a type to the variable 'if_condition_95316' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'if_condition_95316', if_condition_95316)
    # SSA begins for if statement (line 267)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 268):
    
    # Call to getfortranname(...): (line 268)
    # Processing the call arguments (line 268)
    # Getting the type of 'rout' (line 268)
    rout_95318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 37), 'rout', False)
    # Processing the call keyword arguments (line 268)
    kwargs_95319 = {}
    # Getting the type of 'getfortranname' (line 268)
    getfortranname_95317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 22), 'getfortranname', False)
    # Calling getfortranname(args, kwargs) (line 268)
    getfortranname_call_result_95320 = invoke(stypy.reporting.localization.Localization(__file__, 268, 22), getfortranname_95317, *[rout_95318], **kwargs_95319)
    
    # Assigning a type to the variable 'fortranname' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'fortranname', getfortranname_call_result_95320)
    
    # Assigning a Subscript to a Name (line 269):
    
    # Obtaining the type of the subscript
    str_95321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 20), 'str', 'name')
    # Getting the type of 'rout' (line 269)
    rout_95322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 15), 'rout')
    # Obtaining the member '__getitem__' of a type (line 269)
    getitem___95323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 15), rout_95322, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 269)
    subscript_call_result_95324 = invoke(stypy.reporting.localization.Localization(__file__, 269, 15), getitem___95323, str_95321)
    
    # Assigning a type to the variable 'name' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'name', subscript_call_result_95324)
    
    # Call to outmess(...): (line 270)
    # Processing the call arguments (line 270)
    str_95326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 16), 'str', '\t\tCreating wrapper for Fortran function "%s"("%s")...\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 271)
    tuple_95327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 271)
    # Adding element type (line 271)
    # Getting the type of 'name' (line 271)
    name_95328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 12), tuple_95327, name_95328)
    # Adding element type (line 271)
    # Getting the type of 'fortranname' (line 271)
    fortranname_95329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 18), 'fortranname', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 12), tuple_95327, fortranname_95329)
    
    # Applying the binary operator '%' (line 270)
    result_mod_95330 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 16), '%', str_95326, tuple_95327)
    
    # Processing the call keyword arguments (line 270)
    kwargs_95331 = {}
    # Getting the type of 'outmess' (line 270)
    outmess_95325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'outmess', False)
    # Calling outmess(args, kwargs) (line 270)
    outmess_call_result_95332 = invoke(stypy.reporting.localization.Localization(__file__, 270, 8), outmess_95325, *[result_mod_95330], **kwargs_95331)
    
    
    # Assigning a Call to a Name (line 272):
    
    # Call to copy(...): (line 272)
    # Processing the call arguments (line 272)
    # Getting the type of 'rout' (line 272)
    rout_95335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 25), 'rout', False)
    # Processing the call keyword arguments (line 272)
    kwargs_95336 = {}
    # Getting the type of 'copy' (line 272)
    copy_95333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 15), 'copy', False)
    # Obtaining the member 'copy' of a type (line 272)
    copy_95334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 15), copy_95333, 'copy')
    # Calling copy(args, kwargs) (line 272)
    copy_call_result_95337 = invoke(stypy.reporting.localization.Localization(__file__, 272, 15), copy_95334, *[rout_95335], **kwargs_95336)
    
    # Assigning a type to the variable 'rout' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'rout', copy_call_result_95337)
    
    # Assigning a Name to a Name (line 273):
    # Getting the type of 'name' (line 273)
    name_95338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'name')
    # Assigning a type to the variable 'fname' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'fname', name_95338)
    
    # Assigning a Name to a Name (line 274):
    # Getting the type of 'fname' (line 274)
    fname_95339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'fname')
    # Assigning a type to the variable 'rname' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'rname', fname_95339)
    
    
    str_95340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 11), 'str', 'result')
    # Getting the type of 'rout' (line 275)
    rout_95341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 23), 'rout')
    # Applying the binary operator 'in' (line 275)
    result_contains_95342 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 11), 'in', str_95340, rout_95341)
    
    # Testing the type of an if condition (line 275)
    if_condition_95343 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 275, 8), result_contains_95342)
    # Assigning a type to the variable 'if_condition_95343' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'if_condition_95343', if_condition_95343)
    # SSA begins for if statement (line 275)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 276):
    
    # Obtaining the type of the subscript
    str_95344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 25), 'str', 'result')
    # Getting the type of 'rout' (line 276)
    rout_95345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 20), 'rout')
    # Obtaining the member '__getitem__' of a type (line 276)
    getitem___95346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 20), rout_95345, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 276)
    subscript_call_result_95347 = invoke(stypy.reporting.localization.Localization(__file__, 276, 20), getitem___95346, str_95344)
    
    # Assigning a type to the variable 'rname' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'rname', subscript_call_result_95347)
    
    # Assigning a Subscript to a Subscript (line 277):
    
    # Obtaining the type of the subscript
    # Getting the type of 'rname' (line 277)
    rname_95348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 47), 'rname')
    
    # Obtaining the type of the subscript
    str_95349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 39), 'str', 'vars')
    # Getting the type of 'rout' (line 277)
    rout_95350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 34), 'rout')
    # Obtaining the member '__getitem__' of a type (line 277)
    getitem___95351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 34), rout_95350, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 277)
    subscript_call_result_95352 = invoke(stypy.reporting.localization.Localization(__file__, 277, 34), getitem___95351, str_95349)
    
    # Obtaining the member '__getitem__' of a type (line 277)
    getitem___95353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 34), subscript_call_result_95352, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 277)
    subscript_call_result_95354 = invoke(stypy.reporting.localization.Localization(__file__, 277, 34), getitem___95353, rname_95348)
    
    
    # Obtaining the type of the subscript
    str_95355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 17), 'str', 'vars')
    # Getting the type of 'rout' (line 277)
    rout_95356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'rout')
    # Obtaining the member '__getitem__' of a type (line 277)
    getitem___95357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 12), rout_95356, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 277)
    subscript_call_result_95358 = invoke(stypy.reporting.localization.Localization(__file__, 277, 12), getitem___95357, str_95355)
    
    # Getting the type of 'fname' (line 277)
    fname_95359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 25), 'fname')
    # Storing an element on a container (line 277)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 12), subscript_call_result_95358, (fname_95359, subscript_call_result_95354))
    # SSA join for if statement (line 275)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 278):
    
    # Obtaining the type of the subscript
    # Getting the type of 'fname' (line 278)
    fname_95360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 28), 'fname')
    
    # Obtaining the type of the subscript
    str_95361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 20), 'str', 'vars')
    # Getting the type of 'rout' (line 278)
    rout_95362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 15), 'rout')
    # Obtaining the member '__getitem__' of a type (line 278)
    getitem___95363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 15), rout_95362, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 278)
    subscript_call_result_95364 = invoke(stypy.reporting.localization.Localization(__file__, 278, 15), getitem___95363, str_95361)
    
    # Obtaining the member '__getitem__' of a type (line 278)
    getitem___95365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 15), subscript_call_result_95364, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 278)
    subscript_call_result_95366 = invoke(stypy.reporting.localization.Localization(__file__, 278, 15), getitem___95365, fname_95360)
    
    # Assigning a type to the variable 'fvar' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'fvar', subscript_call_result_95366)
    
    
    
    # Call to isintent_out(...): (line 279)
    # Processing the call arguments (line 279)
    # Getting the type of 'fvar' (line 279)
    fvar_95368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 28), 'fvar', False)
    # Processing the call keyword arguments (line 279)
    kwargs_95369 = {}
    # Getting the type of 'isintent_out' (line 279)
    isintent_out_95367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 15), 'isintent_out', False)
    # Calling isintent_out(args, kwargs) (line 279)
    isintent_out_call_result_95370 = invoke(stypy.reporting.localization.Localization(__file__, 279, 15), isintent_out_95367, *[fvar_95368], **kwargs_95369)
    
    # Applying the 'not' unary operator (line 279)
    result_not__95371 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 11), 'not', isintent_out_call_result_95370)
    
    # Testing the type of an if condition (line 279)
    if_condition_95372 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 8), result_not__95371)
    # Assigning a type to the variable 'if_condition_95372' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'if_condition_95372', if_condition_95372)
    # SSA begins for if statement (line 279)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    str_95373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 15), 'str', 'intent')
    # Getting the type of 'fvar' (line 280)
    fvar_95374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 31), 'fvar')
    # Applying the binary operator 'notin' (line 280)
    result_contains_95375 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 15), 'notin', str_95373, fvar_95374)
    
    # Testing the type of an if condition (line 280)
    if_condition_95376 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 280, 12), result_contains_95375)
    # Assigning a type to the variable 'if_condition_95376' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'if_condition_95376', if_condition_95376)
    # SSA begins for if statement (line 280)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Subscript (line 281):
    
    # Obtaining an instance of the builtin type 'list' (line 281)
    list_95377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 281)
    
    # Getting the type of 'fvar' (line 281)
    fvar_95378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'fvar')
    str_95379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 21), 'str', 'intent')
    # Storing an element on a container (line 281)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 16), fvar_95378, (str_95379, list_95377))
    # SSA join for if statement (line 280)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 282)
    # Processing the call arguments (line 282)
    str_95385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 34), 'str', 'out')
    # Processing the call keyword arguments (line 282)
    kwargs_95386 = {}
    
    # Obtaining the type of the subscript
    str_95380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 17), 'str', 'intent')
    # Getting the type of 'fvar' (line 282)
    fvar_95381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'fvar', False)
    # Obtaining the member '__getitem__' of a type (line 282)
    getitem___95382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 12), fvar_95381, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 282)
    subscript_call_result_95383 = invoke(stypy.reporting.localization.Localization(__file__, 282, 12), getitem___95382, str_95380)
    
    # Obtaining the member 'append' of a type (line 282)
    append_95384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 12), subscript_call_result_95383, 'append')
    # Calling append(args, kwargs) (line 282)
    append_call_result_95387 = invoke(stypy.reporting.localization.Localization(__file__, 282, 12), append_95384, *[str_95385], **kwargs_95386)
    
    
    # Assigning a Num to a Name (line 283):
    int_95388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 19), 'int')
    # Assigning a type to the variable 'flag' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'flag', int_95388)
    
    
    # Obtaining the type of the subscript
    str_95389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 26), 'str', 'intent')
    # Getting the type of 'fvar' (line 284)
    fvar_95390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 21), 'fvar')
    # Obtaining the member '__getitem__' of a type (line 284)
    getitem___95391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 21), fvar_95390, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 284)
    subscript_call_result_95392 = invoke(stypy.reporting.localization.Localization(__file__, 284, 21), getitem___95391, str_95389)
    
    # Testing the type of a for loop iterable (line 284)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 284, 12), subscript_call_result_95392)
    # Getting the type of the for loop variable (line 284)
    for_loop_var_95393 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 284, 12), subscript_call_result_95392)
    # Assigning a type to the variable 'i' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'i', for_loop_var_95393)
    # SSA begins for a for statement (line 284)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to startswith(...): (line 285)
    # Processing the call arguments (line 285)
    str_95396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 32), 'str', 'out=')
    # Processing the call keyword arguments (line 285)
    kwargs_95397 = {}
    # Getting the type of 'i' (line 285)
    i_95394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 19), 'i', False)
    # Obtaining the member 'startswith' of a type (line 285)
    startswith_95395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 19), i_95394, 'startswith')
    # Calling startswith(args, kwargs) (line 285)
    startswith_call_result_95398 = invoke(stypy.reporting.localization.Localization(__file__, 285, 19), startswith_95395, *[str_95396], **kwargs_95397)
    
    # Testing the type of an if condition (line 285)
    if_condition_95399 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 285, 16), startswith_call_result_95398)
    # Assigning a type to the variable 'if_condition_95399' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 16), 'if_condition_95399', if_condition_95399)
    # SSA begins for if statement (line 285)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 286):
    int_95400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 27), 'int')
    # Assigning a type to the variable 'flag' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 20), 'flag', int_95400)
    # SSA join for if statement (line 285)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'flag' (line 288)
    flag_95401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 15), 'flag')
    # Testing the type of an if condition (line 288)
    if_condition_95402 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 288, 12), flag_95401)
    # Assigning a type to the variable 'if_condition_95402' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'if_condition_95402', if_condition_95402)
    # SSA begins for if statement (line 288)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 289)
    # Processing the call arguments (line 289)
    str_95408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 38), 'str', 'out=%s')
    # Getting the type of 'rname' (line 289)
    rname_95409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 50), 'rname', False)
    # Applying the binary operator '%' (line 289)
    result_mod_95410 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 38), '%', str_95408, rname_95409)
    
    # Processing the call keyword arguments (line 289)
    kwargs_95411 = {}
    
    # Obtaining the type of the subscript
    str_95403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 21), 'str', 'intent')
    # Getting the type of 'fvar' (line 289)
    fvar_95404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 16), 'fvar', False)
    # Obtaining the member '__getitem__' of a type (line 289)
    getitem___95405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 16), fvar_95404, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 289)
    subscript_call_result_95406 = invoke(stypy.reporting.localization.Localization(__file__, 289, 16), getitem___95405, str_95403)
    
    # Obtaining the member 'append' of a type (line 289)
    append_95407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 16), subscript_call_result_95406, 'append')
    # Calling append(args, kwargs) (line 289)
    append_call_result_95412 = invoke(stypy.reporting.localization.Localization(__file__, 289, 16), append_95407, *[result_mod_95410], **kwargs_95411)
    
    # SSA join for if statement (line 288)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 279)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Subscript (line 290):
    
    # Obtaining an instance of the builtin type 'list' (line 290)
    list_95413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 290)
    # Adding element type (line 290)
    # Getting the type of 'fname' (line 290)
    fname_95414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 27), 'fname')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 26), list_95413, fname_95414)
    
    
    # Obtaining the type of the subscript
    str_95415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 41), 'str', 'args')
    # Getting the type of 'rout' (line 290)
    rout_95416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 36), 'rout')
    # Obtaining the member '__getitem__' of a type (line 290)
    getitem___95417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 36), rout_95416, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 290)
    subscript_call_result_95418 = invoke(stypy.reporting.localization.Localization(__file__, 290, 36), getitem___95417, str_95415)
    
    # Applying the binary operator '+' (line 290)
    result_add_95419 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 26), '+', list_95413, subscript_call_result_95418)
    
    
    # Obtaining the type of the subscript
    str_95420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 13), 'str', 'args')
    # Getting the type of 'rout' (line 290)
    rout_95421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'rout')
    # Obtaining the member '__getitem__' of a type (line 290)
    getitem___95422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), rout_95421, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 290)
    subscript_call_result_95423 = invoke(stypy.reporting.localization.Localization(__file__, 290, 8), getitem___95422, str_95420)
    
    slice_95424 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 290, 8), None, None, None)
    # Storing an element on a container (line 290)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 8), subscript_call_result_95423, (slice_95424, result_add_95419))
    
    # Obtaining an instance of the builtin type 'tuple' (line 291)
    tuple_95425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 291)
    # Adding element type (line 291)
    # Getting the type of 'rout' (line 291)
    rout_95426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 15), 'rout')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 15), tuple_95425, rout_95426)
    # Adding element type (line 291)
    
    # Call to createfuncwrapper(...): (line 291)
    # Processing the call arguments (line 291)
    # Getting the type of 'rout' (line 291)
    rout_95428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 39), 'rout', False)
    # Processing the call keyword arguments (line 291)
    kwargs_95429 = {}
    # Getting the type of 'createfuncwrapper' (line 291)
    createfuncwrapper_95427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 21), 'createfuncwrapper', False)
    # Calling createfuncwrapper(args, kwargs) (line 291)
    createfuncwrapper_call_result_95430 = invoke(stypy.reporting.localization.Localization(__file__, 291, 21), createfuncwrapper_95427, *[rout_95428], **kwargs_95429)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 15), tuple_95425, createfuncwrapper_call_result_95430)
    
    # Assigning a type to the variable 'stypy_return_type' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'stypy_return_type', tuple_95425)
    # SSA join for if statement (line 267)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to issubroutine_wrap(...): (line 292)
    # Processing the call arguments (line 292)
    # Getting the type of 'rout' (line 292)
    rout_95432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 25), 'rout', False)
    # Processing the call keyword arguments (line 292)
    kwargs_95433 = {}
    # Getting the type of 'issubroutine_wrap' (line 292)
    issubroutine_wrap_95431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 7), 'issubroutine_wrap', False)
    # Calling issubroutine_wrap(args, kwargs) (line 292)
    issubroutine_wrap_call_result_95434 = invoke(stypy.reporting.localization.Localization(__file__, 292, 7), issubroutine_wrap_95431, *[rout_95432], **kwargs_95433)
    
    # Testing the type of an if condition (line 292)
    if_condition_95435 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 292, 4), issubroutine_wrap_call_result_95434)
    # Assigning a type to the variable 'if_condition_95435' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'if_condition_95435', if_condition_95435)
    # SSA begins for if statement (line 292)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 293):
    
    # Call to getfortranname(...): (line 293)
    # Processing the call arguments (line 293)
    # Getting the type of 'rout' (line 293)
    rout_95437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 37), 'rout', False)
    # Processing the call keyword arguments (line 293)
    kwargs_95438 = {}
    # Getting the type of 'getfortranname' (line 293)
    getfortranname_95436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 22), 'getfortranname', False)
    # Calling getfortranname(args, kwargs) (line 293)
    getfortranname_call_result_95439 = invoke(stypy.reporting.localization.Localization(__file__, 293, 22), getfortranname_95436, *[rout_95437], **kwargs_95438)
    
    # Assigning a type to the variable 'fortranname' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'fortranname', getfortranname_call_result_95439)
    
    # Assigning a Subscript to a Name (line 294):
    
    # Obtaining the type of the subscript
    str_95440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 20), 'str', 'name')
    # Getting the type of 'rout' (line 294)
    rout_95441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 15), 'rout')
    # Obtaining the member '__getitem__' of a type (line 294)
    getitem___95442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 15), rout_95441, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 294)
    subscript_call_result_95443 = invoke(stypy.reporting.localization.Localization(__file__, 294, 15), getitem___95442, str_95440)
    
    # Assigning a type to the variable 'name' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'name', subscript_call_result_95443)
    
    # Call to outmess(...): (line 295)
    # Processing the call arguments (line 295)
    str_95445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 16), 'str', '\t\tCreating wrapper for Fortran subroutine "%s"("%s")...\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 296)
    tuple_95446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 296)
    # Adding element type (line 296)
    # Getting the type of 'name' (line 296)
    name_95447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 12), tuple_95446, name_95447)
    # Adding element type (line 296)
    # Getting the type of 'fortranname' (line 296)
    fortranname_95448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 18), 'fortranname', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 12), tuple_95446, fortranname_95448)
    
    # Applying the binary operator '%' (line 295)
    result_mod_95449 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 16), '%', str_95445, tuple_95446)
    
    # Processing the call keyword arguments (line 295)
    kwargs_95450 = {}
    # Getting the type of 'outmess' (line 295)
    outmess_95444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'outmess', False)
    # Calling outmess(args, kwargs) (line 295)
    outmess_call_result_95451 = invoke(stypy.reporting.localization.Localization(__file__, 295, 8), outmess_95444, *[result_mod_95449], **kwargs_95450)
    
    
    # Assigning a Call to a Name (line 297):
    
    # Call to copy(...): (line 297)
    # Processing the call arguments (line 297)
    # Getting the type of 'rout' (line 297)
    rout_95454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 25), 'rout', False)
    # Processing the call keyword arguments (line 297)
    kwargs_95455 = {}
    # Getting the type of 'copy' (line 297)
    copy_95452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 15), 'copy', False)
    # Obtaining the member 'copy' of a type (line 297)
    copy_95453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 15), copy_95452, 'copy')
    # Calling copy(args, kwargs) (line 297)
    copy_call_result_95456 = invoke(stypy.reporting.localization.Localization(__file__, 297, 15), copy_95453, *[rout_95454], **kwargs_95455)
    
    # Assigning a type to the variable 'rout' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'rout', copy_call_result_95456)
    
    # Obtaining an instance of the builtin type 'tuple' (line 298)
    tuple_95457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 298)
    # Adding element type (line 298)
    # Getting the type of 'rout' (line 298)
    rout_95458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 15), 'rout')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 15), tuple_95457, rout_95458)
    # Adding element type (line 298)
    
    # Call to createsubrwrapper(...): (line 298)
    # Processing the call arguments (line 298)
    # Getting the type of 'rout' (line 298)
    rout_95460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 39), 'rout', False)
    # Processing the call keyword arguments (line 298)
    kwargs_95461 = {}
    # Getting the type of 'createsubrwrapper' (line 298)
    createsubrwrapper_95459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 21), 'createsubrwrapper', False)
    # Calling createsubrwrapper(args, kwargs) (line 298)
    createsubrwrapper_call_result_95462 = invoke(stypy.reporting.localization.Localization(__file__, 298, 21), createsubrwrapper_95459, *[rout_95460], **kwargs_95461)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 15), tuple_95457, createsubrwrapper_call_result_95462)
    
    # Assigning a type to the variable 'stypy_return_type' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'stypy_return_type', tuple_95457)
    # SSA join for if statement (line 292)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 299)
    tuple_95463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 299)
    # Adding element type (line 299)
    # Getting the type of 'rout' (line 299)
    rout_95464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 11), 'rout')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 11), tuple_95463, rout_95464)
    # Adding element type (line 299)
    str_95465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 17), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 11), tuple_95463, str_95465)
    
    # Assigning a type to the variable 'stypy_return_type' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'stypy_return_type', tuple_95463)
    
    # ################# End of 'assubr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assubr' in the type store
    # Getting the type of 'stypy_return_type' (line 266)
    stypy_return_type_95466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_95466)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assubr'
    return stypy_return_type_95466

# Assigning a type to the variable 'assubr' (line 266)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 0), 'assubr', assubr)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
