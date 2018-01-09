
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: '''
3: 
4: Build F90 module support for f2py2e.
5: 
6: Copyright 2000 Pearu Peterson all rights reserved,
7: Pearu Peterson <pearu@ioc.ee>
8: Permission to use, modify, and distribute this software is given under the
9: terms of the NumPy License.
10: 
11: NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
12: $Date: 2005/02/03 19:30:23 $
13: Pearu Peterson
14: 
15: '''
16: from __future__ import division, absolute_import, print_function
17: 
18: __version__ = "$Revision: 1.27 $"[10:-1]
19: 
20: f2py_version = 'See `f2py -v`'
21: 
22: import numpy as np
23: 
24: from . import capi_maps
25: from . import func2subr
26: from .crackfortran import undo_rmbadname, undo_rmbadname1
27: 
28: # The eviroment provided by auxfuncs.py is needed for some calls to eval.
29: # As the needed functions cannot be determined by static inspection of the
30: # code, it is safest to use import * pending a major refactoring of f2py.
31: from .auxfuncs import *
32: 
33: options = {}
34: 
35: 
36: def findf90modules(m):
37:     if ismodule(m):
38:         return [m]
39:     if not hasbody(m):
40:         return []
41:     ret = []
42:     for b in m['body']:
43:         if ismodule(b):
44:             ret.append(b)
45:         else:
46:             ret = ret + findf90modules(b)
47:     return ret
48: 
49: fgetdims1 = '''\
50:       external f2pysetdata
51:       logical ns
52:       integer r,i
53:       integer(%d) s(*)
54:       ns = .FALSE.
55:       if (allocated(d)) then
56:          do i=1,r
57:             if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
58:                ns = .TRUE.
59:             end if
60:          end do
61:          if (ns) then
62:             deallocate(d)
63:          end if
64:       end if
65:       if ((.not.allocated(d)).and.(s(1).ge.1)) then''' % np.intp().itemsize
66: 
67: fgetdims2 = '''\
68:       end if
69:       if (allocated(d)) then
70:          do i=1,r
71:             s(i) = size(d,i)
72:          end do
73:       end if
74:       flag = 1
75:       call f2pysetdata(d,allocated(d))'''
76: 
77: fgetdims2_sa = '''\
78:       end if
79:       if (allocated(d)) then
80:          do i=1,r
81:             s(i) = size(d,i)
82:          end do
83:          !s(r) must be equal to len(d(1))
84:       end if
85:       flag = 2
86:       call f2pysetdata(d,allocated(d))'''
87: 
88: 
89: def buildhooks(pymod):
90:     global fgetdims1, fgetdims2
91:     from . import rules
92:     ret = {'f90modhooks': [], 'initf90modhooks': [], 'body': [],
93:            'need': ['F_FUNC', 'arrayobject.h'],
94:            'separatorsfor': {'includes0': '\n', 'includes': '\n'},
95:            'docs': ['"Fortran 90/95 modules:\\n"'],
96:            'latexdoc': []}
97:     fhooks = ['']
98: 
99:     def fadd(line, s=fhooks):
100:         s[0] = '%s\n      %s' % (s[0], line)
101:     doc = ['']
102: 
103:     def dadd(line, s=doc):
104:         s[0] = '%s\n%s' % (s[0], line)
105:     for m in findf90modules(pymod):
106:         sargs, fargs, efargs, modobjs, notvars, onlyvars = [], [], [], [], [
107:             m['name']], []
108:         sargsp = []
109:         ifargs = []
110:         mfargs = []
111:         if hasbody(m):
112:             for b in m['body']:
113:                 notvars.append(b['name'])
114:         for n in m['vars'].keys():
115:             var = m['vars'][n]
116:             if (n not in notvars) and (not l_or(isintent_hide, isprivate)(var)):
117:                 onlyvars.append(n)
118:                 mfargs.append(n)
119:         outmess('\t\tConstructing F90 module support for "%s"...\n' %
120:                 (m['name']))
121:         if onlyvars:
122:             outmess('\t\t  Variables: %s\n' % (' '.join(onlyvars)))
123:         chooks = ['']
124: 
125:         def cadd(line, s=chooks):
126:             s[0] = '%s\n%s' % (s[0], line)
127:         ihooks = ['']
128: 
129:         def iadd(line, s=ihooks):
130:             s[0] = '%s\n%s' % (s[0], line)
131: 
132:         vrd = capi_maps.modsign2map(m)
133:         cadd('static FortranDataDef f2py_%s_def[] = {' % (m['name']))
134:         dadd('\\subsection{Fortran 90/95 module \\texttt{%s}}\n' % (m['name']))
135:         if hasnote(m):
136:             note = m['note']
137:             if isinstance(note, list):
138:                 note = '\n'.join(note)
139:             dadd(note)
140:         if onlyvars:
141:             dadd('\\begin{description}')
142:         for n in onlyvars:
143:             var = m['vars'][n]
144:             modobjs.append(n)
145:             ct = capi_maps.getctype(var)
146:             at = capi_maps.c2capi_map[ct]
147:             dm = capi_maps.getarrdims(n, var)
148:             dms = dm['dims'].replace('*', '-1').strip()
149:             dms = dms.replace(':', '-1').strip()
150:             if not dms:
151:                 dms = '-1'
152:             use_fgetdims2 = fgetdims2
153:             if isstringarray(var):
154:                 if 'charselector' in var and 'len' in var['charselector']:
155:                     cadd('\t{"%s",%s,{{%s,%s}},%s},'
156:                          % (undo_rmbadname1(n), dm['rank'], dms, var['charselector']['len'], at))
157:                     use_fgetdims2 = fgetdims2_sa
158:                 else:
159:                     cadd('\t{"%s",%s,{{%s}},%s},' %
160:                          (undo_rmbadname1(n), dm['rank'], dms, at))
161:             else:
162:                 cadd('\t{"%s",%s,{{%s}},%s},' %
163:                      (undo_rmbadname1(n), dm['rank'], dms, at))
164:             dadd('\\item[]{{}\\verb@%s@{}}' %
165:                  (capi_maps.getarrdocsign(n, var)))
166:             if hasnote(var):
167:                 note = var['note']
168:                 if isinstance(note, list):
169:                     note = '\n'.join(note)
170:                 dadd('--- %s' % (note))
171:             if isallocatable(var):
172:                 fargs.append('f2py_%s_getdims_%s' % (m['name'], n))
173:                 efargs.append(fargs[-1])
174:                 sargs.append(
175:                     'void (*%s)(int*,int*,void(*)(char*,int*),int*)' % (n))
176:                 sargsp.append('void (*)(int*,int*,void(*)(char*,int*),int*)')
177:                 iadd('\tf2py_%s_def[i_f2py++].func = %s;' % (m['name'], n))
178:                 fadd('subroutine %s(r,s,f2pysetdata,flag)' % (fargs[-1]))
179:                 fadd('use %s, only: d => %s\n' %
180:                      (m['name'], undo_rmbadname1(n)))
181:                 fadd('integer flag\n')
182:                 fhooks[0] = fhooks[0] + fgetdims1
183:                 dms = eval('range(1,%s+1)' % (dm['rank']))
184:                 fadd(' allocate(d(%s))\n' %
185:                      (','.join(['s(%s)' % i for i in dms])))
186:                 fhooks[0] = fhooks[0] + use_fgetdims2
187:                 fadd('end subroutine %s' % (fargs[-1]))
188:             else:
189:                 fargs.append(n)
190:                 sargs.append('char *%s' % (n))
191:                 sargsp.append('char*')
192:                 iadd('\tf2py_%s_def[i_f2py++].data = %s;' % (m['name'], n))
193:         if onlyvars:
194:             dadd('\\end{description}')
195:         if hasbody(m):
196:             for b in m['body']:
197:                 if not isroutine(b):
198:                     print('Skipping', b['block'], b['name'])
199:                     continue
200:                 modobjs.append('%s()' % (b['name']))
201:                 b['modulename'] = m['name']
202:                 api, wrap = rules.buildapi(b)
203:                 if isfunction(b):
204:                     fhooks[0] = fhooks[0] + wrap
205:                     fargs.append('f2pywrap_%s_%s' % (m['name'], b['name']))
206:                     ifargs.append(func2subr.createfuncwrapper(b, signature=1))
207:                 else:
208:                     if wrap:
209:                         fhooks[0] = fhooks[0] + wrap
210:                         fargs.append('f2pywrap_%s_%s' % (m['name'], b['name']))
211:                         ifargs.append(
212:                             func2subr.createsubrwrapper(b, signature=1))
213:                     else:
214:                         fargs.append(b['name'])
215:                         mfargs.append(fargs[-1])
216:                 api['externroutines'] = []
217:                 ar = applyrules(api, vrd)
218:                 ar['docs'] = []
219:                 ar['docshort'] = []
220:                 ret = dictappend(ret, ar)
221:                 cadd('\t{"%s",-1,{{-1}},0,NULL,(void *)f2py_rout_#modulename#_%s_%s,doc_f2py_rout_#modulename#_%s_%s},' %
222:                      (b['name'], m['name'], b['name'], m['name'], b['name']))
223:                 sargs.append('char *%s' % (b['name']))
224:                 sargsp.append('char *')
225:                 iadd('\tf2py_%s_def[i_f2py++].data = %s;' %
226:                      (m['name'], b['name']))
227:         cadd('\t{NULL}\n};\n')
228:         iadd('}')
229:         ihooks[0] = 'static void f2py_setup_%s(%s) {\n\tint i_f2py=0;%s' % (
230:             m['name'], ','.join(sargs), ihooks[0])
231:         if '_' in m['name']:
232:             F_FUNC = 'F_FUNC_US'
233:         else:
234:             F_FUNC = 'F_FUNC'
235:         iadd('extern void %s(f2pyinit%s,F2PYINIT%s)(void (*)(%s));'
236:              % (F_FUNC, m['name'], m['name'].upper(), ','.join(sargsp)))
237:         iadd('static void f2py_init_%s(void) {' % (m['name']))
238:         iadd('\t%s(f2pyinit%s,F2PYINIT%s)(f2py_setup_%s);'
239:              % (F_FUNC, m['name'], m['name'].upper(), m['name']))
240:         iadd('}\n')
241:         ret['f90modhooks'] = ret['f90modhooks'] + chooks + ihooks
242:         ret['initf90modhooks'] = ['\tPyDict_SetItemString(d, "%s", PyFortranObject_New(f2py_%s_def,f2py_init_%s));' % (
243:             m['name'], m['name'], m['name'])] + ret['initf90modhooks']
244:         fadd('')
245:         fadd('subroutine f2pyinit%s(f2pysetupfunc)' % (m['name']))
246:         if mfargs:
247:             for a in undo_rmbadname(mfargs):
248:                 fadd('use %s, only : %s' % (m['name'], a))
249:         if ifargs:
250:             fadd(' '.join(['interface'] + ifargs))
251:             fadd('end interface')
252:         fadd('external f2pysetupfunc')
253:         if efargs:
254:             for a in undo_rmbadname(efargs):
255:                 fadd('external %s' % (a))
256:         fadd('call f2pysetupfunc(%s)' % (','.join(undo_rmbadname(fargs))))
257:         fadd('end subroutine f2pyinit%s\n' % (m['name']))
258: 
259:         dadd('\n'.join(ret['latexdoc']).replace(
260:             r'\subsection{', r'\subsubsection{'))
261: 
262:         ret['latexdoc'] = []
263:         ret['docs'].append('"\t%s --- %s"' % (m['name'],
264:                                               ','.join(undo_rmbadname(modobjs))))
265: 
266:     ret['routine_defs'] = ''
267:     ret['doc'] = []
268:     ret['docshort'] = []
269:     ret['latexdoc'] = doc[0]
270:     if len(ret['docs']) <= 1:
271:         ret['docs'] = ''
272:     return ret, fhooks[0]
273: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_93169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, (-1)), 'str', '\n\nBuild F90 module support for f2py2e.\n\nCopyright 2000 Pearu Peterson all rights reserved,\nPearu Peterson <pearu@ioc.ee>\nPermission to use, modify, and distribute this software is given under the\nterms of the NumPy License.\n\nNO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.\n$Date: 2005/02/03 19:30:23 $\nPearu Peterson\n\n')

# Assigning a Subscript to a Name (line 18):

# Assigning a Subscript to a Name (line 18):

# Obtaining the type of the subscript
int_93170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 34), 'int')
int_93171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 37), 'int')
slice_93172 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 18, 14), int_93170, int_93171, None)
str_93173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 14), 'str', '$Revision: 1.27 $')
# Obtaining the member '__getitem__' of a type (line 18)
getitem___93174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 14), str_93173, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 18)
subscript_call_result_93175 = invoke(stypy.reporting.localization.Localization(__file__, 18, 14), getitem___93174, slice_93172)

# Assigning a type to the variable '__version__' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), '__version__', subscript_call_result_93175)

# Assigning a Str to a Name (line 20):

# Assigning a Str to a Name (line 20):
str_93176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 15), 'str', 'See `f2py -v`')
# Assigning a type to the variable 'f2py_version' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'f2py_version', str_93176)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'import numpy' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_93177 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy')

if (type(import_93177) is not StypyTypeError):

    if (import_93177 != 'pyd_module'):
        __import__(import_93177)
        sys_modules_93178 = sys.modules[import_93177]
        import_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'np', sys_modules_93178.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy', import_93177)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from numpy.f2py import capi_maps' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_93179 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.f2py')

if (type(import_93179) is not StypyTypeError):

    if (import_93179 != 'pyd_module'):
        __import__(import_93179)
        sys_modules_93180 = sys.modules[import_93179]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.f2py', sys_modules_93180.module_type_store, module_type_store, ['capi_maps'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_93180, sys_modules_93180.module_type_store, module_type_store)
    else:
        from numpy.f2py import capi_maps

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.f2py', None, module_type_store, ['capi_maps'], [capi_maps])

else:
    # Assigning a type to the variable 'numpy.f2py' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.f2py', import_93179)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from numpy.f2py import func2subr' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_93181 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.f2py')

if (type(import_93181) is not StypyTypeError):

    if (import_93181 != 'pyd_module'):
        __import__(import_93181)
        sys_modules_93182 = sys.modules[import_93181]
        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.f2py', sys_modules_93182.module_type_store, module_type_store, ['func2subr'])
        nest_module(stypy.reporting.localization.Localization(__file__, 25, 0), __file__, sys_modules_93182, sys_modules_93182.module_type_store, module_type_store)
    else:
        from numpy.f2py import func2subr

        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.f2py', None, module_type_store, ['func2subr'], [func2subr])

else:
    # Assigning a type to the variable 'numpy.f2py' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.f2py', import_93181)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from numpy.f2py.crackfortran import undo_rmbadname, undo_rmbadname1' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_93183 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.f2py.crackfortran')

if (type(import_93183) is not StypyTypeError):

    if (import_93183 != 'pyd_module'):
        __import__(import_93183)
        sys_modules_93184 = sys.modules[import_93183]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.f2py.crackfortran', sys_modules_93184.module_type_store, module_type_store, ['undo_rmbadname', 'undo_rmbadname1'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_93184, sys_modules_93184.module_type_store, module_type_store)
    else:
        from numpy.f2py.crackfortran import undo_rmbadname, undo_rmbadname1

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.f2py.crackfortran', None, module_type_store, ['undo_rmbadname', 'undo_rmbadname1'], [undo_rmbadname, undo_rmbadname1])

else:
    # Assigning a type to the variable 'numpy.f2py.crackfortran' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.f2py.crackfortran', import_93183)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 31, 0))

# 'from numpy.f2py.auxfuncs import ' statement (line 31)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_93185 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'numpy.f2py.auxfuncs')

if (type(import_93185) is not StypyTypeError):

    if (import_93185 != 'pyd_module'):
        __import__(import_93185)
        sys_modules_93186 = sys.modules[import_93185]
        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'numpy.f2py.auxfuncs', sys_modules_93186.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 31, 0), __file__, sys_modules_93186, sys_modules_93186.module_type_store, module_type_store)
    else:
        from numpy.f2py.auxfuncs import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'numpy.f2py.auxfuncs', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.f2py.auxfuncs' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'numpy.f2py.auxfuncs', import_93185)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')


# Assigning a Dict to a Name (line 33):

# Assigning a Dict to a Name (line 33):

# Obtaining an instance of the builtin type 'dict' (line 33)
dict_93187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 10), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 33)

# Assigning a type to the variable 'options' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'options', dict_93187)

@norecursion
def findf90modules(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'findf90modules'
    module_type_store = module_type_store.open_function_context('findf90modules', 36, 0, False)
    
    # Passed parameters checking function
    findf90modules.stypy_localization = localization
    findf90modules.stypy_type_of_self = None
    findf90modules.stypy_type_store = module_type_store
    findf90modules.stypy_function_name = 'findf90modules'
    findf90modules.stypy_param_names_list = ['m']
    findf90modules.stypy_varargs_param_name = None
    findf90modules.stypy_kwargs_param_name = None
    findf90modules.stypy_call_defaults = defaults
    findf90modules.stypy_call_varargs = varargs
    findf90modules.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'findf90modules', ['m'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'findf90modules', localization, ['m'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'findf90modules(...)' code ##################

    
    
    # Call to ismodule(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'm' (line 37)
    m_93189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 16), 'm', False)
    # Processing the call keyword arguments (line 37)
    kwargs_93190 = {}
    # Getting the type of 'ismodule' (line 37)
    ismodule_93188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 7), 'ismodule', False)
    # Calling ismodule(args, kwargs) (line 37)
    ismodule_call_result_93191 = invoke(stypy.reporting.localization.Localization(__file__, 37, 7), ismodule_93188, *[m_93189], **kwargs_93190)
    
    # Testing the type of an if condition (line 37)
    if_condition_93192 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 4), ismodule_call_result_93191)
    # Assigning a type to the variable 'if_condition_93192' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'if_condition_93192', if_condition_93192)
    # SSA begins for if statement (line 37)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'list' (line 38)
    list_93193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 38)
    # Adding element type (line 38)
    # Getting the type of 'm' (line 38)
    m_93194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 15), list_93193, m_93194)
    
    # Assigning a type to the variable 'stypy_return_type' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'stypy_return_type', list_93193)
    # SSA join for if statement (line 37)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to hasbody(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'm' (line 39)
    m_93196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 19), 'm', False)
    # Processing the call keyword arguments (line 39)
    kwargs_93197 = {}
    # Getting the type of 'hasbody' (line 39)
    hasbody_93195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 11), 'hasbody', False)
    # Calling hasbody(args, kwargs) (line 39)
    hasbody_call_result_93198 = invoke(stypy.reporting.localization.Localization(__file__, 39, 11), hasbody_93195, *[m_93196], **kwargs_93197)
    
    # Applying the 'not' unary operator (line 39)
    result_not__93199 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 7), 'not', hasbody_call_result_93198)
    
    # Testing the type of an if condition (line 39)
    if_condition_93200 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 4), result_not__93199)
    # Assigning a type to the variable 'if_condition_93200' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'if_condition_93200', if_condition_93200)
    # SSA begins for if statement (line 39)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'list' (line 40)
    list_93201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 40)
    
    # Assigning a type to the variable 'stypy_return_type' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'stypy_return_type', list_93201)
    # SSA join for if statement (line 39)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 41):
    
    # Assigning a List to a Name (line 41):
    
    # Obtaining an instance of the builtin type 'list' (line 41)
    list_93202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 41)
    
    # Assigning a type to the variable 'ret' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'ret', list_93202)
    
    
    # Obtaining the type of the subscript
    str_93203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 15), 'str', 'body')
    # Getting the type of 'm' (line 42)
    m_93204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'm')
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___93205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 13), m_93204, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_93206 = invoke(stypy.reporting.localization.Localization(__file__, 42, 13), getitem___93205, str_93203)
    
    # Testing the type of a for loop iterable (line 42)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 42, 4), subscript_call_result_93206)
    # Getting the type of the for loop variable (line 42)
    for_loop_var_93207 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 42, 4), subscript_call_result_93206)
    # Assigning a type to the variable 'b' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'b', for_loop_var_93207)
    # SSA begins for a for statement (line 42)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to ismodule(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'b' (line 43)
    b_93209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 20), 'b', False)
    # Processing the call keyword arguments (line 43)
    kwargs_93210 = {}
    # Getting the type of 'ismodule' (line 43)
    ismodule_93208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 11), 'ismodule', False)
    # Calling ismodule(args, kwargs) (line 43)
    ismodule_call_result_93211 = invoke(stypy.reporting.localization.Localization(__file__, 43, 11), ismodule_93208, *[b_93209], **kwargs_93210)
    
    # Testing the type of an if condition (line 43)
    if_condition_93212 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 8), ismodule_call_result_93211)
    # Assigning a type to the variable 'if_condition_93212' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'if_condition_93212', if_condition_93212)
    # SSA begins for if statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'b' (line 44)
    b_93215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 23), 'b', False)
    # Processing the call keyword arguments (line 44)
    kwargs_93216 = {}
    # Getting the type of 'ret' (line 44)
    ret_93213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'ret', False)
    # Obtaining the member 'append' of a type (line 44)
    append_93214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 12), ret_93213, 'append')
    # Calling append(args, kwargs) (line 44)
    append_call_result_93217 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), append_93214, *[b_93215], **kwargs_93216)
    
    # SSA branch for the else part of an if statement (line 43)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 46):
    
    # Assigning a BinOp to a Name (line 46):
    # Getting the type of 'ret' (line 46)
    ret_93218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 18), 'ret')
    
    # Call to findf90modules(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'b' (line 46)
    b_93220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 39), 'b', False)
    # Processing the call keyword arguments (line 46)
    kwargs_93221 = {}
    # Getting the type of 'findf90modules' (line 46)
    findf90modules_93219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 24), 'findf90modules', False)
    # Calling findf90modules(args, kwargs) (line 46)
    findf90modules_call_result_93222 = invoke(stypy.reporting.localization.Localization(__file__, 46, 24), findf90modules_93219, *[b_93220], **kwargs_93221)
    
    # Applying the binary operator '+' (line 46)
    result_add_93223 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 18), '+', ret_93218, findf90modules_call_result_93222)
    
    # Assigning a type to the variable 'ret' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'ret', result_add_93223)
    # SSA join for if statement (line 43)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ret' (line 47)
    ret_93224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'stypy_return_type', ret_93224)
    
    # ################# End of 'findf90modules(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'findf90modules' in the type store
    # Getting the type of 'stypy_return_type' (line 36)
    stypy_return_type_93225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_93225)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'findf90modules'
    return stypy_return_type_93225

# Assigning a type to the variable 'findf90modules' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'findf90modules', findf90modules)

# Assigning a BinOp to a Name (line 49):

# Assigning a BinOp to a Name (line 49):
str_93226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, (-1)), 'str', '      external f2pysetdata\n      logical ns\n      integer r,i\n      integer(%d) s(*)\n      ns = .FALSE.\n      if (allocated(d)) then\n         do i=1,r\n            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then\n               ns = .TRUE.\n            end if\n         end do\n         if (ns) then\n            deallocate(d)\n         end if\n      end if\n      if ((.not.allocated(d)).and.(s(1).ge.1)) then')

# Call to intp(...): (line 65)
# Processing the call keyword arguments (line 65)
kwargs_93229 = {}
# Getting the type of 'np' (line 65)
np_93227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 57), 'np', False)
# Obtaining the member 'intp' of a type (line 65)
intp_93228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 57), np_93227, 'intp')
# Calling intp(args, kwargs) (line 65)
intp_call_result_93230 = invoke(stypy.reporting.localization.Localization(__file__, 65, 57), intp_93228, *[], **kwargs_93229)

# Obtaining the member 'itemsize' of a type (line 65)
itemsize_93231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 57), intp_call_result_93230, 'itemsize')
# Applying the binary operator '%' (line 65)
result_mod_93232 = python_operator(stypy.reporting.localization.Localization(__file__, 65, (-1)), '%', str_93226, itemsize_93231)

# Assigning a type to the variable 'fgetdims1' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'fgetdims1', result_mod_93232)

# Assigning a Str to a Name (line 67):

# Assigning a Str to a Name (line 67):
str_93233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, (-1)), 'str', '      end if\n      if (allocated(d)) then\n         do i=1,r\n            s(i) = size(d,i)\n         end do\n      end if\n      flag = 1\n      call f2pysetdata(d,allocated(d))')
# Assigning a type to the variable 'fgetdims2' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'fgetdims2', str_93233)

# Assigning a Str to a Name (line 77):

# Assigning a Str to a Name (line 77):
str_93234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, (-1)), 'str', '      end if\n      if (allocated(d)) then\n         do i=1,r\n            s(i) = size(d,i)\n         end do\n         !s(r) must be equal to len(d(1))\n      end if\n      flag = 2\n      call f2pysetdata(d,allocated(d))')
# Assigning a type to the variable 'fgetdims2_sa' (line 77)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'fgetdims2_sa', str_93234)

@norecursion
def buildhooks(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'buildhooks'
    module_type_store = module_type_store.open_function_context('buildhooks', 89, 0, False)
    
    # Passed parameters checking function
    buildhooks.stypy_localization = localization
    buildhooks.stypy_type_of_self = None
    buildhooks.stypy_type_store = module_type_store
    buildhooks.stypy_function_name = 'buildhooks'
    buildhooks.stypy_param_names_list = ['pymod']
    buildhooks.stypy_varargs_param_name = None
    buildhooks.stypy_kwargs_param_name = None
    buildhooks.stypy_call_defaults = defaults
    buildhooks.stypy_call_varargs = varargs
    buildhooks.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'buildhooks', ['pymod'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'buildhooks', localization, ['pymod'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'buildhooks(...)' code ##################

    # Marking variables as global (line 90)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 90, 4), 'fgetdims1')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 90, 4), 'fgetdims2')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 91, 4))
    
    # 'from numpy.f2py import rules' statement (line 91)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
    import_93235 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 91, 4), 'numpy.f2py')

    if (type(import_93235) is not StypyTypeError):

        if (import_93235 != 'pyd_module'):
            __import__(import_93235)
            sys_modules_93236 = sys.modules[import_93235]
            import_from_module(stypy.reporting.localization.Localization(__file__, 91, 4), 'numpy.f2py', sys_modules_93236.module_type_store, module_type_store, ['rules'])
            nest_module(stypy.reporting.localization.Localization(__file__, 91, 4), __file__, sys_modules_93236, sys_modules_93236.module_type_store, module_type_store)
        else:
            from numpy.f2py import rules

            import_from_module(stypy.reporting.localization.Localization(__file__, 91, 4), 'numpy.f2py', None, module_type_store, ['rules'], [rules])

    else:
        # Assigning a type to the variable 'numpy.f2py' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'numpy.f2py', import_93235)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')
    
    
    # Assigning a Dict to a Name (line 92):
    
    # Assigning a Dict to a Name (line 92):
    
    # Obtaining an instance of the builtin type 'dict' (line 92)
    dict_93237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 10), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 92)
    # Adding element type (key, value) (line 92)
    str_93238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 11), 'str', 'f90modhooks')
    
    # Obtaining an instance of the builtin type 'list' (line 92)
    list_93239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 92)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 10), dict_93237, (str_93238, list_93239))
    # Adding element type (key, value) (line 92)
    str_93240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 30), 'str', 'initf90modhooks')
    
    # Obtaining an instance of the builtin type 'list' (line 92)
    list_93241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 49), 'list')
    # Adding type elements to the builtin type 'list' instance (line 92)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 10), dict_93237, (str_93240, list_93241))
    # Adding element type (key, value) (line 92)
    str_93242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 53), 'str', 'body')
    
    # Obtaining an instance of the builtin type 'list' (line 92)
    list_93243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 61), 'list')
    # Adding type elements to the builtin type 'list' instance (line 92)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 10), dict_93237, (str_93242, list_93243))
    # Adding element type (key, value) (line 92)
    str_93244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 11), 'str', 'need')
    
    # Obtaining an instance of the builtin type 'list' (line 93)
    list_93245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 93)
    # Adding element type (line 93)
    str_93246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 20), 'str', 'F_FUNC')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 19), list_93245, str_93246)
    # Adding element type (line 93)
    str_93247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 30), 'str', 'arrayobject.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 19), list_93245, str_93247)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 10), dict_93237, (str_93244, list_93245))
    # Adding element type (key, value) (line 92)
    str_93248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 11), 'str', 'separatorsfor')
    
    # Obtaining an instance of the builtin type 'dict' (line 94)
    dict_93249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 28), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 94)
    # Adding element type (key, value) (line 94)
    str_93250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 29), 'str', 'includes0')
    str_93251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 42), 'str', '\n')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 28), dict_93249, (str_93250, str_93251))
    # Adding element type (key, value) (line 94)
    str_93252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 48), 'str', 'includes')
    str_93253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 60), 'str', '\n')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 28), dict_93249, (str_93252, str_93253))
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 10), dict_93237, (str_93248, dict_93249))
    # Adding element type (key, value) (line 92)
    str_93254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 11), 'str', 'docs')
    
    # Obtaining an instance of the builtin type 'list' (line 95)
    list_93255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 95)
    # Adding element type (line 95)
    str_93256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 20), 'str', '"Fortran 90/95 modules:\\n"')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 19), list_93255, str_93256)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 10), dict_93237, (str_93254, list_93255))
    # Adding element type (key, value) (line 92)
    str_93257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 11), 'str', 'latexdoc')
    
    # Obtaining an instance of the builtin type 'list' (line 96)
    list_93258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 96)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 10), dict_93237, (str_93257, list_93258))
    
    # Assigning a type to the variable 'ret' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'ret', dict_93237)
    
    # Assigning a List to a Name (line 97):
    
    # Assigning a List to a Name (line 97):
    
    # Obtaining an instance of the builtin type 'list' (line 97)
    list_93259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 97)
    # Adding element type (line 97)
    str_93260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 14), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 13), list_93259, str_93260)
    
    # Assigning a type to the variable 'fhooks' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'fhooks', list_93259)

    @norecursion
    def fadd(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'fhooks' (line 99)
        fhooks_93261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 21), 'fhooks')
        defaults = [fhooks_93261]
        # Create a new context for function 'fadd'
        module_type_store = module_type_store.open_function_context('fadd', 99, 4, False)
        
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

        
        # Assigning a BinOp to a Subscript (line 100):
        
        # Assigning a BinOp to a Subscript (line 100):
        str_93262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 15), 'str', '%s\n      %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 100)
        tuple_93263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 100)
        # Adding element type (line 100)
        
        # Obtaining the type of the subscript
        int_93264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 35), 'int')
        # Getting the type of 's' (line 100)
        s_93265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 33), 's')
        # Obtaining the member '__getitem__' of a type (line 100)
        getitem___93266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 33), s_93265, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 100)
        subscript_call_result_93267 = invoke(stypy.reporting.localization.Localization(__file__, 100, 33), getitem___93266, int_93264)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 33), tuple_93263, subscript_call_result_93267)
        # Adding element type (line 100)
        # Getting the type of 'line' (line 100)
        line_93268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 39), 'line')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 33), tuple_93263, line_93268)
        
        # Applying the binary operator '%' (line 100)
        result_mod_93269 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 15), '%', str_93262, tuple_93263)
        
        # Getting the type of 's' (line 100)
        s_93270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 's')
        int_93271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 10), 'int')
        # Storing an element on a container (line 100)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 8), s_93270, (int_93271, result_mod_93269))
        
        # ################# End of 'fadd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fadd' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_93272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_93272)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fadd'
        return stypy_return_type_93272

    # Assigning a type to the variable 'fadd' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'fadd', fadd)
    
    # Assigning a List to a Name (line 101):
    
    # Assigning a List to a Name (line 101):
    
    # Obtaining an instance of the builtin type 'list' (line 101)
    list_93273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 101)
    # Adding element type (line 101)
    str_93274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 11), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 10), list_93273, str_93274)
    
    # Assigning a type to the variable 'doc' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'doc', list_93273)

    @norecursion
    def dadd(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'doc' (line 103)
        doc_93275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 21), 'doc')
        defaults = [doc_93275]
        # Create a new context for function 'dadd'
        module_type_store = module_type_store.open_function_context('dadd', 103, 4, False)
        
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

        
        # Assigning a BinOp to a Subscript (line 104):
        
        # Assigning a BinOp to a Subscript (line 104):
        str_93276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 15), 'str', '%s\n%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 104)
        tuple_93277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 104)
        # Adding element type (line 104)
        
        # Obtaining the type of the subscript
        int_93278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 29), 'int')
        # Getting the type of 's' (line 104)
        s_93279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 27), 's')
        # Obtaining the member '__getitem__' of a type (line 104)
        getitem___93280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 27), s_93279, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 104)
        subscript_call_result_93281 = invoke(stypy.reporting.localization.Localization(__file__, 104, 27), getitem___93280, int_93278)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 27), tuple_93277, subscript_call_result_93281)
        # Adding element type (line 104)
        # Getting the type of 'line' (line 104)
        line_93282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 33), 'line')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 27), tuple_93277, line_93282)
        
        # Applying the binary operator '%' (line 104)
        result_mod_93283 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 15), '%', str_93276, tuple_93277)
        
        # Getting the type of 's' (line 104)
        s_93284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 's')
        int_93285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 10), 'int')
        # Storing an element on a container (line 104)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 8), s_93284, (int_93285, result_mod_93283))
        
        # ################# End of 'dadd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dadd' in the type store
        # Getting the type of 'stypy_return_type' (line 103)
        stypy_return_type_93286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_93286)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dadd'
        return stypy_return_type_93286

    # Assigning a type to the variable 'dadd' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'dadd', dadd)
    
    
    # Call to findf90modules(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'pymod' (line 105)
    pymod_93288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 28), 'pymod', False)
    # Processing the call keyword arguments (line 105)
    kwargs_93289 = {}
    # Getting the type of 'findf90modules' (line 105)
    findf90modules_93287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 13), 'findf90modules', False)
    # Calling findf90modules(args, kwargs) (line 105)
    findf90modules_call_result_93290 = invoke(stypy.reporting.localization.Localization(__file__, 105, 13), findf90modules_93287, *[pymod_93288], **kwargs_93289)
    
    # Testing the type of a for loop iterable (line 105)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 105, 4), findf90modules_call_result_93290)
    # Getting the type of the for loop variable (line 105)
    for_loop_var_93291 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 105, 4), findf90modules_call_result_93290)
    # Assigning a type to the variable 'm' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'm', for_loop_var_93291)
    # SSA begins for a for statement (line 105)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Tuple to a Tuple (line 106):
    
    # Assigning a List to a Name (line 106):
    
    # Obtaining an instance of the builtin type 'list' (line 106)
    list_93292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 59), 'list')
    # Adding type elements to the builtin type 'list' instance (line 106)
    
    # Assigning a type to the variable 'tuple_assignment_93160' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_assignment_93160', list_93292)
    
    # Assigning a List to a Name (line 106):
    
    # Obtaining an instance of the builtin type 'list' (line 106)
    list_93293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 63), 'list')
    # Adding type elements to the builtin type 'list' instance (line 106)
    
    # Assigning a type to the variable 'tuple_assignment_93161' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_assignment_93161', list_93293)
    
    # Assigning a List to a Name (line 106):
    
    # Obtaining an instance of the builtin type 'list' (line 106)
    list_93294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 67), 'list')
    # Adding type elements to the builtin type 'list' instance (line 106)
    
    # Assigning a type to the variable 'tuple_assignment_93162' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_assignment_93162', list_93294)
    
    # Assigning a List to a Name (line 106):
    
    # Obtaining an instance of the builtin type 'list' (line 106)
    list_93295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 71), 'list')
    # Adding type elements to the builtin type 'list' instance (line 106)
    
    # Assigning a type to the variable 'tuple_assignment_93163' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_assignment_93163', list_93295)
    
    # Assigning a List to a Name (line 106):
    
    # Obtaining an instance of the builtin type 'list' (line 106)
    list_93296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 75), 'list')
    # Adding type elements to the builtin type 'list' instance (line 106)
    # Adding element type (line 106)
    
    # Obtaining the type of the subscript
    str_93297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 14), 'str', 'name')
    # Getting the type of 'm' (line 107)
    m_93298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'm')
    # Obtaining the member '__getitem__' of a type (line 107)
    getitem___93299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 12), m_93298, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 107)
    subscript_call_result_93300 = invoke(stypy.reporting.localization.Localization(__file__, 107, 12), getitem___93299, str_93297)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 75), list_93296, subscript_call_result_93300)
    
    # Assigning a type to the variable 'tuple_assignment_93164' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_assignment_93164', list_93296)
    
    # Assigning a List to a Name (line 106):
    
    # Obtaining an instance of the builtin type 'list' (line 107)
    list_93301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 107)
    
    # Assigning a type to the variable 'tuple_assignment_93165' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_assignment_93165', list_93301)
    
    # Assigning a Name to a Name (line 106):
    # Getting the type of 'tuple_assignment_93160' (line 106)
    tuple_assignment_93160_93302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_assignment_93160')
    # Assigning a type to the variable 'sargs' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'sargs', tuple_assignment_93160_93302)
    
    # Assigning a Name to a Name (line 106):
    # Getting the type of 'tuple_assignment_93161' (line 106)
    tuple_assignment_93161_93303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_assignment_93161')
    # Assigning a type to the variable 'fargs' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 15), 'fargs', tuple_assignment_93161_93303)
    
    # Assigning a Name to a Name (line 106):
    # Getting the type of 'tuple_assignment_93162' (line 106)
    tuple_assignment_93162_93304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_assignment_93162')
    # Assigning a type to the variable 'efargs' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 22), 'efargs', tuple_assignment_93162_93304)
    
    # Assigning a Name to a Name (line 106):
    # Getting the type of 'tuple_assignment_93163' (line 106)
    tuple_assignment_93163_93305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_assignment_93163')
    # Assigning a type to the variable 'modobjs' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 30), 'modobjs', tuple_assignment_93163_93305)
    
    # Assigning a Name to a Name (line 106):
    # Getting the type of 'tuple_assignment_93164' (line 106)
    tuple_assignment_93164_93306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_assignment_93164')
    # Assigning a type to the variable 'notvars' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 39), 'notvars', tuple_assignment_93164_93306)
    
    # Assigning a Name to a Name (line 106):
    # Getting the type of 'tuple_assignment_93165' (line 106)
    tuple_assignment_93165_93307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_assignment_93165')
    # Assigning a type to the variable 'onlyvars' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 48), 'onlyvars', tuple_assignment_93165_93307)
    
    # Assigning a List to a Name (line 108):
    
    # Assigning a List to a Name (line 108):
    
    # Obtaining an instance of the builtin type 'list' (line 108)
    list_93308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 108)
    
    # Assigning a type to the variable 'sargsp' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'sargsp', list_93308)
    
    # Assigning a List to a Name (line 109):
    
    # Assigning a List to a Name (line 109):
    
    # Obtaining an instance of the builtin type 'list' (line 109)
    list_93309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 109)
    
    # Assigning a type to the variable 'ifargs' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'ifargs', list_93309)
    
    # Assigning a List to a Name (line 110):
    
    # Assigning a List to a Name (line 110):
    
    # Obtaining an instance of the builtin type 'list' (line 110)
    list_93310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 110)
    
    # Assigning a type to the variable 'mfargs' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'mfargs', list_93310)
    
    
    # Call to hasbody(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'm' (line 111)
    m_93312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'm', False)
    # Processing the call keyword arguments (line 111)
    kwargs_93313 = {}
    # Getting the type of 'hasbody' (line 111)
    hasbody_93311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 11), 'hasbody', False)
    # Calling hasbody(args, kwargs) (line 111)
    hasbody_call_result_93314 = invoke(stypy.reporting.localization.Localization(__file__, 111, 11), hasbody_93311, *[m_93312], **kwargs_93313)
    
    # Testing the type of an if condition (line 111)
    if_condition_93315 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 8), hasbody_call_result_93314)
    # Assigning a type to the variable 'if_condition_93315' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'if_condition_93315', if_condition_93315)
    # SSA begins for if statement (line 111)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Obtaining the type of the subscript
    str_93316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 23), 'str', 'body')
    # Getting the type of 'm' (line 112)
    m_93317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 21), 'm')
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___93318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 21), m_93317, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_93319 = invoke(stypy.reporting.localization.Localization(__file__, 112, 21), getitem___93318, str_93316)
    
    # Testing the type of a for loop iterable (line 112)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 112, 12), subscript_call_result_93319)
    # Getting the type of the for loop variable (line 112)
    for_loop_var_93320 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 112, 12), subscript_call_result_93319)
    # Assigning a type to the variable 'b' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'b', for_loop_var_93320)
    # SSA begins for a for statement (line 112)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 113)
    # Processing the call arguments (line 113)
    
    # Obtaining the type of the subscript
    str_93323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 33), 'str', 'name')
    # Getting the type of 'b' (line 113)
    b_93324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 31), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 113)
    getitem___93325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 31), b_93324, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
    subscript_call_result_93326 = invoke(stypy.reporting.localization.Localization(__file__, 113, 31), getitem___93325, str_93323)
    
    # Processing the call keyword arguments (line 113)
    kwargs_93327 = {}
    # Getting the type of 'notvars' (line 113)
    notvars_93321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'notvars', False)
    # Obtaining the member 'append' of a type (line 113)
    append_93322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 16), notvars_93321, 'append')
    # Calling append(args, kwargs) (line 113)
    append_call_result_93328 = invoke(stypy.reporting.localization.Localization(__file__, 113, 16), append_93322, *[subscript_call_result_93326], **kwargs_93327)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 111)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to keys(...): (line 114)
    # Processing the call keyword arguments (line 114)
    kwargs_93334 = {}
    
    # Obtaining the type of the subscript
    str_93329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 19), 'str', 'vars')
    # Getting the type of 'm' (line 114)
    m_93330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 17), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 114)
    getitem___93331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 17), m_93330, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 114)
    subscript_call_result_93332 = invoke(stypy.reporting.localization.Localization(__file__, 114, 17), getitem___93331, str_93329)
    
    # Obtaining the member 'keys' of a type (line 114)
    keys_93333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 17), subscript_call_result_93332, 'keys')
    # Calling keys(args, kwargs) (line 114)
    keys_call_result_93335 = invoke(stypy.reporting.localization.Localization(__file__, 114, 17), keys_93333, *[], **kwargs_93334)
    
    # Testing the type of a for loop iterable (line 114)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 114, 8), keys_call_result_93335)
    # Getting the type of the for loop variable (line 114)
    for_loop_var_93336 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 114, 8), keys_call_result_93335)
    # Assigning a type to the variable 'n' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'n', for_loop_var_93336)
    # SSA begins for a for statement (line 114)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 115):
    
    # Assigning a Subscript to a Name (line 115):
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 115)
    n_93337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 28), 'n')
    
    # Obtaining the type of the subscript
    str_93338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 20), 'str', 'vars')
    # Getting the type of 'm' (line 115)
    m_93339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 18), 'm')
    # Obtaining the member '__getitem__' of a type (line 115)
    getitem___93340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 18), m_93339, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 115)
    subscript_call_result_93341 = invoke(stypy.reporting.localization.Localization(__file__, 115, 18), getitem___93340, str_93338)
    
    # Obtaining the member '__getitem__' of a type (line 115)
    getitem___93342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 18), subscript_call_result_93341, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 115)
    subscript_call_result_93343 = invoke(stypy.reporting.localization.Localization(__file__, 115, 18), getitem___93342, n_93337)
    
    # Assigning a type to the variable 'var' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'var', subscript_call_result_93343)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'n' (line 116)
    n_93344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'n')
    # Getting the type of 'notvars' (line 116)
    notvars_93345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 25), 'notvars')
    # Applying the binary operator 'notin' (line 116)
    result_contains_93346 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 16), 'notin', n_93344, notvars_93345)
    
    
    
    # Call to (...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'var' (line 116)
    var_93352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 74), 'var', False)
    # Processing the call keyword arguments (line 116)
    kwargs_93353 = {}
    
    # Call to l_or(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'isintent_hide' (line 116)
    isintent_hide_93348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 48), 'isintent_hide', False)
    # Getting the type of 'isprivate' (line 116)
    isprivate_93349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 63), 'isprivate', False)
    # Processing the call keyword arguments (line 116)
    kwargs_93350 = {}
    # Getting the type of 'l_or' (line 116)
    l_or_93347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 43), 'l_or', False)
    # Calling l_or(args, kwargs) (line 116)
    l_or_call_result_93351 = invoke(stypy.reporting.localization.Localization(__file__, 116, 43), l_or_93347, *[isintent_hide_93348, isprivate_93349], **kwargs_93350)
    
    # Calling (args, kwargs) (line 116)
    _call_result_93354 = invoke(stypy.reporting.localization.Localization(__file__, 116, 43), l_or_call_result_93351, *[var_93352], **kwargs_93353)
    
    # Applying the 'not' unary operator (line 116)
    result_not__93355 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 39), 'not', _call_result_93354)
    
    # Applying the binary operator 'and' (line 116)
    result_and_keyword_93356 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 15), 'and', result_contains_93346, result_not__93355)
    
    # Testing the type of an if condition (line 116)
    if_condition_93357 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 12), result_and_keyword_93356)
    # Assigning a type to the variable 'if_condition_93357' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'if_condition_93357', if_condition_93357)
    # SSA begins for if statement (line 116)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'n' (line 117)
    n_93360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 32), 'n', False)
    # Processing the call keyword arguments (line 117)
    kwargs_93361 = {}
    # Getting the type of 'onlyvars' (line 117)
    onlyvars_93358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'onlyvars', False)
    # Obtaining the member 'append' of a type (line 117)
    append_93359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 16), onlyvars_93358, 'append')
    # Calling append(args, kwargs) (line 117)
    append_call_result_93362 = invoke(stypy.reporting.localization.Localization(__file__, 117, 16), append_93359, *[n_93360], **kwargs_93361)
    
    
    # Call to append(...): (line 118)
    # Processing the call arguments (line 118)
    # Getting the type of 'n' (line 118)
    n_93365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 30), 'n', False)
    # Processing the call keyword arguments (line 118)
    kwargs_93366 = {}
    # Getting the type of 'mfargs' (line 118)
    mfargs_93363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'mfargs', False)
    # Obtaining the member 'append' of a type (line 118)
    append_93364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 16), mfargs_93363, 'append')
    # Calling append(args, kwargs) (line 118)
    append_call_result_93367 = invoke(stypy.reporting.localization.Localization(__file__, 118, 16), append_93364, *[n_93365], **kwargs_93366)
    
    # SSA join for if statement (line 116)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to outmess(...): (line 119)
    # Processing the call arguments (line 119)
    str_93369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 16), 'str', '\t\tConstructing F90 module support for "%s"...\n')
    
    # Obtaining the type of the subscript
    str_93370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 19), 'str', 'name')
    # Getting the type of 'm' (line 120)
    m_93371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 17), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 120)
    getitem___93372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 17), m_93371, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 120)
    subscript_call_result_93373 = invoke(stypy.reporting.localization.Localization(__file__, 120, 17), getitem___93372, str_93370)
    
    # Applying the binary operator '%' (line 119)
    result_mod_93374 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 16), '%', str_93369, subscript_call_result_93373)
    
    # Processing the call keyword arguments (line 119)
    kwargs_93375 = {}
    # Getting the type of 'outmess' (line 119)
    outmess_93368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'outmess', False)
    # Calling outmess(args, kwargs) (line 119)
    outmess_call_result_93376 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), outmess_93368, *[result_mod_93374], **kwargs_93375)
    
    
    # Getting the type of 'onlyvars' (line 121)
    onlyvars_93377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'onlyvars')
    # Testing the type of an if condition (line 121)
    if_condition_93378 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 8), onlyvars_93377)
    # Assigning a type to the variable 'if_condition_93378' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'if_condition_93378', if_condition_93378)
    # SSA begins for if statement (line 121)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to outmess(...): (line 122)
    # Processing the call arguments (line 122)
    str_93380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 20), 'str', '\t\t  Variables: %s\n')
    
    # Call to join(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'onlyvars' (line 122)
    onlyvars_93383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 56), 'onlyvars', False)
    # Processing the call keyword arguments (line 122)
    kwargs_93384 = {}
    str_93381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 47), 'str', ' ')
    # Obtaining the member 'join' of a type (line 122)
    join_93382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 47), str_93381, 'join')
    # Calling join(args, kwargs) (line 122)
    join_call_result_93385 = invoke(stypy.reporting.localization.Localization(__file__, 122, 47), join_93382, *[onlyvars_93383], **kwargs_93384)
    
    # Applying the binary operator '%' (line 122)
    result_mod_93386 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 20), '%', str_93380, join_call_result_93385)
    
    # Processing the call keyword arguments (line 122)
    kwargs_93387 = {}
    # Getting the type of 'outmess' (line 122)
    outmess_93379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'outmess', False)
    # Calling outmess(args, kwargs) (line 122)
    outmess_call_result_93388 = invoke(stypy.reporting.localization.Localization(__file__, 122, 12), outmess_93379, *[result_mod_93386], **kwargs_93387)
    
    # SSA join for if statement (line 121)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 123):
    
    # Assigning a List to a Name (line 123):
    
    # Obtaining an instance of the builtin type 'list' (line 123)
    list_93389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 123)
    # Adding element type (line 123)
    str_93390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 18), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 17), list_93389, str_93390)
    
    # Assigning a type to the variable 'chooks' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'chooks', list_93389)

    @norecursion
    def cadd(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'chooks' (line 125)
        chooks_93391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 25), 'chooks')
        defaults = [chooks_93391]
        # Create a new context for function 'cadd'
        module_type_store = module_type_store.open_function_context('cadd', 125, 8, False)
        
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

        
        # Assigning a BinOp to a Subscript (line 126):
        
        # Assigning a BinOp to a Subscript (line 126):
        str_93392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 19), 'str', '%s\n%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 126)
        tuple_93393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 126)
        # Adding element type (line 126)
        
        # Obtaining the type of the subscript
        int_93394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 33), 'int')
        # Getting the type of 's' (line 126)
        s_93395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 31), 's')
        # Obtaining the member '__getitem__' of a type (line 126)
        getitem___93396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 31), s_93395, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 126)
        subscript_call_result_93397 = invoke(stypy.reporting.localization.Localization(__file__, 126, 31), getitem___93396, int_93394)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 31), tuple_93393, subscript_call_result_93397)
        # Adding element type (line 126)
        # Getting the type of 'line' (line 126)
        line_93398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 37), 'line')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 31), tuple_93393, line_93398)
        
        # Applying the binary operator '%' (line 126)
        result_mod_93399 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 19), '%', str_93392, tuple_93393)
        
        # Getting the type of 's' (line 126)
        s_93400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 's')
        int_93401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 14), 'int')
        # Storing an element on a container (line 126)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 12), s_93400, (int_93401, result_mod_93399))
        
        # ################# End of 'cadd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cadd' in the type store
        # Getting the type of 'stypy_return_type' (line 125)
        stypy_return_type_93402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_93402)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cadd'
        return stypy_return_type_93402

    # Assigning a type to the variable 'cadd' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'cadd', cadd)
    
    # Assigning a List to a Name (line 127):
    
    # Assigning a List to a Name (line 127):
    
    # Obtaining an instance of the builtin type 'list' (line 127)
    list_93403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 127)
    # Adding element type (line 127)
    str_93404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 18), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 17), list_93403, str_93404)
    
    # Assigning a type to the variable 'ihooks' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'ihooks', list_93403)

    @norecursion
    def iadd(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'ihooks' (line 129)
        ihooks_93405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 25), 'ihooks')
        defaults = [ihooks_93405]
        # Create a new context for function 'iadd'
        module_type_store = module_type_store.open_function_context('iadd', 129, 8, False)
        
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

        
        # Assigning a BinOp to a Subscript (line 130):
        
        # Assigning a BinOp to a Subscript (line 130):
        str_93406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 19), 'str', '%s\n%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 130)
        tuple_93407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 130)
        # Adding element type (line 130)
        
        # Obtaining the type of the subscript
        int_93408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 33), 'int')
        # Getting the type of 's' (line 130)
        s_93409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 31), 's')
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___93410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 31), s_93409, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
        subscript_call_result_93411 = invoke(stypy.reporting.localization.Localization(__file__, 130, 31), getitem___93410, int_93408)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 31), tuple_93407, subscript_call_result_93411)
        # Adding element type (line 130)
        # Getting the type of 'line' (line 130)
        line_93412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 37), 'line')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 31), tuple_93407, line_93412)
        
        # Applying the binary operator '%' (line 130)
        result_mod_93413 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 19), '%', str_93406, tuple_93407)
        
        # Getting the type of 's' (line 130)
        s_93414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 's')
        int_93415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 14), 'int')
        # Storing an element on a container (line 130)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 12), s_93414, (int_93415, result_mod_93413))
        
        # ################# End of 'iadd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'iadd' in the type store
        # Getting the type of 'stypy_return_type' (line 129)
        stypy_return_type_93416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_93416)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'iadd'
        return stypy_return_type_93416

    # Assigning a type to the variable 'iadd' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'iadd', iadd)
    
    # Assigning a Call to a Name (line 132):
    
    # Assigning a Call to a Name (line 132):
    
    # Call to modsign2map(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'm' (line 132)
    m_93419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 36), 'm', False)
    # Processing the call keyword arguments (line 132)
    kwargs_93420 = {}
    # Getting the type of 'capi_maps' (line 132)
    capi_maps_93417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 14), 'capi_maps', False)
    # Obtaining the member 'modsign2map' of a type (line 132)
    modsign2map_93418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 14), capi_maps_93417, 'modsign2map')
    # Calling modsign2map(args, kwargs) (line 132)
    modsign2map_call_result_93421 = invoke(stypy.reporting.localization.Localization(__file__, 132, 14), modsign2map_93418, *[m_93419], **kwargs_93420)
    
    # Assigning a type to the variable 'vrd' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'vrd', modsign2map_call_result_93421)
    
    # Call to cadd(...): (line 133)
    # Processing the call arguments (line 133)
    str_93423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 13), 'str', 'static FortranDataDef f2py_%s_def[] = {')
    
    # Obtaining the type of the subscript
    str_93424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 60), 'str', 'name')
    # Getting the type of 'm' (line 133)
    m_93425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 58), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 133)
    getitem___93426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 58), m_93425, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 133)
    subscript_call_result_93427 = invoke(stypy.reporting.localization.Localization(__file__, 133, 58), getitem___93426, str_93424)
    
    # Applying the binary operator '%' (line 133)
    result_mod_93428 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 13), '%', str_93423, subscript_call_result_93427)
    
    # Processing the call keyword arguments (line 133)
    kwargs_93429 = {}
    # Getting the type of 'cadd' (line 133)
    cadd_93422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'cadd', False)
    # Calling cadd(args, kwargs) (line 133)
    cadd_call_result_93430 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), cadd_93422, *[result_mod_93428], **kwargs_93429)
    
    
    # Call to dadd(...): (line 134)
    # Processing the call arguments (line 134)
    str_93432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 13), 'str', '\\subsection{Fortran 90/95 module \\texttt{%s}}\n')
    
    # Obtaining the type of the subscript
    str_93433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 70), 'str', 'name')
    # Getting the type of 'm' (line 134)
    m_93434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 68), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 134)
    getitem___93435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 68), m_93434, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 134)
    subscript_call_result_93436 = invoke(stypy.reporting.localization.Localization(__file__, 134, 68), getitem___93435, str_93433)
    
    # Applying the binary operator '%' (line 134)
    result_mod_93437 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 13), '%', str_93432, subscript_call_result_93436)
    
    # Processing the call keyword arguments (line 134)
    kwargs_93438 = {}
    # Getting the type of 'dadd' (line 134)
    dadd_93431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'dadd', False)
    # Calling dadd(args, kwargs) (line 134)
    dadd_call_result_93439 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), dadd_93431, *[result_mod_93437], **kwargs_93438)
    
    
    
    # Call to hasnote(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'm' (line 135)
    m_93441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 19), 'm', False)
    # Processing the call keyword arguments (line 135)
    kwargs_93442 = {}
    # Getting the type of 'hasnote' (line 135)
    hasnote_93440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 11), 'hasnote', False)
    # Calling hasnote(args, kwargs) (line 135)
    hasnote_call_result_93443 = invoke(stypy.reporting.localization.Localization(__file__, 135, 11), hasnote_93440, *[m_93441], **kwargs_93442)
    
    # Testing the type of an if condition (line 135)
    if_condition_93444 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 8), hasnote_call_result_93443)
    # Assigning a type to the variable 'if_condition_93444' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'if_condition_93444', if_condition_93444)
    # SSA begins for if statement (line 135)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 136):
    
    # Assigning a Subscript to a Name (line 136):
    
    # Obtaining the type of the subscript
    str_93445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 21), 'str', 'note')
    # Getting the type of 'm' (line 136)
    m_93446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 19), 'm')
    # Obtaining the member '__getitem__' of a type (line 136)
    getitem___93447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 19), m_93446, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 136)
    subscript_call_result_93448 = invoke(stypy.reporting.localization.Localization(__file__, 136, 19), getitem___93447, str_93445)
    
    # Assigning a type to the variable 'note' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'note', subscript_call_result_93448)
    
    # Type idiom detected: calculating its left and rigth part (line 137)
    # Getting the type of 'list' (line 137)
    list_93449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 32), 'list')
    # Getting the type of 'note' (line 137)
    note_93450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 26), 'note')
    
    (may_be_93451, more_types_in_union_93452) = may_be_subtype(list_93449, note_93450)

    if may_be_93451:

        if more_types_in_union_93452:
            # Runtime conditional SSA (line 137)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'note' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'note', remove_not_subtype_from_union(note_93450, list))
        
        # Assigning a Call to a Name (line 138):
        
        # Assigning a Call to a Name (line 138):
        
        # Call to join(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'note' (line 138)
        note_93455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 33), 'note', False)
        # Processing the call keyword arguments (line 138)
        kwargs_93456 = {}
        str_93453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 23), 'str', '\n')
        # Obtaining the member 'join' of a type (line 138)
        join_93454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 23), str_93453, 'join')
        # Calling join(args, kwargs) (line 138)
        join_call_result_93457 = invoke(stypy.reporting.localization.Localization(__file__, 138, 23), join_93454, *[note_93455], **kwargs_93456)
        
        # Assigning a type to the variable 'note' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'note', join_call_result_93457)

        if more_types_in_union_93452:
            # SSA join for if statement (line 137)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to dadd(...): (line 139)
    # Processing the call arguments (line 139)
    # Getting the type of 'note' (line 139)
    note_93459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 17), 'note', False)
    # Processing the call keyword arguments (line 139)
    kwargs_93460 = {}
    # Getting the type of 'dadd' (line 139)
    dadd_93458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'dadd', False)
    # Calling dadd(args, kwargs) (line 139)
    dadd_call_result_93461 = invoke(stypy.reporting.localization.Localization(__file__, 139, 12), dadd_93458, *[note_93459], **kwargs_93460)
    
    # SSA join for if statement (line 135)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'onlyvars' (line 140)
    onlyvars_93462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 11), 'onlyvars')
    # Testing the type of an if condition (line 140)
    if_condition_93463 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 8), onlyvars_93462)
    # Assigning a type to the variable 'if_condition_93463' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'if_condition_93463', if_condition_93463)
    # SSA begins for if statement (line 140)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to dadd(...): (line 141)
    # Processing the call arguments (line 141)
    str_93465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 17), 'str', '\\begin{description}')
    # Processing the call keyword arguments (line 141)
    kwargs_93466 = {}
    # Getting the type of 'dadd' (line 141)
    dadd_93464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'dadd', False)
    # Calling dadd(args, kwargs) (line 141)
    dadd_call_result_93467 = invoke(stypy.reporting.localization.Localization(__file__, 141, 12), dadd_93464, *[str_93465], **kwargs_93466)
    
    # SSA join for if statement (line 140)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'onlyvars' (line 142)
    onlyvars_93468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 17), 'onlyvars')
    # Testing the type of a for loop iterable (line 142)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 142, 8), onlyvars_93468)
    # Getting the type of the for loop variable (line 142)
    for_loop_var_93469 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 142, 8), onlyvars_93468)
    # Assigning a type to the variable 'n' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'n', for_loop_var_93469)
    # SSA begins for a for statement (line 142)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 143):
    
    # Assigning a Subscript to a Name (line 143):
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 143)
    n_93470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 28), 'n')
    
    # Obtaining the type of the subscript
    str_93471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 20), 'str', 'vars')
    # Getting the type of 'm' (line 143)
    m_93472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 18), 'm')
    # Obtaining the member '__getitem__' of a type (line 143)
    getitem___93473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 18), m_93472, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 143)
    subscript_call_result_93474 = invoke(stypy.reporting.localization.Localization(__file__, 143, 18), getitem___93473, str_93471)
    
    # Obtaining the member '__getitem__' of a type (line 143)
    getitem___93475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 18), subscript_call_result_93474, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 143)
    subscript_call_result_93476 = invoke(stypy.reporting.localization.Localization(__file__, 143, 18), getitem___93475, n_93470)
    
    # Assigning a type to the variable 'var' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'var', subscript_call_result_93476)
    
    # Call to append(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 'n' (line 144)
    n_93479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 27), 'n', False)
    # Processing the call keyword arguments (line 144)
    kwargs_93480 = {}
    # Getting the type of 'modobjs' (line 144)
    modobjs_93477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'modobjs', False)
    # Obtaining the member 'append' of a type (line 144)
    append_93478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 12), modobjs_93477, 'append')
    # Calling append(args, kwargs) (line 144)
    append_call_result_93481 = invoke(stypy.reporting.localization.Localization(__file__, 144, 12), append_93478, *[n_93479], **kwargs_93480)
    
    
    # Assigning a Call to a Name (line 145):
    
    # Assigning a Call to a Name (line 145):
    
    # Call to getctype(...): (line 145)
    # Processing the call arguments (line 145)
    # Getting the type of 'var' (line 145)
    var_93484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 36), 'var', False)
    # Processing the call keyword arguments (line 145)
    kwargs_93485 = {}
    # Getting the type of 'capi_maps' (line 145)
    capi_maps_93482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 17), 'capi_maps', False)
    # Obtaining the member 'getctype' of a type (line 145)
    getctype_93483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 17), capi_maps_93482, 'getctype')
    # Calling getctype(args, kwargs) (line 145)
    getctype_call_result_93486 = invoke(stypy.reporting.localization.Localization(__file__, 145, 17), getctype_93483, *[var_93484], **kwargs_93485)
    
    # Assigning a type to the variable 'ct' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'ct', getctype_call_result_93486)
    
    # Assigning a Subscript to a Name (line 146):
    
    # Assigning a Subscript to a Name (line 146):
    
    # Obtaining the type of the subscript
    # Getting the type of 'ct' (line 146)
    ct_93487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 38), 'ct')
    # Getting the type of 'capi_maps' (line 146)
    capi_maps_93488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 17), 'capi_maps')
    # Obtaining the member 'c2capi_map' of a type (line 146)
    c2capi_map_93489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 17), capi_maps_93488, 'c2capi_map')
    # Obtaining the member '__getitem__' of a type (line 146)
    getitem___93490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 17), c2capi_map_93489, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 146)
    subscript_call_result_93491 = invoke(stypy.reporting.localization.Localization(__file__, 146, 17), getitem___93490, ct_93487)
    
    # Assigning a type to the variable 'at' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'at', subscript_call_result_93491)
    
    # Assigning a Call to a Name (line 147):
    
    # Assigning a Call to a Name (line 147):
    
    # Call to getarrdims(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'n' (line 147)
    n_93494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 38), 'n', False)
    # Getting the type of 'var' (line 147)
    var_93495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 41), 'var', False)
    # Processing the call keyword arguments (line 147)
    kwargs_93496 = {}
    # Getting the type of 'capi_maps' (line 147)
    capi_maps_93492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 17), 'capi_maps', False)
    # Obtaining the member 'getarrdims' of a type (line 147)
    getarrdims_93493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 17), capi_maps_93492, 'getarrdims')
    # Calling getarrdims(args, kwargs) (line 147)
    getarrdims_call_result_93497 = invoke(stypy.reporting.localization.Localization(__file__, 147, 17), getarrdims_93493, *[n_93494, var_93495], **kwargs_93496)
    
    # Assigning a type to the variable 'dm' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'dm', getarrdims_call_result_93497)
    
    # Assigning a Call to a Name (line 148):
    
    # Assigning a Call to a Name (line 148):
    
    # Call to strip(...): (line 148)
    # Processing the call keyword arguments (line 148)
    kwargs_93508 = {}
    
    # Call to replace(...): (line 148)
    # Processing the call arguments (line 148)
    str_93503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 37), 'str', '*')
    str_93504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 42), 'str', '-1')
    # Processing the call keyword arguments (line 148)
    kwargs_93505 = {}
    
    # Obtaining the type of the subscript
    str_93498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 21), 'str', 'dims')
    # Getting the type of 'dm' (line 148)
    dm_93499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 18), 'dm', False)
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___93500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 18), dm_93499, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_93501 = invoke(stypy.reporting.localization.Localization(__file__, 148, 18), getitem___93500, str_93498)
    
    # Obtaining the member 'replace' of a type (line 148)
    replace_93502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 18), subscript_call_result_93501, 'replace')
    # Calling replace(args, kwargs) (line 148)
    replace_call_result_93506 = invoke(stypy.reporting.localization.Localization(__file__, 148, 18), replace_93502, *[str_93503, str_93504], **kwargs_93505)
    
    # Obtaining the member 'strip' of a type (line 148)
    strip_93507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 18), replace_call_result_93506, 'strip')
    # Calling strip(args, kwargs) (line 148)
    strip_call_result_93509 = invoke(stypy.reporting.localization.Localization(__file__, 148, 18), strip_93507, *[], **kwargs_93508)
    
    # Assigning a type to the variable 'dms' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'dms', strip_call_result_93509)
    
    # Assigning a Call to a Name (line 149):
    
    # Assigning a Call to a Name (line 149):
    
    # Call to strip(...): (line 149)
    # Processing the call keyword arguments (line 149)
    kwargs_93517 = {}
    
    # Call to replace(...): (line 149)
    # Processing the call arguments (line 149)
    str_93512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 30), 'str', ':')
    str_93513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 35), 'str', '-1')
    # Processing the call keyword arguments (line 149)
    kwargs_93514 = {}
    # Getting the type of 'dms' (line 149)
    dms_93510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 18), 'dms', False)
    # Obtaining the member 'replace' of a type (line 149)
    replace_93511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 18), dms_93510, 'replace')
    # Calling replace(args, kwargs) (line 149)
    replace_call_result_93515 = invoke(stypy.reporting.localization.Localization(__file__, 149, 18), replace_93511, *[str_93512, str_93513], **kwargs_93514)
    
    # Obtaining the member 'strip' of a type (line 149)
    strip_93516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 18), replace_call_result_93515, 'strip')
    # Calling strip(args, kwargs) (line 149)
    strip_call_result_93518 = invoke(stypy.reporting.localization.Localization(__file__, 149, 18), strip_93516, *[], **kwargs_93517)
    
    # Assigning a type to the variable 'dms' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'dms', strip_call_result_93518)
    
    
    # Getting the type of 'dms' (line 150)
    dms_93519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 19), 'dms')
    # Applying the 'not' unary operator (line 150)
    result_not__93520 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 15), 'not', dms_93519)
    
    # Testing the type of an if condition (line 150)
    if_condition_93521 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 12), result_not__93520)
    # Assigning a type to the variable 'if_condition_93521' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'if_condition_93521', if_condition_93521)
    # SSA begins for if statement (line 150)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 151):
    
    # Assigning a Str to a Name (line 151):
    str_93522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 22), 'str', '-1')
    # Assigning a type to the variable 'dms' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'dms', str_93522)
    # SSA join for if statement (line 150)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 152):
    
    # Assigning a Name to a Name (line 152):
    # Getting the type of 'fgetdims2' (line 152)
    fgetdims2_93523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 28), 'fgetdims2')
    # Assigning a type to the variable 'use_fgetdims2' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'use_fgetdims2', fgetdims2_93523)
    
    
    # Call to isstringarray(...): (line 153)
    # Processing the call arguments (line 153)
    # Getting the type of 'var' (line 153)
    var_93525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 29), 'var', False)
    # Processing the call keyword arguments (line 153)
    kwargs_93526 = {}
    # Getting the type of 'isstringarray' (line 153)
    isstringarray_93524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 15), 'isstringarray', False)
    # Calling isstringarray(args, kwargs) (line 153)
    isstringarray_call_result_93527 = invoke(stypy.reporting.localization.Localization(__file__, 153, 15), isstringarray_93524, *[var_93525], **kwargs_93526)
    
    # Testing the type of an if condition (line 153)
    if_condition_93528 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 12), isstringarray_call_result_93527)
    # Assigning a type to the variable 'if_condition_93528' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'if_condition_93528', if_condition_93528)
    # SSA begins for if statement (line 153)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    str_93529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 19), 'str', 'charselector')
    # Getting the type of 'var' (line 154)
    var_93530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 37), 'var')
    # Applying the binary operator 'in' (line 154)
    result_contains_93531 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 19), 'in', str_93529, var_93530)
    
    
    str_93532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 45), 'str', 'len')
    
    # Obtaining the type of the subscript
    str_93533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 58), 'str', 'charselector')
    # Getting the type of 'var' (line 154)
    var_93534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 54), 'var')
    # Obtaining the member '__getitem__' of a type (line 154)
    getitem___93535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 54), var_93534, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 154)
    subscript_call_result_93536 = invoke(stypy.reporting.localization.Localization(__file__, 154, 54), getitem___93535, str_93533)
    
    # Applying the binary operator 'in' (line 154)
    result_contains_93537 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 45), 'in', str_93532, subscript_call_result_93536)
    
    # Applying the binary operator 'and' (line 154)
    result_and_keyword_93538 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 19), 'and', result_contains_93531, result_contains_93537)
    
    # Testing the type of an if condition (line 154)
    if_condition_93539 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 16), result_and_keyword_93538)
    # Assigning a type to the variable 'if_condition_93539' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'if_condition_93539', if_condition_93539)
    # SSA begins for if statement (line 154)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to cadd(...): (line 155)
    # Processing the call arguments (line 155)
    str_93541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 25), 'str', '\t{"%s",%s,{{%s,%s}},%s},')
    
    # Obtaining an instance of the builtin type 'tuple' (line 156)
    tuple_93542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 156)
    # Adding element type (line 156)
    
    # Call to undo_rmbadname1(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'n' (line 156)
    n_93544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 44), 'n', False)
    # Processing the call keyword arguments (line 156)
    kwargs_93545 = {}
    # Getting the type of 'undo_rmbadname1' (line 156)
    undo_rmbadname1_93543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 28), 'undo_rmbadname1', False)
    # Calling undo_rmbadname1(args, kwargs) (line 156)
    undo_rmbadname1_call_result_93546 = invoke(stypy.reporting.localization.Localization(__file__, 156, 28), undo_rmbadname1_93543, *[n_93544], **kwargs_93545)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 28), tuple_93542, undo_rmbadname1_call_result_93546)
    # Adding element type (line 156)
    
    # Obtaining the type of the subscript
    str_93547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 51), 'str', 'rank')
    # Getting the type of 'dm' (line 156)
    dm_93548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 48), 'dm', False)
    # Obtaining the member '__getitem__' of a type (line 156)
    getitem___93549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 48), dm_93548, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 156)
    subscript_call_result_93550 = invoke(stypy.reporting.localization.Localization(__file__, 156, 48), getitem___93549, str_93547)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 28), tuple_93542, subscript_call_result_93550)
    # Adding element type (line 156)
    # Getting the type of 'dms' (line 156)
    dms_93551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 60), 'dms', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 28), tuple_93542, dms_93551)
    # Adding element type (line 156)
    
    # Obtaining the type of the subscript
    str_93552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 85), 'str', 'len')
    
    # Obtaining the type of the subscript
    str_93553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 69), 'str', 'charselector')
    # Getting the type of 'var' (line 156)
    var_93554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 65), 'var', False)
    # Obtaining the member '__getitem__' of a type (line 156)
    getitem___93555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 65), var_93554, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 156)
    subscript_call_result_93556 = invoke(stypy.reporting.localization.Localization(__file__, 156, 65), getitem___93555, str_93553)
    
    # Obtaining the member '__getitem__' of a type (line 156)
    getitem___93557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 65), subscript_call_result_93556, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 156)
    subscript_call_result_93558 = invoke(stypy.reporting.localization.Localization(__file__, 156, 65), getitem___93557, str_93552)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 28), tuple_93542, subscript_call_result_93558)
    # Adding element type (line 156)
    # Getting the type of 'at' (line 156)
    at_93559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 93), 'at', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 28), tuple_93542, at_93559)
    
    # Applying the binary operator '%' (line 155)
    result_mod_93560 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 25), '%', str_93541, tuple_93542)
    
    # Processing the call keyword arguments (line 155)
    kwargs_93561 = {}
    # Getting the type of 'cadd' (line 155)
    cadd_93540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 20), 'cadd', False)
    # Calling cadd(args, kwargs) (line 155)
    cadd_call_result_93562 = invoke(stypy.reporting.localization.Localization(__file__, 155, 20), cadd_93540, *[result_mod_93560], **kwargs_93561)
    
    
    # Assigning a Name to a Name (line 157):
    
    # Assigning a Name to a Name (line 157):
    # Getting the type of 'fgetdims2_sa' (line 157)
    fgetdims2_sa_93563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 36), 'fgetdims2_sa')
    # Assigning a type to the variable 'use_fgetdims2' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'use_fgetdims2', fgetdims2_sa_93563)
    # SSA branch for the else part of an if statement (line 154)
    module_type_store.open_ssa_branch('else')
    
    # Call to cadd(...): (line 159)
    # Processing the call arguments (line 159)
    str_93565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 25), 'str', '\t{"%s",%s,{{%s}},%s},')
    
    # Obtaining an instance of the builtin type 'tuple' (line 160)
    tuple_93566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 160)
    # Adding element type (line 160)
    
    # Call to undo_rmbadname1(...): (line 160)
    # Processing the call arguments (line 160)
    # Getting the type of 'n' (line 160)
    n_93568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 42), 'n', False)
    # Processing the call keyword arguments (line 160)
    kwargs_93569 = {}
    # Getting the type of 'undo_rmbadname1' (line 160)
    undo_rmbadname1_93567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 26), 'undo_rmbadname1', False)
    # Calling undo_rmbadname1(args, kwargs) (line 160)
    undo_rmbadname1_call_result_93570 = invoke(stypy.reporting.localization.Localization(__file__, 160, 26), undo_rmbadname1_93567, *[n_93568], **kwargs_93569)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 26), tuple_93566, undo_rmbadname1_call_result_93570)
    # Adding element type (line 160)
    
    # Obtaining the type of the subscript
    str_93571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 49), 'str', 'rank')
    # Getting the type of 'dm' (line 160)
    dm_93572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 46), 'dm', False)
    # Obtaining the member '__getitem__' of a type (line 160)
    getitem___93573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 46), dm_93572, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 160)
    subscript_call_result_93574 = invoke(stypy.reporting.localization.Localization(__file__, 160, 46), getitem___93573, str_93571)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 26), tuple_93566, subscript_call_result_93574)
    # Adding element type (line 160)
    # Getting the type of 'dms' (line 160)
    dms_93575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 58), 'dms', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 26), tuple_93566, dms_93575)
    # Adding element type (line 160)
    # Getting the type of 'at' (line 160)
    at_93576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 63), 'at', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 26), tuple_93566, at_93576)
    
    # Applying the binary operator '%' (line 159)
    result_mod_93577 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 25), '%', str_93565, tuple_93566)
    
    # Processing the call keyword arguments (line 159)
    kwargs_93578 = {}
    # Getting the type of 'cadd' (line 159)
    cadd_93564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 20), 'cadd', False)
    # Calling cadd(args, kwargs) (line 159)
    cadd_call_result_93579 = invoke(stypy.reporting.localization.Localization(__file__, 159, 20), cadd_93564, *[result_mod_93577], **kwargs_93578)
    
    # SSA join for if statement (line 154)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 153)
    module_type_store.open_ssa_branch('else')
    
    # Call to cadd(...): (line 162)
    # Processing the call arguments (line 162)
    str_93581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 21), 'str', '\t{"%s",%s,{{%s}},%s},')
    
    # Obtaining an instance of the builtin type 'tuple' (line 163)
    tuple_93582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 163)
    # Adding element type (line 163)
    
    # Call to undo_rmbadname1(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'n' (line 163)
    n_93584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 38), 'n', False)
    # Processing the call keyword arguments (line 163)
    kwargs_93585 = {}
    # Getting the type of 'undo_rmbadname1' (line 163)
    undo_rmbadname1_93583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 22), 'undo_rmbadname1', False)
    # Calling undo_rmbadname1(args, kwargs) (line 163)
    undo_rmbadname1_call_result_93586 = invoke(stypy.reporting.localization.Localization(__file__, 163, 22), undo_rmbadname1_93583, *[n_93584], **kwargs_93585)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 22), tuple_93582, undo_rmbadname1_call_result_93586)
    # Adding element type (line 163)
    
    # Obtaining the type of the subscript
    str_93587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 45), 'str', 'rank')
    # Getting the type of 'dm' (line 163)
    dm_93588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 42), 'dm', False)
    # Obtaining the member '__getitem__' of a type (line 163)
    getitem___93589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 42), dm_93588, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 163)
    subscript_call_result_93590 = invoke(stypy.reporting.localization.Localization(__file__, 163, 42), getitem___93589, str_93587)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 22), tuple_93582, subscript_call_result_93590)
    # Adding element type (line 163)
    # Getting the type of 'dms' (line 163)
    dms_93591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 54), 'dms', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 22), tuple_93582, dms_93591)
    # Adding element type (line 163)
    # Getting the type of 'at' (line 163)
    at_93592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 59), 'at', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 22), tuple_93582, at_93592)
    
    # Applying the binary operator '%' (line 162)
    result_mod_93593 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 21), '%', str_93581, tuple_93582)
    
    # Processing the call keyword arguments (line 162)
    kwargs_93594 = {}
    # Getting the type of 'cadd' (line 162)
    cadd_93580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'cadd', False)
    # Calling cadd(args, kwargs) (line 162)
    cadd_call_result_93595 = invoke(stypy.reporting.localization.Localization(__file__, 162, 16), cadd_93580, *[result_mod_93593], **kwargs_93594)
    
    # SSA join for if statement (line 153)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to dadd(...): (line 164)
    # Processing the call arguments (line 164)
    str_93597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 17), 'str', '\\item[]{{}\\verb@%s@{}}')
    
    # Call to getarrdocsign(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'n' (line 165)
    n_93600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 42), 'n', False)
    # Getting the type of 'var' (line 165)
    var_93601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 45), 'var', False)
    # Processing the call keyword arguments (line 165)
    kwargs_93602 = {}
    # Getting the type of 'capi_maps' (line 165)
    capi_maps_93598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 18), 'capi_maps', False)
    # Obtaining the member 'getarrdocsign' of a type (line 165)
    getarrdocsign_93599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 18), capi_maps_93598, 'getarrdocsign')
    # Calling getarrdocsign(args, kwargs) (line 165)
    getarrdocsign_call_result_93603 = invoke(stypy.reporting.localization.Localization(__file__, 165, 18), getarrdocsign_93599, *[n_93600, var_93601], **kwargs_93602)
    
    # Applying the binary operator '%' (line 164)
    result_mod_93604 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 17), '%', str_93597, getarrdocsign_call_result_93603)
    
    # Processing the call keyword arguments (line 164)
    kwargs_93605 = {}
    # Getting the type of 'dadd' (line 164)
    dadd_93596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'dadd', False)
    # Calling dadd(args, kwargs) (line 164)
    dadd_call_result_93606 = invoke(stypy.reporting.localization.Localization(__file__, 164, 12), dadd_93596, *[result_mod_93604], **kwargs_93605)
    
    
    
    # Call to hasnote(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'var' (line 166)
    var_93608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 23), 'var', False)
    # Processing the call keyword arguments (line 166)
    kwargs_93609 = {}
    # Getting the type of 'hasnote' (line 166)
    hasnote_93607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 15), 'hasnote', False)
    # Calling hasnote(args, kwargs) (line 166)
    hasnote_call_result_93610 = invoke(stypy.reporting.localization.Localization(__file__, 166, 15), hasnote_93607, *[var_93608], **kwargs_93609)
    
    # Testing the type of an if condition (line 166)
    if_condition_93611 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 166, 12), hasnote_call_result_93610)
    # Assigning a type to the variable 'if_condition_93611' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'if_condition_93611', if_condition_93611)
    # SSA begins for if statement (line 166)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 167):
    
    # Assigning a Subscript to a Name (line 167):
    
    # Obtaining the type of the subscript
    str_93612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 27), 'str', 'note')
    # Getting the type of 'var' (line 167)
    var_93613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 23), 'var')
    # Obtaining the member '__getitem__' of a type (line 167)
    getitem___93614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 23), var_93613, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 167)
    subscript_call_result_93615 = invoke(stypy.reporting.localization.Localization(__file__, 167, 23), getitem___93614, str_93612)
    
    # Assigning a type to the variable 'note' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'note', subscript_call_result_93615)
    
    # Type idiom detected: calculating its left and rigth part (line 168)
    # Getting the type of 'list' (line 168)
    list_93616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 36), 'list')
    # Getting the type of 'note' (line 168)
    note_93617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 30), 'note')
    
    (may_be_93618, more_types_in_union_93619) = may_be_subtype(list_93616, note_93617)

    if may_be_93618:

        if more_types_in_union_93619:
            # Runtime conditional SSA (line 168)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'note' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'note', remove_not_subtype_from_union(note_93617, list))
        
        # Assigning a Call to a Name (line 169):
        
        # Assigning a Call to a Name (line 169):
        
        # Call to join(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'note' (line 169)
        note_93622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 37), 'note', False)
        # Processing the call keyword arguments (line 169)
        kwargs_93623 = {}
        str_93620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 27), 'str', '\n')
        # Obtaining the member 'join' of a type (line 169)
        join_93621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 27), str_93620, 'join')
        # Calling join(args, kwargs) (line 169)
        join_call_result_93624 = invoke(stypy.reporting.localization.Localization(__file__, 169, 27), join_93621, *[note_93622], **kwargs_93623)
        
        # Assigning a type to the variable 'note' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 20), 'note', join_call_result_93624)

        if more_types_in_union_93619:
            # SSA join for if statement (line 168)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to dadd(...): (line 170)
    # Processing the call arguments (line 170)
    str_93626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 21), 'str', '--- %s')
    # Getting the type of 'note' (line 170)
    note_93627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 33), 'note', False)
    # Applying the binary operator '%' (line 170)
    result_mod_93628 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 21), '%', str_93626, note_93627)
    
    # Processing the call keyword arguments (line 170)
    kwargs_93629 = {}
    # Getting the type of 'dadd' (line 170)
    dadd_93625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'dadd', False)
    # Calling dadd(args, kwargs) (line 170)
    dadd_call_result_93630 = invoke(stypy.reporting.localization.Localization(__file__, 170, 16), dadd_93625, *[result_mod_93628], **kwargs_93629)
    
    # SSA join for if statement (line 166)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isallocatable(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'var' (line 171)
    var_93632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 29), 'var', False)
    # Processing the call keyword arguments (line 171)
    kwargs_93633 = {}
    # Getting the type of 'isallocatable' (line 171)
    isallocatable_93631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 15), 'isallocatable', False)
    # Calling isallocatable(args, kwargs) (line 171)
    isallocatable_call_result_93634 = invoke(stypy.reporting.localization.Localization(__file__, 171, 15), isallocatable_93631, *[var_93632], **kwargs_93633)
    
    # Testing the type of an if condition (line 171)
    if_condition_93635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 12), isallocatable_call_result_93634)
    # Assigning a type to the variable 'if_condition_93635' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'if_condition_93635', if_condition_93635)
    # SSA begins for if statement (line 171)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 172)
    # Processing the call arguments (line 172)
    str_93638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 29), 'str', 'f2py_%s_getdims_%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 172)
    tuple_93639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 53), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 172)
    # Adding element type (line 172)
    
    # Obtaining the type of the subscript
    str_93640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 55), 'str', 'name')
    # Getting the type of 'm' (line 172)
    m_93641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 53), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 172)
    getitem___93642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 53), m_93641, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 172)
    subscript_call_result_93643 = invoke(stypy.reporting.localization.Localization(__file__, 172, 53), getitem___93642, str_93640)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 53), tuple_93639, subscript_call_result_93643)
    # Adding element type (line 172)
    # Getting the type of 'n' (line 172)
    n_93644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 64), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 53), tuple_93639, n_93644)
    
    # Applying the binary operator '%' (line 172)
    result_mod_93645 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 29), '%', str_93638, tuple_93639)
    
    # Processing the call keyword arguments (line 172)
    kwargs_93646 = {}
    # Getting the type of 'fargs' (line 172)
    fargs_93636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'fargs', False)
    # Obtaining the member 'append' of a type (line 172)
    append_93637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 16), fargs_93636, 'append')
    # Calling append(args, kwargs) (line 172)
    append_call_result_93647 = invoke(stypy.reporting.localization.Localization(__file__, 172, 16), append_93637, *[result_mod_93645], **kwargs_93646)
    
    
    # Call to append(...): (line 173)
    # Processing the call arguments (line 173)
    
    # Obtaining the type of the subscript
    int_93650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 36), 'int')
    # Getting the type of 'fargs' (line 173)
    fargs_93651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 30), 'fargs', False)
    # Obtaining the member '__getitem__' of a type (line 173)
    getitem___93652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 30), fargs_93651, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 173)
    subscript_call_result_93653 = invoke(stypy.reporting.localization.Localization(__file__, 173, 30), getitem___93652, int_93650)
    
    # Processing the call keyword arguments (line 173)
    kwargs_93654 = {}
    # Getting the type of 'efargs' (line 173)
    efargs_93648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'efargs', False)
    # Obtaining the member 'append' of a type (line 173)
    append_93649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 16), efargs_93648, 'append')
    # Calling append(args, kwargs) (line 173)
    append_call_result_93655 = invoke(stypy.reporting.localization.Localization(__file__, 173, 16), append_93649, *[subscript_call_result_93653], **kwargs_93654)
    
    
    # Call to append(...): (line 174)
    # Processing the call arguments (line 174)
    str_93658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 20), 'str', 'void (*%s)(int*,int*,void(*)(char*,int*),int*)')
    # Getting the type of 'n' (line 175)
    n_93659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 72), 'n', False)
    # Applying the binary operator '%' (line 175)
    result_mod_93660 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 20), '%', str_93658, n_93659)
    
    # Processing the call keyword arguments (line 174)
    kwargs_93661 = {}
    # Getting the type of 'sargs' (line 174)
    sargs_93656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'sargs', False)
    # Obtaining the member 'append' of a type (line 174)
    append_93657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 16), sargs_93656, 'append')
    # Calling append(args, kwargs) (line 174)
    append_call_result_93662 = invoke(stypy.reporting.localization.Localization(__file__, 174, 16), append_93657, *[result_mod_93660], **kwargs_93661)
    
    
    # Call to append(...): (line 176)
    # Processing the call arguments (line 176)
    str_93665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 30), 'str', 'void (*)(int*,int*,void(*)(char*,int*),int*)')
    # Processing the call keyword arguments (line 176)
    kwargs_93666 = {}
    # Getting the type of 'sargsp' (line 176)
    sargsp_93663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 16), 'sargsp', False)
    # Obtaining the member 'append' of a type (line 176)
    append_93664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 16), sargsp_93663, 'append')
    # Calling append(args, kwargs) (line 176)
    append_call_result_93667 = invoke(stypy.reporting.localization.Localization(__file__, 176, 16), append_93664, *[str_93665], **kwargs_93666)
    
    
    # Call to iadd(...): (line 177)
    # Processing the call arguments (line 177)
    str_93669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 21), 'str', '\tf2py_%s_def[i_f2py++].func = %s;')
    
    # Obtaining an instance of the builtin type 'tuple' (line 177)
    tuple_93670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 61), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 177)
    # Adding element type (line 177)
    
    # Obtaining the type of the subscript
    str_93671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 63), 'str', 'name')
    # Getting the type of 'm' (line 177)
    m_93672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 61), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 177)
    getitem___93673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 61), m_93672, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 177)
    subscript_call_result_93674 = invoke(stypy.reporting.localization.Localization(__file__, 177, 61), getitem___93673, str_93671)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 61), tuple_93670, subscript_call_result_93674)
    # Adding element type (line 177)
    # Getting the type of 'n' (line 177)
    n_93675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 72), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 61), tuple_93670, n_93675)
    
    # Applying the binary operator '%' (line 177)
    result_mod_93676 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 21), '%', str_93669, tuple_93670)
    
    # Processing the call keyword arguments (line 177)
    kwargs_93677 = {}
    # Getting the type of 'iadd' (line 177)
    iadd_93668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 16), 'iadd', False)
    # Calling iadd(args, kwargs) (line 177)
    iadd_call_result_93678 = invoke(stypy.reporting.localization.Localization(__file__, 177, 16), iadd_93668, *[result_mod_93676], **kwargs_93677)
    
    
    # Call to fadd(...): (line 178)
    # Processing the call arguments (line 178)
    str_93680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 21), 'str', 'subroutine %s(r,s,f2pysetdata,flag)')
    
    # Obtaining the type of the subscript
    int_93681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 68), 'int')
    # Getting the type of 'fargs' (line 178)
    fargs_93682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 62), 'fargs', False)
    # Obtaining the member '__getitem__' of a type (line 178)
    getitem___93683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 62), fargs_93682, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 178)
    subscript_call_result_93684 = invoke(stypy.reporting.localization.Localization(__file__, 178, 62), getitem___93683, int_93681)
    
    # Applying the binary operator '%' (line 178)
    result_mod_93685 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 21), '%', str_93680, subscript_call_result_93684)
    
    # Processing the call keyword arguments (line 178)
    kwargs_93686 = {}
    # Getting the type of 'fadd' (line 178)
    fadd_93679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'fadd', False)
    # Calling fadd(args, kwargs) (line 178)
    fadd_call_result_93687 = invoke(stypy.reporting.localization.Localization(__file__, 178, 16), fadd_93679, *[result_mod_93685], **kwargs_93686)
    
    
    # Call to fadd(...): (line 179)
    # Processing the call arguments (line 179)
    str_93689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 21), 'str', 'use %s, only: d => %s\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 180)
    tuple_93690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 180)
    # Adding element type (line 180)
    
    # Obtaining the type of the subscript
    str_93691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 24), 'str', 'name')
    # Getting the type of 'm' (line 180)
    m_93692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 22), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 180)
    getitem___93693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 22), m_93692, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 180)
    subscript_call_result_93694 = invoke(stypy.reporting.localization.Localization(__file__, 180, 22), getitem___93693, str_93691)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 22), tuple_93690, subscript_call_result_93694)
    # Adding element type (line 180)
    
    # Call to undo_rmbadname1(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'n' (line 180)
    n_93696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 49), 'n', False)
    # Processing the call keyword arguments (line 180)
    kwargs_93697 = {}
    # Getting the type of 'undo_rmbadname1' (line 180)
    undo_rmbadname1_93695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 33), 'undo_rmbadname1', False)
    # Calling undo_rmbadname1(args, kwargs) (line 180)
    undo_rmbadname1_call_result_93698 = invoke(stypy.reporting.localization.Localization(__file__, 180, 33), undo_rmbadname1_93695, *[n_93696], **kwargs_93697)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 22), tuple_93690, undo_rmbadname1_call_result_93698)
    
    # Applying the binary operator '%' (line 179)
    result_mod_93699 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 21), '%', str_93689, tuple_93690)
    
    # Processing the call keyword arguments (line 179)
    kwargs_93700 = {}
    # Getting the type of 'fadd' (line 179)
    fadd_93688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'fadd', False)
    # Calling fadd(args, kwargs) (line 179)
    fadd_call_result_93701 = invoke(stypy.reporting.localization.Localization(__file__, 179, 16), fadd_93688, *[result_mod_93699], **kwargs_93700)
    
    
    # Call to fadd(...): (line 181)
    # Processing the call arguments (line 181)
    str_93703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 21), 'str', 'integer flag\n')
    # Processing the call keyword arguments (line 181)
    kwargs_93704 = {}
    # Getting the type of 'fadd' (line 181)
    fadd_93702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'fadd', False)
    # Calling fadd(args, kwargs) (line 181)
    fadd_call_result_93705 = invoke(stypy.reporting.localization.Localization(__file__, 181, 16), fadd_93702, *[str_93703], **kwargs_93704)
    
    
    # Assigning a BinOp to a Subscript (line 182):
    
    # Assigning a BinOp to a Subscript (line 182):
    
    # Obtaining the type of the subscript
    int_93706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 35), 'int')
    # Getting the type of 'fhooks' (line 182)
    fhooks_93707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 28), 'fhooks')
    # Obtaining the member '__getitem__' of a type (line 182)
    getitem___93708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 28), fhooks_93707, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 182)
    subscript_call_result_93709 = invoke(stypy.reporting.localization.Localization(__file__, 182, 28), getitem___93708, int_93706)
    
    # Getting the type of 'fgetdims1' (line 182)
    fgetdims1_93710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 40), 'fgetdims1')
    # Applying the binary operator '+' (line 182)
    result_add_93711 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 28), '+', subscript_call_result_93709, fgetdims1_93710)
    
    # Getting the type of 'fhooks' (line 182)
    fhooks_93712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'fhooks')
    int_93713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 23), 'int')
    # Storing an element on a container (line 182)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 16), fhooks_93712, (int_93713, result_add_93711))
    
    # Assigning a Call to a Name (line 183):
    
    # Assigning a Call to a Name (line 183):
    
    # Call to eval(...): (line 183)
    # Processing the call arguments (line 183)
    str_93715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 27), 'str', 'range(1,%s+1)')
    
    # Obtaining the type of the subscript
    str_93716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 49), 'str', 'rank')
    # Getting the type of 'dm' (line 183)
    dm_93717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 46), 'dm', False)
    # Obtaining the member '__getitem__' of a type (line 183)
    getitem___93718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 46), dm_93717, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 183)
    subscript_call_result_93719 = invoke(stypy.reporting.localization.Localization(__file__, 183, 46), getitem___93718, str_93716)
    
    # Applying the binary operator '%' (line 183)
    result_mod_93720 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 27), '%', str_93715, subscript_call_result_93719)
    
    # Processing the call keyword arguments (line 183)
    kwargs_93721 = {}
    # Getting the type of 'eval' (line 183)
    eval_93714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 22), 'eval', False)
    # Calling eval(args, kwargs) (line 183)
    eval_call_result_93722 = invoke(stypy.reporting.localization.Localization(__file__, 183, 22), eval_93714, *[result_mod_93720], **kwargs_93721)
    
    # Assigning a type to the variable 'dms' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'dms', eval_call_result_93722)
    
    # Call to fadd(...): (line 184)
    # Processing the call arguments (line 184)
    str_93724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 21), 'str', ' allocate(d(%s))\n')
    
    # Call to join(...): (line 185)
    # Processing the call arguments (line 185)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'dms' (line 185)
    dms_93730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 53), 'dms', False)
    comprehension_93731 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 32), dms_93730)
    # Assigning a type to the variable 'i' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 32), 'i', comprehension_93731)
    str_93727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 32), 'str', 's(%s)')
    # Getting the type of 'i' (line 185)
    i_93728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 42), 'i', False)
    # Applying the binary operator '%' (line 185)
    result_mod_93729 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 32), '%', str_93727, i_93728)
    
    list_93732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 32), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 32), list_93732, result_mod_93729)
    # Processing the call keyword arguments (line 185)
    kwargs_93733 = {}
    str_93725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 22), 'str', ',')
    # Obtaining the member 'join' of a type (line 185)
    join_93726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 22), str_93725, 'join')
    # Calling join(args, kwargs) (line 185)
    join_call_result_93734 = invoke(stypy.reporting.localization.Localization(__file__, 185, 22), join_93726, *[list_93732], **kwargs_93733)
    
    # Applying the binary operator '%' (line 184)
    result_mod_93735 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 21), '%', str_93724, join_call_result_93734)
    
    # Processing the call keyword arguments (line 184)
    kwargs_93736 = {}
    # Getting the type of 'fadd' (line 184)
    fadd_93723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'fadd', False)
    # Calling fadd(args, kwargs) (line 184)
    fadd_call_result_93737 = invoke(stypy.reporting.localization.Localization(__file__, 184, 16), fadd_93723, *[result_mod_93735], **kwargs_93736)
    
    
    # Assigning a BinOp to a Subscript (line 186):
    
    # Assigning a BinOp to a Subscript (line 186):
    
    # Obtaining the type of the subscript
    int_93738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 35), 'int')
    # Getting the type of 'fhooks' (line 186)
    fhooks_93739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 28), 'fhooks')
    # Obtaining the member '__getitem__' of a type (line 186)
    getitem___93740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 28), fhooks_93739, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 186)
    subscript_call_result_93741 = invoke(stypy.reporting.localization.Localization(__file__, 186, 28), getitem___93740, int_93738)
    
    # Getting the type of 'use_fgetdims2' (line 186)
    use_fgetdims2_93742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 40), 'use_fgetdims2')
    # Applying the binary operator '+' (line 186)
    result_add_93743 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 28), '+', subscript_call_result_93741, use_fgetdims2_93742)
    
    # Getting the type of 'fhooks' (line 186)
    fhooks_93744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 16), 'fhooks')
    int_93745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 23), 'int')
    # Storing an element on a container (line 186)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 16), fhooks_93744, (int_93745, result_add_93743))
    
    # Call to fadd(...): (line 187)
    # Processing the call arguments (line 187)
    str_93747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 21), 'str', 'end subroutine %s')
    
    # Obtaining the type of the subscript
    int_93748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 50), 'int')
    # Getting the type of 'fargs' (line 187)
    fargs_93749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 44), 'fargs', False)
    # Obtaining the member '__getitem__' of a type (line 187)
    getitem___93750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 44), fargs_93749, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 187)
    subscript_call_result_93751 = invoke(stypy.reporting.localization.Localization(__file__, 187, 44), getitem___93750, int_93748)
    
    # Applying the binary operator '%' (line 187)
    result_mod_93752 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 21), '%', str_93747, subscript_call_result_93751)
    
    # Processing the call keyword arguments (line 187)
    kwargs_93753 = {}
    # Getting the type of 'fadd' (line 187)
    fadd_93746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 16), 'fadd', False)
    # Calling fadd(args, kwargs) (line 187)
    fadd_call_result_93754 = invoke(stypy.reporting.localization.Localization(__file__, 187, 16), fadd_93746, *[result_mod_93752], **kwargs_93753)
    
    # SSA branch for the else part of an if statement (line 171)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of 'n' (line 189)
    n_93757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 29), 'n', False)
    # Processing the call keyword arguments (line 189)
    kwargs_93758 = {}
    # Getting the type of 'fargs' (line 189)
    fargs_93755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 16), 'fargs', False)
    # Obtaining the member 'append' of a type (line 189)
    append_93756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 16), fargs_93755, 'append')
    # Calling append(args, kwargs) (line 189)
    append_call_result_93759 = invoke(stypy.reporting.localization.Localization(__file__, 189, 16), append_93756, *[n_93757], **kwargs_93758)
    
    
    # Call to append(...): (line 190)
    # Processing the call arguments (line 190)
    str_93762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 29), 'str', 'char *%s')
    # Getting the type of 'n' (line 190)
    n_93763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 43), 'n', False)
    # Applying the binary operator '%' (line 190)
    result_mod_93764 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 29), '%', str_93762, n_93763)
    
    # Processing the call keyword arguments (line 190)
    kwargs_93765 = {}
    # Getting the type of 'sargs' (line 190)
    sargs_93760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), 'sargs', False)
    # Obtaining the member 'append' of a type (line 190)
    append_93761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 16), sargs_93760, 'append')
    # Calling append(args, kwargs) (line 190)
    append_call_result_93766 = invoke(stypy.reporting.localization.Localization(__file__, 190, 16), append_93761, *[result_mod_93764], **kwargs_93765)
    
    
    # Call to append(...): (line 191)
    # Processing the call arguments (line 191)
    str_93769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 30), 'str', 'char*')
    # Processing the call keyword arguments (line 191)
    kwargs_93770 = {}
    # Getting the type of 'sargsp' (line 191)
    sargsp_93767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'sargsp', False)
    # Obtaining the member 'append' of a type (line 191)
    append_93768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 16), sargsp_93767, 'append')
    # Calling append(args, kwargs) (line 191)
    append_call_result_93771 = invoke(stypy.reporting.localization.Localization(__file__, 191, 16), append_93768, *[str_93769], **kwargs_93770)
    
    
    # Call to iadd(...): (line 192)
    # Processing the call arguments (line 192)
    str_93773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 21), 'str', '\tf2py_%s_def[i_f2py++].data = %s;')
    
    # Obtaining an instance of the builtin type 'tuple' (line 192)
    tuple_93774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 61), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 192)
    # Adding element type (line 192)
    
    # Obtaining the type of the subscript
    str_93775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 63), 'str', 'name')
    # Getting the type of 'm' (line 192)
    m_93776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 61), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 192)
    getitem___93777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 61), m_93776, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 192)
    subscript_call_result_93778 = invoke(stypy.reporting.localization.Localization(__file__, 192, 61), getitem___93777, str_93775)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 61), tuple_93774, subscript_call_result_93778)
    # Adding element type (line 192)
    # Getting the type of 'n' (line 192)
    n_93779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 72), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 61), tuple_93774, n_93779)
    
    # Applying the binary operator '%' (line 192)
    result_mod_93780 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 21), '%', str_93773, tuple_93774)
    
    # Processing the call keyword arguments (line 192)
    kwargs_93781 = {}
    # Getting the type of 'iadd' (line 192)
    iadd_93772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 16), 'iadd', False)
    # Calling iadd(args, kwargs) (line 192)
    iadd_call_result_93782 = invoke(stypy.reporting.localization.Localization(__file__, 192, 16), iadd_93772, *[result_mod_93780], **kwargs_93781)
    
    # SSA join for if statement (line 171)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'onlyvars' (line 193)
    onlyvars_93783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 11), 'onlyvars')
    # Testing the type of an if condition (line 193)
    if_condition_93784 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 193, 8), onlyvars_93783)
    # Assigning a type to the variable 'if_condition_93784' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'if_condition_93784', if_condition_93784)
    # SSA begins for if statement (line 193)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to dadd(...): (line 194)
    # Processing the call arguments (line 194)
    str_93786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 17), 'str', '\\end{description}')
    # Processing the call keyword arguments (line 194)
    kwargs_93787 = {}
    # Getting the type of 'dadd' (line 194)
    dadd_93785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'dadd', False)
    # Calling dadd(args, kwargs) (line 194)
    dadd_call_result_93788 = invoke(stypy.reporting.localization.Localization(__file__, 194, 12), dadd_93785, *[str_93786], **kwargs_93787)
    
    # SSA join for if statement (line 193)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to hasbody(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of 'm' (line 195)
    m_93790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 19), 'm', False)
    # Processing the call keyword arguments (line 195)
    kwargs_93791 = {}
    # Getting the type of 'hasbody' (line 195)
    hasbody_93789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 11), 'hasbody', False)
    # Calling hasbody(args, kwargs) (line 195)
    hasbody_call_result_93792 = invoke(stypy.reporting.localization.Localization(__file__, 195, 11), hasbody_93789, *[m_93790], **kwargs_93791)
    
    # Testing the type of an if condition (line 195)
    if_condition_93793 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 195, 8), hasbody_call_result_93792)
    # Assigning a type to the variable 'if_condition_93793' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'if_condition_93793', if_condition_93793)
    # SSA begins for if statement (line 195)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Obtaining the type of the subscript
    str_93794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 23), 'str', 'body')
    # Getting the type of 'm' (line 196)
    m_93795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 21), 'm')
    # Obtaining the member '__getitem__' of a type (line 196)
    getitem___93796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 21), m_93795, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 196)
    subscript_call_result_93797 = invoke(stypy.reporting.localization.Localization(__file__, 196, 21), getitem___93796, str_93794)
    
    # Testing the type of a for loop iterable (line 196)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 196, 12), subscript_call_result_93797)
    # Getting the type of the for loop variable (line 196)
    for_loop_var_93798 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 196, 12), subscript_call_result_93797)
    # Assigning a type to the variable 'b' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'b', for_loop_var_93798)
    # SSA begins for a for statement (line 196)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Call to isroutine(...): (line 197)
    # Processing the call arguments (line 197)
    # Getting the type of 'b' (line 197)
    b_93800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 33), 'b', False)
    # Processing the call keyword arguments (line 197)
    kwargs_93801 = {}
    # Getting the type of 'isroutine' (line 197)
    isroutine_93799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 23), 'isroutine', False)
    # Calling isroutine(args, kwargs) (line 197)
    isroutine_call_result_93802 = invoke(stypy.reporting.localization.Localization(__file__, 197, 23), isroutine_93799, *[b_93800], **kwargs_93801)
    
    # Applying the 'not' unary operator (line 197)
    result_not__93803 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 19), 'not', isroutine_call_result_93802)
    
    # Testing the type of an if condition (line 197)
    if_condition_93804 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 16), result_not__93803)
    # Assigning a type to the variable 'if_condition_93804' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 16), 'if_condition_93804', if_condition_93804)
    # SSA begins for if statement (line 197)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 198)
    # Processing the call arguments (line 198)
    str_93806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 26), 'str', 'Skipping')
    
    # Obtaining the type of the subscript
    str_93807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 40), 'str', 'block')
    # Getting the type of 'b' (line 198)
    b_93808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 38), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 198)
    getitem___93809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 38), b_93808, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 198)
    subscript_call_result_93810 = invoke(stypy.reporting.localization.Localization(__file__, 198, 38), getitem___93809, str_93807)
    
    
    # Obtaining the type of the subscript
    str_93811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 52), 'str', 'name')
    # Getting the type of 'b' (line 198)
    b_93812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 50), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 198)
    getitem___93813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 50), b_93812, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 198)
    subscript_call_result_93814 = invoke(stypy.reporting.localization.Localization(__file__, 198, 50), getitem___93813, str_93811)
    
    # Processing the call keyword arguments (line 198)
    kwargs_93815 = {}
    # Getting the type of 'print' (line 198)
    print_93805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 20), 'print', False)
    # Calling print(args, kwargs) (line 198)
    print_call_result_93816 = invoke(stypy.reporting.localization.Localization(__file__, 198, 20), print_93805, *[str_93806, subscript_call_result_93810, subscript_call_result_93814], **kwargs_93815)
    
    # SSA join for if statement (line 197)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 200)
    # Processing the call arguments (line 200)
    str_93819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 31), 'str', '%s()')
    
    # Obtaining the type of the subscript
    str_93820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 43), 'str', 'name')
    # Getting the type of 'b' (line 200)
    b_93821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 41), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___93822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 41), b_93821, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 200)
    subscript_call_result_93823 = invoke(stypy.reporting.localization.Localization(__file__, 200, 41), getitem___93822, str_93820)
    
    # Applying the binary operator '%' (line 200)
    result_mod_93824 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 31), '%', str_93819, subscript_call_result_93823)
    
    # Processing the call keyword arguments (line 200)
    kwargs_93825 = {}
    # Getting the type of 'modobjs' (line 200)
    modobjs_93817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 16), 'modobjs', False)
    # Obtaining the member 'append' of a type (line 200)
    append_93818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 16), modobjs_93817, 'append')
    # Calling append(args, kwargs) (line 200)
    append_call_result_93826 = invoke(stypy.reporting.localization.Localization(__file__, 200, 16), append_93818, *[result_mod_93824], **kwargs_93825)
    
    
    # Assigning a Subscript to a Subscript (line 201):
    
    # Assigning a Subscript to a Subscript (line 201):
    
    # Obtaining the type of the subscript
    str_93827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 36), 'str', 'name')
    # Getting the type of 'm' (line 201)
    m_93828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 34), 'm')
    # Obtaining the member '__getitem__' of a type (line 201)
    getitem___93829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 34), m_93828, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 201)
    subscript_call_result_93830 = invoke(stypy.reporting.localization.Localization(__file__, 201, 34), getitem___93829, str_93827)
    
    # Getting the type of 'b' (line 201)
    b_93831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'b')
    str_93832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 18), 'str', 'modulename')
    # Storing an element on a container (line 201)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 16), b_93831, (str_93832, subscript_call_result_93830))
    
    # Assigning a Call to a Tuple (line 202):
    
    # Assigning a Call to a Name:
    
    # Call to buildapi(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'b' (line 202)
    b_93835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 43), 'b', False)
    # Processing the call keyword arguments (line 202)
    kwargs_93836 = {}
    # Getting the type of 'rules' (line 202)
    rules_93833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 28), 'rules', False)
    # Obtaining the member 'buildapi' of a type (line 202)
    buildapi_93834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 28), rules_93833, 'buildapi')
    # Calling buildapi(args, kwargs) (line 202)
    buildapi_call_result_93837 = invoke(stypy.reporting.localization.Localization(__file__, 202, 28), buildapi_93834, *[b_93835], **kwargs_93836)
    
    # Assigning a type to the variable 'call_assignment_93166' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'call_assignment_93166', buildapi_call_result_93837)
    
    # Assigning a Call to a Name (line 202):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_93840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 16), 'int')
    # Processing the call keyword arguments
    kwargs_93841 = {}
    # Getting the type of 'call_assignment_93166' (line 202)
    call_assignment_93166_93838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'call_assignment_93166', False)
    # Obtaining the member '__getitem__' of a type (line 202)
    getitem___93839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 16), call_assignment_93166_93838, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_93842 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___93839, *[int_93840], **kwargs_93841)
    
    # Assigning a type to the variable 'call_assignment_93167' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'call_assignment_93167', getitem___call_result_93842)
    
    # Assigning a Name to a Name (line 202):
    # Getting the type of 'call_assignment_93167' (line 202)
    call_assignment_93167_93843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'call_assignment_93167')
    # Assigning a type to the variable 'api' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'api', call_assignment_93167_93843)
    
    # Assigning a Call to a Name (line 202):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_93846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 16), 'int')
    # Processing the call keyword arguments
    kwargs_93847 = {}
    # Getting the type of 'call_assignment_93166' (line 202)
    call_assignment_93166_93844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'call_assignment_93166', False)
    # Obtaining the member '__getitem__' of a type (line 202)
    getitem___93845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 16), call_assignment_93166_93844, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_93848 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___93845, *[int_93846], **kwargs_93847)
    
    # Assigning a type to the variable 'call_assignment_93168' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'call_assignment_93168', getitem___call_result_93848)
    
    # Assigning a Name to a Name (line 202):
    # Getting the type of 'call_assignment_93168' (line 202)
    call_assignment_93168_93849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'call_assignment_93168')
    # Assigning a type to the variable 'wrap' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 21), 'wrap', call_assignment_93168_93849)
    
    
    # Call to isfunction(...): (line 203)
    # Processing the call arguments (line 203)
    # Getting the type of 'b' (line 203)
    b_93851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 30), 'b', False)
    # Processing the call keyword arguments (line 203)
    kwargs_93852 = {}
    # Getting the type of 'isfunction' (line 203)
    isfunction_93850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 19), 'isfunction', False)
    # Calling isfunction(args, kwargs) (line 203)
    isfunction_call_result_93853 = invoke(stypy.reporting.localization.Localization(__file__, 203, 19), isfunction_93850, *[b_93851], **kwargs_93852)
    
    # Testing the type of an if condition (line 203)
    if_condition_93854 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 16), isfunction_call_result_93853)
    # Assigning a type to the variable 'if_condition_93854' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 'if_condition_93854', if_condition_93854)
    # SSA begins for if statement (line 203)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 204):
    
    # Assigning a BinOp to a Subscript (line 204):
    
    # Obtaining the type of the subscript
    int_93855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 39), 'int')
    # Getting the type of 'fhooks' (line 204)
    fhooks_93856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 32), 'fhooks')
    # Obtaining the member '__getitem__' of a type (line 204)
    getitem___93857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 32), fhooks_93856, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 204)
    subscript_call_result_93858 = invoke(stypy.reporting.localization.Localization(__file__, 204, 32), getitem___93857, int_93855)
    
    # Getting the type of 'wrap' (line 204)
    wrap_93859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 44), 'wrap')
    # Applying the binary operator '+' (line 204)
    result_add_93860 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 32), '+', subscript_call_result_93858, wrap_93859)
    
    # Getting the type of 'fhooks' (line 204)
    fhooks_93861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 20), 'fhooks')
    int_93862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 27), 'int')
    # Storing an element on a container (line 204)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 20), fhooks_93861, (int_93862, result_add_93860))
    
    # Call to append(...): (line 205)
    # Processing the call arguments (line 205)
    str_93865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 33), 'str', 'f2pywrap_%s_%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 205)
    tuple_93866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 53), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 205)
    # Adding element type (line 205)
    
    # Obtaining the type of the subscript
    str_93867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 55), 'str', 'name')
    # Getting the type of 'm' (line 205)
    m_93868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 53), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___93869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 53), m_93868, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_93870 = invoke(stypy.reporting.localization.Localization(__file__, 205, 53), getitem___93869, str_93867)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 53), tuple_93866, subscript_call_result_93870)
    # Adding element type (line 205)
    
    # Obtaining the type of the subscript
    str_93871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 66), 'str', 'name')
    # Getting the type of 'b' (line 205)
    b_93872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 64), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___93873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 64), b_93872, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_93874 = invoke(stypy.reporting.localization.Localization(__file__, 205, 64), getitem___93873, str_93871)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 53), tuple_93866, subscript_call_result_93874)
    
    # Applying the binary operator '%' (line 205)
    result_mod_93875 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 33), '%', str_93865, tuple_93866)
    
    # Processing the call keyword arguments (line 205)
    kwargs_93876 = {}
    # Getting the type of 'fargs' (line 205)
    fargs_93863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'fargs', False)
    # Obtaining the member 'append' of a type (line 205)
    append_93864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 20), fargs_93863, 'append')
    # Calling append(args, kwargs) (line 205)
    append_call_result_93877 = invoke(stypy.reporting.localization.Localization(__file__, 205, 20), append_93864, *[result_mod_93875], **kwargs_93876)
    
    
    # Call to append(...): (line 206)
    # Processing the call arguments (line 206)
    
    # Call to createfuncwrapper(...): (line 206)
    # Processing the call arguments (line 206)
    # Getting the type of 'b' (line 206)
    b_93882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 62), 'b', False)
    # Processing the call keyword arguments (line 206)
    int_93883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 75), 'int')
    keyword_93884 = int_93883
    kwargs_93885 = {'signature': keyword_93884}
    # Getting the type of 'func2subr' (line 206)
    func2subr_93880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 34), 'func2subr', False)
    # Obtaining the member 'createfuncwrapper' of a type (line 206)
    createfuncwrapper_93881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 34), func2subr_93880, 'createfuncwrapper')
    # Calling createfuncwrapper(args, kwargs) (line 206)
    createfuncwrapper_call_result_93886 = invoke(stypy.reporting.localization.Localization(__file__, 206, 34), createfuncwrapper_93881, *[b_93882], **kwargs_93885)
    
    # Processing the call keyword arguments (line 206)
    kwargs_93887 = {}
    # Getting the type of 'ifargs' (line 206)
    ifargs_93878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 20), 'ifargs', False)
    # Obtaining the member 'append' of a type (line 206)
    append_93879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 20), ifargs_93878, 'append')
    # Calling append(args, kwargs) (line 206)
    append_call_result_93888 = invoke(stypy.reporting.localization.Localization(__file__, 206, 20), append_93879, *[createfuncwrapper_call_result_93886], **kwargs_93887)
    
    # SSA branch for the else part of an if statement (line 203)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'wrap' (line 208)
    wrap_93889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 23), 'wrap')
    # Testing the type of an if condition (line 208)
    if_condition_93890 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 208, 20), wrap_93889)
    # Assigning a type to the variable 'if_condition_93890' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 20), 'if_condition_93890', if_condition_93890)
    # SSA begins for if statement (line 208)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 209):
    
    # Assigning a BinOp to a Subscript (line 209):
    
    # Obtaining the type of the subscript
    int_93891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 43), 'int')
    # Getting the type of 'fhooks' (line 209)
    fhooks_93892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 36), 'fhooks')
    # Obtaining the member '__getitem__' of a type (line 209)
    getitem___93893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 36), fhooks_93892, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 209)
    subscript_call_result_93894 = invoke(stypy.reporting.localization.Localization(__file__, 209, 36), getitem___93893, int_93891)
    
    # Getting the type of 'wrap' (line 209)
    wrap_93895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 48), 'wrap')
    # Applying the binary operator '+' (line 209)
    result_add_93896 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 36), '+', subscript_call_result_93894, wrap_93895)
    
    # Getting the type of 'fhooks' (line 209)
    fhooks_93897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 24), 'fhooks')
    int_93898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 31), 'int')
    # Storing an element on a container (line 209)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 24), fhooks_93897, (int_93898, result_add_93896))
    
    # Call to append(...): (line 210)
    # Processing the call arguments (line 210)
    str_93901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 37), 'str', 'f2pywrap_%s_%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 210)
    tuple_93902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 57), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 210)
    # Adding element type (line 210)
    
    # Obtaining the type of the subscript
    str_93903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 59), 'str', 'name')
    # Getting the type of 'm' (line 210)
    m_93904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 57), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 210)
    getitem___93905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 57), m_93904, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 210)
    subscript_call_result_93906 = invoke(stypy.reporting.localization.Localization(__file__, 210, 57), getitem___93905, str_93903)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 57), tuple_93902, subscript_call_result_93906)
    # Adding element type (line 210)
    
    # Obtaining the type of the subscript
    str_93907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 70), 'str', 'name')
    # Getting the type of 'b' (line 210)
    b_93908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 68), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 210)
    getitem___93909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 68), b_93908, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 210)
    subscript_call_result_93910 = invoke(stypy.reporting.localization.Localization(__file__, 210, 68), getitem___93909, str_93907)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 57), tuple_93902, subscript_call_result_93910)
    
    # Applying the binary operator '%' (line 210)
    result_mod_93911 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 37), '%', str_93901, tuple_93902)
    
    # Processing the call keyword arguments (line 210)
    kwargs_93912 = {}
    # Getting the type of 'fargs' (line 210)
    fargs_93899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 24), 'fargs', False)
    # Obtaining the member 'append' of a type (line 210)
    append_93900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 24), fargs_93899, 'append')
    # Calling append(args, kwargs) (line 210)
    append_call_result_93913 = invoke(stypy.reporting.localization.Localization(__file__, 210, 24), append_93900, *[result_mod_93911], **kwargs_93912)
    
    
    # Call to append(...): (line 211)
    # Processing the call arguments (line 211)
    
    # Call to createsubrwrapper(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of 'b' (line 212)
    b_93918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 56), 'b', False)
    # Processing the call keyword arguments (line 212)
    int_93919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 69), 'int')
    keyword_93920 = int_93919
    kwargs_93921 = {'signature': keyword_93920}
    # Getting the type of 'func2subr' (line 212)
    func2subr_93916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 28), 'func2subr', False)
    # Obtaining the member 'createsubrwrapper' of a type (line 212)
    createsubrwrapper_93917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 28), func2subr_93916, 'createsubrwrapper')
    # Calling createsubrwrapper(args, kwargs) (line 212)
    createsubrwrapper_call_result_93922 = invoke(stypy.reporting.localization.Localization(__file__, 212, 28), createsubrwrapper_93917, *[b_93918], **kwargs_93921)
    
    # Processing the call keyword arguments (line 211)
    kwargs_93923 = {}
    # Getting the type of 'ifargs' (line 211)
    ifargs_93914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 24), 'ifargs', False)
    # Obtaining the member 'append' of a type (line 211)
    append_93915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 24), ifargs_93914, 'append')
    # Calling append(args, kwargs) (line 211)
    append_call_result_93924 = invoke(stypy.reporting.localization.Localization(__file__, 211, 24), append_93915, *[createsubrwrapper_call_result_93922], **kwargs_93923)
    
    # SSA branch for the else part of an if statement (line 208)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 214)
    # Processing the call arguments (line 214)
    
    # Obtaining the type of the subscript
    str_93927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 39), 'str', 'name')
    # Getting the type of 'b' (line 214)
    b_93928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 37), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 214)
    getitem___93929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 37), b_93928, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 214)
    subscript_call_result_93930 = invoke(stypy.reporting.localization.Localization(__file__, 214, 37), getitem___93929, str_93927)
    
    # Processing the call keyword arguments (line 214)
    kwargs_93931 = {}
    # Getting the type of 'fargs' (line 214)
    fargs_93925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 24), 'fargs', False)
    # Obtaining the member 'append' of a type (line 214)
    append_93926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 24), fargs_93925, 'append')
    # Calling append(args, kwargs) (line 214)
    append_call_result_93932 = invoke(stypy.reporting.localization.Localization(__file__, 214, 24), append_93926, *[subscript_call_result_93930], **kwargs_93931)
    
    
    # Call to append(...): (line 215)
    # Processing the call arguments (line 215)
    
    # Obtaining the type of the subscript
    int_93935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 44), 'int')
    # Getting the type of 'fargs' (line 215)
    fargs_93936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 38), 'fargs', False)
    # Obtaining the member '__getitem__' of a type (line 215)
    getitem___93937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 38), fargs_93936, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 215)
    subscript_call_result_93938 = invoke(stypy.reporting.localization.Localization(__file__, 215, 38), getitem___93937, int_93935)
    
    # Processing the call keyword arguments (line 215)
    kwargs_93939 = {}
    # Getting the type of 'mfargs' (line 215)
    mfargs_93933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 24), 'mfargs', False)
    # Obtaining the member 'append' of a type (line 215)
    append_93934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 24), mfargs_93933, 'append')
    # Calling append(args, kwargs) (line 215)
    append_call_result_93940 = invoke(stypy.reporting.localization.Localization(__file__, 215, 24), append_93934, *[subscript_call_result_93938], **kwargs_93939)
    
    # SSA join for if statement (line 208)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 203)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Subscript (line 216):
    
    # Assigning a List to a Subscript (line 216):
    
    # Obtaining an instance of the builtin type 'list' (line 216)
    list_93941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 216)
    
    # Getting the type of 'api' (line 216)
    api_93942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'api')
    str_93943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 20), 'str', 'externroutines')
    # Storing an element on a container (line 216)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 16), api_93942, (str_93943, list_93941))
    
    # Assigning a Call to a Name (line 217):
    
    # Assigning a Call to a Name (line 217):
    
    # Call to applyrules(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'api' (line 217)
    api_93945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 32), 'api', False)
    # Getting the type of 'vrd' (line 217)
    vrd_93946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 37), 'vrd', False)
    # Processing the call keyword arguments (line 217)
    kwargs_93947 = {}
    # Getting the type of 'applyrules' (line 217)
    applyrules_93944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 21), 'applyrules', False)
    # Calling applyrules(args, kwargs) (line 217)
    applyrules_call_result_93948 = invoke(stypy.reporting.localization.Localization(__file__, 217, 21), applyrules_93944, *[api_93945, vrd_93946], **kwargs_93947)
    
    # Assigning a type to the variable 'ar' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 16), 'ar', applyrules_call_result_93948)
    
    # Assigning a List to a Subscript (line 218):
    
    # Assigning a List to a Subscript (line 218):
    
    # Obtaining an instance of the builtin type 'list' (line 218)
    list_93949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 218)
    
    # Getting the type of 'ar' (line 218)
    ar_93950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'ar')
    str_93951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 19), 'str', 'docs')
    # Storing an element on a container (line 218)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 16), ar_93950, (str_93951, list_93949))
    
    # Assigning a List to a Subscript (line 219):
    
    # Assigning a List to a Subscript (line 219):
    
    # Obtaining an instance of the builtin type 'list' (line 219)
    list_93952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 219)
    
    # Getting the type of 'ar' (line 219)
    ar_93953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'ar')
    str_93954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 19), 'str', 'docshort')
    # Storing an element on a container (line 219)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 16), ar_93953, (str_93954, list_93952))
    
    # Assigning a Call to a Name (line 220):
    
    # Assigning a Call to a Name (line 220):
    
    # Call to dictappend(...): (line 220)
    # Processing the call arguments (line 220)
    # Getting the type of 'ret' (line 220)
    ret_93956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 33), 'ret', False)
    # Getting the type of 'ar' (line 220)
    ar_93957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 38), 'ar', False)
    # Processing the call keyword arguments (line 220)
    kwargs_93958 = {}
    # Getting the type of 'dictappend' (line 220)
    dictappend_93955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 22), 'dictappend', False)
    # Calling dictappend(args, kwargs) (line 220)
    dictappend_call_result_93959 = invoke(stypy.reporting.localization.Localization(__file__, 220, 22), dictappend_93955, *[ret_93956, ar_93957], **kwargs_93958)
    
    # Assigning a type to the variable 'ret' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'ret', dictappend_call_result_93959)
    
    # Call to cadd(...): (line 221)
    # Processing the call arguments (line 221)
    str_93961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 21), 'str', '\t{"%s",-1,{{-1}},0,NULL,(void *)f2py_rout_#modulename#_%s_%s,doc_f2py_rout_#modulename#_%s_%s},')
    
    # Obtaining an instance of the builtin type 'tuple' (line 222)
    tuple_93962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 222)
    # Adding element type (line 222)
    
    # Obtaining the type of the subscript
    str_93963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 24), 'str', 'name')
    # Getting the type of 'b' (line 222)
    b_93964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 22), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 222)
    getitem___93965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 22), b_93964, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 222)
    subscript_call_result_93966 = invoke(stypy.reporting.localization.Localization(__file__, 222, 22), getitem___93965, str_93963)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 22), tuple_93962, subscript_call_result_93966)
    # Adding element type (line 222)
    
    # Obtaining the type of the subscript
    str_93967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 35), 'str', 'name')
    # Getting the type of 'm' (line 222)
    m_93968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 33), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 222)
    getitem___93969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 33), m_93968, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 222)
    subscript_call_result_93970 = invoke(stypy.reporting.localization.Localization(__file__, 222, 33), getitem___93969, str_93967)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 22), tuple_93962, subscript_call_result_93970)
    # Adding element type (line 222)
    
    # Obtaining the type of the subscript
    str_93971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 46), 'str', 'name')
    # Getting the type of 'b' (line 222)
    b_93972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 44), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 222)
    getitem___93973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 44), b_93972, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 222)
    subscript_call_result_93974 = invoke(stypy.reporting.localization.Localization(__file__, 222, 44), getitem___93973, str_93971)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 22), tuple_93962, subscript_call_result_93974)
    # Adding element type (line 222)
    
    # Obtaining the type of the subscript
    str_93975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 57), 'str', 'name')
    # Getting the type of 'm' (line 222)
    m_93976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 55), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 222)
    getitem___93977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 55), m_93976, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 222)
    subscript_call_result_93978 = invoke(stypy.reporting.localization.Localization(__file__, 222, 55), getitem___93977, str_93975)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 22), tuple_93962, subscript_call_result_93978)
    # Adding element type (line 222)
    
    # Obtaining the type of the subscript
    str_93979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 68), 'str', 'name')
    # Getting the type of 'b' (line 222)
    b_93980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 66), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 222)
    getitem___93981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 66), b_93980, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 222)
    subscript_call_result_93982 = invoke(stypy.reporting.localization.Localization(__file__, 222, 66), getitem___93981, str_93979)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 22), tuple_93962, subscript_call_result_93982)
    
    # Applying the binary operator '%' (line 221)
    result_mod_93983 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 21), '%', str_93961, tuple_93962)
    
    # Processing the call keyword arguments (line 221)
    kwargs_93984 = {}
    # Getting the type of 'cadd' (line 221)
    cadd_93960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'cadd', False)
    # Calling cadd(args, kwargs) (line 221)
    cadd_call_result_93985 = invoke(stypy.reporting.localization.Localization(__file__, 221, 16), cadd_93960, *[result_mod_93983], **kwargs_93984)
    
    
    # Call to append(...): (line 223)
    # Processing the call arguments (line 223)
    str_93988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 29), 'str', 'char *%s')
    
    # Obtaining the type of the subscript
    str_93989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 45), 'str', 'name')
    # Getting the type of 'b' (line 223)
    b_93990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 43), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 223)
    getitem___93991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 43), b_93990, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 223)
    subscript_call_result_93992 = invoke(stypy.reporting.localization.Localization(__file__, 223, 43), getitem___93991, str_93989)
    
    # Applying the binary operator '%' (line 223)
    result_mod_93993 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 29), '%', str_93988, subscript_call_result_93992)
    
    # Processing the call keyword arguments (line 223)
    kwargs_93994 = {}
    # Getting the type of 'sargs' (line 223)
    sargs_93986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 16), 'sargs', False)
    # Obtaining the member 'append' of a type (line 223)
    append_93987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 16), sargs_93986, 'append')
    # Calling append(args, kwargs) (line 223)
    append_call_result_93995 = invoke(stypy.reporting.localization.Localization(__file__, 223, 16), append_93987, *[result_mod_93993], **kwargs_93994)
    
    
    # Call to append(...): (line 224)
    # Processing the call arguments (line 224)
    str_93998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 30), 'str', 'char *')
    # Processing the call keyword arguments (line 224)
    kwargs_93999 = {}
    # Getting the type of 'sargsp' (line 224)
    sargsp_93996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'sargsp', False)
    # Obtaining the member 'append' of a type (line 224)
    append_93997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 16), sargsp_93996, 'append')
    # Calling append(args, kwargs) (line 224)
    append_call_result_94000 = invoke(stypy.reporting.localization.Localization(__file__, 224, 16), append_93997, *[str_93998], **kwargs_93999)
    
    
    # Call to iadd(...): (line 225)
    # Processing the call arguments (line 225)
    str_94002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 21), 'str', '\tf2py_%s_def[i_f2py++].data = %s;')
    
    # Obtaining an instance of the builtin type 'tuple' (line 226)
    tuple_94003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 226)
    # Adding element type (line 226)
    
    # Obtaining the type of the subscript
    str_94004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 24), 'str', 'name')
    # Getting the type of 'm' (line 226)
    m_94005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 22), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 226)
    getitem___94006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 22), m_94005, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 226)
    subscript_call_result_94007 = invoke(stypy.reporting.localization.Localization(__file__, 226, 22), getitem___94006, str_94004)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 22), tuple_94003, subscript_call_result_94007)
    # Adding element type (line 226)
    
    # Obtaining the type of the subscript
    str_94008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 35), 'str', 'name')
    # Getting the type of 'b' (line 226)
    b_94009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 33), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 226)
    getitem___94010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 33), b_94009, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 226)
    subscript_call_result_94011 = invoke(stypy.reporting.localization.Localization(__file__, 226, 33), getitem___94010, str_94008)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 22), tuple_94003, subscript_call_result_94011)
    
    # Applying the binary operator '%' (line 225)
    result_mod_94012 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 21), '%', str_94002, tuple_94003)
    
    # Processing the call keyword arguments (line 225)
    kwargs_94013 = {}
    # Getting the type of 'iadd' (line 225)
    iadd_94001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'iadd', False)
    # Calling iadd(args, kwargs) (line 225)
    iadd_call_result_94014 = invoke(stypy.reporting.localization.Localization(__file__, 225, 16), iadd_94001, *[result_mod_94012], **kwargs_94013)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 195)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to cadd(...): (line 227)
    # Processing the call arguments (line 227)
    str_94016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 13), 'str', '\t{NULL}\n};\n')
    # Processing the call keyword arguments (line 227)
    kwargs_94017 = {}
    # Getting the type of 'cadd' (line 227)
    cadd_94015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'cadd', False)
    # Calling cadd(args, kwargs) (line 227)
    cadd_call_result_94018 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), cadd_94015, *[str_94016], **kwargs_94017)
    
    
    # Call to iadd(...): (line 228)
    # Processing the call arguments (line 228)
    str_94020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 13), 'str', '}')
    # Processing the call keyword arguments (line 228)
    kwargs_94021 = {}
    # Getting the type of 'iadd' (line 228)
    iadd_94019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'iadd', False)
    # Calling iadd(args, kwargs) (line 228)
    iadd_call_result_94022 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), iadd_94019, *[str_94020], **kwargs_94021)
    
    
    # Assigning a BinOp to a Subscript (line 229):
    
    # Assigning a BinOp to a Subscript (line 229):
    str_94023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 20), 'str', 'static void f2py_setup_%s(%s) {\n\tint i_f2py=0;%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 230)
    tuple_94024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 230)
    # Adding element type (line 230)
    
    # Obtaining the type of the subscript
    str_94025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 14), 'str', 'name')
    # Getting the type of 'm' (line 230)
    m_94026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'm')
    # Obtaining the member '__getitem__' of a type (line 230)
    getitem___94027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 12), m_94026, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 230)
    subscript_call_result_94028 = invoke(stypy.reporting.localization.Localization(__file__, 230, 12), getitem___94027, str_94025)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 12), tuple_94024, subscript_call_result_94028)
    # Adding element type (line 230)
    
    # Call to join(...): (line 230)
    # Processing the call arguments (line 230)
    # Getting the type of 'sargs' (line 230)
    sargs_94031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 32), 'sargs', False)
    # Processing the call keyword arguments (line 230)
    kwargs_94032 = {}
    str_94029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 23), 'str', ',')
    # Obtaining the member 'join' of a type (line 230)
    join_94030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 23), str_94029, 'join')
    # Calling join(args, kwargs) (line 230)
    join_call_result_94033 = invoke(stypy.reporting.localization.Localization(__file__, 230, 23), join_94030, *[sargs_94031], **kwargs_94032)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 12), tuple_94024, join_call_result_94033)
    # Adding element type (line 230)
    
    # Obtaining the type of the subscript
    int_94034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 47), 'int')
    # Getting the type of 'ihooks' (line 230)
    ihooks_94035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 40), 'ihooks')
    # Obtaining the member '__getitem__' of a type (line 230)
    getitem___94036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 40), ihooks_94035, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 230)
    subscript_call_result_94037 = invoke(stypy.reporting.localization.Localization(__file__, 230, 40), getitem___94036, int_94034)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 12), tuple_94024, subscript_call_result_94037)
    
    # Applying the binary operator '%' (line 229)
    result_mod_94038 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 20), '%', str_94023, tuple_94024)
    
    # Getting the type of 'ihooks' (line 229)
    ihooks_94039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'ihooks')
    int_94040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 15), 'int')
    # Storing an element on a container (line 229)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 8), ihooks_94039, (int_94040, result_mod_94038))
    
    
    str_94041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 11), 'str', '_')
    
    # Obtaining the type of the subscript
    str_94042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 20), 'str', 'name')
    # Getting the type of 'm' (line 231)
    m_94043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 18), 'm')
    # Obtaining the member '__getitem__' of a type (line 231)
    getitem___94044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 18), m_94043, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 231)
    subscript_call_result_94045 = invoke(stypy.reporting.localization.Localization(__file__, 231, 18), getitem___94044, str_94042)
    
    # Applying the binary operator 'in' (line 231)
    result_contains_94046 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 11), 'in', str_94041, subscript_call_result_94045)
    
    # Testing the type of an if condition (line 231)
    if_condition_94047 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 231, 8), result_contains_94046)
    # Assigning a type to the variable 'if_condition_94047' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'if_condition_94047', if_condition_94047)
    # SSA begins for if statement (line 231)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 232):
    
    # Assigning a Str to a Name (line 232):
    str_94048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 21), 'str', 'F_FUNC_US')
    # Assigning a type to the variable 'F_FUNC' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'F_FUNC', str_94048)
    # SSA branch for the else part of an if statement (line 231)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 234):
    
    # Assigning a Str to a Name (line 234):
    str_94049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 21), 'str', 'F_FUNC')
    # Assigning a type to the variable 'F_FUNC' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'F_FUNC', str_94049)
    # SSA join for if statement (line 231)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to iadd(...): (line 235)
    # Processing the call arguments (line 235)
    str_94051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 13), 'str', 'extern void %s(f2pyinit%s,F2PYINIT%s)(void (*)(%s));')
    
    # Obtaining an instance of the builtin type 'tuple' (line 236)
    tuple_94052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 236)
    # Adding element type (line 236)
    # Getting the type of 'F_FUNC' (line 236)
    F_FUNC_94053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 16), 'F_FUNC', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 16), tuple_94052, F_FUNC_94053)
    # Adding element type (line 236)
    
    # Obtaining the type of the subscript
    str_94054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 26), 'str', 'name')
    # Getting the type of 'm' (line 236)
    m_94055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 24), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 236)
    getitem___94056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 24), m_94055, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 236)
    subscript_call_result_94057 = invoke(stypy.reporting.localization.Localization(__file__, 236, 24), getitem___94056, str_94054)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 16), tuple_94052, subscript_call_result_94057)
    # Adding element type (line 236)
    
    # Call to upper(...): (line 236)
    # Processing the call keyword arguments (line 236)
    kwargs_94063 = {}
    
    # Obtaining the type of the subscript
    str_94058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 37), 'str', 'name')
    # Getting the type of 'm' (line 236)
    m_94059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 35), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 236)
    getitem___94060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 35), m_94059, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 236)
    subscript_call_result_94061 = invoke(stypy.reporting.localization.Localization(__file__, 236, 35), getitem___94060, str_94058)
    
    # Obtaining the member 'upper' of a type (line 236)
    upper_94062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 35), subscript_call_result_94061, 'upper')
    # Calling upper(args, kwargs) (line 236)
    upper_call_result_94064 = invoke(stypy.reporting.localization.Localization(__file__, 236, 35), upper_94062, *[], **kwargs_94063)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 16), tuple_94052, upper_call_result_94064)
    # Adding element type (line 236)
    
    # Call to join(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'sargsp' (line 236)
    sargsp_94067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 63), 'sargsp', False)
    # Processing the call keyword arguments (line 236)
    kwargs_94068 = {}
    str_94065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 54), 'str', ',')
    # Obtaining the member 'join' of a type (line 236)
    join_94066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 54), str_94065, 'join')
    # Calling join(args, kwargs) (line 236)
    join_call_result_94069 = invoke(stypy.reporting.localization.Localization(__file__, 236, 54), join_94066, *[sargsp_94067], **kwargs_94068)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 16), tuple_94052, join_call_result_94069)
    
    # Applying the binary operator '%' (line 235)
    result_mod_94070 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 13), '%', str_94051, tuple_94052)
    
    # Processing the call keyword arguments (line 235)
    kwargs_94071 = {}
    # Getting the type of 'iadd' (line 235)
    iadd_94050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'iadd', False)
    # Calling iadd(args, kwargs) (line 235)
    iadd_call_result_94072 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), iadd_94050, *[result_mod_94070], **kwargs_94071)
    
    
    # Call to iadd(...): (line 237)
    # Processing the call arguments (line 237)
    str_94074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 13), 'str', 'static void f2py_init_%s(void) {')
    
    # Obtaining the type of the subscript
    str_94075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 53), 'str', 'name')
    # Getting the type of 'm' (line 237)
    m_94076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 51), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 237)
    getitem___94077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 51), m_94076, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 237)
    subscript_call_result_94078 = invoke(stypy.reporting.localization.Localization(__file__, 237, 51), getitem___94077, str_94075)
    
    # Applying the binary operator '%' (line 237)
    result_mod_94079 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 13), '%', str_94074, subscript_call_result_94078)
    
    # Processing the call keyword arguments (line 237)
    kwargs_94080 = {}
    # Getting the type of 'iadd' (line 237)
    iadd_94073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'iadd', False)
    # Calling iadd(args, kwargs) (line 237)
    iadd_call_result_94081 = invoke(stypy.reporting.localization.Localization(__file__, 237, 8), iadd_94073, *[result_mod_94079], **kwargs_94080)
    
    
    # Call to iadd(...): (line 238)
    # Processing the call arguments (line 238)
    str_94083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 13), 'str', '\t%s(f2pyinit%s,F2PYINIT%s)(f2py_setup_%s);')
    
    # Obtaining an instance of the builtin type 'tuple' (line 239)
    tuple_94084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 239)
    # Adding element type (line 239)
    # Getting the type of 'F_FUNC' (line 239)
    F_FUNC_94085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 16), 'F_FUNC', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 16), tuple_94084, F_FUNC_94085)
    # Adding element type (line 239)
    
    # Obtaining the type of the subscript
    str_94086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 26), 'str', 'name')
    # Getting the type of 'm' (line 239)
    m_94087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 24), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 239)
    getitem___94088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 24), m_94087, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 239)
    subscript_call_result_94089 = invoke(stypy.reporting.localization.Localization(__file__, 239, 24), getitem___94088, str_94086)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 16), tuple_94084, subscript_call_result_94089)
    # Adding element type (line 239)
    
    # Call to upper(...): (line 239)
    # Processing the call keyword arguments (line 239)
    kwargs_94095 = {}
    
    # Obtaining the type of the subscript
    str_94090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 37), 'str', 'name')
    # Getting the type of 'm' (line 239)
    m_94091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 35), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 239)
    getitem___94092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 35), m_94091, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 239)
    subscript_call_result_94093 = invoke(stypy.reporting.localization.Localization(__file__, 239, 35), getitem___94092, str_94090)
    
    # Obtaining the member 'upper' of a type (line 239)
    upper_94094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 35), subscript_call_result_94093, 'upper')
    # Calling upper(args, kwargs) (line 239)
    upper_call_result_94096 = invoke(stypy.reporting.localization.Localization(__file__, 239, 35), upper_94094, *[], **kwargs_94095)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 16), tuple_94084, upper_call_result_94096)
    # Adding element type (line 239)
    
    # Obtaining the type of the subscript
    str_94097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 56), 'str', 'name')
    # Getting the type of 'm' (line 239)
    m_94098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 54), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 239)
    getitem___94099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 54), m_94098, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 239)
    subscript_call_result_94100 = invoke(stypy.reporting.localization.Localization(__file__, 239, 54), getitem___94099, str_94097)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 16), tuple_94084, subscript_call_result_94100)
    
    # Applying the binary operator '%' (line 238)
    result_mod_94101 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 13), '%', str_94083, tuple_94084)
    
    # Processing the call keyword arguments (line 238)
    kwargs_94102 = {}
    # Getting the type of 'iadd' (line 238)
    iadd_94082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'iadd', False)
    # Calling iadd(args, kwargs) (line 238)
    iadd_call_result_94103 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), iadd_94082, *[result_mod_94101], **kwargs_94102)
    
    
    # Call to iadd(...): (line 240)
    # Processing the call arguments (line 240)
    str_94105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 13), 'str', '}\n')
    # Processing the call keyword arguments (line 240)
    kwargs_94106 = {}
    # Getting the type of 'iadd' (line 240)
    iadd_94104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'iadd', False)
    # Calling iadd(args, kwargs) (line 240)
    iadd_call_result_94107 = invoke(stypy.reporting.localization.Localization(__file__, 240, 8), iadd_94104, *[str_94105], **kwargs_94106)
    
    
    # Assigning a BinOp to a Subscript (line 241):
    
    # Assigning a BinOp to a Subscript (line 241):
    
    # Obtaining the type of the subscript
    str_94108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 33), 'str', 'f90modhooks')
    # Getting the type of 'ret' (line 241)
    ret_94109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 29), 'ret')
    # Obtaining the member '__getitem__' of a type (line 241)
    getitem___94110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 29), ret_94109, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 241)
    subscript_call_result_94111 = invoke(stypy.reporting.localization.Localization(__file__, 241, 29), getitem___94110, str_94108)
    
    # Getting the type of 'chooks' (line 241)
    chooks_94112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 50), 'chooks')
    # Applying the binary operator '+' (line 241)
    result_add_94113 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 29), '+', subscript_call_result_94111, chooks_94112)
    
    # Getting the type of 'ihooks' (line 241)
    ihooks_94114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 59), 'ihooks')
    # Applying the binary operator '+' (line 241)
    result_add_94115 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 57), '+', result_add_94113, ihooks_94114)
    
    # Getting the type of 'ret' (line 241)
    ret_94116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'ret')
    str_94117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 12), 'str', 'f90modhooks')
    # Storing an element on a container (line 241)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 8), ret_94116, (str_94117, result_add_94115))
    
    # Assigning a BinOp to a Subscript (line 242):
    
    # Assigning a BinOp to a Subscript (line 242):
    
    # Obtaining an instance of the builtin type 'list' (line 242)
    list_94118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 242)
    # Adding element type (line 242)
    str_94119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 34), 'str', '\tPyDict_SetItemString(d, "%s", PyFortranObject_New(f2py_%s_def,f2py_init_%s));')
    
    # Obtaining an instance of the builtin type 'tuple' (line 243)
    tuple_94120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 243)
    # Adding element type (line 243)
    
    # Obtaining the type of the subscript
    str_94121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 14), 'str', 'name')
    # Getting the type of 'm' (line 243)
    m_94122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'm')
    # Obtaining the member '__getitem__' of a type (line 243)
    getitem___94123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 12), m_94122, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 243)
    subscript_call_result_94124 = invoke(stypy.reporting.localization.Localization(__file__, 243, 12), getitem___94123, str_94121)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 12), tuple_94120, subscript_call_result_94124)
    # Adding element type (line 243)
    
    # Obtaining the type of the subscript
    str_94125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 25), 'str', 'name')
    # Getting the type of 'm' (line 243)
    m_94126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 23), 'm')
    # Obtaining the member '__getitem__' of a type (line 243)
    getitem___94127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 23), m_94126, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 243)
    subscript_call_result_94128 = invoke(stypy.reporting.localization.Localization(__file__, 243, 23), getitem___94127, str_94125)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 12), tuple_94120, subscript_call_result_94128)
    # Adding element type (line 243)
    
    # Obtaining the type of the subscript
    str_94129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 36), 'str', 'name')
    # Getting the type of 'm' (line 243)
    m_94130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 34), 'm')
    # Obtaining the member '__getitem__' of a type (line 243)
    getitem___94131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 34), m_94130, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 243)
    subscript_call_result_94132 = invoke(stypy.reporting.localization.Localization(__file__, 243, 34), getitem___94131, str_94129)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 12), tuple_94120, subscript_call_result_94132)
    
    # Applying the binary operator '%' (line 242)
    result_mod_94133 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 34), '%', str_94119, tuple_94120)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 33), list_94118, result_mod_94133)
    
    
    # Obtaining the type of the subscript
    str_94134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 52), 'str', 'initf90modhooks')
    # Getting the type of 'ret' (line 243)
    ret_94135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 48), 'ret')
    # Obtaining the member '__getitem__' of a type (line 243)
    getitem___94136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 48), ret_94135, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 243)
    subscript_call_result_94137 = invoke(stypy.reporting.localization.Localization(__file__, 243, 48), getitem___94136, str_94134)
    
    # Applying the binary operator '+' (line 242)
    result_add_94138 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 33), '+', list_94118, subscript_call_result_94137)
    
    # Getting the type of 'ret' (line 242)
    ret_94139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'ret')
    str_94140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 12), 'str', 'initf90modhooks')
    # Storing an element on a container (line 242)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 8), ret_94139, (str_94140, result_add_94138))
    
    # Call to fadd(...): (line 244)
    # Processing the call arguments (line 244)
    str_94142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 13), 'str', '')
    # Processing the call keyword arguments (line 244)
    kwargs_94143 = {}
    # Getting the type of 'fadd' (line 244)
    fadd_94141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'fadd', False)
    # Calling fadd(args, kwargs) (line 244)
    fadd_call_result_94144 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), fadd_94141, *[str_94142], **kwargs_94143)
    
    
    # Call to fadd(...): (line 245)
    # Processing the call arguments (line 245)
    str_94146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 13), 'str', 'subroutine f2pyinit%s(f2pysetupfunc)')
    
    # Obtaining the type of the subscript
    str_94147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 57), 'str', 'name')
    # Getting the type of 'm' (line 245)
    m_94148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 55), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 245)
    getitem___94149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 55), m_94148, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 245)
    subscript_call_result_94150 = invoke(stypy.reporting.localization.Localization(__file__, 245, 55), getitem___94149, str_94147)
    
    # Applying the binary operator '%' (line 245)
    result_mod_94151 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 13), '%', str_94146, subscript_call_result_94150)
    
    # Processing the call keyword arguments (line 245)
    kwargs_94152 = {}
    # Getting the type of 'fadd' (line 245)
    fadd_94145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'fadd', False)
    # Calling fadd(args, kwargs) (line 245)
    fadd_call_result_94153 = invoke(stypy.reporting.localization.Localization(__file__, 245, 8), fadd_94145, *[result_mod_94151], **kwargs_94152)
    
    
    # Getting the type of 'mfargs' (line 246)
    mfargs_94154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 11), 'mfargs')
    # Testing the type of an if condition (line 246)
    if_condition_94155 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 8), mfargs_94154)
    # Assigning a type to the variable 'if_condition_94155' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'if_condition_94155', if_condition_94155)
    # SSA begins for if statement (line 246)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to undo_rmbadname(...): (line 247)
    # Processing the call arguments (line 247)
    # Getting the type of 'mfargs' (line 247)
    mfargs_94157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 36), 'mfargs', False)
    # Processing the call keyword arguments (line 247)
    kwargs_94158 = {}
    # Getting the type of 'undo_rmbadname' (line 247)
    undo_rmbadname_94156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 21), 'undo_rmbadname', False)
    # Calling undo_rmbadname(args, kwargs) (line 247)
    undo_rmbadname_call_result_94159 = invoke(stypy.reporting.localization.Localization(__file__, 247, 21), undo_rmbadname_94156, *[mfargs_94157], **kwargs_94158)
    
    # Testing the type of a for loop iterable (line 247)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 247, 12), undo_rmbadname_call_result_94159)
    # Getting the type of the for loop variable (line 247)
    for_loop_var_94160 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 247, 12), undo_rmbadname_call_result_94159)
    # Assigning a type to the variable 'a' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'a', for_loop_var_94160)
    # SSA begins for a for statement (line 247)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to fadd(...): (line 248)
    # Processing the call arguments (line 248)
    str_94162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 21), 'str', 'use %s, only : %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 248)
    tuple_94163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 248)
    # Adding element type (line 248)
    
    # Obtaining the type of the subscript
    str_94164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 46), 'str', 'name')
    # Getting the type of 'm' (line 248)
    m_94165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 44), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 248)
    getitem___94166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 44), m_94165, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 248)
    subscript_call_result_94167 = invoke(stypy.reporting.localization.Localization(__file__, 248, 44), getitem___94166, str_94164)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 44), tuple_94163, subscript_call_result_94167)
    # Adding element type (line 248)
    # Getting the type of 'a' (line 248)
    a_94168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 55), 'a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 44), tuple_94163, a_94168)
    
    # Applying the binary operator '%' (line 248)
    result_mod_94169 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 21), '%', str_94162, tuple_94163)
    
    # Processing the call keyword arguments (line 248)
    kwargs_94170 = {}
    # Getting the type of 'fadd' (line 248)
    fadd_94161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'fadd', False)
    # Calling fadd(args, kwargs) (line 248)
    fadd_call_result_94171 = invoke(stypy.reporting.localization.Localization(__file__, 248, 16), fadd_94161, *[result_mod_94169], **kwargs_94170)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 246)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'ifargs' (line 249)
    ifargs_94172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 11), 'ifargs')
    # Testing the type of an if condition (line 249)
    if_condition_94173 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 8), ifargs_94172)
    # Assigning a type to the variable 'if_condition_94173' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'if_condition_94173', if_condition_94173)
    # SSA begins for if statement (line 249)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to fadd(...): (line 250)
    # Processing the call arguments (line 250)
    
    # Call to join(...): (line 250)
    # Processing the call arguments (line 250)
    
    # Obtaining an instance of the builtin type 'list' (line 250)
    list_94177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 250)
    # Adding element type (line 250)
    str_94178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 27), 'str', 'interface')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 26), list_94177, str_94178)
    
    # Getting the type of 'ifargs' (line 250)
    ifargs_94179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 42), 'ifargs', False)
    # Applying the binary operator '+' (line 250)
    result_add_94180 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 26), '+', list_94177, ifargs_94179)
    
    # Processing the call keyword arguments (line 250)
    kwargs_94181 = {}
    str_94175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 17), 'str', ' ')
    # Obtaining the member 'join' of a type (line 250)
    join_94176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 17), str_94175, 'join')
    # Calling join(args, kwargs) (line 250)
    join_call_result_94182 = invoke(stypy.reporting.localization.Localization(__file__, 250, 17), join_94176, *[result_add_94180], **kwargs_94181)
    
    # Processing the call keyword arguments (line 250)
    kwargs_94183 = {}
    # Getting the type of 'fadd' (line 250)
    fadd_94174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'fadd', False)
    # Calling fadd(args, kwargs) (line 250)
    fadd_call_result_94184 = invoke(stypy.reporting.localization.Localization(__file__, 250, 12), fadd_94174, *[join_call_result_94182], **kwargs_94183)
    
    
    # Call to fadd(...): (line 251)
    # Processing the call arguments (line 251)
    str_94186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 17), 'str', 'end interface')
    # Processing the call keyword arguments (line 251)
    kwargs_94187 = {}
    # Getting the type of 'fadd' (line 251)
    fadd_94185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'fadd', False)
    # Calling fadd(args, kwargs) (line 251)
    fadd_call_result_94188 = invoke(stypy.reporting.localization.Localization(__file__, 251, 12), fadd_94185, *[str_94186], **kwargs_94187)
    
    # SSA join for if statement (line 249)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to fadd(...): (line 252)
    # Processing the call arguments (line 252)
    str_94190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 13), 'str', 'external f2pysetupfunc')
    # Processing the call keyword arguments (line 252)
    kwargs_94191 = {}
    # Getting the type of 'fadd' (line 252)
    fadd_94189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'fadd', False)
    # Calling fadd(args, kwargs) (line 252)
    fadd_call_result_94192 = invoke(stypy.reporting.localization.Localization(__file__, 252, 8), fadd_94189, *[str_94190], **kwargs_94191)
    
    
    # Getting the type of 'efargs' (line 253)
    efargs_94193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 11), 'efargs')
    # Testing the type of an if condition (line 253)
    if_condition_94194 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 253, 8), efargs_94193)
    # Assigning a type to the variable 'if_condition_94194' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'if_condition_94194', if_condition_94194)
    # SSA begins for if statement (line 253)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to undo_rmbadname(...): (line 254)
    # Processing the call arguments (line 254)
    # Getting the type of 'efargs' (line 254)
    efargs_94196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 36), 'efargs', False)
    # Processing the call keyword arguments (line 254)
    kwargs_94197 = {}
    # Getting the type of 'undo_rmbadname' (line 254)
    undo_rmbadname_94195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 21), 'undo_rmbadname', False)
    # Calling undo_rmbadname(args, kwargs) (line 254)
    undo_rmbadname_call_result_94198 = invoke(stypy.reporting.localization.Localization(__file__, 254, 21), undo_rmbadname_94195, *[efargs_94196], **kwargs_94197)
    
    # Testing the type of a for loop iterable (line 254)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 254, 12), undo_rmbadname_call_result_94198)
    # Getting the type of the for loop variable (line 254)
    for_loop_var_94199 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 254, 12), undo_rmbadname_call_result_94198)
    # Assigning a type to the variable 'a' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'a', for_loop_var_94199)
    # SSA begins for a for statement (line 254)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to fadd(...): (line 255)
    # Processing the call arguments (line 255)
    str_94201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 21), 'str', 'external %s')
    # Getting the type of 'a' (line 255)
    a_94202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 38), 'a', False)
    # Applying the binary operator '%' (line 255)
    result_mod_94203 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 21), '%', str_94201, a_94202)
    
    # Processing the call keyword arguments (line 255)
    kwargs_94204 = {}
    # Getting the type of 'fadd' (line 255)
    fadd_94200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 16), 'fadd', False)
    # Calling fadd(args, kwargs) (line 255)
    fadd_call_result_94205 = invoke(stypy.reporting.localization.Localization(__file__, 255, 16), fadd_94200, *[result_mod_94203], **kwargs_94204)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 253)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to fadd(...): (line 256)
    # Processing the call arguments (line 256)
    str_94207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 13), 'str', 'call f2pysetupfunc(%s)')
    
    # Call to join(...): (line 256)
    # Processing the call arguments (line 256)
    
    # Call to undo_rmbadname(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'fargs' (line 256)
    fargs_94211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 65), 'fargs', False)
    # Processing the call keyword arguments (line 256)
    kwargs_94212 = {}
    # Getting the type of 'undo_rmbadname' (line 256)
    undo_rmbadname_94210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 50), 'undo_rmbadname', False)
    # Calling undo_rmbadname(args, kwargs) (line 256)
    undo_rmbadname_call_result_94213 = invoke(stypy.reporting.localization.Localization(__file__, 256, 50), undo_rmbadname_94210, *[fargs_94211], **kwargs_94212)
    
    # Processing the call keyword arguments (line 256)
    kwargs_94214 = {}
    str_94208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 41), 'str', ',')
    # Obtaining the member 'join' of a type (line 256)
    join_94209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 41), str_94208, 'join')
    # Calling join(args, kwargs) (line 256)
    join_call_result_94215 = invoke(stypy.reporting.localization.Localization(__file__, 256, 41), join_94209, *[undo_rmbadname_call_result_94213], **kwargs_94214)
    
    # Applying the binary operator '%' (line 256)
    result_mod_94216 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 13), '%', str_94207, join_call_result_94215)
    
    # Processing the call keyword arguments (line 256)
    kwargs_94217 = {}
    # Getting the type of 'fadd' (line 256)
    fadd_94206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'fadd', False)
    # Calling fadd(args, kwargs) (line 256)
    fadd_call_result_94218 = invoke(stypy.reporting.localization.Localization(__file__, 256, 8), fadd_94206, *[result_mod_94216], **kwargs_94217)
    
    
    # Call to fadd(...): (line 257)
    # Processing the call arguments (line 257)
    str_94220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 13), 'str', 'end subroutine f2pyinit%s\n')
    
    # Obtaining the type of the subscript
    str_94221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 48), 'str', 'name')
    # Getting the type of 'm' (line 257)
    m_94222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 46), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 257)
    getitem___94223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 46), m_94222, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 257)
    subscript_call_result_94224 = invoke(stypy.reporting.localization.Localization(__file__, 257, 46), getitem___94223, str_94221)
    
    # Applying the binary operator '%' (line 257)
    result_mod_94225 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 13), '%', str_94220, subscript_call_result_94224)
    
    # Processing the call keyword arguments (line 257)
    kwargs_94226 = {}
    # Getting the type of 'fadd' (line 257)
    fadd_94219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'fadd', False)
    # Calling fadd(args, kwargs) (line 257)
    fadd_call_result_94227 = invoke(stypy.reporting.localization.Localization(__file__, 257, 8), fadd_94219, *[result_mod_94225], **kwargs_94226)
    
    
    # Call to dadd(...): (line 259)
    # Processing the call arguments (line 259)
    
    # Call to replace(...): (line 259)
    # Processing the call arguments (line 259)
    str_94238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 12), 'str', '\\subsection{')
    str_94239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 29), 'str', '\\subsubsection{')
    # Processing the call keyword arguments (line 259)
    kwargs_94240 = {}
    
    # Call to join(...): (line 259)
    # Processing the call arguments (line 259)
    
    # Obtaining the type of the subscript
    str_94231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 27), 'str', 'latexdoc')
    # Getting the type of 'ret' (line 259)
    ret_94232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 23), 'ret', False)
    # Obtaining the member '__getitem__' of a type (line 259)
    getitem___94233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 23), ret_94232, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 259)
    subscript_call_result_94234 = invoke(stypy.reporting.localization.Localization(__file__, 259, 23), getitem___94233, str_94231)
    
    # Processing the call keyword arguments (line 259)
    kwargs_94235 = {}
    str_94229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 13), 'str', '\n')
    # Obtaining the member 'join' of a type (line 259)
    join_94230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 13), str_94229, 'join')
    # Calling join(args, kwargs) (line 259)
    join_call_result_94236 = invoke(stypy.reporting.localization.Localization(__file__, 259, 13), join_94230, *[subscript_call_result_94234], **kwargs_94235)
    
    # Obtaining the member 'replace' of a type (line 259)
    replace_94237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 13), join_call_result_94236, 'replace')
    # Calling replace(args, kwargs) (line 259)
    replace_call_result_94241 = invoke(stypy.reporting.localization.Localization(__file__, 259, 13), replace_94237, *[str_94238, str_94239], **kwargs_94240)
    
    # Processing the call keyword arguments (line 259)
    kwargs_94242 = {}
    # Getting the type of 'dadd' (line 259)
    dadd_94228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'dadd', False)
    # Calling dadd(args, kwargs) (line 259)
    dadd_call_result_94243 = invoke(stypy.reporting.localization.Localization(__file__, 259, 8), dadd_94228, *[replace_call_result_94241], **kwargs_94242)
    
    
    # Assigning a List to a Subscript (line 262):
    
    # Assigning a List to a Subscript (line 262):
    
    # Obtaining an instance of the builtin type 'list' (line 262)
    list_94244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 262)
    
    # Getting the type of 'ret' (line 262)
    ret_94245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'ret')
    str_94246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 12), 'str', 'latexdoc')
    # Storing an element on a container (line 262)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 8), ret_94245, (str_94246, list_94244))
    
    # Call to append(...): (line 263)
    # Processing the call arguments (line 263)
    str_94252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 27), 'str', '"\t%s --- %s"')
    
    # Obtaining an instance of the builtin type 'tuple' (line 263)
    tuple_94253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 263)
    # Adding element type (line 263)
    
    # Obtaining the type of the subscript
    str_94254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 48), 'str', 'name')
    # Getting the type of 'm' (line 263)
    m_94255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 46), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 263)
    getitem___94256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 46), m_94255, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 263)
    subscript_call_result_94257 = invoke(stypy.reporting.localization.Localization(__file__, 263, 46), getitem___94256, str_94254)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 46), tuple_94253, subscript_call_result_94257)
    # Adding element type (line 263)
    
    # Call to join(...): (line 264)
    # Processing the call arguments (line 264)
    
    # Call to undo_rmbadname(...): (line 264)
    # Processing the call arguments (line 264)
    # Getting the type of 'modobjs' (line 264)
    modobjs_94261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 70), 'modobjs', False)
    # Processing the call keyword arguments (line 264)
    kwargs_94262 = {}
    # Getting the type of 'undo_rmbadname' (line 264)
    undo_rmbadname_94260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 55), 'undo_rmbadname', False)
    # Calling undo_rmbadname(args, kwargs) (line 264)
    undo_rmbadname_call_result_94263 = invoke(stypy.reporting.localization.Localization(__file__, 264, 55), undo_rmbadname_94260, *[modobjs_94261], **kwargs_94262)
    
    # Processing the call keyword arguments (line 264)
    kwargs_94264 = {}
    str_94258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 46), 'str', ',')
    # Obtaining the member 'join' of a type (line 264)
    join_94259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 46), str_94258, 'join')
    # Calling join(args, kwargs) (line 264)
    join_call_result_94265 = invoke(stypy.reporting.localization.Localization(__file__, 264, 46), join_94259, *[undo_rmbadname_call_result_94263], **kwargs_94264)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 46), tuple_94253, join_call_result_94265)
    
    # Applying the binary operator '%' (line 263)
    result_mod_94266 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 27), '%', str_94252, tuple_94253)
    
    # Processing the call keyword arguments (line 263)
    kwargs_94267 = {}
    
    # Obtaining the type of the subscript
    str_94247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 12), 'str', 'docs')
    # Getting the type of 'ret' (line 263)
    ret_94248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'ret', False)
    # Obtaining the member '__getitem__' of a type (line 263)
    getitem___94249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 8), ret_94248, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 263)
    subscript_call_result_94250 = invoke(stypy.reporting.localization.Localization(__file__, 263, 8), getitem___94249, str_94247)
    
    # Obtaining the member 'append' of a type (line 263)
    append_94251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 8), subscript_call_result_94250, 'append')
    # Calling append(args, kwargs) (line 263)
    append_call_result_94268 = invoke(stypy.reporting.localization.Localization(__file__, 263, 8), append_94251, *[result_mod_94266], **kwargs_94267)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Str to a Subscript (line 266):
    
    # Assigning a Str to a Subscript (line 266):
    str_94269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 26), 'str', '')
    # Getting the type of 'ret' (line 266)
    ret_94270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'ret')
    str_94271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 8), 'str', 'routine_defs')
    # Storing an element on a container (line 266)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 4), ret_94270, (str_94271, str_94269))
    
    # Assigning a List to a Subscript (line 267):
    
    # Assigning a List to a Subscript (line 267):
    
    # Obtaining an instance of the builtin type 'list' (line 267)
    list_94272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 267)
    
    # Getting the type of 'ret' (line 267)
    ret_94273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'ret')
    str_94274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 8), 'str', 'doc')
    # Storing an element on a container (line 267)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 4), ret_94273, (str_94274, list_94272))
    
    # Assigning a List to a Subscript (line 268):
    
    # Assigning a List to a Subscript (line 268):
    
    # Obtaining an instance of the builtin type 'list' (line 268)
    list_94275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 268)
    
    # Getting the type of 'ret' (line 268)
    ret_94276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'ret')
    str_94277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 8), 'str', 'docshort')
    # Storing an element on a container (line 268)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 4), ret_94276, (str_94277, list_94275))
    
    # Assigning a Subscript to a Subscript (line 269):
    
    # Assigning a Subscript to a Subscript (line 269):
    
    # Obtaining the type of the subscript
    int_94278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 26), 'int')
    # Getting the type of 'doc' (line 269)
    doc_94279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 22), 'doc')
    # Obtaining the member '__getitem__' of a type (line 269)
    getitem___94280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 22), doc_94279, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 269)
    subscript_call_result_94281 = invoke(stypy.reporting.localization.Localization(__file__, 269, 22), getitem___94280, int_94278)
    
    # Getting the type of 'ret' (line 269)
    ret_94282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'ret')
    str_94283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 8), 'str', 'latexdoc')
    # Storing an element on a container (line 269)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 4), ret_94282, (str_94283, subscript_call_result_94281))
    
    
    
    # Call to len(...): (line 270)
    # Processing the call arguments (line 270)
    
    # Obtaining the type of the subscript
    str_94285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 15), 'str', 'docs')
    # Getting the type of 'ret' (line 270)
    ret_94286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 11), 'ret', False)
    # Obtaining the member '__getitem__' of a type (line 270)
    getitem___94287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 11), ret_94286, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 270)
    subscript_call_result_94288 = invoke(stypy.reporting.localization.Localization(__file__, 270, 11), getitem___94287, str_94285)
    
    # Processing the call keyword arguments (line 270)
    kwargs_94289 = {}
    # Getting the type of 'len' (line 270)
    len_94284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 7), 'len', False)
    # Calling len(args, kwargs) (line 270)
    len_call_result_94290 = invoke(stypy.reporting.localization.Localization(__file__, 270, 7), len_94284, *[subscript_call_result_94288], **kwargs_94289)
    
    int_94291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 27), 'int')
    # Applying the binary operator '<=' (line 270)
    result_le_94292 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 7), '<=', len_call_result_94290, int_94291)
    
    # Testing the type of an if condition (line 270)
    if_condition_94293 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 270, 4), result_le_94292)
    # Assigning a type to the variable 'if_condition_94293' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'if_condition_94293', if_condition_94293)
    # SSA begins for if statement (line 270)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Subscript (line 271):
    
    # Assigning a Str to a Subscript (line 271):
    str_94294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 22), 'str', '')
    # Getting the type of 'ret' (line 271)
    ret_94295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'ret')
    str_94296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 12), 'str', 'docs')
    # Storing an element on a container (line 271)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 8), ret_94295, (str_94296, str_94294))
    # SSA join for if statement (line 270)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 272)
    tuple_94297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 272)
    # Adding element type (line 272)
    # Getting the type of 'ret' (line 272)
    ret_94298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 11), 'ret')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 11), tuple_94297, ret_94298)
    # Adding element type (line 272)
    
    # Obtaining the type of the subscript
    int_94299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 23), 'int')
    # Getting the type of 'fhooks' (line 272)
    fhooks_94300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 16), 'fhooks')
    # Obtaining the member '__getitem__' of a type (line 272)
    getitem___94301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 16), fhooks_94300, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 272)
    subscript_call_result_94302 = invoke(stypy.reporting.localization.Localization(__file__, 272, 16), getitem___94301, int_94299)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 11), tuple_94297, subscript_call_result_94302)
    
    # Assigning a type to the variable 'stypy_return_type' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'stypy_return_type', tuple_94297)
    
    # ################# End of 'buildhooks(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'buildhooks' in the type store
    # Getting the type of 'stypy_return_type' (line 89)
    stypy_return_type_94303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_94303)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'buildhooks'
    return stypy_return_type_94303

# Assigning a type to the variable 'buildhooks' (line 89)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), 'buildhooks', buildhooks)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
