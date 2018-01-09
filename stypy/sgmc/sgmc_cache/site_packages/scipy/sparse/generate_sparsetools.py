
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: python generate_sparsetools.py
3: 
4: Generate manual wrappers for C++ sparsetools code.
5: 
6: Type codes used:
7: 
8:     'i':  integer scalar
9:     'I':  integer array
10:     'T':  data array
11:     'B':  boolean array
12:     'V':  std::vector<integer>*
13:     'W':  std::vector<data>*
14:     '*':  indicates that the next argument is an output argument
15:     'v':  void
16: 
17: See sparsetools.cxx for more details.
18: 
19: '''
20: import optparse
21: import os
22: from distutils.dep_util import newer
23: 
24: #
25: # List of all routines and their argument types.
26: #
27: # The first code indicates the return value, the rest the arguments.
28: #
29: 
30: # bsr.h
31: BSR_ROUTINES = '''
32: bsr_diagonal        v iiiiiIIT*T
33: bsr_scale_rows      v iiiiII*TT
34: bsr_scale_columns   v iiiiII*TT
35: bsr_sort_indices    v iiii*I*I*T
36: bsr_transpose       v iiiiIIT*I*I*T
37: bsr_matmat_pass2    v iiiiiIITIIT*I*I*T
38: bsr_matvec          v iiiiIITT*T
39: bsr_matvecs         v iiiiiIITT*T
40: bsr_elmul_bsr       v iiiiIITIIT*I*I*T
41: bsr_eldiv_bsr       v iiiiIITIIT*I*I*T
42: bsr_plus_bsr        v iiiiIITIIT*I*I*T
43: bsr_minus_bsr       v iiiiIITIIT*I*I*T
44: bsr_maximum_bsr     v iiiiIITIIT*I*I*T
45: bsr_minimum_bsr     v iiiiIITIIT*I*I*T
46: bsr_ne_bsr          v iiiiIITIIT*I*I*B
47: bsr_lt_bsr          v iiiiIITIIT*I*I*B
48: bsr_gt_bsr          v iiiiIITIIT*I*I*B
49: bsr_le_bsr          v iiiiIITIIT*I*I*B
50: bsr_ge_bsr          v iiiiIITIIT*I*I*B
51: '''
52: 
53: # csc.h
54: CSC_ROUTINES = '''
55: csc_diagonal        v iiiIIT*T
56: csc_tocsr           v iiIIT*I*I*T
57: csc_matmat_pass1    v iiIIII*I
58: csc_matmat_pass2    v iiIITIIT*I*I*T
59: csc_matvec          v iiIITT*T
60: csc_matvecs         v iiiIITT*T
61: csc_elmul_csc       v iiIITIIT*I*I*T
62: csc_eldiv_csc       v iiIITIIT*I*I*T
63: csc_plus_csc        v iiIITIIT*I*I*T
64: csc_minus_csc       v iiIITIIT*I*I*T
65: csc_maximum_csc     v iiIITIIT*I*I*T
66: csc_minimum_csc     v iiIITIIT*I*I*T
67: csc_ne_csc          v iiIITIIT*I*I*B
68: csc_lt_csc          v iiIITIIT*I*I*B
69: csc_gt_csc          v iiIITIIT*I*I*B
70: csc_le_csc          v iiIITIIT*I*I*B
71: csc_ge_csc          v iiIITIIT*I*I*B
72: '''
73: 
74: # csr.h
75: CSR_ROUTINES = '''
76: csr_matmat_pass1    v iiIIII*I
77: csr_matmat_pass2    v iiIITIIT*I*I*T
78: csr_diagonal        v iiiIIT*T
79: csr_tocsc           v iiIIT*I*I*T
80: csr_tobsr           v iiiiIIT*I*I*T
81: csr_todense         v iiIIT*T
82: csr_matvec          v iiIITT*T
83: csr_matvecs         v iiiIITT*T
84: csr_elmul_csr       v iiIITIIT*I*I*T
85: csr_eldiv_csr       v iiIITIIT*I*I*T
86: csr_plus_csr        v iiIITIIT*I*I*T
87: csr_minus_csr       v iiIITIIT*I*I*T
88: csr_maximum_csr     v iiIITIIT*I*I*T
89: csr_minimum_csr     v iiIITIIT*I*I*T
90: csr_ne_csr          v iiIITIIT*I*I*B
91: csr_lt_csr          v iiIITIIT*I*I*B
92: csr_gt_csr          v iiIITIIT*I*I*B
93: csr_le_csr          v iiIITIIT*I*I*B
94: csr_ge_csr          v iiIITIIT*I*I*B
95: csr_scale_rows      v iiII*TT
96: csr_scale_columns   v iiII*TT
97: csr_sort_indices    v iI*I*T
98: csr_eliminate_zeros v ii*I*I*T
99: csr_sum_duplicates  v ii*I*I*T
100: get_csr_submatrix   v iiIITiiii*V*V*W
101: csr_sample_values   v iiIITiII*T
102: csr_count_blocks    i iiiiII
103: csr_sample_offsets  i iiIIiII*I
104: expandptr           v iI*I
105: test_throw_error    i
106: csr_has_sorted_indices    i iII
107: csr_has_canonical_format  i iII
108: '''
109: 
110: # coo.h, dia.h, csgraph.h
111: OTHER_ROUTINES = '''
112: coo_tocsr           v iiiIIT*I*I*T
113: coo_todense         v iiiIIT*Ti
114: coo_matvec          v iIITT*T
115: dia_matvec          v iiiiITT*T
116: cs_graph_components i iII*I
117: '''
118: 
119: # List of compilation units
120: COMPILATION_UNITS = [
121:     ('bsr', BSR_ROUTINES),
122:     ('csr', CSR_ROUTINES),
123:     ('csc', CSC_ROUTINES),
124:     ('other', OTHER_ROUTINES),
125: ]
126: 
127: #
128: # List of the supported index typenums and the corresponding C++ types
129: #
130: I_TYPES = [
131:     ('NPY_INT32', 'npy_int32'),
132:     ('NPY_INT64', 'npy_int64'),
133: ]
134: 
135: #
136: # List of the supported data typenums and the corresponding C++ types
137: #
138: T_TYPES = [
139:     ('NPY_BOOL', 'npy_bool_wrapper'),
140:     ('NPY_BYTE', 'npy_byte'),
141:     ('NPY_UBYTE', 'npy_ubyte'),
142:     ('NPY_SHORT', 'npy_short'),
143:     ('NPY_USHORT', 'npy_ushort'),
144:     ('NPY_INT', 'npy_int'),
145:     ('NPY_UINT', 'npy_uint'),
146:     ('NPY_LONG', 'npy_long'),
147:     ('NPY_ULONG', 'npy_ulong'),
148:     ('NPY_LONGLONG', 'npy_longlong'),
149:     ('NPY_ULONGLONG', 'npy_ulonglong'),
150:     ('NPY_FLOAT', 'npy_float'),
151:     ('NPY_DOUBLE', 'npy_double'),
152:     ('NPY_LONGDOUBLE', 'npy_longdouble'),
153:     ('NPY_CFLOAT', 'npy_cfloat_wrapper'),
154:     ('NPY_CDOUBLE', 'npy_cdouble_wrapper'),
155:     ('NPY_CLONGDOUBLE', 'npy_clongdouble_wrapper'),
156: ]
157: 
158: #
159: # Code templates
160: #
161: 
162: THUNK_TEMPLATE = '''
163: static Py_ssize_t %(name)s_thunk(int I_typenum, int T_typenum, void **a)
164: {
165:     %(thunk_content)s
166: }
167: '''
168: 
169: METHOD_TEMPLATE = '''
170: NPY_VISIBILITY_HIDDEN PyObject *
171: %(name)s_method(PyObject *self, PyObject *args)
172: {
173:     return call_thunk('%(ret_spec)s', "%(arg_spec)s", %(name)s_thunk, args);
174: }
175: '''
176: 
177: GET_THUNK_CASE_TEMPLATE = '''
178: static int get_thunk_case(int I_typenum, int T_typenum)
179: {
180:     %(content)s;
181:     return -1;
182: }
183: '''
184: 
185: 
186: #
187: # Code generation
188: #
189: 
190: def get_thunk_type_set():
191:     '''
192:     Get a list containing cartesian product of data types, plus a getter routine.
193: 
194:     Returns
195:     -------
196:     i_types : list [(j, I_typenum, None, I_type, None), ...]
197:          Pairing of index type numbers and the corresponding C++ types,
198:          and an unique index `j`. This is for routines that are parameterized
199:          only by I but not by T.
200:     it_types : list [(j, I_typenum, T_typenum, I_type, T_type), ...]
201:          Same as `i_types`, but for routines parameterized both by T and I.
202:     getter_code : str
203:          C++ code for a function that takes I_typenum, T_typenum and returns
204:          the unique index corresponding to the lists, or -1 if no match was
205:          found.
206: 
207:     '''
208:     it_types = []
209:     i_types = []
210: 
211:     j = 0
212: 
213:     getter_code = "    if (0) {}"
214: 
215:     for I_typenum, I_type in I_TYPES:
216:         piece = '''
217:         else if (I_typenum == %(I_typenum)s) {
218:             if (T_typenum == -1) { return %(j)s; }'''
219:         getter_code += piece % dict(I_typenum=I_typenum, j=j)
220: 
221:         i_types.append((j, I_typenum, None, I_type, None))
222:         j += 1
223: 
224:         for T_typenum, T_type in T_TYPES:
225:             piece = '''
226:             else if (T_typenum == %(T_typenum)s) { return %(j)s; }'''
227:             getter_code += piece % dict(T_typenum=T_typenum, j=j)
228: 
229:             it_types.append((j, I_typenum, T_typenum, I_type, T_type))
230:             j += 1
231: 
232:         getter_code += '''
233:         }'''
234: 
235:     return i_types, it_types, GET_THUNK_CASE_TEMPLATE % dict(content=getter_code)
236: 
237: 
238: def parse_routine(name, args, types):
239:     '''
240:     Generate thunk and method code for a given routine.
241: 
242:     Parameters
243:     ----------
244:     name : str
245:         Name of the C++ routine
246:     args : str
247:         Argument list specification (in format explained above)
248:     types : list
249:         List of types to instantiate, as returned `get_thunk_type_set`
250: 
251:     '''
252: 
253:     ret_spec = args[0]
254:     arg_spec = args[1:]
255: 
256:     def get_arglist(I_type, T_type):
257:         '''
258:         Generate argument list for calling the C++ function
259:         '''
260:         args = []
261:         next_is_writeable = False
262:         j = 0
263:         for t in arg_spec:
264:             const = '' if next_is_writeable else 'const '
265:             next_is_writeable = False
266:             if t == '*':
267:                 next_is_writeable = True
268:                 continue
269:             elif t == 'i':
270:                 args.append("*(%s*)a[%d]" % (const + I_type, j))
271:             elif t == 'I':
272:                 args.append("(%s*)a[%d]" % (const + I_type, j))
273:             elif t == 'T':
274:                 args.append("(%s*)a[%d]" % (const + T_type, j))
275:             elif t == 'B':
276:                 args.append("(npy_bool_wrapper*)a[%d]" % (j,))
277:             elif t == 'V':
278:                 if const:
279:                     raise ValueError("'V' argument must be an output arg")
280:                 args.append("(std::vector<%s>*)a[%d]" % (I_type, j,))
281:             elif t == 'W':
282:                 if const:
283:                     raise ValueError("'W' argument must be an output arg")
284:                 args.append("(std::vector<%s>*)a[%d]" % (T_type, j,))
285:             else:
286:                 raise ValueError("Invalid spec character %r" % (t,))
287:             j += 1
288:         return ", ".join(args)
289: 
290:     # Generate thunk code: a giant switch statement with different
291:     # type combinations inside.
292:     thunk_content = '''int j = get_thunk_case(I_typenum, T_typenum);
293:     switch (j) {'''
294:     for j, I_typenum, T_typenum, I_type, T_type in types:
295:         arglist = get_arglist(I_type, T_type)
296:         if T_type is None:
297:             dispatch = "%s" % (I_type,)
298:         else:
299:             dispatch = "%s,%s" % (I_type, T_type)
300:         if 'B' in arg_spec:
301:             dispatch += ",npy_bool_wrapper"
302: 
303:         piece = '''
304:         case %(j)s:'''
305:         if ret_spec == 'v':
306:             piece += '''
307:             (void)%(name)s<%(dispatch)s>(%(arglist)s);
308:             return 0;'''
309:         else:
310:             piece += '''
311:             return %(name)s<%(dispatch)s>(%(arglist)s);'''
312:         thunk_content += piece % dict(j=j, I_type=I_type, T_type=T_type,
313:                                       I_typenum=I_typenum, T_typenum=T_typenum,
314:                                       arglist=arglist, name=name,
315:                                       dispatch=dispatch)
316: 
317:     thunk_content += '''
318:     default:
319:         throw std::runtime_error("internal error: invalid argument typenums");
320:     }'''
321: 
322:     thunk_code = THUNK_TEMPLATE % dict(name=name,
323:                                        thunk_content=thunk_content)
324: 
325:     # Generate method code
326:     method_code = METHOD_TEMPLATE % dict(name=name,
327:                                          ret_spec=ret_spec,
328:                                          arg_spec=arg_spec)
329: 
330:     return thunk_code, method_code
331: 
332: 
333: def main():
334:     p = optparse.OptionParser(usage=__doc__.strip())
335:     p.add_option("--no-force", action="store_false",
336:                  dest="force", default=True)
337:     options, args = p.parse_args()
338: 
339:     names = []
340: 
341:     i_types, it_types, getter_code = get_thunk_type_set()
342: 
343:     # Generate *_impl.h for each compilation unit
344:     for unit_name, routines in COMPILATION_UNITS:
345:         thunks = []
346:         methods = []
347: 
348:         # Generate thunks and methods for all routines
349:         for line in routines.splitlines():
350:             line = line.strip()
351:             if not line or line.startswith('#'):
352:                 continue
353: 
354:             try:
355:                 name, args = line.split(None, 1)
356:             except ValueError:
357:                 raise ValueError("Malformed line: %r" % (line,))
358: 
359:             args = "".join(args.split())
360:             if 't' in args or 'T' in args:
361:                 thunk, method = parse_routine(name, args, it_types)
362:             else:
363:                 thunk, method = parse_routine(name, args, i_types)
364: 
365:             if name in names:
366:                 raise ValueError("Duplicate routine %r" % (name,))
367: 
368:             names.append(name)
369:             thunks.append(thunk)
370:             methods.append(method)
371: 
372:         # Produce output
373:         dst = os.path.join(os.path.dirname(__file__),
374:                            'sparsetools',
375:                            unit_name + '_impl.h')
376:         if newer(__file__, dst) or options.force:
377:             print("[generate_sparsetools] generating %r" % (dst,))
378:             with open(dst, 'w') as f:
379:                 write_autogen_blurb(f)
380:                 f.write(getter_code)
381:                 for thunk in thunks:
382:                     f.write(thunk)
383:                 for method in methods:
384:                     f.write(method)
385:         else:
386:             print("[generate_sparsetools] %r already up-to-date" % (dst,))
387: 
388:     # Generate code for method struct
389:     method_defs = ""
390:     for name in names:
391:         method_defs += "NPY_VISIBILITY_HIDDEN PyObject *%s_method(PyObject *, PyObject *);\n" % (name,)
392: 
393:     method_struct = '''\nstatic struct PyMethodDef sparsetools_methods[] = {'''
394:     for name in names:
395:         method_struct += '''
396:         {"%(name)s", (PyCFunction)%(name)s_method, METH_VARARGS, NULL},''' % dict(name=name)
397:     method_struct += '''
398:         {NULL, NULL, 0, NULL}
399:     };'''
400: 
401:     # Produce sparsetools_impl.h
402:     dst = os.path.join(os.path.dirname(__file__),
403:                        'sparsetools',
404:                        'sparsetools_impl.h')
405: 
406:     if newer(__file__, dst) or options.force:
407:         print("[generate_sparsetools] generating %r" % (dst,))
408:         with open(dst, 'w') as f:
409:             write_autogen_blurb(f)
410:             f.write(method_defs)
411:             f.write(method_struct)
412:     else:
413:         print("[generate_sparsetools] %r already up-to-date" % (dst,))
414: 
415: 
416: def write_autogen_blurb(stream):
417:     stream.write('''\
418: /* This file is autogenerated by generate_sparsetools.py
419:  * Do not edit manually or check into VCS.
420:  */
421: ''')
422: 
423: 
424: if __name__ == "__main__":
425:     main()
426: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_376727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, (-1)), 'str', "\npython generate_sparsetools.py\n\nGenerate manual wrappers for C++ sparsetools code.\n\nType codes used:\n\n    'i':  integer scalar\n    'I':  integer array\n    'T':  data array\n    'B':  boolean array\n    'V':  std::vector<integer>*\n    'W':  std::vector<data>*\n    '*':  indicates that the next argument is an output argument\n    'v':  void\n\nSee sparsetools.cxx for more details.\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import optparse' statement (line 20)
import optparse

import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'optparse', optparse, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'import os' statement (line 21)
import os

import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from distutils.dep_util import newer' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_376728 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.dep_util')

if (type(import_376728) is not StypyTypeError):

    if (import_376728 != 'pyd_module'):
        __import__(import_376728)
        sys_modules_376729 = sys.modules[import_376728]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.dep_util', sys_modules_376729.module_type_store, module_type_store, ['newer'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_376729, sys_modules_376729.module_type_store, module_type_store)
    else:
        from distutils.dep_util import newer

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.dep_util', None, module_type_store, ['newer'], [newer])

else:
    # Assigning a type to the variable 'distutils.dep_util' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.dep_util', import_376728)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')


# Assigning a Str to a Name (line 31):

# Assigning a Str to a Name (line 31):
str_376730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, (-1)), 'str', '\nbsr_diagonal        v iiiiiIIT*T\nbsr_scale_rows      v iiiiII*TT\nbsr_scale_columns   v iiiiII*TT\nbsr_sort_indices    v iiii*I*I*T\nbsr_transpose       v iiiiIIT*I*I*T\nbsr_matmat_pass2    v iiiiiIITIIT*I*I*T\nbsr_matvec          v iiiiIITT*T\nbsr_matvecs         v iiiiiIITT*T\nbsr_elmul_bsr       v iiiiIITIIT*I*I*T\nbsr_eldiv_bsr       v iiiiIITIIT*I*I*T\nbsr_plus_bsr        v iiiiIITIIT*I*I*T\nbsr_minus_bsr       v iiiiIITIIT*I*I*T\nbsr_maximum_bsr     v iiiiIITIIT*I*I*T\nbsr_minimum_bsr     v iiiiIITIIT*I*I*T\nbsr_ne_bsr          v iiiiIITIIT*I*I*B\nbsr_lt_bsr          v iiiiIITIIT*I*I*B\nbsr_gt_bsr          v iiiiIITIIT*I*I*B\nbsr_le_bsr          v iiiiIITIIT*I*I*B\nbsr_ge_bsr          v iiiiIITIIT*I*I*B\n')
# Assigning a type to the variable 'BSR_ROUTINES' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'BSR_ROUTINES', str_376730)

# Assigning a Str to a Name (line 54):

# Assigning a Str to a Name (line 54):
str_376731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, (-1)), 'str', '\ncsc_diagonal        v iiiIIT*T\ncsc_tocsr           v iiIIT*I*I*T\ncsc_matmat_pass1    v iiIIII*I\ncsc_matmat_pass2    v iiIITIIT*I*I*T\ncsc_matvec          v iiIITT*T\ncsc_matvecs         v iiiIITT*T\ncsc_elmul_csc       v iiIITIIT*I*I*T\ncsc_eldiv_csc       v iiIITIIT*I*I*T\ncsc_plus_csc        v iiIITIIT*I*I*T\ncsc_minus_csc       v iiIITIIT*I*I*T\ncsc_maximum_csc     v iiIITIIT*I*I*T\ncsc_minimum_csc     v iiIITIIT*I*I*T\ncsc_ne_csc          v iiIITIIT*I*I*B\ncsc_lt_csc          v iiIITIIT*I*I*B\ncsc_gt_csc          v iiIITIIT*I*I*B\ncsc_le_csc          v iiIITIIT*I*I*B\ncsc_ge_csc          v iiIITIIT*I*I*B\n')
# Assigning a type to the variable 'CSC_ROUTINES' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'CSC_ROUTINES', str_376731)

# Assigning a Str to a Name (line 75):

# Assigning a Str to a Name (line 75):
str_376732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, (-1)), 'str', '\ncsr_matmat_pass1    v iiIIII*I\ncsr_matmat_pass2    v iiIITIIT*I*I*T\ncsr_diagonal        v iiiIIT*T\ncsr_tocsc           v iiIIT*I*I*T\ncsr_tobsr           v iiiiIIT*I*I*T\ncsr_todense         v iiIIT*T\ncsr_matvec          v iiIITT*T\ncsr_matvecs         v iiiIITT*T\ncsr_elmul_csr       v iiIITIIT*I*I*T\ncsr_eldiv_csr       v iiIITIIT*I*I*T\ncsr_plus_csr        v iiIITIIT*I*I*T\ncsr_minus_csr       v iiIITIIT*I*I*T\ncsr_maximum_csr     v iiIITIIT*I*I*T\ncsr_minimum_csr     v iiIITIIT*I*I*T\ncsr_ne_csr          v iiIITIIT*I*I*B\ncsr_lt_csr          v iiIITIIT*I*I*B\ncsr_gt_csr          v iiIITIIT*I*I*B\ncsr_le_csr          v iiIITIIT*I*I*B\ncsr_ge_csr          v iiIITIIT*I*I*B\ncsr_scale_rows      v iiII*TT\ncsr_scale_columns   v iiII*TT\ncsr_sort_indices    v iI*I*T\ncsr_eliminate_zeros v ii*I*I*T\ncsr_sum_duplicates  v ii*I*I*T\nget_csr_submatrix   v iiIITiiii*V*V*W\ncsr_sample_values   v iiIITiII*T\ncsr_count_blocks    i iiiiII\ncsr_sample_offsets  i iiIIiII*I\nexpandptr           v iI*I\ntest_throw_error    i\ncsr_has_sorted_indices    i iII\ncsr_has_canonical_format  i iII\n')
# Assigning a type to the variable 'CSR_ROUTINES' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'CSR_ROUTINES', str_376732)

# Assigning a Str to a Name (line 111):

# Assigning a Str to a Name (line 111):
str_376733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, (-1)), 'str', '\ncoo_tocsr           v iiiIIT*I*I*T\ncoo_todense         v iiiIIT*Ti\ncoo_matvec          v iIITT*T\ndia_matvec          v iiiiITT*T\ncs_graph_components i iII*I\n')
# Assigning a type to the variable 'OTHER_ROUTINES' (line 111)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'OTHER_ROUTINES', str_376733)

# Assigning a List to a Name (line 120):

# Assigning a List to a Name (line 120):

# Obtaining an instance of the builtin type 'list' (line 120)
list_376734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 120)
# Adding element type (line 120)

# Obtaining an instance of the builtin type 'tuple' (line 121)
tuple_376735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 121)
# Adding element type (line 121)
str_376736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 5), 'str', 'bsr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 5), tuple_376735, str_376736)
# Adding element type (line 121)
# Getting the type of 'BSR_ROUTINES' (line 121)
BSR_ROUTINES_376737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'BSR_ROUTINES')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 5), tuple_376735, BSR_ROUTINES_376737)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 20), list_376734, tuple_376735)
# Adding element type (line 120)

# Obtaining an instance of the builtin type 'tuple' (line 122)
tuple_376738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 122)
# Adding element type (line 122)
str_376739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 5), 'str', 'csr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 5), tuple_376738, str_376739)
# Adding element type (line 122)
# Getting the type of 'CSR_ROUTINES' (line 122)
CSR_ROUTINES_376740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'CSR_ROUTINES')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 5), tuple_376738, CSR_ROUTINES_376740)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 20), list_376734, tuple_376738)
# Adding element type (line 120)

# Obtaining an instance of the builtin type 'tuple' (line 123)
tuple_376741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 123)
# Adding element type (line 123)
str_376742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 5), 'str', 'csc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 5), tuple_376741, str_376742)
# Adding element type (line 123)
# Getting the type of 'CSC_ROUTINES' (line 123)
CSC_ROUTINES_376743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'CSC_ROUTINES')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 5), tuple_376741, CSC_ROUTINES_376743)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 20), list_376734, tuple_376741)
# Adding element type (line 120)

# Obtaining an instance of the builtin type 'tuple' (line 124)
tuple_376744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 124)
# Adding element type (line 124)
str_376745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 5), 'str', 'other')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 5), tuple_376744, str_376745)
# Adding element type (line 124)
# Getting the type of 'OTHER_ROUTINES' (line 124)
OTHER_ROUTINES_376746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 14), 'OTHER_ROUTINES')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 5), tuple_376744, OTHER_ROUTINES_376746)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 20), list_376734, tuple_376744)

# Assigning a type to the variable 'COMPILATION_UNITS' (line 120)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 0), 'COMPILATION_UNITS', list_376734)

# Assigning a List to a Name (line 130):

# Assigning a List to a Name (line 130):

# Obtaining an instance of the builtin type 'list' (line 130)
list_376747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 130)
# Adding element type (line 130)

# Obtaining an instance of the builtin type 'tuple' (line 131)
tuple_376748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 131)
# Adding element type (line 131)
str_376749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 5), 'str', 'NPY_INT32')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 5), tuple_376748, str_376749)
# Adding element type (line 131)
str_376750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 18), 'str', 'npy_int32')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 5), tuple_376748, str_376750)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 10), list_376747, tuple_376748)
# Adding element type (line 130)

# Obtaining an instance of the builtin type 'tuple' (line 132)
tuple_376751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 132)
# Adding element type (line 132)
str_376752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 5), 'str', 'NPY_INT64')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 5), tuple_376751, str_376752)
# Adding element type (line 132)
str_376753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 18), 'str', 'npy_int64')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 5), tuple_376751, str_376753)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 10), list_376747, tuple_376751)

# Assigning a type to the variable 'I_TYPES' (line 130)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 0), 'I_TYPES', list_376747)

# Assigning a List to a Name (line 138):

# Assigning a List to a Name (line 138):

# Obtaining an instance of the builtin type 'list' (line 138)
list_376754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 138)
# Adding element type (line 138)

# Obtaining an instance of the builtin type 'tuple' (line 139)
tuple_376755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 139)
# Adding element type (line 139)
str_376756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 5), 'str', 'NPY_BOOL')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 5), tuple_376755, str_376756)
# Adding element type (line 139)
str_376757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 17), 'str', 'npy_bool_wrapper')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 5), tuple_376755, str_376757)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 10), list_376754, tuple_376755)
# Adding element type (line 138)

# Obtaining an instance of the builtin type 'tuple' (line 140)
tuple_376758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 140)
# Adding element type (line 140)
str_376759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 5), 'str', 'NPY_BYTE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 5), tuple_376758, str_376759)
# Adding element type (line 140)
str_376760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 17), 'str', 'npy_byte')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 5), tuple_376758, str_376760)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 10), list_376754, tuple_376758)
# Adding element type (line 138)

# Obtaining an instance of the builtin type 'tuple' (line 141)
tuple_376761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 141)
# Adding element type (line 141)
str_376762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 5), 'str', 'NPY_UBYTE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 5), tuple_376761, str_376762)
# Adding element type (line 141)
str_376763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 18), 'str', 'npy_ubyte')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 5), tuple_376761, str_376763)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 10), list_376754, tuple_376761)
# Adding element type (line 138)

# Obtaining an instance of the builtin type 'tuple' (line 142)
tuple_376764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 142)
# Adding element type (line 142)
str_376765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 5), 'str', 'NPY_SHORT')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 5), tuple_376764, str_376765)
# Adding element type (line 142)
str_376766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 18), 'str', 'npy_short')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 5), tuple_376764, str_376766)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 10), list_376754, tuple_376764)
# Adding element type (line 138)

# Obtaining an instance of the builtin type 'tuple' (line 143)
tuple_376767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 143)
# Adding element type (line 143)
str_376768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 5), 'str', 'NPY_USHORT')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 5), tuple_376767, str_376768)
# Adding element type (line 143)
str_376769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 19), 'str', 'npy_ushort')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 5), tuple_376767, str_376769)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 10), list_376754, tuple_376767)
# Adding element type (line 138)

# Obtaining an instance of the builtin type 'tuple' (line 144)
tuple_376770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 144)
# Adding element type (line 144)
str_376771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 5), 'str', 'NPY_INT')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 5), tuple_376770, str_376771)
# Adding element type (line 144)
str_376772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 16), 'str', 'npy_int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 5), tuple_376770, str_376772)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 10), list_376754, tuple_376770)
# Adding element type (line 138)

# Obtaining an instance of the builtin type 'tuple' (line 145)
tuple_376773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 145)
# Adding element type (line 145)
str_376774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 5), 'str', 'NPY_UINT')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 5), tuple_376773, str_376774)
# Adding element type (line 145)
str_376775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 17), 'str', 'npy_uint')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 5), tuple_376773, str_376775)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 10), list_376754, tuple_376773)
# Adding element type (line 138)

# Obtaining an instance of the builtin type 'tuple' (line 146)
tuple_376776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 146)
# Adding element type (line 146)
str_376777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 5), 'str', 'NPY_LONG')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 5), tuple_376776, str_376777)
# Adding element type (line 146)
str_376778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 17), 'str', 'npy_long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 5), tuple_376776, str_376778)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 10), list_376754, tuple_376776)
# Adding element type (line 138)

# Obtaining an instance of the builtin type 'tuple' (line 147)
tuple_376779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 147)
# Adding element type (line 147)
str_376780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 5), 'str', 'NPY_ULONG')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 5), tuple_376779, str_376780)
# Adding element type (line 147)
str_376781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 18), 'str', 'npy_ulong')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 5), tuple_376779, str_376781)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 10), list_376754, tuple_376779)
# Adding element type (line 138)

# Obtaining an instance of the builtin type 'tuple' (line 148)
tuple_376782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 148)
# Adding element type (line 148)
str_376783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 5), 'str', 'NPY_LONGLONG')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 5), tuple_376782, str_376783)
# Adding element type (line 148)
str_376784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 21), 'str', 'npy_longlong')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 5), tuple_376782, str_376784)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 10), list_376754, tuple_376782)
# Adding element type (line 138)

# Obtaining an instance of the builtin type 'tuple' (line 149)
tuple_376785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 149)
# Adding element type (line 149)
str_376786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 5), 'str', 'NPY_ULONGLONG')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 5), tuple_376785, str_376786)
# Adding element type (line 149)
str_376787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 22), 'str', 'npy_ulonglong')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 5), tuple_376785, str_376787)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 10), list_376754, tuple_376785)
# Adding element type (line 138)

# Obtaining an instance of the builtin type 'tuple' (line 150)
tuple_376788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 150)
# Adding element type (line 150)
str_376789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 5), 'str', 'NPY_FLOAT')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 5), tuple_376788, str_376789)
# Adding element type (line 150)
str_376790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 18), 'str', 'npy_float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 5), tuple_376788, str_376790)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 10), list_376754, tuple_376788)
# Adding element type (line 138)

# Obtaining an instance of the builtin type 'tuple' (line 151)
tuple_376791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 151)
# Adding element type (line 151)
str_376792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 5), 'str', 'NPY_DOUBLE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 5), tuple_376791, str_376792)
# Adding element type (line 151)
str_376793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 19), 'str', 'npy_double')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 5), tuple_376791, str_376793)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 10), list_376754, tuple_376791)
# Adding element type (line 138)

# Obtaining an instance of the builtin type 'tuple' (line 152)
tuple_376794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 152)
# Adding element type (line 152)
str_376795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 5), 'str', 'NPY_LONGDOUBLE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 5), tuple_376794, str_376795)
# Adding element type (line 152)
str_376796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 23), 'str', 'npy_longdouble')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 5), tuple_376794, str_376796)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 10), list_376754, tuple_376794)
# Adding element type (line 138)

# Obtaining an instance of the builtin type 'tuple' (line 153)
tuple_376797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 153)
# Adding element type (line 153)
str_376798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 5), 'str', 'NPY_CFLOAT')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 5), tuple_376797, str_376798)
# Adding element type (line 153)
str_376799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 19), 'str', 'npy_cfloat_wrapper')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 5), tuple_376797, str_376799)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 10), list_376754, tuple_376797)
# Adding element type (line 138)

# Obtaining an instance of the builtin type 'tuple' (line 154)
tuple_376800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 154)
# Adding element type (line 154)
str_376801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 5), 'str', 'NPY_CDOUBLE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 5), tuple_376800, str_376801)
# Adding element type (line 154)
str_376802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 20), 'str', 'npy_cdouble_wrapper')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 5), tuple_376800, str_376802)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 10), list_376754, tuple_376800)
# Adding element type (line 138)

# Obtaining an instance of the builtin type 'tuple' (line 155)
tuple_376803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 155)
# Adding element type (line 155)
str_376804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 5), 'str', 'NPY_CLONGDOUBLE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 5), tuple_376803, str_376804)
# Adding element type (line 155)
str_376805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 24), 'str', 'npy_clongdouble_wrapper')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 5), tuple_376803, str_376805)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 10), list_376754, tuple_376803)

# Assigning a type to the variable 'T_TYPES' (line 138)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), 'T_TYPES', list_376754)

# Assigning a Str to a Name (line 162):

# Assigning a Str to a Name (line 162):
str_376806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, (-1)), 'str', '\nstatic Py_ssize_t %(name)s_thunk(int I_typenum, int T_typenum, void **a)\n{\n    %(thunk_content)s\n}\n')
# Assigning a type to the variable 'THUNK_TEMPLATE' (line 162)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 0), 'THUNK_TEMPLATE', str_376806)

# Assigning a Str to a Name (line 169):

# Assigning a Str to a Name (line 169):
str_376807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, (-1)), 'str', '\nNPY_VISIBILITY_HIDDEN PyObject *\n%(name)s_method(PyObject *self, PyObject *args)\n{\n    return call_thunk(\'%(ret_spec)s\', "%(arg_spec)s", %(name)s_thunk, args);\n}\n')
# Assigning a type to the variable 'METHOD_TEMPLATE' (line 169)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 0), 'METHOD_TEMPLATE', str_376807)

# Assigning a Str to a Name (line 177):

# Assigning a Str to a Name (line 177):
str_376808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, (-1)), 'str', '\nstatic int get_thunk_case(int I_typenum, int T_typenum)\n{\n    %(content)s;\n    return -1;\n}\n')
# Assigning a type to the variable 'GET_THUNK_CASE_TEMPLATE' (line 177)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'GET_THUNK_CASE_TEMPLATE', str_376808)

@norecursion
def get_thunk_type_set(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_thunk_type_set'
    module_type_store = module_type_store.open_function_context('get_thunk_type_set', 190, 0, False)
    
    # Passed parameters checking function
    get_thunk_type_set.stypy_localization = localization
    get_thunk_type_set.stypy_type_of_self = None
    get_thunk_type_set.stypy_type_store = module_type_store
    get_thunk_type_set.stypy_function_name = 'get_thunk_type_set'
    get_thunk_type_set.stypy_param_names_list = []
    get_thunk_type_set.stypy_varargs_param_name = None
    get_thunk_type_set.stypy_kwargs_param_name = None
    get_thunk_type_set.stypy_call_defaults = defaults
    get_thunk_type_set.stypy_call_varargs = varargs
    get_thunk_type_set.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_thunk_type_set', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_thunk_type_set', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_thunk_type_set(...)' code ##################

    str_376809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, (-1)), 'str', '\n    Get a list containing cartesian product of data types, plus a getter routine.\n\n    Returns\n    -------\n    i_types : list [(j, I_typenum, None, I_type, None), ...]\n         Pairing of index type numbers and the corresponding C++ types,\n         and an unique index `j`. This is for routines that are parameterized\n         only by I but not by T.\n    it_types : list [(j, I_typenum, T_typenum, I_type, T_type), ...]\n         Same as `i_types`, but for routines parameterized both by T and I.\n    getter_code : str\n         C++ code for a function that takes I_typenum, T_typenum and returns\n         the unique index corresponding to the lists, or -1 if no match was\n         found.\n\n    ')
    
    # Assigning a List to a Name (line 208):
    
    # Assigning a List to a Name (line 208):
    
    # Obtaining an instance of the builtin type 'list' (line 208)
    list_376810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 208)
    
    # Assigning a type to the variable 'it_types' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'it_types', list_376810)
    
    # Assigning a List to a Name (line 209):
    
    # Assigning a List to a Name (line 209):
    
    # Obtaining an instance of the builtin type 'list' (line 209)
    list_376811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 209)
    
    # Assigning a type to the variable 'i_types' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'i_types', list_376811)
    
    # Assigning a Num to a Name (line 211):
    
    # Assigning a Num to a Name (line 211):
    int_376812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 8), 'int')
    # Assigning a type to the variable 'j' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'j', int_376812)
    
    # Assigning a Str to a Name (line 213):
    
    # Assigning a Str to a Name (line 213):
    str_376813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 18), 'str', '    if (0) {}')
    # Assigning a type to the variable 'getter_code' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'getter_code', str_376813)
    
    # Getting the type of 'I_TYPES' (line 215)
    I_TYPES_376814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 29), 'I_TYPES')
    # Testing the type of a for loop iterable (line 215)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 215, 4), I_TYPES_376814)
    # Getting the type of the for loop variable (line 215)
    for_loop_var_376815 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 215, 4), I_TYPES_376814)
    # Assigning a type to the variable 'I_typenum' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'I_typenum', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 4), for_loop_var_376815))
    # Assigning a type to the variable 'I_type' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'I_type', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 4), for_loop_var_376815))
    # SSA begins for a for statement (line 215)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Str to a Name (line 216):
    
    # Assigning a Str to a Name (line 216):
    str_376816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, (-1)), 'str', '\n        else if (I_typenum == %(I_typenum)s) {\n            if (T_typenum == -1) { return %(j)s; }')
    # Assigning a type to the variable 'piece' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'piece', str_376816)
    
    # Getting the type of 'getter_code' (line 219)
    getter_code_376817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'getter_code')
    # Getting the type of 'piece' (line 219)
    piece_376818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 23), 'piece')
    
    # Call to dict(...): (line 219)
    # Processing the call keyword arguments (line 219)
    # Getting the type of 'I_typenum' (line 219)
    I_typenum_376820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 46), 'I_typenum', False)
    keyword_376821 = I_typenum_376820
    # Getting the type of 'j' (line 219)
    j_376822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 59), 'j', False)
    keyword_376823 = j_376822
    kwargs_376824 = {'j': keyword_376823, 'I_typenum': keyword_376821}
    # Getting the type of 'dict' (line 219)
    dict_376819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 31), 'dict', False)
    # Calling dict(args, kwargs) (line 219)
    dict_call_result_376825 = invoke(stypy.reporting.localization.Localization(__file__, 219, 31), dict_376819, *[], **kwargs_376824)
    
    # Applying the binary operator '%' (line 219)
    result_mod_376826 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 23), '%', piece_376818, dict_call_result_376825)
    
    # Applying the binary operator '+=' (line 219)
    result_iadd_376827 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 8), '+=', getter_code_376817, result_mod_376826)
    # Assigning a type to the variable 'getter_code' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'getter_code', result_iadd_376827)
    
    
    # Call to append(...): (line 221)
    # Processing the call arguments (line 221)
    
    # Obtaining an instance of the builtin type 'tuple' (line 221)
    tuple_376830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 221)
    # Adding element type (line 221)
    # Getting the type of 'j' (line 221)
    j_376831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 24), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 24), tuple_376830, j_376831)
    # Adding element type (line 221)
    # Getting the type of 'I_typenum' (line 221)
    I_typenum_376832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 27), 'I_typenum', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 24), tuple_376830, I_typenum_376832)
    # Adding element type (line 221)
    # Getting the type of 'None' (line 221)
    None_376833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 38), 'None', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 24), tuple_376830, None_376833)
    # Adding element type (line 221)
    # Getting the type of 'I_type' (line 221)
    I_type_376834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 44), 'I_type', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 24), tuple_376830, I_type_376834)
    # Adding element type (line 221)
    # Getting the type of 'None' (line 221)
    None_376835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 52), 'None', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 24), tuple_376830, None_376835)
    
    # Processing the call keyword arguments (line 221)
    kwargs_376836 = {}
    # Getting the type of 'i_types' (line 221)
    i_types_376828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'i_types', False)
    # Obtaining the member 'append' of a type (line 221)
    append_376829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), i_types_376828, 'append')
    # Calling append(args, kwargs) (line 221)
    append_call_result_376837 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), append_376829, *[tuple_376830], **kwargs_376836)
    
    
    # Getting the type of 'j' (line 222)
    j_376838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'j')
    int_376839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 13), 'int')
    # Applying the binary operator '+=' (line 222)
    result_iadd_376840 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 8), '+=', j_376838, int_376839)
    # Assigning a type to the variable 'j' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'j', result_iadd_376840)
    
    
    # Getting the type of 'T_TYPES' (line 224)
    T_TYPES_376841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 33), 'T_TYPES')
    # Testing the type of a for loop iterable (line 224)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 224, 8), T_TYPES_376841)
    # Getting the type of the for loop variable (line 224)
    for_loop_var_376842 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 224, 8), T_TYPES_376841)
    # Assigning a type to the variable 'T_typenum' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'T_typenum', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 8), for_loop_var_376842))
    # Assigning a type to the variable 'T_type' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'T_type', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 8), for_loop_var_376842))
    # SSA begins for a for statement (line 224)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Str to a Name (line 225):
    
    # Assigning a Str to a Name (line 225):
    str_376843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, (-1)), 'str', '\n            else if (T_typenum == %(T_typenum)s) { return %(j)s; }')
    # Assigning a type to the variable 'piece' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'piece', str_376843)
    
    # Getting the type of 'getter_code' (line 227)
    getter_code_376844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'getter_code')
    # Getting the type of 'piece' (line 227)
    piece_376845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 27), 'piece')
    
    # Call to dict(...): (line 227)
    # Processing the call keyword arguments (line 227)
    # Getting the type of 'T_typenum' (line 227)
    T_typenum_376847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 50), 'T_typenum', False)
    keyword_376848 = T_typenum_376847
    # Getting the type of 'j' (line 227)
    j_376849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 63), 'j', False)
    keyword_376850 = j_376849
    kwargs_376851 = {'j': keyword_376850, 'T_typenum': keyword_376848}
    # Getting the type of 'dict' (line 227)
    dict_376846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 35), 'dict', False)
    # Calling dict(args, kwargs) (line 227)
    dict_call_result_376852 = invoke(stypy.reporting.localization.Localization(__file__, 227, 35), dict_376846, *[], **kwargs_376851)
    
    # Applying the binary operator '%' (line 227)
    result_mod_376853 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 27), '%', piece_376845, dict_call_result_376852)
    
    # Applying the binary operator '+=' (line 227)
    result_iadd_376854 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 12), '+=', getter_code_376844, result_mod_376853)
    # Assigning a type to the variable 'getter_code' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'getter_code', result_iadd_376854)
    
    
    # Call to append(...): (line 229)
    # Processing the call arguments (line 229)
    
    # Obtaining an instance of the builtin type 'tuple' (line 229)
    tuple_376857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 229)
    # Adding element type (line 229)
    # Getting the type of 'j' (line 229)
    j_376858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 29), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 29), tuple_376857, j_376858)
    # Adding element type (line 229)
    # Getting the type of 'I_typenum' (line 229)
    I_typenum_376859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 32), 'I_typenum', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 29), tuple_376857, I_typenum_376859)
    # Adding element type (line 229)
    # Getting the type of 'T_typenum' (line 229)
    T_typenum_376860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 43), 'T_typenum', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 29), tuple_376857, T_typenum_376860)
    # Adding element type (line 229)
    # Getting the type of 'I_type' (line 229)
    I_type_376861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 54), 'I_type', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 29), tuple_376857, I_type_376861)
    # Adding element type (line 229)
    # Getting the type of 'T_type' (line 229)
    T_type_376862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 62), 'T_type', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 29), tuple_376857, T_type_376862)
    
    # Processing the call keyword arguments (line 229)
    kwargs_376863 = {}
    # Getting the type of 'it_types' (line 229)
    it_types_376855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'it_types', False)
    # Obtaining the member 'append' of a type (line 229)
    append_376856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 12), it_types_376855, 'append')
    # Calling append(args, kwargs) (line 229)
    append_call_result_376864 = invoke(stypy.reporting.localization.Localization(__file__, 229, 12), append_376856, *[tuple_376857], **kwargs_376863)
    
    
    # Getting the type of 'j' (line 230)
    j_376865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'j')
    int_376866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 17), 'int')
    # Applying the binary operator '+=' (line 230)
    result_iadd_376867 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 12), '+=', j_376865, int_376866)
    # Assigning a type to the variable 'j' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'j', result_iadd_376867)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'getter_code' (line 232)
    getter_code_376868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'getter_code')
    str_376869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, (-1)), 'str', '\n        }')
    # Applying the binary operator '+=' (line 232)
    result_iadd_376870 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 8), '+=', getter_code_376868, str_376869)
    # Assigning a type to the variable 'getter_code' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'getter_code', result_iadd_376870)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 235)
    tuple_376871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 235)
    # Adding element type (line 235)
    # Getting the type of 'i_types' (line 235)
    i_types_376872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 11), 'i_types')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 11), tuple_376871, i_types_376872)
    # Adding element type (line 235)
    # Getting the type of 'it_types' (line 235)
    it_types_376873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'it_types')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 11), tuple_376871, it_types_376873)
    # Adding element type (line 235)
    # Getting the type of 'GET_THUNK_CASE_TEMPLATE' (line 235)
    GET_THUNK_CASE_TEMPLATE_376874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 30), 'GET_THUNK_CASE_TEMPLATE')
    
    # Call to dict(...): (line 235)
    # Processing the call keyword arguments (line 235)
    # Getting the type of 'getter_code' (line 235)
    getter_code_376876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 69), 'getter_code', False)
    keyword_376877 = getter_code_376876
    kwargs_376878 = {'content': keyword_376877}
    # Getting the type of 'dict' (line 235)
    dict_376875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 56), 'dict', False)
    # Calling dict(args, kwargs) (line 235)
    dict_call_result_376879 = invoke(stypy.reporting.localization.Localization(__file__, 235, 56), dict_376875, *[], **kwargs_376878)
    
    # Applying the binary operator '%' (line 235)
    result_mod_376880 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 30), '%', GET_THUNK_CASE_TEMPLATE_376874, dict_call_result_376879)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 11), tuple_376871, result_mod_376880)
    
    # Assigning a type to the variable 'stypy_return_type' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'stypy_return_type', tuple_376871)
    
    # ################# End of 'get_thunk_type_set(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_thunk_type_set' in the type store
    # Getting the type of 'stypy_return_type' (line 190)
    stypy_return_type_376881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_376881)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_thunk_type_set'
    return stypy_return_type_376881

# Assigning a type to the variable 'get_thunk_type_set' (line 190)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'get_thunk_type_set', get_thunk_type_set)

@norecursion
def parse_routine(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'parse_routine'
    module_type_store = module_type_store.open_function_context('parse_routine', 238, 0, False)
    
    # Passed parameters checking function
    parse_routine.stypy_localization = localization
    parse_routine.stypy_type_of_self = None
    parse_routine.stypy_type_store = module_type_store
    parse_routine.stypy_function_name = 'parse_routine'
    parse_routine.stypy_param_names_list = ['name', 'args', 'types']
    parse_routine.stypy_varargs_param_name = None
    parse_routine.stypy_kwargs_param_name = None
    parse_routine.stypy_call_defaults = defaults
    parse_routine.stypy_call_varargs = varargs
    parse_routine.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'parse_routine', ['name', 'args', 'types'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'parse_routine', localization, ['name', 'args', 'types'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'parse_routine(...)' code ##################

    str_376882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, (-1)), 'str', '\n    Generate thunk and method code for a given routine.\n\n    Parameters\n    ----------\n    name : str\n        Name of the C++ routine\n    args : str\n        Argument list specification (in format explained above)\n    types : list\n        List of types to instantiate, as returned `get_thunk_type_set`\n\n    ')
    
    # Assigning a Subscript to a Name (line 253):
    
    # Assigning a Subscript to a Name (line 253):
    
    # Obtaining the type of the subscript
    int_376883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 20), 'int')
    # Getting the type of 'args' (line 253)
    args_376884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 15), 'args')
    # Obtaining the member '__getitem__' of a type (line 253)
    getitem___376885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 15), args_376884, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 253)
    subscript_call_result_376886 = invoke(stypy.reporting.localization.Localization(__file__, 253, 15), getitem___376885, int_376883)
    
    # Assigning a type to the variable 'ret_spec' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'ret_spec', subscript_call_result_376886)
    
    # Assigning a Subscript to a Name (line 254):
    
    # Assigning a Subscript to a Name (line 254):
    
    # Obtaining the type of the subscript
    int_376887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 20), 'int')
    slice_376888 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 254, 15), int_376887, None, None)
    # Getting the type of 'args' (line 254)
    args_376889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 15), 'args')
    # Obtaining the member '__getitem__' of a type (line 254)
    getitem___376890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 15), args_376889, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 254)
    subscript_call_result_376891 = invoke(stypy.reporting.localization.Localization(__file__, 254, 15), getitem___376890, slice_376888)
    
    # Assigning a type to the variable 'arg_spec' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'arg_spec', subscript_call_result_376891)

    @norecursion
    def get_arglist(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_arglist'
        module_type_store = module_type_store.open_function_context('get_arglist', 256, 4, False)
        
        # Passed parameters checking function
        get_arglist.stypy_localization = localization
        get_arglist.stypy_type_of_self = None
        get_arglist.stypy_type_store = module_type_store
        get_arglist.stypy_function_name = 'get_arglist'
        get_arglist.stypy_param_names_list = ['I_type', 'T_type']
        get_arglist.stypy_varargs_param_name = None
        get_arglist.stypy_kwargs_param_name = None
        get_arglist.stypy_call_defaults = defaults
        get_arglist.stypy_call_varargs = varargs
        get_arglist.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'get_arglist', ['I_type', 'T_type'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_arglist', localization, ['I_type', 'T_type'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_arglist(...)' code ##################

        str_376892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, (-1)), 'str', '\n        Generate argument list for calling the C++ function\n        ')
        
        # Assigning a List to a Name (line 260):
        
        # Assigning a List to a Name (line 260):
        
        # Obtaining an instance of the builtin type 'list' (line 260)
        list_376893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 260)
        
        # Assigning a type to the variable 'args' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'args', list_376893)
        
        # Assigning a Name to a Name (line 261):
        
        # Assigning a Name to a Name (line 261):
        # Getting the type of 'False' (line 261)
        False_376894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 28), 'False')
        # Assigning a type to the variable 'next_is_writeable' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'next_is_writeable', False_376894)
        
        # Assigning a Num to a Name (line 262):
        
        # Assigning a Num to a Name (line 262):
        int_376895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 12), 'int')
        # Assigning a type to the variable 'j' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'j', int_376895)
        
        # Getting the type of 'arg_spec' (line 263)
        arg_spec_376896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 17), 'arg_spec')
        # Testing the type of a for loop iterable (line 263)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 263, 8), arg_spec_376896)
        # Getting the type of the for loop variable (line 263)
        for_loop_var_376897 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 263, 8), arg_spec_376896)
        # Assigning a type to the variable 't' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 't', for_loop_var_376897)
        # SSA begins for a for statement (line 263)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a IfExp to a Name (line 264):
        
        # Assigning a IfExp to a Name (line 264):
        
        # Getting the type of 'next_is_writeable' (line 264)
        next_is_writeable_376898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 26), 'next_is_writeable')
        # Testing the type of an if expression (line 264)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 264, 20), next_is_writeable_376898)
        # SSA begins for if expression (line 264)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        str_376899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 20), 'str', '')
        # SSA branch for the else part of an if expression (line 264)
        module_type_store.open_ssa_branch('if expression else')
        str_376900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 49), 'str', 'const ')
        # SSA join for if expression (line 264)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_376901 = union_type.UnionType.add(str_376899, str_376900)
        
        # Assigning a type to the variable 'const' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'const', if_exp_376901)
        
        # Assigning a Name to a Name (line 265):
        
        # Assigning a Name to a Name (line 265):
        # Getting the type of 'False' (line 265)
        False_376902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 32), 'False')
        # Assigning a type to the variable 'next_is_writeable' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'next_is_writeable', False_376902)
        
        
        # Getting the type of 't' (line 266)
        t_376903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 15), 't')
        str_376904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 20), 'str', '*')
        # Applying the binary operator '==' (line 266)
        result_eq_376905 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 15), '==', t_376903, str_376904)
        
        # Testing the type of an if condition (line 266)
        if_condition_376906 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 266, 12), result_eq_376905)
        # Assigning a type to the variable 'if_condition_376906' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'if_condition_376906', if_condition_376906)
        # SSA begins for if statement (line 266)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 267):
        
        # Assigning a Name to a Name (line 267):
        # Getting the type of 'True' (line 267)
        True_376907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 36), 'True')
        # Assigning a type to the variable 'next_is_writeable' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 16), 'next_is_writeable', True_376907)
        # SSA branch for the else part of an if statement (line 266)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 't' (line 269)
        t_376908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 17), 't')
        str_376909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 22), 'str', 'i')
        # Applying the binary operator '==' (line 269)
        result_eq_376910 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 17), '==', t_376908, str_376909)
        
        # Testing the type of an if condition (line 269)
        if_condition_376911 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 269, 17), result_eq_376910)
        # Assigning a type to the variable 'if_condition_376911' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 17), 'if_condition_376911', if_condition_376911)
        # SSA begins for if statement (line 269)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 270)
        # Processing the call arguments (line 270)
        str_376914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 28), 'str', '*(%s*)a[%d]')
        
        # Obtaining an instance of the builtin type 'tuple' (line 270)
        tuple_376915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 270)
        # Adding element type (line 270)
        # Getting the type of 'const' (line 270)
        const_376916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 45), 'const', False)
        # Getting the type of 'I_type' (line 270)
        I_type_376917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 53), 'I_type', False)
        # Applying the binary operator '+' (line 270)
        result_add_376918 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 45), '+', const_376916, I_type_376917)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 45), tuple_376915, result_add_376918)
        # Adding element type (line 270)
        # Getting the type of 'j' (line 270)
        j_376919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 61), 'j', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 45), tuple_376915, j_376919)
        
        # Applying the binary operator '%' (line 270)
        result_mod_376920 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 28), '%', str_376914, tuple_376915)
        
        # Processing the call keyword arguments (line 270)
        kwargs_376921 = {}
        # Getting the type of 'args' (line 270)
        args_376912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 16), 'args', False)
        # Obtaining the member 'append' of a type (line 270)
        append_376913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 16), args_376912, 'append')
        # Calling append(args, kwargs) (line 270)
        append_call_result_376922 = invoke(stypy.reporting.localization.Localization(__file__, 270, 16), append_376913, *[result_mod_376920], **kwargs_376921)
        
        # SSA branch for the else part of an if statement (line 269)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 't' (line 271)
        t_376923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 17), 't')
        str_376924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 22), 'str', 'I')
        # Applying the binary operator '==' (line 271)
        result_eq_376925 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 17), '==', t_376923, str_376924)
        
        # Testing the type of an if condition (line 271)
        if_condition_376926 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 271, 17), result_eq_376925)
        # Assigning a type to the variable 'if_condition_376926' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 17), 'if_condition_376926', if_condition_376926)
        # SSA begins for if statement (line 271)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 272)
        # Processing the call arguments (line 272)
        str_376929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 28), 'str', '(%s*)a[%d]')
        
        # Obtaining an instance of the builtin type 'tuple' (line 272)
        tuple_376930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 272)
        # Adding element type (line 272)
        # Getting the type of 'const' (line 272)
        const_376931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 44), 'const', False)
        # Getting the type of 'I_type' (line 272)
        I_type_376932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 52), 'I_type', False)
        # Applying the binary operator '+' (line 272)
        result_add_376933 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 44), '+', const_376931, I_type_376932)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 44), tuple_376930, result_add_376933)
        # Adding element type (line 272)
        # Getting the type of 'j' (line 272)
        j_376934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 60), 'j', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 44), tuple_376930, j_376934)
        
        # Applying the binary operator '%' (line 272)
        result_mod_376935 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 28), '%', str_376929, tuple_376930)
        
        # Processing the call keyword arguments (line 272)
        kwargs_376936 = {}
        # Getting the type of 'args' (line 272)
        args_376927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 16), 'args', False)
        # Obtaining the member 'append' of a type (line 272)
        append_376928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 16), args_376927, 'append')
        # Calling append(args, kwargs) (line 272)
        append_call_result_376937 = invoke(stypy.reporting.localization.Localization(__file__, 272, 16), append_376928, *[result_mod_376935], **kwargs_376936)
        
        # SSA branch for the else part of an if statement (line 271)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 't' (line 273)
        t_376938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 17), 't')
        str_376939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 22), 'str', 'T')
        # Applying the binary operator '==' (line 273)
        result_eq_376940 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 17), '==', t_376938, str_376939)
        
        # Testing the type of an if condition (line 273)
        if_condition_376941 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 17), result_eq_376940)
        # Assigning a type to the variable 'if_condition_376941' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 17), 'if_condition_376941', if_condition_376941)
        # SSA begins for if statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 274)
        # Processing the call arguments (line 274)
        str_376944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 28), 'str', '(%s*)a[%d]')
        
        # Obtaining an instance of the builtin type 'tuple' (line 274)
        tuple_376945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 274)
        # Adding element type (line 274)
        # Getting the type of 'const' (line 274)
        const_376946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 44), 'const', False)
        # Getting the type of 'T_type' (line 274)
        T_type_376947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 52), 'T_type', False)
        # Applying the binary operator '+' (line 274)
        result_add_376948 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 44), '+', const_376946, T_type_376947)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 44), tuple_376945, result_add_376948)
        # Adding element type (line 274)
        # Getting the type of 'j' (line 274)
        j_376949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 60), 'j', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 44), tuple_376945, j_376949)
        
        # Applying the binary operator '%' (line 274)
        result_mod_376950 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 28), '%', str_376944, tuple_376945)
        
        # Processing the call keyword arguments (line 274)
        kwargs_376951 = {}
        # Getting the type of 'args' (line 274)
        args_376942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'args', False)
        # Obtaining the member 'append' of a type (line 274)
        append_376943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 16), args_376942, 'append')
        # Calling append(args, kwargs) (line 274)
        append_call_result_376952 = invoke(stypy.reporting.localization.Localization(__file__, 274, 16), append_376943, *[result_mod_376950], **kwargs_376951)
        
        # SSA branch for the else part of an if statement (line 273)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 't' (line 275)
        t_376953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 17), 't')
        str_376954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 22), 'str', 'B')
        # Applying the binary operator '==' (line 275)
        result_eq_376955 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 17), '==', t_376953, str_376954)
        
        # Testing the type of an if condition (line 275)
        if_condition_376956 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 275, 17), result_eq_376955)
        # Assigning a type to the variable 'if_condition_376956' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 17), 'if_condition_376956', if_condition_376956)
        # SSA begins for if statement (line 275)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 276)
        # Processing the call arguments (line 276)
        str_376959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 28), 'str', '(npy_bool_wrapper*)a[%d]')
        
        # Obtaining an instance of the builtin type 'tuple' (line 276)
        tuple_376960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 276)
        # Adding element type (line 276)
        # Getting the type of 'j' (line 276)
        j_376961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 58), 'j', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 58), tuple_376960, j_376961)
        
        # Applying the binary operator '%' (line 276)
        result_mod_376962 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 28), '%', str_376959, tuple_376960)
        
        # Processing the call keyword arguments (line 276)
        kwargs_376963 = {}
        # Getting the type of 'args' (line 276)
        args_376957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 16), 'args', False)
        # Obtaining the member 'append' of a type (line 276)
        append_376958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 16), args_376957, 'append')
        # Calling append(args, kwargs) (line 276)
        append_call_result_376964 = invoke(stypy.reporting.localization.Localization(__file__, 276, 16), append_376958, *[result_mod_376962], **kwargs_376963)
        
        # SSA branch for the else part of an if statement (line 275)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 't' (line 277)
        t_376965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 17), 't')
        str_376966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 22), 'str', 'V')
        # Applying the binary operator '==' (line 277)
        result_eq_376967 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 17), '==', t_376965, str_376966)
        
        # Testing the type of an if condition (line 277)
        if_condition_376968 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 277, 17), result_eq_376967)
        # Assigning a type to the variable 'if_condition_376968' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 17), 'if_condition_376968', if_condition_376968)
        # SSA begins for if statement (line 277)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'const' (line 278)
        const_376969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 19), 'const')
        # Testing the type of an if condition (line 278)
        if_condition_376970 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 278, 16), const_376969)
        # Assigning a type to the variable 'if_condition_376970' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 16), 'if_condition_376970', if_condition_376970)
        # SSA begins for if statement (line 278)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 279)
        # Processing the call arguments (line 279)
        str_376972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 37), 'str', "'V' argument must be an output arg")
        # Processing the call keyword arguments (line 279)
        kwargs_376973 = {}
        # Getting the type of 'ValueError' (line 279)
        ValueError_376971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 26), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 279)
        ValueError_call_result_376974 = invoke(stypy.reporting.localization.Localization(__file__, 279, 26), ValueError_376971, *[str_376972], **kwargs_376973)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 279, 20), ValueError_call_result_376974, 'raise parameter', BaseException)
        # SSA join for if statement (line 278)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 280)
        # Processing the call arguments (line 280)
        str_376977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 28), 'str', '(std::vector<%s>*)a[%d]')
        
        # Obtaining an instance of the builtin type 'tuple' (line 280)
        tuple_376978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 57), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 280)
        # Adding element type (line 280)
        # Getting the type of 'I_type' (line 280)
        I_type_376979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 57), 'I_type', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 57), tuple_376978, I_type_376979)
        # Adding element type (line 280)
        # Getting the type of 'j' (line 280)
        j_376980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 65), 'j', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 57), tuple_376978, j_376980)
        
        # Applying the binary operator '%' (line 280)
        result_mod_376981 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 28), '%', str_376977, tuple_376978)
        
        # Processing the call keyword arguments (line 280)
        kwargs_376982 = {}
        # Getting the type of 'args' (line 280)
        args_376975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 16), 'args', False)
        # Obtaining the member 'append' of a type (line 280)
        append_376976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 16), args_376975, 'append')
        # Calling append(args, kwargs) (line 280)
        append_call_result_376983 = invoke(stypy.reporting.localization.Localization(__file__, 280, 16), append_376976, *[result_mod_376981], **kwargs_376982)
        
        # SSA branch for the else part of an if statement (line 277)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 't' (line 281)
        t_376984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 17), 't')
        str_376985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 22), 'str', 'W')
        # Applying the binary operator '==' (line 281)
        result_eq_376986 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 17), '==', t_376984, str_376985)
        
        # Testing the type of an if condition (line 281)
        if_condition_376987 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 281, 17), result_eq_376986)
        # Assigning a type to the variable 'if_condition_376987' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 17), 'if_condition_376987', if_condition_376987)
        # SSA begins for if statement (line 281)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'const' (line 282)
        const_376988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 19), 'const')
        # Testing the type of an if condition (line 282)
        if_condition_376989 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 282, 16), const_376988)
        # Assigning a type to the variable 'if_condition_376989' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 16), 'if_condition_376989', if_condition_376989)
        # SSA begins for if statement (line 282)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 283)
        # Processing the call arguments (line 283)
        str_376991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 37), 'str', "'W' argument must be an output arg")
        # Processing the call keyword arguments (line 283)
        kwargs_376992 = {}
        # Getting the type of 'ValueError' (line 283)
        ValueError_376990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 26), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 283)
        ValueError_call_result_376993 = invoke(stypy.reporting.localization.Localization(__file__, 283, 26), ValueError_376990, *[str_376991], **kwargs_376992)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 283, 20), ValueError_call_result_376993, 'raise parameter', BaseException)
        # SSA join for if statement (line 282)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 284)
        # Processing the call arguments (line 284)
        str_376996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 28), 'str', '(std::vector<%s>*)a[%d]')
        
        # Obtaining an instance of the builtin type 'tuple' (line 284)
        tuple_376997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 57), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 284)
        # Adding element type (line 284)
        # Getting the type of 'T_type' (line 284)
        T_type_376998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 57), 'T_type', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 57), tuple_376997, T_type_376998)
        # Adding element type (line 284)
        # Getting the type of 'j' (line 284)
        j_376999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 65), 'j', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 57), tuple_376997, j_376999)
        
        # Applying the binary operator '%' (line 284)
        result_mod_377000 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 28), '%', str_376996, tuple_376997)
        
        # Processing the call keyword arguments (line 284)
        kwargs_377001 = {}
        # Getting the type of 'args' (line 284)
        args_376994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'args', False)
        # Obtaining the member 'append' of a type (line 284)
        append_376995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 16), args_376994, 'append')
        # Calling append(args, kwargs) (line 284)
        append_call_result_377002 = invoke(stypy.reporting.localization.Localization(__file__, 284, 16), append_376995, *[result_mod_377000], **kwargs_377001)
        
        # SSA branch for the else part of an if statement (line 281)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 286)
        # Processing the call arguments (line 286)
        str_377004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 33), 'str', 'Invalid spec character %r')
        
        # Obtaining an instance of the builtin type 'tuple' (line 286)
        tuple_377005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 64), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 286)
        # Adding element type (line 286)
        # Getting the type of 't' (line 286)
        t_377006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 64), 't', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 64), tuple_377005, t_377006)
        
        # Applying the binary operator '%' (line 286)
        result_mod_377007 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 33), '%', str_377004, tuple_377005)
        
        # Processing the call keyword arguments (line 286)
        kwargs_377008 = {}
        # Getting the type of 'ValueError' (line 286)
        ValueError_377003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 286)
        ValueError_call_result_377009 = invoke(stypy.reporting.localization.Localization(__file__, 286, 22), ValueError_377003, *[result_mod_377007], **kwargs_377008)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 286, 16), ValueError_call_result_377009, 'raise parameter', BaseException)
        # SSA join for if statement (line 281)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 277)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 275)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 273)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 271)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 269)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 266)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'j' (line 287)
        j_377010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'j')
        int_377011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 17), 'int')
        # Applying the binary operator '+=' (line 287)
        result_iadd_377012 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 12), '+=', j_377010, int_377011)
        # Assigning a type to the variable 'j' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'j', result_iadd_377012)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to join(...): (line 288)
        # Processing the call arguments (line 288)
        # Getting the type of 'args' (line 288)
        args_377015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 25), 'args', False)
        # Processing the call keyword arguments (line 288)
        kwargs_377016 = {}
        str_377013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 15), 'str', ', ')
        # Obtaining the member 'join' of a type (line 288)
        join_377014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 15), str_377013, 'join')
        # Calling join(args, kwargs) (line 288)
        join_call_result_377017 = invoke(stypy.reporting.localization.Localization(__file__, 288, 15), join_377014, *[args_377015], **kwargs_377016)
        
        # Assigning a type to the variable 'stypy_return_type' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'stypy_return_type', join_call_result_377017)
        
        # ################# End of 'get_arglist(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_arglist' in the type store
        # Getting the type of 'stypy_return_type' (line 256)
        stypy_return_type_377018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_377018)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_arglist'
        return stypy_return_type_377018

    # Assigning a type to the variable 'get_arglist' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'get_arglist', get_arglist)
    
    # Assigning a Str to a Name (line 292):
    
    # Assigning a Str to a Name (line 292):
    str_377019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, (-1)), 'str', 'int j = get_thunk_case(I_typenum, T_typenum);\n    switch (j) {')
    # Assigning a type to the variable 'thunk_content' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'thunk_content', str_377019)
    
    # Getting the type of 'types' (line 294)
    types_377020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 51), 'types')
    # Testing the type of a for loop iterable (line 294)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 294, 4), types_377020)
    # Getting the type of the for loop variable (line 294)
    for_loop_var_377021 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 294, 4), types_377020)
    # Assigning a type to the variable 'j' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 4), for_loop_var_377021))
    # Assigning a type to the variable 'I_typenum' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'I_typenum', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 4), for_loop_var_377021))
    # Assigning a type to the variable 'T_typenum' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'T_typenum', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 4), for_loop_var_377021))
    # Assigning a type to the variable 'I_type' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'I_type', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 4), for_loop_var_377021))
    # Assigning a type to the variable 'T_type' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'T_type', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 4), for_loop_var_377021))
    # SSA begins for a for statement (line 294)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 295):
    
    # Assigning a Call to a Name (line 295):
    
    # Call to get_arglist(...): (line 295)
    # Processing the call arguments (line 295)
    # Getting the type of 'I_type' (line 295)
    I_type_377023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 30), 'I_type', False)
    # Getting the type of 'T_type' (line 295)
    T_type_377024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 38), 'T_type', False)
    # Processing the call keyword arguments (line 295)
    kwargs_377025 = {}
    # Getting the type of 'get_arglist' (line 295)
    get_arglist_377022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 18), 'get_arglist', False)
    # Calling get_arglist(args, kwargs) (line 295)
    get_arglist_call_result_377026 = invoke(stypy.reporting.localization.Localization(__file__, 295, 18), get_arglist_377022, *[I_type_377023, T_type_377024], **kwargs_377025)
    
    # Assigning a type to the variable 'arglist' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'arglist', get_arglist_call_result_377026)
    
    # Type idiom detected: calculating its left and rigth part (line 296)
    # Getting the type of 'T_type' (line 296)
    T_type_377027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 11), 'T_type')
    # Getting the type of 'None' (line 296)
    None_377028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 21), 'None')
    
    (may_be_377029, more_types_in_union_377030) = may_be_none(T_type_377027, None_377028)

    if may_be_377029:

        if more_types_in_union_377030:
            # Runtime conditional SSA (line 296)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 297):
        
        # Assigning a BinOp to a Name (line 297):
        str_377031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 23), 'str', '%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 297)
        tuple_377032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 297)
        # Adding element type (line 297)
        # Getting the type of 'I_type' (line 297)
        I_type_377033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 31), 'I_type')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 31), tuple_377032, I_type_377033)
        
        # Applying the binary operator '%' (line 297)
        result_mod_377034 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 23), '%', str_377031, tuple_377032)
        
        # Assigning a type to the variable 'dispatch' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'dispatch', result_mod_377034)

        if more_types_in_union_377030:
            # Runtime conditional SSA for else branch (line 296)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_377029) or more_types_in_union_377030):
        
        # Assigning a BinOp to a Name (line 299):
        
        # Assigning a BinOp to a Name (line 299):
        str_377035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 23), 'str', '%s,%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 299)
        tuple_377036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 299)
        # Adding element type (line 299)
        # Getting the type of 'I_type' (line 299)
        I_type_377037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 34), 'I_type')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 34), tuple_377036, I_type_377037)
        # Adding element type (line 299)
        # Getting the type of 'T_type' (line 299)
        T_type_377038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 42), 'T_type')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 34), tuple_377036, T_type_377038)
        
        # Applying the binary operator '%' (line 299)
        result_mod_377039 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 23), '%', str_377035, tuple_377036)
        
        # Assigning a type to the variable 'dispatch' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'dispatch', result_mod_377039)

        if (may_be_377029 and more_types_in_union_377030):
            # SSA join for if statement (line 296)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    str_377040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 11), 'str', 'B')
    # Getting the type of 'arg_spec' (line 300)
    arg_spec_377041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 18), 'arg_spec')
    # Applying the binary operator 'in' (line 300)
    result_contains_377042 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 11), 'in', str_377040, arg_spec_377041)
    
    # Testing the type of an if condition (line 300)
    if_condition_377043 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 300, 8), result_contains_377042)
    # Assigning a type to the variable 'if_condition_377043' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'if_condition_377043', if_condition_377043)
    # SSA begins for if statement (line 300)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'dispatch' (line 301)
    dispatch_377044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'dispatch')
    str_377045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 24), 'str', ',npy_bool_wrapper')
    # Applying the binary operator '+=' (line 301)
    result_iadd_377046 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 12), '+=', dispatch_377044, str_377045)
    # Assigning a type to the variable 'dispatch' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'dispatch', result_iadd_377046)
    
    # SSA join for if statement (line 300)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Str to a Name (line 303):
    
    # Assigning a Str to a Name (line 303):
    str_377047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, (-1)), 'str', '\n        case %(j)s:')
    # Assigning a type to the variable 'piece' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'piece', str_377047)
    
    
    # Getting the type of 'ret_spec' (line 305)
    ret_spec_377048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 11), 'ret_spec')
    str_377049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 23), 'str', 'v')
    # Applying the binary operator '==' (line 305)
    result_eq_377050 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 11), '==', ret_spec_377048, str_377049)
    
    # Testing the type of an if condition (line 305)
    if_condition_377051 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 8), result_eq_377050)
    # Assigning a type to the variable 'if_condition_377051' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'if_condition_377051', if_condition_377051)
    # SSA begins for if statement (line 305)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'piece' (line 306)
    piece_377052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'piece')
    str_377053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, (-1)), 'str', '\n            (void)%(name)s<%(dispatch)s>(%(arglist)s);\n            return 0;')
    # Applying the binary operator '+=' (line 306)
    result_iadd_377054 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 12), '+=', piece_377052, str_377053)
    # Assigning a type to the variable 'piece' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'piece', result_iadd_377054)
    
    # SSA branch for the else part of an if statement (line 305)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'piece' (line 310)
    piece_377055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'piece')
    str_377056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, (-1)), 'str', '\n            return %(name)s<%(dispatch)s>(%(arglist)s);')
    # Applying the binary operator '+=' (line 310)
    result_iadd_377057 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 12), '+=', piece_377055, str_377056)
    # Assigning a type to the variable 'piece' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'piece', result_iadd_377057)
    
    # SSA join for if statement (line 305)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'thunk_content' (line 312)
    thunk_content_377058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'thunk_content')
    # Getting the type of 'piece' (line 312)
    piece_377059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 25), 'piece')
    
    # Call to dict(...): (line 312)
    # Processing the call keyword arguments (line 312)
    # Getting the type of 'j' (line 312)
    j_377061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 40), 'j', False)
    keyword_377062 = j_377061
    # Getting the type of 'I_type' (line 312)
    I_type_377063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 50), 'I_type', False)
    keyword_377064 = I_type_377063
    # Getting the type of 'T_type' (line 312)
    T_type_377065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 65), 'T_type', False)
    keyword_377066 = T_type_377065
    # Getting the type of 'I_typenum' (line 313)
    I_typenum_377067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 48), 'I_typenum', False)
    keyword_377068 = I_typenum_377067
    # Getting the type of 'T_typenum' (line 313)
    T_typenum_377069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 69), 'T_typenum', False)
    keyword_377070 = T_typenum_377069
    # Getting the type of 'arglist' (line 314)
    arglist_377071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 46), 'arglist', False)
    keyword_377072 = arglist_377071
    # Getting the type of 'name' (line 314)
    name_377073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 60), 'name', False)
    keyword_377074 = name_377073
    # Getting the type of 'dispatch' (line 315)
    dispatch_377075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 47), 'dispatch', False)
    keyword_377076 = dispatch_377075
    kwargs_377077 = {'I_typenum': keyword_377068, 'I_type': keyword_377064, 'j': keyword_377062, 'T_typenum': keyword_377070, 'T_type': keyword_377066, 'dispatch': keyword_377076, 'arglist': keyword_377072, 'name': keyword_377074}
    # Getting the type of 'dict' (line 312)
    dict_377060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 33), 'dict', False)
    # Calling dict(args, kwargs) (line 312)
    dict_call_result_377078 = invoke(stypy.reporting.localization.Localization(__file__, 312, 33), dict_377060, *[], **kwargs_377077)
    
    # Applying the binary operator '%' (line 312)
    result_mod_377079 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 25), '%', piece_377059, dict_call_result_377078)
    
    # Applying the binary operator '+=' (line 312)
    result_iadd_377080 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 8), '+=', thunk_content_377058, result_mod_377079)
    # Assigning a type to the variable 'thunk_content' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'thunk_content', result_iadd_377080)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'thunk_content' (line 317)
    thunk_content_377081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'thunk_content')
    str_377082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, (-1)), 'str', '\n    default:\n        throw std::runtime_error("internal error: invalid argument typenums");\n    }')
    # Applying the binary operator '+=' (line 317)
    result_iadd_377083 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 4), '+=', thunk_content_377081, str_377082)
    # Assigning a type to the variable 'thunk_content' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'thunk_content', result_iadd_377083)
    
    
    # Assigning a BinOp to a Name (line 322):
    
    # Assigning a BinOp to a Name (line 322):
    # Getting the type of 'THUNK_TEMPLATE' (line 322)
    THUNK_TEMPLATE_377084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 17), 'THUNK_TEMPLATE')
    
    # Call to dict(...): (line 322)
    # Processing the call keyword arguments (line 322)
    # Getting the type of 'name' (line 322)
    name_377086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 44), 'name', False)
    keyword_377087 = name_377086
    # Getting the type of 'thunk_content' (line 323)
    thunk_content_377088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 53), 'thunk_content', False)
    keyword_377089 = thunk_content_377088
    kwargs_377090 = {'name': keyword_377087, 'thunk_content': keyword_377089}
    # Getting the type of 'dict' (line 322)
    dict_377085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 34), 'dict', False)
    # Calling dict(args, kwargs) (line 322)
    dict_call_result_377091 = invoke(stypy.reporting.localization.Localization(__file__, 322, 34), dict_377085, *[], **kwargs_377090)
    
    # Applying the binary operator '%' (line 322)
    result_mod_377092 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 17), '%', THUNK_TEMPLATE_377084, dict_call_result_377091)
    
    # Assigning a type to the variable 'thunk_code' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'thunk_code', result_mod_377092)
    
    # Assigning a BinOp to a Name (line 326):
    
    # Assigning a BinOp to a Name (line 326):
    # Getting the type of 'METHOD_TEMPLATE' (line 326)
    METHOD_TEMPLATE_377093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 18), 'METHOD_TEMPLATE')
    
    # Call to dict(...): (line 326)
    # Processing the call keyword arguments (line 326)
    # Getting the type of 'name' (line 326)
    name_377095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 46), 'name', False)
    keyword_377096 = name_377095
    # Getting the type of 'ret_spec' (line 327)
    ret_spec_377097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 50), 'ret_spec', False)
    keyword_377098 = ret_spec_377097
    # Getting the type of 'arg_spec' (line 328)
    arg_spec_377099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 50), 'arg_spec', False)
    keyword_377100 = arg_spec_377099
    kwargs_377101 = {'ret_spec': keyword_377098, 'name': keyword_377096, 'arg_spec': keyword_377100}
    # Getting the type of 'dict' (line 326)
    dict_377094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 36), 'dict', False)
    # Calling dict(args, kwargs) (line 326)
    dict_call_result_377102 = invoke(stypy.reporting.localization.Localization(__file__, 326, 36), dict_377094, *[], **kwargs_377101)
    
    # Applying the binary operator '%' (line 326)
    result_mod_377103 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 18), '%', METHOD_TEMPLATE_377093, dict_call_result_377102)
    
    # Assigning a type to the variable 'method_code' (line 326)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'method_code', result_mod_377103)
    
    # Obtaining an instance of the builtin type 'tuple' (line 330)
    tuple_377104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 330)
    # Adding element type (line 330)
    # Getting the type of 'thunk_code' (line 330)
    thunk_code_377105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 11), 'thunk_code')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 11), tuple_377104, thunk_code_377105)
    # Adding element type (line 330)
    # Getting the type of 'method_code' (line 330)
    method_code_377106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 23), 'method_code')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 11), tuple_377104, method_code_377106)
    
    # Assigning a type to the variable 'stypy_return_type' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'stypy_return_type', tuple_377104)
    
    # ################# End of 'parse_routine(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'parse_routine' in the type store
    # Getting the type of 'stypy_return_type' (line 238)
    stypy_return_type_377107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_377107)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'parse_routine'
    return stypy_return_type_377107

# Assigning a type to the variable 'parse_routine' (line 238)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 0), 'parse_routine', parse_routine)

@norecursion
def main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'main'
    module_type_store = module_type_store.open_function_context('main', 333, 0, False)
    
    # Passed parameters checking function
    main.stypy_localization = localization
    main.stypy_type_of_self = None
    main.stypy_type_store = module_type_store
    main.stypy_function_name = 'main'
    main.stypy_param_names_list = []
    main.stypy_varargs_param_name = None
    main.stypy_kwargs_param_name = None
    main.stypy_call_defaults = defaults
    main.stypy_call_varargs = varargs
    main.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'main', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'main', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'main(...)' code ##################

    
    # Assigning a Call to a Name (line 334):
    
    # Assigning a Call to a Name (line 334):
    
    # Call to OptionParser(...): (line 334)
    # Processing the call keyword arguments (line 334)
    
    # Call to strip(...): (line 334)
    # Processing the call keyword arguments (line 334)
    kwargs_377112 = {}
    # Getting the type of '__doc__' (line 334)
    doc___377110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 36), '__doc__', False)
    # Obtaining the member 'strip' of a type (line 334)
    strip_377111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 36), doc___377110, 'strip')
    # Calling strip(args, kwargs) (line 334)
    strip_call_result_377113 = invoke(stypy.reporting.localization.Localization(__file__, 334, 36), strip_377111, *[], **kwargs_377112)
    
    keyword_377114 = strip_call_result_377113
    kwargs_377115 = {'usage': keyword_377114}
    # Getting the type of 'optparse' (line 334)
    optparse_377108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'optparse', False)
    # Obtaining the member 'OptionParser' of a type (line 334)
    OptionParser_377109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 8), optparse_377108, 'OptionParser')
    # Calling OptionParser(args, kwargs) (line 334)
    OptionParser_call_result_377116 = invoke(stypy.reporting.localization.Localization(__file__, 334, 8), OptionParser_377109, *[], **kwargs_377115)
    
    # Assigning a type to the variable 'p' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'p', OptionParser_call_result_377116)
    
    # Call to add_option(...): (line 335)
    # Processing the call arguments (line 335)
    str_377119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 17), 'str', '--no-force')
    # Processing the call keyword arguments (line 335)
    str_377120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 38), 'str', 'store_false')
    keyword_377121 = str_377120
    str_377122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 22), 'str', 'force')
    keyword_377123 = str_377122
    # Getting the type of 'True' (line 336)
    True_377124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 39), 'True', False)
    keyword_377125 = True_377124
    kwargs_377126 = {'action': keyword_377121, 'dest': keyword_377123, 'default': keyword_377125}
    # Getting the type of 'p' (line 335)
    p_377117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'p', False)
    # Obtaining the member 'add_option' of a type (line 335)
    add_option_377118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 4), p_377117, 'add_option')
    # Calling add_option(args, kwargs) (line 335)
    add_option_call_result_377127 = invoke(stypy.reporting.localization.Localization(__file__, 335, 4), add_option_377118, *[str_377119], **kwargs_377126)
    
    
    # Assigning a Call to a Tuple (line 337):
    
    # Assigning a Subscript to a Name (line 337):
    
    # Obtaining the type of the subscript
    int_377128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 4), 'int')
    
    # Call to parse_args(...): (line 337)
    # Processing the call keyword arguments (line 337)
    kwargs_377131 = {}
    # Getting the type of 'p' (line 337)
    p_377129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 20), 'p', False)
    # Obtaining the member 'parse_args' of a type (line 337)
    parse_args_377130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 20), p_377129, 'parse_args')
    # Calling parse_args(args, kwargs) (line 337)
    parse_args_call_result_377132 = invoke(stypy.reporting.localization.Localization(__file__, 337, 20), parse_args_377130, *[], **kwargs_377131)
    
    # Obtaining the member '__getitem__' of a type (line 337)
    getitem___377133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 4), parse_args_call_result_377132, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 337)
    subscript_call_result_377134 = invoke(stypy.reporting.localization.Localization(__file__, 337, 4), getitem___377133, int_377128)
    
    # Assigning a type to the variable 'tuple_var_assignment_376716' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'tuple_var_assignment_376716', subscript_call_result_377134)
    
    # Assigning a Subscript to a Name (line 337):
    
    # Obtaining the type of the subscript
    int_377135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 4), 'int')
    
    # Call to parse_args(...): (line 337)
    # Processing the call keyword arguments (line 337)
    kwargs_377138 = {}
    # Getting the type of 'p' (line 337)
    p_377136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 20), 'p', False)
    # Obtaining the member 'parse_args' of a type (line 337)
    parse_args_377137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 20), p_377136, 'parse_args')
    # Calling parse_args(args, kwargs) (line 337)
    parse_args_call_result_377139 = invoke(stypy.reporting.localization.Localization(__file__, 337, 20), parse_args_377137, *[], **kwargs_377138)
    
    # Obtaining the member '__getitem__' of a type (line 337)
    getitem___377140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 4), parse_args_call_result_377139, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 337)
    subscript_call_result_377141 = invoke(stypy.reporting.localization.Localization(__file__, 337, 4), getitem___377140, int_377135)
    
    # Assigning a type to the variable 'tuple_var_assignment_376717' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'tuple_var_assignment_376717', subscript_call_result_377141)
    
    # Assigning a Name to a Name (line 337):
    # Getting the type of 'tuple_var_assignment_376716' (line 337)
    tuple_var_assignment_376716_377142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'tuple_var_assignment_376716')
    # Assigning a type to the variable 'options' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'options', tuple_var_assignment_376716_377142)
    
    # Assigning a Name to a Name (line 337):
    # Getting the type of 'tuple_var_assignment_376717' (line 337)
    tuple_var_assignment_376717_377143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'tuple_var_assignment_376717')
    # Assigning a type to the variable 'args' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 13), 'args', tuple_var_assignment_376717_377143)
    
    # Assigning a List to a Name (line 339):
    
    # Assigning a List to a Name (line 339):
    
    # Obtaining an instance of the builtin type 'list' (line 339)
    list_377144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 339)
    
    # Assigning a type to the variable 'names' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'names', list_377144)
    
    # Assigning a Call to a Tuple (line 341):
    
    # Assigning a Subscript to a Name (line 341):
    
    # Obtaining the type of the subscript
    int_377145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 4), 'int')
    
    # Call to get_thunk_type_set(...): (line 341)
    # Processing the call keyword arguments (line 341)
    kwargs_377147 = {}
    # Getting the type of 'get_thunk_type_set' (line 341)
    get_thunk_type_set_377146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 37), 'get_thunk_type_set', False)
    # Calling get_thunk_type_set(args, kwargs) (line 341)
    get_thunk_type_set_call_result_377148 = invoke(stypy.reporting.localization.Localization(__file__, 341, 37), get_thunk_type_set_377146, *[], **kwargs_377147)
    
    # Obtaining the member '__getitem__' of a type (line 341)
    getitem___377149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 4), get_thunk_type_set_call_result_377148, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 341)
    subscript_call_result_377150 = invoke(stypy.reporting.localization.Localization(__file__, 341, 4), getitem___377149, int_377145)
    
    # Assigning a type to the variable 'tuple_var_assignment_376718' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'tuple_var_assignment_376718', subscript_call_result_377150)
    
    # Assigning a Subscript to a Name (line 341):
    
    # Obtaining the type of the subscript
    int_377151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 4), 'int')
    
    # Call to get_thunk_type_set(...): (line 341)
    # Processing the call keyword arguments (line 341)
    kwargs_377153 = {}
    # Getting the type of 'get_thunk_type_set' (line 341)
    get_thunk_type_set_377152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 37), 'get_thunk_type_set', False)
    # Calling get_thunk_type_set(args, kwargs) (line 341)
    get_thunk_type_set_call_result_377154 = invoke(stypy.reporting.localization.Localization(__file__, 341, 37), get_thunk_type_set_377152, *[], **kwargs_377153)
    
    # Obtaining the member '__getitem__' of a type (line 341)
    getitem___377155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 4), get_thunk_type_set_call_result_377154, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 341)
    subscript_call_result_377156 = invoke(stypy.reporting.localization.Localization(__file__, 341, 4), getitem___377155, int_377151)
    
    # Assigning a type to the variable 'tuple_var_assignment_376719' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'tuple_var_assignment_376719', subscript_call_result_377156)
    
    # Assigning a Subscript to a Name (line 341):
    
    # Obtaining the type of the subscript
    int_377157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 4), 'int')
    
    # Call to get_thunk_type_set(...): (line 341)
    # Processing the call keyword arguments (line 341)
    kwargs_377159 = {}
    # Getting the type of 'get_thunk_type_set' (line 341)
    get_thunk_type_set_377158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 37), 'get_thunk_type_set', False)
    # Calling get_thunk_type_set(args, kwargs) (line 341)
    get_thunk_type_set_call_result_377160 = invoke(stypy.reporting.localization.Localization(__file__, 341, 37), get_thunk_type_set_377158, *[], **kwargs_377159)
    
    # Obtaining the member '__getitem__' of a type (line 341)
    getitem___377161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 4), get_thunk_type_set_call_result_377160, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 341)
    subscript_call_result_377162 = invoke(stypy.reporting.localization.Localization(__file__, 341, 4), getitem___377161, int_377157)
    
    # Assigning a type to the variable 'tuple_var_assignment_376720' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'tuple_var_assignment_376720', subscript_call_result_377162)
    
    # Assigning a Name to a Name (line 341):
    # Getting the type of 'tuple_var_assignment_376718' (line 341)
    tuple_var_assignment_376718_377163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'tuple_var_assignment_376718')
    # Assigning a type to the variable 'i_types' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'i_types', tuple_var_assignment_376718_377163)
    
    # Assigning a Name to a Name (line 341):
    # Getting the type of 'tuple_var_assignment_376719' (line 341)
    tuple_var_assignment_376719_377164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'tuple_var_assignment_376719')
    # Assigning a type to the variable 'it_types' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 13), 'it_types', tuple_var_assignment_376719_377164)
    
    # Assigning a Name to a Name (line 341):
    # Getting the type of 'tuple_var_assignment_376720' (line 341)
    tuple_var_assignment_376720_377165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'tuple_var_assignment_376720')
    # Assigning a type to the variable 'getter_code' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 23), 'getter_code', tuple_var_assignment_376720_377165)
    
    # Getting the type of 'COMPILATION_UNITS' (line 344)
    COMPILATION_UNITS_377166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 31), 'COMPILATION_UNITS')
    # Testing the type of a for loop iterable (line 344)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 344, 4), COMPILATION_UNITS_377166)
    # Getting the type of the for loop variable (line 344)
    for_loop_var_377167 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 344, 4), COMPILATION_UNITS_377166)
    # Assigning a type to the variable 'unit_name' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'unit_name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 4), for_loop_var_377167))
    # Assigning a type to the variable 'routines' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'routines', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 4), for_loop_var_377167))
    # SSA begins for a for statement (line 344)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a List to a Name (line 345):
    
    # Assigning a List to a Name (line 345):
    
    # Obtaining an instance of the builtin type 'list' (line 345)
    list_377168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 345)
    
    # Assigning a type to the variable 'thunks' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'thunks', list_377168)
    
    # Assigning a List to a Name (line 346):
    
    # Assigning a List to a Name (line 346):
    
    # Obtaining an instance of the builtin type 'list' (line 346)
    list_377169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 346)
    
    # Assigning a type to the variable 'methods' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'methods', list_377169)
    
    
    # Call to splitlines(...): (line 349)
    # Processing the call keyword arguments (line 349)
    kwargs_377172 = {}
    # Getting the type of 'routines' (line 349)
    routines_377170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 20), 'routines', False)
    # Obtaining the member 'splitlines' of a type (line 349)
    splitlines_377171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 20), routines_377170, 'splitlines')
    # Calling splitlines(args, kwargs) (line 349)
    splitlines_call_result_377173 = invoke(stypy.reporting.localization.Localization(__file__, 349, 20), splitlines_377171, *[], **kwargs_377172)
    
    # Testing the type of a for loop iterable (line 349)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 349, 8), splitlines_call_result_377173)
    # Getting the type of the for loop variable (line 349)
    for_loop_var_377174 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 349, 8), splitlines_call_result_377173)
    # Assigning a type to the variable 'line' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'line', for_loop_var_377174)
    # SSA begins for a for statement (line 349)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 350):
    
    # Assigning a Call to a Name (line 350):
    
    # Call to strip(...): (line 350)
    # Processing the call keyword arguments (line 350)
    kwargs_377177 = {}
    # Getting the type of 'line' (line 350)
    line_377175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 19), 'line', False)
    # Obtaining the member 'strip' of a type (line 350)
    strip_377176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 19), line_377175, 'strip')
    # Calling strip(args, kwargs) (line 350)
    strip_call_result_377178 = invoke(stypy.reporting.localization.Localization(__file__, 350, 19), strip_377176, *[], **kwargs_377177)
    
    # Assigning a type to the variable 'line' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'line', strip_call_result_377178)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'line' (line 351)
    line_377179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 19), 'line')
    # Applying the 'not' unary operator (line 351)
    result_not__377180 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 15), 'not', line_377179)
    
    
    # Call to startswith(...): (line 351)
    # Processing the call arguments (line 351)
    str_377183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 43), 'str', '#')
    # Processing the call keyword arguments (line 351)
    kwargs_377184 = {}
    # Getting the type of 'line' (line 351)
    line_377181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 27), 'line', False)
    # Obtaining the member 'startswith' of a type (line 351)
    startswith_377182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 27), line_377181, 'startswith')
    # Calling startswith(args, kwargs) (line 351)
    startswith_call_result_377185 = invoke(stypy.reporting.localization.Localization(__file__, 351, 27), startswith_377182, *[str_377183], **kwargs_377184)
    
    # Applying the binary operator 'or' (line 351)
    result_or_keyword_377186 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 15), 'or', result_not__377180, startswith_call_result_377185)
    
    # Testing the type of an if condition (line 351)
    if_condition_377187 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 351, 12), result_or_keyword_377186)
    # Assigning a type to the variable 'if_condition_377187' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'if_condition_377187', if_condition_377187)
    # SSA begins for if statement (line 351)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 351)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 354)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Tuple (line 355):
    
    # Assigning a Subscript to a Name (line 355):
    
    # Obtaining the type of the subscript
    int_377188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 16), 'int')
    
    # Call to split(...): (line 355)
    # Processing the call arguments (line 355)
    # Getting the type of 'None' (line 355)
    None_377191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 40), 'None', False)
    int_377192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 46), 'int')
    # Processing the call keyword arguments (line 355)
    kwargs_377193 = {}
    # Getting the type of 'line' (line 355)
    line_377189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 29), 'line', False)
    # Obtaining the member 'split' of a type (line 355)
    split_377190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 29), line_377189, 'split')
    # Calling split(args, kwargs) (line 355)
    split_call_result_377194 = invoke(stypy.reporting.localization.Localization(__file__, 355, 29), split_377190, *[None_377191, int_377192], **kwargs_377193)
    
    # Obtaining the member '__getitem__' of a type (line 355)
    getitem___377195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 16), split_call_result_377194, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 355)
    subscript_call_result_377196 = invoke(stypy.reporting.localization.Localization(__file__, 355, 16), getitem___377195, int_377188)
    
    # Assigning a type to the variable 'tuple_var_assignment_376721' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 16), 'tuple_var_assignment_376721', subscript_call_result_377196)
    
    # Assigning a Subscript to a Name (line 355):
    
    # Obtaining the type of the subscript
    int_377197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 16), 'int')
    
    # Call to split(...): (line 355)
    # Processing the call arguments (line 355)
    # Getting the type of 'None' (line 355)
    None_377200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 40), 'None', False)
    int_377201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 46), 'int')
    # Processing the call keyword arguments (line 355)
    kwargs_377202 = {}
    # Getting the type of 'line' (line 355)
    line_377198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 29), 'line', False)
    # Obtaining the member 'split' of a type (line 355)
    split_377199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 29), line_377198, 'split')
    # Calling split(args, kwargs) (line 355)
    split_call_result_377203 = invoke(stypy.reporting.localization.Localization(__file__, 355, 29), split_377199, *[None_377200, int_377201], **kwargs_377202)
    
    # Obtaining the member '__getitem__' of a type (line 355)
    getitem___377204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 16), split_call_result_377203, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 355)
    subscript_call_result_377205 = invoke(stypy.reporting.localization.Localization(__file__, 355, 16), getitem___377204, int_377197)
    
    # Assigning a type to the variable 'tuple_var_assignment_376722' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 16), 'tuple_var_assignment_376722', subscript_call_result_377205)
    
    # Assigning a Name to a Name (line 355):
    # Getting the type of 'tuple_var_assignment_376721' (line 355)
    tuple_var_assignment_376721_377206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 16), 'tuple_var_assignment_376721')
    # Assigning a type to the variable 'name' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 16), 'name', tuple_var_assignment_376721_377206)
    
    # Assigning a Name to a Name (line 355):
    # Getting the type of 'tuple_var_assignment_376722' (line 355)
    tuple_var_assignment_376722_377207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 16), 'tuple_var_assignment_376722')
    # Assigning a type to the variable 'args' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 22), 'args', tuple_var_assignment_376722_377207)
    # SSA branch for the except part of a try statement (line 354)
    # SSA branch for the except 'ValueError' branch of a try statement (line 354)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 357)
    # Processing the call arguments (line 357)
    str_377209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 33), 'str', 'Malformed line: %r')
    
    # Obtaining an instance of the builtin type 'tuple' (line 357)
    tuple_377210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 57), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 357)
    # Adding element type (line 357)
    # Getting the type of 'line' (line 357)
    line_377211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 57), 'line', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 57), tuple_377210, line_377211)
    
    # Applying the binary operator '%' (line 357)
    result_mod_377212 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 33), '%', str_377209, tuple_377210)
    
    # Processing the call keyword arguments (line 357)
    kwargs_377213 = {}
    # Getting the type of 'ValueError' (line 357)
    ValueError_377208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 357)
    ValueError_call_result_377214 = invoke(stypy.reporting.localization.Localization(__file__, 357, 22), ValueError_377208, *[result_mod_377212], **kwargs_377213)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 357, 16), ValueError_call_result_377214, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 354)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 359):
    
    # Assigning a Call to a Name (line 359):
    
    # Call to join(...): (line 359)
    # Processing the call arguments (line 359)
    
    # Call to split(...): (line 359)
    # Processing the call keyword arguments (line 359)
    kwargs_377219 = {}
    # Getting the type of 'args' (line 359)
    args_377217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 27), 'args', False)
    # Obtaining the member 'split' of a type (line 359)
    split_377218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 27), args_377217, 'split')
    # Calling split(args, kwargs) (line 359)
    split_call_result_377220 = invoke(stypy.reporting.localization.Localization(__file__, 359, 27), split_377218, *[], **kwargs_377219)
    
    # Processing the call keyword arguments (line 359)
    kwargs_377221 = {}
    str_377215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 19), 'str', '')
    # Obtaining the member 'join' of a type (line 359)
    join_377216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 19), str_377215, 'join')
    # Calling join(args, kwargs) (line 359)
    join_call_result_377222 = invoke(stypy.reporting.localization.Localization(__file__, 359, 19), join_377216, *[split_call_result_377220], **kwargs_377221)
    
    # Assigning a type to the variable 'args' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'args', join_call_result_377222)
    
    
    # Evaluating a boolean operation
    
    str_377223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 15), 'str', 't')
    # Getting the type of 'args' (line 360)
    args_377224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 22), 'args')
    # Applying the binary operator 'in' (line 360)
    result_contains_377225 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 15), 'in', str_377223, args_377224)
    
    
    str_377226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 30), 'str', 'T')
    # Getting the type of 'args' (line 360)
    args_377227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 37), 'args')
    # Applying the binary operator 'in' (line 360)
    result_contains_377228 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 30), 'in', str_377226, args_377227)
    
    # Applying the binary operator 'or' (line 360)
    result_or_keyword_377229 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 15), 'or', result_contains_377225, result_contains_377228)
    
    # Testing the type of an if condition (line 360)
    if_condition_377230 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 360, 12), result_or_keyword_377229)
    # Assigning a type to the variable 'if_condition_377230' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'if_condition_377230', if_condition_377230)
    # SSA begins for if statement (line 360)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 361):
    
    # Assigning a Subscript to a Name (line 361):
    
    # Obtaining the type of the subscript
    int_377231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 16), 'int')
    
    # Call to parse_routine(...): (line 361)
    # Processing the call arguments (line 361)
    # Getting the type of 'name' (line 361)
    name_377233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 46), 'name', False)
    # Getting the type of 'args' (line 361)
    args_377234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 52), 'args', False)
    # Getting the type of 'it_types' (line 361)
    it_types_377235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 58), 'it_types', False)
    # Processing the call keyword arguments (line 361)
    kwargs_377236 = {}
    # Getting the type of 'parse_routine' (line 361)
    parse_routine_377232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 32), 'parse_routine', False)
    # Calling parse_routine(args, kwargs) (line 361)
    parse_routine_call_result_377237 = invoke(stypy.reporting.localization.Localization(__file__, 361, 32), parse_routine_377232, *[name_377233, args_377234, it_types_377235], **kwargs_377236)
    
    # Obtaining the member '__getitem__' of a type (line 361)
    getitem___377238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 16), parse_routine_call_result_377237, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 361)
    subscript_call_result_377239 = invoke(stypy.reporting.localization.Localization(__file__, 361, 16), getitem___377238, int_377231)
    
    # Assigning a type to the variable 'tuple_var_assignment_376723' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 16), 'tuple_var_assignment_376723', subscript_call_result_377239)
    
    # Assigning a Subscript to a Name (line 361):
    
    # Obtaining the type of the subscript
    int_377240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 16), 'int')
    
    # Call to parse_routine(...): (line 361)
    # Processing the call arguments (line 361)
    # Getting the type of 'name' (line 361)
    name_377242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 46), 'name', False)
    # Getting the type of 'args' (line 361)
    args_377243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 52), 'args', False)
    # Getting the type of 'it_types' (line 361)
    it_types_377244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 58), 'it_types', False)
    # Processing the call keyword arguments (line 361)
    kwargs_377245 = {}
    # Getting the type of 'parse_routine' (line 361)
    parse_routine_377241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 32), 'parse_routine', False)
    # Calling parse_routine(args, kwargs) (line 361)
    parse_routine_call_result_377246 = invoke(stypy.reporting.localization.Localization(__file__, 361, 32), parse_routine_377241, *[name_377242, args_377243, it_types_377244], **kwargs_377245)
    
    # Obtaining the member '__getitem__' of a type (line 361)
    getitem___377247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 16), parse_routine_call_result_377246, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 361)
    subscript_call_result_377248 = invoke(stypy.reporting.localization.Localization(__file__, 361, 16), getitem___377247, int_377240)
    
    # Assigning a type to the variable 'tuple_var_assignment_376724' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 16), 'tuple_var_assignment_376724', subscript_call_result_377248)
    
    # Assigning a Name to a Name (line 361):
    # Getting the type of 'tuple_var_assignment_376723' (line 361)
    tuple_var_assignment_376723_377249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 16), 'tuple_var_assignment_376723')
    # Assigning a type to the variable 'thunk' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 16), 'thunk', tuple_var_assignment_376723_377249)
    
    # Assigning a Name to a Name (line 361):
    # Getting the type of 'tuple_var_assignment_376724' (line 361)
    tuple_var_assignment_376724_377250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 16), 'tuple_var_assignment_376724')
    # Assigning a type to the variable 'method' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 23), 'method', tuple_var_assignment_376724_377250)
    # SSA branch for the else part of an if statement (line 360)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 363):
    
    # Assigning a Subscript to a Name (line 363):
    
    # Obtaining the type of the subscript
    int_377251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 16), 'int')
    
    # Call to parse_routine(...): (line 363)
    # Processing the call arguments (line 363)
    # Getting the type of 'name' (line 363)
    name_377253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 46), 'name', False)
    # Getting the type of 'args' (line 363)
    args_377254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 52), 'args', False)
    # Getting the type of 'i_types' (line 363)
    i_types_377255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 58), 'i_types', False)
    # Processing the call keyword arguments (line 363)
    kwargs_377256 = {}
    # Getting the type of 'parse_routine' (line 363)
    parse_routine_377252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 32), 'parse_routine', False)
    # Calling parse_routine(args, kwargs) (line 363)
    parse_routine_call_result_377257 = invoke(stypy.reporting.localization.Localization(__file__, 363, 32), parse_routine_377252, *[name_377253, args_377254, i_types_377255], **kwargs_377256)
    
    # Obtaining the member '__getitem__' of a type (line 363)
    getitem___377258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 16), parse_routine_call_result_377257, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 363)
    subscript_call_result_377259 = invoke(stypy.reporting.localization.Localization(__file__, 363, 16), getitem___377258, int_377251)
    
    # Assigning a type to the variable 'tuple_var_assignment_376725' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 16), 'tuple_var_assignment_376725', subscript_call_result_377259)
    
    # Assigning a Subscript to a Name (line 363):
    
    # Obtaining the type of the subscript
    int_377260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 16), 'int')
    
    # Call to parse_routine(...): (line 363)
    # Processing the call arguments (line 363)
    # Getting the type of 'name' (line 363)
    name_377262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 46), 'name', False)
    # Getting the type of 'args' (line 363)
    args_377263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 52), 'args', False)
    # Getting the type of 'i_types' (line 363)
    i_types_377264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 58), 'i_types', False)
    # Processing the call keyword arguments (line 363)
    kwargs_377265 = {}
    # Getting the type of 'parse_routine' (line 363)
    parse_routine_377261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 32), 'parse_routine', False)
    # Calling parse_routine(args, kwargs) (line 363)
    parse_routine_call_result_377266 = invoke(stypy.reporting.localization.Localization(__file__, 363, 32), parse_routine_377261, *[name_377262, args_377263, i_types_377264], **kwargs_377265)
    
    # Obtaining the member '__getitem__' of a type (line 363)
    getitem___377267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 16), parse_routine_call_result_377266, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 363)
    subscript_call_result_377268 = invoke(stypy.reporting.localization.Localization(__file__, 363, 16), getitem___377267, int_377260)
    
    # Assigning a type to the variable 'tuple_var_assignment_376726' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 16), 'tuple_var_assignment_376726', subscript_call_result_377268)
    
    # Assigning a Name to a Name (line 363):
    # Getting the type of 'tuple_var_assignment_376725' (line 363)
    tuple_var_assignment_376725_377269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 16), 'tuple_var_assignment_376725')
    # Assigning a type to the variable 'thunk' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 16), 'thunk', tuple_var_assignment_376725_377269)
    
    # Assigning a Name to a Name (line 363):
    # Getting the type of 'tuple_var_assignment_376726' (line 363)
    tuple_var_assignment_376726_377270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 16), 'tuple_var_assignment_376726')
    # Assigning a type to the variable 'method' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 23), 'method', tuple_var_assignment_376726_377270)
    # SSA join for if statement (line 360)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'name' (line 365)
    name_377271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 15), 'name')
    # Getting the type of 'names' (line 365)
    names_377272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 23), 'names')
    # Applying the binary operator 'in' (line 365)
    result_contains_377273 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 15), 'in', name_377271, names_377272)
    
    # Testing the type of an if condition (line 365)
    if_condition_377274 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 365, 12), result_contains_377273)
    # Assigning a type to the variable 'if_condition_377274' (line 365)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'if_condition_377274', if_condition_377274)
    # SSA begins for if statement (line 365)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 366)
    # Processing the call arguments (line 366)
    str_377276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 33), 'str', 'Duplicate routine %r')
    
    # Obtaining an instance of the builtin type 'tuple' (line 366)
    tuple_377277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 59), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 366)
    # Adding element type (line 366)
    # Getting the type of 'name' (line 366)
    name_377278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 59), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 59), tuple_377277, name_377278)
    
    # Applying the binary operator '%' (line 366)
    result_mod_377279 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 33), '%', str_377276, tuple_377277)
    
    # Processing the call keyword arguments (line 366)
    kwargs_377280 = {}
    # Getting the type of 'ValueError' (line 366)
    ValueError_377275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 366)
    ValueError_call_result_377281 = invoke(stypy.reporting.localization.Localization(__file__, 366, 22), ValueError_377275, *[result_mod_377279], **kwargs_377280)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 366, 16), ValueError_call_result_377281, 'raise parameter', BaseException)
    # SSA join for if statement (line 365)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 368)
    # Processing the call arguments (line 368)
    # Getting the type of 'name' (line 368)
    name_377284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 25), 'name', False)
    # Processing the call keyword arguments (line 368)
    kwargs_377285 = {}
    # Getting the type of 'names' (line 368)
    names_377282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'names', False)
    # Obtaining the member 'append' of a type (line 368)
    append_377283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 12), names_377282, 'append')
    # Calling append(args, kwargs) (line 368)
    append_call_result_377286 = invoke(stypy.reporting.localization.Localization(__file__, 368, 12), append_377283, *[name_377284], **kwargs_377285)
    
    
    # Call to append(...): (line 369)
    # Processing the call arguments (line 369)
    # Getting the type of 'thunk' (line 369)
    thunk_377289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 26), 'thunk', False)
    # Processing the call keyword arguments (line 369)
    kwargs_377290 = {}
    # Getting the type of 'thunks' (line 369)
    thunks_377287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'thunks', False)
    # Obtaining the member 'append' of a type (line 369)
    append_377288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 12), thunks_377287, 'append')
    # Calling append(args, kwargs) (line 369)
    append_call_result_377291 = invoke(stypy.reporting.localization.Localization(__file__, 369, 12), append_377288, *[thunk_377289], **kwargs_377290)
    
    
    # Call to append(...): (line 370)
    # Processing the call arguments (line 370)
    # Getting the type of 'method' (line 370)
    method_377294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 27), 'method', False)
    # Processing the call keyword arguments (line 370)
    kwargs_377295 = {}
    # Getting the type of 'methods' (line 370)
    methods_377292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'methods', False)
    # Obtaining the member 'append' of a type (line 370)
    append_377293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 12), methods_377292, 'append')
    # Calling append(args, kwargs) (line 370)
    append_call_result_377296 = invoke(stypy.reporting.localization.Localization(__file__, 370, 12), append_377293, *[method_377294], **kwargs_377295)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 373):
    
    # Assigning a Call to a Name (line 373):
    
    # Call to join(...): (line 373)
    # Processing the call arguments (line 373)
    
    # Call to dirname(...): (line 373)
    # Processing the call arguments (line 373)
    # Getting the type of '__file__' (line 373)
    file___377303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 43), '__file__', False)
    # Processing the call keyword arguments (line 373)
    kwargs_377304 = {}
    # Getting the type of 'os' (line 373)
    os_377300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 27), 'os', False)
    # Obtaining the member 'path' of a type (line 373)
    path_377301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 27), os_377300, 'path')
    # Obtaining the member 'dirname' of a type (line 373)
    dirname_377302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 27), path_377301, 'dirname')
    # Calling dirname(args, kwargs) (line 373)
    dirname_call_result_377305 = invoke(stypy.reporting.localization.Localization(__file__, 373, 27), dirname_377302, *[file___377303], **kwargs_377304)
    
    str_377306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 27), 'str', 'sparsetools')
    # Getting the type of 'unit_name' (line 375)
    unit_name_377307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 27), 'unit_name', False)
    str_377308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 39), 'str', '_impl.h')
    # Applying the binary operator '+' (line 375)
    result_add_377309 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 27), '+', unit_name_377307, str_377308)
    
    # Processing the call keyword arguments (line 373)
    kwargs_377310 = {}
    # Getting the type of 'os' (line 373)
    os_377297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 14), 'os', False)
    # Obtaining the member 'path' of a type (line 373)
    path_377298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 14), os_377297, 'path')
    # Obtaining the member 'join' of a type (line 373)
    join_377299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 14), path_377298, 'join')
    # Calling join(args, kwargs) (line 373)
    join_call_result_377311 = invoke(stypy.reporting.localization.Localization(__file__, 373, 14), join_377299, *[dirname_call_result_377305, str_377306, result_add_377309], **kwargs_377310)
    
    # Assigning a type to the variable 'dst' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'dst', join_call_result_377311)
    
    
    # Evaluating a boolean operation
    
    # Call to newer(...): (line 376)
    # Processing the call arguments (line 376)
    # Getting the type of '__file__' (line 376)
    file___377313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 17), '__file__', False)
    # Getting the type of 'dst' (line 376)
    dst_377314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 27), 'dst', False)
    # Processing the call keyword arguments (line 376)
    kwargs_377315 = {}
    # Getting the type of 'newer' (line 376)
    newer_377312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 11), 'newer', False)
    # Calling newer(args, kwargs) (line 376)
    newer_call_result_377316 = invoke(stypy.reporting.localization.Localization(__file__, 376, 11), newer_377312, *[file___377313, dst_377314], **kwargs_377315)
    
    # Getting the type of 'options' (line 376)
    options_377317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 35), 'options')
    # Obtaining the member 'force' of a type (line 376)
    force_377318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 35), options_377317, 'force')
    # Applying the binary operator 'or' (line 376)
    result_or_keyword_377319 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 11), 'or', newer_call_result_377316, force_377318)
    
    # Testing the type of an if condition (line 376)
    if_condition_377320 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 376, 8), result_or_keyword_377319)
    # Assigning a type to the variable 'if_condition_377320' (line 376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'if_condition_377320', if_condition_377320)
    # SSA begins for if statement (line 376)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_377321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 18), 'str', '[generate_sparsetools] generating %r')
    
    # Obtaining an instance of the builtin type 'tuple' (line 377)
    tuple_377322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 60), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 377)
    # Adding element type (line 377)
    # Getting the type of 'dst' (line 377)
    dst_377323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 60), 'dst')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 60), tuple_377322, dst_377323)
    
    # Applying the binary operator '%' (line 377)
    result_mod_377324 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 18), '%', str_377321, tuple_377322)
    
    
    # Call to open(...): (line 378)
    # Processing the call arguments (line 378)
    # Getting the type of 'dst' (line 378)
    dst_377326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 22), 'dst', False)
    str_377327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 27), 'str', 'w')
    # Processing the call keyword arguments (line 378)
    kwargs_377328 = {}
    # Getting the type of 'open' (line 378)
    open_377325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 17), 'open', False)
    # Calling open(args, kwargs) (line 378)
    open_call_result_377329 = invoke(stypy.reporting.localization.Localization(__file__, 378, 17), open_377325, *[dst_377326, str_377327], **kwargs_377328)
    
    with_377330 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 378, 17), open_call_result_377329, 'with parameter', '__enter__', '__exit__')

    if with_377330:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 378)
        enter___377331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 17), open_call_result_377329, '__enter__')
        with_enter_377332 = invoke(stypy.reporting.localization.Localization(__file__, 378, 17), enter___377331)
        # Assigning a type to the variable 'f' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 17), 'f', with_enter_377332)
        
        # Call to write_autogen_blurb(...): (line 379)
        # Processing the call arguments (line 379)
        # Getting the type of 'f' (line 379)
        f_377334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 36), 'f', False)
        # Processing the call keyword arguments (line 379)
        kwargs_377335 = {}
        # Getting the type of 'write_autogen_blurb' (line 379)
        write_autogen_blurb_377333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 16), 'write_autogen_blurb', False)
        # Calling write_autogen_blurb(args, kwargs) (line 379)
        write_autogen_blurb_call_result_377336 = invoke(stypy.reporting.localization.Localization(__file__, 379, 16), write_autogen_blurb_377333, *[f_377334], **kwargs_377335)
        
        
        # Call to write(...): (line 380)
        # Processing the call arguments (line 380)
        # Getting the type of 'getter_code' (line 380)
        getter_code_377339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 24), 'getter_code', False)
        # Processing the call keyword arguments (line 380)
        kwargs_377340 = {}
        # Getting the type of 'f' (line 380)
        f_377337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 16), 'f', False)
        # Obtaining the member 'write' of a type (line 380)
        write_377338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 16), f_377337, 'write')
        # Calling write(args, kwargs) (line 380)
        write_call_result_377341 = invoke(stypy.reporting.localization.Localization(__file__, 380, 16), write_377338, *[getter_code_377339], **kwargs_377340)
        
        
        # Getting the type of 'thunks' (line 381)
        thunks_377342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 29), 'thunks')
        # Testing the type of a for loop iterable (line 381)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 381, 16), thunks_377342)
        # Getting the type of the for loop variable (line 381)
        for_loop_var_377343 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 381, 16), thunks_377342)
        # Assigning a type to the variable 'thunk' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 16), 'thunk', for_loop_var_377343)
        # SSA begins for a for statement (line 381)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to write(...): (line 382)
        # Processing the call arguments (line 382)
        # Getting the type of 'thunk' (line 382)
        thunk_377346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 28), 'thunk', False)
        # Processing the call keyword arguments (line 382)
        kwargs_377347 = {}
        # Getting the type of 'f' (line 382)
        f_377344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 20), 'f', False)
        # Obtaining the member 'write' of a type (line 382)
        write_377345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 20), f_377344, 'write')
        # Calling write(args, kwargs) (line 382)
        write_call_result_377348 = invoke(stypy.reporting.localization.Localization(__file__, 382, 20), write_377345, *[thunk_377346], **kwargs_377347)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'methods' (line 383)
        methods_377349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 30), 'methods')
        # Testing the type of a for loop iterable (line 383)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 383, 16), methods_377349)
        # Getting the type of the for loop variable (line 383)
        for_loop_var_377350 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 383, 16), methods_377349)
        # Assigning a type to the variable 'method' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 16), 'method', for_loop_var_377350)
        # SSA begins for a for statement (line 383)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to write(...): (line 384)
        # Processing the call arguments (line 384)
        # Getting the type of 'method' (line 384)
        method_377353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 28), 'method', False)
        # Processing the call keyword arguments (line 384)
        kwargs_377354 = {}
        # Getting the type of 'f' (line 384)
        f_377351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 20), 'f', False)
        # Obtaining the member 'write' of a type (line 384)
        write_377352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 20), f_377351, 'write')
        # Calling write(args, kwargs) (line 384)
        write_call_result_377355 = invoke(stypy.reporting.localization.Localization(__file__, 384, 20), write_377352, *[method_377353], **kwargs_377354)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 378)
        exit___377356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 17), open_call_result_377329, '__exit__')
        with_exit_377357 = invoke(stypy.reporting.localization.Localization(__file__, 378, 17), exit___377356, None, None, None)

    # SSA branch for the else part of an if statement (line 376)
    module_type_store.open_ssa_branch('else')
    str_377358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 18), 'str', '[generate_sparsetools] %r already up-to-date')
    
    # Obtaining an instance of the builtin type 'tuple' (line 386)
    tuple_377359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 68), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 386)
    # Adding element type (line 386)
    # Getting the type of 'dst' (line 386)
    dst_377360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 68), 'dst')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 68), tuple_377359, dst_377360)
    
    # Applying the binary operator '%' (line 386)
    result_mod_377361 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 18), '%', str_377358, tuple_377359)
    
    # SSA join for if statement (line 376)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Str to a Name (line 389):
    
    # Assigning a Str to a Name (line 389):
    str_377362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 18), 'str', '')
    # Assigning a type to the variable 'method_defs' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'method_defs', str_377362)
    
    # Getting the type of 'names' (line 390)
    names_377363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 16), 'names')
    # Testing the type of a for loop iterable (line 390)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 390, 4), names_377363)
    # Getting the type of the for loop variable (line 390)
    for_loop_var_377364 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 390, 4), names_377363)
    # Assigning a type to the variable 'name' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'name', for_loop_var_377364)
    # SSA begins for a for statement (line 390)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'method_defs' (line 391)
    method_defs_377365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'method_defs')
    str_377366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 23), 'str', 'NPY_VISIBILITY_HIDDEN PyObject *%s_method(PyObject *, PyObject *);\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 391)
    tuple_377367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 97), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 391)
    # Adding element type (line 391)
    # Getting the type of 'name' (line 391)
    name_377368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 97), 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 97), tuple_377367, name_377368)
    
    # Applying the binary operator '%' (line 391)
    result_mod_377369 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 23), '%', str_377366, tuple_377367)
    
    # Applying the binary operator '+=' (line 391)
    result_iadd_377370 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 8), '+=', method_defs_377365, result_mod_377369)
    # Assigning a type to the variable 'method_defs' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'method_defs', result_iadd_377370)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Str to a Name (line 393):
    
    # Assigning a Str to a Name (line 393):
    str_377371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 20), 'str', '\nstatic struct PyMethodDef sparsetools_methods[] = {')
    # Assigning a type to the variable 'method_struct' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'method_struct', str_377371)
    
    # Getting the type of 'names' (line 394)
    names_377372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 16), 'names')
    # Testing the type of a for loop iterable (line 394)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 394, 4), names_377372)
    # Getting the type of the for loop variable (line 394)
    for_loop_var_377373 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 394, 4), names_377372)
    # Assigning a type to the variable 'name' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'name', for_loop_var_377373)
    # SSA begins for a for statement (line 394)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'method_struct' (line 395)
    method_struct_377374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'method_struct')
    str_377375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, (-1)), 'str', '\n        {"%(name)s", (PyCFunction)%(name)s_method, METH_VARARGS, NULL},')
    
    # Call to dict(...): (line 396)
    # Processing the call keyword arguments (line 396)
    # Getting the type of 'name' (line 396)
    name_377377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 87), 'name', False)
    keyword_377378 = name_377377
    kwargs_377379 = {'name': keyword_377378}
    # Getting the type of 'dict' (line 396)
    dict_377376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 77), 'dict', False)
    # Calling dict(args, kwargs) (line 396)
    dict_call_result_377380 = invoke(stypy.reporting.localization.Localization(__file__, 396, 77), dict_377376, *[], **kwargs_377379)
    
    # Applying the binary operator '%' (line 396)
    result_mod_377381 = python_operator(stypy.reporting.localization.Localization(__file__, 396, (-1)), '%', str_377375, dict_call_result_377380)
    
    # Applying the binary operator '+=' (line 395)
    result_iadd_377382 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 8), '+=', method_struct_377374, result_mod_377381)
    # Assigning a type to the variable 'method_struct' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'method_struct', result_iadd_377382)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'method_struct' (line 397)
    method_struct_377383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'method_struct')
    str_377384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, (-1)), 'str', '\n        {NULL, NULL, 0, NULL}\n    };')
    # Applying the binary operator '+=' (line 397)
    result_iadd_377385 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 4), '+=', method_struct_377383, str_377384)
    # Assigning a type to the variable 'method_struct' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'method_struct', result_iadd_377385)
    
    
    # Assigning a Call to a Name (line 402):
    
    # Assigning a Call to a Name (line 402):
    
    # Call to join(...): (line 402)
    # Processing the call arguments (line 402)
    
    # Call to dirname(...): (line 402)
    # Processing the call arguments (line 402)
    # Getting the type of '__file__' (line 402)
    file___377392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 39), '__file__', False)
    # Processing the call keyword arguments (line 402)
    kwargs_377393 = {}
    # Getting the type of 'os' (line 402)
    os_377389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 23), 'os', False)
    # Obtaining the member 'path' of a type (line 402)
    path_377390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 23), os_377389, 'path')
    # Obtaining the member 'dirname' of a type (line 402)
    dirname_377391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 23), path_377390, 'dirname')
    # Calling dirname(args, kwargs) (line 402)
    dirname_call_result_377394 = invoke(stypy.reporting.localization.Localization(__file__, 402, 23), dirname_377391, *[file___377392], **kwargs_377393)
    
    str_377395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 23), 'str', 'sparsetools')
    str_377396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 23), 'str', 'sparsetools_impl.h')
    # Processing the call keyword arguments (line 402)
    kwargs_377397 = {}
    # Getting the type of 'os' (line 402)
    os_377386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 10), 'os', False)
    # Obtaining the member 'path' of a type (line 402)
    path_377387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 10), os_377386, 'path')
    # Obtaining the member 'join' of a type (line 402)
    join_377388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 10), path_377387, 'join')
    # Calling join(args, kwargs) (line 402)
    join_call_result_377398 = invoke(stypy.reporting.localization.Localization(__file__, 402, 10), join_377388, *[dirname_call_result_377394, str_377395, str_377396], **kwargs_377397)
    
    # Assigning a type to the variable 'dst' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'dst', join_call_result_377398)
    
    
    # Evaluating a boolean operation
    
    # Call to newer(...): (line 406)
    # Processing the call arguments (line 406)
    # Getting the type of '__file__' (line 406)
    file___377400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 13), '__file__', False)
    # Getting the type of 'dst' (line 406)
    dst_377401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 23), 'dst', False)
    # Processing the call keyword arguments (line 406)
    kwargs_377402 = {}
    # Getting the type of 'newer' (line 406)
    newer_377399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 7), 'newer', False)
    # Calling newer(args, kwargs) (line 406)
    newer_call_result_377403 = invoke(stypy.reporting.localization.Localization(__file__, 406, 7), newer_377399, *[file___377400, dst_377401], **kwargs_377402)
    
    # Getting the type of 'options' (line 406)
    options_377404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 31), 'options')
    # Obtaining the member 'force' of a type (line 406)
    force_377405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 31), options_377404, 'force')
    # Applying the binary operator 'or' (line 406)
    result_or_keyword_377406 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 7), 'or', newer_call_result_377403, force_377405)
    
    # Testing the type of an if condition (line 406)
    if_condition_377407 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 406, 4), result_or_keyword_377406)
    # Assigning a type to the variable 'if_condition_377407' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'if_condition_377407', if_condition_377407)
    # SSA begins for if statement (line 406)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_377408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 14), 'str', '[generate_sparsetools] generating %r')
    
    # Obtaining an instance of the builtin type 'tuple' (line 407)
    tuple_377409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 56), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 407)
    # Adding element type (line 407)
    # Getting the type of 'dst' (line 407)
    dst_377410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 56), 'dst')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 56), tuple_377409, dst_377410)
    
    # Applying the binary operator '%' (line 407)
    result_mod_377411 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 14), '%', str_377408, tuple_377409)
    
    
    # Call to open(...): (line 408)
    # Processing the call arguments (line 408)
    # Getting the type of 'dst' (line 408)
    dst_377413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 18), 'dst', False)
    str_377414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 23), 'str', 'w')
    # Processing the call keyword arguments (line 408)
    kwargs_377415 = {}
    # Getting the type of 'open' (line 408)
    open_377412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 13), 'open', False)
    # Calling open(args, kwargs) (line 408)
    open_call_result_377416 = invoke(stypy.reporting.localization.Localization(__file__, 408, 13), open_377412, *[dst_377413, str_377414], **kwargs_377415)
    
    with_377417 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 408, 13), open_call_result_377416, 'with parameter', '__enter__', '__exit__')

    if with_377417:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 408)
        enter___377418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 13), open_call_result_377416, '__enter__')
        with_enter_377419 = invoke(stypy.reporting.localization.Localization(__file__, 408, 13), enter___377418)
        # Assigning a type to the variable 'f' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 13), 'f', with_enter_377419)
        
        # Call to write_autogen_blurb(...): (line 409)
        # Processing the call arguments (line 409)
        # Getting the type of 'f' (line 409)
        f_377421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 32), 'f', False)
        # Processing the call keyword arguments (line 409)
        kwargs_377422 = {}
        # Getting the type of 'write_autogen_blurb' (line 409)
        write_autogen_blurb_377420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'write_autogen_blurb', False)
        # Calling write_autogen_blurb(args, kwargs) (line 409)
        write_autogen_blurb_call_result_377423 = invoke(stypy.reporting.localization.Localization(__file__, 409, 12), write_autogen_blurb_377420, *[f_377421], **kwargs_377422)
        
        
        # Call to write(...): (line 410)
        # Processing the call arguments (line 410)
        # Getting the type of 'method_defs' (line 410)
        method_defs_377426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 20), 'method_defs', False)
        # Processing the call keyword arguments (line 410)
        kwargs_377427 = {}
        # Getting the type of 'f' (line 410)
        f_377424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'f', False)
        # Obtaining the member 'write' of a type (line 410)
        write_377425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 12), f_377424, 'write')
        # Calling write(args, kwargs) (line 410)
        write_call_result_377428 = invoke(stypy.reporting.localization.Localization(__file__, 410, 12), write_377425, *[method_defs_377426], **kwargs_377427)
        
        
        # Call to write(...): (line 411)
        # Processing the call arguments (line 411)
        # Getting the type of 'method_struct' (line 411)
        method_struct_377431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 20), 'method_struct', False)
        # Processing the call keyword arguments (line 411)
        kwargs_377432 = {}
        # Getting the type of 'f' (line 411)
        f_377429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 12), 'f', False)
        # Obtaining the member 'write' of a type (line 411)
        write_377430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 12), f_377429, 'write')
        # Calling write(args, kwargs) (line 411)
        write_call_result_377433 = invoke(stypy.reporting.localization.Localization(__file__, 411, 12), write_377430, *[method_struct_377431], **kwargs_377432)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 408)
        exit___377434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 13), open_call_result_377416, '__exit__')
        with_exit_377435 = invoke(stypy.reporting.localization.Localization(__file__, 408, 13), exit___377434, None, None, None)

    # SSA branch for the else part of an if statement (line 406)
    module_type_store.open_ssa_branch('else')
    str_377436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 14), 'str', '[generate_sparsetools] %r already up-to-date')
    
    # Obtaining an instance of the builtin type 'tuple' (line 413)
    tuple_377437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 64), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 413)
    # Adding element type (line 413)
    # Getting the type of 'dst' (line 413)
    dst_377438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 64), 'dst')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 64), tuple_377437, dst_377438)
    
    # Applying the binary operator '%' (line 413)
    result_mod_377439 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 14), '%', str_377436, tuple_377437)
    
    # SSA join for if statement (line 406)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 333)
    stypy_return_type_377440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_377440)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_377440

# Assigning a type to the variable 'main' (line 333)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 0), 'main', main)

@norecursion
def write_autogen_blurb(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'write_autogen_blurb'
    module_type_store = module_type_store.open_function_context('write_autogen_blurb', 416, 0, False)
    
    # Passed parameters checking function
    write_autogen_blurb.stypy_localization = localization
    write_autogen_blurb.stypy_type_of_self = None
    write_autogen_blurb.stypy_type_store = module_type_store
    write_autogen_blurb.stypy_function_name = 'write_autogen_blurb'
    write_autogen_blurb.stypy_param_names_list = ['stream']
    write_autogen_blurb.stypy_varargs_param_name = None
    write_autogen_blurb.stypy_kwargs_param_name = None
    write_autogen_blurb.stypy_call_defaults = defaults
    write_autogen_blurb.stypy_call_varargs = varargs
    write_autogen_blurb.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'write_autogen_blurb', ['stream'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'write_autogen_blurb', localization, ['stream'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'write_autogen_blurb(...)' code ##################

    
    # Call to write(...): (line 417)
    # Processing the call arguments (line 417)
    str_377443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, (-1)), 'str', '/* This file is autogenerated by generate_sparsetools.py\n * Do not edit manually or check into VCS.\n */\n')
    # Processing the call keyword arguments (line 417)
    kwargs_377444 = {}
    # Getting the type of 'stream' (line 417)
    stream_377441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'stream', False)
    # Obtaining the member 'write' of a type (line 417)
    write_377442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 4), stream_377441, 'write')
    # Calling write(args, kwargs) (line 417)
    write_call_result_377445 = invoke(stypy.reporting.localization.Localization(__file__, 417, 4), write_377442, *[str_377443], **kwargs_377444)
    
    
    # ################# End of 'write_autogen_blurb(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'write_autogen_blurb' in the type store
    # Getting the type of 'stypy_return_type' (line 416)
    stypy_return_type_377446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_377446)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'write_autogen_blurb'
    return stypy_return_type_377446

# Assigning a type to the variable 'write_autogen_blurb' (line 416)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 0), 'write_autogen_blurb', write_autogen_blurb)

if (__name__ == '__main__'):
    
    # Call to main(...): (line 425)
    # Processing the call keyword arguments (line 425)
    kwargs_377448 = {}
    # Getting the type of 'main' (line 425)
    main_377447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'main', False)
    # Calling main(args, kwargs) (line 425)
    main_call_result_377449 = invoke(stypy.reporting.localization.Localization(__file__, 425, 4), main_377447, *[], **kwargs_377448)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
