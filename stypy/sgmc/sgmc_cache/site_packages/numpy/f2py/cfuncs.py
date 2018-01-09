
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: '''
3: 
4: C declarations, CPP macros, and C functions for f2py2e.
5: Only required declarations/macros/functions will be used.
6: 
7: Copyright 1999,2000 Pearu Peterson all rights reserved,
8: Pearu Peterson <pearu@ioc.ee>
9: Permission to use, modify, and distribute this software is given under the
10: terms of the NumPy License.
11: 
12: NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
13: $Date: 2005/05/06 11:42:34 $
14: Pearu Peterson
15: 
16: '''
17: from __future__ import division, absolute_import, print_function
18: 
19: import sys
20: import copy
21: 
22: from . import __version__
23: 
24: f2py_version = __version__.version
25: errmess = sys.stderr.write
26: 
27: ##################### Definitions ##################
28: 
29: outneeds = {'includes0': [], 'includes': [], 'typedefs': [], 'typedefs_generated': [],
30:             'userincludes': [],
31:             'cppmacros': [], 'cfuncs': [], 'callbacks': [], 'f90modhooks': [],
32:             'commonhooks': []}
33: needs = {}
34: includes0 = {'includes0': '/*need_includes0*/'}
35: includes = {'includes': '/*need_includes*/'}
36: userincludes = {'userincludes': '/*need_userincludes*/'}
37: typedefs = {'typedefs': '/*need_typedefs*/'}
38: typedefs_generated = {'typedefs_generated': '/*need_typedefs_generated*/'}
39: cppmacros = {'cppmacros': '/*need_cppmacros*/'}
40: cfuncs = {'cfuncs': '/*need_cfuncs*/'}
41: callbacks = {'callbacks': '/*need_callbacks*/'}
42: f90modhooks = {'f90modhooks': '/*need_f90modhooks*/',
43:                'initf90modhooksstatic': '/*initf90modhooksstatic*/',
44:                'initf90modhooksdynamic': '/*initf90modhooksdynamic*/',
45:                }
46: commonhooks = {'commonhooks': '/*need_commonhooks*/',
47:                'initcommonhooks': '/*need_initcommonhooks*/',
48:                }
49: 
50: ############ Includes ###################
51: 
52: includes0['math.h'] = '#include <math.h>'
53: includes0['string.h'] = '#include <string.h>'
54: includes0['setjmp.h'] = '#include <setjmp.h>'
55: 
56: includes['Python.h'] = '#include "Python.h"'
57: needs['arrayobject.h'] = ['Python.h']
58: includes['arrayobject.h'] = '''#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API
59: #include "arrayobject.h"'''
60: 
61: includes['arrayobject.h'] = '#include "fortranobject.h"'
62: includes['stdarg.h'] = '#include <stdarg.h>'
63: 
64: ############# Type definitions ###############
65: 
66: typedefs['unsigned_char'] = 'typedef unsigned char unsigned_char;'
67: typedefs['unsigned_short'] = 'typedef unsigned short unsigned_short;'
68: typedefs['unsigned_long'] = 'typedef unsigned long unsigned_long;'
69: typedefs['signed_char'] = 'typedef signed char signed_char;'
70: typedefs['long_long'] = '''\
71: #ifdef _WIN32
72: typedef __int64 long_long;
73: #else
74: typedef long long long_long;
75: typedef unsigned long long unsigned_long_long;
76: #endif
77: '''
78: typedefs['unsigned_long_long'] = '''\
79: #ifdef _WIN32
80: typedef __uint64 long_long;
81: #else
82: typedef unsigned long long unsigned_long_long;
83: #endif
84: '''
85: typedefs['long_double'] = '''\
86: #ifndef _LONG_DOUBLE
87: typedef long double long_double;
88: #endif
89: '''
90: typedefs[
91:     'complex_long_double'] = 'typedef struct {long double r,i;} complex_long_double;'
92: typedefs['complex_float'] = 'typedef struct {float r,i;} complex_float;'
93: typedefs['complex_double'] = 'typedef struct {double r,i;} complex_double;'
94: typedefs['string'] = '''typedef char * string;'''
95: 
96: 
97: ############### CPP macros ####################
98: cppmacros['CFUNCSMESS'] = '''\
99: #ifdef DEBUGCFUNCS
100: #define CFUNCSMESS(mess) fprintf(stderr,\"debug-capi:\"mess);
101: #define CFUNCSMESSPY(mess,obj) CFUNCSMESS(mess) \\
102: \tPyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\\
103: \tfprintf(stderr,\"\\n\");
104: #else
105: #define CFUNCSMESS(mess)
106: #define CFUNCSMESSPY(mess,obj)
107: #endif
108: '''
109: cppmacros['F_FUNC'] = '''\
110: #if defined(PREPEND_FORTRAN)
111: #if defined(NO_APPEND_FORTRAN)
112: #if defined(UPPERCASE_FORTRAN)
113: #define F_FUNC(f,F) _##F
114: #else
115: #define F_FUNC(f,F) _##f
116: #endif
117: #else
118: #if defined(UPPERCASE_FORTRAN)
119: #define F_FUNC(f,F) _##F##_
120: #else
121: #define F_FUNC(f,F) _##f##_
122: #endif
123: #endif
124: #else
125: #if defined(NO_APPEND_FORTRAN)
126: #if defined(UPPERCASE_FORTRAN)
127: #define F_FUNC(f,F) F
128: #else
129: #define F_FUNC(f,F) f
130: #endif
131: #else
132: #if defined(UPPERCASE_FORTRAN)
133: #define F_FUNC(f,F) F##_
134: #else
135: #define F_FUNC(f,F) f##_
136: #endif
137: #endif
138: #endif
139: #if defined(UNDERSCORE_G77)
140: #define F_FUNC_US(f,F) F_FUNC(f##_,F##_)
141: #else
142: #define F_FUNC_US(f,F) F_FUNC(f,F)
143: #endif
144: '''
145: cppmacros['F_WRAPPEDFUNC'] = '''\
146: #if defined(PREPEND_FORTRAN)
147: #if defined(NO_APPEND_FORTRAN)
148: #if defined(UPPERCASE_FORTRAN)
149: #define F_WRAPPEDFUNC(f,F) _F2PYWRAP##F
150: #else
151: #define F_WRAPPEDFUNC(f,F) _f2pywrap##f
152: #endif
153: #else
154: #if defined(UPPERCASE_FORTRAN)
155: #define F_WRAPPEDFUNC(f,F) _F2PYWRAP##F##_
156: #else
157: #define F_WRAPPEDFUNC(f,F) _f2pywrap##f##_
158: #endif
159: #endif
160: #else
161: #if defined(NO_APPEND_FORTRAN)
162: #if defined(UPPERCASE_FORTRAN)
163: #define F_WRAPPEDFUNC(f,F) F2PYWRAP##F
164: #else
165: #define F_WRAPPEDFUNC(f,F) f2pywrap##f
166: #endif
167: #else
168: #if defined(UPPERCASE_FORTRAN)
169: #define F_WRAPPEDFUNC(f,F) F2PYWRAP##F##_
170: #else
171: #define F_WRAPPEDFUNC(f,F) f2pywrap##f##_
172: #endif
173: #endif
174: #endif
175: #if defined(UNDERSCORE_G77)
176: #define F_WRAPPEDFUNC_US(f,F) F_WRAPPEDFUNC(f##_,F##_)
177: #else
178: #define F_WRAPPEDFUNC_US(f,F) F_WRAPPEDFUNC(f,F)
179: #endif
180: '''
181: cppmacros['F_MODFUNC'] = '''\
182: #if defined(F90MOD2CCONV1) /*E.g. Compaq Fortran */
183: #if defined(NO_APPEND_FORTRAN)
184: #define F_MODFUNCNAME(m,f) $ ## m ## $ ## f
185: #else
186: #define F_MODFUNCNAME(m,f) $ ## m ## $ ## f ## _
187: #endif
188: #endif
189: 
190: #if defined(F90MOD2CCONV2) /*E.g. IBM XL Fortran, not tested though */
191: #if defined(NO_APPEND_FORTRAN)
192: #define F_MODFUNCNAME(m,f)  __ ## m ## _MOD_ ## f
193: #else
194: #define F_MODFUNCNAME(m,f)  __ ## m ## _MOD_ ## f ## _
195: #endif
196: #endif
197: 
198: #if defined(F90MOD2CCONV3) /*E.g. MIPSPro Compilers */
199: #if defined(NO_APPEND_FORTRAN)
200: #define F_MODFUNCNAME(m,f)  f ## .in. ## m
201: #else
202: #define F_MODFUNCNAME(m,f)  f ## .in. ## m ## _
203: #endif
204: #endif
205: /*
206: #if defined(UPPERCASE_FORTRAN)
207: #define F_MODFUNC(m,M,f,F) F_MODFUNCNAME(M,F)
208: #else
209: #define F_MODFUNC(m,M,f,F) F_MODFUNCNAME(m,f)
210: #endif
211: */
212: 
213: #define F_MODFUNC(m,f) (*(f2pymodstruct##m##.##f))
214: '''
215: cppmacros['SWAPUNSAFE'] = '''\
216: #define SWAP(a,b) (size_t)(a) = ((size_t)(a) ^ (size_t)(b));\\
217:  (size_t)(b) = ((size_t)(a) ^ (size_t)(b));\\
218:  (size_t)(a) = ((size_t)(a) ^ (size_t)(b))
219: '''
220: cppmacros['SWAP'] = '''\
221: #define SWAP(a,b,t) {\\
222: \tt *c;\\
223: \tc = a;\\
224: \ta = b;\\
225: \tb = c;}
226: '''
227: # cppmacros['ISCONTIGUOUS']='#define ISCONTIGUOUS(m) (PyArray_FLAGS(m) &
228: # NPY_ARRAY_C_CONTIGUOUS)'
229: cppmacros['PRINTPYOBJERR'] = '''\
230: #define PRINTPYOBJERR(obj)\\
231: \tfprintf(stderr,\"#modulename#.error is related to \");\\
232: \tPyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\\
233: \tfprintf(stderr,\"\\n\");
234: '''
235: cppmacros['MINMAX'] = '''\
236: #ifndef max
237: #define max(a,b) ((a > b) ? (a) : (b))
238: #endif
239: #ifndef min
240: #define min(a,b) ((a < b) ? (a) : (b))
241: #endif
242: #ifndef MAX
243: #define MAX(a,b) ((a > b) ? (a) : (b))
244: #endif
245: #ifndef MIN
246: #define MIN(a,b) ((a < b) ? (a) : (b))
247: #endif
248: '''
249: needs['len..'] = ['f2py_size']
250: cppmacros['len..'] = '''\
251: #define rank(var) var ## _Rank
252: #define shape(var,dim) var ## _Dims[dim]
253: #define old_rank(var) (PyArray_NDIM((PyArrayObject *)(capi_ ## var ## _tmp)))
254: #define old_shape(var,dim) PyArray_DIM(((PyArrayObject *)(capi_ ## var ## _tmp)),dim)
255: #define fshape(var,dim) shape(var,rank(var)-dim-1)
256: #define len(var) shape(var,0)
257: #define flen(var) fshape(var,0)
258: #define old_size(var) PyArray_SIZE((PyArrayObject *)(capi_ ## var ## _tmp))
259: /* #define index(i) capi_i ## i */
260: #define slen(var) capi_ ## var ## _len
261: #define size(var, ...) f2py_size((PyArrayObject *)(capi_ ## var ## _tmp), ## __VA_ARGS__, -1)
262: '''
263: needs['f2py_size'] = ['stdarg.h']
264: cfuncs['f2py_size'] = '''\
265: static int f2py_size(PyArrayObject* var, ...)
266: {
267:   npy_int sz = 0;
268:   npy_int dim;
269:   npy_int rank;
270:   va_list argp;
271:   va_start(argp, var);
272:   dim = va_arg(argp, npy_int);
273:   if (dim==-1)
274:     {
275:       sz = PyArray_SIZE(var);
276:     }
277:   else
278:     {
279:       rank = PyArray_NDIM(var);
280:       if (dim>=1 && dim<=rank)
281:         sz = PyArray_DIM(var, dim-1);
282:       else
283:         fprintf(stderr, \"f2py_size: 2nd argument value=%d fails to satisfy 1<=value<=%d. Result will be 0.\\n\", dim, rank);
284:     }
285:   va_end(argp);
286:   return sz;
287: }
288: '''
289: 
290: cppmacros[
291:     'pyobj_from_char1'] = '#define pyobj_from_char1(v) (PyInt_FromLong(v))'
292: cppmacros[
293:     'pyobj_from_short1'] = '#define pyobj_from_short1(v) (PyInt_FromLong(v))'
294: needs['pyobj_from_int1'] = ['signed_char']
295: cppmacros['pyobj_from_int1'] = '#define pyobj_from_int1(v) (PyInt_FromLong(v))'
296: cppmacros[
297:     'pyobj_from_long1'] = '#define pyobj_from_long1(v) (PyLong_FromLong(v))'
298: needs['pyobj_from_long_long1'] = ['long_long']
299: cppmacros['pyobj_from_long_long1'] = '''\
300: #ifdef HAVE_LONG_LONG
301: #define pyobj_from_long_long1(v) (PyLong_FromLongLong(v))
302: #else
303: #warning HAVE_LONG_LONG is not available. Redefining pyobj_from_long_long.
304: #define pyobj_from_long_long1(v) (PyLong_FromLong(v))
305: #endif
306: '''
307: needs['pyobj_from_long_double1'] = ['long_double']
308: cppmacros[
309:     'pyobj_from_long_double1'] = '#define pyobj_from_long_double1(v) (PyFloat_FromDouble(v))'
310: cppmacros[
311:     'pyobj_from_double1'] = '#define pyobj_from_double1(v) (PyFloat_FromDouble(v))'
312: cppmacros[
313:     'pyobj_from_float1'] = '#define pyobj_from_float1(v) (PyFloat_FromDouble(v))'
314: needs['pyobj_from_complex_long_double1'] = ['complex_long_double']
315: cppmacros[
316:     'pyobj_from_complex_long_double1'] = '#define pyobj_from_complex_long_double1(v) (PyComplex_FromDoubles(v.r,v.i))'
317: needs['pyobj_from_complex_double1'] = ['complex_double']
318: cppmacros[
319:     'pyobj_from_complex_double1'] = '#define pyobj_from_complex_double1(v) (PyComplex_FromDoubles(v.r,v.i))'
320: needs['pyobj_from_complex_float1'] = ['complex_float']
321: cppmacros[
322:     'pyobj_from_complex_float1'] = '#define pyobj_from_complex_float1(v) (PyComplex_FromDoubles(v.r,v.i))'
323: needs['pyobj_from_string1'] = ['string']
324: cppmacros[
325:     'pyobj_from_string1'] = '#define pyobj_from_string1(v) (PyString_FromString((char *)v))'
326: needs['pyobj_from_string1size'] = ['string']
327: cppmacros[
328:     'pyobj_from_string1size'] = '#define pyobj_from_string1size(v,len) (PyUString_FromStringAndSize((char *)v, len))'
329: needs['TRYPYARRAYTEMPLATE'] = ['PRINTPYOBJERR']
330: cppmacros['TRYPYARRAYTEMPLATE'] = '''\
331: /* New SciPy */
332: #define TRYPYARRAYTEMPLATECHAR case NPY_STRING: *(char *)(PyArray_DATA(arr))=*v; break;
333: #define TRYPYARRAYTEMPLATELONG case NPY_LONG: *(long *)(PyArray_DATA(arr))=*v; break;
334: #define TRYPYARRAYTEMPLATEOBJECT case NPY_OBJECT: (PyArray_DESCR(arr)->f->setitem)(pyobj_from_ ## ctype ## 1(*v),PyArray_DATA(arr)); break;
335: 
336: #define TRYPYARRAYTEMPLATE(ctype,typecode) \\
337:         PyArrayObject *arr = NULL;\\
338:         if (!obj) return -2;\\
339:         if (!PyArray_Check(obj)) return -1;\\
340:         if (!(arr=(PyArrayObject *)obj)) {fprintf(stderr,\"TRYPYARRAYTEMPLATE:\");PRINTPYOBJERR(obj);return 0;}\\
341:         if (PyArray_DESCR(arr)->type==typecode)  {*(ctype *)(PyArray_DATA(arr))=*v; return 1;}\\
342:         switch (PyArray_TYPE(arr)) {\\
343:                 case NPY_DOUBLE: *(double *)(PyArray_DATA(arr))=*v; break;\\
344:                 case NPY_INT: *(int *)(PyArray_DATA(arr))=*v; break;\\
345:                 case NPY_LONG: *(long *)(PyArray_DATA(arr))=*v; break;\\
346:                 case NPY_FLOAT: *(float *)(PyArray_DATA(arr))=*v; break;\\
347:                 case NPY_CDOUBLE: *(double *)(PyArray_DATA(arr))=*v; break;\\
348:                 case NPY_CFLOAT: *(float *)(PyArray_DATA(arr))=*v; break;\\
349:                 case NPY_BOOL: *(npy_bool *)(PyArray_DATA(arr))=(*v!=0); break;\\
350:                 case NPY_UBYTE: *(unsigned char *)(PyArray_DATA(arr))=*v; break;\\
351:                 case NPY_BYTE: *(signed char *)(PyArray_DATA(arr))=*v; break;\\
352:                 case NPY_SHORT: *(short *)(PyArray_DATA(arr))=*v; break;\\
353:                 case NPY_USHORT: *(npy_ushort *)(PyArray_DATA(arr))=*v; break;\\
354:                 case NPY_UINT: *(npy_uint *)(PyArray_DATA(arr))=*v; break;\\
355:                 case NPY_ULONG: *(npy_ulong *)(PyArray_DATA(arr))=*v; break;\\
356:                 case NPY_LONGLONG: *(npy_longlong *)(PyArray_DATA(arr))=*v; break;\\
357:                 case NPY_ULONGLONG: *(npy_ulonglong *)(PyArray_DATA(arr))=*v; break;\\
358:                 case NPY_LONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=*v; break;\\
359:                 case NPY_CLONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=*v; break;\\
360:                 case NPY_OBJECT: (PyArray_DESCR(arr)->f->setitem)(pyobj_from_ ## ctype ## 1(*v),PyArray_DATA(arr), arr); break;\\
361:         default: return -2;\\
362:         };\\
363:         return 1
364: '''
365: 
366: needs['TRYCOMPLEXPYARRAYTEMPLATE'] = ['PRINTPYOBJERR']
367: cppmacros['TRYCOMPLEXPYARRAYTEMPLATE'] = '''\
368: #define TRYCOMPLEXPYARRAYTEMPLATEOBJECT case NPY_OBJECT: (PyArray_DESCR(arr)->f->setitem)(pyobj_from_complex_ ## ctype ## 1((*v)),PyArray_DATA(arr), arr); break;
369: #define TRYCOMPLEXPYARRAYTEMPLATE(ctype,typecode)\\
370:         PyArrayObject *arr = NULL;\\
371:         if (!obj) return -2;\\
372:         if (!PyArray_Check(obj)) return -1;\\
373:         if (!(arr=(PyArrayObject *)obj)) {fprintf(stderr,\"TRYCOMPLEXPYARRAYTEMPLATE:\");PRINTPYOBJERR(obj);return 0;}\\
374:         if (PyArray_DESCR(arr)->type==typecode) {\\
375:             *(ctype *)(PyArray_DATA(arr))=(*v).r;\\
376:             *(ctype *)(PyArray_DATA(arr)+sizeof(ctype))=(*v).i;\\
377:             return 1;\\
378:         }\\
379:         switch (PyArray_TYPE(arr)) {\\
380:                 case NPY_CDOUBLE: *(double *)(PyArray_DATA(arr))=(*v).r;*(double *)(PyArray_DATA(arr)+sizeof(double))=(*v).i;break;\\
381:                 case NPY_CFLOAT: *(float *)(PyArray_DATA(arr))=(*v).r;*(float *)(PyArray_DATA(arr)+sizeof(float))=(*v).i;break;\\
382:                 case NPY_DOUBLE: *(double *)(PyArray_DATA(arr))=(*v).r; break;\\
383:                 case NPY_LONG: *(long *)(PyArray_DATA(arr))=(*v).r; break;\\
384:                 case NPY_FLOAT: *(float *)(PyArray_DATA(arr))=(*v).r; break;\\
385:                 case NPY_INT: *(int *)(PyArray_DATA(arr))=(*v).r; break;\\
386:                 case NPY_SHORT: *(short *)(PyArray_DATA(arr))=(*v).r; break;\\
387:                 case NPY_UBYTE: *(unsigned char *)(PyArray_DATA(arr))=(*v).r; break;\\
388:                 case NPY_BYTE: *(signed char *)(PyArray_DATA(arr))=(*v).r; break;\\
389:                 case NPY_BOOL: *(npy_bool *)(PyArray_DATA(arr))=((*v).r!=0 && (*v).i!=0); break;\\
390:                 case NPY_USHORT: *(npy_ushort *)(PyArray_DATA(arr))=(*v).r; break;\\
391:                 case NPY_UINT: *(npy_uint *)(PyArray_DATA(arr))=(*v).r; break;\\
392:                 case NPY_ULONG: *(npy_ulong *)(PyArray_DATA(arr))=(*v).r; break;\\
393:                 case NPY_LONGLONG: *(npy_longlong *)(PyArray_DATA(arr))=(*v).r; break;\\
394:                 case NPY_ULONGLONG: *(npy_ulonglong *)(PyArray_DATA(arr))=(*v).r; break;\\
395:                 case NPY_LONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=(*v).r; break;\\
396:                 case NPY_CLONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=(*v).r;*(npy_longdouble *)(PyArray_DATA(arr)+sizeof(npy_longdouble))=(*v).i;break;\\
397:                 case NPY_OBJECT: (PyArray_DESCR(arr)->f->setitem)(pyobj_from_complex_ ## ctype ## 1((*v)),PyArray_DATA(arr), arr); break;\\
398:                 default: return -2;\\
399:         };\\
400:         return -1;
401: '''
402: # cppmacros['NUMFROMARROBJ']='''\
403: # define NUMFROMARROBJ(typenum,ctype) \\
404: # \tif (PyArray_Check(obj)) arr = (PyArrayObject *)obj;\\
405: # \telse arr = (PyArrayObject *)PyArray_ContiguousFromObject(obj,typenum,0,0);\\
406: # \tif (arr) {\\
407: # \t\tif (PyArray_TYPE(arr)==NPY_OBJECT) {\\
408: # \t\t\tif (!ctype ## _from_pyobj(v,(PyArray_DESCR(arr)->getitem)(PyArray_DATA(arr)),\"\"))\\
409: # \t\t\tgoto capi_fail;\\
410: # \t\t} else {\\
411: # \t\t\t(PyArray_DESCR(arr)->cast[typenum])(PyArray_DATA(arr),1,(char*)v,1,1);\\
412: # \t\t}\\
413: # \t\tif ((PyObject *)arr != obj) { Py_DECREF(arr); }\\
414: # \t\treturn 1;\\
415: # \t}
416: # '''
417: # XXX: Note that CNUMFROMARROBJ is identical with NUMFROMARROBJ
418: # cppmacros['CNUMFROMARROBJ']='''\
419: # define CNUMFROMARROBJ(typenum,ctype) \\
420: # \tif (PyArray_Check(obj)) arr = (PyArrayObject *)obj;\\
421: # \telse arr = (PyArrayObject *)PyArray_ContiguousFromObject(obj,typenum,0,0);\\
422: # \tif (arr) {\\
423: # \t\tif (PyArray_TYPE(arr)==NPY_OBJECT) {\\
424: # \t\t\tif (!ctype ## _from_pyobj(v,(PyArray_DESCR(arr)->getitem)(PyArray_DATA(arr)),\"\"))\\
425: # \t\t\tgoto capi_fail;\\
426: # \t\t} else {\\
427: # \t\t\t(PyArray_DESCR(arr)->cast[typenum])((void *)(PyArray_DATA(arr)),1,(void *)(v),1,1);\\
428: # \t\t}\\
429: # \t\tif ((PyObject *)arr != obj) { Py_DECREF(arr); }\\
430: # \t\treturn 1;\\
431: # \t}
432: # '''
433: 
434: 
435: needs['GETSTRFROMPYTUPLE'] = ['STRINGCOPYN', 'PRINTPYOBJERR']
436: cppmacros['GETSTRFROMPYTUPLE'] = '''\
437: #define GETSTRFROMPYTUPLE(tuple,index,str,len) {\\
438: \t\tPyObject *rv_cb_str = PyTuple_GetItem((tuple),(index));\\
439: \t\tif (rv_cb_str == NULL)\\
440: \t\t\tgoto capi_fail;\\
441: \t\tif (PyString_Check(rv_cb_str)) {\\
442: \t\t\tstr[len-1]='\\0';\\
443: \t\t\tSTRINGCOPYN((str),PyString_AS_STRING((PyStringObject*)rv_cb_str),(len));\\
444: \t\t} else {\\
445: \t\t\tPRINTPYOBJERR(rv_cb_str);\\
446: \t\t\tPyErr_SetString(#modulename#_error,\"string object expected\");\\
447: \t\t\tgoto capi_fail;\\
448: \t\t}\\
449: \t}
450: '''
451: cppmacros['GETSCALARFROMPYTUPLE'] = '''\
452: #define GETSCALARFROMPYTUPLE(tuple,index,var,ctype,mess) {\\
453: \t\tif ((capi_tmp = PyTuple_GetItem((tuple),(index)))==NULL) goto capi_fail;\\
454: \t\tif (!(ctype ## _from_pyobj((var),capi_tmp,mess)))\\
455: \t\t\tgoto capi_fail;\\
456: \t}
457: '''
458: 
459: cppmacros['FAILNULL'] = '''\\
460: #define FAILNULL(p) do {                                            \\
461:     if ((p) == NULL) {                                              \\
462:         PyErr_SetString(PyExc_MemoryError, "NULL pointer found");   \\
463:         goto capi_fail;                                             \\
464:     }                                                               \\
465: } while (0)
466: '''
467: needs['MEMCOPY'] = ['string.h', 'FAILNULL']
468: cppmacros['MEMCOPY'] = '''\
469: #define MEMCOPY(to,from,n)\\
470:     do { FAILNULL(to); FAILNULL(from); (void)memcpy(to,from,n); } while (0)
471: '''
472: cppmacros['STRINGMALLOC'] = '''\
473: #define STRINGMALLOC(str,len)\\
474: \tif ((str = (string)malloc(sizeof(char)*(len+1))) == NULL) {\\
475: \t\tPyErr_SetString(PyExc_MemoryError, \"out of memory\");\\
476: \t\tgoto capi_fail;\\
477: \t} else {\\
478: \t\t(str)[len] = '\\0';\\
479: \t}
480: '''
481: cppmacros['STRINGFREE'] = '''\
482: #define STRINGFREE(str) do {if (!(str == NULL)) free(str);} while (0)
483: '''
484: needs['STRINGCOPYN'] = ['string.h', 'FAILNULL']
485: cppmacros['STRINGCOPYN'] = '''\
486: #define STRINGCOPYN(to,from,buf_size)                           \\
487:     do {                                                        \\
488:         int _m = (buf_size);                                    \\
489:         char *_to = (to);                                       \\
490:         char *_from = (from);                                   \\
491:         FAILNULL(_to); FAILNULL(_from);                         \\
492:         (void)strncpy(_to, _from, sizeof(char)*_m);             \\
493:         _to[_m-1] = '\\0';                                      \\
494:         /* Padding with spaces instead of nulls */              \\
495:         for (_m -= 2; _m >= 0 && _to[_m] == '\\0'; _m--) {      \\
496:             _to[_m] = ' ';                                      \\
497:         }                                                       \\
498:     } while (0)
499: '''
500: needs['STRINGCOPY'] = ['string.h', 'FAILNULL']
501: cppmacros['STRINGCOPY'] = '''\
502: #define STRINGCOPY(to,from)\\
503:     do { FAILNULL(to); FAILNULL(from); (void)strcpy(to,from); } while (0)
504: '''
505: cppmacros['CHECKGENERIC'] = '''\
506: #define CHECKGENERIC(check,tcheck,name) \\
507: \tif (!(check)) {\\
508: \t\tPyErr_SetString(#modulename#_error,\"(\"tcheck\") failed for \"name);\\
509: \t\t/*goto capi_fail;*/\\
510: \t} else '''
511: cppmacros['CHECKARRAY'] = '''\
512: #define CHECKARRAY(check,tcheck,name) \\
513: \tif (!(check)) {\\
514: \t\tPyErr_SetString(#modulename#_error,\"(\"tcheck\") failed for \"name);\\
515: \t\t/*goto capi_fail;*/\\
516: \t} else '''
517: cppmacros['CHECKSTRING'] = '''\
518: #define CHECKSTRING(check,tcheck,name,show,var)\\
519: \tif (!(check)) {\\
520: \t\tchar errstring[256];\\
521: \t\tsprintf(errstring, \"%s: \"show, \"(\"tcheck\") failed for \"name, slen(var), var);\\
522: \t\tPyErr_SetString(#modulename#_error, errstring);\\
523: \t\t/*goto capi_fail;*/\\
524: \t} else '''
525: cppmacros['CHECKSCALAR'] = '''\
526: #define CHECKSCALAR(check,tcheck,name,show,var)\\
527: \tif (!(check)) {\\
528: \t\tchar errstring[256];\\
529: \t\tsprintf(errstring, \"%s: \"show, \"(\"tcheck\") failed for \"name, var);\\
530: \t\tPyErr_SetString(#modulename#_error,errstring);\\
531: \t\t/*goto capi_fail;*/\\
532: \t} else '''
533: # cppmacros['CHECKDIMS']='''\
534: # define CHECKDIMS(dims,rank) \\
535: # \tfor (int i=0;i<(rank);i++)\\
536: # \t\tif (dims[i]<0) {\\
537: # \t\t\tfprintf(stderr,\"Unspecified array argument requires a complete dimension specification.\\n\");\\
538: # \t\t\tgoto capi_fail;\\
539: # \t\t}
540: # '''
541: cppmacros[
542:     'ARRSIZE'] = '#define ARRSIZE(dims,rank) (_PyArray_multiply_list(dims,rank))'
543: cppmacros['OLDPYNUM'] = '''\
544: #ifdef OLDPYNUM
545: #error You need to intall Numeric Python version 13 or higher. Get it from http:/sourceforge.net/project/?group_id=1369
546: #endif
547: '''
548: ################# C functions ###############
549: 
550: cfuncs['calcarrindex'] = '''\
551: static int calcarrindex(int *i,PyArrayObject *arr) {
552: \tint k,ii = i[0];
553: \tfor (k=1; k < PyArray_NDIM(arr); k++)
554: \t\tii += (ii*(PyArray_DIM(arr,k) - 1)+i[k]); /* assuming contiguous arr */
555: \treturn ii;
556: }'''
557: cfuncs['calcarrindextr'] = '''\
558: static int calcarrindextr(int *i,PyArrayObject *arr) {
559: \tint k,ii = i[PyArray_NDIM(arr)-1];
560: \tfor (k=1; k < PyArray_NDIM(arr); k++)
561: \t\tii += (ii*(PyArray_DIM(arr,PyArray_NDIM(arr)-k-1) - 1)+i[PyArray_NDIM(arr)-k-1]); /* assuming contiguous arr */
562: \treturn ii;
563: }'''
564: cfuncs['forcomb'] = '''\
565: static struct { int nd;npy_intp *d;int *i,*i_tr,tr; } forcombcache;
566: static int initforcomb(npy_intp *dims,int nd,int tr) {
567:   int k;
568:   if (dims==NULL) return 0;
569:   if (nd<0) return 0;
570:   forcombcache.nd = nd;
571:   forcombcache.d = dims;
572:   forcombcache.tr = tr;
573:   if ((forcombcache.i = (int *)malloc(sizeof(int)*nd))==NULL) return 0;
574:   if ((forcombcache.i_tr = (int *)malloc(sizeof(int)*nd))==NULL) return 0;
575:   for (k=1;k<nd;k++) {
576:     forcombcache.i[k] = forcombcache.i_tr[nd-k-1] = 0;
577:   }
578:   forcombcache.i[0] = forcombcache.i_tr[nd-1] = -1;
579:   return 1;
580: }
581: static int *nextforcomb(void) {
582:   int j,*i,*i_tr,k;
583:   int nd=forcombcache.nd;
584:   if ((i=forcombcache.i) == NULL) return NULL;
585:   if ((i_tr=forcombcache.i_tr) == NULL) return NULL;
586:   if (forcombcache.d == NULL) return NULL;
587:   i[0]++;
588:   if (i[0]==forcombcache.d[0]) {
589:     j=1;
590:     while ((j<nd) && (i[j]==forcombcache.d[j]-1)) j++;
591:     if (j==nd) {
592:       free(i);
593:       free(i_tr);
594:       return NULL;
595:     }
596:     for (k=0;k<j;k++) i[k] = i_tr[nd-k-1] = 0;
597:     i[j]++;
598:     i_tr[nd-j-1]++;
599:   } else
600:     i_tr[nd-1]++;
601:   if (forcombcache.tr) return i_tr;
602:   return i;
603: }'''
604: needs['try_pyarr_from_string'] = ['STRINGCOPYN', 'PRINTPYOBJERR', 'string']
605: cfuncs['try_pyarr_from_string'] = '''\
606: static int try_pyarr_from_string(PyObject *obj,const string str) {
607: \tPyArrayObject *arr = NULL;
608: \tif (PyArray_Check(obj) && (!((arr = (PyArrayObject *)obj) == NULL)))
609: \t\t{ STRINGCOPYN(PyArray_DATA(arr),str,PyArray_NBYTES(arr)); }
610: \treturn 1;
611: capi_fail:
612: \tPRINTPYOBJERR(obj);
613: \tPyErr_SetString(#modulename#_error,\"try_pyarr_from_string failed\");
614: \treturn 0;
615: }
616: '''
617: needs['string_from_pyobj'] = ['string', 'STRINGMALLOC', 'STRINGCOPYN']
618: cfuncs['string_from_pyobj'] = '''\
619: static int string_from_pyobj(string *str,int *len,const string inistr,PyObject *obj,const char *errmess) {
620: \tPyArrayObject *arr = NULL;
621: \tPyObject *tmp = NULL;
622: #ifdef DEBUGCFUNCS
623: fprintf(stderr,\"string_from_pyobj(str='%s',len=%d,inistr='%s',obj=%p)\\n\",(char*)str,*len,(char *)inistr,obj);
624: #endif
625: \tif (obj == Py_None) {
626: \t\tif (*len == -1)
627: \t\t\t*len = strlen(inistr); /* Will this cause problems? */
628: \t\tSTRINGMALLOC(*str,*len);
629: \t\tSTRINGCOPYN(*str,inistr,*len+1);
630: \t\treturn 1;
631: \t}
632: \tif (PyArray_Check(obj)) {
633: \t\tif ((arr = (PyArrayObject *)obj) == NULL)
634: \t\t\tgoto capi_fail;
635: \t\tif (!ISCONTIGUOUS(arr)) {
636: \t\t\tPyErr_SetString(PyExc_ValueError,\"array object is non-contiguous.\");
637: \t\t\tgoto capi_fail;
638: \t\t}
639: \t\tif (*len == -1)
640: \t\t\t*len = (PyArray_ITEMSIZE(arr))*PyArray_SIZE(arr);
641: \t\tSTRINGMALLOC(*str,*len);
642: \t\tSTRINGCOPYN(*str,PyArray_DATA(arr),*len+1);
643: \t\treturn 1;
644: \t}
645: \tif (PyString_Check(obj)) {
646: \t\ttmp = obj;
647: \t\tPy_INCREF(tmp);
648: \t}
649: #if PY_VERSION_HEX >= 0x03000000
650: \telse if (PyUnicode_Check(obj)) {
651: \t\ttmp = PyUnicode_AsASCIIString(obj);
652: \t}
653: \telse {
654: \t\tPyObject *tmp2;
655: \t\ttmp2 = PyObject_Str(obj);
656: \t\tif (tmp2) {
657: \t\t\ttmp = PyUnicode_AsASCIIString(tmp2);
658: \t\t\tPy_DECREF(tmp2);
659: \t\t}
660: \t\telse {
661: \t\t\ttmp = NULL;
662: \t\t}
663: \t}
664: #else
665: \telse {
666: \t\ttmp = PyObject_Str(obj);
667: \t}
668: #endif
669: \tif (tmp == NULL) goto capi_fail;
670: \tif (*len == -1)
671: \t\t*len = PyString_GET_SIZE(tmp);
672: \tSTRINGMALLOC(*str,*len);
673: \tSTRINGCOPYN(*str,PyString_AS_STRING(tmp),*len+1);
674: \tPy_DECREF(tmp);
675: \treturn 1;
676: capi_fail:
677: \tPy_XDECREF(tmp);
678: \t{
679: \t\tPyObject* err = PyErr_Occurred();
680: \t\tif (err==NULL) err = #modulename#_error;
681: \t\tPyErr_SetString(err,errmess);
682: \t}
683: \treturn 0;
684: }
685: '''
686: needs['char_from_pyobj'] = ['int_from_pyobj']
687: cfuncs['char_from_pyobj'] = '''\
688: static int char_from_pyobj(char* v,PyObject *obj,const char *errmess) {
689: \tint i=0;
690: \tif (int_from_pyobj(&i,obj,errmess)) {
691: \t\t*v = (char)i;
692: \t\treturn 1;
693: \t}
694: \treturn 0;
695: }
696: '''
697: needs['signed_char_from_pyobj'] = ['int_from_pyobj', 'signed_char']
698: cfuncs['signed_char_from_pyobj'] = '''\
699: static int signed_char_from_pyobj(signed_char* v,PyObject *obj,const char *errmess) {
700: \tint i=0;
701: \tif (int_from_pyobj(&i,obj,errmess)) {
702: \t\t*v = (signed_char)i;
703: \t\treturn 1;
704: \t}
705: \treturn 0;
706: }
707: '''
708: needs['short_from_pyobj'] = ['int_from_pyobj']
709: cfuncs['short_from_pyobj'] = '''\
710: static int short_from_pyobj(short* v,PyObject *obj,const char *errmess) {
711: \tint i=0;
712: \tif (int_from_pyobj(&i,obj,errmess)) {
713: \t\t*v = (short)i;
714: \t\treturn 1;
715: \t}
716: \treturn 0;
717: }
718: '''
719: cfuncs['int_from_pyobj'] = '''\
720: static int int_from_pyobj(int* v,PyObject *obj,const char *errmess) {
721: \tPyObject* tmp = NULL;
722: \tif (PyInt_Check(obj)) {
723: \t\t*v = (int)PyInt_AS_LONG(obj);
724: \t\treturn 1;
725: \t}
726: \ttmp = PyNumber_Int(obj);
727: \tif (tmp) {
728: \t\t*v = PyInt_AS_LONG(tmp);
729: \t\tPy_DECREF(tmp);
730: \t\treturn 1;
731: \t}
732: \tif (PyComplex_Check(obj))
733: \t\ttmp = PyObject_GetAttrString(obj,\"real\");
734: \telse if (PyString_Check(obj) || PyUnicode_Check(obj))
735: \t\t/*pass*/;
736: \telse if (PySequence_Check(obj))
737: \t\ttmp = PySequence_GetItem(obj,0);
738: \tif (tmp) {
739: \t\tPyErr_Clear();
740: \t\tif (int_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}
741: \t\tPy_DECREF(tmp);
742: \t}
743: \t{
744: \t\tPyObject* err = PyErr_Occurred();
745: \t\tif (err==NULL) err = #modulename#_error;
746: \t\tPyErr_SetString(err,errmess);
747: \t}
748: \treturn 0;
749: }
750: '''
751: cfuncs['long_from_pyobj'] = '''\
752: static int long_from_pyobj(long* v,PyObject *obj,const char *errmess) {
753: \tPyObject* tmp = NULL;
754: \tif (PyInt_Check(obj)) {
755: \t\t*v = PyInt_AS_LONG(obj);
756: \t\treturn 1;
757: \t}
758: \ttmp = PyNumber_Int(obj);
759: \tif (tmp) {
760: \t\t*v = PyInt_AS_LONG(tmp);
761: \t\tPy_DECREF(tmp);
762: \t\treturn 1;
763: \t}
764: \tif (PyComplex_Check(obj))
765: \t\ttmp = PyObject_GetAttrString(obj,\"real\");
766: \telse if (PyString_Check(obj) || PyUnicode_Check(obj))
767: \t\t/*pass*/;
768: \telse if (PySequence_Check(obj))
769: \t\ttmp = PySequence_GetItem(obj,0);
770: \tif (tmp) {
771: \t\tPyErr_Clear();
772: \t\tif (long_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}
773: \t\tPy_DECREF(tmp);
774: \t}
775: \t{
776: \t\tPyObject* err = PyErr_Occurred();
777: \t\tif (err==NULL) err = #modulename#_error;
778: \t\tPyErr_SetString(err,errmess);
779: \t}
780: \treturn 0;
781: }
782: '''
783: needs['long_long_from_pyobj'] = ['long_long']
784: cfuncs['long_long_from_pyobj'] = '''\
785: static int long_long_from_pyobj(long_long* v,PyObject *obj,const char *errmess) {
786: \tPyObject* tmp = NULL;
787: \tif (PyLong_Check(obj)) {
788: \t\t*v = PyLong_AsLongLong(obj);
789: \t\treturn (!PyErr_Occurred());
790: \t}
791: \tif (PyInt_Check(obj)) {
792: \t\t*v = (long_long)PyInt_AS_LONG(obj);
793: \t\treturn 1;
794: \t}
795: \ttmp = PyNumber_Long(obj);
796: \tif (tmp) {
797: \t\t*v = PyLong_AsLongLong(tmp);
798: \t\tPy_DECREF(tmp);
799: \t\treturn (!PyErr_Occurred());
800: \t}
801: \tif (PyComplex_Check(obj))
802: \t\ttmp = PyObject_GetAttrString(obj,\"real\");
803: \telse if (PyString_Check(obj) || PyUnicode_Check(obj))
804: \t\t/*pass*/;
805: \telse if (PySequence_Check(obj))
806: \t\ttmp = PySequence_GetItem(obj,0);
807: \tif (tmp) {
808: \t\tPyErr_Clear();
809: \t\tif (long_long_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}
810: \t\tPy_DECREF(tmp);
811: \t}
812: \t{
813: \t\tPyObject* err = PyErr_Occurred();
814: \t\tif (err==NULL) err = #modulename#_error;
815: \t\tPyErr_SetString(err,errmess);
816: \t}
817: \treturn 0;
818: }
819: '''
820: needs['long_double_from_pyobj'] = ['double_from_pyobj', 'long_double']
821: cfuncs['long_double_from_pyobj'] = '''\
822: static int long_double_from_pyobj(long_double* v,PyObject *obj,const char *errmess) {
823: \tdouble d=0;
824: \tif (PyArray_CheckScalar(obj)){
825: \t\tif PyArray_IsScalar(obj, LongDouble) {
826: \t\t\tPyArray_ScalarAsCtype(obj, v);
827: \t\t\treturn 1;
828: \t\t}
829: \t\telse if (PyArray_Check(obj) && PyArray_TYPE(obj)==NPY_LONGDOUBLE) {
830: \t\t\t(*v) = *((npy_longdouble *)PyArray_DATA(obj));
831: \t\t\treturn 1;
832: \t\t}
833: \t}
834: \tif (double_from_pyobj(&d,obj,errmess)) {
835: \t\t*v = (long_double)d;
836: \t\treturn 1;
837: \t}
838: \treturn 0;
839: }
840: '''
841: cfuncs['double_from_pyobj'] = '''\
842: static int double_from_pyobj(double* v,PyObject *obj,const char *errmess) {
843: \tPyObject* tmp = NULL;
844: \tif (PyFloat_Check(obj)) {
845: #ifdef __sgi
846: \t\t*v = PyFloat_AsDouble(obj);
847: #else
848: \t\t*v = PyFloat_AS_DOUBLE(obj);
849: #endif
850: \t\treturn 1;
851: \t}
852: \ttmp = PyNumber_Float(obj);
853: \tif (tmp) {
854: #ifdef __sgi
855: \t\t*v = PyFloat_AsDouble(tmp);
856: #else
857: \t\t*v = PyFloat_AS_DOUBLE(tmp);
858: #endif
859: \t\tPy_DECREF(tmp);
860: \t\treturn 1;
861: \t}
862: \tif (PyComplex_Check(obj))
863: \t\ttmp = PyObject_GetAttrString(obj,\"real\");
864: \telse if (PyString_Check(obj) || PyUnicode_Check(obj))
865: \t\t/*pass*/;
866: \telse if (PySequence_Check(obj))
867: \t\ttmp = PySequence_GetItem(obj,0);
868: \tif (tmp) {
869: \t\tPyErr_Clear();
870: \t\tif (double_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}
871: \t\tPy_DECREF(tmp);
872: \t}
873: \t{
874: \t\tPyObject* err = PyErr_Occurred();
875: \t\tif (err==NULL) err = #modulename#_error;
876: \t\tPyErr_SetString(err,errmess);
877: \t}
878: \treturn 0;
879: }
880: '''
881: needs['float_from_pyobj'] = ['double_from_pyobj']
882: cfuncs['float_from_pyobj'] = '''\
883: static int float_from_pyobj(float* v,PyObject *obj,const char *errmess) {
884: \tdouble d=0.0;
885: \tif (double_from_pyobj(&d,obj,errmess)) {
886: \t\t*v = (float)d;
887: \t\treturn 1;
888: \t}
889: \treturn 0;
890: }
891: '''
892: needs['complex_long_double_from_pyobj'] = ['complex_long_double', 'long_double',
893:                                            'complex_double_from_pyobj']
894: cfuncs['complex_long_double_from_pyobj'] = '''\
895: static int complex_long_double_from_pyobj(complex_long_double* v,PyObject *obj,const char *errmess) {
896: \tcomplex_double cd={0.0,0.0};
897: \tif (PyArray_CheckScalar(obj)){
898: \t\tif PyArray_IsScalar(obj, CLongDouble) {
899: \t\t\tPyArray_ScalarAsCtype(obj, v);
900: \t\t\treturn 1;
901: \t\t}
902: \t\telse if (PyArray_Check(obj) && PyArray_TYPE(obj)==NPY_CLONGDOUBLE) {
903: \t\t\t(*v).r = ((npy_clongdouble *)PyArray_DATA(obj))->real;
904: \t\t\t(*v).i = ((npy_clongdouble *)PyArray_DATA(obj))->imag;
905: \t\t\treturn 1;
906: \t\t}
907: \t}
908: \tif (complex_double_from_pyobj(&cd,obj,errmess)) {
909: \t\t(*v).r = (long_double)cd.r;
910: \t\t(*v).i = (long_double)cd.i;
911: \t\treturn 1;
912: \t}
913: \treturn 0;
914: }
915: '''
916: needs['complex_double_from_pyobj'] = ['complex_double']
917: cfuncs['complex_double_from_pyobj'] = '''\
918: static int complex_double_from_pyobj(complex_double* v,PyObject *obj,const char *errmess) {
919: \tPy_complex c;
920: \tif (PyComplex_Check(obj)) {
921: \t\tc=PyComplex_AsCComplex(obj);
922: \t\t(*v).r=c.real, (*v).i=c.imag;
923: \t\treturn 1;
924: \t}
925: \tif (PyArray_IsScalar(obj, ComplexFloating)) {
926: \t\tif (PyArray_IsScalar(obj, CFloat)) {
927: \t\t\tnpy_cfloat new;
928: \t\t\tPyArray_ScalarAsCtype(obj, &new);
929: \t\t\t(*v).r = (double)new.real;
930: \t\t\t(*v).i = (double)new.imag;
931: \t\t}
932: \t\telse if (PyArray_IsScalar(obj, CLongDouble)) {
933: \t\t\tnpy_clongdouble new;
934: \t\t\tPyArray_ScalarAsCtype(obj, &new);
935: \t\t\t(*v).r = (double)new.real;
936: \t\t\t(*v).i = (double)new.imag;
937: \t\t}
938: \t\telse { /* if (PyArray_IsScalar(obj, CDouble)) */
939: \t\t\tPyArray_ScalarAsCtype(obj, v);
940: \t\t}
941: \t\treturn 1;
942: \t}
943: \tif (PyArray_CheckScalar(obj)) { /* 0-dim array or still array scalar */
944: \t\tPyObject *arr;
945: \t\tif (PyArray_Check(obj)) {
946: \t\t\tarr = PyArray_Cast((PyArrayObject *)obj, NPY_CDOUBLE);
947: \t\t}
948: \t\telse {
949: \t\t\tarr = PyArray_FromScalar(obj, PyArray_DescrFromType(NPY_CDOUBLE));
950: \t\t}
951: \t\tif (arr==NULL) return 0;
952: \t\t(*v).r = ((npy_cdouble *)PyArray_DATA(arr))->real;
953: \t\t(*v).i = ((npy_cdouble *)PyArray_DATA(arr))->imag;
954: \t\treturn 1;
955: \t}
956: \t/* Python does not provide PyNumber_Complex function :-( */
957: \t(*v).i=0.0;
958: \tif (PyFloat_Check(obj)) {
959: #ifdef __sgi
960: \t\t(*v).r = PyFloat_AsDouble(obj);
961: #else
962: \t\t(*v).r = PyFloat_AS_DOUBLE(obj);
963: #endif
964: \t\treturn 1;
965: \t}
966: \tif (PyInt_Check(obj)) {
967: \t\t(*v).r = (double)PyInt_AS_LONG(obj);
968: \t\treturn 1;
969: \t}
970: \tif (PyLong_Check(obj)) {
971: \t\t(*v).r = PyLong_AsDouble(obj);
972: \t\treturn (!PyErr_Occurred());
973: \t}
974: \tif (PySequence_Check(obj) && !(PyString_Check(obj) || PyUnicode_Check(obj))) {
975: \t\tPyObject *tmp = PySequence_GetItem(obj,0);
976: \t\tif (tmp) {
977: \t\t\tif (complex_double_from_pyobj(v,tmp,errmess)) {
978: \t\t\t\tPy_DECREF(tmp);
979: \t\t\t\treturn 1;
980: \t\t\t}
981: \t\t\tPy_DECREF(tmp);
982: \t\t}
983: \t}
984: \t{
985: \t\tPyObject* err = PyErr_Occurred();
986: \t\tif (err==NULL)
987: \t\t\terr = PyExc_TypeError;
988: \t\tPyErr_SetString(err,errmess);
989: \t}
990: \treturn 0;
991: }
992: '''
993: needs['complex_float_from_pyobj'] = [
994:     'complex_float', 'complex_double_from_pyobj']
995: cfuncs['complex_float_from_pyobj'] = '''\
996: static int complex_float_from_pyobj(complex_float* v,PyObject *obj,const char *errmess) {
997: \tcomplex_double cd={0.0,0.0};
998: \tif (complex_double_from_pyobj(&cd,obj,errmess)) {
999: \t\t(*v).r = (float)cd.r;
1000: \t\t(*v).i = (float)cd.i;
1001: \t\treturn 1;
1002: \t}
1003: \treturn 0;
1004: }
1005: '''
1006: needs['try_pyarr_from_char'] = ['pyobj_from_char1', 'TRYPYARRAYTEMPLATE']
1007: cfuncs[
1008:     'try_pyarr_from_char'] = 'static int try_pyarr_from_char(PyObject* obj,char* v) {\n\tTRYPYARRAYTEMPLATE(char,\'c\');\n}\n'
1009: needs['try_pyarr_from_signed_char'] = ['TRYPYARRAYTEMPLATE', 'unsigned_char']
1010: cfuncs[
1011:     'try_pyarr_from_unsigned_char'] = 'static int try_pyarr_from_unsigned_char(PyObject* obj,unsigned_char* v) {\n\tTRYPYARRAYTEMPLATE(unsigned_char,\'b\');\n}\n'
1012: needs['try_pyarr_from_signed_char'] = ['TRYPYARRAYTEMPLATE', 'signed_char']
1013: cfuncs[
1014:     'try_pyarr_from_signed_char'] = 'static int try_pyarr_from_signed_char(PyObject* obj,signed_char* v) {\n\tTRYPYARRAYTEMPLATE(signed_char,\'1\');\n}\n'
1015: needs['try_pyarr_from_short'] = ['pyobj_from_short1', 'TRYPYARRAYTEMPLATE']
1016: cfuncs[
1017:     'try_pyarr_from_short'] = 'static int try_pyarr_from_short(PyObject* obj,short* v) {\n\tTRYPYARRAYTEMPLATE(short,\'s\');\n}\n'
1018: needs['try_pyarr_from_int'] = ['pyobj_from_int1', 'TRYPYARRAYTEMPLATE']
1019: cfuncs[
1020:     'try_pyarr_from_int'] = 'static int try_pyarr_from_int(PyObject* obj,int* v) {\n\tTRYPYARRAYTEMPLATE(int,\'i\');\n}\n'
1021: needs['try_pyarr_from_long'] = ['pyobj_from_long1', 'TRYPYARRAYTEMPLATE']
1022: cfuncs[
1023:     'try_pyarr_from_long'] = 'static int try_pyarr_from_long(PyObject* obj,long* v) {\n\tTRYPYARRAYTEMPLATE(long,\'l\');\n}\n'
1024: needs['try_pyarr_from_long_long'] = [
1025:     'pyobj_from_long_long1', 'TRYPYARRAYTEMPLATE', 'long_long']
1026: cfuncs[
1027:     'try_pyarr_from_long_long'] = 'static int try_pyarr_from_long_long(PyObject* obj,long_long* v) {\n\tTRYPYARRAYTEMPLATE(long_long,\'L\');\n}\n'
1028: needs['try_pyarr_from_float'] = ['pyobj_from_float1', 'TRYPYARRAYTEMPLATE']
1029: cfuncs[
1030:     'try_pyarr_from_float'] = 'static int try_pyarr_from_float(PyObject* obj,float* v) {\n\tTRYPYARRAYTEMPLATE(float,\'f\');\n}\n'
1031: needs['try_pyarr_from_double'] = ['pyobj_from_double1', 'TRYPYARRAYTEMPLATE']
1032: cfuncs[
1033:     'try_pyarr_from_double'] = 'static int try_pyarr_from_double(PyObject* obj,double* v) {\n\tTRYPYARRAYTEMPLATE(double,\'d\');\n}\n'
1034: needs['try_pyarr_from_complex_float'] = [
1035:     'pyobj_from_complex_float1', 'TRYCOMPLEXPYARRAYTEMPLATE', 'complex_float']
1036: cfuncs[
1037:     'try_pyarr_from_complex_float'] = 'static int try_pyarr_from_complex_float(PyObject* obj,complex_float* v) {\n\tTRYCOMPLEXPYARRAYTEMPLATE(float,\'F\');\n}\n'
1038: needs['try_pyarr_from_complex_double'] = [
1039:     'pyobj_from_complex_double1', 'TRYCOMPLEXPYARRAYTEMPLATE', 'complex_double']
1040: cfuncs[
1041:     'try_pyarr_from_complex_double'] = 'static int try_pyarr_from_complex_double(PyObject* obj,complex_double* v) {\n\tTRYCOMPLEXPYARRAYTEMPLATE(double,\'D\');\n}\n'
1042: 
1043: needs['create_cb_arglist'] = ['CFUNCSMESS', 'PRINTPYOBJERR', 'MINMAX']
1044: cfuncs['create_cb_arglist'] = '''\
1045: static int create_cb_arglist(PyObject* fun,PyTupleObject* xa,const int maxnofargs,const int nofoptargs,int *nofargs,PyTupleObject **args,const char *errmess) {
1046: \tPyObject *tmp = NULL;
1047: \tPyObject *tmp_fun = NULL;
1048: \tint tot,opt,ext,siz,i,di=0;
1049: \tCFUNCSMESS(\"create_cb_arglist\\n\");
1050: \ttot=opt=ext=siz=0;
1051: \t/* Get the total number of arguments */
1052: \tif (PyFunction_Check(fun))
1053: \t\ttmp_fun = fun;
1054: \telse {
1055: \t\tdi = 1;
1056: \t\tif (PyObject_HasAttrString(fun,\"im_func\")) {
1057: \t\t\ttmp_fun = PyObject_GetAttrString(fun,\"im_func\");
1058: \t\t}
1059: \t\telse if (PyObject_HasAttrString(fun,\"__call__\")) {
1060: \t\t\ttmp = PyObject_GetAttrString(fun,\"__call__\");
1061: \t\t\tif (PyObject_HasAttrString(tmp,\"im_func\"))
1062: \t\t\t\ttmp_fun = PyObject_GetAttrString(tmp,\"im_func\");
1063: \t\t\telse {
1064: \t\t\t\ttmp_fun = fun; /* built-in function */
1065: \t\t\t\ttot = maxnofargs;
1066: \t\t\t\tif (xa != NULL)
1067: \t\t\t\t\ttot += PyTuple_Size((PyObject *)xa);
1068: \t\t\t}
1069: \t\t\tPy_XDECREF(tmp);
1070: \t\t}
1071: \t\telse if (PyFortran_Check(fun) || PyFortran_Check1(fun)) {
1072: \t\t\ttot = maxnofargs;
1073: \t\t\tif (xa != NULL)
1074: \t\t\t\ttot += PyTuple_Size((PyObject *)xa);
1075: \t\t\ttmp_fun = fun;
1076: \t\t}
1077: \t\telse if (F2PyCapsule_Check(fun)) {
1078: \t\t\ttot = maxnofargs;
1079: \t\t\tif (xa != NULL)
1080: \t\t\t\text = PyTuple_Size((PyObject *)xa);
1081: \t\t\tif(ext>0) {
1082: \t\t\t\tfprintf(stderr,\"extra arguments tuple cannot be used with CObject call-back\\n\");
1083: \t\t\t\tgoto capi_fail;
1084: \t\t\t}
1085: \t\t\ttmp_fun = fun;
1086: \t\t}
1087: \t}
1088: if (tmp_fun==NULL) {
1089: fprintf(stderr,\"Call-back argument must be function|instance|instance.__call__|f2py-function but got %s.\\n\",(fun==NULL?\"NULL\":Py_TYPE(fun)->tp_name));
1090: goto capi_fail;
1091: }
1092: #if PY_VERSION_HEX >= 0x03000000
1093: \tif (PyObject_HasAttrString(tmp_fun,\"__code__\")) {
1094: \t\tif (PyObject_HasAttrString(tmp = PyObject_GetAttrString(tmp_fun,\"__code__\"),\"co_argcount\"))
1095: #else
1096: \tif (PyObject_HasAttrString(tmp_fun,\"func_code\")) {
1097: \t\tif (PyObject_HasAttrString(tmp = PyObject_GetAttrString(tmp_fun,\"func_code\"),\"co_argcount\"))
1098: #endif
1099: \t\t\ttot = PyInt_AsLong(PyObject_GetAttrString(tmp,\"co_argcount\")) - di;
1100: \t\tPy_XDECREF(tmp);
1101: \t}
1102: \t/* Get the number of optional arguments */
1103: #if PY_VERSION_HEX >= 0x03000000
1104: \tif (PyObject_HasAttrString(tmp_fun,\"__defaults__\")) {
1105: \t\tif (PyTuple_Check(tmp = PyObject_GetAttrString(tmp_fun,\"__defaults__\")))
1106: #else
1107: \tif (PyObject_HasAttrString(tmp_fun,\"func_defaults\")) {
1108: \t\tif (PyTuple_Check(tmp = PyObject_GetAttrString(tmp_fun,\"func_defaults\")))
1109: #endif
1110: \t\t\topt = PyTuple_Size(tmp);
1111: \t\tPy_XDECREF(tmp);
1112: \t}
1113: \t/* Get the number of extra arguments */
1114: \tif (xa != NULL)
1115: \t\text = PyTuple_Size((PyObject *)xa);
1116: \t/* Calculate the size of call-backs argument list */
1117: \tsiz = MIN(maxnofargs+ext,tot);
1118: \t*nofargs = MAX(0,siz-ext);
1119: #ifdef DEBUGCFUNCS
1120: \tfprintf(stderr,\"debug-capi:create_cb_arglist:maxnofargs(-nofoptargs),tot,opt,ext,siz,nofargs=%d(-%d),%d,%d,%d,%d,%d\\n\",maxnofargs,nofoptargs,tot,opt,ext,siz,*nofargs);
1121: #endif
1122: \tif (siz<tot-opt) {
1123: \t\tfprintf(stderr,\"create_cb_arglist: Failed to build argument list (siz) with enough arguments (tot-opt) required by user-supplied function (siz,tot,opt=%d,%d,%d).\\n\",siz,tot,opt);
1124: \t\tgoto capi_fail;
1125: \t}
1126: \t/* Initialize argument list */
1127: \t*args = (PyTupleObject *)PyTuple_New(siz);
1128: \tfor (i=0;i<*nofargs;i++) {
1129: \t\tPy_INCREF(Py_None);
1130: \t\tPyTuple_SET_ITEM((PyObject *)(*args),i,Py_None);
1131: \t}
1132: \tif (xa != NULL)
1133: \t\tfor (i=(*nofargs);i<siz;i++) {
1134: \t\t\ttmp = PyTuple_GetItem((PyObject *)xa,i-(*nofargs));
1135: \t\t\tPy_INCREF(tmp);
1136: \t\t\tPyTuple_SET_ITEM(*args,i,tmp);
1137: \t\t}
1138: \tCFUNCSMESS(\"create_cb_arglist-end\\n\");
1139: \treturn 1;
1140: capi_fail:
1141: \tif ((PyErr_Occurred())==NULL)
1142: \t\tPyErr_SetString(#modulename#_error,errmess);
1143: \treturn 0;
1144: }
1145: '''
1146: 
1147: 
1148: def buildcfuncs():
1149:     from .capi_maps import c2capi_map
1150:     for k in c2capi_map.keys():
1151:         m = 'pyarr_from_p_%s1' % k
1152:         cppmacros[
1153:             m] = '#define %s(v) (PyArray_SimpleNewFromData(0,NULL,%s,(char *)v))' % (m, c2capi_map[k])
1154:     k = 'string'
1155:     m = 'pyarr_from_p_%s1' % k
1156:     cppmacros[
1157:         m] = '#define %s(v,dims) (PyArray_SimpleNewFromData(1,dims,NPY_CHAR,(char *)v))' % (m)
1158: 
1159: 
1160: ############ Auxiliary functions for sorting needs ###################
1161: 
1162: def append_needs(need, flag=1):
1163:     global outneeds, needs
1164:     if isinstance(need, list):
1165:         for n in need:
1166:             append_needs(n, flag)
1167:     elif isinstance(need, str):
1168:         if not need:
1169:             return
1170:         if need in includes0:
1171:             n = 'includes0'
1172:         elif need in includes:
1173:             n = 'includes'
1174:         elif need in typedefs:
1175:             n = 'typedefs'
1176:         elif need in typedefs_generated:
1177:             n = 'typedefs_generated'
1178:         elif need in cppmacros:
1179:             n = 'cppmacros'
1180:         elif need in cfuncs:
1181:             n = 'cfuncs'
1182:         elif need in callbacks:
1183:             n = 'callbacks'
1184:         elif need in f90modhooks:
1185:             n = 'f90modhooks'
1186:         elif need in commonhooks:
1187:             n = 'commonhooks'
1188:         else:
1189:             errmess('append_needs: unknown need %s\n' % (repr(need)))
1190:             return
1191:         if need in outneeds[n]:
1192:             return
1193:         if flag:
1194:             tmp = {}
1195:             if need in needs:
1196:                 for nn in needs[need]:
1197:                     t = append_needs(nn, 0)
1198:                     if isinstance(t, dict):
1199:                         for nnn in t.keys():
1200:                             if nnn in tmp:
1201:                                 tmp[nnn] = tmp[nnn] + t[nnn]
1202:                             else:
1203:                                 tmp[nnn] = t[nnn]
1204:             for nn in tmp.keys():
1205:                 for nnn in tmp[nn]:
1206:                     if nnn not in outneeds[nn]:
1207:                         outneeds[nn] = [nnn] + outneeds[nn]
1208:             outneeds[n].append(need)
1209:         else:
1210:             tmp = {}
1211:             if need in needs:
1212:                 for nn in needs[need]:
1213:                     t = append_needs(nn, flag)
1214:                     if isinstance(t, dict):
1215:                         for nnn in t.keys():
1216:                             if nnn in tmp:
1217:                                 tmp[nnn] = t[nnn] + tmp[nnn]
1218:                             else:
1219:                                 tmp[nnn] = t[nnn]
1220:             if n not in tmp:
1221:                 tmp[n] = []
1222:             tmp[n].append(need)
1223:             return tmp
1224:     else:
1225:         errmess('append_needs: expected list or string but got :%s\n' %
1226:                 (repr(need)))
1227: 
1228: 
1229: def get_needs():
1230:     global outneeds, needs
1231:     res = {}
1232:     for n in outneeds.keys():
1233:         out = []
1234:         saveout = copy.copy(outneeds[n])
1235:         while len(outneeds[n]) > 0:
1236:             if outneeds[n][0] not in needs:
1237:                 out.append(outneeds[n][0])
1238:                 del outneeds[n][0]
1239:             else:
1240:                 flag = 0
1241:                 for k in outneeds[n][1:]:
1242:                     if k in needs[outneeds[n][0]]:
1243:                         flag = 1
1244:                         break
1245:                 if flag:
1246:                     outneeds[n] = outneeds[n][1:] + [outneeds[n][0]]
1247:                 else:
1248:                     out.append(outneeds[n][0])
1249:                     del outneeds[n][0]
1250:             if saveout and (0 not in map(lambda x, y: x == y, saveout, outneeds[n])) \
1251:                     and outneeds[n] != []:
1252:                 print(n, saveout)
1253:                 errmess(
1254:                     'get_needs: no progress in sorting needs, probably circular dependence, skipping.\n')
1255:                 out = out + saveout
1256:                 break
1257:             saveout = copy.copy(outneeds[n])
1258:         if out == []:
1259:             out = [n]
1260:         res[n] = out
1261:     return res
1262: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_74199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, (-1)), 'str', '\n\nC declarations, CPP macros, and C functions for f2py2e.\nOnly required declarations/macros/functions will be used.\n\nCopyright 1999,2000 Pearu Peterson all rights reserved,\nPearu Peterson <pearu@ioc.ee>\nPermission to use, modify, and distribute this software is given under the\nterms of the NumPy License.\n\nNO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.\n$Date: 2005/05/06 11:42:34 $\nPearu Peterson\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import sys' statement (line 19)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import copy' statement (line 20)
import copy

import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'copy', copy, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from numpy.f2py import __version__' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_74200 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.f2py')

if (type(import_74200) is not StypyTypeError):

    if (import_74200 != 'pyd_module'):
        __import__(import_74200)
        sys_modules_74201 = sys.modules[import_74200]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.f2py', sys_modules_74201.module_type_store, module_type_store, ['__version__'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_74201, sys_modules_74201.module_type_store, module_type_store)
    else:
        from numpy.f2py import __version__

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.f2py', None, module_type_store, ['__version__'], [__version__])

else:
    # Assigning a type to the variable 'numpy.f2py' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.f2py', import_74200)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')


# Assigning a Attribute to a Name (line 24):
# Getting the type of '__version__' (line 24)
version___74202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 15), '__version__')
# Obtaining the member 'version' of a type (line 24)
version_74203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 15), version___74202, 'version')
# Assigning a type to the variable 'f2py_version' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'f2py_version', version_74203)

# Assigning a Attribute to a Name (line 25):
# Getting the type of 'sys' (line 25)
sys_74204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 10), 'sys')
# Obtaining the member 'stderr' of a type (line 25)
stderr_74205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 10), sys_74204, 'stderr')
# Obtaining the member 'write' of a type (line 25)
write_74206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 10), stderr_74205, 'write')
# Assigning a type to the variable 'errmess' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'errmess', write_74206)

# Assigning a Dict to a Name (line 29):

# Obtaining an instance of the builtin type 'dict' (line 29)
dict_74207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 11), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 29)
# Adding element type (key, value) (line 29)
str_74208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 12), 'str', 'includes0')

# Obtaining an instance of the builtin type 'list' (line 29)
list_74209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 29)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 11), dict_74207, (str_74208, list_74209))
# Adding element type (key, value) (line 29)
str_74210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 29), 'str', 'includes')

# Obtaining an instance of the builtin type 'list' (line 29)
list_74211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 41), 'list')
# Adding type elements to the builtin type 'list' instance (line 29)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 11), dict_74207, (str_74210, list_74211))
# Adding element type (key, value) (line 29)
str_74212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 45), 'str', 'typedefs')

# Obtaining an instance of the builtin type 'list' (line 29)
list_74213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 57), 'list')
# Adding type elements to the builtin type 'list' instance (line 29)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 11), dict_74207, (str_74212, list_74213))
# Adding element type (key, value) (line 29)
str_74214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 61), 'str', 'typedefs_generated')

# Obtaining an instance of the builtin type 'list' (line 29)
list_74215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 83), 'list')
# Adding type elements to the builtin type 'list' instance (line 29)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 11), dict_74207, (str_74214, list_74215))
# Adding element type (key, value) (line 29)
str_74216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 12), 'str', 'userincludes')

# Obtaining an instance of the builtin type 'list' (line 30)
list_74217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 30)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 11), dict_74207, (str_74216, list_74217))
# Adding element type (key, value) (line 29)
str_74218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 12), 'str', 'cppmacros')

# Obtaining an instance of the builtin type 'list' (line 31)
list_74219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 31)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 11), dict_74207, (str_74218, list_74219))
# Adding element type (key, value) (line 29)
str_74220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 29), 'str', 'cfuncs')

# Obtaining an instance of the builtin type 'list' (line 31)
list_74221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 39), 'list')
# Adding type elements to the builtin type 'list' instance (line 31)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 11), dict_74207, (str_74220, list_74221))
# Adding element type (key, value) (line 29)
str_74222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 43), 'str', 'callbacks')

# Obtaining an instance of the builtin type 'list' (line 31)
list_74223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 56), 'list')
# Adding type elements to the builtin type 'list' instance (line 31)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 11), dict_74207, (str_74222, list_74223))
# Adding element type (key, value) (line 29)
str_74224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 60), 'str', 'f90modhooks')

# Obtaining an instance of the builtin type 'list' (line 31)
list_74225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 75), 'list')
# Adding type elements to the builtin type 'list' instance (line 31)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 11), dict_74207, (str_74224, list_74225))
# Adding element type (key, value) (line 29)
str_74226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 12), 'str', 'commonhooks')

# Obtaining an instance of the builtin type 'list' (line 32)
list_74227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 32)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 11), dict_74207, (str_74226, list_74227))

# Assigning a type to the variable 'outneeds' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'outneeds', dict_74207)

# Assigning a Dict to a Name (line 33):

# Obtaining an instance of the builtin type 'dict' (line 33)
dict_74228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 8), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 33)

# Assigning a type to the variable 'needs' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'needs', dict_74228)

# Assigning a Dict to a Name (line 34):

# Obtaining an instance of the builtin type 'dict' (line 34)
dict_74229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 34)
# Adding element type (key, value) (line 34)
str_74230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 13), 'str', 'includes0')
str_74231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 26), 'str', '/*need_includes0*/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 12), dict_74229, (str_74230, str_74231))

# Assigning a type to the variable 'includes0' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'includes0', dict_74229)

# Assigning a Dict to a Name (line 35):

# Obtaining an instance of the builtin type 'dict' (line 35)
dict_74232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 11), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 35)
# Adding element type (key, value) (line 35)
str_74233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 12), 'str', 'includes')
str_74234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 24), 'str', '/*need_includes*/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 11), dict_74232, (str_74233, str_74234))

# Assigning a type to the variable 'includes' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'includes', dict_74232)

# Assigning a Dict to a Name (line 36):

# Obtaining an instance of the builtin type 'dict' (line 36)
dict_74235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 15), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 36)
# Adding element type (key, value) (line 36)
str_74236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 16), 'str', 'userincludes')
str_74237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 32), 'str', '/*need_userincludes*/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 15), dict_74235, (str_74236, str_74237))

# Assigning a type to the variable 'userincludes' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'userincludes', dict_74235)

# Assigning a Dict to a Name (line 37):

# Obtaining an instance of the builtin type 'dict' (line 37)
dict_74238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 11), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 37)
# Adding element type (key, value) (line 37)
str_74239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 12), 'str', 'typedefs')
str_74240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 24), 'str', '/*need_typedefs*/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 11), dict_74238, (str_74239, str_74240))

# Assigning a type to the variable 'typedefs' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'typedefs', dict_74238)

# Assigning a Dict to a Name (line 38):

# Obtaining an instance of the builtin type 'dict' (line 38)
dict_74241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 21), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 38)
# Adding element type (key, value) (line 38)
str_74242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 22), 'str', 'typedefs_generated')
str_74243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 44), 'str', '/*need_typedefs_generated*/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 21), dict_74241, (str_74242, str_74243))

# Assigning a type to the variable 'typedefs_generated' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'typedefs_generated', dict_74241)

# Assigning a Dict to a Name (line 39):

# Obtaining an instance of the builtin type 'dict' (line 39)
dict_74244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 39)
# Adding element type (key, value) (line 39)
str_74245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 13), 'str', 'cppmacros')
str_74246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 26), 'str', '/*need_cppmacros*/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 12), dict_74244, (str_74245, str_74246))

# Assigning a type to the variable 'cppmacros' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'cppmacros', dict_74244)

# Assigning a Dict to a Name (line 40):

# Obtaining an instance of the builtin type 'dict' (line 40)
dict_74247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 40)
# Adding element type (key, value) (line 40)
str_74248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 10), 'str', 'cfuncs')
str_74249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 20), 'str', '/*need_cfuncs*/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 9), dict_74247, (str_74248, str_74249))

# Assigning a type to the variable 'cfuncs' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'cfuncs', dict_74247)

# Assigning a Dict to a Name (line 41):

# Obtaining an instance of the builtin type 'dict' (line 41)
dict_74250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 41)
# Adding element type (key, value) (line 41)
str_74251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 13), 'str', 'callbacks')
str_74252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 26), 'str', '/*need_callbacks*/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 12), dict_74250, (str_74251, str_74252))

# Assigning a type to the variable 'callbacks' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'callbacks', dict_74250)

# Assigning a Dict to a Name (line 42):

# Obtaining an instance of the builtin type 'dict' (line 42)
dict_74253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 42)
# Adding element type (key, value) (line 42)
str_74254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 15), 'str', 'f90modhooks')
str_74255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 30), 'str', '/*need_f90modhooks*/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 14), dict_74253, (str_74254, str_74255))
# Adding element type (key, value) (line 42)
str_74256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 15), 'str', 'initf90modhooksstatic')
str_74257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 40), 'str', '/*initf90modhooksstatic*/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 14), dict_74253, (str_74256, str_74257))
# Adding element type (key, value) (line 42)
str_74258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 15), 'str', 'initf90modhooksdynamic')
str_74259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 41), 'str', '/*initf90modhooksdynamic*/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 14), dict_74253, (str_74258, str_74259))

# Assigning a type to the variable 'f90modhooks' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'f90modhooks', dict_74253)

# Assigning a Dict to a Name (line 46):

# Obtaining an instance of the builtin type 'dict' (line 46)
dict_74260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 46)
# Adding element type (key, value) (line 46)
str_74261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 15), 'str', 'commonhooks')
str_74262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 30), 'str', '/*need_commonhooks*/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 14), dict_74260, (str_74261, str_74262))
# Adding element type (key, value) (line 46)
str_74263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 15), 'str', 'initcommonhooks')
str_74264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 34), 'str', '/*need_initcommonhooks*/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 14), dict_74260, (str_74263, str_74264))

# Assigning a type to the variable 'commonhooks' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'commonhooks', dict_74260)

# Assigning a Str to a Subscript (line 52):
str_74265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 22), 'str', '#include <math.h>')
# Getting the type of 'includes0' (line 52)
includes0_74266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'includes0')
str_74267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 10), 'str', 'math.h')
# Storing an element on a container (line 52)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 0), includes0_74266, (str_74267, str_74265))

# Assigning a Str to a Subscript (line 53):
str_74268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 24), 'str', '#include <string.h>')
# Getting the type of 'includes0' (line 53)
includes0_74269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'includes0')
str_74270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 10), 'str', 'string.h')
# Storing an element on a container (line 53)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 0), includes0_74269, (str_74270, str_74268))

# Assigning a Str to a Subscript (line 54):
str_74271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 24), 'str', '#include <setjmp.h>')
# Getting the type of 'includes0' (line 54)
includes0_74272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'includes0')
str_74273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 10), 'str', 'setjmp.h')
# Storing an element on a container (line 54)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 0), includes0_74272, (str_74273, str_74271))

# Assigning a Str to a Subscript (line 56):
str_74274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 23), 'str', '#include "Python.h"')
# Getting the type of 'includes' (line 56)
includes_74275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'includes')
str_74276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 9), 'str', 'Python.h')
# Storing an element on a container (line 56)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 0), includes_74275, (str_74276, str_74274))

# Assigning a List to a Subscript (line 57):

# Obtaining an instance of the builtin type 'list' (line 57)
list_74277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 57)
# Adding element type (line 57)
str_74278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 26), 'str', 'Python.h')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 25), list_74277, str_74278)

# Getting the type of 'needs' (line 57)
needs_74279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'needs')
str_74280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 6), 'str', 'arrayobject.h')
# Storing an element on a container (line 57)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 0), needs_74279, (str_74280, list_74277))

# Assigning a Str to a Subscript (line 58):
str_74281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, (-1)), 'str', '#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API\n#include "arrayobject.h"')
# Getting the type of 'includes' (line 58)
includes_74282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'includes')
str_74283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 9), 'str', 'arrayobject.h')
# Storing an element on a container (line 58)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 0), includes_74282, (str_74283, str_74281))

# Assigning a Str to a Subscript (line 61):
str_74284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 28), 'str', '#include "fortranobject.h"')
# Getting the type of 'includes' (line 61)
includes_74285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'includes')
str_74286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 9), 'str', 'arrayobject.h')
# Storing an element on a container (line 61)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 0), includes_74285, (str_74286, str_74284))

# Assigning a Str to a Subscript (line 62):
str_74287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 23), 'str', '#include <stdarg.h>')
# Getting the type of 'includes' (line 62)
includes_74288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'includes')
str_74289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 9), 'str', 'stdarg.h')
# Storing an element on a container (line 62)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 0), includes_74288, (str_74289, str_74287))

# Assigning a Str to a Subscript (line 66):
str_74290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 28), 'str', 'typedef unsigned char unsigned_char;')
# Getting the type of 'typedefs' (line 66)
typedefs_74291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'typedefs')
str_74292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 9), 'str', 'unsigned_char')
# Storing an element on a container (line 66)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 0), typedefs_74291, (str_74292, str_74290))

# Assigning a Str to a Subscript (line 67):
str_74293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 29), 'str', 'typedef unsigned short unsigned_short;')
# Getting the type of 'typedefs' (line 67)
typedefs_74294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'typedefs')
str_74295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 9), 'str', 'unsigned_short')
# Storing an element on a container (line 67)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 0), typedefs_74294, (str_74295, str_74293))

# Assigning a Str to a Subscript (line 68):
str_74296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 28), 'str', 'typedef unsigned long unsigned_long;')
# Getting the type of 'typedefs' (line 68)
typedefs_74297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'typedefs')
str_74298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 9), 'str', 'unsigned_long')
# Storing an element on a container (line 68)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 0), typedefs_74297, (str_74298, str_74296))

# Assigning a Str to a Subscript (line 69):
str_74299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 26), 'str', 'typedef signed char signed_char;')
# Getting the type of 'typedefs' (line 69)
typedefs_74300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'typedefs')
str_74301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 9), 'str', 'signed_char')
# Storing an element on a container (line 69)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 0), typedefs_74300, (str_74301, str_74299))

# Assigning a Str to a Subscript (line 70):
str_74302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, (-1)), 'str', '#ifdef _WIN32\ntypedef __int64 long_long;\n#else\ntypedef long long long_long;\ntypedef unsigned long long unsigned_long_long;\n#endif\n')
# Getting the type of 'typedefs' (line 70)
typedefs_74303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'typedefs')
str_74304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 9), 'str', 'long_long')
# Storing an element on a container (line 70)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 0), typedefs_74303, (str_74304, str_74302))

# Assigning a Str to a Subscript (line 78):
str_74305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, (-1)), 'str', '#ifdef _WIN32\ntypedef __uint64 long_long;\n#else\ntypedef unsigned long long unsigned_long_long;\n#endif\n')
# Getting the type of 'typedefs' (line 78)
typedefs_74306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'typedefs')
str_74307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 9), 'str', 'unsigned_long_long')
# Storing an element on a container (line 78)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 0), typedefs_74306, (str_74307, str_74305))

# Assigning a Str to a Subscript (line 85):
str_74308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, (-1)), 'str', '#ifndef _LONG_DOUBLE\ntypedef long double long_double;\n#endif\n')
# Getting the type of 'typedefs' (line 85)
typedefs_74309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'typedefs')
str_74310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 9), 'str', 'long_double')
# Storing an element on a container (line 85)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 0), typedefs_74309, (str_74310, str_74308))

# Assigning a Str to a Subscript (line 90):
str_74311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 29), 'str', 'typedef struct {long double r,i;} complex_long_double;')
# Getting the type of 'typedefs' (line 90)
typedefs_74312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'typedefs')
str_74313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 4), 'str', 'complex_long_double')
# Storing an element on a container (line 90)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 0), typedefs_74312, (str_74313, str_74311))

# Assigning a Str to a Subscript (line 92):
str_74314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 28), 'str', 'typedef struct {float r,i;} complex_float;')
# Getting the type of 'typedefs' (line 92)
typedefs_74315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'typedefs')
str_74316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 9), 'str', 'complex_float')
# Storing an element on a container (line 92)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 0), typedefs_74315, (str_74316, str_74314))

# Assigning a Str to a Subscript (line 93):
str_74317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 29), 'str', 'typedef struct {double r,i;} complex_double;')
# Getting the type of 'typedefs' (line 93)
typedefs_74318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'typedefs')
str_74319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 9), 'str', 'complex_double')
# Storing an element on a container (line 93)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 0), typedefs_74318, (str_74319, str_74317))

# Assigning a Str to a Subscript (line 94):
str_74320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 21), 'str', 'typedef char * string;')
# Getting the type of 'typedefs' (line 94)
typedefs_74321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'typedefs')
str_74322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 9), 'str', 'string')
# Storing an element on a container (line 94)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 0), typedefs_74321, (str_74322, str_74320))

# Assigning a Str to a Subscript (line 98):
str_74323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, (-1)), 'str', '#ifdef DEBUGCFUNCS\n#define CFUNCSMESS(mess) fprintf(stderr,"debug-capi:"mess);\n#define CFUNCSMESSPY(mess,obj) CFUNCSMESS(mess) \\\n\tPyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\\\n\tfprintf(stderr,"\\n");\n#else\n#define CFUNCSMESS(mess)\n#define CFUNCSMESSPY(mess,obj)\n#endif\n')
# Getting the type of 'cppmacros' (line 98)
cppmacros_74324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'cppmacros')
str_74325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 10), 'str', 'CFUNCSMESS')
# Storing an element on a container (line 98)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 0), cppmacros_74324, (str_74325, str_74323))

# Assigning a Str to a Subscript (line 109):
str_74326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, (-1)), 'str', '#if defined(PREPEND_FORTRAN)\n#if defined(NO_APPEND_FORTRAN)\n#if defined(UPPERCASE_FORTRAN)\n#define F_FUNC(f,F) _##F\n#else\n#define F_FUNC(f,F) _##f\n#endif\n#else\n#if defined(UPPERCASE_FORTRAN)\n#define F_FUNC(f,F) _##F##_\n#else\n#define F_FUNC(f,F) _##f##_\n#endif\n#endif\n#else\n#if defined(NO_APPEND_FORTRAN)\n#if defined(UPPERCASE_FORTRAN)\n#define F_FUNC(f,F) F\n#else\n#define F_FUNC(f,F) f\n#endif\n#else\n#if defined(UPPERCASE_FORTRAN)\n#define F_FUNC(f,F) F##_\n#else\n#define F_FUNC(f,F) f##_\n#endif\n#endif\n#endif\n#if defined(UNDERSCORE_G77)\n#define F_FUNC_US(f,F) F_FUNC(f##_,F##_)\n#else\n#define F_FUNC_US(f,F) F_FUNC(f,F)\n#endif\n')
# Getting the type of 'cppmacros' (line 109)
cppmacros_74327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'cppmacros')
str_74328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 10), 'str', 'F_FUNC')
# Storing an element on a container (line 109)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 0), cppmacros_74327, (str_74328, str_74326))

# Assigning a Str to a Subscript (line 145):
str_74329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, (-1)), 'str', '#if defined(PREPEND_FORTRAN)\n#if defined(NO_APPEND_FORTRAN)\n#if defined(UPPERCASE_FORTRAN)\n#define F_WRAPPEDFUNC(f,F) _F2PYWRAP##F\n#else\n#define F_WRAPPEDFUNC(f,F) _f2pywrap##f\n#endif\n#else\n#if defined(UPPERCASE_FORTRAN)\n#define F_WRAPPEDFUNC(f,F) _F2PYWRAP##F##_\n#else\n#define F_WRAPPEDFUNC(f,F) _f2pywrap##f##_\n#endif\n#endif\n#else\n#if defined(NO_APPEND_FORTRAN)\n#if defined(UPPERCASE_FORTRAN)\n#define F_WRAPPEDFUNC(f,F) F2PYWRAP##F\n#else\n#define F_WRAPPEDFUNC(f,F) f2pywrap##f\n#endif\n#else\n#if defined(UPPERCASE_FORTRAN)\n#define F_WRAPPEDFUNC(f,F) F2PYWRAP##F##_\n#else\n#define F_WRAPPEDFUNC(f,F) f2pywrap##f##_\n#endif\n#endif\n#endif\n#if defined(UNDERSCORE_G77)\n#define F_WRAPPEDFUNC_US(f,F) F_WRAPPEDFUNC(f##_,F##_)\n#else\n#define F_WRAPPEDFUNC_US(f,F) F_WRAPPEDFUNC(f,F)\n#endif\n')
# Getting the type of 'cppmacros' (line 145)
cppmacros_74330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 0), 'cppmacros')
str_74331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 10), 'str', 'F_WRAPPEDFUNC')
# Storing an element on a container (line 145)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 0), cppmacros_74330, (str_74331, str_74329))

# Assigning a Str to a Subscript (line 181):
str_74332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, (-1)), 'str', '#if defined(F90MOD2CCONV1) /*E.g. Compaq Fortran */\n#if defined(NO_APPEND_FORTRAN)\n#define F_MODFUNCNAME(m,f) $ ## m ## $ ## f\n#else\n#define F_MODFUNCNAME(m,f) $ ## m ## $ ## f ## _\n#endif\n#endif\n\n#if defined(F90MOD2CCONV2) /*E.g. IBM XL Fortran, not tested though */\n#if defined(NO_APPEND_FORTRAN)\n#define F_MODFUNCNAME(m,f)  __ ## m ## _MOD_ ## f\n#else\n#define F_MODFUNCNAME(m,f)  __ ## m ## _MOD_ ## f ## _\n#endif\n#endif\n\n#if defined(F90MOD2CCONV3) /*E.g. MIPSPro Compilers */\n#if defined(NO_APPEND_FORTRAN)\n#define F_MODFUNCNAME(m,f)  f ## .in. ## m\n#else\n#define F_MODFUNCNAME(m,f)  f ## .in. ## m ## _\n#endif\n#endif\n/*\n#if defined(UPPERCASE_FORTRAN)\n#define F_MODFUNC(m,M,f,F) F_MODFUNCNAME(M,F)\n#else\n#define F_MODFUNC(m,M,f,F) F_MODFUNCNAME(m,f)\n#endif\n*/\n\n#define F_MODFUNC(m,f) (*(f2pymodstruct##m##.##f))\n')
# Getting the type of 'cppmacros' (line 181)
cppmacros_74333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 0), 'cppmacros')
str_74334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 10), 'str', 'F_MODFUNC')
# Storing an element on a container (line 181)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 0), cppmacros_74333, (str_74334, str_74332))

# Assigning a Str to a Subscript (line 215):
str_74335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, (-1)), 'str', '#define SWAP(a,b) (size_t)(a) = ((size_t)(a) ^ (size_t)(b));\\\n (size_t)(b) = ((size_t)(a) ^ (size_t)(b));\\\n (size_t)(a) = ((size_t)(a) ^ (size_t)(b))\n')
# Getting the type of 'cppmacros' (line 215)
cppmacros_74336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 0), 'cppmacros')
str_74337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 10), 'str', 'SWAPUNSAFE')
# Storing an element on a container (line 215)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 0), cppmacros_74336, (str_74337, str_74335))

# Assigning a Str to a Subscript (line 220):
str_74338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, (-1)), 'str', '#define SWAP(a,b,t) {\\\n\tt *c;\\\n\tc = a;\\\n\ta = b;\\\n\tb = c;}\n')
# Getting the type of 'cppmacros' (line 220)
cppmacros_74339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 0), 'cppmacros')
str_74340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 10), 'str', 'SWAP')
# Storing an element on a container (line 220)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 0), cppmacros_74339, (str_74340, str_74338))

# Assigning a Str to a Subscript (line 229):
str_74341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, (-1)), 'str', '#define PRINTPYOBJERR(obj)\\\n\tfprintf(stderr,"#modulename#.error is related to ");\\\n\tPyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\\\n\tfprintf(stderr,"\\n");\n')
# Getting the type of 'cppmacros' (line 229)
cppmacros_74342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 0), 'cppmacros')
str_74343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 10), 'str', 'PRINTPYOBJERR')
# Storing an element on a container (line 229)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 0), cppmacros_74342, (str_74343, str_74341))

# Assigning a Str to a Subscript (line 235):
str_74344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, (-1)), 'str', '#ifndef max\n#define max(a,b) ((a > b) ? (a) : (b))\n#endif\n#ifndef min\n#define min(a,b) ((a < b) ? (a) : (b))\n#endif\n#ifndef MAX\n#define MAX(a,b) ((a > b) ? (a) : (b))\n#endif\n#ifndef MIN\n#define MIN(a,b) ((a < b) ? (a) : (b))\n#endif\n')
# Getting the type of 'cppmacros' (line 235)
cppmacros_74345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 0), 'cppmacros')
str_74346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 10), 'str', 'MINMAX')
# Storing an element on a container (line 235)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 0), cppmacros_74345, (str_74346, str_74344))

# Assigning a List to a Subscript (line 249):

# Obtaining an instance of the builtin type 'list' (line 249)
list_74347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 249)
# Adding element type (line 249)
str_74348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 18), 'str', 'f2py_size')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 17), list_74347, str_74348)

# Getting the type of 'needs' (line 249)
needs_74349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 0), 'needs')
str_74350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 6), 'str', 'len..')
# Storing an element on a container (line 249)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 0), needs_74349, (str_74350, list_74347))

# Assigning a Str to a Subscript (line 250):
str_74351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, (-1)), 'str', '#define rank(var) var ## _Rank\n#define shape(var,dim) var ## _Dims[dim]\n#define old_rank(var) (PyArray_NDIM((PyArrayObject *)(capi_ ## var ## _tmp)))\n#define old_shape(var,dim) PyArray_DIM(((PyArrayObject *)(capi_ ## var ## _tmp)),dim)\n#define fshape(var,dim) shape(var,rank(var)-dim-1)\n#define len(var) shape(var,0)\n#define flen(var) fshape(var,0)\n#define old_size(var) PyArray_SIZE((PyArrayObject *)(capi_ ## var ## _tmp))\n/* #define index(i) capi_i ## i */\n#define slen(var) capi_ ## var ## _len\n#define size(var, ...) f2py_size((PyArrayObject *)(capi_ ## var ## _tmp), ## __VA_ARGS__, -1)\n')
# Getting the type of 'cppmacros' (line 250)
cppmacros_74352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 0), 'cppmacros')
str_74353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 10), 'str', 'len..')
# Storing an element on a container (line 250)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 0), cppmacros_74352, (str_74353, str_74351))

# Assigning a List to a Subscript (line 263):

# Obtaining an instance of the builtin type 'list' (line 263)
list_74354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 263)
# Adding element type (line 263)
str_74355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 22), 'str', 'stdarg.h')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 21), list_74354, str_74355)

# Getting the type of 'needs' (line 263)
needs_74356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'needs')
str_74357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 6), 'str', 'f2py_size')
# Storing an element on a container (line 263)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 0), needs_74356, (str_74357, list_74354))

# Assigning a Str to a Subscript (line 264):
str_74358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, (-1)), 'str', 'static int f2py_size(PyArrayObject* var, ...)\n{\n  npy_int sz = 0;\n  npy_int dim;\n  npy_int rank;\n  va_list argp;\n  va_start(argp, var);\n  dim = va_arg(argp, npy_int);\n  if (dim==-1)\n    {\n      sz = PyArray_SIZE(var);\n    }\n  else\n    {\n      rank = PyArray_NDIM(var);\n      if (dim>=1 && dim<=rank)\n        sz = PyArray_DIM(var, dim-1);\n      else\n        fprintf(stderr, "f2py_size: 2nd argument value=%d fails to satisfy 1<=value<=%d. Result will be 0.\\n", dim, rank);\n    }\n  va_end(argp);\n  return sz;\n}\n')
# Getting the type of 'cfuncs' (line 264)
cfuncs_74359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 0), 'cfuncs')
str_74360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 7), 'str', 'f2py_size')
# Storing an element on a container (line 264)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 0), cfuncs_74359, (str_74360, str_74358))

# Assigning a Str to a Subscript (line 290):
str_74361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 26), 'str', '#define pyobj_from_char1(v) (PyInt_FromLong(v))')
# Getting the type of 'cppmacros' (line 290)
cppmacros_74362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 0), 'cppmacros')
str_74363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 4), 'str', 'pyobj_from_char1')
# Storing an element on a container (line 290)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 0), cppmacros_74362, (str_74363, str_74361))

# Assigning a Str to a Subscript (line 292):
str_74364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 27), 'str', '#define pyobj_from_short1(v) (PyInt_FromLong(v))')
# Getting the type of 'cppmacros' (line 292)
cppmacros_74365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 0), 'cppmacros')
str_74366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 4), 'str', 'pyobj_from_short1')
# Storing an element on a container (line 292)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 0), cppmacros_74365, (str_74366, str_74364))

# Assigning a List to a Subscript (line 294):

# Obtaining an instance of the builtin type 'list' (line 294)
list_74367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 294)
# Adding element type (line 294)
str_74368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 28), 'str', 'signed_char')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 27), list_74367, str_74368)

# Getting the type of 'needs' (line 294)
needs_74369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 0), 'needs')
str_74370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 6), 'str', 'pyobj_from_int1')
# Storing an element on a container (line 294)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 0), needs_74369, (str_74370, list_74367))

# Assigning a Str to a Subscript (line 295):
str_74371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 31), 'str', '#define pyobj_from_int1(v) (PyInt_FromLong(v))')
# Getting the type of 'cppmacros' (line 295)
cppmacros_74372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 0), 'cppmacros')
str_74373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 10), 'str', 'pyobj_from_int1')
# Storing an element on a container (line 295)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 0), cppmacros_74372, (str_74373, str_74371))

# Assigning a Str to a Subscript (line 296):
str_74374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 26), 'str', '#define pyobj_from_long1(v) (PyLong_FromLong(v))')
# Getting the type of 'cppmacros' (line 296)
cppmacros_74375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 0), 'cppmacros')
str_74376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 4), 'str', 'pyobj_from_long1')
# Storing an element on a container (line 296)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 0), cppmacros_74375, (str_74376, str_74374))

# Assigning a List to a Subscript (line 298):

# Obtaining an instance of the builtin type 'list' (line 298)
list_74377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 33), 'list')
# Adding type elements to the builtin type 'list' instance (line 298)
# Adding element type (line 298)
str_74378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 34), 'str', 'long_long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 33), list_74377, str_74378)

# Getting the type of 'needs' (line 298)
needs_74379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 0), 'needs')
str_74380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 6), 'str', 'pyobj_from_long_long1')
# Storing an element on a container (line 298)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 0), needs_74379, (str_74380, list_74377))

# Assigning a Str to a Subscript (line 299):
str_74381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, (-1)), 'str', '#ifdef HAVE_LONG_LONG\n#define pyobj_from_long_long1(v) (PyLong_FromLongLong(v))\n#else\n#warning HAVE_LONG_LONG is not available. Redefining pyobj_from_long_long.\n#define pyobj_from_long_long1(v) (PyLong_FromLong(v))\n#endif\n')
# Getting the type of 'cppmacros' (line 299)
cppmacros_74382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 0), 'cppmacros')
str_74383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 10), 'str', 'pyobj_from_long_long1')
# Storing an element on a container (line 299)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 0), cppmacros_74382, (str_74383, str_74381))

# Assigning a List to a Subscript (line 307):

# Obtaining an instance of the builtin type 'list' (line 307)
list_74384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 35), 'list')
# Adding type elements to the builtin type 'list' instance (line 307)
# Adding element type (line 307)
str_74385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 36), 'str', 'long_double')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 35), list_74384, str_74385)

# Getting the type of 'needs' (line 307)
needs_74386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 0), 'needs')
str_74387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 6), 'str', 'pyobj_from_long_double1')
# Storing an element on a container (line 307)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 0), needs_74386, (str_74387, list_74384))

# Assigning a Str to a Subscript (line 308):
str_74388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 33), 'str', '#define pyobj_from_long_double1(v) (PyFloat_FromDouble(v))')
# Getting the type of 'cppmacros' (line 308)
cppmacros_74389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 0), 'cppmacros')
str_74390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 4), 'str', 'pyobj_from_long_double1')
# Storing an element on a container (line 308)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 0), cppmacros_74389, (str_74390, str_74388))

# Assigning a Str to a Subscript (line 310):
str_74391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 28), 'str', '#define pyobj_from_double1(v) (PyFloat_FromDouble(v))')
# Getting the type of 'cppmacros' (line 310)
cppmacros_74392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 0), 'cppmacros')
str_74393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 4), 'str', 'pyobj_from_double1')
# Storing an element on a container (line 310)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 0), cppmacros_74392, (str_74393, str_74391))

# Assigning a Str to a Subscript (line 312):
str_74394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 27), 'str', '#define pyobj_from_float1(v) (PyFloat_FromDouble(v))')
# Getting the type of 'cppmacros' (line 312)
cppmacros_74395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 0), 'cppmacros')
str_74396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 4), 'str', 'pyobj_from_float1')
# Storing an element on a container (line 312)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 0), cppmacros_74395, (str_74396, str_74394))

# Assigning a List to a Subscript (line 314):

# Obtaining an instance of the builtin type 'list' (line 314)
list_74397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 43), 'list')
# Adding type elements to the builtin type 'list' instance (line 314)
# Adding element type (line 314)
str_74398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 44), 'str', 'complex_long_double')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 43), list_74397, str_74398)

# Getting the type of 'needs' (line 314)
needs_74399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 0), 'needs')
str_74400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 6), 'str', 'pyobj_from_complex_long_double1')
# Storing an element on a container (line 314)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 0), needs_74399, (str_74400, list_74397))

# Assigning a Str to a Subscript (line 315):
str_74401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 41), 'str', '#define pyobj_from_complex_long_double1(v) (PyComplex_FromDoubles(v.r,v.i))')
# Getting the type of 'cppmacros' (line 315)
cppmacros_74402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 0), 'cppmacros')
str_74403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 4), 'str', 'pyobj_from_complex_long_double1')
# Storing an element on a container (line 315)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 0), cppmacros_74402, (str_74403, str_74401))

# Assigning a List to a Subscript (line 317):

# Obtaining an instance of the builtin type 'list' (line 317)
list_74404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 38), 'list')
# Adding type elements to the builtin type 'list' instance (line 317)
# Adding element type (line 317)
str_74405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 39), 'str', 'complex_double')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 38), list_74404, str_74405)

# Getting the type of 'needs' (line 317)
needs_74406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 0), 'needs')
str_74407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 6), 'str', 'pyobj_from_complex_double1')
# Storing an element on a container (line 317)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 0), needs_74406, (str_74407, list_74404))

# Assigning a Str to a Subscript (line 318):
str_74408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 36), 'str', '#define pyobj_from_complex_double1(v) (PyComplex_FromDoubles(v.r,v.i))')
# Getting the type of 'cppmacros' (line 318)
cppmacros_74409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 0), 'cppmacros')
str_74410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 4), 'str', 'pyobj_from_complex_double1')
# Storing an element on a container (line 318)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 0), cppmacros_74409, (str_74410, str_74408))

# Assigning a List to a Subscript (line 320):

# Obtaining an instance of the builtin type 'list' (line 320)
list_74411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 37), 'list')
# Adding type elements to the builtin type 'list' instance (line 320)
# Adding element type (line 320)
str_74412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 38), 'str', 'complex_float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 37), list_74411, str_74412)

# Getting the type of 'needs' (line 320)
needs_74413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 0), 'needs')
str_74414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 6), 'str', 'pyobj_from_complex_float1')
# Storing an element on a container (line 320)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 0), needs_74413, (str_74414, list_74411))

# Assigning a Str to a Subscript (line 321):
str_74415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 35), 'str', '#define pyobj_from_complex_float1(v) (PyComplex_FromDoubles(v.r,v.i))')
# Getting the type of 'cppmacros' (line 321)
cppmacros_74416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 0), 'cppmacros')
str_74417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 4), 'str', 'pyobj_from_complex_float1')
# Storing an element on a container (line 321)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 0), cppmacros_74416, (str_74417, str_74415))

# Assigning a List to a Subscript (line 323):

# Obtaining an instance of the builtin type 'list' (line 323)
list_74418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 30), 'list')
# Adding type elements to the builtin type 'list' instance (line 323)
# Adding element type (line 323)
str_74419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 31), 'str', 'string')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 30), list_74418, str_74419)

# Getting the type of 'needs' (line 323)
needs_74420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 0), 'needs')
str_74421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 6), 'str', 'pyobj_from_string1')
# Storing an element on a container (line 323)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 0), needs_74420, (str_74421, list_74418))

# Assigning a Str to a Subscript (line 324):
str_74422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 28), 'str', '#define pyobj_from_string1(v) (PyString_FromString((char *)v))')
# Getting the type of 'cppmacros' (line 324)
cppmacros_74423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 0), 'cppmacros')
str_74424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 4), 'str', 'pyobj_from_string1')
# Storing an element on a container (line 324)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 0), cppmacros_74423, (str_74424, str_74422))

# Assigning a List to a Subscript (line 326):

# Obtaining an instance of the builtin type 'list' (line 326)
list_74425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 34), 'list')
# Adding type elements to the builtin type 'list' instance (line 326)
# Adding element type (line 326)
str_74426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 35), 'str', 'string')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 34), list_74425, str_74426)

# Getting the type of 'needs' (line 326)
needs_74427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 0), 'needs')
str_74428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 6), 'str', 'pyobj_from_string1size')
# Storing an element on a container (line 326)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 0), needs_74427, (str_74428, list_74425))

# Assigning a Str to a Subscript (line 327):
str_74429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 32), 'str', '#define pyobj_from_string1size(v,len) (PyUString_FromStringAndSize((char *)v, len))')
# Getting the type of 'cppmacros' (line 327)
cppmacros_74430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 0), 'cppmacros')
str_74431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 4), 'str', 'pyobj_from_string1size')
# Storing an element on a container (line 327)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 0), cppmacros_74430, (str_74431, str_74429))

# Assigning a List to a Subscript (line 329):

# Obtaining an instance of the builtin type 'list' (line 329)
list_74432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 30), 'list')
# Adding type elements to the builtin type 'list' instance (line 329)
# Adding element type (line 329)
str_74433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 31), 'str', 'PRINTPYOBJERR')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 30), list_74432, str_74433)

# Getting the type of 'needs' (line 329)
needs_74434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 0), 'needs')
str_74435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 6), 'str', 'TRYPYARRAYTEMPLATE')
# Storing an element on a container (line 329)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 0), needs_74434, (str_74435, list_74432))

# Assigning a Str to a Subscript (line 330):
str_74436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, (-1)), 'str', '/* New SciPy */\n#define TRYPYARRAYTEMPLATECHAR case NPY_STRING: *(char *)(PyArray_DATA(arr))=*v; break;\n#define TRYPYARRAYTEMPLATELONG case NPY_LONG: *(long *)(PyArray_DATA(arr))=*v; break;\n#define TRYPYARRAYTEMPLATEOBJECT case NPY_OBJECT: (PyArray_DESCR(arr)->f->setitem)(pyobj_from_ ## ctype ## 1(*v),PyArray_DATA(arr)); break;\n\n#define TRYPYARRAYTEMPLATE(ctype,typecode) \\\n        PyArrayObject *arr = NULL;\\\n        if (!obj) return -2;\\\n        if (!PyArray_Check(obj)) return -1;\\\n        if (!(arr=(PyArrayObject *)obj)) {fprintf(stderr,"TRYPYARRAYTEMPLATE:");PRINTPYOBJERR(obj);return 0;}\\\n        if (PyArray_DESCR(arr)->type==typecode)  {*(ctype *)(PyArray_DATA(arr))=*v; return 1;}\\\n        switch (PyArray_TYPE(arr)) {\\\n                case NPY_DOUBLE: *(double *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_INT: *(int *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_LONG: *(long *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_FLOAT: *(float *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_CDOUBLE: *(double *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_CFLOAT: *(float *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_BOOL: *(npy_bool *)(PyArray_DATA(arr))=(*v!=0); break;\\\n                case NPY_UBYTE: *(unsigned char *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_BYTE: *(signed char *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_SHORT: *(short *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_USHORT: *(npy_ushort *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_UINT: *(npy_uint *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_ULONG: *(npy_ulong *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_LONGLONG: *(npy_longlong *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_ULONGLONG: *(npy_ulonglong *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_LONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_CLONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=*v; break;\\\n                case NPY_OBJECT: (PyArray_DESCR(arr)->f->setitem)(pyobj_from_ ## ctype ## 1(*v),PyArray_DATA(arr), arr); break;\\\n        default: return -2;\\\n        };\\\n        return 1\n')
# Getting the type of 'cppmacros' (line 330)
cppmacros_74437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 0), 'cppmacros')
str_74438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 10), 'str', 'TRYPYARRAYTEMPLATE')
# Storing an element on a container (line 330)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 0), cppmacros_74437, (str_74438, str_74436))

# Assigning a List to a Subscript (line 366):

# Obtaining an instance of the builtin type 'list' (line 366)
list_74439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 37), 'list')
# Adding type elements to the builtin type 'list' instance (line 366)
# Adding element type (line 366)
str_74440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 38), 'str', 'PRINTPYOBJERR')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 37), list_74439, str_74440)

# Getting the type of 'needs' (line 366)
needs_74441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 0), 'needs')
str_74442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 6), 'str', 'TRYCOMPLEXPYARRAYTEMPLATE')
# Storing an element on a container (line 366)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 0), needs_74441, (str_74442, list_74439))

# Assigning a Str to a Subscript (line 367):
str_74443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, (-1)), 'str', '#define TRYCOMPLEXPYARRAYTEMPLATEOBJECT case NPY_OBJECT: (PyArray_DESCR(arr)->f->setitem)(pyobj_from_complex_ ## ctype ## 1((*v)),PyArray_DATA(arr), arr); break;\n#define TRYCOMPLEXPYARRAYTEMPLATE(ctype,typecode)\\\n        PyArrayObject *arr = NULL;\\\n        if (!obj) return -2;\\\n        if (!PyArray_Check(obj)) return -1;\\\n        if (!(arr=(PyArrayObject *)obj)) {fprintf(stderr,"TRYCOMPLEXPYARRAYTEMPLATE:");PRINTPYOBJERR(obj);return 0;}\\\n        if (PyArray_DESCR(arr)->type==typecode) {\\\n            *(ctype *)(PyArray_DATA(arr))=(*v).r;\\\n            *(ctype *)(PyArray_DATA(arr)+sizeof(ctype))=(*v).i;\\\n            return 1;\\\n        }\\\n        switch (PyArray_TYPE(arr)) {\\\n                case NPY_CDOUBLE: *(double *)(PyArray_DATA(arr))=(*v).r;*(double *)(PyArray_DATA(arr)+sizeof(double))=(*v).i;break;\\\n                case NPY_CFLOAT: *(float *)(PyArray_DATA(arr))=(*v).r;*(float *)(PyArray_DATA(arr)+sizeof(float))=(*v).i;break;\\\n                case NPY_DOUBLE: *(double *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_LONG: *(long *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_FLOAT: *(float *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_INT: *(int *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_SHORT: *(short *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_UBYTE: *(unsigned char *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_BYTE: *(signed char *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_BOOL: *(npy_bool *)(PyArray_DATA(arr))=((*v).r!=0 && (*v).i!=0); break;\\\n                case NPY_USHORT: *(npy_ushort *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_UINT: *(npy_uint *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_ULONG: *(npy_ulong *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_LONGLONG: *(npy_longlong *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_ULONGLONG: *(npy_ulonglong *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_LONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=(*v).r; break;\\\n                case NPY_CLONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=(*v).r;*(npy_longdouble *)(PyArray_DATA(arr)+sizeof(npy_longdouble))=(*v).i;break;\\\n                case NPY_OBJECT: (PyArray_DESCR(arr)->f->setitem)(pyobj_from_complex_ ## ctype ## 1((*v)),PyArray_DATA(arr), arr); break;\\\n                default: return -2;\\\n        };\\\n        return -1;\n')
# Getting the type of 'cppmacros' (line 367)
cppmacros_74444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 0), 'cppmacros')
str_74445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 10), 'str', 'TRYCOMPLEXPYARRAYTEMPLATE')
# Storing an element on a container (line 367)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 0), cppmacros_74444, (str_74445, str_74443))

# Assigning a List to a Subscript (line 435):

# Obtaining an instance of the builtin type 'list' (line 435)
list_74446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 29), 'list')
# Adding type elements to the builtin type 'list' instance (line 435)
# Adding element type (line 435)
str_74447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 30), 'str', 'STRINGCOPYN')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 29), list_74446, str_74447)
# Adding element type (line 435)
str_74448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 45), 'str', 'PRINTPYOBJERR')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 29), list_74446, str_74448)

# Getting the type of 'needs' (line 435)
needs_74449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 0), 'needs')
str_74450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 6), 'str', 'GETSTRFROMPYTUPLE')
# Storing an element on a container (line 435)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 0), needs_74449, (str_74450, list_74446))

# Assigning a Str to a Subscript (line 436):
str_74451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, (-1)), 'str', '#define GETSTRFROMPYTUPLE(tuple,index,str,len) {\\\n\t\tPyObject *rv_cb_str = PyTuple_GetItem((tuple),(index));\\\n\t\tif (rv_cb_str == NULL)\\\n\t\t\tgoto capi_fail;\\\n\t\tif (PyString_Check(rv_cb_str)) {\\\n\t\t\tstr[len-1]=\'\\0\';\\\n\t\t\tSTRINGCOPYN((str),PyString_AS_STRING((PyStringObject*)rv_cb_str),(len));\\\n\t\t} else {\\\n\t\t\tPRINTPYOBJERR(rv_cb_str);\\\n\t\t\tPyErr_SetString(#modulename#_error,"string object expected");\\\n\t\t\tgoto capi_fail;\\\n\t\t}\\\n\t}\n')
# Getting the type of 'cppmacros' (line 436)
cppmacros_74452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 0), 'cppmacros')
str_74453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 10), 'str', 'GETSTRFROMPYTUPLE')
# Storing an element on a container (line 436)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 0), cppmacros_74452, (str_74453, str_74451))

# Assigning a Str to a Subscript (line 451):
str_74454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, (-1)), 'str', '#define GETSCALARFROMPYTUPLE(tuple,index,var,ctype,mess) {\\\n\t\tif ((capi_tmp = PyTuple_GetItem((tuple),(index)))==NULL) goto capi_fail;\\\n\t\tif (!(ctype ## _from_pyobj((var),capi_tmp,mess)))\\\n\t\t\tgoto capi_fail;\\\n\t}\n')
# Getting the type of 'cppmacros' (line 451)
cppmacros_74455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 0), 'cppmacros')
str_74456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 10), 'str', 'GETSCALARFROMPYTUPLE')
# Storing an element on a container (line 451)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 0), cppmacros_74455, (str_74456, str_74454))

# Assigning a Str to a Subscript (line 459):
str_74457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, (-1)), 'str', '\\\n#define FAILNULL(p) do {                                            \\\n    if ((p) == NULL) {                                              \\\n        PyErr_SetString(PyExc_MemoryError, "NULL pointer found");   \\\n        goto capi_fail;                                             \\\n    }                                                               \\\n} while (0)\n')
# Getting the type of 'cppmacros' (line 459)
cppmacros_74458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 0), 'cppmacros')
str_74459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 10), 'str', 'FAILNULL')
# Storing an element on a container (line 459)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 0), cppmacros_74458, (str_74459, str_74457))

# Assigning a List to a Subscript (line 467):

# Obtaining an instance of the builtin type 'list' (line 467)
list_74460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 467)
# Adding element type (line 467)
str_74461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 20), 'str', 'string.h')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 19), list_74460, str_74461)
# Adding element type (line 467)
str_74462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 32), 'str', 'FAILNULL')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 19), list_74460, str_74462)

# Getting the type of 'needs' (line 467)
needs_74463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 0), 'needs')
str_74464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 6), 'str', 'MEMCOPY')
# Storing an element on a container (line 467)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 0), needs_74463, (str_74464, list_74460))

# Assigning a Str to a Subscript (line 468):
str_74465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, (-1)), 'str', '#define MEMCOPY(to,from,n)\\\n    do { FAILNULL(to); FAILNULL(from); (void)memcpy(to,from,n); } while (0)\n')
# Getting the type of 'cppmacros' (line 468)
cppmacros_74466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 0), 'cppmacros')
str_74467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 10), 'str', 'MEMCOPY')
# Storing an element on a container (line 468)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 0), cppmacros_74466, (str_74467, str_74465))

# Assigning a Str to a Subscript (line 472):
str_74468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, (-1)), 'str', '#define STRINGMALLOC(str,len)\\\n\tif ((str = (string)malloc(sizeof(char)*(len+1))) == NULL) {\\\n\t\tPyErr_SetString(PyExc_MemoryError, "out of memory");\\\n\t\tgoto capi_fail;\\\n\t} else {\\\n\t\t(str)[len] = \'\\0\';\\\n\t}\n')
# Getting the type of 'cppmacros' (line 472)
cppmacros_74469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 0), 'cppmacros')
str_74470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 10), 'str', 'STRINGMALLOC')
# Storing an element on a container (line 472)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 472, 0), cppmacros_74469, (str_74470, str_74468))

# Assigning a Str to a Subscript (line 481):
str_74471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, (-1)), 'str', '#define STRINGFREE(str) do {if (!(str == NULL)) free(str);} while (0)\n')
# Getting the type of 'cppmacros' (line 481)
cppmacros_74472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 0), 'cppmacros')
str_74473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 10), 'str', 'STRINGFREE')
# Storing an element on a container (line 481)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 481, 0), cppmacros_74472, (str_74473, str_74471))

# Assigning a List to a Subscript (line 484):

# Obtaining an instance of the builtin type 'list' (line 484)
list_74474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 484)
# Adding element type (line 484)
str_74475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 24), 'str', 'string.h')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 23), list_74474, str_74475)
# Adding element type (line 484)
str_74476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 36), 'str', 'FAILNULL')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 23), list_74474, str_74476)

# Getting the type of 'needs' (line 484)
needs_74477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 0), 'needs')
str_74478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 6), 'str', 'STRINGCOPYN')
# Storing an element on a container (line 484)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 0), needs_74477, (str_74478, list_74474))

# Assigning a Str to a Subscript (line 485):
str_74479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, (-1)), 'str', "#define STRINGCOPYN(to,from,buf_size)                           \\\n    do {                                                        \\\n        int _m = (buf_size);                                    \\\n        char *_to = (to);                                       \\\n        char *_from = (from);                                   \\\n        FAILNULL(_to); FAILNULL(_from);                         \\\n        (void)strncpy(_to, _from, sizeof(char)*_m);             \\\n        _to[_m-1] = '\\0';                                      \\\n        /* Padding with spaces instead of nulls */              \\\n        for (_m -= 2; _m >= 0 && _to[_m] == '\\0'; _m--) {      \\\n            _to[_m] = ' ';                                      \\\n        }                                                       \\\n    } while (0)\n")
# Getting the type of 'cppmacros' (line 485)
cppmacros_74480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 0), 'cppmacros')
str_74481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 10), 'str', 'STRINGCOPYN')
# Storing an element on a container (line 485)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 0), cppmacros_74480, (str_74481, str_74479))

# Assigning a List to a Subscript (line 500):

# Obtaining an instance of the builtin type 'list' (line 500)
list_74482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 500)
# Adding element type (line 500)
str_74483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 23), 'str', 'string.h')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 22), list_74482, str_74483)
# Adding element type (line 500)
str_74484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 35), 'str', 'FAILNULL')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 22), list_74482, str_74484)

# Getting the type of 'needs' (line 500)
needs_74485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 0), 'needs')
str_74486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 6), 'str', 'STRINGCOPY')
# Storing an element on a container (line 500)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 0), needs_74485, (str_74486, list_74482))

# Assigning a Str to a Subscript (line 501):
str_74487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, (-1)), 'str', '#define STRINGCOPY(to,from)\\\n    do { FAILNULL(to); FAILNULL(from); (void)strcpy(to,from); } while (0)\n')
# Getting the type of 'cppmacros' (line 501)
cppmacros_74488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 0), 'cppmacros')
str_74489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 10), 'str', 'STRINGCOPY')
# Storing an element on a container (line 501)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 0), cppmacros_74488, (str_74489, str_74487))

# Assigning a Str to a Subscript (line 505):
str_74490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, (-1)), 'str', '#define CHECKGENERIC(check,tcheck,name) \\\n\tif (!(check)) {\\\n\t\tPyErr_SetString(#modulename#_error,"("tcheck") failed for "name);\\\n\t\t/*goto capi_fail;*/\\\n\t} else ')
# Getting the type of 'cppmacros' (line 505)
cppmacros_74491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 0), 'cppmacros')
str_74492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 10), 'str', 'CHECKGENERIC')
# Storing an element on a container (line 505)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 0), cppmacros_74491, (str_74492, str_74490))

# Assigning a Str to a Subscript (line 511):
str_74493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, (-1)), 'str', '#define CHECKARRAY(check,tcheck,name) \\\n\tif (!(check)) {\\\n\t\tPyErr_SetString(#modulename#_error,"("tcheck") failed for "name);\\\n\t\t/*goto capi_fail;*/\\\n\t} else ')
# Getting the type of 'cppmacros' (line 511)
cppmacros_74494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 0), 'cppmacros')
str_74495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 10), 'str', 'CHECKARRAY')
# Storing an element on a container (line 511)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 511, 0), cppmacros_74494, (str_74495, str_74493))

# Assigning a Str to a Subscript (line 517):
str_74496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, (-1)), 'str', '#define CHECKSTRING(check,tcheck,name,show,var)\\\n\tif (!(check)) {\\\n\t\tchar errstring[256];\\\n\t\tsprintf(errstring, "%s: "show, "("tcheck") failed for "name, slen(var), var);\\\n\t\tPyErr_SetString(#modulename#_error, errstring);\\\n\t\t/*goto capi_fail;*/\\\n\t} else ')
# Getting the type of 'cppmacros' (line 517)
cppmacros_74497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 0), 'cppmacros')
str_74498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 10), 'str', 'CHECKSTRING')
# Storing an element on a container (line 517)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 0), cppmacros_74497, (str_74498, str_74496))

# Assigning a Str to a Subscript (line 525):
str_74499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, (-1)), 'str', '#define CHECKSCALAR(check,tcheck,name,show,var)\\\n\tif (!(check)) {\\\n\t\tchar errstring[256];\\\n\t\tsprintf(errstring, "%s: "show, "("tcheck") failed for "name, var);\\\n\t\tPyErr_SetString(#modulename#_error,errstring);\\\n\t\t/*goto capi_fail;*/\\\n\t} else ')
# Getting the type of 'cppmacros' (line 525)
cppmacros_74500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 0), 'cppmacros')
str_74501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 10), 'str', 'CHECKSCALAR')
# Storing an element on a container (line 525)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 525, 0), cppmacros_74500, (str_74501, str_74499))

# Assigning a Str to a Subscript (line 541):
str_74502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 17), 'str', '#define ARRSIZE(dims,rank) (_PyArray_multiply_list(dims,rank))')
# Getting the type of 'cppmacros' (line 541)
cppmacros_74503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 0), 'cppmacros')
str_74504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 4), 'str', 'ARRSIZE')
# Storing an element on a container (line 541)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 541, 0), cppmacros_74503, (str_74504, str_74502))

# Assigning a Str to a Subscript (line 543):
str_74505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, (-1)), 'str', '#ifdef OLDPYNUM\n#error You need to intall Numeric Python version 13 or higher. Get it from http:/sourceforge.net/project/?group_id=1369\n#endif\n')
# Getting the type of 'cppmacros' (line 543)
cppmacros_74506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 0), 'cppmacros')
str_74507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 10), 'str', 'OLDPYNUM')
# Storing an element on a container (line 543)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 543, 0), cppmacros_74506, (str_74507, str_74505))

# Assigning a Str to a Subscript (line 550):
str_74508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, (-1)), 'str', 'static int calcarrindex(int *i,PyArrayObject *arr) {\n\tint k,ii = i[0];\n\tfor (k=1; k < PyArray_NDIM(arr); k++)\n\t\tii += (ii*(PyArray_DIM(arr,k) - 1)+i[k]); /* assuming contiguous arr */\n\treturn ii;\n}')
# Getting the type of 'cfuncs' (line 550)
cfuncs_74509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 0), 'cfuncs')
str_74510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 7), 'str', 'calcarrindex')
# Storing an element on a container (line 550)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 0), cfuncs_74509, (str_74510, str_74508))

# Assigning a Str to a Subscript (line 557):
str_74511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, (-1)), 'str', 'static int calcarrindextr(int *i,PyArrayObject *arr) {\n\tint k,ii = i[PyArray_NDIM(arr)-1];\n\tfor (k=1; k < PyArray_NDIM(arr); k++)\n\t\tii += (ii*(PyArray_DIM(arr,PyArray_NDIM(arr)-k-1) - 1)+i[PyArray_NDIM(arr)-k-1]); /* assuming contiguous arr */\n\treturn ii;\n}')
# Getting the type of 'cfuncs' (line 557)
cfuncs_74512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 0), 'cfuncs')
str_74513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 7), 'str', 'calcarrindextr')
# Storing an element on a container (line 557)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 0), cfuncs_74512, (str_74513, str_74511))

# Assigning a Str to a Subscript (line 564):
str_74514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, (-1)), 'str', 'static struct { int nd;npy_intp *d;int *i,*i_tr,tr; } forcombcache;\nstatic int initforcomb(npy_intp *dims,int nd,int tr) {\n  int k;\n  if (dims==NULL) return 0;\n  if (nd<0) return 0;\n  forcombcache.nd = nd;\n  forcombcache.d = dims;\n  forcombcache.tr = tr;\n  if ((forcombcache.i = (int *)malloc(sizeof(int)*nd))==NULL) return 0;\n  if ((forcombcache.i_tr = (int *)malloc(sizeof(int)*nd))==NULL) return 0;\n  for (k=1;k<nd;k++) {\n    forcombcache.i[k] = forcombcache.i_tr[nd-k-1] = 0;\n  }\n  forcombcache.i[0] = forcombcache.i_tr[nd-1] = -1;\n  return 1;\n}\nstatic int *nextforcomb(void) {\n  int j,*i,*i_tr,k;\n  int nd=forcombcache.nd;\n  if ((i=forcombcache.i) == NULL) return NULL;\n  if ((i_tr=forcombcache.i_tr) == NULL) return NULL;\n  if (forcombcache.d == NULL) return NULL;\n  i[0]++;\n  if (i[0]==forcombcache.d[0]) {\n    j=1;\n    while ((j<nd) && (i[j]==forcombcache.d[j]-1)) j++;\n    if (j==nd) {\n      free(i);\n      free(i_tr);\n      return NULL;\n    }\n    for (k=0;k<j;k++) i[k] = i_tr[nd-k-1] = 0;\n    i[j]++;\n    i_tr[nd-j-1]++;\n  } else\n    i_tr[nd-1]++;\n  if (forcombcache.tr) return i_tr;\n  return i;\n}')
# Getting the type of 'cfuncs' (line 564)
cfuncs_74515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 0), 'cfuncs')
str_74516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 7), 'str', 'forcomb')
# Storing an element on a container (line 564)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 564, 0), cfuncs_74515, (str_74516, str_74514))

# Assigning a List to a Subscript (line 604):

# Obtaining an instance of the builtin type 'list' (line 604)
list_74517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 33), 'list')
# Adding type elements to the builtin type 'list' instance (line 604)
# Adding element type (line 604)
str_74518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 34), 'str', 'STRINGCOPYN')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 604, 33), list_74517, str_74518)
# Adding element type (line 604)
str_74519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 49), 'str', 'PRINTPYOBJERR')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 604, 33), list_74517, str_74519)
# Adding element type (line 604)
str_74520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 66), 'str', 'string')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 604, 33), list_74517, str_74520)

# Getting the type of 'needs' (line 604)
needs_74521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 0), 'needs')
str_74522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 6), 'str', 'try_pyarr_from_string')
# Storing an element on a container (line 604)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 604, 0), needs_74521, (str_74522, list_74517))

# Assigning a Str to a Subscript (line 605):
str_74523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, (-1)), 'str', 'static int try_pyarr_from_string(PyObject *obj,const string str) {\n\tPyArrayObject *arr = NULL;\n\tif (PyArray_Check(obj) && (!((arr = (PyArrayObject *)obj) == NULL)))\n\t\t{ STRINGCOPYN(PyArray_DATA(arr),str,PyArray_NBYTES(arr)); }\n\treturn 1;\ncapi_fail:\n\tPRINTPYOBJERR(obj);\n\tPyErr_SetString(#modulename#_error,"try_pyarr_from_string failed");\n\treturn 0;\n}\n')
# Getting the type of 'cfuncs' (line 605)
cfuncs_74524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 0), 'cfuncs')
str_74525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 7), 'str', 'try_pyarr_from_string')
# Storing an element on a container (line 605)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 0), cfuncs_74524, (str_74525, str_74523))

# Assigning a List to a Subscript (line 617):

# Obtaining an instance of the builtin type 'list' (line 617)
list_74526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 29), 'list')
# Adding type elements to the builtin type 'list' instance (line 617)
# Adding element type (line 617)
str_74527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 30), 'str', 'string')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 29), list_74526, str_74527)
# Adding element type (line 617)
str_74528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 40), 'str', 'STRINGMALLOC')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 29), list_74526, str_74528)
# Adding element type (line 617)
str_74529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 56), 'str', 'STRINGCOPYN')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 29), list_74526, str_74529)

# Getting the type of 'needs' (line 617)
needs_74530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 0), 'needs')
str_74531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 6), 'str', 'string_from_pyobj')
# Storing an element on a container (line 617)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 0), needs_74530, (str_74531, list_74526))

# Assigning a Str to a Subscript (line 618):
str_74532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, (-1)), 'str', 'static int string_from_pyobj(string *str,int *len,const string inistr,PyObject *obj,const char *errmess) {\n\tPyArrayObject *arr = NULL;\n\tPyObject *tmp = NULL;\n#ifdef DEBUGCFUNCS\nfprintf(stderr,"string_from_pyobj(str=\'%s\',len=%d,inistr=\'%s\',obj=%p)\\n",(char*)str,*len,(char *)inistr,obj);\n#endif\n\tif (obj == Py_None) {\n\t\tif (*len == -1)\n\t\t\t*len = strlen(inistr); /* Will this cause problems? */\n\t\tSTRINGMALLOC(*str,*len);\n\t\tSTRINGCOPYN(*str,inistr,*len+1);\n\t\treturn 1;\n\t}\n\tif (PyArray_Check(obj)) {\n\t\tif ((arr = (PyArrayObject *)obj) == NULL)\n\t\t\tgoto capi_fail;\n\t\tif (!ISCONTIGUOUS(arr)) {\n\t\t\tPyErr_SetString(PyExc_ValueError,"array object is non-contiguous.");\n\t\t\tgoto capi_fail;\n\t\t}\n\t\tif (*len == -1)\n\t\t\t*len = (PyArray_ITEMSIZE(arr))*PyArray_SIZE(arr);\n\t\tSTRINGMALLOC(*str,*len);\n\t\tSTRINGCOPYN(*str,PyArray_DATA(arr),*len+1);\n\t\treturn 1;\n\t}\n\tif (PyString_Check(obj)) {\n\t\ttmp = obj;\n\t\tPy_INCREF(tmp);\n\t}\n#if PY_VERSION_HEX >= 0x03000000\n\telse if (PyUnicode_Check(obj)) {\n\t\ttmp = PyUnicode_AsASCIIString(obj);\n\t}\n\telse {\n\t\tPyObject *tmp2;\n\t\ttmp2 = PyObject_Str(obj);\n\t\tif (tmp2) {\n\t\t\ttmp = PyUnicode_AsASCIIString(tmp2);\n\t\t\tPy_DECREF(tmp2);\n\t\t}\n\t\telse {\n\t\t\ttmp = NULL;\n\t\t}\n\t}\n#else\n\telse {\n\t\ttmp = PyObject_Str(obj);\n\t}\n#endif\n\tif (tmp == NULL) goto capi_fail;\n\tif (*len == -1)\n\t\t*len = PyString_GET_SIZE(tmp);\n\tSTRINGMALLOC(*str,*len);\n\tSTRINGCOPYN(*str,PyString_AS_STRING(tmp),*len+1);\n\tPy_DECREF(tmp);\n\treturn 1;\ncapi_fail:\n\tPy_XDECREF(tmp);\n\t{\n\t\tPyObject* err = PyErr_Occurred();\n\t\tif (err==NULL) err = #modulename#_error;\n\t\tPyErr_SetString(err,errmess);\n\t}\n\treturn 0;\n}\n')
# Getting the type of 'cfuncs' (line 618)
cfuncs_74533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 0), 'cfuncs')
str_74534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 7), 'str', 'string_from_pyobj')
# Storing an element on a container (line 618)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 618, 0), cfuncs_74533, (str_74534, str_74532))

# Assigning a List to a Subscript (line 686):

# Obtaining an instance of the builtin type 'list' (line 686)
list_74535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 686)
# Adding element type (line 686)
str_74536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 28), 'str', 'int_from_pyobj')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 686, 27), list_74535, str_74536)

# Getting the type of 'needs' (line 686)
needs_74537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 0), 'needs')
str_74538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 6), 'str', 'char_from_pyobj')
# Storing an element on a container (line 686)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 686, 0), needs_74537, (str_74538, list_74535))

# Assigning a Str to a Subscript (line 687):
str_74539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, (-1)), 'str', 'static int char_from_pyobj(char* v,PyObject *obj,const char *errmess) {\n\tint i=0;\n\tif (int_from_pyobj(&i,obj,errmess)) {\n\t\t*v = (char)i;\n\t\treturn 1;\n\t}\n\treturn 0;\n}\n')
# Getting the type of 'cfuncs' (line 687)
cfuncs_74540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 0), 'cfuncs')
str_74541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 7), 'str', 'char_from_pyobj')
# Storing an element on a container (line 687)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 687, 0), cfuncs_74540, (str_74541, str_74539))

# Assigning a List to a Subscript (line 697):

# Obtaining an instance of the builtin type 'list' (line 697)
list_74542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 34), 'list')
# Adding type elements to the builtin type 'list' instance (line 697)
# Adding element type (line 697)
str_74543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 35), 'str', 'int_from_pyobj')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 697, 34), list_74542, str_74543)
# Adding element type (line 697)
str_74544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 53), 'str', 'signed_char')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 697, 34), list_74542, str_74544)

# Getting the type of 'needs' (line 697)
needs_74545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 0), 'needs')
str_74546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 6), 'str', 'signed_char_from_pyobj')
# Storing an element on a container (line 697)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 697, 0), needs_74545, (str_74546, list_74542))

# Assigning a Str to a Subscript (line 698):
str_74547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, (-1)), 'str', 'static int signed_char_from_pyobj(signed_char* v,PyObject *obj,const char *errmess) {\n\tint i=0;\n\tif (int_from_pyobj(&i,obj,errmess)) {\n\t\t*v = (signed_char)i;\n\t\treturn 1;\n\t}\n\treturn 0;\n}\n')
# Getting the type of 'cfuncs' (line 698)
cfuncs_74548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 0), 'cfuncs')
str_74549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 7), 'str', 'signed_char_from_pyobj')
# Storing an element on a container (line 698)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 698, 0), cfuncs_74548, (str_74549, str_74547))

# Assigning a List to a Subscript (line 708):

# Obtaining an instance of the builtin type 'list' (line 708)
list_74550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 708)
# Adding element type (line 708)
str_74551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 29), 'str', 'int_from_pyobj')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 28), list_74550, str_74551)

# Getting the type of 'needs' (line 708)
needs_74552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 0), 'needs')
str_74553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 6), 'str', 'short_from_pyobj')
# Storing an element on a container (line 708)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 0), needs_74552, (str_74553, list_74550))

# Assigning a Str to a Subscript (line 709):
str_74554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, (-1)), 'str', 'static int short_from_pyobj(short* v,PyObject *obj,const char *errmess) {\n\tint i=0;\n\tif (int_from_pyobj(&i,obj,errmess)) {\n\t\t*v = (short)i;\n\t\treturn 1;\n\t}\n\treturn 0;\n}\n')
# Getting the type of 'cfuncs' (line 709)
cfuncs_74555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 0), 'cfuncs')
str_74556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 7), 'str', 'short_from_pyobj')
# Storing an element on a container (line 709)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 709, 0), cfuncs_74555, (str_74556, str_74554))

# Assigning a Str to a Subscript (line 719):
str_74557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, (-1)), 'str', 'static int int_from_pyobj(int* v,PyObject *obj,const char *errmess) {\n\tPyObject* tmp = NULL;\n\tif (PyInt_Check(obj)) {\n\t\t*v = (int)PyInt_AS_LONG(obj);\n\t\treturn 1;\n\t}\n\ttmp = PyNumber_Int(obj);\n\tif (tmp) {\n\t\t*v = PyInt_AS_LONG(tmp);\n\t\tPy_DECREF(tmp);\n\t\treturn 1;\n\t}\n\tif (PyComplex_Check(obj))\n\t\ttmp = PyObject_GetAttrString(obj,"real");\n\telse if (PyString_Check(obj) || PyUnicode_Check(obj))\n\t\t/*pass*/;\n\telse if (PySequence_Check(obj))\n\t\ttmp = PySequence_GetItem(obj,0);\n\tif (tmp) {\n\t\tPyErr_Clear();\n\t\tif (int_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}\n\t\tPy_DECREF(tmp);\n\t}\n\t{\n\t\tPyObject* err = PyErr_Occurred();\n\t\tif (err==NULL) err = #modulename#_error;\n\t\tPyErr_SetString(err,errmess);\n\t}\n\treturn 0;\n}\n')
# Getting the type of 'cfuncs' (line 719)
cfuncs_74558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 0), 'cfuncs')
str_74559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 7), 'str', 'int_from_pyobj')
# Storing an element on a container (line 719)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 719, 0), cfuncs_74558, (str_74559, str_74557))

# Assigning a Str to a Subscript (line 751):
str_74560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 782, (-1)), 'str', 'static int long_from_pyobj(long* v,PyObject *obj,const char *errmess) {\n\tPyObject* tmp = NULL;\n\tif (PyInt_Check(obj)) {\n\t\t*v = PyInt_AS_LONG(obj);\n\t\treturn 1;\n\t}\n\ttmp = PyNumber_Int(obj);\n\tif (tmp) {\n\t\t*v = PyInt_AS_LONG(tmp);\n\t\tPy_DECREF(tmp);\n\t\treturn 1;\n\t}\n\tif (PyComplex_Check(obj))\n\t\ttmp = PyObject_GetAttrString(obj,"real");\n\telse if (PyString_Check(obj) || PyUnicode_Check(obj))\n\t\t/*pass*/;\n\telse if (PySequence_Check(obj))\n\t\ttmp = PySequence_GetItem(obj,0);\n\tif (tmp) {\n\t\tPyErr_Clear();\n\t\tif (long_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}\n\t\tPy_DECREF(tmp);\n\t}\n\t{\n\t\tPyObject* err = PyErr_Occurred();\n\t\tif (err==NULL) err = #modulename#_error;\n\t\tPyErr_SetString(err,errmess);\n\t}\n\treturn 0;\n}\n')
# Getting the type of 'cfuncs' (line 751)
cfuncs_74561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 0), 'cfuncs')
str_74562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 7), 'str', 'long_from_pyobj')
# Storing an element on a container (line 751)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 751, 0), cfuncs_74561, (str_74562, str_74560))

# Assigning a List to a Subscript (line 783):

# Obtaining an instance of the builtin type 'list' (line 783)
list_74563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 32), 'list')
# Adding type elements to the builtin type 'list' instance (line 783)
# Adding element type (line 783)
str_74564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 33), 'str', 'long_long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 783, 32), list_74563, str_74564)

# Getting the type of 'needs' (line 783)
needs_74565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 0), 'needs')
str_74566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 6), 'str', 'long_long_from_pyobj')
# Storing an element on a container (line 783)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 783, 0), needs_74565, (str_74566, list_74563))

# Assigning a Str to a Subscript (line 784):
str_74567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, (-1)), 'str', 'static int long_long_from_pyobj(long_long* v,PyObject *obj,const char *errmess) {\n\tPyObject* tmp = NULL;\n\tif (PyLong_Check(obj)) {\n\t\t*v = PyLong_AsLongLong(obj);\n\t\treturn (!PyErr_Occurred());\n\t}\n\tif (PyInt_Check(obj)) {\n\t\t*v = (long_long)PyInt_AS_LONG(obj);\n\t\treturn 1;\n\t}\n\ttmp = PyNumber_Long(obj);\n\tif (tmp) {\n\t\t*v = PyLong_AsLongLong(tmp);\n\t\tPy_DECREF(tmp);\n\t\treturn (!PyErr_Occurred());\n\t}\n\tif (PyComplex_Check(obj))\n\t\ttmp = PyObject_GetAttrString(obj,"real");\n\telse if (PyString_Check(obj) || PyUnicode_Check(obj))\n\t\t/*pass*/;\n\telse if (PySequence_Check(obj))\n\t\ttmp = PySequence_GetItem(obj,0);\n\tif (tmp) {\n\t\tPyErr_Clear();\n\t\tif (long_long_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}\n\t\tPy_DECREF(tmp);\n\t}\n\t{\n\t\tPyObject* err = PyErr_Occurred();\n\t\tif (err==NULL) err = #modulename#_error;\n\t\tPyErr_SetString(err,errmess);\n\t}\n\treturn 0;\n}\n')
# Getting the type of 'cfuncs' (line 784)
cfuncs_74568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 0), 'cfuncs')
str_74569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 7), 'str', 'long_long_from_pyobj')
# Storing an element on a container (line 784)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 784, 0), cfuncs_74568, (str_74569, str_74567))

# Assigning a List to a Subscript (line 820):

# Obtaining an instance of the builtin type 'list' (line 820)
list_74570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, 34), 'list')
# Adding type elements to the builtin type 'list' instance (line 820)
# Adding element type (line 820)
str_74571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, 35), 'str', 'double_from_pyobj')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 820, 34), list_74570, str_74571)
# Adding element type (line 820)
str_74572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, 56), 'str', 'long_double')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 820, 34), list_74570, str_74572)

# Getting the type of 'needs' (line 820)
needs_74573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 0), 'needs')
str_74574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, 6), 'str', 'long_double_from_pyobj')
# Storing an element on a container (line 820)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 820, 0), needs_74573, (str_74574, list_74570))

# Assigning a Str to a Subscript (line 821):
str_74575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 840, (-1)), 'str', 'static int long_double_from_pyobj(long_double* v,PyObject *obj,const char *errmess) {\n\tdouble d=0;\n\tif (PyArray_CheckScalar(obj)){\n\t\tif PyArray_IsScalar(obj, LongDouble) {\n\t\t\tPyArray_ScalarAsCtype(obj, v);\n\t\t\treturn 1;\n\t\t}\n\t\telse if (PyArray_Check(obj) && PyArray_TYPE(obj)==NPY_LONGDOUBLE) {\n\t\t\t(*v) = *((npy_longdouble *)PyArray_DATA(obj));\n\t\t\treturn 1;\n\t\t}\n\t}\n\tif (double_from_pyobj(&d,obj,errmess)) {\n\t\t*v = (long_double)d;\n\t\treturn 1;\n\t}\n\treturn 0;\n}\n')
# Getting the type of 'cfuncs' (line 821)
cfuncs_74576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 0), 'cfuncs')
str_74577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 7), 'str', 'long_double_from_pyobj')
# Storing an element on a container (line 821)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 0), cfuncs_74576, (str_74577, str_74575))

# Assigning a Str to a Subscript (line 841):
str_74578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 880, (-1)), 'str', 'static int double_from_pyobj(double* v,PyObject *obj,const char *errmess) {\n\tPyObject* tmp = NULL;\n\tif (PyFloat_Check(obj)) {\n#ifdef __sgi\n\t\t*v = PyFloat_AsDouble(obj);\n#else\n\t\t*v = PyFloat_AS_DOUBLE(obj);\n#endif\n\t\treturn 1;\n\t}\n\ttmp = PyNumber_Float(obj);\n\tif (tmp) {\n#ifdef __sgi\n\t\t*v = PyFloat_AsDouble(tmp);\n#else\n\t\t*v = PyFloat_AS_DOUBLE(tmp);\n#endif\n\t\tPy_DECREF(tmp);\n\t\treturn 1;\n\t}\n\tif (PyComplex_Check(obj))\n\t\ttmp = PyObject_GetAttrString(obj,"real");\n\telse if (PyString_Check(obj) || PyUnicode_Check(obj))\n\t\t/*pass*/;\n\telse if (PySequence_Check(obj))\n\t\ttmp = PySequence_GetItem(obj,0);\n\tif (tmp) {\n\t\tPyErr_Clear();\n\t\tif (double_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}\n\t\tPy_DECREF(tmp);\n\t}\n\t{\n\t\tPyObject* err = PyErr_Occurred();\n\t\tif (err==NULL) err = #modulename#_error;\n\t\tPyErr_SetString(err,errmess);\n\t}\n\treturn 0;\n}\n')
# Getting the type of 'cfuncs' (line 841)
cfuncs_74579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 0), 'cfuncs')
str_74580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 841, 7), 'str', 'double_from_pyobj')
# Storing an element on a container (line 841)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 841, 0), cfuncs_74579, (str_74580, str_74578))

# Assigning a List to a Subscript (line 881):

# Obtaining an instance of the builtin type 'list' (line 881)
list_74581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 881, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 881)
# Adding element type (line 881)
str_74582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 881, 29), 'str', 'double_from_pyobj')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 881, 28), list_74581, str_74582)

# Getting the type of 'needs' (line 881)
needs_74583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 0), 'needs')
str_74584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 881, 6), 'str', 'float_from_pyobj')
# Storing an element on a container (line 881)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 881, 0), needs_74583, (str_74584, list_74581))

# Assigning a Str to a Subscript (line 882):
str_74585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 891, (-1)), 'str', 'static int float_from_pyobj(float* v,PyObject *obj,const char *errmess) {\n\tdouble d=0.0;\n\tif (double_from_pyobj(&d,obj,errmess)) {\n\t\t*v = (float)d;\n\t\treturn 1;\n\t}\n\treturn 0;\n}\n')
# Getting the type of 'cfuncs' (line 882)
cfuncs_74586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 0), 'cfuncs')
str_74587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 882, 7), 'str', 'float_from_pyobj')
# Storing an element on a container (line 882)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 882, 0), cfuncs_74586, (str_74587, str_74585))

# Assigning a List to a Subscript (line 892):

# Obtaining an instance of the builtin type 'list' (line 892)
list_74588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 892, 42), 'list')
# Adding type elements to the builtin type 'list' instance (line 892)
# Adding element type (line 892)
str_74589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 892, 43), 'str', 'complex_long_double')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 892, 42), list_74588, str_74589)
# Adding element type (line 892)
str_74590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 892, 66), 'str', 'long_double')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 892, 42), list_74588, str_74590)
# Adding element type (line 892)
str_74591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 43), 'str', 'complex_double_from_pyobj')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 892, 42), list_74588, str_74591)

# Getting the type of 'needs' (line 892)
needs_74592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 0), 'needs')
str_74593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 892, 6), 'str', 'complex_long_double_from_pyobj')
# Storing an element on a container (line 892)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 892, 0), needs_74592, (str_74593, list_74588))

# Assigning a Str to a Subscript (line 894):
str_74594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 915, (-1)), 'str', 'static int complex_long_double_from_pyobj(complex_long_double* v,PyObject *obj,const char *errmess) {\n\tcomplex_double cd={0.0,0.0};\n\tif (PyArray_CheckScalar(obj)){\n\t\tif PyArray_IsScalar(obj, CLongDouble) {\n\t\t\tPyArray_ScalarAsCtype(obj, v);\n\t\t\treturn 1;\n\t\t}\n\t\telse if (PyArray_Check(obj) && PyArray_TYPE(obj)==NPY_CLONGDOUBLE) {\n\t\t\t(*v).r = ((npy_clongdouble *)PyArray_DATA(obj))->real;\n\t\t\t(*v).i = ((npy_clongdouble *)PyArray_DATA(obj))->imag;\n\t\t\treturn 1;\n\t\t}\n\t}\n\tif (complex_double_from_pyobj(&cd,obj,errmess)) {\n\t\t(*v).r = (long_double)cd.r;\n\t\t(*v).i = (long_double)cd.i;\n\t\treturn 1;\n\t}\n\treturn 0;\n}\n')
# Getting the type of 'cfuncs' (line 894)
cfuncs_74595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 0), 'cfuncs')
str_74596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 894, 7), 'str', 'complex_long_double_from_pyobj')
# Storing an element on a container (line 894)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 894, 0), cfuncs_74595, (str_74596, str_74594))

# Assigning a List to a Subscript (line 916):

# Obtaining an instance of the builtin type 'list' (line 916)
list_74597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 916, 37), 'list')
# Adding type elements to the builtin type 'list' instance (line 916)
# Adding element type (line 916)
str_74598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 916, 38), 'str', 'complex_double')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 916, 37), list_74597, str_74598)

# Getting the type of 'needs' (line 916)
needs_74599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 0), 'needs')
str_74600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 916, 6), 'str', 'complex_double_from_pyobj')
# Storing an element on a container (line 916)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 916, 0), needs_74599, (str_74600, list_74597))

# Assigning a Str to a Subscript (line 917):
str_74601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 992, (-1)), 'str', 'static int complex_double_from_pyobj(complex_double* v,PyObject *obj,const char *errmess) {\n\tPy_complex c;\n\tif (PyComplex_Check(obj)) {\n\t\tc=PyComplex_AsCComplex(obj);\n\t\t(*v).r=c.real, (*v).i=c.imag;\n\t\treturn 1;\n\t}\n\tif (PyArray_IsScalar(obj, ComplexFloating)) {\n\t\tif (PyArray_IsScalar(obj, CFloat)) {\n\t\t\tnpy_cfloat new;\n\t\t\tPyArray_ScalarAsCtype(obj, &new);\n\t\t\t(*v).r = (double)new.real;\n\t\t\t(*v).i = (double)new.imag;\n\t\t}\n\t\telse if (PyArray_IsScalar(obj, CLongDouble)) {\n\t\t\tnpy_clongdouble new;\n\t\t\tPyArray_ScalarAsCtype(obj, &new);\n\t\t\t(*v).r = (double)new.real;\n\t\t\t(*v).i = (double)new.imag;\n\t\t}\n\t\telse { /* if (PyArray_IsScalar(obj, CDouble)) */\n\t\t\tPyArray_ScalarAsCtype(obj, v);\n\t\t}\n\t\treturn 1;\n\t}\n\tif (PyArray_CheckScalar(obj)) { /* 0-dim array or still array scalar */\n\t\tPyObject *arr;\n\t\tif (PyArray_Check(obj)) {\n\t\t\tarr = PyArray_Cast((PyArrayObject *)obj, NPY_CDOUBLE);\n\t\t}\n\t\telse {\n\t\t\tarr = PyArray_FromScalar(obj, PyArray_DescrFromType(NPY_CDOUBLE));\n\t\t}\n\t\tif (arr==NULL) return 0;\n\t\t(*v).r = ((npy_cdouble *)PyArray_DATA(arr))->real;\n\t\t(*v).i = ((npy_cdouble *)PyArray_DATA(arr))->imag;\n\t\treturn 1;\n\t}\n\t/* Python does not provide PyNumber_Complex function :-( */\n\t(*v).i=0.0;\n\tif (PyFloat_Check(obj)) {\n#ifdef __sgi\n\t\t(*v).r = PyFloat_AsDouble(obj);\n#else\n\t\t(*v).r = PyFloat_AS_DOUBLE(obj);\n#endif\n\t\treturn 1;\n\t}\n\tif (PyInt_Check(obj)) {\n\t\t(*v).r = (double)PyInt_AS_LONG(obj);\n\t\treturn 1;\n\t}\n\tif (PyLong_Check(obj)) {\n\t\t(*v).r = PyLong_AsDouble(obj);\n\t\treturn (!PyErr_Occurred());\n\t}\n\tif (PySequence_Check(obj) && !(PyString_Check(obj) || PyUnicode_Check(obj))) {\n\t\tPyObject *tmp = PySequence_GetItem(obj,0);\n\t\tif (tmp) {\n\t\t\tif (complex_double_from_pyobj(v,tmp,errmess)) {\n\t\t\t\tPy_DECREF(tmp);\n\t\t\t\treturn 1;\n\t\t\t}\n\t\t\tPy_DECREF(tmp);\n\t\t}\n\t}\n\t{\n\t\tPyObject* err = PyErr_Occurred();\n\t\tif (err==NULL)\n\t\t\terr = PyExc_TypeError;\n\t\tPyErr_SetString(err,errmess);\n\t}\n\treturn 0;\n}\n')
# Getting the type of 'cfuncs' (line 917)
cfuncs_74602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 0), 'cfuncs')
str_74603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 917, 7), 'str', 'complex_double_from_pyobj')
# Storing an element on a container (line 917)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 917, 0), cfuncs_74602, (str_74603, str_74601))

# Assigning a List to a Subscript (line 993):

# Obtaining an instance of the builtin type 'list' (line 993)
list_74604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 993, 36), 'list')
# Adding type elements to the builtin type 'list' instance (line 993)
# Adding element type (line 993)
str_74605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 994, 4), 'str', 'complex_float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 993, 36), list_74604, str_74605)
# Adding element type (line 993)
str_74606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 994, 21), 'str', 'complex_double_from_pyobj')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 993, 36), list_74604, str_74606)

# Getting the type of 'needs' (line 993)
needs_74607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 0), 'needs')
str_74608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 993, 6), 'str', 'complex_float_from_pyobj')
# Storing an element on a container (line 993)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 993, 0), needs_74607, (str_74608, list_74604))

# Assigning a Str to a Subscript (line 995):
str_74609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1005, (-1)), 'str', 'static int complex_float_from_pyobj(complex_float* v,PyObject *obj,const char *errmess) {\n\tcomplex_double cd={0.0,0.0};\n\tif (complex_double_from_pyobj(&cd,obj,errmess)) {\n\t\t(*v).r = (float)cd.r;\n\t\t(*v).i = (float)cd.i;\n\t\treturn 1;\n\t}\n\treturn 0;\n}\n')
# Getting the type of 'cfuncs' (line 995)
cfuncs_74610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 0), 'cfuncs')
str_74611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 995, 7), 'str', 'complex_float_from_pyobj')
# Storing an element on a container (line 995)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 995, 0), cfuncs_74610, (str_74611, str_74609))

# Assigning a List to a Subscript (line 1006):

# Obtaining an instance of the builtin type 'list' (line 1006)
list_74612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1006, 31), 'list')
# Adding type elements to the builtin type 'list' instance (line 1006)
# Adding element type (line 1006)
str_74613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1006, 32), 'str', 'pyobj_from_char1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1006, 31), list_74612, str_74613)
# Adding element type (line 1006)
str_74614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1006, 52), 'str', 'TRYPYARRAYTEMPLATE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1006, 31), list_74612, str_74614)

# Getting the type of 'needs' (line 1006)
needs_74615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1006, 0), 'needs')
str_74616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1006, 6), 'str', 'try_pyarr_from_char')
# Storing an element on a container (line 1006)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1006, 0), needs_74615, (str_74616, list_74612))

# Assigning a Str to a Subscript (line 1007):
str_74617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1008, 29), 'str', "static int try_pyarr_from_char(PyObject* obj,char* v) {\n\tTRYPYARRAYTEMPLATE(char,'c');\n}\n")
# Getting the type of 'cfuncs' (line 1007)
cfuncs_74618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 0), 'cfuncs')
str_74619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1008, 4), 'str', 'try_pyarr_from_char')
# Storing an element on a container (line 1007)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1007, 0), cfuncs_74618, (str_74619, str_74617))

# Assigning a List to a Subscript (line 1009):

# Obtaining an instance of the builtin type 'list' (line 1009)
list_74620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1009, 38), 'list')
# Adding type elements to the builtin type 'list' instance (line 1009)
# Adding element type (line 1009)
str_74621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1009, 39), 'str', 'TRYPYARRAYTEMPLATE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1009, 38), list_74620, str_74621)
# Adding element type (line 1009)
str_74622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1009, 61), 'str', 'unsigned_char')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1009, 38), list_74620, str_74622)

# Getting the type of 'needs' (line 1009)
needs_74623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 0), 'needs')
str_74624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1009, 6), 'str', 'try_pyarr_from_signed_char')
# Storing an element on a container (line 1009)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1009, 0), needs_74623, (str_74624, list_74620))

# Assigning a Str to a Subscript (line 1010):
str_74625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1011, 38), 'str', "static int try_pyarr_from_unsigned_char(PyObject* obj,unsigned_char* v) {\n\tTRYPYARRAYTEMPLATE(unsigned_char,'b');\n}\n")
# Getting the type of 'cfuncs' (line 1010)
cfuncs_74626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1010, 0), 'cfuncs')
str_74627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1011, 4), 'str', 'try_pyarr_from_unsigned_char')
# Storing an element on a container (line 1010)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1010, 0), cfuncs_74626, (str_74627, str_74625))

# Assigning a List to a Subscript (line 1012):

# Obtaining an instance of the builtin type 'list' (line 1012)
list_74628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1012, 38), 'list')
# Adding type elements to the builtin type 'list' instance (line 1012)
# Adding element type (line 1012)
str_74629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1012, 39), 'str', 'TRYPYARRAYTEMPLATE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1012, 38), list_74628, str_74629)
# Adding element type (line 1012)
str_74630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1012, 61), 'str', 'signed_char')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1012, 38), list_74628, str_74630)

# Getting the type of 'needs' (line 1012)
needs_74631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1012, 0), 'needs')
str_74632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1012, 6), 'str', 'try_pyarr_from_signed_char')
# Storing an element on a container (line 1012)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1012, 0), needs_74631, (str_74632, list_74628))

# Assigning a Str to a Subscript (line 1013):
str_74633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1014, 36), 'str', "static int try_pyarr_from_signed_char(PyObject* obj,signed_char* v) {\n\tTRYPYARRAYTEMPLATE(signed_char,'1');\n}\n")
# Getting the type of 'cfuncs' (line 1013)
cfuncs_74634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1013, 0), 'cfuncs')
str_74635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1014, 4), 'str', 'try_pyarr_from_signed_char')
# Storing an element on a container (line 1013)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1013, 0), cfuncs_74634, (str_74635, str_74633))

# Assigning a List to a Subscript (line 1015):

# Obtaining an instance of the builtin type 'list' (line 1015)
list_74636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1015, 32), 'list')
# Adding type elements to the builtin type 'list' instance (line 1015)
# Adding element type (line 1015)
str_74637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1015, 33), 'str', 'pyobj_from_short1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1015, 32), list_74636, str_74637)
# Adding element type (line 1015)
str_74638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1015, 54), 'str', 'TRYPYARRAYTEMPLATE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1015, 32), list_74636, str_74638)

# Getting the type of 'needs' (line 1015)
needs_74639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1015, 0), 'needs')
str_74640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1015, 6), 'str', 'try_pyarr_from_short')
# Storing an element on a container (line 1015)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1015, 0), needs_74639, (str_74640, list_74636))

# Assigning a Str to a Subscript (line 1016):
str_74641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1017, 30), 'str', "static int try_pyarr_from_short(PyObject* obj,short* v) {\n\tTRYPYARRAYTEMPLATE(short,'s');\n}\n")
# Getting the type of 'cfuncs' (line 1016)
cfuncs_74642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1016, 0), 'cfuncs')
str_74643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1017, 4), 'str', 'try_pyarr_from_short')
# Storing an element on a container (line 1016)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1016, 0), cfuncs_74642, (str_74643, str_74641))

# Assigning a List to a Subscript (line 1018):

# Obtaining an instance of the builtin type 'list' (line 1018)
list_74644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1018, 30), 'list')
# Adding type elements to the builtin type 'list' instance (line 1018)
# Adding element type (line 1018)
str_74645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1018, 31), 'str', 'pyobj_from_int1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1018, 30), list_74644, str_74645)
# Adding element type (line 1018)
str_74646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1018, 50), 'str', 'TRYPYARRAYTEMPLATE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1018, 30), list_74644, str_74646)

# Getting the type of 'needs' (line 1018)
needs_74647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1018, 0), 'needs')
str_74648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1018, 6), 'str', 'try_pyarr_from_int')
# Storing an element on a container (line 1018)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1018, 0), needs_74647, (str_74648, list_74644))

# Assigning a Str to a Subscript (line 1019):
str_74649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1020, 28), 'str', "static int try_pyarr_from_int(PyObject* obj,int* v) {\n\tTRYPYARRAYTEMPLATE(int,'i');\n}\n")
# Getting the type of 'cfuncs' (line 1019)
cfuncs_74650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1019, 0), 'cfuncs')
str_74651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1020, 4), 'str', 'try_pyarr_from_int')
# Storing an element on a container (line 1019)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1019, 0), cfuncs_74650, (str_74651, str_74649))

# Assigning a List to a Subscript (line 1021):

# Obtaining an instance of the builtin type 'list' (line 1021)
list_74652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1021, 31), 'list')
# Adding type elements to the builtin type 'list' instance (line 1021)
# Adding element type (line 1021)
str_74653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1021, 32), 'str', 'pyobj_from_long1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 31), list_74652, str_74653)
# Adding element type (line 1021)
str_74654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1021, 52), 'str', 'TRYPYARRAYTEMPLATE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 31), list_74652, str_74654)

# Getting the type of 'needs' (line 1021)
needs_74655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1021, 0), 'needs')
str_74656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1021, 6), 'str', 'try_pyarr_from_long')
# Storing an element on a container (line 1021)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 0), needs_74655, (str_74656, list_74652))

# Assigning a Str to a Subscript (line 1022):
str_74657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1023, 29), 'str', "static int try_pyarr_from_long(PyObject* obj,long* v) {\n\tTRYPYARRAYTEMPLATE(long,'l');\n}\n")
# Getting the type of 'cfuncs' (line 1022)
cfuncs_74658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1022, 0), 'cfuncs')
str_74659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1023, 4), 'str', 'try_pyarr_from_long')
# Storing an element on a container (line 1022)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1022, 0), cfuncs_74658, (str_74659, str_74657))

# Assigning a List to a Subscript (line 1024):

# Obtaining an instance of the builtin type 'list' (line 1024)
list_74660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1024, 36), 'list')
# Adding type elements to the builtin type 'list' instance (line 1024)
# Adding element type (line 1024)
str_74661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1025, 4), 'str', 'pyobj_from_long_long1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1024, 36), list_74660, str_74661)
# Adding element type (line 1024)
str_74662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1025, 29), 'str', 'TRYPYARRAYTEMPLATE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1024, 36), list_74660, str_74662)
# Adding element type (line 1024)
str_74663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1025, 51), 'str', 'long_long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1024, 36), list_74660, str_74663)

# Getting the type of 'needs' (line 1024)
needs_74664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1024, 0), 'needs')
str_74665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1024, 6), 'str', 'try_pyarr_from_long_long')
# Storing an element on a container (line 1024)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1024, 0), needs_74664, (str_74665, list_74660))

# Assigning a Str to a Subscript (line 1026):
str_74666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1027, 34), 'str', "static int try_pyarr_from_long_long(PyObject* obj,long_long* v) {\n\tTRYPYARRAYTEMPLATE(long_long,'L');\n}\n")
# Getting the type of 'cfuncs' (line 1026)
cfuncs_74667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1026, 0), 'cfuncs')
str_74668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1027, 4), 'str', 'try_pyarr_from_long_long')
# Storing an element on a container (line 1026)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1026, 0), cfuncs_74667, (str_74668, str_74666))

# Assigning a List to a Subscript (line 1028):

# Obtaining an instance of the builtin type 'list' (line 1028)
list_74669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1028, 32), 'list')
# Adding type elements to the builtin type 'list' instance (line 1028)
# Adding element type (line 1028)
str_74670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1028, 33), 'str', 'pyobj_from_float1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1028, 32), list_74669, str_74670)
# Adding element type (line 1028)
str_74671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1028, 54), 'str', 'TRYPYARRAYTEMPLATE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1028, 32), list_74669, str_74671)

# Getting the type of 'needs' (line 1028)
needs_74672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 0), 'needs')
str_74673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1028, 6), 'str', 'try_pyarr_from_float')
# Storing an element on a container (line 1028)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1028, 0), needs_74672, (str_74673, list_74669))

# Assigning a Str to a Subscript (line 1029):
str_74674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, 30), 'str', "static int try_pyarr_from_float(PyObject* obj,float* v) {\n\tTRYPYARRAYTEMPLATE(float,'f');\n}\n")
# Getting the type of 'cfuncs' (line 1029)
cfuncs_74675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1029, 0), 'cfuncs')
str_74676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, 4), 'str', 'try_pyarr_from_float')
# Storing an element on a container (line 1029)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1029, 0), cfuncs_74675, (str_74676, str_74674))

# Assigning a List to a Subscript (line 1031):

# Obtaining an instance of the builtin type 'list' (line 1031)
list_74677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1031, 33), 'list')
# Adding type elements to the builtin type 'list' instance (line 1031)
# Adding element type (line 1031)
str_74678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1031, 34), 'str', 'pyobj_from_double1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1031, 33), list_74677, str_74678)
# Adding element type (line 1031)
str_74679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1031, 56), 'str', 'TRYPYARRAYTEMPLATE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1031, 33), list_74677, str_74679)

# Getting the type of 'needs' (line 1031)
needs_74680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 0), 'needs')
str_74681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1031, 6), 'str', 'try_pyarr_from_double')
# Storing an element on a container (line 1031)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1031, 0), needs_74680, (str_74681, list_74677))

# Assigning a Str to a Subscript (line 1032):
str_74682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1033, 31), 'str', "static int try_pyarr_from_double(PyObject* obj,double* v) {\n\tTRYPYARRAYTEMPLATE(double,'d');\n}\n")
# Getting the type of 'cfuncs' (line 1032)
cfuncs_74683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1032, 0), 'cfuncs')
str_74684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1033, 4), 'str', 'try_pyarr_from_double')
# Storing an element on a container (line 1032)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1032, 0), cfuncs_74683, (str_74684, str_74682))

# Assigning a List to a Subscript (line 1034):

# Obtaining an instance of the builtin type 'list' (line 1034)
list_74685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1034, 40), 'list')
# Adding type elements to the builtin type 'list' instance (line 1034)
# Adding element type (line 1034)
str_74686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1035, 4), 'str', 'pyobj_from_complex_float1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1034, 40), list_74685, str_74686)
# Adding element type (line 1034)
str_74687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1035, 33), 'str', 'TRYCOMPLEXPYARRAYTEMPLATE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1034, 40), list_74685, str_74687)
# Adding element type (line 1034)
str_74688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1035, 62), 'str', 'complex_float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1034, 40), list_74685, str_74688)

# Getting the type of 'needs' (line 1034)
needs_74689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 0), 'needs')
str_74690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1034, 6), 'str', 'try_pyarr_from_complex_float')
# Storing an element on a container (line 1034)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1034, 0), needs_74689, (str_74690, list_74685))

# Assigning a Str to a Subscript (line 1036):
str_74691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1037, 38), 'str', "static int try_pyarr_from_complex_float(PyObject* obj,complex_float* v) {\n\tTRYCOMPLEXPYARRAYTEMPLATE(float,'F');\n}\n")
# Getting the type of 'cfuncs' (line 1036)
cfuncs_74692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 0), 'cfuncs')
str_74693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1037, 4), 'str', 'try_pyarr_from_complex_float')
# Storing an element on a container (line 1036)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1036, 0), cfuncs_74692, (str_74693, str_74691))

# Assigning a List to a Subscript (line 1038):

# Obtaining an instance of the builtin type 'list' (line 1038)
list_74694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1038, 41), 'list')
# Adding type elements to the builtin type 'list' instance (line 1038)
# Adding element type (line 1038)
str_74695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1039, 4), 'str', 'pyobj_from_complex_double1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1038, 41), list_74694, str_74695)
# Adding element type (line 1038)
str_74696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1039, 34), 'str', 'TRYCOMPLEXPYARRAYTEMPLATE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1038, 41), list_74694, str_74696)
# Adding element type (line 1038)
str_74697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1039, 63), 'str', 'complex_double')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1038, 41), list_74694, str_74697)

# Getting the type of 'needs' (line 1038)
needs_74698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 0), 'needs')
str_74699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1038, 6), 'str', 'try_pyarr_from_complex_double')
# Storing an element on a container (line 1038)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1038, 0), needs_74698, (str_74699, list_74694))

# Assigning a Str to a Subscript (line 1040):
str_74700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1041, 39), 'str', "static int try_pyarr_from_complex_double(PyObject* obj,complex_double* v) {\n\tTRYCOMPLEXPYARRAYTEMPLATE(double,'D');\n}\n")
# Getting the type of 'cfuncs' (line 1040)
cfuncs_74701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 0), 'cfuncs')
str_74702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1041, 4), 'str', 'try_pyarr_from_complex_double')
# Storing an element on a container (line 1040)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1040, 0), cfuncs_74701, (str_74702, str_74700))

# Assigning a List to a Subscript (line 1043):

# Obtaining an instance of the builtin type 'list' (line 1043)
list_74703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1043, 29), 'list')
# Adding type elements to the builtin type 'list' instance (line 1043)
# Adding element type (line 1043)
str_74704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1043, 30), 'str', 'CFUNCSMESS')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1043, 29), list_74703, str_74704)
# Adding element type (line 1043)
str_74705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1043, 44), 'str', 'PRINTPYOBJERR')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1043, 29), list_74703, str_74705)
# Adding element type (line 1043)
str_74706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1043, 61), 'str', 'MINMAX')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1043, 29), list_74703, str_74706)

# Getting the type of 'needs' (line 1043)
needs_74707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 0), 'needs')
str_74708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1043, 6), 'str', 'create_cb_arglist')
# Storing an element on a container (line 1043)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1043, 0), needs_74707, (str_74708, list_74703))

# Assigning a Str to a Subscript (line 1044):
str_74709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1145, (-1)), 'str', 'static int create_cb_arglist(PyObject* fun,PyTupleObject* xa,const int maxnofargs,const int nofoptargs,int *nofargs,PyTupleObject **args,const char *errmess) {\n\tPyObject *tmp = NULL;\n\tPyObject *tmp_fun = NULL;\n\tint tot,opt,ext,siz,i,di=0;\n\tCFUNCSMESS("create_cb_arglist\\n");\n\ttot=opt=ext=siz=0;\n\t/* Get the total number of arguments */\n\tif (PyFunction_Check(fun))\n\t\ttmp_fun = fun;\n\telse {\n\t\tdi = 1;\n\t\tif (PyObject_HasAttrString(fun,"im_func")) {\n\t\t\ttmp_fun = PyObject_GetAttrString(fun,"im_func");\n\t\t}\n\t\telse if (PyObject_HasAttrString(fun,"__call__")) {\n\t\t\ttmp = PyObject_GetAttrString(fun,"__call__");\n\t\t\tif (PyObject_HasAttrString(tmp,"im_func"))\n\t\t\t\ttmp_fun = PyObject_GetAttrString(tmp,"im_func");\n\t\t\telse {\n\t\t\t\ttmp_fun = fun; /* built-in function */\n\t\t\t\ttot = maxnofargs;\n\t\t\t\tif (xa != NULL)\n\t\t\t\t\ttot += PyTuple_Size((PyObject *)xa);\n\t\t\t}\n\t\t\tPy_XDECREF(tmp);\n\t\t}\n\t\telse if (PyFortran_Check(fun) || PyFortran_Check1(fun)) {\n\t\t\ttot = maxnofargs;\n\t\t\tif (xa != NULL)\n\t\t\t\ttot += PyTuple_Size((PyObject *)xa);\n\t\t\ttmp_fun = fun;\n\t\t}\n\t\telse if (F2PyCapsule_Check(fun)) {\n\t\t\ttot = maxnofargs;\n\t\t\tif (xa != NULL)\n\t\t\t\text = PyTuple_Size((PyObject *)xa);\n\t\t\tif(ext>0) {\n\t\t\t\tfprintf(stderr,"extra arguments tuple cannot be used with CObject call-back\\n");\n\t\t\t\tgoto capi_fail;\n\t\t\t}\n\t\t\ttmp_fun = fun;\n\t\t}\n\t}\nif (tmp_fun==NULL) {\nfprintf(stderr,"Call-back argument must be function|instance|instance.__call__|f2py-function but got %s.\\n",(fun==NULL?"NULL":Py_TYPE(fun)->tp_name));\ngoto capi_fail;\n}\n#if PY_VERSION_HEX >= 0x03000000\n\tif (PyObject_HasAttrString(tmp_fun,"__code__")) {\n\t\tif (PyObject_HasAttrString(tmp = PyObject_GetAttrString(tmp_fun,"__code__"),"co_argcount"))\n#else\n\tif (PyObject_HasAttrString(tmp_fun,"func_code")) {\n\t\tif (PyObject_HasAttrString(tmp = PyObject_GetAttrString(tmp_fun,"func_code"),"co_argcount"))\n#endif\n\t\t\ttot = PyInt_AsLong(PyObject_GetAttrString(tmp,"co_argcount")) - di;\n\t\tPy_XDECREF(tmp);\n\t}\n\t/* Get the number of optional arguments */\n#if PY_VERSION_HEX >= 0x03000000\n\tif (PyObject_HasAttrString(tmp_fun,"__defaults__")) {\n\t\tif (PyTuple_Check(tmp = PyObject_GetAttrString(tmp_fun,"__defaults__")))\n#else\n\tif (PyObject_HasAttrString(tmp_fun,"func_defaults")) {\n\t\tif (PyTuple_Check(tmp = PyObject_GetAttrString(tmp_fun,"func_defaults")))\n#endif\n\t\t\topt = PyTuple_Size(tmp);\n\t\tPy_XDECREF(tmp);\n\t}\n\t/* Get the number of extra arguments */\n\tif (xa != NULL)\n\t\text = PyTuple_Size((PyObject *)xa);\n\t/* Calculate the size of call-backs argument list */\n\tsiz = MIN(maxnofargs+ext,tot);\n\t*nofargs = MAX(0,siz-ext);\n#ifdef DEBUGCFUNCS\n\tfprintf(stderr,"debug-capi:create_cb_arglist:maxnofargs(-nofoptargs),tot,opt,ext,siz,nofargs=%d(-%d),%d,%d,%d,%d,%d\\n",maxnofargs,nofoptargs,tot,opt,ext,siz,*nofargs);\n#endif\n\tif (siz<tot-opt) {\n\t\tfprintf(stderr,"create_cb_arglist: Failed to build argument list (siz) with enough arguments (tot-opt) required by user-supplied function (siz,tot,opt=%d,%d,%d).\\n",siz,tot,opt);\n\t\tgoto capi_fail;\n\t}\n\t/* Initialize argument list */\n\t*args = (PyTupleObject *)PyTuple_New(siz);\n\tfor (i=0;i<*nofargs;i++) {\n\t\tPy_INCREF(Py_None);\n\t\tPyTuple_SET_ITEM((PyObject *)(*args),i,Py_None);\n\t}\n\tif (xa != NULL)\n\t\tfor (i=(*nofargs);i<siz;i++) {\n\t\t\ttmp = PyTuple_GetItem((PyObject *)xa,i-(*nofargs));\n\t\t\tPy_INCREF(tmp);\n\t\t\tPyTuple_SET_ITEM(*args,i,tmp);\n\t\t}\n\tCFUNCSMESS("create_cb_arglist-end\\n");\n\treturn 1;\ncapi_fail:\n\tif ((PyErr_Occurred())==NULL)\n\t\tPyErr_SetString(#modulename#_error,errmess);\n\treturn 0;\n}\n')
# Getting the type of 'cfuncs' (line 1044)
cfuncs_74710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 0), 'cfuncs')
str_74711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1044, 7), 'str', 'create_cb_arglist')
# Storing an element on a container (line 1044)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1044, 0), cfuncs_74710, (str_74711, str_74709))

@norecursion
def buildcfuncs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'buildcfuncs'
    module_type_store = module_type_store.open_function_context('buildcfuncs', 1148, 0, False)
    
    # Passed parameters checking function
    buildcfuncs.stypy_localization = localization
    buildcfuncs.stypy_type_of_self = None
    buildcfuncs.stypy_type_store = module_type_store
    buildcfuncs.stypy_function_name = 'buildcfuncs'
    buildcfuncs.stypy_param_names_list = []
    buildcfuncs.stypy_varargs_param_name = None
    buildcfuncs.stypy_kwargs_param_name = None
    buildcfuncs.stypy_call_defaults = defaults
    buildcfuncs.stypy_call_varargs = varargs
    buildcfuncs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'buildcfuncs', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'buildcfuncs', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'buildcfuncs(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1149, 4))
    
    # 'from numpy.f2py.capi_maps import c2capi_map' statement (line 1149)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
    import_74712 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1149, 4), 'numpy.f2py.capi_maps')

    if (type(import_74712) is not StypyTypeError):

        if (import_74712 != 'pyd_module'):
            __import__(import_74712)
            sys_modules_74713 = sys.modules[import_74712]
            import_from_module(stypy.reporting.localization.Localization(__file__, 1149, 4), 'numpy.f2py.capi_maps', sys_modules_74713.module_type_store, module_type_store, ['c2capi_map'])
            nest_module(stypy.reporting.localization.Localization(__file__, 1149, 4), __file__, sys_modules_74713, sys_modules_74713.module_type_store, module_type_store)
        else:
            from numpy.f2py.capi_maps import c2capi_map

            import_from_module(stypy.reporting.localization.Localization(__file__, 1149, 4), 'numpy.f2py.capi_maps', None, module_type_store, ['c2capi_map'], [c2capi_map])

    else:
        # Assigning a type to the variable 'numpy.f2py.capi_maps' (line 1149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1149, 4), 'numpy.f2py.capi_maps', import_74712)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')
    
    
    
    # Call to keys(...): (line 1150)
    # Processing the call keyword arguments (line 1150)
    kwargs_74716 = {}
    # Getting the type of 'c2capi_map' (line 1150)
    c2capi_map_74714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1150, 13), 'c2capi_map', False)
    # Obtaining the member 'keys' of a type (line 1150)
    keys_74715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1150, 13), c2capi_map_74714, 'keys')
    # Calling keys(args, kwargs) (line 1150)
    keys_call_result_74717 = invoke(stypy.reporting.localization.Localization(__file__, 1150, 13), keys_74715, *[], **kwargs_74716)
    
    # Testing the type of a for loop iterable (line 1150)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1150, 4), keys_call_result_74717)
    # Getting the type of the for loop variable (line 1150)
    for_loop_var_74718 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1150, 4), keys_call_result_74717)
    # Assigning a type to the variable 'k' (line 1150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1150, 4), 'k', for_loop_var_74718)
    # SSA begins for a for statement (line 1150)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 1151):
    str_74719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1151, 12), 'str', 'pyarr_from_p_%s1')
    # Getting the type of 'k' (line 1151)
    k_74720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1151, 33), 'k')
    # Applying the binary operator '%' (line 1151)
    result_mod_74721 = python_operator(stypy.reporting.localization.Localization(__file__, 1151, 12), '%', str_74719, k_74720)
    
    # Assigning a type to the variable 'm' (line 1151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1151, 8), 'm', result_mod_74721)
    
    # Assigning a BinOp to a Subscript (line 1152):
    str_74722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1153, 17), 'str', '#define %s(v) (PyArray_SimpleNewFromData(0,NULL,%s,(char *)v))')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1153)
    tuple_74723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1153, 85), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1153)
    # Adding element type (line 1153)
    # Getting the type of 'm' (line 1153)
    m_74724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 85), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1153, 85), tuple_74723, m_74724)
    # Adding element type (line 1153)
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 1153)
    k_74725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 99), 'k')
    # Getting the type of 'c2capi_map' (line 1153)
    c2capi_map_74726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 88), 'c2capi_map')
    # Obtaining the member '__getitem__' of a type (line 1153)
    getitem___74727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1153, 88), c2capi_map_74726, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1153)
    subscript_call_result_74728 = invoke(stypy.reporting.localization.Localization(__file__, 1153, 88), getitem___74727, k_74725)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1153, 85), tuple_74723, subscript_call_result_74728)
    
    # Applying the binary operator '%' (line 1153)
    result_mod_74729 = python_operator(stypy.reporting.localization.Localization(__file__, 1153, 17), '%', str_74722, tuple_74723)
    
    # Getting the type of 'cppmacros' (line 1152)
    cppmacros_74730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1152, 8), 'cppmacros')
    # Getting the type of 'm' (line 1153)
    m_74731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 12), 'm')
    # Storing an element on a container (line 1152)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1152, 8), cppmacros_74730, (m_74731, result_mod_74729))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Str to a Name (line 1154):
    str_74732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1154, 8), 'str', 'string')
    # Assigning a type to the variable 'k' (line 1154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1154, 4), 'k', str_74732)
    
    # Assigning a BinOp to a Name (line 1155):
    str_74733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1155, 8), 'str', 'pyarr_from_p_%s1')
    # Getting the type of 'k' (line 1155)
    k_74734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 29), 'k')
    # Applying the binary operator '%' (line 1155)
    result_mod_74735 = python_operator(stypy.reporting.localization.Localization(__file__, 1155, 8), '%', str_74733, k_74734)
    
    # Assigning a type to the variable 'm' (line 1155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1155, 4), 'm', result_mod_74735)
    
    # Assigning a BinOp to a Subscript (line 1156):
    str_74736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1157, 13), 'str', '#define %s(v,dims) (PyArray_SimpleNewFromData(1,dims,NPY_CHAR,(char *)v))')
    # Getting the type of 'm' (line 1157)
    m_74737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1157, 92), 'm')
    # Applying the binary operator '%' (line 1157)
    result_mod_74738 = python_operator(stypy.reporting.localization.Localization(__file__, 1157, 13), '%', str_74736, m_74737)
    
    # Getting the type of 'cppmacros' (line 1156)
    cppmacros_74739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 4), 'cppmacros')
    # Getting the type of 'm' (line 1157)
    m_74740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1157, 8), 'm')
    # Storing an element on a container (line 1156)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1156, 4), cppmacros_74739, (m_74740, result_mod_74738))
    
    # ################# End of 'buildcfuncs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'buildcfuncs' in the type store
    # Getting the type of 'stypy_return_type' (line 1148)
    stypy_return_type_74741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_74741)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'buildcfuncs'
    return stypy_return_type_74741

# Assigning a type to the variable 'buildcfuncs' (line 1148)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1148, 0), 'buildcfuncs', buildcfuncs)

@norecursion
def append_needs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_74742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1162, 28), 'int')
    defaults = [int_74742]
    # Create a new context for function 'append_needs'
    module_type_store = module_type_store.open_function_context('append_needs', 1162, 0, False)
    
    # Passed parameters checking function
    append_needs.stypy_localization = localization
    append_needs.stypy_type_of_self = None
    append_needs.stypy_type_store = module_type_store
    append_needs.stypy_function_name = 'append_needs'
    append_needs.stypy_param_names_list = ['need', 'flag']
    append_needs.stypy_varargs_param_name = None
    append_needs.stypy_kwargs_param_name = None
    append_needs.stypy_call_defaults = defaults
    append_needs.stypy_call_varargs = varargs
    append_needs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'append_needs', ['need', 'flag'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'append_needs', localization, ['need', 'flag'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'append_needs(...)' code ##################

    # Marking variables as global (line 1163)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 1163, 4), 'outneeds')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 1163, 4), 'needs')
    
    # Type idiom detected: calculating its left and rigth part (line 1164)
    # Getting the type of 'list' (line 1164)
    list_74743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 24), 'list')
    # Getting the type of 'need' (line 1164)
    need_74744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 18), 'need')
    
    (may_be_74745, more_types_in_union_74746) = may_be_subtype(list_74743, need_74744)

    if may_be_74745:

        if more_types_in_union_74746:
            # Runtime conditional SSA (line 1164)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'need' (line 1164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1164, 4), 'need', remove_not_subtype_from_union(need_74744, list))
        
        # Getting the type of 'need' (line 1165)
        need_74747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1165, 17), 'need')
        # Testing the type of a for loop iterable (line 1165)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1165, 8), need_74747)
        # Getting the type of the for loop variable (line 1165)
        for_loop_var_74748 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1165, 8), need_74747)
        # Assigning a type to the variable 'n' (line 1165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1165, 8), 'n', for_loop_var_74748)
        # SSA begins for a for statement (line 1165)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append_needs(...): (line 1166)
        # Processing the call arguments (line 1166)
        # Getting the type of 'n' (line 1166)
        n_74750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 25), 'n', False)
        # Getting the type of 'flag' (line 1166)
        flag_74751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 28), 'flag', False)
        # Processing the call keyword arguments (line 1166)
        kwargs_74752 = {}
        # Getting the type of 'append_needs' (line 1166)
        append_needs_74749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 12), 'append_needs', False)
        # Calling append_needs(args, kwargs) (line 1166)
        append_needs_call_result_74753 = invoke(stypy.reporting.localization.Localization(__file__, 1166, 12), append_needs_74749, *[n_74750, flag_74751], **kwargs_74752)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_74746:
            # Runtime conditional SSA for else branch (line 1164)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_74745) or more_types_in_union_74746):
        # Assigning a type to the variable 'need' (line 1164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1164, 4), 'need', remove_subtype_from_union(need_74744, list))
        
        # Type idiom detected: calculating its left and rigth part (line 1167)
        # Getting the type of 'str' (line 1167)
        str_74754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1167, 26), 'str')
        # Getting the type of 'need' (line 1167)
        need_74755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1167, 20), 'need')
        
        (may_be_74756, more_types_in_union_74757) = may_be_subtype(str_74754, need_74755)

        if may_be_74756:

            if more_types_in_union_74757:
                # Runtime conditional SSA (line 1167)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'need' (line 1167)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1167, 9), 'need', remove_not_subtype_from_union(need_74755, str))
            
            
            # Getting the type of 'need' (line 1168)
            need_74758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1168, 15), 'need')
            # Applying the 'not' unary operator (line 1168)
            result_not__74759 = python_operator(stypy.reporting.localization.Localization(__file__, 1168, 11), 'not', need_74758)
            
            # Testing the type of an if condition (line 1168)
            if_condition_74760 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1168, 8), result_not__74759)
            # Assigning a type to the variable 'if_condition_74760' (line 1168)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1168, 8), 'if_condition_74760', if_condition_74760)
            # SSA begins for if statement (line 1168)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Assigning a type to the variable 'stypy_return_type' (line 1169)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1169, 12), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 1168)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Getting the type of 'need' (line 1170)
            need_74761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1170, 11), 'need')
            # Getting the type of 'includes0' (line 1170)
            includes0_74762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1170, 19), 'includes0')
            # Applying the binary operator 'in' (line 1170)
            result_contains_74763 = python_operator(stypy.reporting.localization.Localization(__file__, 1170, 11), 'in', need_74761, includes0_74762)
            
            # Testing the type of an if condition (line 1170)
            if_condition_74764 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1170, 8), result_contains_74763)
            # Assigning a type to the variable 'if_condition_74764' (line 1170)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1170, 8), 'if_condition_74764', if_condition_74764)
            # SSA begins for if statement (line 1170)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 1171):
            str_74765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1171, 16), 'str', 'includes0')
            # Assigning a type to the variable 'n' (line 1171)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1171, 12), 'n', str_74765)
            # SSA branch for the else part of an if statement (line 1170)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'need' (line 1172)
            need_74766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1172, 13), 'need')
            # Getting the type of 'includes' (line 1172)
            includes_74767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1172, 21), 'includes')
            # Applying the binary operator 'in' (line 1172)
            result_contains_74768 = python_operator(stypy.reporting.localization.Localization(__file__, 1172, 13), 'in', need_74766, includes_74767)
            
            # Testing the type of an if condition (line 1172)
            if_condition_74769 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1172, 13), result_contains_74768)
            # Assigning a type to the variable 'if_condition_74769' (line 1172)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1172, 13), 'if_condition_74769', if_condition_74769)
            # SSA begins for if statement (line 1172)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 1173):
            str_74770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1173, 16), 'str', 'includes')
            # Assigning a type to the variable 'n' (line 1173)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1173, 12), 'n', str_74770)
            # SSA branch for the else part of an if statement (line 1172)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'need' (line 1174)
            need_74771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1174, 13), 'need')
            # Getting the type of 'typedefs' (line 1174)
            typedefs_74772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1174, 21), 'typedefs')
            # Applying the binary operator 'in' (line 1174)
            result_contains_74773 = python_operator(stypy.reporting.localization.Localization(__file__, 1174, 13), 'in', need_74771, typedefs_74772)
            
            # Testing the type of an if condition (line 1174)
            if_condition_74774 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1174, 13), result_contains_74773)
            # Assigning a type to the variable 'if_condition_74774' (line 1174)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1174, 13), 'if_condition_74774', if_condition_74774)
            # SSA begins for if statement (line 1174)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 1175):
            str_74775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1175, 16), 'str', 'typedefs')
            # Assigning a type to the variable 'n' (line 1175)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1175, 12), 'n', str_74775)
            # SSA branch for the else part of an if statement (line 1174)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'need' (line 1176)
            need_74776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1176, 13), 'need')
            # Getting the type of 'typedefs_generated' (line 1176)
            typedefs_generated_74777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1176, 21), 'typedefs_generated')
            # Applying the binary operator 'in' (line 1176)
            result_contains_74778 = python_operator(stypy.reporting.localization.Localization(__file__, 1176, 13), 'in', need_74776, typedefs_generated_74777)
            
            # Testing the type of an if condition (line 1176)
            if_condition_74779 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1176, 13), result_contains_74778)
            # Assigning a type to the variable 'if_condition_74779' (line 1176)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1176, 13), 'if_condition_74779', if_condition_74779)
            # SSA begins for if statement (line 1176)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 1177):
            str_74780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1177, 16), 'str', 'typedefs_generated')
            # Assigning a type to the variable 'n' (line 1177)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1177, 12), 'n', str_74780)
            # SSA branch for the else part of an if statement (line 1176)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'need' (line 1178)
            need_74781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1178, 13), 'need')
            # Getting the type of 'cppmacros' (line 1178)
            cppmacros_74782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1178, 21), 'cppmacros')
            # Applying the binary operator 'in' (line 1178)
            result_contains_74783 = python_operator(stypy.reporting.localization.Localization(__file__, 1178, 13), 'in', need_74781, cppmacros_74782)
            
            # Testing the type of an if condition (line 1178)
            if_condition_74784 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1178, 13), result_contains_74783)
            # Assigning a type to the variable 'if_condition_74784' (line 1178)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1178, 13), 'if_condition_74784', if_condition_74784)
            # SSA begins for if statement (line 1178)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 1179):
            str_74785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1179, 16), 'str', 'cppmacros')
            # Assigning a type to the variable 'n' (line 1179)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1179, 12), 'n', str_74785)
            # SSA branch for the else part of an if statement (line 1178)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'need' (line 1180)
            need_74786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1180, 13), 'need')
            # Getting the type of 'cfuncs' (line 1180)
            cfuncs_74787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1180, 21), 'cfuncs')
            # Applying the binary operator 'in' (line 1180)
            result_contains_74788 = python_operator(stypy.reporting.localization.Localization(__file__, 1180, 13), 'in', need_74786, cfuncs_74787)
            
            # Testing the type of an if condition (line 1180)
            if_condition_74789 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1180, 13), result_contains_74788)
            # Assigning a type to the variable 'if_condition_74789' (line 1180)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1180, 13), 'if_condition_74789', if_condition_74789)
            # SSA begins for if statement (line 1180)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 1181):
            str_74790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1181, 16), 'str', 'cfuncs')
            # Assigning a type to the variable 'n' (line 1181)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1181, 12), 'n', str_74790)
            # SSA branch for the else part of an if statement (line 1180)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'need' (line 1182)
            need_74791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 13), 'need')
            # Getting the type of 'callbacks' (line 1182)
            callbacks_74792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 21), 'callbacks')
            # Applying the binary operator 'in' (line 1182)
            result_contains_74793 = python_operator(stypy.reporting.localization.Localization(__file__, 1182, 13), 'in', need_74791, callbacks_74792)
            
            # Testing the type of an if condition (line 1182)
            if_condition_74794 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1182, 13), result_contains_74793)
            # Assigning a type to the variable 'if_condition_74794' (line 1182)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1182, 13), 'if_condition_74794', if_condition_74794)
            # SSA begins for if statement (line 1182)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 1183):
            str_74795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1183, 16), 'str', 'callbacks')
            # Assigning a type to the variable 'n' (line 1183)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1183, 12), 'n', str_74795)
            # SSA branch for the else part of an if statement (line 1182)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'need' (line 1184)
            need_74796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1184, 13), 'need')
            # Getting the type of 'f90modhooks' (line 1184)
            f90modhooks_74797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1184, 21), 'f90modhooks')
            # Applying the binary operator 'in' (line 1184)
            result_contains_74798 = python_operator(stypy.reporting.localization.Localization(__file__, 1184, 13), 'in', need_74796, f90modhooks_74797)
            
            # Testing the type of an if condition (line 1184)
            if_condition_74799 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1184, 13), result_contains_74798)
            # Assigning a type to the variable 'if_condition_74799' (line 1184)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1184, 13), 'if_condition_74799', if_condition_74799)
            # SSA begins for if statement (line 1184)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 1185):
            str_74800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1185, 16), 'str', 'f90modhooks')
            # Assigning a type to the variable 'n' (line 1185)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1185, 12), 'n', str_74800)
            # SSA branch for the else part of an if statement (line 1184)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'need' (line 1186)
            need_74801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1186, 13), 'need')
            # Getting the type of 'commonhooks' (line 1186)
            commonhooks_74802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1186, 21), 'commonhooks')
            # Applying the binary operator 'in' (line 1186)
            result_contains_74803 = python_operator(stypy.reporting.localization.Localization(__file__, 1186, 13), 'in', need_74801, commonhooks_74802)
            
            # Testing the type of an if condition (line 1186)
            if_condition_74804 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1186, 13), result_contains_74803)
            # Assigning a type to the variable 'if_condition_74804' (line 1186)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1186, 13), 'if_condition_74804', if_condition_74804)
            # SSA begins for if statement (line 1186)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 1187):
            str_74805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1187, 16), 'str', 'commonhooks')
            # Assigning a type to the variable 'n' (line 1187)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1187, 12), 'n', str_74805)
            # SSA branch for the else part of an if statement (line 1186)
            module_type_store.open_ssa_branch('else')
            
            # Call to errmess(...): (line 1189)
            # Processing the call arguments (line 1189)
            str_74807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1189, 20), 'str', 'append_needs: unknown need %s\n')
            
            # Call to repr(...): (line 1189)
            # Processing the call arguments (line 1189)
            # Getting the type of 'need' (line 1189)
            need_74809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1189, 62), 'need', False)
            # Processing the call keyword arguments (line 1189)
            kwargs_74810 = {}
            # Getting the type of 'repr' (line 1189)
            repr_74808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1189, 57), 'repr', False)
            # Calling repr(args, kwargs) (line 1189)
            repr_call_result_74811 = invoke(stypy.reporting.localization.Localization(__file__, 1189, 57), repr_74808, *[need_74809], **kwargs_74810)
            
            # Applying the binary operator '%' (line 1189)
            result_mod_74812 = python_operator(stypy.reporting.localization.Localization(__file__, 1189, 20), '%', str_74807, repr_call_result_74811)
            
            # Processing the call keyword arguments (line 1189)
            kwargs_74813 = {}
            # Getting the type of 'errmess' (line 1189)
            errmess_74806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1189, 12), 'errmess', False)
            # Calling errmess(args, kwargs) (line 1189)
            errmess_call_result_74814 = invoke(stypy.reporting.localization.Localization(__file__, 1189, 12), errmess_74806, *[result_mod_74812], **kwargs_74813)
            
            # Assigning a type to the variable 'stypy_return_type' (line 1190)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1190, 12), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 1186)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 1184)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 1182)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 1180)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 1178)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 1176)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 1174)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 1172)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 1170)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Getting the type of 'need' (line 1191)
            need_74815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 11), 'need')
            
            # Obtaining the type of the subscript
            # Getting the type of 'n' (line 1191)
            n_74816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 28), 'n')
            # Getting the type of 'outneeds' (line 1191)
            outneeds_74817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 19), 'outneeds')
            # Obtaining the member '__getitem__' of a type (line 1191)
            getitem___74818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1191, 19), outneeds_74817, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 1191)
            subscript_call_result_74819 = invoke(stypy.reporting.localization.Localization(__file__, 1191, 19), getitem___74818, n_74816)
            
            # Applying the binary operator 'in' (line 1191)
            result_contains_74820 = python_operator(stypy.reporting.localization.Localization(__file__, 1191, 11), 'in', need_74815, subscript_call_result_74819)
            
            # Testing the type of an if condition (line 1191)
            if_condition_74821 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1191, 8), result_contains_74820)
            # Assigning a type to the variable 'if_condition_74821' (line 1191)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1191, 8), 'if_condition_74821', if_condition_74821)
            # SSA begins for if statement (line 1191)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Assigning a type to the variable 'stypy_return_type' (line 1192)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1192, 12), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 1191)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Getting the type of 'flag' (line 1193)
            flag_74822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1193, 11), 'flag')
            # Testing the type of an if condition (line 1193)
            if_condition_74823 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1193, 8), flag_74822)
            # Assigning a type to the variable 'if_condition_74823' (line 1193)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1193, 8), 'if_condition_74823', if_condition_74823)
            # SSA begins for if statement (line 1193)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Dict to a Name (line 1194):
            
            # Obtaining an instance of the builtin type 'dict' (line 1194)
            dict_74824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1194, 18), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 1194)
            
            # Assigning a type to the variable 'tmp' (line 1194)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1194, 12), 'tmp', dict_74824)
            
            
            # Getting the type of 'need' (line 1195)
            need_74825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 15), 'need')
            # Getting the type of 'needs' (line 1195)
            needs_74826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 23), 'needs')
            # Applying the binary operator 'in' (line 1195)
            result_contains_74827 = python_operator(stypy.reporting.localization.Localization(__file__, 1195, 15), 'in', need_74825, needs_74826)
            
            # Testing the type of an if condition (line 1195)
            if_condition_74828 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1195, 12), result_contains_74827)
            # Assigning a type to the variable 'if_condition_74828' (line 1195)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1195, 12), 'if_condition_74828', if_condition_74828)
            # SSA begins for if statement (line 1195)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'need' (line 1196)
            need_74829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1196, 32), 'need')
            # Getting the type of 'needs' (line 1196)
            needs_74830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1196, 26), 'needs')
            # Obtaining the member '__getitem__' of a type (line 1196)
            getitem___74831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1196, 26), needs_74830, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 1196)
            subscript_call_result_74832 = invoke(stypy.reporting.localization.Localization(__file__, 1196, 26), getitem___74831, need_74829)
            
            # Testing the type of a for loop iterable (line 1196)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1196, 16), subscript_call_result_74832)
            # Getting the type of the for loop variable (line 1196)
            for_loop_var_74833 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1196, 16), subscript_call_result_74832)
            # Assigning a type to the variable 'nn' (line 1196)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1196, 16), 'nn', for_loop_var_74833)
            # SSA begins for a for statement (line 1196)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 1197):
            
            # Call to append_needs(...): (line 1197)
            # Processing the call arguments (line 1197)
            # Getting the type of 'nn' (line 1197)
            nn_74835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1197, 37), 'nn', False)
            int_74836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1197, 41), 'int')
            # Processing the call keyword arguments (line 1197)
            kwargs_74837 = {}
            # Getting the type of 'append_needs' (line 1197)
            append_needs_74834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1197, 24), 'append_needs', False)
            # Calling append_needs(args, kwargs) (line 1197)
            append_needs_call_result_74838 = invoke(stypy.reporting.localization.Localization(__file__, 1197, 24), append_needs_74834, *[nn_74835, int_74836], **kwargs_74837)
            
            # Assigning a type to the variable 't' (line 1197)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1197, 20), 't', append_needs_call_result_74838)
            
            # Type idiom detected: calculating its left and rigth part (line 1198)
            # Getting the type of 'dict' (line 1198)
            dict_74839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1198, 37), 'dict')
            # Getting the type of 't' (line 1198)
            t_74840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1198, 34), 't')
            
            (may_be_74841, more_types_in_union_74842) = may_be_subtype(dict_74839, t_74840)

            if may_be_74841:

                if more_types_in_union_74842:
                    # Runtime conditional SSA (line 1198)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 't' (line 1198)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1198, 20), 't', remove_not_subtype_from_union(t_74840, dict))
                
                
                # Call to keys(...): (line 1199)
                # Processing the call keyword arguments (line 1199)
                kwargs_74845 = {}
                # Getting the type of 't' (line 1199)
                t_74843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1199, 35), 't', False)
                # Obtaining the member 'keys' of a type (line 1199)
                keys_74844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1199, 35), t_74843, 'keys')
                # Calling keys(args, kwargs) (line 1199)
                keys_call_result_74846 = invoke(stypy.reporting.localization.Localization(__file__, 1199, 35), keys_74844, *[], **kwargs_74845)
                
                # Testing the type of a for loop iterable (line 1199)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1199, 24), keys_call_result_74846)
                # Getting the type of the for loop variable (line 1199)
                for_loop_var_74847 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1199, 24), keys_call_result_74846)
                # Assigning a type to the variable 'nnn' (line 1199)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1199, 24), 'nnn', for_loop_var_74847)
                # SSA begins for a for statement (line 1199)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Getting the type of 'nnn' (line 1200)
                nnn_74848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1200, 31), 'nnn')
                # Getting the type of 'tmp' (line 1200)
                tmp_74849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1200, 38), 'tmp')
                # Applying the binary operator 'in' (line 1200)
                result_contains_74850 = python_operator(stypy.reporting.localization.Localization(__file__, 1200, 31), 'in', nnn_74848, tmp_74849)
                
                # Testing the type of an if condition (line 1200)
                if_condition_74851 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1200, 28), result_contains_74850)
                # Assigning a type to the variable 'if_condition_74851' (line 1200)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1200, 28), 'if_condition_74851', if_condition_74851)
                # SSA begins for if statement (line 1200)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Subscript (line 1201):
                
                # Obtaining the type of the subscript
                # Getting the type of 'nnn' (line 1201)
                nnn_74852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1201, 47), 'nnn')
                # Getting the type of 'tmp' (line 1201)
                tmp_74853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1201, 43), 'tmp')
                # Obtaining the member '__getitem__' of a type (line 1201)
                getitem___74854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1201, 43), tmp_74853, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 1201)
                subscript_call_result_74855 = invoke(stypy.reporting.localization.Localization(__file__, 1201, 43), getitem___74854, nnn_74852)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'nnn' (line 1201)
                nnn_74856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1201, 56), 'nnn')
                # Getting the type of 't' (line 1201)
                t_74857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1201, 54), 't')
                # Obtaining the member '__getitem__' of a type (line 1201)
                getitem___74858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1201, 54), t_74857, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 1201)
                subscript_call_result_74859 = invoke(stypy.reporting.localization.Localization(__file__, 1201, 54), getitem___74858, nnn_74856)
                
                # Applying the binary operator '+' (line 1201)
                result_add_74860 = python_operator(stypy.reporting.localization.Localization(__file__, 1201, 43), '+', subscript_call_result_74855, subscript_call_result_74859)
                
                # Getting the type of 'tmp' (line 1201)
                tmp_74861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1201, 32), 'tmp')
                # Getting the type of 'nnn' (line 1201)
                nnn_74862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1201, 36), 'nnn')
                # Storing an element on a container (line 1201)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1201, 32), tmp_74861, (nnn_74862, result_add_74860))
                # SSA branch for the else part of an if statement (line 1200)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Subscript to a Subscript (line 1203):
                
                # Obtaining the type of the subscript
                # Getting the type of 'nnn' (line 1203)
                nnn_74863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1203, 45), 'nnn')
                # Getting the type of 't' (line 1203)
                t_74864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1203, 43), 't')
                # Obtaining the member '__getitem__' of a type (line 1203)
                getitem___74865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1203, 43), t_74864, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 1203)
                subscript_call_result_74866 = invoke(stypy.reporting.localization.Localization(__file__, 1203, 43), getitem___74865, nnn_74863)
                
                # Getting the type of 'tmp' (line 1203)
                tmp_74867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1203, 32), 'tmp')
                # Getting the type of 'nnn' (line 1203)
                nnn_74868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1203, 36), 'nnn')
                # Storing an element on a container (line 1203)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1203, 32), tmp_74867, (nnn_74868, subscript_call_result_74866))
                # SSA join for if statement (line 1200)
                module_type_store = module_type_store.join_ssa_context()
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()
                

                if more_types_in_union_74842:
                    # SSA join for if statement (line 1198)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 1195)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Call to keys(...): (line 1204)
            # Processing the call keyword arguments (line 1204)
            kwargs_74871 = {}
            # Getting the type of 'tmp' (line 1204)
            tmp_74869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1204, 22), 'tmp', False)
            # Obtaining the member 'keys' of a type (line 1204)
            keys_74870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1204, 22), tmp_74869, 'keys')
            # Calling keys(args, kwargs) (line 1204)
            keys_call_result_74872 = invoke(stypy.reporting.localization.Localization(__file__, 1204, 22), keys_74870, *[], **kwargs_74871)
            
            # Testing the type of a for loop iterable (line 1204)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1204, 12), keys_call_result_74872)
            # Getting the type of the for loop variable (line 1204)
            for_loop_var_74873 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1204, 12), keys_call_result_74872)
            # Assigning a type to the variable 'nn' (line 1204)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1204, 12), 'nn', for_loop_var_74873)
            # SSA begins for a for statement (line 1204)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'nn' (line 1205)
            nn_74874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 31), 'nn')
            # Getting the type of 'tmp' (line 1205)
            tmp_74875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 27), 'tmp')
            # Obtaining the member '__getitem__' of a type (line 1205)
            getitem___74876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1205, 27), tmp_74875, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 1205)
            subscript_call_result_74877 = invoke(stypy.reporting.localization.Localization(__file__, 1205, 27), getitem___74876, nn_74874)
            
            # Testing the type of a for loop iterable (line 1205)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1205, 16), subscript_call_result_74877)
            # Getting the type of the for loop variable (line 1205)
            for_loop_var_74878 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1205, 16), subscript_call_result_74877)
            # Assigning a type to the variable 'nnn' (line 1205)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1205, 16), 'nnn', for_loop_var_74878)
            # SSA begins for a for statement (line 1205)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 'nnn' (line 1206)
            nnn_74879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 23), 'nnn')
            
            # Obtaining the type of the subscript
            # Getting the type of 'nn' (line 1206)
            nn_74880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 43), 'nn')
            # Getting the type of 'outneeds' (line 1206)
            outneeds_74881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 34), 'outneeds')
            # Obtaining the member '__getitem__' of a type (line 1206)
            getitem___74882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1206, 34), outneeds_74881, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 1206)
            subscript_call_result_74883 = invoke(stypy.reporting.localization.Localization(__file__, 1206, 34), getitem___74882, nn_74880)
            
            # Applying the binary operator 'notin' (line 1206)
            result_contains_74884 = python_operator(stypy.reporting.localization.Localization(__file__, 1206, 23), 'notin', nnn_74879, subscript_call_result_74883)
            
            # Testing the type of an if condition (line 1206)
            if_condition_74885 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1206, 20), result_contains_74884)
            # Assigning a type to the variable 'if_condition_74885' (line 1206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1206, 20), 'if_condition_74885', if_condition_74885)
            # SSA begins for if statement (line 1206)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Subscript (line 1207):
            
            # Obtaining an instance of the builtin type 'list' (line 1207)
            list_74886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1207, 39), 'list')
            # Adding type elements to the builtin type 'list' instance (line 1207)
            # Adding element type (line 1207)
            # Getting the type of 'nnn' (line 1207)
            nnn_74887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 40), 'nnn')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1207, 39), list_74886, nnn_74887)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'nn' (line 1207)
            nn_74888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 56), 'nn')
            # Getting the type of 'outneeds' (line 1207)
            outneeds_74889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 47), 'outneeds')
            # Obtaining the member '__getitem__' of a type (line 1207)
            getitem___74890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1207, 47), outneeds_74889, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 1207)
            subscript_call_result_74891 = invoke(stypy.reporting.localization.Localization(__file__, 1207, 47), getitem___74890, nn_74888)
            
            # Applying the binary operator '+' (line 1207)
            result_add_74892 = python_operator(stypy.reporting.localization.Localization(__file__, 1207, 39), '+', list_74886, subscript_call_result_74891)
            
            # Getting the type of 'outneeds' (line 1207)
            outneeds_74893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 24), 'outneeds')
            # Getting the type of 'nn' (line 1207)
            nn_74894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 33), 'nn')
            # Storing an element on a container (line 1207)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1207, 24), outneeds_74893, (nn_74894, result_add_74892))
            # SSA join for if statement (line 1206)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to append(...): (line 1208)
            # Processing the call arguments (line 1208)
            # Getting the type of 'need' (line 1208)
            need_74900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1208, 31), 'need', False)
            # Processing the call keyword arguments (line 1208)
            kwargs_74901 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'n' (line 1208)
            n_74895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1208, 21), 'n', False)
            # Getting the type of 'outneeds' (line 1208)
            outneeds_74896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1208, 12), 'outneeds', False)
            # Obtaining the member '__getitem__' of a type (line 1208)
            getitem___74897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1208, 12), outneeds_74896, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 1208)
            subscript_call_result_74898 = invoke(stypy.reporting.localization.Localization(__file__, 1208, 12), getitem___74897, n_74895)
            
            # Obtaining the member 'append' of a type (line 1208)
            append_74899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1208, 12), subscript_call_result_74898, 'append')
            # Calling append(args, kwargs) (line 1208)
            append_call_result_74902 = invoke(stypy.reporting.localization.Localization(__file__, 1208, 12), append_74899, *[need_74900], **kwargs_74901)
            
            # SSA branch for the else part of an if statement (line 1193)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Dict to a Name (line 1210):
            
            # Obtaining an instance of the builtin type 'dict' (line 1210)
            dict_74903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1210, 18), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 1210)
            
            # Assigning a type to the variable 'tmp' (line 1210)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1210, 12), 'tmp', dict_74903)
            
            
            # Getting the type of 'need' (line 1211)
            need_74904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 15), 'need')
            # Getting the type of 'needs' (line 1211)
            needs_74905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 23), 'needs')
            # Applying the binary operator 'in' (line 1211)
            result_contains_74906 = python_operator(stypy.reporting.localization.Localization(__file__, 1211, 15), 'in', need_74904, needs_74905)
            
            # Testing the type of an if condition (line 1211)
            if_condition_74907 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1211, 12), result_contains_74906)
            # Assigning a type to the variable 'if_condition_74907' (line 1211)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1211, 12), 'if_condition_74907', if_condition_74907)
            # SSA begins for if statement (line 1211)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'need' (line 1212)
            need_74908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1212, 32), 'need')
            # Getting the type of 'needs' (line 1212)
            needs_74909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1212, 26), 'needs')
            # Obtaining the member '__getitem__' of a type (line 1212)
            getitem___74910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1212, 26), needs_74909, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 1212)
            subscript_call_result_74911 = invoke(stypy.reporting.localization.Localization(__file__, 1212, 26), getitem___74910, need_74908)
            
            # Testing the type of a for loop iterable (line 1212)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1212, 16), subscript_call_result_74911)
            # Getting the type of the for loop variable (line 1212)
            for_loop_var_74912 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1212, 16), subscript_call_result_74911)
            # Assigning a type to the variable 'nn' (line 1212)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1212, 16), 'nn', for_loop_var_74912)
            # SSA begins for a for statement (line 1212)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 1213):
            
            # Call to append_needs(...): (line 1213)
            # Processing the call arguments (line 1213)
            # Getting the type of 'nn' (line 1213)
            nn_74914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1213, 37), 'nn', False)
            # Getting the type of 'flag' (line 1213)
            flag_74915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1213, 41), 'flag', False)
            # Processing the call keyword arguments (line 1213)
            kwargs_74916 = {}
            # Getting the type of 'append_needs' (line 1213)
            append_needs_74913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1213, 24), 'append_needs', False)
            # Calling append_needs(args, kwargs) (line 1213)
            append_needs_call_result_74917 = invoke(stypy.reporting.localization.Localization(__file__, 1213, 24), append_needs_74913, *[nn_74914, flag_74915], **kwargs_74916)
            
            # Assigning a type to the variable 't' (line 1213)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1213, 20), 't', append_needs_call_result_74917)
            
            # Type idiom detected: calculating its left and rigth part (line 1214)
            # Getting the type of 'dict' (line 1214)
            dict_74918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1214, 37), 'dict')
            # Getting the type of 't' (line 1214)
            t_74919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1214, 34), 't')
            
            (may_be_74920, more_types_in_union_74921) = may_be_subtype(dict_74918, t_74919)

            if may_be_74920:

                if more_types_in_union_74921:
                    # Runtime conditional SSA (line 1214)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 't' (line 1214)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1214, 20), 't', remove_not_subtype_from_union(t_74919, dict))
                
                
                # Call to keys(...): (line 1215)
                # Processing the call keyword arguments (line 1215)
                kwargs_74924 = {}
                # Getting the type of 't' (line 1215)
                t_74922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1215, 35), 't', False)
                # Obtaining the member 'keys' of a type (line 1215)
                keys_74923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1215, 35), t_74922, 'keys')
                # Calling keys(args, kwargs) (line 1215)
                keys_call_result_74925 = invoke(stypy.reporting.localization.Localization(__file__, 1215, 35), keys_74923, *[], **kwargs_74924)
                
                # Testing the type of a for loop iterable (line 1215)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1215, 24), keys_call_result_74925)
                # Getting the type of the for loop variable (line 1215)
                for_loop_var_74926 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1215, 24), keys_call_result_74925)
                # Assigning a type to the variable 'nnn' (line 1215)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1215, 24), 'nnn', for_loop_var_74926)
                # SSA begins for a for statement (line 1215)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Getting the type of 'nnn' (line 1216)
                nnn_74927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1216, 31), 'nnn')
                # Getting the type of 'tmp' (line 1216)
                tmp_74928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1216, 38), 'tmp')
                # Applying the binary operator 'in' (line 1216)
                result_contains_74929 = python_operator(stypy.reporting.localization.Localization(__file__, 1216, 31), 'in', nnn_74927, tmp_74928)
                
                # Testing the type of an if condition (line 1216)
                if_condition_74930 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1216, 28), result_contains_74929)
                # Assigning a type to the variable 'if_condition_74930' (line 1216)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1216, 28), 'if_condition_74930', if_condition_74930)
                # SSA begins for if statement (line 1216)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Subscript (line 1217):
                
                # Obtaining the type of the subscript
                # Getting the type of 'nnn' (line 1217)
                nnn_74931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 45), 'nnn')
                # Getting the type of 't' (line 1217)
                t_74932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 43), 't')
                # Obtaining the member '__getitem__' of a type (line 1217)
                getitem___74933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1217, 43), t_74932, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 1217)
                subscript_call_result_74934 = invoke(stypy.reporting.localization.Localization(__file__, 1217, 43), getitem___74933, nnn_74931)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'nnn' (line 1217)
                nnn_74935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 56), 'nnn')
                # Getting the type of 'tmp' (line 1217)
                tmp_74936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 52), 'tmp')
                # Obtaining the member '__getitem__' of a type (line 1217)
                getitem___74937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1217, 52), tmp_74936, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 1217)
                subscript_call_result_74938 = invoke(stypy.reporting.localization.Localization(__file__, 1217, 52), getitem___74937, nnn_74935)
                
                # Applying the binary operator '+' (line 1217)
                result_add_74939 = python_operator(stypy.reporting.localization.Localization(__file__, 1217, 43), '+', subscript_call_result_74934, subscript_call_result_74938)
                
                # Getting the type of 'tmp' (line 1217)
                tmp_74940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 32), 'tmp')
                # Getting the type of 'nnn' (line 1217)
                nnn_74941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 36), 'nnn')
                # Storing an element on a container (line 1217)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1217, 32), tmp_74940, (nnn_74941, result_add_74939))
                # SSA branch for the else part of an if statement (line 1216)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Subscript to a Subscript (line 1219):
                
                # Obtaining the type of the subscript
                # Getting the type of 'nnn' (line 1219)
                nnn_74942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1219, 45), 'nnn')
                # Getting the type of 't' (line 1219)
                t_74943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1219, 43), 't')
                # Obtaining the member '__getitem__' of a type (line 1219)
                getitem___74944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1219, 43), t_74943, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 1219)
                subscript_call_result_74945 = invoke(stypy.reporting.localization.Localization(__file__, 1219, 43), getitem___74944, nnn_74942)
                
                # Getting the type of 'tmp' (line 1219)
                tmp_74946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1219, 32), 'tmp')
                # Getting the type of 'nnn' (line 1219)
                nnn_74947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1219, 36), 'nnn')
                # Storing an element on a container (line 1219)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1219, 32), tmp_74946, (nnn_74947, subscript_call_result_74945))
                # SSA join for if statement (line 1216)
                module_type_store = module_type_store.join_ssa_context()
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()
                

                if more_types_in_union_74921:
                    # SSA join for if statement (line 1214)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 1211)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Getting the type of 'n' (line 1220)
            n_74948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1220, 15), 'n')
            # Getting the type of 'tmp' (line 1220)
            tmp_74949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1220, 24), 'tmp')
            # Applying the binary operator 'notin' (line 1220)
            result_contains_74950 = python_operator(stypy.reporting.localization.Localization(__file__, 1220, 15), 'notin', n_74948, tmp_74949)
            
            # Testing the type of an if condition (line 1220)
            if_condition_74951 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1220, 12), result_contains_74950)
            # Assigning a type to the variable 'if_condition_74951' (line 1220)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1220, 12), 'if_condition_74951', if_condition_74951)
            # SSA begins for if statement (line 1220)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a List to a Subscript (line 1221):
            
            # Obtaining an instance of the builtin type 'list' (line 1221)
            list_74952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1221, 25), 'list')
            # Adding type elements to the builtin type 'list' instance (line 1221)
            
            # Getting the type of 'tmp' (line 1221)
            tmp_74953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 16), 'tmp')
            # Getting the type of 'n' (line 1221)
            n_74954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 20), 'n')
            # Storing an element on a container (line 1221)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1221, 16), tmp_74953, (n_74954, list_74952))
            # SSA join for if statement (line 1220)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to append(...): (line 1222)
            # Processing the call arguments (line 1222)
            # Getting the type of 'need' (line 1222)
            need_74960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1222, 26), 'need', False)
            # Processing the call keyword arguments (line 1222)
            kwargs_74961 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'n' (line 1222)
            n_74955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1222, 16), 'n', False)
            # Getting the type of 'tmp' (line 1222)
            tmp_74956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1222, 12), 'tmp', False)
            # Obtaining the member '__getitem__' of a type (line 1222)
            getitem___74957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1222, 12), tmp_74956, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 1222)
            subscript_call_result_74958 = invoke(stypy.reporting.localization.Localization(__file__, 1222, 12), getitem___74957, n_74955)
            
            # Obtaining the member 'append' of a type (line 1222)
            append_74959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1222, 12), subscript_call_result_74958, 'append')
            # Calling append(args, kwargs) (line 1222)
            append_call_result_74962 = invoke(stypy.reporting.localization.Localization(__file__, 1222, 12), append_74959, *[need_74960], **kwargs_74961)
            
            # Getting the type of 'tmp' (line 1223)
            tmp_74963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1223, 19), 'tmp')
            # Assigning a type to the variable 'stypy_return_type' (line 1223)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1223, 12), 'stypy_return_type', tmp_74963)
            # SSA join for if statement (line 1193)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_74757:
                # Runtime conditional SSA for else branch (line 1167)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_74756) or more_types_in_union_74757):
            # Assigning a type to the variable 'need' (line 1167)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1167, 9), 'need', remove_subtype_from_union(need_74755, str))
            
            # Call to errmess(...): (line 1225)
            # Processing the call arguments (line 1225)
            str_74965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1225, 16), 'str', 'append_needs: expected list or string but got :%s\n')
            
            # Call to repr(...): (line 1226)
            # Processing the call arguments (line 1226)
            # Getting the type of 'need' (line 1226)
            need_74967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1226, 22), 'need', False)
            # Processing the call keyword arguments (line 1226)
            kwargs_74968 = {}
            # Getting the type of 'repr' (line 1226)
            repr_74966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1226, 17), 'repr', False)
            # Calling repr(args, kwargs) (line 1226)
            repr_call_result_74969 = invoke(stypy.reporting.localization.Localization(__file__, 1226, 17), repr_74966, *[need_74967], **kwargs_74968)
            
            # Applying the binary operator '%' (line 1225)
            result_mod_74970 = python_operator(stypy.reporting.localization.Localization(__file__, 1225, 16), '%', str_74965, repr_call_result_74969)
            
            # Processing the call keyword arguments (line 1225)
            kwargs_74971 = {}
            # Getting the type of 'errmess' (line 1225)
            errmess_74964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1225, 8), 'errmess', False)
            # Calling errmess(args, kwargs) (line 1225)
            errmess_call_result_74972 = invoke(stypy.reporting.localization.Localization(__file__, 1225, 8), errmess_74964, *[result_mod_74970], **kwargs_74971)
            

            if (may_be_74756 and more_types_in_union_74757):
                # SSA join for if statement (line 1167)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_74745 and more_types_in_union_74746):
            # SSA join for if statement (line 1164)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'append_needs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'append_needs' in the type store
    # Getting the type of 'stypy_return_type' (line 1162)
    stypy_return_type_74973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1162, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_74973)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'append_needs'
    return stypy_return_type_74973

# Assigning a type to the variable 'append_needs' (line 1162)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1162, 0), 'append_needs', append_needs)

@norecursion
def get_needs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_needs'
    module_type_store = module_type_store.open_function_context('get_needs', 1229, 0, False)
    
    # Passed parameters checking function
    get_needs.stypy_localization = localization
    get_needs.stypy_type_of_self = None
    get_needs.stypy_type_store = module_type_store
    get_needs.stypy_function_name = 'get_needs'
    get_needs.stypy_param_names_list = []
    get_needs.stypy_varargs_param_name = None
    get_needs.stypy_kwargs_param_name = None
    get_needs.stypy_call_defaults = defaults
    get_needs.stypy_call_varargs = varargs
    get_needs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_needs', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_needs', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_needs(...)' code ##################

    # Marking variables as global (line 1230)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 1230, 4), 'outneeds')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 1230, 4), 'needs')
    
    # Assigning a Dict to a Name (line 1231):
    
    # Obtaining an instance of the builtin type 'dict' (line 1231)
    dict_74974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1231, 10), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 1231)
    
    # Assigning a type to the variable 'res' (line 1231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1231, 4), 'res', dict_74974)
    
    
    # Call to keys(...): (line 1232)
    # Processing the call keyword arguments (line 1232)
    kwargs_74977 = {}
    # Getting the type of 'outneeds' (line 1232)
    outneeds_74975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1232, 13), 'outneeds', False)
    # Obtaining the member 'keys' of a type (line 1232)
    keys_74976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1232, 13), outneeds_74975, 'keys')
    # Calling keys(args, kwargs) (line 1232)
    keys_call_result_74978 = invoke(stypy.reporting.localization.Localization(__file__, 1232, 13), keys_74976, *[], **kwargs_74977)
    
    # Testing the type of a for loop iterable (line 1232)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1232, 4), keys_call_result_74978)
    # Getting the type of the for loop variable (line 1232)
    for_loop_var_74979 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1232, 4), keys_call_result_74978)
    # Assigning a type to the variable 'n' (line 1232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1232, 4), 'n', for_loop_var_74979)
    # SSA begins for a for statement (line 1232)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a List to a Name (line 1233):
    
    # Obtaining an instance of the builtin type 'list' (line 1233)
    list_74980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1233, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1233)
    
    # Assigning a type to the variable 'out' (line 1233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1233, 8), 'out', list_74980)
    
    # Assigning a Call to a Name (line 1234):
    
    # Call to copy(...): (line 1234)
    # Processing the call arguments (line 1234)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1234)
    n_74983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 37), 'n', False)
    # Getting the type of 'outneeds' (line 1234)
    outneeds_74984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 28), 'outneeds', False)
    # Obtaining the member '__getitem__' of a type (line 1234)
    getitem___74985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1234, 28), outneeds_74984, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1234)
    subscript_call_result_74986 = invoke(stypy.reporting.localization.Localization(__file__, 1234, 28), getitem___74985, n_74983)
    
    # Processing the call keyword arguments (line 1234)
    kwargs_74987 = {}
    # Getting the type of 'copy' (line 1234)
    copy_74981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 18), 'copy', False)
    # Obtaining the member 'copy' of a type (line 1234)
    copy_74982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1234, 18), copy_74981, 'copy')
    # Calling copy(args, kwargs) (line 1234)
    copy_call_result_74988 = invoke(stypy.reporting.localization.Localization(__file__, 1234, 18), copy_74982, *[subscript_call_result_74986], **kwargs_74987)
    
    # Assigning a type to the variable 'saveout' (line 1234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1234, 8), 'saveout', copy_call_result_74988)
    
    
    
    # Call to len(...): (line 1235)
    # Processing the call arguments (line 1235)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1235)
    n_74990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 27), 'n', False)
    # Getting the type of 'outneeds' (line 1235)
    outneeds_74991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 18), 'outneeds', False)
    # Obtaining the member '__getitem__' of a type (line 1235)
    getitem___74992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1235, 18), outneeds_74991, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1235)
    subscript_call_result_74993 = invoke(stypy.reporting.localization.Localization(__file__, 1235, 18), getitem___74992, n_74990)
    
    # Processing the call keyword arguments (line 1235)
    kwargs_74994 = {}
    # Getting the type of 'len' (line 1235)
    len_74989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 14), 'len', False)
    # Calling len(args, kwargs) (line 1235)
    len_call_result_74995 = invoke(stypy.reporting.localization.Localization(__file__, 1235, 14), len_74989, *[subscript_call_result_74993], **kwargs_74994)
    
    int_74996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1235, 33), 'int')
    # Applying the binary operator '>' (line 1235)
    result_gt_74997 = python_operator(stypy.reporting.localization.Localization(__file__, 1235, 14), '>', len_call_result_74995, int_74996)
    
    # Testing the type of an if condition (line 1235)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1235, 8), result_gt_74997)
    # SSA begins for while statement (line 1235)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    
    # Obtaining the type of the subscript
    int_74998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1236, 27), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1236)
    n_74999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 24), 'n')
    # Getting the type of 'outneeds' (line 1236)
    outneeds_75000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 15), 'outneeds')
    # Obtaining the member '__getitem__' of a type (line 1236)
    getitem___75001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1236, 15), outneeds_75000, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1236)
    subscript_call_result_75002 = invoke(stypy.reporting.localization.Localization(__file__, 1236, 15), getitem___75001, n_74999)
    
    # Obtaining the member '__getitem__' of a type (line 1236)
    getitem___75003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1236, 15), subscript_call_result_75002, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1236)
    subscript_call_result_75004 = invoke(stypy.reporting.localization.Localization(__file__, 1236, 15), getitem___75003, int_74998)
    
    # Getting the type of 'needs' (line 1236)
    needs_75005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 37), 'needs')
    # Applying the binary operator 'notin' (line 1236)
    result_contains_75006 = python_operator(stypy.reporting.localization.Localization(__file__, 1236, 15), 'notin', subscript_call_result_75004, needs_75005)
    
    # Testing the type of an if condition (line 1236)
    if_condition_75007 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1236, 12), result_contains_75006)
    # Assigning a type to the variable 'if_condition_75007' (line 1236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1236, 12), 'if_condition_75007', if_condition_75007)
    # SSA begins for if statement (line 1236)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 1237)
    # Processing the call arguments (line 1237)
    
    # Obtaining the type of the subscript
    int_75010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1237, 39), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1237)
    n_75011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1237, 36), 'n', False)
    # Getting the type of 'outneeds' (line 1237)
    outneeds_75012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1237, 27), 'outneeds', False)
    # Obtaining the member '__getitem__' of a type (line 1237)
    getitem___75013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1237, 27), outneeds_75012, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1237)
    subscript_call_result_75014 = invoke(stypy.reporting.localization.Localization(__file__, 1237, 27), getitem___75013, n_75011)
    
    # Obtaining the member '__getitem__' of a type (line 1237)
    getitem___75015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1237, 27), subscript_call_result_75014, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1237)
    subscript_call_result_75016 = invoke(stypy.reporting.localization.Localization(__file__, 1237, 27), getitem___75015, int_75010)
    
    # Processing the call keyword arguments (line 1237)
    kwargs_75017 = {}
    # Getting the type of 'out' (line 1237)
    out_75008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1237, 16), 'out', False)
    # Obtaining the member 'append' of a type (line 1237)
    append_75009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1237, 16), out_75008, 'append')
    # Calling append(args, kwargs) (line 1237)
    append_call_result_75018 = invoke(stypy.reporting.localization.Localization(__file__, 1237, 16), append_75009, *[subscript_call_result_75016], **kwargs_75017)
    
    # Deleting a member
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1238)
    n_75019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 29), 'n')
    # Getting the type of 'outneeds' (line 1238)
    outneeds_75020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 20), 'outneeds')
    # Obtaining the member '__getitem__' of a type (line 1238)
    getitem___75021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1238, 20), outneeds_75020, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1238)
    subscript_call_result_75022 = invoke(stypy.reporting.localization.Localization(__file__, 1238, 20), getitem___75021, n_75019)
    
    
    # Obtaining the type of the subscript
    int_75023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1238, 32), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1238)
    n_75024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 29), 'n')
    # Getting the type of 'outneeds' (line 1238)
    outneeds_75025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 20), 'outneeds')
    # Obtaining the member '__getitem__' of a type (line 1238)
    getitem___75026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1238, 20), outneeds_75025, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1238)
    subscript_call_result_75027 = invoke(stypy.reporting.localization.Localization(__file__, 1238, 20), getitem___75026, n_75024)
    
    # Obtaining the member '__getitem__' of a type (line 1238)
    getitem___75028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1238, 20), subscript_call_result_75027, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1238)
    subscript_call_result_75029 = invoke(stypy.reporting.localization.Localization(__file__, 1238, 20), getitem___75028, int_75023)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1238, 16), subscript_call_result_75022, subscript_call_result_75029)
    # SSA branch for the else part of an if statement (line 1236)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 1240):
    int_75030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1240, 23), 'int')
    # Assigning a type to the variable 'flag' (line 1240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1240, 16), 'flag', int_75030)
    
    
    # Obtaining the type of the subscript
    int_75031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1241, 37), 'int')
    slice_75032 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1241, 25), int_75031, None, None)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1241)
    n_75033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1241, 34), 'n')
    # Getting the type of 'outneeds' (line 1241)
    outneeds_75034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1241, 25), 'outneeds')
    # Obtaining the member '__getitem__' of a type (line 1241)
    getitem___75035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1241, 25), outneeds_75034, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1241)
    subscript_call_result_75036 = invoke(stypy.reporting.localization.Localization(__file__, 1241, 25), getitem___75035, n_75033)
    
    # Obtaining the member '__getitem__' of a type (line 1241)
    getitem___75037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1241, 25), subscript_call_result_75036, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1241)
    subscript_call_result_75038 = invoke(stypy.reporting.localization.Localization(__file__, 1241, 25), getitem___75037, slice_75032)
    
    # Testing the type of a for loop iterable (line 1241)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1241, 16), subscript_call_result_75038)
    # Getting the type of the for loop variable (line 1241)
    for_loop_var_75039 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1241, 16), subscript_call_result_75038)
    # Assigning a type to the variable 'k' (line 1241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1241, 16), 'k', for_loop_var_75039)
    # SSA begins for a for statement (line 1241)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'k' (line 1242)
    k_75040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1242, 23), 'k')
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_75041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1242, 46), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1242)
    n_75042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1242, 43), 'n')
    # Getting the type of 'outneeds' (line 1242)
    outneeds_75043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1242, 34), 'outneeds')
    # Obtaining the member '__getitem__' of a type (line 1242)
    getitem___75044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1242, 34), outneeds_75043, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1242)
    subscript_call_result_75045 = invoke(stypy.reporting.localization.Localization(__file__, 1242, 34), getitem___75044, n_75042)
    
    # Obtaining the member '__getitem__' of a type (line 1242)
    getitem___75046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1242, 34), subscript_call_result_75045, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1242)
    subscript_call_result_75047 = invoke(stypy.reporting.localization.Localization(__file__, 1242, 34), getitem___75046, int_75041)
    
    # Getting the type of 'needs' (line 1242)
    needs_75048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1242, 28), 'needs')
    # Obtaining the member '__getitem__' of a type (line 1242)
    getitem___75049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1242, 28), needs_75048, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1242)
    subscript_call_result_75050 = invoke(stypy.reporting.localization.Localization(__file__, 1242, 28), getitem___75049, subscript_call_result_75047)
    
    # Applying the binary operator 'in' (line 1242)
    result_contains_75051 = python_operator(stypy.reporting.localization.Localization(__file__, 1242, 23), 'in', k_75040, subscript_call_result_75050)
    
    # Testing the type of an if condition (line 1242)
    if_condition_75052 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1242, 20), result_contains_75051)
    # Assigning a type to the variable 'if_condition_75052' (line 1242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1242, 20), 'if_condition_75052', if_condition_75052)
    # SSA begins for if statement (line 1242)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 1243):
    int_75053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1243, 31), 'int')
    # Assigning a type to the variable 'flag' (line 1243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1243, 24), 'flag', int_75053)
    # SSA join for if statement (line 1242)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'flag' (line 1245)
    flag_75054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1245, 19), 'flag')
    # Testing the type of an if condition (line 1245)
    if_condition_75055 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1245, 16), flag_75054)
    # Assigning a type to the variable 'if_condition_75055' (line 1245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1245, 16), 'if_condition_75055', if_condition_75055)
    # SSA begins for if statement (line 1245)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 1246):
    
    # Obtaining the type of the subscript
    int_75056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1246, 46), 'int')
    slice_75057 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1246, 34), int_75056, None, None)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1246)
    n_75058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1246, 43), 'n')
    # Getting the type of 'outneeds' (line 1246)
    outneeds_75059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1246, 34), 'outneeds')
    # Obtaining the member '__getitem__' of a type (line 1246)
    getitem___75060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1246, 34), outneeds_75059, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1246)
    subscript_call_result_75061 = invoke(stypy.reporting.localization.Localization(__file__, 1246, 34), getitem___75060, n_75058)
    
    # Obtaining the member '__getitem__' of a type (line 1246)
    getitem___75062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1246, 34), subscript_call_result_75061, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1246)
    subscript_call_result_75063 = invoke(stypy.reporting.localization.Localization(__file__, 1246, 34), getitem___75062, slice_75057)
    
    
    # Obtaining an instance of the builtin type 'list' (line 1246)
    list_75064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1246, 52), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1246)
    # Adding element type (line 1246)
    
    # Obtaining the type of the subscript
    int_75065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1246, 65), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1246)
    n_75066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1246, 62), 'n')
    # Getting the type of 'outneeds' (line 1246)
    outneeds_75067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1246, 53), 'outneeds')
    # Obtaining the member '__getitem__' of a type (line 1246)
    getitem___75068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1246, 53), outneeds_75067, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1246)
    subscript_call_result_75069 = invoke(stypy.reporting.localization.Localization(__file__, 1246, 53), getitem___75068, n_75066)
    
    # Obtaining the member '__getitem__' of a type (line 1246)
    getitem___75070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1246, 53), subscript_call_result_75069, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1246)
    subscript_call_result_75071 = invoke(stypy.reporting.localization.Localization(__file__, 1246, 53), getitem___75070, int_75065)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1246, 52), list_75064, subscript_call_result_75071)
    
    # Applying the binary operator '+' (line 1246)
    result_add_75072 = python_operator(stypy.reporting.localization.Localization(__file__, 1246, 34), '+', subscript_call_result_75063, list_75064)
    
    # Getting the type of 'outneeds' (line 1246)
    outneeds_75073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1246, 20), 'outneeds')
    # Getting the type of 'n' (line 1246)
    n_75074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1246, 29), 'n')
    # Storing an element on a container (line 1246)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1246, 20), outneeds_75073, (n_75074, result_add_75072))
    # SSA branch for the else part of an if statement (line 1245)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 1248)
    # Processing the call arguments (line 1248)
    
    # Obtaining the type of the subscript
    int_75077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1248, 43), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1248)
    n_75078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 40), 'n', False)
    # Getting the type of 'outneeds' (line 1248)
    outneeds_75079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 31), 'outneeds', False)
    # Obtaining the member '__getitem__' of a type (line 1248)
    getitem___75080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1248, 31), outneeds_75079, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1248)
    subscript_call_result_75081 = invoke(stypy.reporting.localization.Localization(__file__, 1248, 31), getitem___75080, n_75078)
    
    # Obtaining the member '__getitem__' of a type (line 1248)
    getitem___75082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1248, 31), subscript_call_result_75081, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1248)
    subscript_call_result_75083 = invoke(stypy.reporting.localization.Localization(__file__, 1248, 31), getitem___75082, int_75077)
    
    # Processing the call keyword arguments (line 1248)
    kwargs_75084 = {}
    # Getting the type of 'out' (line 1248)
    out_75075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 20), 'out', False)
    # Obtaining the member 'append' of a type (line 1248)
    append_75076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1248, 20), out_75075, 'append')
    # Calling append(args, kwargs) (line 1248)
    append_call_result_75085 = invoke(stypy.reporting.localization.Localization(__file__, 1248, 20), append_75076, *[subscript_call_result_75083], **kwargs_75084)
    
    # Deleting a member
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1249)
    n_75086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1249, 33), 'n')
    # Getting the type of 'outneeds' (line 1249)
    outneeds_75087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1249, 24), 'outneeds')
    # Obtaining the member '__getitem__' of a type (line 1249)
    getitem___75088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1249, 24), outneeds_75087, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1249)
    subscript_call_result_75089 = invoke(stypy.reporting.localization.Localization(__file__, 1249, 24), getitem___75088, n_75086)
    
    
    # Obtaining the type of the subscript
    int_75090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1249, 36), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1249)
    n_75091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1249, 33), 'n')
    # Getting the type of 'outneeds' (line 1249)
    outneeds_75092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1249, 24), 'outneeds')
    # Obtaining the member '__getitem__' of a type (line 1249)
    getitem___75093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1249, 24), outneeds_75092, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1249)
    subscript_call_result_75094 = invoke(stypy.reporting.localization.Localization(__file__, 1249, 24), getitem___75093, n_75091)
    
    # Obtaining the member '__getitem__' of a type (line 1249)
    getitem___75095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1249, 24), subscript_call_result_75094, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1249)
    subscript_call_result_75096 = invoke(stypy.reporting.localization.Localization(__file__, 1249, 24), getitem___75095, int_75090)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1249, 20), subscript_call_result_75089, subscript_call_result_75096)
    # SSA join for if statement (line 1245)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1236)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'saveout' (line 1250)
    saveout_75097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1250, 15), 'saveout')
    
    int_75098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1250, 28), 'int')
    
    # Call to map(...): (line 1250)
    # Processing the call arguments (line 1250)

    @norecursion
    def _stypy_temp_lambda_25(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_25'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_25', 1250, 41, True)
        # Passed parameters checking function
        _stypy_temp_lambda_25.stypy_localization = localization
        _stypy_temp_lambda_25.stypy_type_of_self = None
        _stypy_temp_lambda_25.stypy_type_store = module_type_store
        _stypy_temp_lambda_25.stypy_function_name = '_stypy_temp_lambda_25'
        _stypy_temp_lambda_25.stypy_param_names_list = ['x', 'y']
        _stypy_temp_lambda_25.stypy_varargs_param_name = None
        _stypy_temp_lambda_25.stypy_kwargs_param_name = None
        _stypy_temp_lambda_25.stypy_call_defaults = defaults
        _stypy_temp_lambda_25.stypy_call_varargs = varargs
        _stypy_temp_lambda_25.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_25', ['x', 'y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_25', ['x', 'y'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Getting the type of 'x' (line 1250)
        x_75100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1250, 54), 'x', False)
        # Getting the type of 'y' (line 1250)
        y_75101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1250, 59), 'y', False)
        # Applying the binary operator '==' (line 1250)
        result_eq_75102 = python_operator(stypy.reporting.localization.Localization(__file__, 1250, 54), '==', x_75100, y_75101)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 1250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1250, 41), 'stypy_return_type', result_eq_75102)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_25' in the type store
        # Getting the type of 'stypy_return_type' (line 1250)
        stypy_return_type_75103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1250, 41), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_75103)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_25'
        return stypy_return_type_75103

    # Assigning a type to the variable '_stypy_temp_lambda_25' (line 1250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1250, 41), '_stypy_temp_lambda_25', _stypy_temp_lambda_25)
    # Getting the type of '_stypy_temp_lambda_25' (line 1250)
    _stypy_temp_lambda_25_75104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1250, 41), '_stypy_temp_lambda_25')
    # Getting the type of 'saveout' (line 1250)
    saveout_75105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1250, 62), 'saveout', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1250)
    n_75106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1250, 80), 'n', False)
    # Getting the type of 'outneeds' (line 1250)
    outneeds_75107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1250, 71), 'outneeds', False)
    # Obtaining the member '__getitem__' of a type (line 1250)
    getitem___75108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1250, 71), outneeds_75107, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1250)
    subscript_call_result_75109 = invoke(stypy.reporting.localization.Localization(__file__, 1250, 71), getitem___75108, n_75106)
    
    # Processing the call keyword arguments (line 1250)
    kwargs_75110 = {}
    # Getting the type of 'map' (line 1250)
    map_75099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1250, 37), 'map', False)
    # Calling map(args, kwargs) (line 1250)
    map_call_result_75111 = invoke(stypy.reporting.localization.Localization(__file__, 1250, 37), map_75099, *[_stypy_temp_lambda_25_75104, saveout_75105, subscript_call_result_75109], **kwargs_75110)
    
    # Applying the binary operator 'notin' (line 1250)
    result_contains_75112 = python_operator(stypy.reporting.localization.Localization(__file__, 1250, 28), 'notin', int_75098, map_call_result_75111)
    
    # Applying the binary operator 'and' (line 1250)
    result_and_keyword_75113 = python_operator(stypy.reporting.localization.Localization(__file__, 1250, 15), 'and', saveout_75097, result_contains_75112)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1251)
    n_75114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1251, 33), 'n')
    # Getting the type of 'outneeds' (line 1251)
    outneeds_75115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1251, 24), 'outneeds')
    # Obtaining the member '__getitem__' of a type (line 1251)
    getitem___75116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1251, 24), outneeds_75115, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1251)
    subscript_call_result_75117 = invoke(stypy.reporting.localization.Localization(__file__, 1251, 24), getitem___75116, n_75114)
    
    
    # Obtaining an instance of the builtin type 'list' (line 1251)
    list_75118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1251, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1251)
    
    # Applying the binary operator '!=' (line 1251)
    result_ne_75119 = python_operator(stypy.reporting.localization.Localization(__file__, 1251, 24), '!=', subscript_call_result_75117, list_75118)
    
    # Applying the binary operator 'and' (line 1250)
    result_and_keyword_75120 = python_operator(stypy.reporting.localization.Localization(__file__, 1250, 15), 'and', result_and_keyword_75113, result_ne_75119)
    
    # Testing the type of an if condition (line 1250)
    if_condition_75121 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1250, 12), result_and_keyword_75120)
    # Assigning a type to the variable 'if_condition_75121' (line 1250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1250, 12), 'if_condition_75121', if_condition_75121)
    # SSA begins for if statement (line 1250)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 1252)
    # Processing the call arguments (line 1252)
    # Getting the type of 'n' (line 1252)
    n_75123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1252, 22), 'n', False)
    # Getting the type of 'saveout' (line 1252)
    saveout_75124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1252, 25), 'saveout', False)
    # Processing the call keyword arguments (line 1252)
    kwargs_75125 = {}
    # Getting the type of 'print' (line 1252)
    print_75122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1252, 16), 'print', False)
    # Calling print(args, kwargs) (line 1252)
    print_call_result_75126 = invoke(stypy.reporting.localization.Localization(__file__, 1252, 16), print_75122, *[n_75123, saveout_75124], **kwargs_75125)
    
    
    # Call to errmess(...): (line 1253)
    # Processing the call arguments (line 1253)
    str_75128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1254, 20), 'str', 'get_needs: no progress in sorting needs, probably circular dependence, skipping.\n')
    # Processing the call keyword arguments (line 1253)
    kwargs_75129 = {}
    # Getting the type of 'errmess' (line 1253)
    errmess_75127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1253, 16), 'errmess', False)
    # Calling errmess(args, kwargs) (line 1253)
    errmess_call_result_75130 = invoke(stypy.reporting.localization.Localization(__file__, 1253, 16), errmess_75127, *[str_75128], **kwargs_75129)
    
    
    # Assigning a BinOp to a Name (line 1255):
    # Getting the type of 'out' (line 1255)
    out_75131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1255, 22), 'out')
    # Getting the type of 'saveout' (line 1255)
    saveout_75132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1255, 28), 'saveout')
    # Applying the binary operator '+' (line 1255)
    result_add_75133 = python_operator(stypy.reporting.localization.Localization(__file__, 1255, 22), '+', out_75131, saveout_75132)
    
    # Assigning a type to the variable 'out' (line 1255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1255, 16), 'out', result_add_75133)
    # SSA join for if statement (line 1250)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1257):
    
    # Call to copy(...): (line 1257)
    # Processing the call arguments (line 1257)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1257)
    n_75136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1257, 41), 'n', False)
    # Getting the type of 'outneeds' (line 1257)
    outneeds_75137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1257, 32), 'outneeds', False)
    # Obtaining the member '__getitem__' of a type (line 1257)
    getitem___75138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1257, 32), outneeds_75137, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1257)
    subscript_call_result_75139 = invoke(stypy.reporting.localization.Localization(__file__, 1257, 32), getitem___75138, n_75136)
    
    # Processing the call keyword arguments (line 1257)
    kwargs_75140 = {}
    # Getting the type of 'copy' (line 1257)
    copy_75134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1257, 22), 'copy', False)
    # Obtaining the member 'copy' of a type (line 1257)
    copy_75135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1257, 22), copy_75134, 'copy')
    # Calling copy(args, kwargs) (line 1257)
    copy_call_result_75141 = invoke(stypy.reporting.localization.Localization(__file__, 1257, 22), copy_75135, *[subscript_call_result_75139], **kwargs_75140)
    
    # Assigning a type to the variable 'saveout' (line 1257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1257, 12), 'saveout', copy_call_result_75141)
    # SSA join for while statement (line 1235)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'out' (line 1258)
    out_75142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 11), 'out')
    
    # Obtaining an instance of the builtin type 'list' (line 1258)
    list_75143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1258, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1258)
    
    # Applying the binary operator '==' (line 1258)
    result_eq_75144 = python_operator(stypy.reporting.localization.Localization(__file__, 1258, 11), '==', out_75142, list_75143)
    
    # Testing the type of an if condition (line 1258)
    if_condition_75145 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1258, 8), result_eq_75144)
    # Assigning a type to the variable 'if_condition_75145' (line 1258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1258, 8), 'if_condition_75145', if_condition_75145)
    # SSA begins for if statement (line 1258)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 1259):
    
    # Obtaining an instance of the builtin type 'list' (line 1259)
    list_75146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1259, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1259)
    # Adding element type (line 1259)
    # Getting the type of 'n' (line 1259)
    n_75147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 19), 'n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1259, 18), list_75146, n_75147)
    
    # Assigning a type to the variable 'out' (line 1259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1259, 12), 'out', list_75146)
    # SSA join for if statement (line 1258)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 1260):
    # Getting the type of 'out' (line 1260)
    out_75148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1260, 17), 'out')
    # Getting the type of 'res' (line 1260)
    res_75149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1260, 8), 'res')
    # Getting the type of 'n' (line 1260)
    n_75150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1260, 12), 'n')
    # Storing an element on a container (line 1260)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1260, 8), res_75149, (n_75150, out_75148))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'res' (line 1261)
    res_75151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1261, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 1261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1261, 4), 'stypy_return_type', res_75151)
    
    # ################# End of 'get_needs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_needs' in the type store
    # Getting the type of 'stypy_return_type' (line 1229)
    stypy_return_type_75152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_75152)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_needs'
    return stypy_return_type_75152

# Assigning a type to the variable 'get_needs' (line 1229)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1229, 0), 'get_needs', get_needs)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
