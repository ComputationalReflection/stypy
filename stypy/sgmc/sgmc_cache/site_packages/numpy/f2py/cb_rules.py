
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: '''
3: 
4: Build call-back mechanism for f2py2e.
5: 
6: Copyright 2000 Pearu Peterson all rights reserved,
7: Pearu Peterson <pearu@ioc.ee>
8: Permission to use, modify, and distribute this software is given under the
9: terms of the NumPy License.
10: 
11: NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
12: $Date: 2005/07/20 11:27:58 $
13: Pearu Peterson
14: 
15: '''
16: from __future__ import division, absolute_import, print_function
17: 
18: from . import __version__
19: from .auxfuncs import (
20:     applyrules, debugcapi, dictappend, errmess, getargs, hasnote, isarray,
21:     iscomplex, iscomplexarray, iscomplexfunction, isfunction, isintent_c,
22:     isintent_hide, isintent_in, isintent_inout, isintent_nothide,
23:     isintent_out, isoptional, isrequired, isscalar, isstring,
24:     isstringfunction, issubroutine, l_and, l_not, l_or, outmess, replace,
25:     stripcomma, throw_error
26: )
27: from . import cfuncs
28: 
29: f2py_version = __version__.version
30: 
31: 
32: ################## Rules for callback function ##############
33: 
34: cb_routine_rules = {
35:     'cbtypedefs': 'typedef #rctype#(*#name#_typedef)(#optargs_td##args_td##strarglens_td##noargs#);',
36:     'body': '''
37: #begintitle#
38: PyObject *#name#_capi = NULL;/*was Py_None*/
39: PyTupleObject *#name#_args_capi = NULL;
40: int #name#_nofargs = 0;
41: jmp_buf #name#_jmpbuf;
42: /*typedef #rctype#(*#name#_typedef)(#optargs_td##args_td##strarglens_td##noargs#);*/
43: #static# #rctype# #callbackname# (#optargs##args##strarglens##noargs#) {
44: \tPyTupleObject *capi_arglist = #name#_args_capi;
45: \tPyObject *capi_return = NULL;
46: \tPyObject *capi_tmp = NULL;
47: \tint capi_j,capi_i = 0;
48: \tint capi_longjmp_ok = 1;
49: #decl#
50: #ifdef F2PY_REPORT_ATEXIT
51: f2py_cb_start_clock();
52: #endif
53: \tCFUNCSMESS(\"cb:Call-back function #name# (maxnofargs=#maxnofargs#(-#nofoptargs#))\\n\");
54: \tCFUNCSMESSPY(\"cb:#name#_capi=\",#name#_capi);
55: \tif (#name#_capi==NULL) {
56: \t\tcapi_longjmp_ok = 0;
57: \t\t#name#_capi = PyObject_GetAttrString(#modulename#_module,\"#argname#\");
58: \t}
59: \tif (#name#_capi==NULL) {
60: \t\tPyErr_SetString(#modulename#_error,\"cb: Callback #argname# not defined (as an argument or module #modulename# attribute).\\n\");
61: \t\tgoto capi_fail;
62: \t}
63: \tif (F2PyCapsule_Check(#name#_capi)) {
64: \t#name#_typedef #name#_cptr;
65: \t#name#_cptr = F2PyCapsule_AsVoidPtr(#name#_capi);
66: \t#returncptr#(*#name#_cptr)(#optargs_nm##args_nm##strarglens_nm#);
67: \t#return#
68: \t}
69: \tif (capi_arglist==NULL) {
70: \t\tcapi_longjmp_ok = 0;
71: \t\tcapi_tmp = PyObject_GetAttrString(#modulename#_module,\"#argname#_extra_args\");
72: \t\tif (capi_tmp) {
73: \t\t\tcapi_arglist = (PyTupleObject *)PySequence_Tuple(capi_tmp);
74: \t\t\tif (capi_arglist==NULL) {
75: \t\t\t\tPyErr_SetString(#modulename#_error,\"Failed to convert #modulename#.#argname#_extra_args to tuple.\\n\");
76: \t\t\t\tgoto capi_fail;
77: \t\t\t}
78: \t\t} else {
79: \t\t\tPyErr_Clear();
80: \t\t\tcapi_arglist = (PyTupleObject *)Py_BuildValue(\"()\");
81: \t\t}
82: \t}
83: \tif (capi_arglist == NULL) {
84: \t\tPyErr_SetString(#modulename#_error,\"Callback #argname# argument list is not set.\\n\");
85: \t\tgoto capi_fail;
86: \t}
87: #setdims#
88: #pyobjfrom#
89: \tCFUNCSMESSPY(\"cb:capi_arglist=\",capi_arglist);
90: \tCFUNCSMESS(\"cb:Call-back calling Python function #argname#.\\n\");
91: #ifdef F2PY_REPORT_ATEXIT
92: f2py_cb_start_call_clock();
93: #endif
94: \tcapi_return = PyObject_CallObject(#name#_capi,(PyObject *)capi_arglist);
95: #ifdef F2PY_REPORT_ATEXIT
96: f2py_cb_stop_call_clock();
97: #endif
98: \tCFUNCSMESSPY(\"cb:capi_return=\",capi_return);
99: \tif (capi_return == NULL) {
100: \t\tfprintf(stderr,\"capi_return is NULL\\n\");
101: \t\tgoto capi_fail;
102: \t}
103: \tif (capi_return == Py_None) {
104: \t\tPy_DECREF(capi_return);
105: \t\tcapi_return = Py_BuildValue(\"()\");
106: \t}
107: \telse if (!PyTuple_Check(capi_return)) {
108: \t\tcapi_return = Py_BuildValue(\"(N)\",capi_return);
109: \t}
110: \tcapi_j = PyTuple_Size(capi_return);
111: \tcapi_i = 0;
112: #frompyobj#
113: \tCFUNCSMESS(\"cb:#name#:successful\\n\");
114: \tPy_DECREF(capi_return);
115: #ifdef F2PY_REPORT_ATEXIT
116: f2py_cb_stop_clock();
117: #endif
118: \tgoto capi_return_pt;
119: capi_fail:
120: \tfprintf(stderr,\"Call-back #name# failed.\\n\");
121: \tPy_XDECREF(capi_return);
122: \tif (capi_longjmp_ok)
123: \t\tlongjmp(#name#_jmpbuf,-1);
124: capi_return_pt:
125: \t;
126: #return#
127: }
128: #endtitle#
129: ''',
130:     'need': ['setjmp.h', 'CFUNCSMESS'],
131:     'maxnofargs': '#maxnofargs#',
132:     'nofoptargs': '#nofoptargs#',
133:     'docstr': '''\
134: \tdef #argname#(#docsignature#): return #docreturn#\\n\\
135: #docstrsigns#''',
136:     'latexdocstr': '''
137: {{}\\verb@def #argname#(#latexdocsignature#): return #docreturn#@{}}
138: #routnote#
139: 
140: #latexdocstrsigns#''',
141:     'docstrshort': 'def #argname#(#docsignature#): return #docreturn#'
142: }
143: cb_rout_rules = [
144:     {  # Init
145:         'separatorsfor': {'decl': '\n',
146:                           'args': ',', 'optargs': '', 'pyobjfrom': '\n', 'freemem': '\n',
147:                           'args_td': ',', 'optargs_td': '',
148:                           'args_nm': ',', 'optargs_nm': '',
149:                           'frompyobj': '\n', 'setdims': '\n',
150:                           'docstrsigns': '\\n"\n"',
151:                           'latexdocstrsigns': '\n',
152:                           'latexdocstrreq': '\n', 'latexdocstropt': '\n',
153:                           'latexdocstrout': '\n', 'latexdocstrcbs': '\n',
154:                           },
155:         'decl': '/*decl*/', 'pyobjfrom': '/*pyobjfrom*/', 'frompyobj': '/*frompyobj*/',
156:         'args': [], 'optargs': '', 'return': '', 'strarglens': '', 'freemem': '/*freemem*/',
157:         'args_td': [], 'optargs_td': '', 'strarglens_td': '',
158:         'args_nm': [], 'optargs_nm': '', 'strarglens_nm': '',
159:         'noargs': '',
160:         'setdims': '/*setdims*/',
161:         'docstrsigns': '', 'latexdocstrsigns': '',
162:         'docstrreq': '\tRequired arguments:',
163:         'docstropt': '\tOptional arguments:',
164:         'docstrout': '\tReturn objects:',
165:         'docstrcbs': '\tCall-back functions:',
166:         'docreturn': '', 'docsign': '', 'docsignopt': '',
167:         'latexdocstrreq': '\\noindent Required arguments:',
168:         'latexdocstropt': '\\noindent Optional arguments:',
169:         'latexdocstrout': '\\noindent Return objects:',
170:         'latexdocstrcbs': '\\noindent Call-back functions:',
171:         'routnote': {hasnote: '--- #note#', l_not(hasnote): ''},
172:     }, {  # Function
173:         'decl': '\t#ctype# return_value;',
174:         'frompyobj': [{debugcapi: '\tCFUNCSMESS("cb:Getting return_value->");'},
175:                       '\tif (capi_j>capi_i)\n\t\tGETSCALARFROMPYTUPLE(capi_return,capi_i++,&return_value,#ctype#,"#ctype#_from_pyobj failed in converting return_value of call-back function #name# to C #ctype#\\n");',
176:                       {debugcapi:
177:                        '\tfprintf(stderr,"#showvalueformat#.\\n",return_value);'}
178:                       ],
179:         'need': ['#ctype#_from_pyobj', {debugcapi: 'CFUNCSMESS'}, 'GETSCALARFROMPYTUPLE'],
180:         'return': '\treturn return_value;',
181:         '_check': l_and(isfunction, l_not(isstringfunction), l_not(iscomplexfunction))
182:     },
183:     {  # String function
184:         'pyobjfrom': {debugcapi: '\tfprintf(stderr,"debug-capi:cb:#name#:%d:\\n",return_value_len);'},
185:         'args': '#ctype# return_value,int return_value_len',
186:         'args_nm': 'return_value,&return_value_len',
187:         'args_td': '#ctype# ,int',
188:         'frompyobj': [{debugcapi: '\tCFUNCSMESS("cb:Getting return_value->\\"");'},
189:                       '''\tif (capi_j>capi_i)
190: \t\tGETSTRFROMPYTUPLE(capi_return,capi_i++,return_value,return_value_len);''',
191:                       {debugcapi:
192:                        '\tfprintf(stderr,"#showvalueformat#\\".\\n",return_value);'}
193:                       ],
194:         'need': ['#ctype#_from_pyobj', {debugcapi: 'CFUNCSMESS'},
195:                  'string.h', 'GETSTRFROMPYTUPLE'],
196:         'return': 'return;',
197:         '_check': isstringfunction
198:     },
199:     {  # Complex function
200:         'optargs': '''
201: #ifndef F2PY_CB_RETURNCOMPLEX
202: #ctype# *return_value
203: #endif
204: ''',
205:         'optargs_nm': '''
206: #ifndef F2PY_CB_RETURNCOMPLEX
207: return_value
208: #endif
209: ''',
210:         'optargs_td': '''
211: #ifndef F2PY_CB_RETURNCOMPLEX
212: #ctype# *
213: #endif
214: ''',
215:         'decl': '''
216: #ifdef F2PY_CB_RETURNCOMPLEX
217: \t#ctype# return_value;
218: #endif
219: ''',
220:         'frompyobj': [{debugcapi: '\tCFUNCSMESS("cb:Getting return_value->");'},
221:                       '''\
222: \tif (capi_j>capi_i)
223: #ifdef F2PY_CB_RETURNCOMPLEX
224: \t\tGETSCALARFROMPYTUPLE(capi_return,capi_i++,&return_value,#ctype#,\"#ctype#_from_pyobj failed in converting return_value of call-back function #name# to C #ctype#\\n\");
225: #else
226: \t\tGETSCALARFROMPYTUPLE(capi_return,capi_i++,return_value,#ctype#,\"#ctype#_from_pyobj failed in converting return_value of call-back function #name# to C #ctype#\\n\");
227: #endif
228: ''',
229:                       {debugcapi: '''
230: #ifdef F2PY_CB_RETURNCOMPLEX
231: \tfprintf(stderr,\"#showvalueformat#.\\n\",(return_value).r,(return_value).i);
232: #else
233: \tfprintf(stderr,\"#showvalueformat#.\\n\",(*return_value).r,(*return_value).i);
234: #endif
235: 
236: '''}
237:                       ],
238:         'return': '''
239: #ifdef F2PY_CB_RETURNCOMPLEX
240: \treturn return_value;
241: #else
242: \treturn;
243: #endif
244: ''',
245:         'need': ['#ctype#_from_pyobj', {debugcapi: 'CFUNCSMESS'},
246:                  'string.h', 'GETSCALARFROMPYTUPLE', '#ctype#'],
247:         '_check': iscomplexfunction
248:     },
249:     {'docstrout': '\t\t#pydocsignout#',
250:      'latexdocstrout': ['\\item[]{{}\\verb@#pydocsignout#@{}}',
251:                         {hasnote: '--- #note#'}],
252:      'docreturn': '#rname#,',
253:      '_check': isfunction},
254:     {'_check': issubroutine, 'return': 'return;'}
255: ]
256: 
257: cb_arg_rules = [
258:     {  # Doc
259:         'docstropt': {l_and(isoptional, isintent_nothide): '\t\t#pydocsign#'},
260:         'docstrreq': {l_and(isrequired, isintent_nothide): '\t\t#pydocsign#'},
261:         'docstrout': {isintent_out: '\t\t#pydocsignout#'},
262:         'latexdocstropt': {l_and(isoptional, isintent_nothide): ['\\item[]{{}\\verb@#pydocsign#@{}}',
263:                                                                  {hasnote: '--- #note#'}]},
264:         'latexdocstrreq': {l_and(isrequired, isintent_nothide): ['\\item[]{{}\\verb@#pydocsign#@{}}',
265:                                                                  {hasnote: '--- #note#'}]},
266:         'latexdocstrout': {isintent_out: ['\\item[]{{}\\verb@#pydocsignout#@{}}',
267:                                           {l_and(hasnote, isintent_hide): '--- #note#',
268:                                            l_and(hasnote, isintent_nothide): '--- See above.'}]},
269:         'docsign': {l_and(isrequired, isintent_nothide): '#varname#,'},
270:         'docsignopt': {l_and(isoptional, isintent_nothide): '#varname#,'},
271:         'depend': ''
272:     },
273:     {
274:         'args': {
275:             l_and(isscalar, isintent_c): '#ctype# #varname_i#',
276:             l_and(isscalar, l_not(isintent_c)): '#ctype# *#varname_i#_cb_capi',
277:             isarray: '#ctype# *#varname_i#',
278:             isstring: '#ctype# #varname_i#'
279:         },
280:         'args_nm': {
281:             l_and(isscalar, isintent_c): '#varname_i#',
282:             l_and(isscalar, l_not(isintent_c)): '#varname_i#_cb_capi',
283:             isarray: '#varname_i#',
284:             isstring: '#varname_i#'
285:         },
286:         'args_td': {
287:             l_and(isscalar, isintent_c): '#ctype#',
288:             l_and(isscalar, l_not(isintent_c)): '#ctype# *',
289:             isarray: '#ctype# *',
290:             isstring: '#ctype#'
291:         },
292:         # untested with multiple args
293:         'strarglens': {isstring: ',int #varname_i#_cb_len'},
294:         'strarglens_td': {isstring: ',int'},  # untested with multiple args
295:         # untested with multiple args
296:         'strarglens_nm': {isstring: ',#varname_i#_cb_len'},
297:     },
298:     {  # Scalars
299:         'decl': {l_not(isintent_c): '\t#ctype# #varname_i#=(*#varname_i#_cb_capi);'},
300:         'error': {l_and(isintent_c, isintent_out,
301:                         throw_error('intent(c,out) is forbidden for callback scalar arguments')):
302:                   ''},
303:         'frompyobj': [{debugcapi: '\tCFUNCSMESS("cb:Getting #varname#->");'},
304:                       {isintent_out:
305:                        '\tif (capi_j>capi_i)\n\t\tGETSCALARFROMPYTUPLE(capi_return,capi_i++,#varname_i#_cb_capi,#ctype#,"#ctype#_from_pyobj failed in converting argument #varname# of call-back function #name# to C #ctype#\\n");'},
306:                       {l_and(debugcapi, l_and(l_not(iscomplex), isintent_c)):
307:                           '\tfprintf(stderr,"#showvalueformat#.\\n",#varname_i#);'},
308:                       {l_and(debugcapi, l_and(l_not(iscomplex), l_not( isintent_c))):
309:                           '\tfprintf(stderr,"#showvalueformat#.\\n",*#varname_i#_cb_capi);'},
310:                       {l_and(debugcapi, l_and(iscomplex, isintent_c)):
311:                           '\tfprintf(stderr,"#showvalueformat#.\\n",(#varname_i#).r,(#varname_i#).i);'},
312:                       {l_and(debugcapi, l_and(iscomplex, l_not( isintent_c))):
313:                           '\tfprintf(stderr,"#showvalueformat#.\\n",(*#varname_i#_cb_capi).r,(*#varname_i#_cb_capi).i);'},
314:                       ],
315:         'need': [{isintent_out: ['#ctype#_from_pyobj', 'GETSCALARFROMPYTUPLE']},
316:                  {debugcapi: 'CFUNCSMESS'}],
317:         '_check': isscalar
318:     }, {
319:         'pyobjfrom': [{isintent_in: '''\
320: \tif (#name#_nofargs>capi_i)
321: \t\tif (PyTuple_SetItem((PyObject *)capi_arglist,capi_i++,pyobj_from_#ctype#1(#varname_i#)))
322: \t\t\tgoto capi_fail;'''},
323:                       {isintent_inout: '''\
324: \tif (#name#_nofargs>capi_i)
325: \t\tif (PyTuple_SetItem((PyObject *)capi_arglist,capi_i++,pyarr_from_p_#ctype#1(#varname_i#_cb_capi)))
326: \t\t\tgoto capi_fail;'''}],
327:         'need': [{isintent_in: 'pyobj_from_#ctype#1'},
328:                  {isintent_inout: 'pyarr_from_p_#ctype#1'},
329:                  {iscomplex: '#ctype#'}],
330:         '_check': l_and(isscalar, isintent_nothide),
331:         '_optional': ''
332:     }, {  # String
333:         'frompyobj': [{debugcapi: '\tCFUNCSMESS("cb:Getting #varname#->\\"");'},
334:                       '''\tif (capi_j>capi_i)
335: \t\tGETSTRFROMPYTUPLE(capi_return,capi_i++,#varname_i#,#varname_i#_cb_len);''',
336:                       {debugcapi:
337:                        '\tfprintf(stderr,"#showvalueformat#\\":%d:.\\n",#varname_i#,#varname_i#_cb_len);'},
338:                       ],
339:         'need': ['#ctype#', 'GETSTRFROMPYTUPLE',
340:                  {debugcapi: 'CFUNCSMESS'}, 'string.h'],
341:         '_check': l_and(isstring, isintent_out)
342:     }, {
343:         'pyobjfrom': [{debugcapi: '\tfprintf(stderr,"debug-capi:cb:#varname#=\\"#showvalueformat#\\":%d:\\n",#varname_i#,#varname_i#_cb_len);'},
344:                       {isintent_in: '''\
345: \tif (#name#_nofargs>capi_i)
346: \t\tif (PyTuple_SetItem((PyObject *)capi_arglist,capi_i++,pyobj_from_#ctype#1size(#varname_i#,#varname_i#_cb_len)))
347: \t\t\tgoto capi_fail;'''},
348:                       {isintent_inout: '''\
349: \tif (#name#_nofargs>capi_i) {
350: \t\tint #varname_i#_cb_dims[] = {#varname_i#_cb_len};
351: \t\tif (PyTuple_SetItem((PyObject *)capi_arglist,capi_i++,pyarr_from_p_#ctype#1(#varname_i#,#varname_i#_cb_dims)))
352: \t\t\tgoto capi_fail;
353: \t}'''}],
354:         'need': [{isintent_in: 'pyobj_from_#ctype#1size'},
355:                  {isintent_inout: 'pyarr_from_p_#ctype#1'}],
356:         '_check': l_and(isstring, isintent_nothide),
357:         '_optional': ''
358:     },
359:     # Array ...
360:     {
361:         'decl': '\tnpy_intp #varname_i#_Dims[#rank#] = {#rank*[-1]#};',
362:         'setdims': '\t#cbsetdims#;',
363:         '_check': isarray,
364:         '_depend': ''
365:     },
366:     {
367:         'pyobjfrom': [{debugcapi: '\tfprintf(stderr,"debug-capi:cb:#varname#\\n");'},
368:                       {isintent_c: '''\
369: \tif (#name#_nofargs>capi_i) {
370: \t\tPyArrayObject *tmp_arr = (PyArrayObject *)PyArray_New(&PyArray_Type,#rank#,#varname_i#_Dims,#atype#,NULL,(char*)#varname_i#,0,NPY_ARRAY_CARRAY,NULL); /*XXX: Hmm, what will destroy this array??? */
371: ''',
372:                        l_not(isintent_c): '''\
373: \tif (#name#_nofargs>capi_i) {
374: \t\tPyArrayObject *tmp_arr = (PyArrayObject *)PyArray_New(&PyArray_Type,#rank#,#varname_i#_Dims,#atype#,NULL,(char*)#varname_i#,0,NPY_ARRAY_FARRAY,NULL); /*XXX: Hmm, what will destroy this array??? */
375: ''',
376:                        },
377:                       '''
378: \t\tif (tmp_arr==NULL)
379: \t\t\tgoto capi_fail;
380: \t\tif (PyTuple_SetItem((PyObject *)capi_arglist,capi_i++,(PyObject *)tmp_arr))
381: \t\t\tgoto capi_fail;
382: }'''],
383:         '_check': l_and(isarray, isintent_nothide, l_or(isintent_in, isintent_inout)),
384:         '_optional': '',
385:     }, {
386:         'frompyobj': [{debugcapi: '\tCFUNCSMESS("cb:Getting #varname#->");'},
387:                       '''\tif (capi_j>capi_i) {
388: \t\tPyArrayObject *rv_cb_arr = NULL;
389: \t\tif ((capi_tmp = PyTuple_GetItem(capi_return,capi_i++))==NULL) goto capi_fail;
390: \t\trv_cb_arr =  array_from_pyobj(#atype#,#varname_i#_Dims,#rank#,F2PY_INTENT_IN''',
391:                       {isintent_c: '|F2PY_INTENT_C'},
392:                       ''',capi_tmp);
393: \t\tif (rv_cb_arr == NULL) {
394: \t\t\tfprintf(stderr,\"rv_cb_arr is NULL\\n\");
395: \t\t\tgoto capi_fail;
396: \t\t}
397: \t\tMEMCOPY(#varname_i#,PyArray_DATA(rv_cb_arr),PyArray_NBYTES(rv_cb_arr));
398: \t\tif (capi_tmp != (PyObject *)rv_cb_arr) {
399: \t\t\tPy_DECREF(rv_cb_arr);
400: \t\t}
401: \t}''',
402:                       {debugcapi: '\tfprintf(stderr,"<-.\\n");'},
403:                       ],
404:         'need': ['MEMCOPY', {iscomplexarray: '#ctype#'}],
405:         '_check': l_and(isarray, isintent_out)
406:     }, {
407:         'docreturn': '#varname#,',
408:         '_check': isintent_out
409:     }
410: ]
411: 
412: ################## Build call-back module #############
413: cb_map = {}
414: 
415: 
416: def buildcallbacks(m):
417:     global cb_map
418:     cb_map[m['name']] = []
419:     for bi in m['body']:
420:         if bi['block'] == 'interface':
421:             for b in bi['body']:
422:                 if b:
423:                     buildcallback(b, m['name'])
424:                 else:
425:                     errmess('warning: empty body for %s\n' % (m['name']))
426: 
427: 
428: def buildcallback(rout, um):
429:     global cb_map
430:     from . import capi_maps
431: 
432:     outmess('\tConstructing call-back function "cb_%s_in_%s"\n' %
433:             (rout['name'], um))
434:     args, depargs = getargs(rout)
435:     capi_maps.depargs = depargs
436:     var = rout['vars']
437:     vrd = capi_maps.cb_routsign2map(rout, um)
438:     rd = dictappend({}, vrd)
439:     cb_map[um].append([rout['name'], rd['name']])
440:     for r in cb_rout_rules:
441:         if ('_check' in r and r['_check'](rout)) or ('_check' not in r):
442:             ar = applyrules(r, vrd, rout)
443:             rd = dictappend(rd, ar)
444:     savevrd = {}
445:     for i, a in enumerate(args):
446:         vrd = capi_maps.cb_sign2map(a, var[a], index=i)
447:         savevrd[a] = vrd
448:         for r in cb_arg_rules:
449:             if '_depend' in r:
450:                 continue
451:             if '_optional' in r and isoptional(var[a]):
452:                 continue
453:             if ('_check' in r and r['_check'](var[a])) or ('_check' not in r):
454:                 ar = applyrules(r, vrd, var[a])
455:                 rd = dictappend(rd, ar)
456:                 if '_break' in r:
457:                     break
458:     for a in args:
459:         vrd = savevrd[a]
460:         for r in cb_arg_rules:
461:             if '_depend' in r:
462:                 continue
463:             if ('_optional' not in r) or ('_optional' in r and isrequired(var[a])):
464:                 continue
465:             if ('_check' in r and r['_check'](var[a])) or ('_check' not in r):
466:                 ar = applyrules(r, vrd, var[a])
467:                 rd = dictappend(rd, ar)
468:                 if '_break' in r:
469:                     break
470:     for a in depargs:
471:         vrd = savevrd[a]
472:         for r in cb_arg_rules:
473:             if '_depend' not in r:
474:                 continue
475:             if '_optional' in r:
476:                 continue
477:             if ('_check' in r and r['_check'](var[a])) or ('_check' not in r):
478:                 ar = applyrules(r, vrd, var[a])
479:                 rd = dictappend(rd, ar)
480:                 if '_break' in r:
481:                     break
482:     if 'args' in rd and 'optargs' in rd:
483:         if isinstance(rd['optargs'], list):
484:             rd['optargs'] = rd['optargs'] + ['''
485: #ifndef F2PY_CB_RETURNCOMPLEX
486: ,
487: #endif
488: ''']
489:             rd['optargs_nm'] = rd['optargs_nm'] + ['''
490: #ifndef F2PY_CB_RETURNCOMPLEX
491: ,
492: #endif
493: ''']
494:             rd['optargs_td'] = rd['optargs_td'] + ['''
495: #ifndef F2PY_CB_RETURNCOMPLEX
496: ,
497: #endif
498: ''']
499:     if isinstance(rd['docreturn'], list):
500:         rd['docreturn'] = stripcomma(
501:             replace('#docreturn#', {'docreturn': rd['docreturn']}))
502:     optargs = stripcomma(replace('#docsignopt#',
503:                                  {'docsignopt': rd['docsignopt']}
504:                                  ))
505:     if optargs == '':
506:         rd['docsignature'] = stripcomma(
507:             replace('#docsign#', {'docsign': rd['docsign']}))
508:     else:
509:         rd['docsignature'] = replace('#docsign#[#docsignopt#]',
510:                                      {'docsign': rd['docsign'],
511:                                       'docsignopt': optargs,
512:                                       })
513:     rd['latexdocsignature'] = rd['docsignature'].replace('_', '\\_')
514:     rd['latexdocsignature'] = rd['latexdocsignature'].replace(',', ', ')
515:     rd['docstrsigns'] = []
516:     rd['latexdocstrsigns'] = []
517:     for k in ['docstrreq', 'docstropt', 'docstrout', 'docstrcbs']:
518:         if k in rd and isinstance(rd[k], list):
519:             rd['docstrsigns'] = rd['docstrsigns'] + rd[k]
520:         k = 'latex' + k
521:         if k in rd and isinstance(rd[k], list):
522:             rd['latexdocstrsigns'] = rd['latexdocstrsigns'] + rd[k][0:1] +\
523:                 ['\\begin{description}'] + rd[k][1:] +\
524:                 ['\\end{description}']
525:     if 'args' not in rd:
526:         rd['args'] = ''
527:         rd['args_td'] = ''
528:         rd['args_nm'] = ''
529:     if not (rd.get('args') or rd.get('optargs') or rd.get('strarglens')):
530:         rd['noargs'] = 'void'
531: 
532:     ar = applyrules(cb_routine_rules, rd)
533:     cfuncs.callbacks[rd['name']] = ar['body']
534:     if isinstance(ar['need'], str):
535:         ar['need'] = [ar['need']]
536: 
537:     if 'need' in rd:
538:         for t in cfuncs.typedefs.keys():
539:             if t in rd['need']:
540:                 ar['need'].append(t)
541: 
542:     cfuncs.typedefs_generated[rd['name'] + '_typedef'] = ar['cbtypedefs']
543:     ar['need'].append(rd['name'] + '_typedef')
544:     cfuncs.needs[rd['name']] = ar['need']
545: 
546:     capi_maps.lcb2_map[rd['name']] = {'maxnofargs': ar['maxnofargs'],
547:                                       'nofoptargs': ar['nofoptargs'],
548:                                       'docstr': ar['docstr'],
549:                                       'latexdocstr': ar['latexdocstr'],
550:                                       'argname': rd['argname']
551:                                       }
552:     outmess('\t  %s\n' % (ar['docstrshort']))
553:     return
554: ################## Build call-back function #############
555: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_72840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, (-1)), 'str', '\n\nBuild call-back mechanism for f2py2e.\n\nCopyright 2000 Pearu Peterson all rights reserved,\nPearu Peterson <pearu@ioc.ee>\nPermission to use, modify, and distribute this software is given under the\nterms of the NumPy License.\n\nNO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.\n$Date: 2005/07/20 11:27:58 $\nPearu Peterson\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from numpy.f2py import __version__' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_72841 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.f2py')

if (type(import_72841) is not StypyTypeError):

    if (import_72841 != 'pyd_module'):
        __import__(import_72841)
        sys_modules_72842 = sys.modules[import_72841]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.f2py', sys_modules_72842.module_type_store, module_type_store, ['__version__'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_72842, sys_modules_72842.module_type_store, module_type_store)
    else:
        from numpy.f2py import __version__

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.f2py', None, module_type_store, ['__version__'], [__version__])

else:
    # Assigning a type to the variable 'numpy.f2py' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.f2py', import_72841)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from numpy.f2py.auxfuncs import applyrules, debugcapi, dictappend, errmess, getargs, hasnote, isarray, iscomplex, iscomplexarray, iscomplexfunction, isfunction, isintent_c, isintent_hide, isintent_in, isintent_inout, isintent_nothide, isintent_out, isoptional, isrequired, isscalar, isstring, isstringfunction, issubroutine, l_and, l_not, l_or, outmess, replace, stripcomma, throw_error' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_72843 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.f2py.auxfuncs')

if (type(import_72843) is not StypyTypeError):

    if (import_72843 != 'pyd_module'):
        __import__(import_72843)
        sys_modules_72844 = sys.modules[import_72843]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.f2py.auxfuncs', sys_modules_72844.module_type_store, module_type_store, ['applyrules', 'debugcapi', 'dictappend', 'errmess', 'getargs', 'hasnote', 'isarray', 'iscomplex', 'iscomplexarray', 'iscomplexfunction', 'isfunction', 'isintent_c', 'isintent_hide', 'isintent_in', 'isintent_inout', 'isintent_nothide', 'isintent_out', 'isoptional', 'isrequired', 'isscalar', 'isstring', 'isstringfunction', 'issubroutine', 'l_and', 'l_not', 'l_or', 'outmess', 'replace', 'stripcomma', 'throw_error'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_72844, sys_modules_72844.module_type_store, module_type_store)
    else:
        from numpy.f2py.auxfuncs import applyrules, debugcapi, dictappend, errmess, getargs, hasnote, isarray, iscomplex, iscomplexarray, iscomplexfunction, isfunction, isintent_c, isintent_hide, isintent_in, isintent_inout, isintent_nothide, isintent_out, isoptional, isrequired, isscalar, isstring, isstringfunction, issubroutine, l_and, l_not, l_or, outmess, replace, stripcomma, throw_error

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.f2py.auxfuncs', None, module_type_store, ['applyrules', 'debugcapi', 'dictappend', 'errmess', 'getargs', 'hasnote', 'isarray', 'iscomplex', 'iscomplexarray', 'iscomplexfunction', 'isfunction', 'isintent_c', 'isintent_hide', 'isintent_in', 'isintent_inout', 'isintent_nothide', 'isintent_out', 'isoptional', 'isrequired', 'isscalar', 'isstring', 'isstringfunction', 'issubroutine', 'l_and', 'l_not', 'l_or', 'outmess', 'replace', 'stripcomma', 'throw_error'], [applyrules, debugcapi, dictappend, errmess, getargs, hasnote, isarray, iscomplex, iscomplexarray, iscomplexfunction, isfunction, isintent_c, isintent_hide, isintent_in, isintent_inout, isintent_nothide, isintent_out, isoptional, isrequired, isscalar, isstring, isstringfunction, issubroutine, l_and, l_not, l_or, outmess, replace, stripcomma, throw_error])

else:
    # Assigning a type to the variable 'numpy.f2py.auxfuncs' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.f2py.auxfuncs', import_72843)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'from numpy.f2py import cfuncs' statement (line 27)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_72845 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.f2py')

if (type(import_72845) is not StypyTypeError):

    if (import_72845 != 'pyd_module'):
        __import__(import_72845)
        sys_modules_72846 = sys.modules[import_72845]
        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.f2py', sys_modules_72846.module_type_store, module_type_store, ['cfuncs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 27, 0), __file__, sys_modules_72846, sys_modules_72846.module_type_store, module_type_store)
    else:
        from numpy.f2py import cfuncs

        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.f2py', None, module_type_store, ['cfuncs'], [cfuncs])

else:
    # Assigning a type to the variable 'numpy.f2py' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.f2py', import_72845)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')


# Assigning a Attribute to a Name (line 29):

# Assigning a Attribute to a Name (line 29):
# Getting the type of '__version__' (line 29)
version___72847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), '__version__')
# Obtaining the member 'version' of a type (line 29)
version_72848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 15), version___72847, 'version')
# Assigning a type to the variable 'f2py_version' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'f2py_version', version_72848)

# Assigning a Dict to a Name (line 34):

# Assigning a Dict to a Name (line 34):

# Obtaining an instance of the builtin type 'dict' (line 34)
dict_72849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 34)
# Adding element type (key, value) (line 34)
str_72850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 4), 'str', 'cbtypedefs')
str_72851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 18), 'str', 'typedef #rctype#(*#name#_typedef)(#optargs_td##args_td##strarglens_td##noargs#);')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 19), dict_72849, (str_72850, str_72851))
# Adding element type (key, value) (line 34)
str_72852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 4), 'str', 'body')
str_72853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, (-1)), 'str', '\n#begintitle#\nPyObject *#name#_capi = NULL;/*was Py_None*/\nPyTupleObject *#name#_args_capi = NULL;\nint #name#_nofargs = 0;\njmp_buf #name#_jmpbuf;\n/*typedef #rctype#(*#name#_typedef)(#optargs_td##args_td##strarglens_td##noargs#);*/\n#static# #rctype# #callbackname# (#optargs##args##strarglens##noargs#) {\n\tPyTupleObject *capi_arglist = #name#_args_capi;\n\tPyObject *capi_return = NULL;\n\tPyObject *capi_tmp = NULL;\n\tint capi_j,capi_i = 0;\n\tint capi_longjmp_ok = 1;\n#decl#\n#ifdef F2PY_REPORT_ATEXIT\nf2py_cb_start_clock();\n#endif\n\tCFUNCSMESS("cb:Call-back function #name# (maxnofargs=#maxnofargs#(-#nofoptargs#))\\n");\n\tCFUNCSMESSPY("cb:#name#_capi=",#name#_capi);\n\tif (#name#_capi==NULL) {\n\t\tcapi_longjmp_ok = 0;\n\t\t#name#_capi = PyObject_GetAttrString(#modulename#_module,"#argname#");\n\t}\n\tif (#name#_capi==NULL) {\n\t\tPyErr_SetString(#modulename#_error,"cb: Callback #argname# not defined (as an argument or module #modulename# attribute).\\n");\n\t\tgoto capi_fail;\n\t}\n\tif (F2PyCapsule_Check(#name#_capi)) {\n\t#name#_typedef #name#_cptr;\n\t#name#_cptr = F2PyCapsule_AsVoidPtr(#name#_capi);\n\t#returncptr#(*#name#_cptr)(#optargs_nm##args_nm##strarglens_nm#);\n\t#return#\n\t}\n\tif (capi_arglist==NULL) {\n\t\tcapi_longjmp_ok = 0;\n\t\tcapi_tmp = PyObject_GetAttrString(#modulename#_module,"#argname#_extra_args");\n\t\tif (capi_tmp) {\n\t\t\tcapi_arglist = (PyTupleObject *)PySequence_Tuple(capi_tmp);\n\t\t\tif (capi_arglist==NULL) {\n\t\t\t\tPyErr_SetString(#modulename#_error,"Failed to convert #modulename#.#argname#_extra_args to tuple.\\n");\n\t\t\t\tgoto capi_fail;\n\t\t\t}\n\t\t} else {\n\t\t\tPyErr_Clear();\n\t\t\tcapi_arglist = (PyTupleObject *)Py_BuildValue("()");\n\t\t}\n\t}\n\tif (capi_arglist == NULL) {\n\t\tPyErr_SetString(#modulename#_error,"Callback #argname# argument list is not set.\\n");\n\t\tgoto capi_fail;\n\t}\n#setdims#\n#pyobjfrom#\n\tCFUNCSMESSPY("cb:capi_arglist=",capi_arglist);\n\tCFUNCSMESS("cb:Call-back calling Python function #argname#.\\n");\n#ifdef F2PY_REPORT_ATEXIT\nf2py_cb_start_call_clock();\n#endif\n\tcapi_return = PyObject_CallObject(#name#_capi,(PyObject *)capi_arglist);\n#ifdef F2PY_REPORT_ATEXIT\nf2py_cb_stop_call_clock();\n#endif\n\tCFUNCSMESSPY("cb:capi_return=",capi_return);\n\tif (capi_return == NULL) {\n\t\tfprintf(stderr,"capi_return is NULL\\n");\n\t\tgoto capi_fail;\n\t}\n\tif (capi_return == Py_None) {\n\t\tPy_DECREF(capi_return);\n\t\tcapi_return = Py_BuildValue("()");\n\t}\n\telse if (!PyTuple_Check(capi_return)) {\n\t\tcapi_return = Py_BuildValue("(N)",capi_return);\n\t}\n\tcapi_j = PyTuple_Size(capi_return);\n\tcapi_i = 0;\n#frompyobj#\n\tCFUNCSMESS("cb:#name#:successful\\n");\n\tPy_DECREF(capi_return);\n#ifdef F2PY_REPORT_ATEXIT\nf2py_cb_stop_clock();\n#endif\n\tgoto capi_return_pt;\ncapi_fail:\n\tfprintf(stderr,"Call-back #name# failed.\\n");\n\tPy_XDECREF(capi_return);\n\tif (capi_longjmp_ok)\n\t\tlongjmp(#name#_jmpbuf,-1);\ncapi_return_pt:\n\t;\n#return#\n}\n#endtitle#\n')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 19), dict_72849, (str_72852, str_72853))
# Adding element type (key, value) (line 34)
str_72854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 4), 'str', 'need')

# Obtaining an instance of the builtin type 'list' (line 130)
list_72855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 130)
# Adding element type (line 130)
str_72856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 13), 'str', 'setjmp.h')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 12), list_72855, str_72856)
# Adding element type (line 130)
str_72857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 25), 'str', 'CFUNCSMESS')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 12), list_72855, str_72857)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 19), dict_72849, (str_72854, list_72855))
# Adding element type (key, value) (line 34)
str_72858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 4), 'str', 'maxnofargs')
str_72859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 18), 'str', '#maxnofargs#')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 19), dict_72849, (str_72858, str_72859))
# Adding element type (key, value) (line 34)
str_72860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 4), 'str', 'nofoptargs')
str_72861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 18), 'str', '#nofoptargs#')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 19), dict_72849, (str_72860, str_72861))
# Adding element type (key, value) (line 34)
str_72862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 4), 'str', 'docstr')
str_72863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, (-1)), 'str', '\tdef #argname#(#docsignature#): return #docreturn#\\n\\\n#docstrsigns#')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 19), dict_72849, (str_72862, str_72863))
# Adding element type (key, value) (line 34)
str_72864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 4), 'str', 'latexdocstr')
str_72865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, (-1)), 'str', '\n{{}\\verb@def #argname#(#latexdocsignature#): return #docreturn#@{}}\n#routnote#\n\n#latexdocstrsigns#')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 19), dict_72849, (str_72864, str_72865))
# Adding element type (key, value) (line 34)
str_72866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 4), 'str', 'docstrshort')
str_72867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 19), 'str', 'def #argname#(#docsignature#): return #docreturn#')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 19), dict_72849, (str_72866, str_72867))

# Assigning a type to the variable 'cb_routine_rules' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'cb_routine_rules', dict_72849)

# Assigning a List to a Name (line 143):

# Assigning a List to a Name (line 143):

# Obtaining an instance of the builtin type 'list' (line 143)
list_72868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 143)
# Adding element type (line 143)

# Obtaining an instance of the builtin type 'dict' (line 144)
dict_72869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 4), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 144)
# Adding element type (key, value) (line 144)
str_72870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 8), 'str', 'separatorsfor')

# Obtaining an instance of the builtin type 'dict' (line 145)
dict_72871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 25), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 145)
# Adding element type (key, value) (line 145)
str_72872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 26), 'str', 'decl')
str_72873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 34), 'str', '\n')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), dict_72871, (str_72872, str_72873))
# Adding element type (key, value) (line 145)
str_72874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 26), 'str', 'args')
str_72875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 34), 'str', ',')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), dict_72871, (str_72874, str_72875))
# Adding element type (key, value) (line 145)
str_72876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 39), 'str', 'optargs')
str_72877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 50), 'str', '')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), dict_72871, (str_72876, str_72877))
# Adding element type (key, value) (line 145)
str_72878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 54), 'str', 'pyobjfrom')
str_72879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 67), 'str', '\n')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), dict_72871, (str_72878, str_72879))
# Adding element type (key, value) (line 145)
str_72880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 73), 'str', 'freemem')
str_72881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 84), 'str', '\n')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), dict_72871, (str_72880, str_72881))
# Adding element type (key, value) (line 145)
str_72882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 26), 'str', 'args_td')
str_72883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 37), 'str', ',')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), dict_72871, (str_72882, str_72883))
# Adding element type (key, value) (line 145)
str_72884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 42), 'str', 'optargs_td')
str_72885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 56), 'str', '')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), dict_72871, (str_72884, str_72885))
# Adding element type (key, value) (line 145)
str_72886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 26), 'str', 'args_nm')
str_72887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 37), 'str', ',')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), dict_72871, (str_72886, str_72887))
# Adding element type (key, value) (line 145)
str_72888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 42), 'str', 'optargs_nm')
str_72889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 56), 'str', '')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), dict_72871, (str_72888, str_72889))
# Adding element type (key, value) (line 145)
str_72890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 26), 'str', 'frompyobj')
str_72891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 39), 'str', '\n')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), dict_72871, (str_72890, str_72891))
# Adding element type (key, value) (line 145)
str_72892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 45), 'str', 'setdims')
str_72893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 56), 'str', '\n')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), dict_72871, (str_72892, str_72893))
# Adding element type (key, value) (line 145)
str_72894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 26), 'str', 'docstrsigns')
str_72895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 41), 'str', '\\n"\n"')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), dict_72871, (str_72894, str_72895))
# Adding element type (key, value) (line 145)
str_72896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 26), 'str', 'latexdocstrsigns')
str_72897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 46), 'str', '\n')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), dict_72871, (str_72896, str_72897))
# Adding element type (key, value) (line 145)
str_72898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 26), 'str', 'latexdocstrreq')
str_72899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 44), 'str', '\n')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), dict_72871, (str_72898, str_72899))
# Adding element type (key, value) (line 145)
str_72900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 50), 'str', 'latexdocstropt')
str_72901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 68), 'str', '\n')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), dict_72871, (str_72900, str_72901))
# Adding element type (key, value) (line 145)
str_72902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 26), 'str', 'latexdocstrout')
str_72903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 44), 'str', '\n')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), dict_72871, (str_72902, str_72903))
# Adding element type (key, value) (line 145)
str_72904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 50), 'str', 'latexdocstrcbs')
str_72905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 68), 'str', '\n')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), dict_72871, (str_72904, str_72905))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72870, dict_72871))
# Adding element type (key, value) (line 144)
str_72906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 8), 'str', 'decl')
str_72907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 16), 'str', '/*decl*/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72906, str_72907))
# Adding element type (key, value) (line 144)
str_72908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 28), 'str', 'pyobjfrom')
str_72909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 41), 'str', '/*pyobjfrom*/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72908, str_72909))
# Adding element type (key, value) (line 144)
str_72910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 58), 'str', 'frompyobj')
str_72911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 71), 'str', '/*frompyobj*/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72910, str_72911))
# Adding element type (key, value) (line 144)
str_72912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 8), 'str', 'args')

# Obtaining an instance of the builtin type 'list' (line 156)
list_72913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 156)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72912, list_72913))
# Adding element type (key, value) (line 144)
str_72914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 20), 'str', 'optargs')
str_72915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 31), 'str', '')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72914, str_72915))
# Adding element type (key, value) (line 144)
str_72916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 35), 'str', 'return')
str_72917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 45), 'str', '')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72916, str_72917))
# Adding element type (key, value) (line 144)
str_72918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 49), 'str', 'strarglens')
str_72919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 63), 'str', '')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72918, str_72919))
# Adding element type (key, value) (line 144)
str_72920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 67), 'str', 'freemem')
str_72921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 78), 'str', '/*freemem*/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72920, str_72921))
# Adding element type (key, value) (line 144)
str_72922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 8), 'str', 'args_td')

# Obtaining an instance of the builtin type 'list' (line 157)
list_72923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 157)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72922, list_72923))
# Adding element type (key, value) (line 144)
str_72924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 23), 'str', 'optargs_td')
str_72925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 37), 'str', '')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72924, str_72925))
# Adding element type (key, value) (line 144)
str_72926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 41), 'str', 'strarglens_td')
str_72927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 58), 'str', '')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72926, str_72927))
# Adding element type (key, value) (line 144)
str_72928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 8), 'str', 'args_nm')

# Obtaining an instance of the builtin type 'list' (line 158)
list_72929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 158)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72928, list_72929))
# Adding element type (key, value) (line 144)
str_72930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 23), 'str', 'optargs_nm')
str_72931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 37), 'str', '')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72930, str_72931))
# Adding element type (key, value) (line 144)
str_72932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 41), 'str', 'strarglens_nm')
str_72933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 58), 'str', '')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72932, str_72933))
# Adding element type (key, value) (line 144)
str_72934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 8), 'str', 'noargs')
str_72935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 18), 'str', '')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72934, str_72935))
# Adding element type (key, value) (line 144)
str_72936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 8), 'str', 'setdims')
str_72937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 19), 'str', '/*setdims*/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72936, str_72937))
# Adding element type (key, value) (line 144)
str_72938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 8), 'str', 'docstrsigns')
str_72939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 23), 'str', '')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72938, str_72939))
# Adding element type (key, value) (line 144)
str_72940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 27), 'str', 'latexdocstrsigns')
str_72941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 47), 'str', '')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72940, str_72941))
# Adding element type (key, value) (line 144)
str_72942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 8), 'str', 'docstrreq')
str_72943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 21), 'str', '\tRequired arguments:')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72942, str_72943))
# Adding element type (key, value) (line 144)
str_72944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 8), 'str', 'docstropt')
str_72945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 21), 'str', '\tOptional arguments:')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72944, str_72945))
# Adding element type (key, value) (line 144)
str_72946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 8), 'str', 'docstrout')
str_72947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 21), 'str', '\tReturn objects:')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72946, str_72947))
# Adding element type (key, value) (line 144)
str_72948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 8), 'str', 'docstrcbs')
str_72949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 21), 'str', '\tCall-back functions:')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72948, str_72949))
# Adding element type (key, value) (line 144)
str_72950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 8), 'str', 'docreturn')
str_72951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 21), 'str', '')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72950, str_72951))
# Adding element type (key, value) (line 144)
str_72952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 25), 'str', 'docsign')
str_72953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 36), 'str', '')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72952, str_72953))
# Adding element type (key, value) (line 144)
str_72954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 40), 'str', 'docsignopt')
str_72955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 54), 'str', '')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72954, str_72955))
# Adding element type (key, value) (line 144)
str_72956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 8), 'str', 'latexdocstrreq')
str_72957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 26), 'str', '\\noindent Required arguments:')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72956, str_72957))
# Adding element type (key, value) (line 144)
str_72958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 8), 'str', 'latexdocstropt')
str_72959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 26), 'str', '\\noindent Optional arguments:')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72958, str_72959))
# Adding element type (key, value) (line 144)
str_72960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 8), 'str', 'latexdocstrout')
str_72961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 26), 'str', '\\noindent Return objects:')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72960, str_72961))
# Adding element type (key, value) (line 144)
str_72962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 8), 'str', 'latexdocstrcbs')
str_72963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 26), 'str', '\\noindent Call-back functions:')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72962, str_72963))
# Adding element type (key, value) (line 144)
str_72964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 8), 'str', 'routnote')

# Obtaining an instance of the builtin type 'dict' (line 171)
dict_72965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 20), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 171)
# Adding element type (key, value) (line 171)
# Getting the type of 'hasnote' (line 171)
hasnote_72966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 21), 'hasnote')
str_72967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 30), 'str', '--- #note#')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 20), dict_72965, (hasnote_72966, str_72967))
# Adding element type (key, value) (line 171)

# Call to l_not(...): (line 171)
# Processing the call arguments (line 171)
# Getting the type of 'hasnote' (line 171)
hasnote_72969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 50), 'hasnote', False)
# Processing the call keyword arguments (line 171)
kwargs_72970 = {}
# Getting the type of 'l_not' (line 171)
l_not_72968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 44), 'l_not', False)
# Calling l_not(args, kwargs) (line 171)
l_not_call_result_72971 = invoke(stypy.reporting.localization.Localization(__file__, 171, 44), l_not_72968, *[hasnote_72969], **kwargs_72970)

str_72972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 60), 'str', '')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 20), dict_72965, (l_not_call_result_72971, str_72972))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), dict_72869, (str_72964, dict_72965))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 16), list_72868, dict_72869)
# Adding element type (line 143)

# Obtaining an instance of the builtin type 'dict' (line 172)
dict_72973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 7), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 172)
# Adding element type (key, value) (line 172)
str_72974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 8), 'str', 'decl')
str_72975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 16), 'str', '\t#ctype# return_value;')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 7), dict_72973, (str_72974, str_72975))
# Adding element type (key, value) (line 172)
str_72976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 8), 'str', 'frompyobj')

# Obtaining an instance of the builtin type 'list' (line 174)
list_72977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 174)
# Adding element type (line 174)

# Obtaining an instance of the builtin type 'dict' (line 174)
dict_72978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 174)
# Adding element type (key, value) (line 174)
# Getting the type of 'debugcapi' (line 174)
debugcapi_72979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 23), 'debugcapi')
str_72980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 34), 'str', '\tCFUNCSMESS("cb:Getting return_value->");')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 22), dict_72978, (debugcapi_72979, str_72980))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 21), list_72977, dict_72978)
# Adding element type (line 174)
str_72981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 22), 'str', '\tif (capi_j>capi_i)\n\t\tGETSCALARFROMPYTUPLE(capi_return,capi_i++,&return_value,#ctype#,"#ctype#_from_pyobj failed in converting return_value of call-back function #name# to C #ctype#\\n");')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 21), list_72977, str_72981)
# Adding element type (line 174)

# Obtaining an instance of the builtin type 'dict' (line 176)
dict_72982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 176)
# Adding element type (key, value) (line 176)
# Getting the type of 'debugcapi' (line 176)
debugcapi_72983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 23), 'debugcapi')
str_72984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 23), 'str', '\tfprintf(stderr,"#showvalueformat#.\\n",return_value);')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 22), dict_72982, (debugcapi_72983, str_72984))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 21), list_72977, dict_72982)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 7), dict_72973, (str_72976, list_72977))
# Adding element type (key, value) (line 172)
str_72985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 8), 'str', 'need')

# Obtaining an instance of the builtin type 'list' (line 179)
list_72986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 179)
# Adding element type (line 179)
str_72987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 17), 'str', '#ctype#_from_pyobj')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 16), list_72986, str_72987)
# Adding element type (line 179)

# Obtaining an instance of the builtin type 'dict' (line 179)
dict_72988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 39), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 179)
# Adding element type (key, value) (line 179)
# Getting the type of 'debugcapi' (line 179)
debugcapi_72989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 40), 'debugcapi')
str_72990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 51), 'str', 'CFUNCSMESS')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 39), dict_72988, (debugcapi_72989, str_72990))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 16), list_72986, dict_72988)
# Adding element type (line 179)
str_72991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 66), 'str', 'GETSCALARFROMPYTUPLE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 16), list_72986, str_72991)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 7), dict_72973, (str_72985, list_72986))
# Adding element type (key, value) (line 172)
str_72992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 8), 'str', 'return')
str_72993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 18), 'str', '\treturn return_value;')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 7), dict_72973, (str_72992, str_72993))
# Adding element type (key, value) (line 172)
str_72994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 8), 'str', '_check')

# Call to l_and(...): (line 181)
# Processing the call arguments (line 181)
# Getting the type of 'isfunction' (line 181)
isfunction_72996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 24), 'isfunction', False)

# Call to l_not(...): (line 181)
# Processing the call arguments (line 181)
# Getting the type of 'isstringfunction' (line 181)
isstringfunction_72998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 42), 'isstringfunction', False)
# Processing the call keyword arguments (line 181)
kwargs_72999 = {}
# Getting the type of 'l_not' (line 181)
l_not_72997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 36), 'l_not', False)
# Calling l_not(args, kwargs) (line 181)
l_not_call_result_73000 = invoke(stypy.reporting.localization.Localization(__file__, 181, 36), l_not_72997, *[isstringfunction_72998], **kwargs_72999)


# Call to l_not(...): (line 181)
# Processing the call arguments (line 181)
# Getting the type of 'iscomplexfunction' (line 181)
iscomplexfunction_73002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 67), 'iscomplexfunction', False)
# Processing the call keyword arguments (line 181)
kwargs_73003 = {}
# Getting the type of 'l_not' (line 181)
l_not_73001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 61), 'l_not', False)
# Calling l_not(args, kwargs) (line 181)
l_not_call_result_73004 = invoke(stypy.reporting.localization.Localization(__file__, 181, 61), l_not_73001, *[iscomplexfunction_73002], **kwargs_73003)

# Processing the call keyword arguments (line 181)
kwargs_73005 = {}
# Getting the type of 'l_and' (line 181)
l_and_72995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 18), 'l_and', False)
# Calling l_and(args, kwargs) (line 181)
l_and_call_result_73006 = invoke(stypy.reporting.localization.Localization(__file__, 181, 18), l_and_72995, *[isfunction_72996, l_not_call_result_73000, l_not_call_result_73004], **kwargs_73005)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 7), dict_72973, (str_72994, l_and_call_result_73006))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 16), list_72868, dict_72973)
# Adding element type (line 143)

# Obtaining an instance of the builtin type 'dict' (line 183)
dict_73007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 4), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 183)
# Adding element type (key, value) (line 183)
str_73008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 8), 'str', 'pyobjfrom')

# Obtaining an instance of the builtin type 'dict' (line 184)
dict_73009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 21), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 184)
# Adding element type (key, value) (line 184)
# Getting the type of 'debugcapi' (line 184)
debugcapi_73010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 22), 'debugcapi')
str_73011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 33), 'str', '\tfprintf(stderr,"debug-capi:cb:#name#:%d:\\n",return_value_len);')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 21), dict_73009, (debugcapi_73010, str_73011))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 4), dict_73007, (str_73008, dict_73009))
# Adding element type (key, value) (line 183)
str_73012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 8), 'str', 'args')
str_73013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 16), 'str', '#ctype# return_value,int return_value_len')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 4), dict_73007, (str_73012, str_73013))
# Adding element type (key, value) (line 183)
str_73014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 8), 'str', 'args_nm')
str_73015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 19), 'str', 'return_value,&return_value_len')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 4), dict_73007, (str_73014, str_73015))
# Adding element type (key, value) (line 183)
str_73016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 8), 'str', 'args_td')
str_73017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 19), 'str', '#ctype# ,int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 4), dict_73007, (str_73016, str_73017))
# Adding element type (key, value) (line 183)
str_73018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 8), 'str', 'frompyobj')

# Obtaining an instance of the builtin type 'list' (line 188)
list_73019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 188)
# Adding element type (line 188)

# Obtaining an instance of the builtin type 'dict' (line 188)
dict_73020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 188)
# Adding element type (key, value) (line 188)
# Getting the type of 'debugcapi' (line 188)
debugcapi_73021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 23), 'debugcapi')
str_73022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 34), 'str', '\tCFUNCSMESS("cb:Getting return_value->\\"");')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 22), dict_73020, (debugcapi_73021, str_73022))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 21), list_73019, dict_73020)
# Adding element type (line 188)
str_73023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, (-1)), 'str', '\tif (capi_j>capi_i)\n\t\tGETSTRFROMPYTUPLE(capi_return,capi_i++,return_value,return_value_len);')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 21), list_73019, str_73023)
# Adding element type (line 188)

# Obtaining an instance of the builtin type 'dict' (line 191)
dict_73024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 191)
# Adding element type (key, value) (line 191)
# Getting the type of 'debugcapi' (line 191)
debugcapi_73025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 23), 'debugcapi')
str_73026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 23), 'str', '\tfprintf(stderr,"#showvalueformat#\\".\\n",return_value);')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 22), dict_73024, (debugcapi_73025, str_73026))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 21), list_73019, dict_73024)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 4), dict_73007, (str_73018, list_73019))
# Adding element type (key, value) (line 183)
str_73027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 8), 'str', 'need')

# Obtaining an instance of the builtin type 'list' (line 194)
list_73028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 194)
# Adding element type (line 194)
str_73029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 17), 'str', '#ctype#_from_pyobj')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 16), list_73028, str_73029)
# Adding element type (line 194)

# Obtaining an instance of the builtin type 'dict' (line 194)
dict_73030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 39), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 194)
# Adding element type (key, value) (line 194)
# Getting the type of 'debugcapi' (line 194)
debugcapi_73031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 40), 'debugcapi')
str_73032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 51), 'str', 'CFUNCSMESS')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 39), dict_73030, (debugcapi_73031, str_73032))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 16), list_73028, dict_73030)
# Adding element type (line 194)
str_73033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 17), 'str', 'string.h')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 16), list_73028, str_73033)
# Adding element type (line 194)
str_73034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 29), 'str', 'GETSTRFROMPYTUPLE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 16), list_73028, str_73034)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 4), dict_73007, (str_73027, list_73028))
# Adding element type (key, value) (line 183)
str_73035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 8), 'str', 'return')
str_73036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 18), 'str', 'return;')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 4), dict_73007, (str_73035, str_73036))
# Adding element type (key, value) (line 183)
str_73037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 8), 'str', '_check')
# Getting the type of 'isstringfunction' (line 197)
isstringfunction_73038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 18), 'isstringfunction')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 4), dict_73007, (str_73037, isstringfunction_73038))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 16), list_72868, dict_73007)
# Adding element type (line 143)

# Obtaining an instance of the builtin type 'dict' (line 199)
dict_73039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 4), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 199)
# Adding element type (key, value) (line 199)
str_73040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 8), 'str', 'optargs')
str_73041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, (-1)), 'str', '\n#ifndef F2PY_CB_RETURNCOMPLEX\n#ctype# *return_value\n#endif\n')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 4), dict_73039, (str_73040, str_73041))
# Adding element type (key, value) (line 199)
str_73042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 8), 'str', 'optargs_nm')
str_73043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, (-1)), 'str', '\n#ifndef F2PY_CB_RETURNCOMPLEX\nreturn_value\n#endif\n')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 4), dict_73039, (str_73042, str_73043))
# Adding element type (key, value) (line 199)
str_73044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 8), 'str', 'optargs_td')
str_73045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, (-1)), 'str', '\n#ifndef F2PY_CB_RETURNCOMPLEX\n#ctype# *\n#endif\n')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 4), dict_73039, (str_73044, str_73045))
# Adding element type (key, value) (line 199)
str_73046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 8), 'str', 'decl')
str_73047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, (-1)), 'str', '\n#ifdef F2PY_CB_RETURNCOMPLEX\n\t#ctype# return_value;\n#endif\n')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 4), dict_73039, (str_73046, str_73047))
# Adding element type (key, value) (line 199)
str_73048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 8), 'str', 'frompyobj')

# Obtaining an instance of the builtin type 'list' (line 220)
list_73049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 220)
# Adding element type (line 220)

# Obtaining an instance of the builtin type 'dict' (line 220)
dict_73050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 220)
# Adding element type (key, value) (line 220)
# Getting the type of 'debugcapi' (line 220)
debugcapi_73051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 23), 'debugcapi')
str_73052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 34), 'str', '\tCFUNCSMESS("cb:Getting return_value->");')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 22), dict_73050, (debugcapi_73051, str_73052))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 21), list_73049, dict_73050)
# Adding element type (line 220)
str_73053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, (-1)), 'str', '\tif (capi_j>capi_i)\n#ifdef F2PY_CB_RETURNCOMPLEX\n\t\tGETSCALARFROMPYTUPLE(capi_return,capi_i++,&return_value,#ctype#,"#ctype#_from_pyobj failed in converting return_value of call-back function #name# to C #ctype#\\n");\n#else\n\t\tGETSCALARFROMPYTUPLE(capi_return,capi_i++,return_value,#ctype#,"#ctype#_from_pyobj failed in converting return_value of call-back function #name# to C #ctype#\\n");\n#endif\n')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 21), list_73049, str_73053)
# Adding element type (line 220)

# Obtaining an instance of the builtin type 'dict' (line 229)
dict_73054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 229)
# Adding element type (key, value) (line 229)
# Getting the type of 'debugcapi' (line 229)
debugcapi_73055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 23), 'debugcapi')
str_73056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, (-1)), 'str', '\n#ifdef F2PY_CB_RETURNCOMPLEX\n\tfprintf(stderr,"#showvalueformat#.\\n",(return_value).r,(return_value).i);\n#else\n\tfprintf(stderr,"#showvalueformat#.\\n",(*return_value).r,(*return_value).i);\n#endif\n\n')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 22), dict_73054, (debugcapi_73055, str_73056))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 21), list_73049, dict_73054)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 4), dict_73039, (str_73048, list_73049))
# Adding element type (key, value) (line 199)
str_73057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 8), 'str', 'return')
str_73058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, (-1)), 'str', '\n#ifdef F2PY_CB_RETURNCOMPLEX\n\treturn return_value;\n#else\n\treturn;\n#endif\n')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 4), dict_73039, (str_73057, str_73058))
# Adding element type (key, value) (line 199)
str_73059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 8), 'str', 'need')

# Obtaining an instance of the builtin type 'list' (line 245)
list_73060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 245)
# Adding element type (line 245)
str_73061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 17), 'str', '#ctype#_from_pyobj')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 16), list_73060, str_73061)
# Adding element type (line 245)

# Obtaining an instance of the builtin type 'dict' (line 245)
dict_73062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 39), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 245)
# Adding element type (key, value) (line 245)
# Getting the type of 'debugcapi' (line 245)
debugcapi_73063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 40), 'debugcapi')
str_73064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 51), 'str', 'CFUNCSMESS')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 39), dict_73062, (debugcapi_73063, str_73064))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 16), list_73060, dict_73062)
# Adding element type (line 245)
str_73065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 17), 'str', 'string.h')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 16), list_73060, str_73065)
# Adding element type (line 245)
str_73066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 29), 'str', 'GETSCALARFROMPYTUPLE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 16), list_73060, str_73066)
# Adding element type (line 245)
str_73067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 53), 'str', '#ctype#')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 16), list_73060, str_73067)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 4), dict_73039, (str_73059, list_73060))
# Adding element type (key, value) (line 199)
str_73068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 8), 'str', '_check')
# Getting the type of 'iscomplexfunction' (line 247)
iscomplexfunction_73069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 18), 'iscomplexfunction')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 4), dict_73039, (str_73068, iscomplexfunction_73069))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 16), list_72868, dict_73039)
# Adding element type (line 143)

# Obtaining an instance of the builtin type 'dict' (line 249)
dict_73070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 4), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 249)
# Adding element type (key, value) (line 249)
str_73071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 5), 'str', 'docstrout')
str_73072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 18), 'str', '\t\t#pydocsignout#')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 4), dict_73070, (str_73071, str_73072))
# Adding element type (key, value) (line 249)
str_73073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 5), 'str', 'latexdocstrout')

# Obtaining an instance of the builtin type 'list' (line 250)
list_73074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 250)
# Adding element type (line 250)
str_73075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 24), 'str', '\\item[]{{}\\verb@#pydocsignout#@{}}')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 23), list_73074, str_73075)
# Adding element type (line 250)

# Obtaining an instance of the builtin type 'dict' (line 251)
dict_73076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 24), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 251)
# Adding element type (key, value) (line 251)
# Getting the type of 'hasnote' (line 251)
hasnote_73077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 25), 'hasnote')
str_73078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 34), 'str', '--- #note#')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 24), dict_73076, (hasnote_73077, str_73078))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 23), list_73074, dict_73076)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 4), dict_73070, (str_73073, list_73074))
# Adding element type (key, value) (line 249)
str_73079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 5), 'str', 'docreturn')
str_73080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 18), 'str', '#rname#,')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 4), dict_73070, (str_73079, str_73080))
# Adding element type (key, value) (line 249)
str_73081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 5), 'str', '_check')
# Getting the type of 'isfunction' (line 253)
isfunction_73082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 15), 'isfunction')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 4), dict_73070, (str_73081, isfunction_73082))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 16), list_72868, dict_73070)
# Adding element type (line 143)

# Obtaining an instance of the builtin type 'dict' (line 254)
dict_73083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 4), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 254)
# Adding element type (key, value) (line 254)
str_73084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 5), 'str', '_check')
# Getting the type of 'issubroutine' (line 254)
issubroutine_73085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 15), 'issubroutine')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 4), dict_73083, (str_73084, issubroutine_73085))
# Adding element type (key, value) (line 254)
str_73086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 29), 'str', 'return')
str_73087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 39), 'str', 'return;')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 4), dict_73083, (str_73086, str_73087))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 16), list_72868, dict_73083)

# Assigning a type to the variable 'cb_rout_rules' (line 143)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 0), 'cb_rout_rules', list_72868)

# Assigning a List to a Name (line 257):

# Assigning a List to a Name (line 257):

# Obtaining an instance of the builtin type 'list' (line 257)
list_73088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 257)
# Adding element type (line 257)

# Obtaining an instance of the builtin type 'dict' (line 258)
dict_73089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 4), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 258)
# Adding element type (key, value) (line 258)
str_73090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 8), 'str', 'docstropt')

# Obtaining an instance of the builtin type 'dict' (line 259)
dict_73091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 21), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 259)
# Adding element type (key, value) (line 259)

# Call to l_and(...): (line 259)
# Processing the call arguments (line 259)
# Getting the type of 'isoptional' (line 259)
isoptional_73093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'isoptional', False)
# Getting the type of 'isintent_nothide' (line 259)
isintent_nothide_73094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 40), 'isintent_nothide', False)
# Processing the call keyword arguments (line 259)
kwargs_73095 = {}
# Getting the type of 'l_and' (line 259)
l_and_73092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 22), 'l_and', False)
# Calling l_and(args, kwargs) (line 259)
l_and_call_result_73096 = invoke(stypy.reporting.localization.Localization(__file__, 259, 22), l_and_73092, *[isoptional_73093, isintent_nothide_73094], **kwargs_73095)

str_73097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 59), 'str', '\t\t#pydocsign#')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 21), dict_73091, (l_and_call_result_73096, str_73097))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 4), dict_73089, (str_73090, dict_73091))
# Adding element type (key, value) (line 258)
str_73098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 8), 'str', 'docstrreq')

# Obtaining an instance of the builtin type 'dict' (line 260)
dict_73099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 21), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 260)
# Adding element type (key, value) (line 260)

# Call to l_and(...): (line 260)
# Processing the call arguments (line 260)
# Getting the type of 'isrequired' (line 260)
isrequired_73101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 28), 'isrequired', False)
# Getting the type of 'isintent_nothide' (line 260)
isintent_nothide_73102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 40), 'isintent_nothide', False)
# Processing the call keyword arguments (line 260)
kwargs_73103 = {}
# Getting the type of 'l_and' (line 260)
l_and_73100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 22), 'l_and', False)
# Calling l_and(args, kwargs) (line 260)
l_and_call_result_73104 = invoke(stypy.reporting.localization.Localization(__file__, 260, 22), l_and_73100, *[isrequired_73101, isintent_nothide_73102], **kwargs_73103)

str_73105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 59), 'str', '\t\t#pydocsign#')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 21), dict_73099, (l_and_call_result_73104, str_73105))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 4), dict_73089, (str_73098, dict_73099))
# Adding element type (key, value) (line 258)
str_73106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 8), 'str', 'docstrout')

# Obtaining an instance of the builtin type 'dict' (line 261)
dict_73107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 21), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 261)
# Adding element type (key, value) (line 261)
# Getting the type of 'isintent_out' (line 261)
isintent_out_73108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 22), 'isintent_out')
str_73109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 36), 'str', '\t\t#pydocsignout#')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 21), dict_73107, (isintent_out_73108, str_73109))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 4), dict_73089, (str_73106, dict_73107))
# Adding element type (key, value) (line 258)
str_73110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 8), 'str', 'latexdocstropt')

# Obtaining an instance of the builtin type 'dict' (line 262)
dict_73111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 26), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 262)
# Adding element type (key, value) (line 262)

# Call to l_and(...): (line 262)
# Processing the call arguments (line 262)
# Getting the type of 'isoptional' (line 262)
isoptional_73113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 33), 'isoptional', False)
# Getting the type of 'isintent_nothide' (line 262)
isintent_nothide_73114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 45), 'isintent_nothide', False)
# Processing the call keyword arguments (line 262)
kwargs_73115 = {}
# Getting the type of 'l_and' (line 262)
l_and_73112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 27), 'l_and', False)
# Calling l_and(args, kwargs) (line 262)
l_and_call_result_73116 = invoke(stypy.reporting.localization.Localization(__file__, 262, 27), l_and_73112, *[isoptional_73113, isintent_nothide_73114], **kwargs_73115)


# Obtaining an instance of the builtin type 'list' (line 262)
list_73117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 64), 'list')
# Adding type elements to the builtin type 'list' instance (line 262)
# Adding element type (line 262)
str_73118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 65), 'str', '\\item[]{{}\\verb@#pydocsign#@{}}')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 64), list_73117, str_73118)
# Adding element type (line 262)

# Obtaining an instance of the builtin type 'dict' (line 263)
dict_73119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 65), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 263)
# Adding element type (key, value) (line 263)
# Getting the type of 'hasnote' (line 263)
hasnote_73120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 66), 'hasnote')
str_73121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 75), 'str', '--- #note#')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 65), dict_73119, (hasnote_73120, str_73121))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 64), list_73117, dict_73119)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 26), dict_73111, (l_and_call_result_73116, list_73117))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 4), dict_73089, (str_73110, dict_73111))
# Adding element type (key, value) (line 258)
str_73122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 8), 'str', 'latexdocstrreq')

# Obtaining an instance of the builtin type 'dict' (line 264)
dict_73123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 26), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 264)
# Adding element type (key, value) (line 264)

# Call to l_and(...): (line 264)
# Processing the call arguments (line 264)
# Getting the type of 'isrequired' (line 264)
isrequired_73125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 33), 'isrequired', False)
# Getting the type of 'isintent_nothide' (line 264)
isintent_nothide_73126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 45), 'isintent_nothide', False)
# Processing the call keyword arguments (line 264)
kwargs_73127 = {}
# Getting the type of 'l_and' (line 264)
l_and_73124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 27), 'l_and', False)
# Calling l_and(args, kwargs) (line 264)
l_and_call_result_73128 = invoke(stypy.reporting.localization.Localization(__file__, 264, 27), l_and_73124, *[isrequired_73125, isintent_nothide_73126], **kwargs_73127)


# Obtaining an instance of the builtin type 'list' (line 264)
list_73129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 64), 'list')
# Adding type elements to the builtin type 'list' instance (line 264)
# Adding element type (line 264)
str_73130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 65), 'str', '\\item[]{{}\\verb@#pydocsign#@{}}')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 64), list_73129, str_73130)
# Adding element type (line 264)

# Obtaining an instance of the builtin type 'dict' (line 265)
dict_73131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 65), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 265)
# Adding element type (key, value) (line 265)
# Getting the type of 'hasnote' (line 265)
hasnote_73132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 66), 'hasnote')
str_73133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 75), 'str', '--- #note#')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 65), dict_73131, (hasnote_73132, str_73133))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 64), list_73129, dict_73131)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 26), dict_73123, (l_and_call_result_73128, list_73129))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 4), dict_73089, (str_73122, dict_73123))
# Adding element type (key, value) (line 258)
str_73134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 8), 'str', 'latexdocstrout')

# Obtaining an instance of the builtin type 'dict' (line 266)
dict_73135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 26), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 266)
# Adding element type (key, value) (line 266)
# Getting the type of 'isintent_out' (line 266)
isintent_out_73136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 27), 'isintent_out')

# Obtaining an instance of the builtin type 'list' (line 266)
list_73137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 41), 'list')
# Adding type elements to the builtin type 'list' instance (line 266)
# Adding element type (line 266)
str_73138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 42), 'str', '\\item[]{{}\\verb@#pydocsignout#@{}}')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 41), list_73137, str_73138)
# Adding element type (line 266)

# Obtaining an instance of the builtin type 'dict' (line 267)
dict_73139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 42), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 267)
# Adding element type (key, value) (line 267)

# Call to l_and(...): (line 267)
# Processing the call arguments (line 267)
# Getting the type of 'hasnote' (line 267)
hasnote_73141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 49), 'hasnote', False)
# Getting the type of 'isintent_hide' (line 267)
isintent_hide_73142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 58), 'isintent_hide', False)
# Processing the call keyword arguments (line 267)
kwargs_73143 = {}
# Getting the type of 'l_and' (line 267)
l_and_73140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 43), 'l_and', False)
# Calling l_and(args, kwargs) (line 267)
l_and_call_result_73144 = invoke(stypy.reporting.localization.Localization(__file__, 267, 43), l_and_73140, *[hasnote_73141, isintent_hide_73142], **kwargs_73143)

str_73145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 74), 'str', '--- #note#')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 42), dict_73139, (l_and_call_result_73144, str_73145))
# Adding element type (key, value) (line 267)

# Call to l_and(...): (line 268)
# Processing the call arguments (line 268)
# Getting the type of 'hasnote' (line 268)
hasnote_73147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 49), 'hasnote', False)
# Getting the type of 'isintent_nothide' (line 268)
isintent_nothide_73148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 58), 'isintent_nothide', False)
# Processing the call keyword arguments (line 268)
kwargs_73149 = {}
# Getting the type of 'l_and' (line 268)
l_and_73146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 43), 'l_and', False)
# Calling l_and(args, kwargs) (line 268)
l_and_call_result_73150 = invoke(stypy.reporting.localization.Localization(__file__, 268, 43), l_and_73146, *[hasnote_73147, isintent_nothide_73148], **kwargs_73149)

str_73151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 77), 'str', '--- See above.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 42), dict_73139, (l_and_call_result_73150, str_73151))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 41), list_73137, dict_73139)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 26), dict_73135, (isintent_out_73136, list_73137))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 4), dict_73089, (str_73134, dict_73135))
# Adding element type (key, value) (line 258)
str_73152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 8), 'str', 'docsign')

# Obtaining an instance of the builtin type 'dict' (line 269)
dict_73153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 269)
# Adding element type (key, value) (line 269)

# Call to l_and(...): (line 269)
# Processing the call arguments (line 269)
# Getting the type of 'isrequired' (line 269)
isrequired_73155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 26), 'isrequired', False)
# Getting the type of 'isintent_nothide' (line 269)
isintent_nothide_73156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 38), 'isintent_nothide', False)
# Processing the call keyword arguments (line 269)
kwargs_73157 = {}
# Getting the type of 'l_and' (line 269)
l_and_73154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 20), 'l_and', False)
# Calling l_and(args, kwargs) (line 269)
l_and_call_result_73158 = invoke(stypy.reporting.localization.Localization(__file__, 269, 20), l_and_73154, *[isrequired_73155, isintent_nothide_73156], **kwargs_73157)

str_73159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 57), 'str', '#varname#,')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 19), dict_73153, (l_and_call_result_73158, str_73159))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 4), dict_73089, (str_73152, dict_73153))
# Adding element type (key, value) (line 258)
str_73160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 8), 'str', 'docsignopt')

# Obtaining an instance of the builtin type 'dict' (line 270)
dict_73161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 270)
# Adding element type (key, value) (line 270)

# Call to l_and(...): (line 270)
# Processing the call arguments (line 270)
# Getting the type of 'isoptional' (line 270)
isoptional_73163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 29), 'isoptional', False)
# Getting the type of 'isintent_nothide' (line 270)
isintent_nothide_73164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 41), 'isintent_nothide', False)
# Processing the call keyword arguments (line 270)
kwargs_73165 = {}
# Getting the type of 'l_and' (line 270)
l_and_73162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 23), 'l_and', False)
# Calling l_and(args, kwargs) (line 270)
l_and_call_result_73166 = invoke(stypy.reporting.localization.Localization(__file__, 270, 23), l_and_73162, *[isoptional_73163, isintent_nothide_73164], **kwargs_73165)

str_73167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 60), 'str', '#varname#,')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 22), dict_73161, (l_and_call_result_73166, str_73167))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 4), dict_73089, (str_73160, dict_73161))
# Adding element type (key, value) (line 258)
str_73168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 8), 'str', 'depend')
str_73169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 18), 'str', '')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 4), dict_73089, (str_73168, str_73169))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_73088, dict_73089)
# Adding element type (line 257)

# Obtaining an instance of the builtin type 'dict' (line 273)
dict_73170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 4), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 273)
# Adding element type (key, value) (line 273)
str_73171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 8), 'str', 'args')

# Obtaining an instance of the builtin type 'dict' (line 274)
dict_73172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 274)
# Adding element type (key, value) (line 274)

# Call to l_and(...): (line 275)
# Processing the call arguments (line 275)
# Getting the type of 'isscalar' (line 275)
isscalar_73174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 18), 'isscalar', False)
# Getting the type of 'isintent_c' (line 275)
isintent_c_73175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 28), 'isintent_c', False)
# Processing the call keyword arguments (line 275)
kwargs_73176 = {}
# Getting the type of 'l_and' (line 275)
l_and_73173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'l_and', False)
# Calling l_and(args, kwargs) (line 275)
l_and_call_result_73177 = invoke(stypy.reporting.localization.Localization(__file__, 275, 12), l_and_73173, *[isscalar_73174, isintent_c_73175], **kwargs_73176)

str_73178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 41), 'str', '#ctype# #varname_i#')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 16), dict_73172, (l_and_call_result_73177, str_73178))
# Adding element type (key, value) (line 274)

# Call to l_and(...): (line 276)
# Processing the call arguments (line 276)
# Getting the type of 'isscalar' (line 276)
isscalar_73180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 18), 'isscalar', False)

# Call to l_not(...): (line 276)
# Processing the call arguments (line 276)
# Getting the type of 'isintent_c' (line 276)
isintent_c_73182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 34), 'isintent_c', False)
# Processing the call keyword arguments (line 276)
kwargs_73183 = {}
# Getting the type of 'l_not' (line 276)
l_not_73181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 28), 'l_not', False)
# Calling l_not(args, kwargs) (line 276)
l_not_call_result_73184 = invoke(stypy.reporting.localization.Localization(__file__, 276, 28), l_not_73181, *[isintent_c_73182], **kwargs_73183)

# Processing the call keyword arguments (line 276)
kwargs_73185 = {}
# Getting the type of 'l_and' (line 276)
l_and_73179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'l_and', False)
# Calling l_and(args, kwargs) (line 276)
l_and_call_result_73186 = invoke(stypy.reporting.localization.Localization(__file__, 276, 12), l_and_73179, *[isscalar_73180, l_not_call_result_73184], **kwargs_73185)

str_73187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 48), 'str', '#ctype# *#varname_i#_cb_capi')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 16), dict_73172, (l_and_call_result_73186, str_73187))
# Adding element type (key, value) (line 274)
# Getting the type of 'isarray' (line 277)
isarray_73188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'isarray')
str_73189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 21), 'str', '#ctype# *#varname_i#')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 16), dict_73172, (isarray_73188, str_73189))
# Adding element type (key, value) (line 274)
# Getting the type of 'isstring' (line 278)
isstring_73190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'isstring')
str_73191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 22), 'str', '#ctype# #varname_i#')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 16), dict_73172, (isstring_73190, str_73191))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 4), dict_73170, (str_73171, dict_73172))
# Adding element type (key, value) (line 273)
str_73192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 8), 'str', 'args_nm')

# Obtaining an instance of the builtin type 'dict' (line 280)
dict_73193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 280)
# Adding element type (key, value) (line 280)

# Call to l_and(...): (line 281)
# Processing the call arguments (line 281)
# Getting the type of 'isscalar' (line 281)
isscalar_73195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 18), 'isscalar', False)
# Getting the type of 'isintent_c' (line 281)
isintent_c_73196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 28), 'isintent_c', False)
# Processing the call keyword arguments (line 281)
kwargs_73197 = {}
# Getting the type of 'l_and' (line 281)
l_and_73194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'l_and', False)
# Calling l_and(args, kwargs) (line 281)
l_and_call_result_73198 = invoke(stypy.reporting.localization.Localization(__file__, 281, 12), l_and_73194, *[isscalar_73195, isintent_c_73196], **kwargs_73197)

str_73199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 41), 'str', '#varname_i#')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 19), dict_73193, (l_and_call_result_73198, str_73199))
# Adding element type (key, value) (line 280)

# Call to l_and(...): (line 282)
# Processing the call arguments (line 282)
# Getting the type of 'isscalar' (line 282)
isscalar_73201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 18), 'isscalar', False)

# Call to l_not(...): (line 282)
# Processing the call arguments (line 282)
# Getting the type of 'isintent_c' (line 282)
isintent_c_73203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 34), 'isintent_c', False)
# Processing the call keyword arguments (line 282)
kwargs_73204 = {}
# Getting the type of 'l_not' (line 282)
l_not_73202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 28), 'l_not', False)
# Calling l_not(args, kwargs) (line 282)
l_not_call_result_73205 = invoke(stypy.reporting.localization.Localization(__file__, 282, 28), l_not_73202, *[isintent_c_73203], **kwargs_73204)

# Processing the call keyword arguments (line 282)
kwargs_73206 = {}
# Getting the type of 'l_and' (line 282)
l_and_73200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'l_and', False)
# Calling l_and(args, kwargs) (line 282)
l_and_call_result_73207 = invoke(stypy.reporting.localization.Localization(__file__, 282, 12), l_and_73200, *[isscalar_73201, l_not_call_result_73205], **kwargs_73206)

str_73208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 48), 'str', '#varname_i#_cb_capi')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 19), dict_73193, (l_and_call_result_73207, str_73208))
# Adding element type (key, value) (line 280)
# Getting the type of 'isarray' (line 283)
isarray_73209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'isarray')
str_73210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 21), 'str', '#varname_i#')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 19), dict_73193, (isarray_73209, str_73210))
# Adding element type (key, value) (line 280)
# Getting the type of 'isstring' (line 284)
isstring_73211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'isstring')
str_73212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 22), 'str', '#varname_i#')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 19), dict_73193, (isstring_73211, str_73212))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 4), dict_73170, (str_73192, dict_73193))
# Adding element type (key, value) (line 273)
str_73213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 8), 'str', 'args_td')

# Obtaining an instance of the builtin type 'dict' (line 286)
dict_73214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 286)
# Adding element type (key, value) (line 286)

# Call to l_and(...): (line 287)
# Processing the call arguments (line 287)
# Getting the type of 'isscalar' (line 287)
isscalar_73216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 18), 'isscalar', False)
# Getting the type of 'isintent_c' (line 287)
isintent_c_73217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 28), 'isintent_c', False)
# Processing the call keyword arguments (line 287)
kwargs_73218 = {}
# Getting the type of 'l_and' (line 287)
l_and_73215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'l_and', False)
# Calling l_and(args, kwargs) (line 287)
l_and_call_result_73219 = invoke(stypy.reporting.localization.Localization(__file__, 287, 12), l_and_73215, *[isscalar_73216, isintent_c_73217], **kwargs_73218)

str_73220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 41), 'str', '#ctype#')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 19), dict_73214, (l_and_call_result_73219, str_73220))
# Adding element type (key, value) (line 286)

# Call to l_and(...): (line 288)
# Processing the call arguments (line 288)
# Getting the type of 'isscalar' (line 288)
isscalar_73222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 18), 'isscalar', False)

# Call to l_not(...): (line 288)
# Processing the call arguments (line 288)
# Getting the type of 'isintent_c' (line 288)
isintent_c_73224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 34), 'isintent_c', False)
# Processing the call keyword arguments (line 288)
kwargs_73225 = {}
# Getting the type of 'l_not' (line 288)
l_not_73223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 28), 'l_not', False)
# Calling l_not(args, kwargs) (line 288)
l_not_call_result_73226 = invoke(stypy.reporting.localization.Localization(__file__, 288, 28), l_not_73223, *[isintent_c_73224], **kwargs_73225)

# Processing the call keyword arguments (line 288)
kwargs_73227 = {}
# Getting the type of 'l_and' (line 288)
l_and_73221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'l_and', False)
# Calling l_and(args, kwargs) (line 288)
l_and_call_result_73228 = invoke(stypy.reporting.localization.Localization(__file__, 288, 12), l_and_73221, *[isscalar_73222, l_not_call_result_73226], **kwargs_73227)

str_73229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 48), 'str', '#ctype# *')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 19), dict_73214, (l_and_call_result_73228, str_73229))
# Adding element type (key, value) (line 286)
# Getting the type of 'isarray' (line 289)
isarray_73230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'isarray')
str_73231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 21), 'str', '#ctype# *')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 19), dict_73214, (isarray_73230, str_73231))
# Adding element type (key, value) (line 286)
# Getting the type of 'isstring' (line 290)
isstring_73232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'isstring')
str_73233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 22), 'str', '#ctype#')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 19), dict_73214, (isstring_73232, str_73233))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 4), dict_73170, (str_73213, dict_73214))
# Adding element type (key, value) (line 273)
str_73234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 8), 'str', 'strarglens')

# Obtaining an instance of the builtin type 'dict' (line 293)
dict_73235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 293)
# Adding element type (key, value) (line 293)
# Getting the type of 'isstring' (line 293)
isstring_73236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 23), 'isstring')
str_73237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 33), 'str', ',int #varname_i#_cb_len')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 22), dict_73235, (isstring_73236, str_73237))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 4), dict_73170, (str_73234, dict_73235))
# Adding element type (key, value) (line 273)
str_73238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 8), 'str', 'strarglens_td')

# Obtaining an instance of the builtin type 'dict' (line 294)
dict_73239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 25), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 294)
# Adding element type (key, value) (line 294)
# Getting the type of 'isstring' (line 294)
isstring_73240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 26), 'isstring')
str_73241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 36), 'str', ',int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 25), dict_73239, (isstring_73240, str_73241))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 4), dict_73170, (str_73238, dict_73239))
# Adding element type (key, value) (line 273)
str_73242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 8), 'str', 'strarglens_nm')

# Obtaining an instance of the builtin type 'dict' (line 296)
dict_73243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 25), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 296)
# Adding element type (key, value) (line 296)
# Getting the type of 'isstring' (line 296)
isstring_73244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 26), 'isstring')
str_73245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 36), 'str', ',#varname_i#_cb_len')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 25), dict_73243, (isstring_73244, str_73245))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 4), dict_73170, (str_73242, dict_73243))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_73088, dict_73170)
# Adding element type (line 257)

# Obtaining an instance of the builtin type 'dict' (line 298)
dict_73246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 4), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 298)
# Adding element type (key, value) (line 298)
str_73247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 8), 'str', 'decl')

# Obtaining an instance of the builtin type 'dict' (line 299)
dict_73248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 299)
# Adding element type (key, value) (line 299)

# Call to l_not(...): (line 299)
# Processing the call arguments (line 299)
# Getting the type of 'isintent_c' (line 299)
isintent_c_73250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 23), 'isintent_c', False)
# Processing the call keyword arguments (line 299)
kwargs_73251 = {}
# Getting the type of 'l_not' (line 299)
l_not_73249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 17), 'l_not', False)
# Calling l_not(args, kwargs) (line 299)
l_not_call_result_73252 = invoke(stypy.reporting.localization.Localization(__file__, 299, 17), l_not_73249, *[isintent_c_73250], **kwargs_73251)

str_73253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 36), 'str', '\t#ctype# #varname_i#=(*#varname_i#_cb_capi);')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 16), dict_73248, (l_not_call_result_73252, str_73253))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 4), dict_73246, (str_73247, dict_73248))
# Adding element type (key, value) (line 298)
str_73254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 8), 'str', 'error')

# Obtaining an instance of the builtin type 'dict' (line 300)
dict_73255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 300)
# Adding element type (key, value) (line 300)

# Call to l_and(...): (line 300)
# Processing the call arguments (line 300)
# Getting the type of 'isintent_c' (line 300)
isintent_c_73257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 24), 'isintent_c', False)
# Getting the type of 'isintent_out' (line 300)
isintent_out_73258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 36), 'isintent_out', False)

# Call to throw_error(...): (line 301)
# Processing the call arguments (line 301)
str_73260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 36), 'str', 'intent(c,out) is forbidden for callback scalar arguments')
# Processing the call keyword arguments (line 301)
kwargs_73261 = {}
# Getting the type of 'throw_error' (line 301)
throw_error_73259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 24), 'throw_error', False)
# Calling throw_error(args, kwargs) (line 301)
throw_error_call_result_73262 = invoke(stypy.reporting.localization.Localization(__file__, 301, 24), throw_error_73259, *[str_73260], **kwargs_73261)

# Processing the call keyword arguments (line 300)
kwargs_73263 = {}
# Getting the type of 'l_and' (line 300)
l_and_73256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 18), 'l_and', False)
# Calling l_and(args, kwargs) (line 300)
l_and_call_result_73264 = invoke(stypy.reporting.localization.Localization(__file__, 300, 18), l_and_73256, *[isintent_c_73257, isintent_out_73258, throw_error_call_result_73262], **kwargs_73263)

str_73265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 18), 'str', '')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 17), dict_73255, (l_and_call_result_73264, str_73265))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 4), dict_73246, (str_73254, dict_73255))
# Adding element type (key, value) (line 298)
str_73266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 8), 'str', 'frompyobj')

# Obtaining an instance of the builtin type 'list' (line 303)
list_73267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 303)
# Adding element type (line 303)

# Obtaining an instance of the builtin type 'dict' (line 303)
dict_73268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 303)
# Adding element type (key, value) (line 303)
# Getting the type of 'debugcapi' (line 303)
debugcapi_73269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 23), 'debugcapi')
str_73270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 34), 'str', '\tCFUNCSMESS("cb:Getting #varname#->");')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 22), dict_73268, (debugcapi_73269, str_73270))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 21), list_73267, dict_73268)
# Adding element type (line 303)

# Obtaining an instance of the builtin type 'dict' (line 304)
dict_73271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 304)
# Adding element type (key, value) (line 304)
# Getting the type of 'isintent_out' (line 304)
isintent_out_73272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 23), 'isintent_out')
str_73273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 23), 'str', '\tif (capi_j>capi_i)\n\t\tGETSCALARFROMPYTUPLE(capi_return,capi_i++,#varname_i#_cb_capi,#ctype#,"#ctype#_from_pyobj failed in converting argument #varname# of call-back function #name# to C #ctype#\\n");')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 22), dict_73271, (isintent_out_73272, str_73273))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 21), list_73267, dict_73271)
# Adding element type (line 303)

# Obtaining an instance of the builtin type 'dict' (line 306)
dict_73274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 306)
# Adding element type (key, value) (line 306)

# Call to l_and(...): (line 306)
# Processing the call arguments (line 306)
# Getting the type of 'debugcapi' (line 306)
debugcapi_73276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 29), 'debugcapi', False)

# Call to l_and(...): (line 306)
# Processing the call arguments (line 306)

# Call to l_not(...): (line 306)
# Processing the call arguments (line 306)
# Getting the type of 'iscomplex' (line 306)
iscomplex_73279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 52), 'iscomplex', False)
# Processing the call keyword arguments (line 306)
kwargs_73280 = {}
# Getting the type of 'l_not' (line 306)
l_not_73278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 46), 'l_not', False)
# Calling l_not(args, kwargs) (line 306)
l_not_call_result_73281 = invoke(stypy.reporting.localization.Localization(__file__, 306, 46), l_not_73278, *[iscomplex_73279], **kwargs_73280)

# Getting the type of 'isintent_c' (line 306)
isintent_c_73282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 64), 'isintent_c', False)
# Processing the call keyword arguments (line 306)
kwargs_73283 = {}
# Getting the type of 'l_and' (line 306)
l_and_73277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 40), 'l_and', False)
# Calling l_and(args, kwargs) (line 306)
l_and_call_result_73284 = invoke(stypy.reporting.localization.Localization(__file__, 306, 40), l_and_73277, *[l_not_call_result_73281, isintent_c_73282], **kwargs_73283)

# Processing the call keyword arguments (line 306)
kwargs_73285 = {}
# Getting the type of 'l_and' (line 306)
l_and_73275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 23), 'l_and', False)
# Calling l_and(args, kwargs) (line 306)
l_and_call_result_73286 = invoke(stypy.reporting.localization.Localization(__file__, 306, 23), l_and_73275, *[debugcapi_73276, l_and_call_result_73284], **kwargs_73285)

str_73287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 26), 'str', '\tfprintf(stderr,"#showvalueformat#.\\n",#varname_i#);')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 22), dict_73274, (l_and_call_result_73286, str_73287))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 21), list_73267, dict_73274)
# Adding element type (line 303)

# Obtaining an instance of the builtin type 'dict' (line 308)
dict_73288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 308)
# Adding element type (key, value) (line 308)

# Call to l_and(...): (line 308)
# Processing the call arguments (line 308)
# Getting the type of 'debugcapi' (line 308)
debugcapi_73290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 29), 'debugcapi', False)

# Call to l_and(...): (line 308)
# Processing the call arguments (line 308)

# Call to l_not(...): (line 308)
# Processing the call arguments (line 308)
# Getting the type of 'iscomplex' (line 308)
iscomplex_73293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 52), 'iscomplex', False)
# Processing the call keyword arguments (line 308)
kwargs_73294 = {}
# Getting the type of 'l_not' (line 308)
l_not_73292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 46), 'l_not', False)
# Calling l_not(args, kwargs) (line 308)
l_not_call_result_73295 = invoke(stypy.reporting.localization.Localization(__file__, 308, 46), l_not_73292, *[iscomplex_73293], **kwargs_73294)


# Call to l_not(...): (line 308)
# Processing the call arguments (line 308)
# Getting the type of 'isintent_c' (line 308)
isintent_c_73297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 71), 'isintent_c', False)
# Processing the call keyword arguments (line 308)
kwargs_73298 = {}
# Getting the type of 'l_not' (line 308)
l_not_73296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 64), 'l_not', False)
# Calling l_not(args, kwargs) (line 308)
l_not_call_result_73299 = invoke(stypy.reporting.localization.Localization(__file__, 308, 64), l_not_73296, *[isintent_c_73297], **kwargs_73298)

# Processing the call keyword arguments (line 308)
kwargs_73300 = {}
# Getting the type of 'l_and' (line 308)
l_and_73291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 40), 'l_and', False)
# Calling l_and(args, kwargs) (line 308)
l_and_call_result_73301 = invoke(stypy.reporting.localization.Localization(__file__, 308, 40), l_and_73291, *[l_not_call_result_73295, l_not_call_result_73299], **kwargs_73300)

# Processing the call keyword arguments (line 308)
kwargs_73302 = {}
# Getting the type of 'l_and' (line 308)
l_and_73289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 23), 'l_and', False)
# Calling l_and(args, kwargs) (line 308)
l_and_call_result_73303 = invoke(stypy.reporting.localization.Localization(__file__, 308, 23), l_and_73289, *[debugcapi_73290, l_and_call_result_73301], **kwargs_73302)

str_73304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 26), 'str', '\tfprintf(stderr,"#showvalueformat#.\\n",*#varname_i#_cb_capi);')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 22), dict_73288, (l_and_call_result_73303, str_73304))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 21), list_73267, dict_73288)
# Adding element type (line 303)

# Obtaining an instance of the builtin type 'dict' (line 310)
dict_73305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 310)
# Adding element type (key, value) (line 310)

# Call to l_and(...): (line 310)
# Processing the call arguments (line 310)
# Getting the type of 'debugcapi' (line 310)
debugcapi_73307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 29), 'debugcapi', False)

# Call to l_and(...): (line 310)
# Processing the call arguments (line 310)
# Getting the type of 'iscomplex' (line 310)
iscomplex_73309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 46), 'iscomplex', False)
# Getting the type of 'isintent_c' (line 310)
isintent_c_73310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 57), 'isintent_c', False)
# Processing the call keyword arguments (line 310)
kwargs_73311 = {}
# Getting the type of 'l_and' (line 310)
l_and_73308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 40), 'l_and', False)
# Calling l_and(args, kwargs) (line 310)
l_and_call_result_73312 = invoke(stypy.reporting.localization.Localization(__file__, 310, 40), l_and_73308, *[iscomplex_73309, isintent_c_73310], **kwargs_73311)

# Processing the call keyword arguments (line 310)
kwargs_73313 = {}
# Getting the type of 'l_and' (line 310)
l_and_73306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 23), 'l_and', False)
# Calling l_and(args, kwargs) (line 310)
l_and_call_result_73314 = invoke(stypy.reporting.localization.Localization(__file__, 310, 23), l_and_73306, *[debugcapi_73307, l_and_call_result_73312], **kwargs_73313)

str_73315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 26), 'str', '\tfprintf(stderr,"#showvalueformat#.\\n",(#varname_i#).r,(#varname_i#).i);')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 22), dict_73305, (l_and_call_result_73314, str_73315))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 21), list_73267, dict_73305)
# Adding element type (line 303)

# Obtaining an instance of the builtin type 'dict' (line 312)
dict_73316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 312)
# Adding element type (key, value) (line 312)

# Call to l_and(...): (line 312)
# Processing the call arguments (line 312)
# Getting the type of 'debugcapi' (line 312)
debugcapi_73318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 29), 'debugcapi', False)

# Call to l_and(...): (line 312)
# Processing the call arguments (line 312)
# Getting the type of 'iscomplex' (line 312)
iscomplex_73320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 46), 'iscomplex', False)

# Call to l_not(...): (line 312)
# Processing the call arguments (line 312)
# Getting the type of 'isintent_c' (line 312)
isintent_c_73322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 64), 'isintent_c', False)
# Processing the call keyword arguments (line 312)
kwargs_73323 = {}
# Getting the type of 'l_not' (line 312)
l_not_73321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 57), 'l_not', False)
# Calling l_not(args, kwargs) (line 312)
l_not_call_result_73324 = invoke(stypy.reporting.localization.Localization(__file__, 312, 57), l_not_73321, *[isintent_c_73322], **kwargs_73323)

# Processing the call keyword arguments (line 312)
kwargs_73325 = {}
# Getting the type of 'l_and' (line 312)
l_and_73319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 40), 'l_and', False)
# Calling l_and(args, kwargs) (line 312)
l_and_call_result_73326 = invoke(stypy.reporting.localization.Localization(__file__, 312, 40), l_and_73319, *[iscomplex_73320, l_not_call_result_73324], **kwargs_73325)

# Processing the call keyword arguments (line 312)
kwargs_73327 = {}
# Getting the type of 'l_and' (line 312)
l_and_73317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 23), 'l_and', False)
# Calling l_and(args, kwargs) (line 312)
l_and_call_result_73328 = invoke(stypy.reporting.localization.Localization(__file__, 312, 23), l_and_73317, *[debugcapi_73318, l_and_call_result_73326], **kwargs_73327)

str_73329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 26), 'str', '\tfprintf(stderr,"#showvalueformat#.\\n",(*#varname_i#_cb_capi).r,(*#varname_i#_cb_capi).i);')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 22), dict_73316, (l_and_call_result_73328, str_73329))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 21), list_73267, dict_73316)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 4), dict_73246, (str_73266, list_73267))
# Adding element type (key, value) (line 298)
str_73330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 8), 'str', 'need')

# Obtaining an instance of the builtin type 'list' (line 315)
list_73331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 315)
# Adding element type (line 315)

# Obtaining an instance of the builtin type 'dict' (line 315)
dict_73332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 315)
# Adding element type (key, value) (line 315)
# Getting the type of 'isintent_out' (line 315)
isintent_out_73333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 18), 'isintent_out')

# Obtaining an instance of the builtin type 'list' (line 315)
list_73334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 32), 'list')
# Adding type elements to the builtin type 'list' instance (line 315)
# Adding element type (line 315)
str_73335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 33), 'str', '#ctype#_from_pyobj')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 32), list_73334, str_73335)
# Adding element type (line 315)
str_73336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 55), 'str', 'GETSCALARFROMPYTUPLE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 32), list_73334, str_73336)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 17), dict_73332, (isintent_out_73333, list_73334))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 16), list_73331, dict_73332)
# Adding element type (line 315)

# Obtaining an instance of the builtin type 'dict' (line 316)
dict_73337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 316)
# Adding element type (key, value) (line 316)
# Getting the type of 'debugcapi' (line 316)
debugcapi_73338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 18), 'debugcapi')
str_73339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 29), 'str', 'CFUNCSMESS')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 17), dict_73337, (debugcapi_73338, str_73339))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 16), list_73331, dict_73337)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 4), dict_73246, (str_73330, list_73331))
# Adding element type (key, value) (line 298)
str_73340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 8), 'str', '_check')
# Getting the type of 'isscalar' (line 317)
isscalar_73341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 18), 'isscalar')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 4), dict_73246, (str_73340, isscalar_73341))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_73088, dict_73246)
# Adding element type (line 257)

# Obtaining an instance of the builtin type 'dict' (line 318)
dict_73342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 7), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 318)
# Adding element type (key, value) (line 318)
str_73343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 8), 'str', 'pyobjfrom')

# Obtaining an instance of the builtin type 'list' (line 319)
list_73344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 319)
# Adding element type (line 319)

# Obtaining an instance of the builtin type 'dict' (line 319)
dict_73345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 319)
# Adding element type (key, value) (line 319)
# Getting the type of 'isintent_in' (line 319)
isintent_in_73346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 23), 'isintent_in')
str_73347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, (-1)), 'str', '\tif (#name#_nofargs>capi_i)\n\t\tif (PyTuple_SetItem((PyObject *)capi_arglist,capi_i++,pyobj_from_#ctype#1(#varname_i#)))\n\t\t\tgoto capi_fail;')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 22), dict_73345, (isintent_in_73346, str_73347))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 21), list_73344, dict_73345)
# Adding element type (line 319)

# Obtaining an instance of the builtin type 'dict' (line 323)
dict_73348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 323)
# Adding element type (key, value) (line 323)
# Getting the type of 'isintent_inout' (line 323)
isintent_inout_73349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 23), 'isintent_inout')
str_73350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, (-1)), 'str', '\tif (#name#_nofargs>capi_i)\n\t\tif (PyTuple_SetItem((PyObject *)capi_arglist,capi_i++,pyarr_from_p_#ctype#1(#varname_i#_cb_capi)))\n\t\t\tgoto capi_fail;')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 22), dict_73348, (isintent_inout_73349, str_73350))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 21), list_73344, dict_73348)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 7), dict_73342, (str_73343, list_73344))
# Adding element type (key, value) (line 318)
str_73351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 8), 'str', 'need')

# Obtaining an instance of the builtin type 'list' (line 327)
list_73352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 327)
# Adding element type (line 327)

# Obtaining an instance of the builtin type 'dict' (line 327)
dict_73353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 327)
# Adding element type (key, value) (line 327)
# Getting the type of 'isintent_in' (line 327)
isintent_in_73354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 18), 'isintent_in')
str_73355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 31), 'str', 'pyobj_from_#ctype#1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 17), dict_73353, (isintent_in_73354, str_73355))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 16), list_73352, dict_73353)
# Adding element type (line 327)

# Obtaining an instance of the builtin type 'dict' (line 328)
dict_73356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 328)
# Adding element type (key, value) (line 328)
# Getting the type of 'isintent_inout' (line 328)
isintent_inout_73357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 18), 'isintent_inout')
str_73358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 34), 'str', 'pyarr_from_p_#ctype#1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 17), dict_73356, (isintent_inout_73357, str_73358))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 16), list_73352, dict_73356)
# Adding element type (line 327)

# Obtaining an instance of the builtin type 'dict' (line 329)
dict_73359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 329)
# Adding element type (key, value) (line 329)
# Getting the type of 'iscomplex' (line 329)
iscomplex_73360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 18), 'iscomplex')
str_73361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 29), 'str', '#ctype#')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 17), dict_73359, (iscomplex_73360, str_73361))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 16), list_73352, dict_73359)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 7), dict_73342, (str_73351, list_73352))
# Adding element type (key, value) (line 318)
str_73362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 8), 'str', '_check')

# Call to l_and(...): (line 330)
# Processing the call arguments (line 330)
# Getting the type of 'isscalar' (line 330)
isscalar_73364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 24), 'isscalar', False)
# Getting the type of 'isintent_nothide' (line 330)
isintent_nothide_73365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 34), 'isintent_nothide', False)
# Processing the call keyword arguments (line 330)
kwargs_73366 = {}
# Getting the type of 'l_and' (line 330)
l_and_73363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 18), 'l_and', False)
# Calling l_and(args, kwargs) (line 330)
l_and_call_result_73367 = invoke(stypy.reporting.localization.Localization(__file__, 330, 18), l_and_73363, *[isscalar_73364, isintent_nothide_73365], **kwargs_73366)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 7), dict_73342, (str_73362, l_and_call_result_73367))
# Adding element type (key, value) (line 318)
str_73368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 8), 'str', '_optional')
str_73369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 21), 'str', '')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 7), dict_73342, (str_73368, str_73369))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_73088, dict_73342)
# Adding element type (line 257)

# Obtaining an instance of the builtin type 'dict' (line 332)
dict_73370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 7), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 332)
# Adding element type (key, value) (line 332)
str_73371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 8), 'str', 'frompyobj')

# Obtaining an instance of the builtin type 'list' (line 333)
list_73372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 333)
# Adding element type (line 333)

# Obtaining an instance of the builtin type 'dict' (line 333)
dict_73373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 333)
# Adding element type (key, value) (line 333)
# Getting the type of 'debugcapi' (line 333)
debugcapi_73374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 23), 'debugcapi')
str_73375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 34), 'str', '\tCFUNCSMESS("cb:Getting #varname#->\\"");')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 22), dict_73373, (debugcapi_73374, str_73375))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 21), list_73372, dict_73373)
# Adding element type (line 333)
str_73376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, (-1)), 'str', '\tif (capi_j>capi_i)\n\t\tGETSTRFROMPYTUPLE(capi_return,capi_i++,#varname_i#,#varname_i#_cb_len);')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 21), list_73372, str_73376)
# Adding element type (line 333)

# Obtaining an instance of the builtin type 'dict' (line 336)
dict_73377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 336)
# Adding element type (key, value) (line 336)
# Getting the type of 'debugcapi' (line 336)
debugcapi_73378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 23), 'debugcapi')
str_73379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 23), 'str', '\tfprintf(stderr,"#showvalueformat#\\":%d:.\\n",#varname_i#,#varname_i#_cb_len);')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 22), dict_73377, (debugcapi_73378, str_73379))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 21), list_73372, dict_73377)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 7), dict_73370, (str_73371, list_73372))
# Adding element type (key, value) (line 332)
str_73380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 8), 'str', 'need')

# Obtaining an instance of the builtin type 'list' (line 339)
list_73381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 339)
# Adding element type (line 339)
str_73382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 17), 'str', '#ctype#')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 16), list_73381, str_73382)
# Adding element type (line 339)
str_73383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 28), 'str', 'GETSTRFROMPYTUPLE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 16), list_73381, str_73383)
# Adding element type (line 339)

# Obtaining an instance of the builtin type 'dict' (line 340)
dict_73384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 340)
# Adding element type (key, value) (line 340)
# Getting the type of 'debugcapi' (line 340)
debugcapi_73385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 18), 'debugcapi')
str_73386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 29), 'str', 'CFUNCSMESS')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 17), dict_73384, (debugcapi_73385, str_73386))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 16), list_73381, dict_73384)
# Adding element type (line 339)
str_73387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 44), 'str', 'string.h')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 16), list_73381, str_73387)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 7), dict_73370, (str_73380, list_73381))
# Adding element type (key, value) (line 332)
str_73388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 8), 'str', '_check')

# Call to l_and(...): (line 341)
# Processing the call arguments (line 341)
# Getting the type of 'isstring' (line 341)
isstring_73390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 24), 'isstring', False)
# Getting the type of 'isintent_out' (line 341)
isintent_out_73391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 34), 'isintent_out', False)
# Processing the call keyword arguments (line 341)
kwargs_73392 = {}
# Getting the type of 'l_and' (line 341)
l_and_73389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 18), 'l_and', False)
# Calling l_and(args, kwargs) (line 341)
l_and_call_result_73393 = invoke(stypy.reporting.localization.Localization(__file__, 341, 18), l_and_73389, *[isstring_73390, isintent_out_73391], **kwargs_73392)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 7), dict_73370, (str_73388, l_and_call_result_73393))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_73088, dict_73370)
# Adding element type (line 257)

# Obtaining an instance of the builtin type 'dict' (line 342)
dict_73394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 7), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 342)
# Adding element type (key, value) (line 342)
str_73395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 8), 'str', 'pyobjfrom')

# Obtaining an instance of the builtin type 'list' (line 343)
list_73396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 343)
# Adding element type (line 343)

# Obtaining an instance of the builtin type 'dict' (line 343)
dict_73397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 343)
# Adding element type (key, value) (line 343)
# Getting the type of 'debugcapi' (line 343)
debugcapi_73398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 23), 'debugcapi')
str_73399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 34), 'str', '\tfprintf(stderr,"debug-capi:cb:#varname#=\\"#showvalueformat#\\":%d:\\n",#varname_i#,#varname_i#_cb_len);')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 22), dict_73397, (debugcapi_73398, str_73399))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 21), list_73396, dict_73397)
# Adding element type (line 343)

# Obtaining an instance of the builtin type 'dict' (line 344)
dict_73400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 344)
# Adding element type (key, value) (line 344)
# Getting the type of 'isintent_in' (line 344)
isintent_in_73401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 23), 'isintent_in')
str_73402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, (-1)), 'str', '\tif (#name#_nofargs>capi_i)\n\t\tif (PyTuple_SetItem((PyObject *)capi_arglist,capi_i++,pyobj_from_#ctype#1size(#varname_i#,#varname_i#_cb_len)))\n\t\t\tgoto capi_fail;')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 22), dict_73400, (isintent_in_73401, str_73402))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 21), list_73396, dict_73400)
# Adding element type (line 343)

# Obtaining an instance of the builtin type 'dict' (line 348)
dict_73403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 348)
# Adding element type (key, value) (line 348)
# Getting the type of 'isintent_inout' (line 348)
isintent_inout_73404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 23), 'isintent_inout')
str_73405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, (-1)), 'str', '\tif (#name#_nofargs>capi_i) {\n\t\tint #varname_i#_cb_dims[] = {#varname_i#_cb_len};\n\t\tif (PyTuple_SetItem((PyObject *)capi_arglist,capi_i++,pyarr_from_p_#ctype#1(#varname_i#,#varname_i#_cb_dims)))\n\t\t\tgoto capi_fail;\n\t}')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 22), dict_73403, (isintent_inout_73404, str_73405))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 21), list_73396, dict_73403)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 7), dict_73394, (str_73395, list_73396))
# Adding element type (key, value) (line 342)
str_73406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 8), 'str', 'need')

# Obtaining an instance of the builtin type 'list' (line 354)
list_73407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 354)
# Adding element type (line 354)

# Obtaining an instance of the builtin type 'dict' (line 354)
dict_73408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 354)
# Adding element type (key, value) (line 354)
# Getting the type of 'isintent_in' (line 354)
isintent_in_73409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 18), 'isintent_in')
str_73410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 31), 'str', 'pyobj_from_#ctype#1size')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 17), dict_73408, (isintent_in_73409, str_73410))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 16), list_73407, dict_73408)
# Adding element type (line 354)

# Obtaining an instance of the builtin type 'dict' (line 355)
dict_73411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 355)
# Adding element type (key, value) (line 355)
# Getting the type of 'isintent_inout' (line 355)
isintent_inout_73412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 18), 'isintent_inout')
str_73413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 34), 'str', 'pyarr_from_p_#ctype#1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 17), dict_73411, (isintent_inout_73412, str_73413))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 16), list_73407, dict_73411)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 7), dict_73394, (str_73406, list_73407))
# Adding element type (key, value) (line 342)
str_73414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 8), 'str', '_check')

# Call to l_and(...): (line 356)
# Processing the call arguments (line 356)
# Getting the type of 'isstring' (line 356)
isstring_73416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 24), 'isstring', False)
# Getting the type of 'isintent_nothide' (line 356)
isintent_nothide_73417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 34), 'isintent_nothide', False)
# Processing the call keyword arguments (line 356)
kwargs_73418 = {}
# Getting the type of 'l_and' (line 356)
l_and_73415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 18), 'l_and', False)
# Calling l_and(args, kwargs) (line 356)
l_and_call_result_73419 = invoke(stypy.reporting.localization.Localization(__file__, 356, 18), l_and_73415, *[isstring_73416, isintent_nothide_73417], **kwargs_73418)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 7), dict_73394, (str_73414, l_and_call_result_73419))
# Adding element type (key, value) (line 342)
str_73420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 8), 'str', '_optional')
str_73421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 21), 'str', '')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 7), dict_73394, (str_73420, str_73421))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_73088, dict_73394)
# Adding element type (line 257)

# Obtaining an instance of the builtin type 'dict' (line 360)
dict_73422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 4), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 360)
# Adding element type (key, value) (line 360)
str_73423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 8), 'str', 'decl')
str_73424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 16), 'str', '\tnpy_intp #varname_i#_Dims[#rank#] = {#rank*[-1]#};')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 4), dict_73422, (str_73423, str_73424))
# Adding element type (key, value) (line 360)
str_73425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 8), 'str', 'setdims')
str_73426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 19), 'str', '\t#cbsetdims#;')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 4), dict_73422, (str_73425, str_73426))
# Adding element type (key, value) (line 360)
str_73427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 8), 'str', '_check')
# Getting the type of 'isarray' (line 363)
isarray_73428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 18), 'isarray')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 4), dict_73422, (str_73427, isarray_73428))
# Adding element type (key, value) (line 360)
str_73429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 8), 'str', '_depend')
str_73430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 19), 'str', '')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 4), dict_73422, (str_73429, str_73430))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_73088, dict_73422)
# Adding element type (line 257)

# Obtaining an instance of the builtin type 'dict' (line 366)
dict_73431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 4), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 366)
# Adding element type (key, value) (line 366)
str_73432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 8), 'str', 'pyobjfrom')

# Obtaining an instance of the builtin type 'list' (line 367)
list_73433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 367)
# Adding element type (line 367)

# Obtaining an instance of the builtin type 'dict' (line 367)
dict_73434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 367)
# Adding element type (key, value) (line 367)
# Getting the type of 'debugcapi' (line 367)
debugcapi_73435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 23), 'debugcapi')
str_73436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 34), 'str', '\tfprintf(stderr,"debug-capi:cb:#varname#\\n");')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 22), dict_73434, (debugcapi_73435, str_73436))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 21), list_73433, dict_73434)
# Adding element type (line 367)

# Obtaining an instance of the builtin type 'dict' (line 368)
dict_73437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 368)
# Adding element type (key, value) (line 368)
# Getting the type of 'isintent_c' (line 368)
isintent_c_73438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 23), 'isintent_c')
str_73439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, (-1)), 'str', '\tif (#name#_nofargs>capi_i) {\n\t\tPyArrayObject *tmp_arr = (PyArrayObject *)PyArray_New(&PyArray_Type,#rank#,#varname_i#_Dims,#atype#,NULL,(char*)#varname_i#,0,NPY_ARRAY_CARRAY,NULL); /*XXX: Hmm, what will destroy this array??? */\n')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 22), dict_73437, (isintent_c_73438, str_73439))
# Adding element type (key, value) (line 368)

# Call to l_not(...): (line 372)
# Processing the call arguments (line 372)
# Getting the type of 'isintent_c' (line 372)
isintent_c_73441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 29), 'isintent_c', False)
# Processing the call keyword arguments (line 372)
kwargs_73442 = {}
# Getting the type of 'l_not' (line 372)
l_not_73440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 23), 'l_not', False)
# Calling l_not(args, kwargs) (line 372)
l_not_call_result_73443 = invoke(stypy.reporting.localization.Localization(__file__, 372, 23), l_not_73440, *[isintent_c_73441], **kwargs_73442)

str_73444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, (-1)), 'str', '\tif (#name#_nofargs>capi_i) {\n\t\tPyArrayObject *tmp_arr = (PyArrayObject *)PyArray_New(&PyArray_Type,#rank#,#varname_i#_Dims,#atype#,NULL,(char*)#varname_i#,0,NPY_ARRAY_FARRAY,NULL); /*XXX: Hmm, what will destroy this array??? */\n')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 22), dict_73437, (l_not_call_result_73443, str_73444))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 21), list_73433, dict_73437)
# Adding element type (line 367)
str_73445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, (-1)), 'str', '\n\t\tif (tmp_arr==NULL)\n\t\t\tgoto capi_fail;\n\t\tif (PyTuple_SetItem((PyObject *)capi_arglist,capi_i++,(PyObject *)tmp_arr))\n\t\t\tgoto capi_fail;\n}')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 21), list_73433, str_73445)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 4), dict_73431, (str_73432, list_73433))
# Adding element type (key, value) (line 366)
str_73446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 8), 'str', '_check')

# Call to l_and(...): (line 383)
# Processing the call arguments (line 383)
# Getting the type of 'isarray' (line 383)
isarray_73448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 24), 'isarray', False)
# Getting the type of 'isintent_nothide' (line 383)
isintent_nothide_73449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 33), 'isintent_nothide', False)

# Call to l_or(...): (line 383)
# Processing the call arguments (line 383)
# Getting the type of 'isintent_in' (line 383)
isintent_in_73451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 56), 'isintent_in', False)
# Getting the type of 'isintent_inout' (line 383)
isintent_inout_73452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 69), 'isintent_inout', False)
# Processing the call keyword arguments (line 383)
kwargs_73453 = {}
# Getting the type of 'l_or' (line 383)
l_or_73450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 51), 'l_or', False)
# Calling l_or(args, kwargs) (line 383)
l_or_call_result_73454 = invoke(stypy.reporting.localization.Localization(__file__, 383, 51), l_or_73450, *[isintent_in_73451, isintent_inout_73452], **kwargs_73453)

# Processing the call keyword arguments (line 383)
kwargs_73455 = {}
# Getting the type of 'l_and' (line 383)
l_and_73447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 18), 'l_and', False)
# Calling l_and(args, kwargs) (line 383)
l_and_call_result_73456 = invoke(stypy.reporting.localization.Localization(__file__, 383, 18), l_and_73447, *[isarray_73448, isintent_nothide_73449, l_or_call_result_73454], **kwargs_73455)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 4), dict_73431, (str_73446, l_and_call_result_73456))
# Adding element type (key, value) (line 366)
str_73457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 8), 'str', '_optional')
str_73458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 21), 'str', '')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 4), dict_73431, (str_73457, str_73458))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_73088, dict_73431)
# Adding element type (line 257)

# Obtaining an instance of the builtin type 'dict' (line 385)
dict_73459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 7), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 385)
# Adding element type (key, value) (line 385)
str_73460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 8), 'str', 'frompyobj')

# Obtaining an instance of the builtin type 'list' (line 386)
list_73461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 386)
# Adding element type (line 386)

# Obtaining an instance of the builtin type 'dict' (line 386)
dict_73462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 386)
# Adding element type (key, value) (line 386)
# Getting the type of 'debugcapi' (line 386)
debugcapi_73463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 23), 'debugcapi')
str_73464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 34), 'str', '\tCFUNCSMESS("cb:Getting #varname#->");')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 22), dict_73462, (debugcapi_73463, str_73464))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 21), list_73461, dict_73462)
# Adding element type (line 386)
str_73465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, (-1)), 'str', '\tif (capi_j>capi_i) {\n\t\tPyArrayObject *rv_cb_arr = NULL;\n\t\tif ((capi_tmp = PyTuple_GetItem(capi_return,capi_i++))==NULL) goto capi_fail;\n\t\trv_cb_arr =  array_from_pyobj(#atype#,#varname_i#_Dims,#rank#,F2PY_INTENT_IN')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 21), list_73461, str_73465)
# Adding element type (line 386)

# Obtaining an instance of the builtin type 'dict' (line 391)
dict_73466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 391)
# Adding element type (key, value) (line 391)
# Getting the type of 'isintent_c' (line 391)
isintent_c_73467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 23), 'isintent_c')
str_73468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 35), 'str', '|F2PY_INTENT_C')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 22), dict_73466, (isintent_c_73467, str_73468))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 21), list_73461, dict_73466)
# Adding element type (line 386)
str_73469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, (-1)), 'str', ',capi_tmp);\n\t\tif (rv_cb_arr == NULL) {\n\t\t\tfprintf(stderr,"rv_cb_arr is NULL\\n");\n\t\t\tgoto capi_fail;\n\t\t}\n\t\tMEMCOPY(#varname_i#,PyArray_DATA(rv_cb_arr),PyArray_NBYTES(rv_cb_arr));\n\t\tif (capi_tmp != (PyObject *)rv_cb_arr) {\n\t\t\tPy_DECREF(rv_cb_arr);\n\t\t}\n\t}')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 21), list_73461, str_73469)
# Adding element type (line 386)

# Obtaining an instance of the builtin type 'dict' (line 402)
dict_73470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 402)
# Adding element type (key, value) (line 402)
# Getting the type of 'debugcapi' (line 402)
debugcapi_73471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 23), 'debugcapi')
str_73472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 34), 'str', '\tfprintf(stderr,"<-.\\n");')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 22), dict_73470, (debugcapi_73471, str_73472))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 21), list_73461, dict_73470)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 7), dict_73459, (str_73460, list_73461))
# Adding element type (key, value) (line 385)
str_73473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 8), 'str', 'need')

# Obtaining an instance of the builtin type 'list' (line 404)
list_73474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 404)
# Adding element type (line 404)
str_73475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 17), 'str', 'MEMCOPY')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 16), list_73474, str_73475)
# Adding element type (line 404)

# Obtaining an instance of the builtin type 'dict' (line 404)
dict_73476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 28), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 404)
# Adding element type (key, value) (line 404)
# Getting the type of 'iscomplexarray' (line 404)
iscomplexarray_73477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 29), 'iscomplexarray')
str_73478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 45), 'str', '#ctype#')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 28), dict_73476, (iscomplexarray_73477, str_73478))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 16), list_73474, dict_73476)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 7), dict_73459, (str_73473, list_73474))
# Adding element type (key, value) (line 385)
str_73479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 8), 'str', '_check')

# Call to l_and(...): (line 405)
# Processing the call arguments (line 405)
# Getting the type of 'isarray' (line 405)
isarray_73481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 24), 'isarray', False)
# Getting the type of 'isintent_out' (line 405)
isintent_out_73482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 33), 'isintent_out', False)
# Processing the call keyword arguments (line 405)
kwargs_73483 = {}
# Getting the type of 'l_and' (line 405)
l_and_73480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 18), 'l_and', False)
# Calling l_and(args, kwargs) (line 405)
l_and_call_result_73484 = invoke(stypy.reporting.localization.Localization(__file__, 405, 18), l_and_73480, *[isarray_73481, isintent_out_73482], **kwargs_73483)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 7), dict_73459, (str_73479, l_and_call_result_73484))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_73088, dict_73459)
# Adding element type (line 257)

# Obtaining an instance of the builtin type 'dict' (line 406)
dict_73485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 7), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 406)
# Adding element type (key, value) (line 406)
str_73486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 8), 'str', 'docreturn')
str_73487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 21), 'str', '#varname#,')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 7), dict_73485, (str_73486, str_73487))
# Adding element type (key, value) (line 406)
str_73488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 8), 'str', '_check')
# Getting the type of 'isintent_out' (line 408)
isintent_out_73489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 18), 'isintent_out')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 7), dict_73485, (str_73488, isintent_out_73489))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_73088, dict_73485)

# Assigning a type to the variable 'cb_arg_rules' (line 257)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 0), 'cb_arg_rules', list_73088)

# Assigning a Dict to a Name (line 413):

# Assigning a Dict to a Name (line 413):

# Obtaining an instance of the builtin type 'dict' (line 413)
dict_73490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 413)

# Assigning a type to the variable 'cb_map' (line 413)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 0), 'cb_map', dict_73490)

@norecursion
def buildcallbacks(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'buildcallbacks'
    module_type_store = module_type_store.open_function_context('buildcallbacks', 416, 0, False)
    
    # Passed parameters checking function
    buildcallbacks.stypy_localization = localization
    buildcallbacks.stypy_type_of_self = None
    buildcallbacks.stypy_type_store = module_type_store
    buildcallbacks.stypy_function_name = 'buildcallbacks'
    buildcallbacks.stypy_param_names_list = ['m']
    buildcallbacks.stypy_varargs_param_name = None
    buildcallbacks.stypy_kwargs_param_name = None
    buildcallbacks.stypy_call_defaults = defaults
    buildcallbacks.stypy_call_varargs = varargs
    buildcallbacks.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'buildcallbacks', ['m'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'buildcallbacks', localization, ['m'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'buildcallbacks(...)' code ##################

    # Marking variables as global (line 417)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 417, 4), 'cb_map')
    
    # Assigning a List to a Subscript (line 418):
    
    # Assigning a List to a Subscript (line 418):
    
    # Obtaining an instance of the builtin type 'list' (line 418)
    list_73491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 418)
    
    # Getting the type of 'cb_map' (line 418)
    cb_map_73492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'cb_map')
    
    # Obtaining the type of the subscript
    str_73493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 13), 'str', 'name')
    # Getting the type of 'm' (line 418)
    m_73494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 11), 'm')
    # Obtaining the member '__getitem__' of a type (line 418)
    getitem___73495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 11), m_73494, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 418)
    subscript_call_result_73496 = invoke(stypy.reporting.localization.Localization(__file__, 418, 11), getitem___73495, str_73493)
    
    # Storing an element on a container (line 418)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 4), cb_map_73492, (subscript_call_result_73496, list_73491))
    
    
    # Obtaining the type of the subscript
    str_73497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 16), 'str', 'body')
    # Getting the type of 'm' (line 419)
    m_73498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 14), 'm')
    # Obtaining the member '__getitem__' of a type (line 419)
    getitem___73499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 14), m_73498, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 419)
    subscript_call_result_73500 = invoke(stypy.reporting.localization.Localization(__file__, 419, 14), getitem___73499, str_73497)
    
    # Testing the type of a for loop iterable (line 419)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 419, 4), subscript_call_result_73500)
    # Getting the type of the for loop variable (line 419)
    for_loop_var_73501 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 419, 4), subscript_call_result_73500)
    # Assigning a type to the variable 'bi' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'bi', for_loop_var_73501)
    # SSA begins for a for statement (line 419)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    str_73502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 14), 'str', 'block')
    # Getting the type of 'bi' (line 420)
    bi_73503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 11), 'bi')
    # Obtaining the member '__getitem__' of a type (line 420)
    getitem___73504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 11), bi_73503, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 420)
    subscript_call_result_73505 = invoke(stypy.reporting.localization.Localization(__file__, 420, 11), getitem___73504, str_73502)
    
    str_73506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 26), 'str', 'interface')
    # Applying the binary operator '==' (line 420)
    result_eq_73507 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 11), '==', subscript_call_result_73505, str_73506)
    
    # Testing the type of an if condition (line 420)
    if_condition_73508 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 420, 8), result_eq_73507)
    # Assigning a type to the variable 'if_condition_73508' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'if_condition_73508', if_condition_73508)
    # SSA begins for if statement (line 420)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Obtaining the type of the subscript
    str_73509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 24), 'str', 'body')
    # Getting the type of 'bi' (line 421)
    bi_73510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 21), 'bi')
    # Obtaining the member '__getitem__' of a type (line 421)
    getitem___73511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 21), bi_73510, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 421)
    subscript_call_result_73512 = invoke(stypy.reporting.localization.Localization(__file__, 421, 21), getitem___73511, str_73509)
    
    # Testing the type of a for loop iterable (line 421)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 421, 12), subscript_call_result_73512)
    # Getting the type of the for loop variable (line 421)
    for_loop_var_73513 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 421, 12), subscript_call_result_73512)
    # Assigning a type to the variable 'b' (line 421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 12), 'b', for_loop_var_73513)
    # SSA begins for a for statement (line 421)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'b' (line 422)
    b_73514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 19), 'b')
    # Testing the type of an if condition (line 422)
    if_condition_73515 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 422, 16), b_73514)
    # Assigning a type to the variable 'if_condition_73515' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 16), 'if_condition_73515', if_condition_73515)
    # SSA begins for if statement (line 422)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to buildcallback(...): (line 423)
    # Processing the call arguments (line 423)
    # Getting the type of 'b' (line 423)
    b_73517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 34), 'b', False)
    
    # Obtaining the type of the subscript
    str_73518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 39), 'str', 'name')
    # Getting the type of 'm' (line 423)
    m_73519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 37), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 423)
    getitem___73520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 37), m_73519, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 423)
    subscript_call_result_73521 = invoke(stypy.reporting.localization.Localization(__file__, 423, 37), getitem___73520, str_73518)
    
    # Processing the call keyword arguments (line 423)
    kwargs_73522 = {}
    # Getting the type of 'buildcallback' (line 423)
    buildcallback_73516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 20), 'buildcallback', False)
    # Calling buildcallback(args, kwargs) (line 423)
    buildcallback_call_result_73523 = invoke(stypy.reporting.localization.Localization(__file__, 423, 20), buildcallback_73516, *[b_73517, subscript_call_result_73521], **kwargs_73522)
    
    # SSA branch for the else part of an if statement (line 422)
    module_type_store.open_ssa_branch('else')
    
    # Call to errmess(...): (line 425)
    # Processing the call arguments (line 425)
    str_73525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 28), 'str', 'warning: empty body for %s\n')
    
    # Obtaining the type of the subscript
    str_73526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 64), 'str', 'name')
    # Getting the type of 'm' (line 425)
    m_73527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 62), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 425)
    getitem___73528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 62), m_73527, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 425)
    subscript_call_result_73529 = invoke(stypy.reporting.localization.Localization(__file__, 425, 62), getitem___73528, str_73526)
    
    # Applying the binary operator '%' (line 425)
    result_mod_73530 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 28), '%', str_73525, subscript_call_result_73529)
    
    # Processing the call keyword arguments (line 425)
    kwargs_73531 = {}
    # Getting the type of 'errmess' (line 425)
    errmess_73524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 20), 'errmess', False)
    # Calling errmess(args, kwargs) (line 425)
    errmess_call_result_73532 = invoke(stypy.reporting.localization.Localization(__file__, 425, 20), errmess_73524, *[result_mod_73530], **kwargs_73531)
    
    # SSA join for if statement (line 422)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 420)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'buildcallbacks(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'buildcallbacks' in the type store
    # Getting the type of 'stypy_return_type' (line 416)
    stypy_return_type_73533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_73533)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'buildcallbacks'
    return stypy_return_type_73533

# Assigning a type to the variable 'buildcallbacks' (line 416)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 0), 'buildcallbacks', buildcallbacks)

@norecursion
def buildcallback(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'buildcallback'
    module_type_store = module_type_store.open_function_context('buildcallback', 428, 0, False)
    
    # Passed parameters checking function
    buildcallback.stypy_localization = localization
    buildcallback.stypy_type_of_self = None
    buildcallback.stypy_type_store = module_type_store
    buildcallback.stypy_function_name = 'buildcallback'
    buildcallback.stypy_param_names_list = ['rout', 'um']
    buildcallback.stypy_varargs_param_name = None
    buildcallback.stypy_kwargs_param_name = None
    buildcallback.stypy_call_defaults = defaults
    buildcallback.stypy_call_varargs = varargs
    buildcallback.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'buildcallback', ['rout', 'um'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'buildcallback', localization, ['rout', 'um'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'buildcallback(...)' code ##################

    # Marking variables as global (line 429)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 429, 4), 'cb_map')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 430, 4))
    
    # 'from numpy.f2py import capi_maps' statement (line 430)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
    import_73534 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 430, 4), 'numpy.f2py')

    if (type(import_73534) is not StypyTypeError):

        if (import_73534 != 'pyd_module'):
            __import__(import_73534)
            sys_modules_73535 = sys.modules[import_73534]
            import_from_module(stypy.reporting.localization.Localization(__file__, 430, 4), 'numpy.f2py', sys_modules_73535.module_type_store, module_type_store, ['capi_maps'])
            nest_module(stypy.reporting.localization.Localization(__file__, 430, 4), __file__, sys_modules_73535, sys_modules_73535.module_type_store, module_type_store)
        else:
            from numpy.f2py import capi_maps

            import_from_module(stypy.reporting.localization.Localization(__file__, 430, 4), 'numpy.f2py', None, module_type_store, ['capi_maps'], [capi_maps])

    else:
        # Assigning a type to the variable 'numpy.f2py' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'numpy.f2py', import_73534)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')
    
    
    # Call to outmess(...): (line 432)
    # Processing the call arguments (line 432)
    str_73537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 12), 'str', '\tConstructing call-back function "cb_%s_in_%s"\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 433)
    tuple_73538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 433)
    # Adding element type (line 433)
    
    # Obtaining the type of the subscript
    str_73539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 18), 'str', 'name')
    # Getting the type of 'rout' (line 433)
    rout_73540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 13), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 433)
    getitem___73541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 13), rout_73540, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 433)
    subscript_call_result_73542 = invoke(stypy.reporting.localization.Localization(__file__, 433, 13), getitem___73541, str_73539)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 13), tuple_73538, subscript_call_result_73542)
    # Adding element type (line 433)
    # Getting the type of 'um' (line 433)
    um_73543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 27), 'um', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 13), tuple_73538, um_73543)
    
    # Applying the binary operator '%' (line 432)
    result_mod_73544 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 12), '%', str_73537, tuple_73538)
    
    # Processing the call keyword arguments (line 432)
    kwargs_73545 = {}
    # Getting the type of 'outmess' (line 432)
    outmess_73536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 'outmess', False)
    # Calling outmess(args, kwargs) (line 432)
    outmess_call_result_73546 = invoke(stypy.reporting.localization.Localization(__file__, 432, 4), outmess_73536, *[result_mod_73544], **kwargs_73545)
    
    
    # Assigning a Call to a Tuple (line 434):
    
    # Assigning a Call to a Name:
    
    # Call to getargs(...): (line 434)
    # Processing the call arguments (line 434)
    # Getting the type of 'rout' (line 434)
    rout_73548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 28), 'rout', False)
    # Processing the call keyword arguments (line 434)
    kwargs_73549 = {}
    # Getting the type of 'getargs' (line 434)
    getargs_73547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 20), 'getargs', False)
    # Calling getargs(args, kwargs) (line 434)
    getargs_call_result_73550 = invoke(stypy.reporting.localization.Localization(__file__, 434, 20), getargs_73547, *[rout_73548], **kwargs_73549)
    
    # Assigning a type to the variable 'call_assignment_72837' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'call_assignment_72837', getargs_call_result_73550)
    
    # Assigning a Call to a Name (line 434):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_73553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 4), 'int')
    # Processing the call keyword arguments
    kwargs_73554 = {}
    # Getting the type of 'call_assignment_72837' (line 434)
    call_assignment_72837_73551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'call_assignment_72837', False)
    # Obtaining the member '__getitem__' of a type (line 434)
    getitem___73552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 4), call_assignment_72837_73551, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_73555 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___73552, *[int_73553], **kwargs_73554)
    
    # Assigning a type to the variable 'call_assignment_72838' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'call_assignment_72838', getitem___call_result_73555)
    
    # Assigning a Name to a Name (line 434):
    # Getting the type of 'call_assignment_72838' (line 434)
    call_assignment_72838_73556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'call_assignment_72838')
    # Assigning a type to the variable 'args' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'args', call_assignment_72838_73556)
    
    # Assigning a Call to a Name (line 434):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_73559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 4), 'int')
    # Processing the call keyword arguments
    kwargs_73560 = {}
    # Getting the type of 'call_assignment_72837' (line 434)
    call_assignment_72837_73557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'call_assignment_72837', False)
    # Obtaining the member '__getitem__' of a type (line 434)
    getitem___73558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 4), call_assignment_72837_73557, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_73561 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___73558, *[int_73559], **kwargs_73560)
    
    # Assigning a type to the variable 'call_assignment_72839' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'call_assignment_72839', getitem___call_result_73561)
    
    # Assigning a Name to a Name (line 434):
    # Getting the type of 'call_assignment_72839' (line 434)
    call_assignment_72839_73562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'call_assignment_72839')
    # Assigning a type to the variable 'depargs' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 10), 'depargs', call_assignment_72839_73562)
    
    # Assigning a Name to a Attribute (line 435):
    
    # Assigning a Name to a Attribute (line 435):
    # Getting the type of 'depargs' (line 435)
    depargs_73563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 24), 'depargs')
    # Getting the type of 'capi_maps' (line 435)
    capi_maps_73564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'capi_maps')
    # Setting the type of the member 'depargs' of a type (line 435)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 4), capi_maps_73564, 'depargs', depargs_73563)
    
    # Assigning a Subscript to a Name (line 436):
    
    # Assigning a Subscript to a Name (line 436):
    
    # Obtaining the type of the subscript
    str_73565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 15), 'str', 'vars')
    # Getting the type of 'rout' (line 436)
    rout_73566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 10), 'rout')
    # Obtaining the member '__getitem__' of a type (line 436)
    getitem___73567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 10), rout_73566, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 436)
    subscript_call_result_73568 = invoke(stypy.reporting.localization.Localization(__file__, 436, 10), getitem___73567, str_73565)
    
    # Assigning a type to the variable 'var' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'var', subscript_call_result_73568)
    
    # Assigning a Call to a Name (line 437):
    
    # Assigning a Call to a Name (line 437):
    
    # Call to cb_routsign2map(...): (line 437)
    # Processing the call arguments (line 437)
    # Getting the type of 'rout' (line 437)
    rout_73571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 36), 'rout', False)
    # Getting the type of 'um' (line 437)
    um_73572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 42), 'um', False)
    # Processing the call keyword arguments (line 437)
    kwargs_73573 = {}
    # Getting the type of 'capi_maps' (line 437)
    capi_maps_73569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 10), 'capi_maps', False)
    # Obtaining the member 'cb_routsign2map' of a type (line 437)
    cb_routsign2map_73570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 10), capi_maps_73569, 'cb_routsign2map')
    # Calling cb_routsign2map(args, kwargs) (line 437)
    cb_routsign2map_call_result_73574 = invoke(stypy.reporting.localization.Localization(__file__, 437, 10), cb_routsign2map_73570, *[rout_73571, um_73572], **kwargs_73573)
    
    # Assigning a type to the variable 'vrd' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'vrd', cb_routsign2map_call_result_73574)
    
    # Assigning a Call to a Name (line 438):
    
    # Assigning a Call to a Name (line 438):
    
    # Call to dictappend(...): (line 438)
    # Processing the call arguments (line 438)
    
    # Obtaining an instance of the builtin type 'dict' (line 438)
    dict_73576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 20), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 438)
    
    # Getting the type of 'vrd' (line 438)
    vrd_73577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 24), 'vrd', False)
    # Processing the call keyword arguments (line 438)
    kwargs_73578 = {}
    # Getting the type of 'dictappend' (line 438)
    dictappend_73575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 9), 'dictappend', False)
    # Calling dictappend(args, kwargs) (line 438)
    dictappend_call_result_73579 = invoke(stypy.reporting.localization.Localization(__file__, 438, 9), dictappend_73575, *[dict_73576, vrd_73577], **kwargs_73578)
    
    # Assigning a type to the variable 'rd' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'rd', dictappend_call_result_73579)
    
    # Call to append(...): (line 439)
    # Processing the call arguments (line 439)
    
    # Obtaining an instance of the builtin type 'list' (line 439)
    list_73585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 439)
    # Adding element type (line 439)
    
    # Obtaining the type of the subscript
    str_73586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 28), 'str', 'name')
    # Getting the type of 'rout' (line 439)
    rout_73587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 23), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 439)
    getitem___73588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 23), rout_73587, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 439)
    subscript_call_result_73589 = invoke(stypy.reporting.localization.Localization(__file__, 439, 23), getitem___73588, str_73586)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 22), list_73585, subscript_call_result_73589)
    # Adding element type (line 439)
    
    # Obtaining the type of the subscript
    str_73590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 40), 'str', 'name')
    # Getting the type of 'rd' (line 439)
    rd_73591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 37), 'rd', False)
    # Obtaining the member '__getitem__' of a type (line 439)
    getitem___73592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 37), rd_73591, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 439)
    subscript_call_result_73593 = invoke(stypy.reporting.localization.Localization(__file__, 439, 37), getitem___73592, str_73590)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 22), list_73585, subscript_call_result_73593)
    
    # Processing the call keyword arguments (line 439)
    kwargs_73594 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'um' (line 439)
    um_73580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 11), 'um', False)
    # Getting the type of 'cb_map' (line 439)
    cb_map_73581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'cb_map', False)
    # Obtaining the member '__getitem__' of a type (line 439)
    getitem___73582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 4), cb_map_73581, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 439)
    subscript_call_result_73583 = invoke(stypy.reporting.localization.Localization(__file__, 439, 4), getitem___73582, um_73580)
    
    # Obtaining the member 'append' of a type (line 439)
    append_73584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 4), subscript_call_result_73583, 'append')
    # Calling append(args, kwargs) (line 439)
    append_call_result_73595 = invoke(stypy.reporting.localization.Localization(__file__, 439, 4), append_73584, *[list_73585], **kwargs_73594)
    
    
    # Getting the type of 'cb_rout_rules' (line 440)
    cb_rout_rules_73596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 13), 'cb_rout_rules')
    # Testing the type of a for loop iterable (line 440)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 440, 4), cb_rout_rules_73596)
    # Getting the type of the for loop variable (line 440)
    for_loop_var_73597 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 440, 4), cb_rout_rules_73596)
    # Assigning a type to the variable 'r' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'r', for_loop_var_73597)
    # SSA begins for a for statement (line 440)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    str_73598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 12), 'str', '_check')
    # Getting the type of 'r' (line 441)
    r_73599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 24), 'r')
    # Applying the binary operator 'in' (line 441)
    result_contains_73600 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 12), 'in', str_73598, r_73599)
    
    
    # Call to (...): (line 441)
    # Processing the call arguments (line 441)
    # Getting the type of 'rout' (line 441)
    rout_73605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 42), 'rout', False)
    # Processing the call keyword arguments (line 441)
    kwargs_73606 = {}
    
    # Obtaining the type of the subscript
    str_73601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 32), 'str', '_check')
    # Getting the type of 'r' (line 441)
    r_73602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 30), 'r', False)
    # Obtaining the member '__getitem__' of a type (line 441)
    getitem___73603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 30), r_73602, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 441)
    subscript_call_result_73604 = invoke(stypy.reporting.localization.Localization(__file__, 441, 30), getitem___73603, str_73601)
    
    # Calling (args, kwargs) (line 441)
    _call_result_73607 = invoke(stypy.reporting.localization.Localization(__file__, 441, 30), subscript_call_result_73604, *[rout_73605], **kwargs_73606)
    
    # Applying the binary operator 'and' (line 441)
    result_and_keyword_73608 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 12), 'and', result_contains_73600, _call_result_73607)
    
    
    str_73609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 53), 'str', '_check')
    # Getting the type of 'r' (line 441)
    r_73610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 69), 'r')
    # Applying the binary operator 'notin' (line 441)
    result_contains_73611 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 53), 'notin', str_73609, r_73610)
    
    # Applying the binary operator 'or' (line 441)
    result_or_keyword_73612 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 11), 'or', result_and_keyword_73608, result_contains_73611)
    
    # Testing the type of an if condition (line 441)
    if_condition_73613 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 441, 8), result_or_keyword_73612)
    # Assigning a type to the variable 'if_condition_73613' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'if_condition_73613', if_condition_73613)
    # SSA begins for if statement (line 441)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 442):
    
    # Assigning a Call to a Name (line 442):
    
    # Call to applyrules(...): (line 442)
    # Processing the call arguments (line 442)
    # Getting the type of 'r' (line 442)
    r_73615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 28), 'r', False)
    # Getting the type of 'vrd' (line 442)
    vrd_73616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 31), 'vrd', False)
    # Getting the type of 'rout' (line 442)
    rout_73617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 36), 'rout', False)
    # Processing the call keyword arguments (line 442)
    kwargs_73618 = {}
    # Getting the type of 'applyrules' (line 442)
    applyrules_73614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 17), 'applyrules', False)
    # Calling applyrules(args, kwargs) (line 442)
    applyrules_call_result_73619 = invoke(stypy.reporting.localization.Localization(__file__, 442, 17), applyrules_73614, *[r_73615, vrd_73616, rout_73617], **kwargs_73618)
    
    # Assigning a type to the variable 'ar' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'ar', applyrules_call_result_73619)
    
    # Assigning a Call to a Name (line 443):
    
    # Assigning a Call to a Name (line 443):
    
    # Call to dictappend(...): (line 443)
    # Processing the call arguments (line 443)
    # Getting the type of 'rd' (line 443)
    rd_73621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 28), 'rd', False)
    # Getting the type of 'ar' (line 443)
    ar_73622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 32), 'ar', False)
    # Processing the call keyword arguments (line 443)
    kwargs_73623 = {}
    # Getting the type of 'dictappend' (line 443)
    dictappend_73620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 17), 'dictappend', False)
    # Calling dictappend(args, kwargs) (line 443)
    dictappend_call_result_73624 = invoke(stypy.reporting.localization.Localization(__file__, 443, 17), dictappend_73620, *[rd_73621, ar_73622], **kwargs_73623)
    
    # Assigning a type to the variable 'rd' (line 443)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 12), 'rd', dictappend_call_result_73624)
    # SSA join for if statement (line 441)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Name (line 444):
    
    # Assigning a Dict to a Name (line 444):
    
    # Obtaining an instance of the builtin type 'dict' (line 444)
    dict_73625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 14), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 444)
    
    # Assigning a type to the variable 'savevrd' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'savevrd', dict_73625)
    
    
    # Call to enumerate(...): (line 445)
    # Processing the call arguments (line 445)
    # Getting the type of 'args' (line 445)
    args_73627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 26), 'args', False)
    # Processing the call keyword arguments (line 445)
    kwargs_73628 = {}
    # Getting the type of 'enumerate' (line 445)
    enumerate_73626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 16), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 445)
    enumerate_call_result_73629 = invoke(stypy.reporting.localization.Localization(__file__, 445, 16), enumerate_73626, *[args_73627], **kwargs_73628)
    
    # Testing the type of a for loop iterable (line 445)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 445, 4), enumerate_call_result_73629)
    # Getting the type of the for loop variable (line 445)
    for_loop_var_73630 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 445, 4), enumerate_call_result_73629)
    # Assigning a type to the variable 'i' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 4), for_loop_var_73630))
    # Assigning a type to the variable 'a' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 4), for_loop_var_73630))
    # SSA begins for a for statement (line 445)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 446):
    
    # Assigning a Call to a Name (line 446):
    
    # Call to cb_sign2map(...): (line 446)
    # Processing the call arguments (line 446)
    # Getting the type of 'a' (line 446)
    a_73633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 36), 'a', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 446)
    a_73634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 43), 'a', False)
    # Getting the type of 'var' (line 446)
    var_73635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 39), 'var', False)
    # Obtaining the member '__getitem__' of a type (line 446)
    getitem___73636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 39), var_73635, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 446)
    subscript_call_result_73637 = invoke(stypy.reporting.localization.Localization(__file__, 446, 39), getitem___73636, a_73634)
    
    # Processing the call keyword arguments (line 446)
    # Getting the type of 'i' (line 446)
    i_73638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 53), 'i', False)
    keyword_73639 = i_73638
    kwargs_73640 = {'index': keyword_73639}
    # Getting the type of 'capi_maps' (line 446)
    capi_maps_73631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 14), 'capi_maps', False)
    # Obtaining the member 'cb_sign2map' of a type (line 446)
    cb_sign2map_73632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 14), capi_maps_73631, 'cb_sign2map')
    # Calling cb_sign2map(args, kwargs) (line 446)
    cb_sign2map_call_result_73641 = invoke(stypy.reporting.localization.Localization(__file__, 446, 14), cb_sign2map_73632, *[a_73633, subscript_call_result_73637], **kwargs_73640)
    
    # Assigning a type to the variable 'vrd' (line 446)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'vrd', cb_sign2map_call_result_73641)
    
    # Assigning a Name to a Subscript (line 447):
    
    # Assigning a Name to a Subscript (line 447):
    # Getting the type of 'vrd' (line 447)
    vrd_73642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 21), 'vrd')
    # Getting the type of 'savevrd' (line 447)
    savevrd_73643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'savevrd')
    # Getting the type of 'a' (line 447)
    a_73644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 16), 'a')
    # Storing an element on a container (line 447)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 8), savevrd_73643, (a_73644, vrd_73642))
    
    # Getting the type of 'cb_arg_rules' (line 448)
    cb_arg_rules_73645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 17), 'cb_arg_rules')
    # Testing the type of a for loop iterable (line 448)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 448, 8), cb_arg_rules_73645)
    # Getting the type of the for loop variable (line 448)
    for_loop_var_73646 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 448, 8), cb_arg_rules_73645)
    # Assigning a type to the variable 'r' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'r', for_loop_var_73646)
    # SSA begins for a for statement (line 448)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    str_73647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 15), 'str', '_depend')
    # Getting the type of 'r' (line 449)
    r_73648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 28), 'r')
    # Applying the binary operator 'in' (line 449)
    result_contains_73649 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 15), 'in', str_73647, r_73648)
    
    # Testing the type of an if condition (line 449)
    if_condition_73650 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 449, 12), result_contains_73649)
    # Assigning a type to the variable 'if_condition_73650' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'if_condition_73650', if_condition_73650)
    # SSA begins for if statement (line 449)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 449)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    str_73651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 15), 'str', '_optional')
    # Getting the type of 'r' (line 451)
    r_73652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 30), 'r')
    # Applying the binary operator 'in' (line 451)
    result_contains_73653 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 15), 'in', str_73651, r_73652)
    
    
    # Call to isoptional(...): (line 451)
    # Processing the call arguments (line 451)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 451)
    a_73655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 51), 'a', False)
    # Getting the type of 'var' (line 451)
    var_73656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 47), 'var', False)
    # Obtaining the member '__getitem__' of a type (line 451)
    getitem___73657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 47), var_73656, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 451)
    subscript_call_result_73658 = invoke(stypy.reporting.localization.Localization(__file__, 451, 47), getitem___73657, a_73655)
    
    # Processing the call keyword arguments (line 451)
    kwargs_73659 = {}
    # Getting the type of 'isoptional' (line 451)
    isoptional_73654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 36), 'isoptional', False)
    # Calling isoptional(args, kwargs) (line 451)
    isoptional_call_result_73660 = invoke(stypy.reporting.localization.Localization(__file__, 451, 36), isoptional_73654, *[subscript_call_result_73658], **kwargs_73659)
    
    # Applying the binary operator 'and' (line 451)
    result_and_keyword_73661 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 15), 'and', result_contains_73653, isoptional_call_result_73660)
    
    # Testing the type of an if condition (line 451)
    if_condition_73662 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 451, 12), result_and_keyword_73661)
    # Assigning a type to the variable 'if_condition_73662' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'if_condition_73662', if_condition_73662)
    # SSA begins for if statement (line 451)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 451)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    str_73663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 16), 'str', '_check')
    # Getting the type of 'r' (line 453)
    r_73664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 28), 'r')
    # Applying the binary operator 'in' (line 453)
    result_contains_73665 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 16), 'in', str_73663, r_73664)
    
    
    # Call to (...): (line 453)
    # Processing the call arguments (line 453)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 453)
    a_73670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 50), 'a', False)
    # Getting the type of 'var' (line 453)
    var_73671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 46), 'var', False)
    # Obtaining the member '__getitem__' of a type (line 453)
    getitem___73672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 46), var_73671, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 453)
    subscript_call_result_73673 = invoke(stypy.reporting.localization.Localization(__file__, 453, 46), getitem___73672, a_73670)
    
    # Processing the call keyword arguments (line 453)
    kwargs_73674 = {}
    
    # Obtaining the type of the subscript
    str_73666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 36), 'str', '_check')
    # Getting the type of 'r' (line 453)
    r_73667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 34), 'r', False)
    # Obtaining the member '__getitem__' of a type (line 453)
    getitem___73668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 34), r_73667, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 453)
    subscript_call_result_73669 = invoke(stypy.reporting.localization.Localization(__file__, 453, 34), getitem___73668, str_73666)
    
    # Calling (args, kwargs) (line 453)
    _call_result_73675 = invoke(stypy.reporting.localization.Localization(__file__, 453, 34), subscript_call_result_73669, *[subscript_call_result_73673], **kwargs_73674)
    
    # Applying the binary operator 'and' (line 453)
    result_and_keyword_73676 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 16), 'and', result_contains_73665, _call_result_73675)
    
    
    str_73677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 59), 'str', '_check')
    # Getting the type of 'r' (line 453)
    r_73678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 75), 'r')
    # Applying the binary operator 'notin' (line 453)
    result_contains_73679 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 59), 'notin', str_73677, r_73678)
    
    # Applying the binary operator 'or' (line 453)
    result_or_keyword_73680 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 15), 'or', result_and_keyword_73676, result_contains_73679)
    
    # Testing the type of an if condition (line 453)
    if_condition_73681 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 453, 12), result_or_keyword_73680)
    # Assigning a type to the variable 'if_condition_73681' (line 453)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 12), 'if_condition_73681', if_condition_73681)
    # SSA begins for if statement (line 453)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 454):
    
    # Assigning a Call to a Name (line 454):
    
    # Call to applyrules(...): (line 454)
    # Processing the call arguments (line 454)
    # Getting the type of 'r' (line 454)
    r_73683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 32), 'r', False)
    # Getting the type of 'vrd' (line 454)
    vrd_73684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 35), 'vrd', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 454)
    a_73685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 44), 'a', False)
    # Getting the type of 'var' (line 454)
    var_73686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 40), 'var', False)
    # Obtaining the member '__getitem__' of a type (line 454)
    getitem___73687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 40), var_73686, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 454)
    subscript_call_result_73688 = invoke(stypy.reporting.localization.Localization(__file__, 454, 40), getitem___73687, a_73685)
    
    # Processing the call keyword arguments (line 454)
    kwargs_73689 = {}
    # Getting the type of 'applyrules' (line 454)
    applyrules_73682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 21), 'applyrules', False)
    # Calling applyrules(args, kwargs) (line 454)
    applyrules_call_result_73690 = invoke(stypy.reporting.localization.Localization(__file__, 454, 21), applyrules_73682, *[r_73683, vrd_73684, subscript_call_result_73688], **kwargs_73689)
    
    # Assigning a type to the variable 'ar' (line 454)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 16), 'ar', applyrules_call_result_73690)
    
    # Assigning a Call to a Name (line 455):
    
    # Assigning a Call to a Name (line 455):
    
    # Call to dictappend(...): (line 455)
    # Processing the call arguments (line 455)
    # Getting the type of 'rd' (line 455)
    rd_73692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 32), 'rd', False)
    # Getting the type of 'ar' (line 455)
    ar_73693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 36), 'ar', False)
    # Processing the call keyword arguments (line 455)
    kwargs_73694 = {}
    # Getting the type of 'dictappend' (line 455)
    dictappend_73691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 21), 'dictappend', False)
    # Calling dictappend(args, kwargs) (line 455)
    dictappend_call_result_73695 = invoke(stypy.reporting.localization.Localization(__file__, 455, 21), dictappend_73691, *[rd_73692, ar_73693], **kwargs_73694)
    
    # Assigning a type to the variable 'rd' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 16), 'rd', dictappend_call_result_73695)
    
    
    str_73696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 19), 'str', '_break')
    # Getting the type of 'r' (line 456)
    r_73697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 31), 'r')
    # Applying the binary operator 'in' (line 456)
    result_contains_73698 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 19), 'in', str_73696, r_73697)
    
    # Testing the type of an if condition (line 456)
    if_condition_73699 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 456, 16), result_contains_73698)
    # Assigning a type to the variable 'if_condition_73699' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'if_condition_73699', if_condition_73699)
    # SSA begins for if statement (line 456)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 456)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 453)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'args' (line 458)
    args_73700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 13), 'args')
    # Testing the type of a for loop iterable (line 458)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 458, 4), args_73700)
    # Getting the type of the for loop variable (line 458)
    for_loop_var_73701 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 458, 4), args_73700)
    # Assigning a type to the variable 'a' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'a', for_loop_var_73701)
    # SSA begins for a for statement (line 458)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 459):
    
    # Assigning a Subscript to a Name (line 459):
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 459)
    a_73702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 22), 'a')
    # Getting the type of 'savevrd' (line 459)
    savevrd_73703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 14), 'savevrd')
    # Obtaining the member '__getitem__' of a type (line 459)
    getitem___73704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 14), savevrd_73703, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 459)
    subscript_call_result_73705 = invoke(stypy.reporting.localization.Localization(__file__, 459, 14), getitem___73704, a_73702)
    
    # Assigning a type to the variable 'vrd' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'vrd', subscript_call_result_73705)
    
    # Getting the type of 'cb_arg_rules' (line 460)
    cb_arg_rules_73706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 17), 'cb_arg_rules')
    # Testing the type of a for loop iterable (line 460)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 460, 8), cb_arg_rules_73706)
    # Getting the type of the for loop variable (line 460)
    for_loop_var_73707 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 460, 8), cb_arg_rules_73706)
    # Assigning a type to the variable 'r' (line 460)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'r', for_loop_var_73707)
    # SSA begins for a for statement (line 460)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    str_73708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 15), 'str', '_depend')
    # Getting the type of 'r' (line 461)
    r_73709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 28), 'r')
    # Applying the binary operator 'in' (line 461)
    result_contains_73710 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 15), 'in', str_73708, r_73709)
    
    # Testing the type of an if condition (line 461)
    if_condition_73711 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 461, 12), result_contains_73710)
    # Assigning a type to the variable 'if_condition_73711' (line 461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'if_condition_73711', if_condition_73711)
    # SSA begins for if statement (line 461)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 461)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    str_73712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 16), 'str', '_optional')
    # Getting the type of 'r' (line 463)
    r_73713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 35), 'r')
    # Applying the binary operator 'notin' (line 463)
    result_contains_73714 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 16), 'notin', str_73712, r_73713)
    
    
    # Evaluating a boolean operation
    
    str_73715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 42), 'str', '_optional')
    # Getting the type of 'r' (line 463)
    r_73716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 57), 'r')
    # Applying the binary operator 'in' (line 463)
    result_contains_73717 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 42), 'in', str_73715, r_73716)
    
    
    # Call to isrequired(...): (line 463)
    # Processing the call arguments (line 463)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 463)
    a_73719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 78), 'a', False)
    # Getting the type of 'var' (line 463)
    var_73720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 74), 'var', False)
    # Obtaining the member '__getitem__' of a type (line 463)
    getitem___73721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 74), var_73720, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 463)
    subscript_call_result_73722 = invoke(stypy.reporting.localization.Localization(__file__, 463, 74), getitem___73721, a_73719)
    
    # Processing the call keyword arguments (line 463)
    kwargs_73723 = {}
    # Getting the type of 'isrequired' (line 463)
    isrequired_73718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 63), 'isrequired', False)
    # Calling isrequired(args, kwargs) (line 463)
    isrequired_call_result_73724 = invoke(stypy.reporting.localization.Localization(__file__, 463, 63), isrequired_73718, *[subscript_call_result_73722], **kwargs_73723)
    
    # Applying the binary operator 'and' (line 463)
    result_and_keyword_73725 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 42), 'and', result_contains_73717, isrequired_call_result_73724)
    
    # Applying the binary operator 'or' (line 463)
    result_or_keyword_73726 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 15), 'or', result_contains_73714, result_and_keyword_73725)
    
    # Testing the type of an if condition (line 463)
    if_condition_73727 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 463, 12), result_or_keyword_73726)
    # Assigning a type to the variable 'if_condition_73727' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 12), 'if_condition_73727', if_condition_73727)
    # SSA begins for if statement (line 463)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 463)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    str_73728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 16), 'str', '_check')
    # Getting the type of 'r' (line 465)
    r_73729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 28), 'r')
    # Applying the binary operator 'in' (line 465)
    result_contains_73730 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 16), 'in', str_73728, r_73729)
    
    
    # Call to (...): (line 465)
    # Processing the call arguments (line 465)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 465)
    a_73735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 50), 'a', False)
    # Getting the type of 'var' (line 465)
    var_73736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 46), 'var', False)
    # Obtaining the member '__getitem__' of a type (line 465)
    getitem___73737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 46), var_73736, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 465)
    subscript_call_result_73738 = invoke(stypy.reporting.localization.Localization(__file__, 465, 46), getitem___73737, a_73735)
    
    # Processing the call keyword arguments (line 465)
    kwargs_73739 = {}
    
    # Obtaining the type of the subscript
    str_73731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 36), 'str', '_check')
    # Getting the type of 'r' (line 465)
    r_73732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 34), 'r', False)
    # Obtaining the member '__getitem__' of a type (line 465)
    getitem___73733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 34), r_73732, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 465)
    subscript_call_result_73734 = invoke(stypy.reporting.localization.Localization(__file__, 465, 34), getitem___73733, str_73731)
    
    # Calling (args, kwargs) (line 465)
    _call_result_73740 = invoke(stypy.reporting.localization.Localization(__file__, 465, 34), subscript_call_result_73734, *[subscript_call_result_73738], **kwargs_73739)
    
    # Applying the binary operator 'and' (line 465)
    result_and_keyword_73741 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 16), 'and', result_contains_73730, _call_result_73740)
    
    
    str_73742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 59), 'str', '_check')
    # Getting the type of 'r' (line 465)
    r_73743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 75), 'r')
    # Applying the binary operator 'notin' (line 465)
    result_contains_73744 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 59), 'notin', str_73742, r_73743)
    
    # Applying the binary operator 'or' (line 465)
    result_or_keyword_73745 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 15), 'or', result_and_keyword_73741, result_contains_73744)
    
    # Testing the type of an if condition (line 465)
    if_condition_73746 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 465, 12), result_or_keyword_73745)
    # Assigning a type to the variable 'if_condition_73746' (line 465)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 12), 'if_condition_73746', if_condition_73746)
    # SSA begins for if statement (line 465)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 466):
    
    # Assigning a Call to a Name (line 466):
    
    # Call to applyrules(...): (line 466)
    # Processing the call arguments (line 466)
    # Getting the type of 'r' (line 466)
    r_73748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 32), 'r', False)
    # Getting the type of 'vrd' (line 466)
    vrd_73749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 35), 'vrd', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 466)
    a_73750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 44), 'a', False)
    # Getting the type of 'var' (line 466)
    var_73751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 40), 'var', False)
    # Obtaining the member '__getitem__' of a type (line 466)
    getitem___73752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 40), var_73751, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 466)
    subscript_call_result_73753 = invoke(stypy.reporting.localization.Localization(__file__, 466, 40), getitem___73752, a_73750)
    
    # Processing the call keyword arguments (line 466)
    kwargs_73754 = {}
    # Getting the type of 'applyrules' (line 466)
    applyrules_73747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 21), 'applyrules', False)
    # Calling applyrules(args, kwargs) (line 466)
    applyrules_call_result_73755 = invoke(stypy.reporting.localization.Localization(__file__, 466, 21), applyrules_73747, *[r_73748, vrd_73749, subscript_call_result_73753], **kwargs_73754)
    
    # Assigning a type to the variable 'ar' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 16), 'ar', applyrules_call_result_73755)
    
    # Assigning a Call to a Name (line 467):
    
    # Assigning a Call to a Name (line 467):
    
    # Call to dictappend(...): (line 467)
    # Processing the call arguments (line 467)
    # Getting the type of 'rd' (line 467)
    rd_73757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 32), 'rd', False)
    # Getting the type of 'ar' (line 467)
    ar_73758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 36), 'ar', False)
    # Processing the call keyword arguments (line 467)
    kwargs_73759 = {}
    # Getting the type of 'dictappend' (line 467)
    dictappend_73756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 21), 'dictappend', False)
    # Calling dictappend(args, kwargs) (line 467)
    dictappend_call_result_73760 = invoke(stypy.reporting.localization.Localization(__file__, 467, 21), dictappend_73756, *[rd_73757, ar_73758], **kwargs_73759)
    
    # Assigning a type to the variable 'rd' (line 467)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 16), 'rd', dictappend_call_result_73760)
    
    
    str_73761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 19), 'str', '_break')
    # Getting the type of 'r' (line 468)
    r_73762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 31), 'r')
    # Applying the binary operator 'in' (line 468)
    result_contains_73763 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 19), 'in', str_73761, r_73762)
    
    # Testing the type of an if condition (line 468)
    if_condition_73764 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 468, 16), result_contains_73763)
    # Assigning a type to the variable 'if_condition_73764' (line 468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 16), 'if_condition_73764', if_condition_73764)
    # SSA begins for if statement (line 468)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 468)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 465)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'depargs' (line 470)
    depargs_73765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 13), 'depargs')
    # Testing the type of a for loop iterable (line 470)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 470, 4), depargs_73765)
    # Getting the type of the for loop variable (line 470)
    for_loop_var_73766 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 470, 4), depargs_73765)
    # Assigning a type to the variable 'a' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'a', for_loop_var_73766)
    # SSA begins for a for statement (line 470)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 471):
    
    # Assigning a Subscript to a Name (line 471):
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 471)
    a_73767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 22), 'a')
    # Getting the type of 'savevrd' (line 471)
    savevrd_73768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 14), 'savevrd')
    # Obtaining the member '__getitem__' of a type (line 471)
    getitem___73769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 14), savevrd_73768, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 471)
    subscript_call_result_73770 = invoke(stypy.reporting.localization.Localization(__file__, 471, 14), getitem___73769, a_73767)
    
    # Assigning a type to the variable 'vrd' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'vrd', subscript_call_result_73770)
    
    # Getting the type of 'cb_arg_rules' (line 472)
    cb_arg_rules_73771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 17), 'cb_arg_rules')
    # Testing the type of a for loop iterable (line 472)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 472, 8), cb_arg_rules_73771)
    # Getting the type of the for loop variable (line 472)
    for_loop_var_73772 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 472, 8), cb_arg_rules_73771)
    # Assigning a type to the variable 'r' (line 472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'r', for_loop_var_73772)
    # SSA begins for a for statement (line 472)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    str_73773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 15), 'str', '_depend')
    # Getting the type of 'r' (line 473)
    r_73774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 32), 'r')
    # Applying the binary operator 'notin' (line 473)
    result_contains_73775 = python_operator(stypy.reporting.localization.Localization(__file__, 473, 15), 'notin', str_73773, r_73774)
    
    # Testing the type of an if condition (line 473)
    if_condition_73776 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 473, 12), result_contains_73775)
    # Assigning a type to the variable 'if_condition_73776' (line 473)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'if_condition_73776', if_condition_73776)
    # SSA begins for if statement (line 473)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 473)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_73777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 15), 'str', '_optional')
    # Getting the type of 'r' (line 475)
    r_73778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 30), 'r')
    # Applying the binary operator 'in' (line 475)
    result_contains_73779 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 15), 'in', str_73777, r_73778)
    
    # Testing the type of an if condition (line 475)
    if_condition_73780 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 475, 12), result_contains_73779)
    # Assigning a type to the variable 'if_condition_73780' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'if_condition_73780', if_condition_73780)
    # SSA begins for if statement (line 475)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 475)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    str_73781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 16), 'str', '_check')
    # Getting the type of 'r' (line 477)
    r_73782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 28), 'r')
    # Applying the binary operator 'in' (line 477)
    result_contains_73783 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 16), 'in', str_73781, r_73782)
    
    
    # Call to (...): (line 477)
    # Processing the call arguments (line 477)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 477)
    a_73788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 50), 'a', False)
    # Getting the type of 'var' (line 477)
    var_73789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 46), 'var', False)
    # Obtaining the member '__getitem__' of a type (line 477)
    getitem___73790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 46), var_73789, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 477)
    subscript_call_result_73791 = invoke(stypy.reporting.localization.Localization(__file__, 477, 46), getitem___73790, a_73788)
    
    # Processing the call keyword arguments (line 477)
    kwargs_73792 = {}
    
    # Obtaining the type of the subscript
    str_73784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 36), 'str', '_check')
    # Getting the type of 'r' (line 477)
    r_73785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 34), 'r', False)
    # Obtaining the member '__getitem__' of a type (line 477)
    getitem___73786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 34), r_73785, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 477)
    subscript_call_result_73787 = invoke(stypy.reporting.localization.Localization(__file__, 477, 34), getitem___73786, str_73784)
    
    # Calling (args, kwargs) (line 477)
    _call_result_73793 = invoke(stypy.reporting.localization.Localization(__file__, 477, 34), subscript_call_result_73787, *[subscript_call_result_73791], **kwargs_73792)
    
    # Applying the binary operator 'and' (line 477)
    result_and_keyword_73794 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 16), 'and', result_contains_73783, _call_result_73793)
    
    
    str_73795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 59), 'str', '_check')
    # Getting the type of 'r' (line 477)
    r_73796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 75), 'r')
    # Applying the binary operator 'notin' (line 477)
    result_contains_73797 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 59), 'notin', str_73795, r_73796)
    
    # Applying the binary operator 'or' (line 477)
    result_or_keyword_73798 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 15), 'or', result_and_keyword_73794, result_contains_73797)
    
    # Testing the type of an if condition (line 477)
    if_condition_73799 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 477, 12), result_or_keyword_73798)
    # Assigning a type to the variable 'if_condition_73799' (line 477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 12), 'if_condition_73799', if_condition_73799)
    # SSA begins for if statement (line 477)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 478):
    
    # Assigning a Call to a Name (line 478):
    
    # Call to applyrules(...): (line 478)
    # Processing the call arguments (line 478)
    # Getting the type of 'r' (line 478)
    r_73801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 32), 'r', False)
    # Getting the type of 'vrd' (line 478)
    vrd_73802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 35), 'vrd', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 478)
    a_73803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 44), 'a', False)
    # Getting the type of 'var' (line 478)
    var_73804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 40), 'var', False)
    # Obtaining the member '__getitem__' of a type (line 478)
    getitem___73805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 40), var_73804, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 478)
    subscript_call_result_73806 = invoke(stypy.reporting.localization.Localization(__file__, 478, 40), getitem___73805, a_73803)
    
    # Processing the call keyword arguments (line 478)
    kwargs_73807 = {}
    # Getting the type of 'applyrules' (line 478)
    applyrules_73800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 21), 'applyrules', False)
    # Calling applyrules(args, kwargs) (line 478)
    applyrules_call_result_73808 = invoke(stypy.reporting.localization.Localization(__file__, 478, 21), applyrules_73800, *[r_73801, vrd_73802, subscript_call_result_73806], **kwargs_73807)
    
    # Assigning a type to the variable 'ar' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 16), 'ar', applyrules_call_result_73808)
    
    # Assigning a Call to a Name (line 479):
    
    # Assigning a Call to a Name (line 479):
    
    # Call to dictappend(...): (line 479)
    # Processing the call arguments (line 479)
    # Getting the type of 'rd' (line 479)
    rd_73810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 32), 'rd', False)
    # Getting the type of 'ar' (line 479)
    ar_73811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 36), 'ar', False)
    # Processing the call keyword arguments (line 479)
    kwargs_73812 = {}
    # Getting the type of 'dictappend' (line 479)
    dictappend_73809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 21), 'dictappend', False)
    # Calling dictappend(args, kwargs) (line 479)
    dictappend_call_result_73813 = invoke(stypy.reporting.localization.Localization(__file__, 479, 21), dictappend_73809, *[rd_73810, ar_73811], **kwargs_73812)
    
    # Assigning a type to the variable 'rd' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 16), 'rd', dictappend_call_result_73813)
    
    
    str_73814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 19), 'str', '_break')
    # Getting the type of 'r' (line 480)
    r_73815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 31), 'r')
    # Applying the binary operator 'in' (line 480)
    result_contains_73816 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 19), 'in', str_73814, r_73815)
    
    # Testing the type of an if condition (line 480)
    if_condition_73817 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 480, 16), result_contains_73816)
    # Assigning a type to the variable 'if_condition_73817' (line 480)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 16), 'if_condition_73817', if_condition_73817)
    # SSA begins for if statement (line 480)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 480)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 477)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    str_73818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 7), 'str', 'args')
    # Getting the type of 'rd' (line 482)
    rd_73819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 17), 'rd')
    # Applying the binary operator 'in' (line 482)
    result_contains_73820 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 7), 'in', str_73818, rd_73819)
    
    
    str_73821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 24), 'str', 'optargs')
    # Getting the type of 'rd' (line 482)
    rd_73822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 37), 'rd')
    # Applying the binary operator 'in' (line 482)
    result_contains_73823 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 24), 'in', str_73821, rd_73822)
    
    # Applying the binary operator 'and' (line 482)
    result_and_keyword_73824 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 7), 'and', result_contains_73820, result_contains_73823)
    
    # Testing the type of an if condition (line 482)
    if_condition_73825 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 482, 4), result_and_keyword_73824)
    # Assigning a type to the variable 'if_condition_73825' (line 482)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'if_condition_73825', if_condition_73825)
    # SSA begins for if statement (line 482)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 483)
    # Getting the type of 'list' (line 483)
    list_73826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 37), 'list')
    
    # Obtaining the type of the subscript
    str_73827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 25), 'str', 'optargs')
    # Getting the type of 'rd' (line 483)
    rd_73828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 22), 'rd')
    # Obtaining the member '__getitem__' of a type (line 483)
    getitem___73829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 22), rd_73828, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 483)
    subscript_call_result_73830 = invoke(stypy.reporting.localization.Localization(__file__, 483, 22), getitem___73829, str_73827)
    
    
    (may_be_73831, more_types_in_union_73832) = may_be_subtype(list_73826, subscript_call_result_73830)

    if may_be_73831:

        if more_types_in_union_73832:
            # Runtime conditional SSA (line 483)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Subscript (line 484):
        
        # Assigning a BinOp to a Subscript (line 484):
        
        # Obtaining the type of the subscript
        str_73833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 31), 'str', 'optargs')
        # Getting the type of 'rd' (line 484)
        rd_73834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 28), 'rd')
        # Obtaining the member '__getitem__' of a type (line 484)
        getitem___73835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 28), rd_73834, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 484)
        subscript_call_result_73836 = invoke(stypy.reporting.localization.Localization(__file__, 484, 28), getitem___73835, str_73833)
        
        
        # Obtaining an instance of the builtin type 'list' (line 484)
        list_73837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 484)
        # Adding element type (line 484)
        str_73838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, (-1)), 'str', '\n#ifndef F2PY_CB_RETURNCOMPLEX\n,\n#endif\n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 44), list_73837, str_73838)
        
        # Applying the binary operator '+' (line 484)
        result_add_73839 = python_operator(stypy.reporting.localization.Localization(__file__, 484, 28), '+', subscript_call_result_73836, list_73837)
        
        # Getting the type of 'rd' (line 484)
        rd_73840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 12), 'rd')
        str_73841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 15), 'str', 'optargs')
        # Storing an element on a container (line 484)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 12), rd_73840, (str_73841, result_add_73839))
        
        # Assigning a BinOp to a Subscript (line 489):
        
        # Assigning a BinOp to a Subscript (line 489):
        
        # Obtaining the type of the subscript
        str_73842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 34), 'str', 'optargs_nm')
        # Getting the type of 'rd' (line 489)
        rd_73843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 31), 'rd')
        # Obtaining the member '__getitem__' of a type (line 489)
        getitem___73844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 31), rd_73843, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 489)
        subscript_call_result_73845 = invoke(stypy.reporting.localization.Localization(__file__, 489, 31), getitem___73844, str_73842)
        
        
        # Obtaining an instance of the builtin type 'list' (line 489)
        list_73846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 489)
        # Adding element type (line 489)
        str_73847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, (-1)), 'str', '\n#ifndef F2PY_CB_RETURNCOMPLEX\n,\n#endif\n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 50), list_73846, str_73847)
        
        # Applying the binary operator '+' (line 489)
        result_add_73848 = python_operator(stypy.reporting.localization.Localization(__file__, 489, 31), '+', subscript_call_result_73845, list_73846)
        
        # Getting the type of 'rd' (line 489)
        rd_73849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 12), 'rd')
        str_73850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 15), 'str', 'optargs_nm')
        # Storing an element on a container (line 489)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 12), rd_73849, (str_73850, result_add_73848))
        
        # Assigning a BinOp to a Subscript (line 494):
        
        # Assigning a BinOp to a Subscript (line 494):
        
        # Obtaining the type of the subscript
        str_73851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 34), 'str', 'optargs_td')
        # Getting the type of 'rd' (line 494)
        rd_73852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 31), 'rd')
        # Obtaining the member '__getitem__' of a type (line 494)
        getitem___73853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 31), rd_73852, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 494)
        subscript_call_result_73854 = invoke(stypy.reporting.localization.Localization(__file__, 494, 31), getitem___73853, str_73851)
        
        
        # Obtaining an instance of the builtin type 'list' (line 494)
        list_73855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 494)
        # Adding element type (line 494)
        str_73856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, (-1)), 'str', '\n#ifndef F2PY_CB_RETURNCOMPLEX\n,\n#endif\n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 494, 50), list_73855, str_73856)
        
        # Applying the binary operator '+' (line 494)
        result_add_73857 = python_operator(stypy.reporting.localization.Localization(__file__, 494, 31), '+', subscript_call_result_73854, list_73855)
        
        # Getting the type of 'rd' (line 494)
        rd_73858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'rd')
        str_73859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 15), 'str', 'optargs_td')
        # Storing an element on a container (line 494)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 494, 12), rd_73858, (str_73859, result_add_73857))

        if more_types_in_union_73832:
            # SSA join for if statement (line 483)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 482)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 499)
    # Getting the type of 'list' (line 499)
    list_73860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 35), 'list')
    
    # Obtaining the type of the subscript
    str_73861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 21), 'str', 'docreturn')
    # Getting the type of 'rd' (line 499)
    rd_73862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 18), 'rd')
    # Obtaining the member '__getitem__' of a type (line 499)
    getitem___73863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 18), rd_73862, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 499)
    subscript_call_result_73864 = invoke(stypy.reporting.localization.Localization(__file__, 499, 18), getitem___73863, str_73861)
    
    
    (may_be_73865, more_types_in_union_73866) = may_be_subtype(list_73860, subscript_call_result_73864)

    if may_be_73865:

        if more_types_in_union_73866:
            # Runtime conditional SSA (line 499)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Subscript (line 500):
        
        # Assigning a Call to a Subscript (line 500):
        
        # Call to stripcomma(...): (line 500)
        # Processing the call arguments (line 500)
        
        # Call to replace(...): (line 501)
        # Processing the call arguments (line 501)
        str_73869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 20), 'str', '#docreturn#')
        
        # Obtaining an instance of the builtin type 'dict' (line 501)
        dict_73870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 35), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 501)
        # Adding element type (key, value) (line 501)
        str_73871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 36), 'str', 'docreturn')
        
        # Obtaining the type of the subscript
        str_73872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 52), 'str', 'docreturn')
        # Getting the type of 'rd' (line 501)
        rd_73873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 49), 'rd', False)
        # Obtaining the member '__getitem__' of a type (line 501)
        getitem___73874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 49), rd_73873, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 501)
        subscript_call_result_73875 = invoke(stypy.reporting.localization.Localization(__file__, 501, 49), getitem___73874, str_73872)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 35), dict_73870, (str_73871, subscript_call_result_73875))
        
        # Processing the call keyword arguments (line 501)
        kwargs_73876 = {}
        # Getting the type of 'replace' (line 501)
        replace_73868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 12), 'replace', False)
        # Calling replace(args, kwargs) (line 501)
        replace_call_result_73877 = invoke(stypy.reporting.localization.Localization(__file__, 501, 12), replace_73868, *[str_73869, dict_73870], **kwargs_73876)
        
        # Processing the call keyword arguments (line 500)
        kwargs_73878 = {}
        # Getting the type of 'stripcomma' (line 500)
        stripcomma_73867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 26), 'stripcomma', False)
        # Calling stripcomma(args, kwargs) (line 500)
        stripcomma_call_result_73879 = invoke(stypy.reporting.localization.Localization(__file__, 500, 26), stripcomma_73867, *[replace_call_result_73877], **kwargs_73878)
        
        # Getting the type of 'rd' (line 500)
        rd_73880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'rd')
        str_73881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 11), 'str', 'docreturn')
        # Storing an element on a container (line 500)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 8), rd_73880, (str_73881, stripcomma_call_result_73879))

        if more_types_in_union_73866:
            # SSA join for if statement (line 499)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 502):
    
    # Assigning a Call to a Name (line 502):
    
    # Call to stripcomma(...): (line 502)
    # Processing the call arguments (line 502)
    
    # Call to replace(...): (line 502)
    # Processing the call arguments (line 502)
    str_73884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 33), 'str', '#docsignopt#')
    
    # Obtaining an instance of the builtin type 'dict' (line 503)
    dict_73885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 33), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 503)
    # Adding element type (key, value) (line 503)
    str_73886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 34), 'str', 'docsignopt')
    
    # Obtaining the type of the subscript
    str_73887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 51), 'str', 'docsignopt')
    # Getting the type of 'rd' (line 503)
    rd_73888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 48), 'rd', False)
    # Obtaining the member '__getitem__' of a type (line 503)
    getitem___73889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 48), rd_73888, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 503)
    subscript_call_result_73890 = invoke(stypy.reporting.localization.Localization(__file__, 503, 48), getitem___73889, str_73887)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 33), dict_73885, (str_73886, subscript_call_result_73890))
    
    # Processing the call keyword arguments (line 502)
    kwargs_73891 = {}
    # Getting the type of 'replace' (line 502)
    replace_73883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 25), 'replace', False)
    # Calling replace(args, kwargs) (line 502)
    replace_call_result_73892 = invoke(stypy.reporting.localization.Localization(__file__, 502, 25), replace_73883, *[str_73884, dict_73885], **kwargs_73891)
    
    # Processing the call keyword arguments (line 502)
    kwargs_73893 = {}
    # Getting the type of 'stripcomma' (line 502)
    stripcomma_73882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 14), 'stripcomma', False)
    # Calling stripcomma(args, kwargs) (line 502)
    stripcomma_call_result_73894 = invoke(stypy.reporting.localization.Localization(__file__, 502, 14), stripcomma_73882, *[replace_call_result_73892], **kwargs_73893)
    
    # Assigning a type to the variable 'optargs' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'optargs', stripcomma_call_result_73894)
    
    
    # Getting the type of 'optargs' (line 505)
    optargs_73895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 7), 'optargs')
    str_73896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 18), 'str', '')
    # Applying the binary operator '==' (line 505)
    result_eq_73897 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 7), '==', optargs_73895, str_73896)
    
    # Testing the type of an if condition (line 505)
    if_condition_73898 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 505, 4), result_eq_73897)
    # Assigning a type to the variable 'if_condition_73898' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'if_condition_73898', if_condition_73898)
    # SSA begins for if statement (line 505)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 506):
    
    # Assigning a Call to a Subscript (line 506):
    
    # Call to stripcomma(...): (line 506)
    # Processing the call arguments (line 506)
    
    # Call to replace(...): (line 507)
    # Processing the call arguments (line 507)
    str_73901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 20), 'str', '#docsign#')
    
    # Obtaining an instance of the builtin type 'dict' (line 507)
    dict_73902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 33), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 507)
    # Adding element type (key, value) (line 507)
    str_73903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 34), 'str', 'docsign')
    
    # Obtaining the type of the subscript
    str_73904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 48), 'str', 'docsign')
    # Getting the type of 'rd' (line 507)
    rd_73905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 45), 'rd', False)
    # Obtaining the member '__getitem__' of a type (line 507)
    getitem___73906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 45), rd_73905, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 507)
    subscript_call_result_73907 = invoke(stypy.reporting.localization.Localization(__file__, 507, 45), getitem___73906, str_73904)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 33), dict_73902, (str_73903, subscript_call_result_73907))
    
    # Processing the call keyword arguments (line 507)
    kwargs_73908 = {}
    # Getting the type of 'replace' (line 507)
    replace_73900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 12), 'replace', False)
    # Calling replace(args, kwargs) (line 507)
    replace_call_result_73909 = invoke(stypy.reporting.localization.Localization(__file__, 507, 12), replace_73900, *[str_73901, dict_73902], **kwargs_73908)
    
    # Processing the call keyword arguments (line 506)
    kwargs_73910 = {}
    # Getting the type of 'stripcomma' (line 506)
    stripcomma_73899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 29), 'stripcomma', False)
    # Calling stripcomma(args, kwargs) (line 506)
    stripcomma_call_result_73911 = invoke(stypy.reporting.localization.Localization(__file__, 506, 29), stripcomma_73899, *[replace_call_result_73909], **kwargs_73910)
    
    # Getting the type of 'rd' (line 506)
    rd_73912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'rd')
    str_73913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 11), 'str', 'docsignature')
    # Storing an element on a container (line 506)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 8), rd_73912, (str_73913, stripcomma_call_result_73911))
    # SSA branch for the else part of an if statement (line 505)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Subscript (line 509):
    
    # Assigning a Call to a Subscript (line 509):
    
    # Call to replace(...): (line 509)
    # Processing the call arguments (line 509)
    str_73915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 37), 'str', '#docsign#[#docsignopt#]')
    
    # Obtaining an instance of the builtin type 'dict' (line 510)
    dict_73916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 37), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 510)
    # Adding element type (key, value) (line 510)
    str_73917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 38), 'str', 'docsign')
    
    # Obtaining the type of the subscript
    str_73918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 52), 'str', 'docsign')
    # Getting the type of 'rd' (line 510)
    rd_73919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 49), 'rd', False)
    # Obtaining the member '__getitem__' of a type (line 510)
    getitem___73920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 49), rd_73919, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 510)
    subscript_call_result_73921 = invoke(stypy.reporting.localization.Localization(__file__, 510, 49), getitem___73920, str_73918)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 37), dict_73916, (str_73917, subscript_call_result_73921))
    # Adding element type (key, value) (line 510)
    str_73922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 38), 'str', 'docsignopt')
    # Getting the type of 'optargs' (line 511)
    optargs_73923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 52), 'optargs', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 37), dict_73916, (str_73922, optargs_73923))
    
    # Processing the call keyword arguments (line 509)
    kwargs_73924 = {}
    # Getting the type of 'replace' (line 509)
    replace_73914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 29), 'replace', False)
    # Calling replace(args, kwargs) (line 509)
    replace_call_result_73925 = invoke(stypy.reporting.localization.Localization(__file__, 509, 29), replace_73914, *[str_73915, dict_73916], **kwargs_73924)
    
    # Getting the type of 'rd' (line 509)
    rd_73926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'rd')
    str_73927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 11), 'str', 'docsignature')
    # Storing an element on a container (line 509)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 8), rd_73926, (str_73927, replace_call_result_73925))
    # SSA join for if statement (line 505)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Subscript (line 513):
    
    # Assigning a Call to a Subscript (line 513):
    
    # Call to replace(...): (line 513)
    # Processing the call arguments (line 513)
    str_73933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 57), 'str', '_')
    str_73934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 62), 'str', '\\_')
    # Processing the call keyword arguments (line 513)
    kwargs_73935 = {}
    
    # Obtaining the type of the subscript
    str_73928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 33), 'str', 'docsignature')
    # Getting the type of 'rd' (line 513)
    rd_73929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 30), 'rd', False)
    # Obtaining the member '__getitem__' of a type (line 513)
    getitem___73930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 30), rd_73929, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 513)
    subscript_call_result_73931 = invoke(stypy.reporting.localization.Localization(__file__, 513, 30), getitem___73930, str_73928)
    
    # Obtaining the member 'replace' of a type (line 513)
    replace_73932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 30), subscript_call_result_73931, 'replace')
    # Calling replace(args, kwargs) (line 513)
    replace_call_result_73936 = invoke(stypy.reporting.localization.Localization(__file__, 513, 30), replace_73932, *[str_73933, str_73934], **kwargs_73935)
    
    # Getting the type of 'rd' (line 513)
    rd_73937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 4), 'rd')
    str_73938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 7), 'str', 'latexdocsignature')
    # Storing an element on a container (line 513)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 4), rd_73937, (str_73938, replace_call_result_73936))
    
    # Assigning a Call to a Subscript (line 514):
    
    # Assigning a Call to a Subscript (line 514):
    
    # Call to replace(...): (line 514)
    # Processing the call arguments (line 514)
    str_73944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 62), 'str', ',')
    str_73945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 67), 'str', ', ')
    # Processing the call keyword arguments (line 514)
    kwargs_73946 = {}
    
    # Obtaining the type of the subscript
    str_73939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 33), 'str', 'latexdocsignature')
    # Getting the type of 'rd' (line 514)
    rd_73940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 30), 'rd', False)
    # Obtaining the member '__getitem__' of a type (line 514)
    getitem___73941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 30), rd_73940, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 514)
    subscript_call_result_73942 = invoke(stypy.reporting.localization.Localization(__file__, 514, 30), getitem___73941, str_73939)
    
    # Obtaining the member 'replace' of a type (line 514)
    replace_73943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 30), subscript_call_result_73942, 'replace')
    # Calling replace(args, kwargs) (line 514)
    replace_call_result_73947 = invoke(stypy.reporting.localization.Localization(__file__, 514, 30), replace_73943, *[str_73944, str_73945], **kwargs_73946)
    
    # Getting the type of 'rd' (line 514)
    rd_73948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 4), 'rd')
    str_73949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 7), 'str', 'latexdocsignature')
    # Storing an element on a container (line 514)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 4), rd_73948, (str_73949, replace_call_result_73947))
    
    # Assigning a List to a Subscript (line 515):
    
    # Assigning a List to a Subscript (line 515):
    
    # Obtaining an instance of the builtin type 'list' (line 515)
    list_73950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 515)
    
    # Getting the type of 'rd' (line 515)
    rd_73951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'rd')
    str_73952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 7), 'str', 'docstrsigns')
    # Storing an element on a container (line 515)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 4), rd_73951, (str_73952, list_73950))
    
    # Assigning a List to a Subscript (line 516):
    
    # Assigning a List to a Subscript (line 516):
    
    # Obtaining an instance of the builtin type 'list' (line 516)
    list_73953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 516)
    
    # Getting the type of 'rd' (line 516)
    rd_73954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'rd')
    str_73955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 7), 'str', 'latexdocstrsigns')
    # Storing an element on a container (line 516)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 4), rd_73954, (str_73955, list_73953))
    
    
    # Obtaining an instance of the builtin type 'list' (line 517)
    list_73956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 517)
    # Adding element type (line 517)
    str_73957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 14), 'str', 'docstrreq')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 13), list_73956, str_73957)
    # Adding element type (line 517)
    str_73958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 27), 'str', 'docstropt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 13), list_73956, str_73958)
    # Adding element type (line 517)
    str_73959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 40), 'str', 'docstrout')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 13), list_73956, str_73959)
    # Adding element type (line 517)
    str_73960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 53), 'str', 'docstrcbs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 13), list_73956, str_73960)
    
    # Testing the type of a for loop iterable (line 517)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 517, 4), list_73956)
    # Getting the type of the for loop variable (line 517)
    for_loop_var_73961 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 517, 4), list_73956)
    # Assigning a type to the variable 'k' (line 517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 4), 'k', for_loop_var_73961)
    # SSA begins for a for statement (line 517)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'k' (line 518)
    k_73962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 11), 'k')
    # Getting the type of 'rd' (line 518)
    rd_73963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 16), 'rd')
    # Applying the binary operator 'in' (line 518)
    result_contains_73964 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 11), 'in', k_73962, rd_73963)
    
    
    # Call to isinstance(...): (line 518)
    # Processing the call arguments (line 518)
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 518)
    k_73966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 37), 'k', False)
    # Getting the type of 'rd' (line 518)
    rd_73967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 34), 'rd', False)
    # Obtaining the member '__getitem__' of a type (line 518)
    getitem___73968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 34), rd_73967, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 518)
    subscript_call_result_73969 = invoke(stypy.reporting.localization.Localization(__file__, 518, 34), getitem___73968, k_73966)
    
    # Getting the type of 'list' (line 518)
    list_73970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 41), 'list', False)
    # Processing the call keyword arguments (line 518)
    kwargs_73971 = {}
    # Getting the type of 'isinstance' (line 518)
    isinstance_73965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 23), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 518)
    isinstance_call_result_73972 = invoke(stypy.reporting.localization.Localization(__file__, 518, 23), isinstance_73965, *[subscript_call_result_73969, list_73970], **kwargs_73971)
    
    # Applying the binary operator 'and' (line 518)
    result_and_keyword_73973 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 11), 'and', result_contains_73964, isinstance_call_result_73972)
    
    # Testing the type of an if condition (line 518)
    if_condition_73974 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 518, 8), result_and_keyword_73973)
    # Assigning a type to the variable 'if_condition_73974' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'if_condition_73974', if_condition_73974)
    # SSA begins for if statement (line 518)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 519):
    
    # Assigning a BinOp to a Subscript (line 519):
    
    # Obtaining the type of the subscript
    str_73975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 35), 'str', 'docstrsigns')
    # Getting the type of 'rd' (line 519)
    rd_73976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 32), 'rd')
    # Obtaining the member '__getitem__' of a type (line 519)
    getitem___73977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 32), rd_73976, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 519)
    subscript_call_result_73978 = invoke(stypy.reporting.localization.Localization(__file__, 519, 32), getitem___73977, str_73975)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 519)
    k_73979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 55), 'k')
    # Getting the type of 'rd' (line 519)
    rd_73980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 52), 'rd')
    # Obtaining the member '__getitem__' of a type (line 519)
    getitem___73981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 52), rd_73980, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 519)
    subscript_call_result_73982 = invoke(stypy.reporting.localization.Localization(__file__, 519, 52), getitem___73981, k_73979)
    
    # Applying the binary operator '+' (line 519)
    result_add_73983 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 32), '+', subscript_call_result_73978, subscript_call_result_73982)
    
    # Getting the type of 'rd' (line 519)
    rd_73984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 12), 'rd')
    str_73985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 15), 'str', 'docstrsigns')
    # Storing an element on a container (line 519)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 12), rd_73984, (str_73985, result_add_73983))
    # SSA join for if statement (line 518)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 520):
    
    # Assigning a BinOp to a Name (line 520):
    str_73986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 12), 'str', 'latex')
    # Getting the type of 'k' (line 520)
    k_73987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 22), 'k')
    # Applying the binary operator '+' (line 520)
    result_add_73988 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 12), '+', str_73986, k_73987)
    
    # Assigning a type to the variable 'k' (line 520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'k', result_add_73988)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'k' (line 521)
    k_73989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 11), 'k')
    # Getting the type of 'rd' (line 521)
    rd_73990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 16), 'rd')
    # Applying the binary operator 'in' (line 521)
    result_contains_73991 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 11), 'in', k_73989, rd_73990)
    
    
    # Call to isinstance(...): (line 521)
    # Processing the call arguments (line 521)
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 521)
    k_73993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 37), 'k', False)
    # Getting the type of 'rd' (line 521)
    rd_73994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 34), 'rd', False)
    # Obtaining the member '__getitem__' of a type (line 521)
    getitem___73995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 34), rd_73994, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 521)
    subscript_call_result_73996 = invoke(stypy.reporting.localization.Localization(__file__, 521, 34), getitem___73995, k_73993)
    
    # Getting the type of 'list' (line 521)
    list_73997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 41), 'list', False)
    # Processing the call keyword arguments (line 521)
    kwargs_73998 = {}
    # Getting the type of 'isinstance' (line 521)
    isinstance_73992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 23), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 521)
    isinstance_call_result_73999 = invoke(stypy.reporting.localization.Localization(__file__, 521, 23), isinstance_73992, *[subscript_call_result_73996, list_73997], **kwargs_73998)
    
    # Applying the binary operator 'and' (line 521)
    result_and_keyword_74000 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 11), 'and', result_contains_73991, isinstance_call_result_73999)
    
    # Testing the type of an if condition (line 521)
    if_condition_74001 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 521, 8), result_and_keyword_74000)
    # Assigning a type to the variable 'if_condition_74001' (line 521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'if_condition_74001', if_condition_74001)
    # SSA begins for if statement (line 521)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 522):
    
    # Assigning a BinOp to a Subscript (line 522):
    
    # Obtaining the type of the subscript
    str_74002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 40), 'str', 'latexdocstrsigns')
    # Getting the type of 'rd' (line 522)
    rd_74003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 37), 'rd')
    # Obtaining the member '__getitem__' of a type (line 522)
    getitem___74004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 37), rd_74003, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 522)
    subscript_call_result_74005 = invoke(stypy.reporting.localization.Localization(__file__, 522, 37), getitem___74004, str_74002)
    
    
    # Obtaining the type of the subscript
    int_74006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 68), 'int')
    int_74007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 70), 'int')
    slice_74008 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 522, 62), int_74006, int_74007, None)
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 522)
    k_74009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 65), 'k')
    # Getting the type of 'rd' (line 522)
    rd_74010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 62), 'rd')
    # Obtaining the member '__getitem__' of a type (line 522)
    getitem___74011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 62), rd_74010, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 522)
    subscript_call_result_74012 = invoke(stypy.reporting.localization.Localization(__file__, 522, 62), getitem___74011, k_74009)
    
    # Obtaining the member '__getitem__' of a type (line 522)
    getitem___74013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 62), subscript_call_result_74012, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 522)
    subscript_call_result_74014 = invoke(stypy.reporting.localization.Localization(__file__, 522, 62), getitem___74013, slice_74008)
    
    # Applying the binary operator '+' (line 522)
    result_add_74015 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 37), '+', subscript_call_result_74005, subscript_call_result_74014)
    
    
    # Obtaining an instance of the builtin type 'list' (line 523)
    list_74016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 523)
    # Adding element type (line 523)
    str_74017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 17), 'str', '\\begin{description}')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 16), list_74016, str_74017)
    
    # Applying the binary operator '+' (line 522)
    result_add_74018 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 73), '+', result_add_74015, list_74016)
    
    
    # Obtaining the type of the subscript
    int_74019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 49), 'int')
    slice_74020 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 523, 43), int_74019, None, None)
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 523)
    k_74021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 46), 'k')
    # Getting the type of 'rd' (line 523)
    rd_74022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 43), 'rd')
    # Obtaining the member '__getitem__' of a type (line 523)
    getitem___74023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 43), rd_74022, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 523)
    subscript_call_result_74024 = invoke(stypy.reporting.localization.Localization(__file__, 523, 43), getitem___74023, k_74021)
    
    # Obtaining the member '__getitem__' of a type (line 523)
    getitem___74025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 43), subscript_call_result_74024, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 523)
    subscript_call_result_74026 = invoke(stypy.reporting.localization.Localization(__file__, 523, 43), getitem___74025, slice_74020)
    
    # Applying the binary operator '+' (line 523)
    result_add_74027 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 41), '+', result_add_74018, subscript_call_result_74026)
    
    
    # Obtaining an instance of the builtin type 'list' (line 524)
    list_74028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 524)
    # Adding element type (line 524)
    str_74029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 17), 'str', '\\end{description}')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 16), list_74028, str_74029)
    
    # Applying the binary operator '+' (line 523)
    result_add_74030 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 53), '+', result_add_74027, list_74028)
    
    # Getting the type of 'rd' (line 522)
    rd_74031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'rd')
    str_74032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 15), 'str', 'latexdocstrsigns')
    # Storing an element on a container (line 522)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 12), rd_74031, (str_74032, result_add_74030))
    # SSA join for if statement (line 521)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_74033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 7), 'str', 'args')
    # Getting the type of 'rd' (line 525)
    rd_74034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 21), 'rd')
    # Applying the binary operator 'notin' (line 525)
    result_contains_74035 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 7), 'notin', str_74033, rd_74034)
    
    # Testing the type of an if condition (line 525)
    if_condition_74036 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 525, 4), result_contains_74035)
    # Assigning a type to the variable 'if_condition_74036' (line 525)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 4), 'if_condition_74036', if_condition_74036)
    # SSA begins for if statement (line 525)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Subscript (line 526):
    
    # Assigning a Str to a Subscript (line 526):
    str_74037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 21), 'str', '')
    # Getting the type of 'rd' (line 526)
    rd_74038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'rd')
    str_74039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 11), 'str', 'args')
    # Storing an element on a container (line 526)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 526, 8), rd_74038, (str_74039, str_74037))
    
    # Assigning a Str to a Subscript (line 527):
    
    # Assigning a Str to a Subscript (line 527):
    str_74040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 24), 'str', '')
    # Getting the type of 'rd' (line 527)
    rd_74041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'rd')
    str_74042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 11), 'str', 'args_td')
    # Storing an element on a container (line 527)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 8), rd_74041, (str_74042, str_74040))
    
    # Assigning a Str to a Subscript (line 528):
    
    # Assigning a Str to a Subscript (line 528):
    str_74043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 24), 'str', '')
    # Getting the type of 'rd' (line 528)
    rd_74044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 8), 'rd')
    str_74045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 11), 'str', 'args_nm')
    # Storing an element on a container (line 528)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 528, 8), rd_74044, (str_74045, str_74043))
    # SSA join for if statement (line 525)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Evaluating a boolean operation
    
    # Call to get(...): (line 529)
    # Processing the call arguments (line 529)
    str_74048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 19), 'str', 'args')
    # Processing the call keyword arguments (line 529)
    kwargs_74049 = {}
    # Getting the type of 'rd' (line 529)
    rd_74046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'rd', False)
    # Obtaining the member 'get' of a type (line 529)
    get_74047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 12), rd_74046, 'get')
    # Calling get(args, kwargs) (line 529)
    get_call_result_74050 = invoke(stypy.reporting.localization.Localization(__file__, 529, 12), get_74047, *[str_74048], **kwargs_74049)
    
    
    # Call to get(...): (line 529)
    # Processing the call arguments (line 529)
    str_74053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 37), 'str', 'optargs')
    # Processing the call keyword arguments (line 529)
    kwargs_74054 = {}
    # Getting the type of 'rd' (line 529)
    rd_74051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 30), 'rd', False)
    # Obtaining the member 'get' of a type (line 529)
    get_74052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 30), rd_74051, 'get')
    # Calling get(args, kwargs) (line 529)
    get_call_result_74055 = invoke(stypy.reporting.localization.Localization(__file__, 529, 30), get_74052, *[str_74053], **kwargs_74054)
    
    # Applying the binary operator 'or' (line 529)
    result_or_keyword_74056 = python_operator(stypy.reporting.localization.Localization(__file__, 529, 12), 'or', get_call_result_74050, get_call_result_74055)
    
    # Call to get(...): (line 529)
    # Processing the call arguments (line 529)
    str_74059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 58), 'str', 'strarglens')
    # Processing the call keyword arguments (line 529)
    kwargs_74060 = {}
    # Getting the type of 'rd' (line 529)
    rd_74057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 51), 'rd', False)
    # Obtaining the member 'get' of a type (line 529)
    get_74058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 51), rd_74057, 'get')
    # Calling get(args, kwargs) (line 529)
    get_call_result_74061 = invoke(stypy.reporting.localization.Localization(__file__, 529, 51), get_74058, *[str_74059], **kwargs_74060)
    
    # Applying the binary operator 'or' (line 529)
    result_or_keyword_74062 = python_operator(stypy.reporting.localization.Localization(__file__, 529, 12), 'or', result_or_keyword_74056, get_call_result_74061)
    
    # Applying the 'not' unary operator (line 529)
    result_not__74063 = python_operator(stypy.reporting.localization.Localization(__file__, 529, 7), 'not', result_or_keyword_74062)
    
    # Testing the type of an if condition (line 529)
    if_condition_74064 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 529, 4), result_not__74063)
    # Assigning a type to the variable 'if_condition_74064' (line 529)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 4), 'if_condition_74064', if_condition_74064)
    # SSA begins for if statement (line 529)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Subscript (line 530):
    
    # Assigning a Str to a Subscript (line 530):
    str_74065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 23), 'str', 'void')
    # Getting the type of 'rd' (line 530)
    rd_74066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'rd')
    str_74067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 11), 'str', 'noargs')
    # Storing an element on a container (line 530)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 530, 8), rd_74066, (str_74067, str_74065))
    # SSA join for if statement (line 529)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 532):
    
    # Assigning a Call to a Name (line 532):
    
    # Call to applyrules(...): (line 532)
    # Processing the call arguments (line 532)
    # Getting the type of 'cb_routine_rules' (line 532)
    cb_routine_rules_74069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 20), 'cb_routine_rules', False)
    # Getting the type of 'rd' (line 532)
    rd_74070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 38), 'rd', False)
    # Processing the call keyword arguments (line 532)
    kwargs_74071 = {}
    # Getting the type of 'applyrules' (line 532)
    applyrules_74068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 9), 'applyrules', False)
    # Calling applyrules(args, kwargs) (line 532)
    applyrules_call_result_74072 = invoke(stypy.reporting.localization.Localization(__file__, 532, 9), applyrules_74068, *[cb_routine_rules_74069, rd_74070], **kwargs_74071)
    
    # Assigning a type to the variable 'ar' (line 532)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 4), 'ar', applyrules_call_result_74072)
    
    # Assigning a Subscript to a Subscript (line 533):
    
    # Assigning a Subscript to a Subscript (line 533):
    
    # Obtaining the type of the subscript
    str_74073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 38), 'str', 'body')
    # Getting the type of 'ar' (line 533)
    ar_74074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 35), 'ar')
    # Obtaining the member '__getitem__' of a type (line 533)
    getitem___74075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 35), ar_74074, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 533)
    subscript_call_result_74076 = invoke(stypy.reporting.localization.Localization(__file__, 533, 35), getitem___74075, str_74073)
    
    # Getting the type of 'cfuncs' (line 533)
    cfuncs_74077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 4), 'cfuncs')
    # Obtaining the member 'callbacks' of a type (line 533)
    callbacks_74078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 4), cfuncs_74077, 'callbacks')
    
    # Obtaining the type of the subscript
    str_74079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 24), 'str', 'name')
    # Getting the type of 'rd' (line 533)
    rd_74080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 21), 'rd')
    # Obtaining the member '__getitem__' of a type (line 533)
    getitem___74081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 21), rd_74080, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 533)
    subscript_call_result_74082 = invoke(stypy.reporting.localization.Localization(__file__, 533, 21), getitem___74081, str_74079)
    
    # Storing an element on a container (line 533)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 4), callbacks_74078, (subscript_call_result_74082, subscript_call_result_74076))
    
    # Type idiom detected: calculating its left and rigth part (line 534)
    # Getting the type of 'str' (line 534)
    str_74083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 30), 'str')
    
    # Obtaining the type of the subscript
    str_74084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 21), 'str', 'need')
    # Getting the type of 'ar' (line 534)
    ar_74085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 18), 'ar')
    # Obtaining the member '__getitem__' of a type (line 534)
    getitem___74086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 18), ar_74085, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 534)
    subscript_call_result_74087 = invoke(stypy.reporting.localization.Localization(__file__, 534, 18), getitem___74086, str_74084)
    
    
    (may_be_74088, more_types_in_union_74089) = may_be_subtype(str_74083, subscript_call_result_74087)

    if may_be_74088:

        if more_types_in_union_74089:
            # Runtime conditional SSA (line 534)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a List to a Subscript (line 535):
        
        # Assigning a List to a Subscript (line 535):
        
        # Obtaining an instance of the builtin type 'list' (line 535)
        list_74090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 535)
        # Adding element type (line 535)
        
        # Obtaining the type of the subscript
        str_74091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 25), 'str', 'need')
        # Getting the type of 'ar' (line 535)
        ar_74092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 22), 'ar')
        # Obtaining the member '__getitem__' of a type (line 535)
        getitem___74093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 22), ar_74092, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 535)
        subscript_call_result_74094 = invoke(stypy.reporting.localization.Localization(__file__, 535, 22), getitem___74093, str_74091)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 21), list_74090, subscript_call_result_74094)
        
        # Getting the type of 'ar' (line 535)
        ar_74095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'ar')
        str_74096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 11), 'str', 'need')
        # Storing an element on a container (line 535)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 8), ar_74095, (str_74096, list_74090))

        if more_types_in_union_74089:
            # SSA join for if statement (line 534)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    str_74097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 7), 'str', 'need')
    # Getting the type of 'rd' (line 537)
    rd_74098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 17), 'rd')
    # Applying the binary operator 'in' (line 537)
    result_contains_74099 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 7), 'in', str_74097, rd_74098)
    
    # Testing the type of an if condition (line 537)
    if_condition_74100 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 537, 4), result_contains_74099)
    # Assigning a type to the variable 'if_condition_74100' (line 537)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 4), 'if_condition_74100', if_condition_74100)
    # SSA begins for if statement (line 537)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to keys(...): (line 538)
    # Processing the call keyword arguments (line 538)
    kwargs_74104 = {}
    # Getting the type of 'cfuncs' (line 538)
    cfuncs_74101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 17), 'cfuncs', False)
    # Obtaining the member 'typedefs' of a type (line 538)
    typedefs_74102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 17), cfuncs_74101, 'typedefs')
    # Obtaining the member 'keys' of a type (line 538)
    keys_74103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 17), typedefs_74102, 'keys')
    # Calling keys(args, kwargs) (line 538)
    keys_call_result_74105 = invoke(stypy.reporting.localization.Localization(__file__, 538, 17), keys_74103, *[], **kwargs_74104)
    
    # Testing the type of a for loop iterable (line 538)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 538, 8), keys_call_result_74105)
    # Getting the type of the for loop variable (line 538)
    for_loop_var_74106 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 538, 8), keys_call_result_74105)
    # Assigning a type to the variable 't' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), 't', for_loop_var_74106)
    # SSA begins for a for statement (line 538)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 't' (line 539)
    t_74107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 15), 't')
    
    # Obtaining the type of the subscript
    str_74108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 23), 'str', 'need')
    # Getting the type of 'rd' (line 539)
    rd_74109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 20), 'rd')
    # Obtaining the member '__getitem__' of a type (line 539)
    getitem___74110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 20), rd_74109, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 539)
    subscript_call_result_74111 = invoke(stypy.reporting.localization.Localization(__file__, 539, 20), getitem___74110, str_74108)
    
    # Applying the binary operator 'in' (line 539)
    result_contains_74112 = python_operator(stypy.reporting.localization.Localization(__file__, 539, 15), 'in', t_74107, subscript_call_result_74111)
    
    # Testing the type of an if condition (line 539)
    if_condition_74113 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 539, 12), result_contains_74112)
    # Assigning a type to the variable 'if_condition_74113' (line 539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 12), 'if_condition_74113', if_condition_74113)
    # SSA begins for if statement (line 539)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 540)
    # Processing the call arguments (line 540)
    # Getting the type of 't' (line 540)
    t_74119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 34), 't', False)
    # Processing the call keyword arguments (line 540)
    kwargs_74120 = {}
    
    # Obtaining the type of the subscript
    str_74114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 19), 'str', 'need')
    # Getting the type of 'ar' (line 540)
    ar_74115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 16), 'ar', False)
    # Obtaining the member '__getitem__' of a type (line 540)
    getitem___74116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 16), ar_74115, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 540)
    subscript_call_result_74117 = invoke(stypy.reporting.localization.Localization(__file__, 540, 16), getitem___74116, str_74114)
    
    # Obtaining the member 'append' of a type (line 540)
    append_74118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 16), subscript_call_result_74117, 'append')
    # Calling append(args, kwargs) (line 540)
    append_call_result_74121 = invoke(stypy.reporting.localization.Localization(__file__, 540, 16), append_74118, *[t_74119], **kwargs_74120)
    
    # SSA join for if statement (line 539)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 537)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Subscript (line 542):
    
    # Assigning a Subscript to a Subscript (line 542):
    
    # Obtaining the type of the subscript
    str_74122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 60), 'str', 'cbtypedefs')
    # Getting the type of 'ar' (line 542)
    ar_74123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 57), 'ar')
    # Obtaining the member '__getitem__' of a type (line 542)
    getitem___74124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 57), ar_74123, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 542)
    subscript_call_result_74125 = invoke(stypy.reporting.localization.Localization(__file__, 542, 57), getitem___74124, str_74122)
    
    # Getting the type of 'cfuncs' (line 542)
    cfuncs_74126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 4), 'cfuncs')
    # Obtaining the member 'typedefs_generated' of a type (line 542)
    typedefs_generated_74127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 4), cfuncs_74126, 'typedefs_generated')
    
    # Obtaining the type of the subscript
    str_74128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 33), 'str', 'name')
    # Getting the type of 'rd' (line 542)
    rd_74129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 30), 'rd')
    # Obtaining the member '__getitem__' of a type (line 542)
    getitem___74130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 30), rd_74129, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 542)
    subscript_call_result_74131 = invoke(stypy.reporting.localization.Localization(__file__, 542, 30), getitem___74130, str_74128)
    
    str_74132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 43), 'str', '_typedef')
    # Applying the binary operator '+' (line 542)
    result_add_74133 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 30), '+', subscript_call_result_74131, str_74132)
    
    # Storing an element on a container (line 542)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 4), typedefs_generated_74127, (result_add_74133, subscript_call_result_74125))
    
    # Call to append(...): (line 543)
    # Processing the call arguments (line 543)
    
    # Obtaining the type of the subscript
    str_74139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 25), 'str', 'name')
    # Getting the type of 'rd' (line 543)
    rd_74140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 22), 'rd', False)
    # Obtaining the member '__getitem__' of a type (line 543)
    getitem___74141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 22), rd_74140, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 543)
    subscript_call_result_74142 = invoke(stypy.reporting.localization.Localization(__file__, 543, 22), getitem___74141, str_74139)
    
    str_74143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 35), 'str', '_typedef')
    # Applying the binary operator '+' (line 543)
    result_add_74144 = python_operator(stypy.reporting.localization.Localization(__file__, 543, 22), '+', subscript_call_result_74142, str_74143)
    
    # Processing the call keyword arguments (line 543)
    kwargs_74145 = {}
    
    # Obtaining the type of the subscript
    str_74134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 7), 'str', 'need')
    # Getting the type of 'ar' (line 543)
    ar_74135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 4), 'ar', False)
    # Obtaining the member '__getitem__' of a type (line 543)
    getitem___74136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 4), ar_74135, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 543)
    subscript_call_result_74137 = invoke(stypy.reporting.localization.Localization(__file__, 543, 4), getitem___74136, str_74134)
    
    # Obtaining the member 'append' of a type (line 543)
    append_74138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 4), subscript_call_result_74137, 'append')
    # Calling append(args, kwargs) (line 543)
    append_call_result_74146 = invoke(stypy.reporting.localization.Localization(__file__, 543, 4), append_74138, *[result_add_74144], **kwargs_74145)
    
    
    # Assigning a Subscript to a Subscript (line 544):
    
    # Assigning a Subscript to a Subscript (line 544):
    
    # Obtaining the type of the subscript
    str_74147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 34), 'str', 'need')
    # Getting the type of 'ar' (line 544)
    ar_74148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 31), 'ar')
    # Obtaining the member '__getitem__' of a type (line 544)
    getitem___74149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 31), ar_74148, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 544)
    subscript_call_result_74150 = invoke(stypy.reporting.localization.Localization(__file__, 544, 31), getitem___74149, str_74147)
    
    # Getting the type of 'cfuncs' (line 544)
    cfuncs_74151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 4), 'cfuncs')
    # Obtaining the member 'needs' of a type (line 544)
    needs_74152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 4), cfuncs_74151, 'needs')
    
    # Obtaining the type of the subscript
    str_74153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 20), 'str', 'name')
    # Getting the type of 'rd' (line 544)
    rd_74154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 17), 'rd')
    # Obtaining the member '__getitem__' of a type (line 544)
    getitem___74155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 17), rd_74154, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 544)
    subscript_call_result_74156 = invoke(stypy.reporting.localization.Localization(__file__, 544, 17), getitem___74155, str_74153)
    
    # Storing an element on a container (line 544)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 544, 4), needs_74152, (subscript_call_result_74156, subscript_call_result_74150))
    
    # Assigning a Dict to a Subscript (line 546):
    
    # Assigning a Dict to a Subscript (line 546):
    
    # Obtaining an instance of the builtin type 'dict' (line 546)
    dict_74157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 37), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 546)
    # Adding element type (key, value) (line 546)
    str_74158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 38), 'str', 'maxnofargs')
    
    # Obtaining the type of the subscript
    str_74159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 55), 'str', 'maxnofargs')
    # Getting the type of 'ar' (line 546)
    ar_74160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 52), 'ar')
    # Obtaining the member '__getitem__' of a type (line 546)
    getitem___74161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 52), ar_74160, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 546)
    subscript_call_result_74162 = invoke(stypy.reporting.localization.Localization(__file__, 546, 52), getitem___74161, str_74159)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 37), dict_74157, (str_74158, subscript_call_result_74162))
    # Adding element type (key, value) (line 546)
    str_74163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 38), 'str', 'nofoptargs')
    
    # Obtaining the type of the subscript
    str_74164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 55), 'str', 'nofoptargs')
    # Getting the type of 'ar' (line 547)
    ar_74165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 52), 'ar')
    # Obtaining the member '__getitem__' of a type (line 547)
    getitem___74166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 52), ar_74165, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 547)
    subscript_call_result_74167 = invoke(stypy.reporting.localization.Localization(__file__, 547, 52), getitem___74166, str_74164)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 37), dict_74157, (str_74163, subscript_call_result_74167))
    # Adding element type (key, value) (line 546)
    str_74168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 38), 'str', 'docstr')
    
    # Obtaining the type of the subscript
    str_74169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 51), 'str', 'docstr')
    # Getting the type of 'ar' (line 548)
    ar_74170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 48), 'ar')
    # Obtaining the member '__getitem__' of a type (line 548)
    getitem___74171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 48), ar_74170, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 548)
    subscript_call_result_74172 = invoke(stypy.reporting.localization.Localization(__file__, 548, 48), getitem___74171, str_74169)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 37), dict_74157, (str_74168, subscript_call_result_74172))
    # Adding element type (key, value) (line 546)
    str_74173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 38), 'str', 'latexdocstr')
    
    # Obtaining the type of the subscript
    str_74174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 56), 'str', 'latexdocstr')
    # Getting the type of 'ar' (line 549)
    ar_74175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 53), 'ar')
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___74176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 53), ar_74175, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 549)
    subscript_call_result_74177 = invoke(stypy.reporting.localization.Localization(__file__, 549, 53), getitem___74176, str_74174)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 37), dict_74157, (str_74173, subscript_call_result_74177))
    # Adding element type (key, value) (line 546)
    str_74178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 38), 'str', 'argname')
    
    # Obtaining the type of the subscript
    str_74179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 52), 'str', 'argname')
    # Getting the type of 'rd' (line 550)
    rd_74180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 49), 'rd')
    # Obtaining the member '__getitem__' of a type (line 550)
    getitem___74181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 49), rd_74180, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 550)
    subscript_call_result_74182 = invoke(stypy.reporting.localization.Localization(__file__, 550, 49), getitem___74181, str_74179)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 37), dict_74157, (str_74178, subscript_call_result_74182))
    
    # Getting the type of 'capi_maps' (line 546)
    capi_maps_74183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 'capi_maps')
    # Obtaining the member 'lcb2_map' of a type (line 546)
    lcb2_map_74184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 4), capi_maps_74183, 'lcb2_map')
    
    # Obtaining the type of the subscript
    str_74185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 26), 'str', 'name')
    # Getting the type of 'rd' (line 546)
    rd_74186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 23), 'rd')
    # Obtaining the member '__getitem__' of a type (line 546)
    getitem___74187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 23), rd_74186, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 546)
    subscript_call_result_74188 = invoke(stypy.reporting.localization.Localization(__file__, 546, 23), getitem___74187, str_74185)
    
    # Storing an element on a container (line 546)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 4), lcb2_map_74184, (subscript_call_result_74188, dict_74157))
    
    # Call to outmess(...): (line 552)
    # Processing the call arguments (line 552)
    str_74190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 12), 'str', '\t  %s\n')
    
    # Obtaining the type of the subscript
    str_74191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 29), 'str', 'docstrshort')
    # Getting the type of 'ar' (line 552)
    ar_74192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 26), 'ar', False)
    # Obtaining the member '__getitem__' of a type (line 552)
    getitem___74193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 26), ar_74192, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 552)
    subscript_call_result_74194 = invoke(stypy.reporting.localization.Localization(__file__, 552, 26), getitem___74193, str_74191)
    
    # Applying the binary operator '%' (line 552)
    result_mod_74195 = python_operator(stypy.reporting.localization.Localization(__file__, 552, 12), '%', str_74190, subscript_call_result_74194)
    
    # Processing the call keyword arguments (line 552)
    kwargs_74196 = {}
    # Getting the type of 'outmess' (line 552)
    outmess_74189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 4), 'outmess', False)
    # Calling outmess(args, kwargs) (line 552)
    outmess_call_result_74197 = invoke(stypy.reporting.localization.Localization(__file__, 552, 4), outmess_74189, *[result_mod_74195], **kwargs_74196)
    
    # Assigning a type to the variable 'stypy_return_type' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'stypy_return_type', types.NoneType)
    
    # ################# End of 'buildcallback(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'buildcallback' in the type store
    # Getting the type of 'stypy_return_type' (line 428)
    stypy_return_type_74198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_74198)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'buildcallback'
    return stypy_return_type_74198

# Assigning a type to the variable 'buildcallback' (line 428)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 0), 'buildcallback', buildcallback)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
