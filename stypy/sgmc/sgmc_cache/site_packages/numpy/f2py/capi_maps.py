
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: '''
3: 
4: Copyright 1999,2000 Pearu Peterson all rights reserved,
5: Pearu Peterson <pearu@ioc.ee>
6: Permission to use, modify, and distribute this software is given under the
7: terms of the NumPy License.
8: 
9: NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
10: $Date: 2005/05/06 10:57:33 $
11: Pearu Peterson
12: 
13: '''
14: from __future__ import division, absolute_import, print_function
15: 
16: __version__ = "$Revision: 1.60 $"[10:-1]
17: 
18: from . import __version__
19: f2py_version = __version__.version
20: 
21: import copy
22: import re
23: import os
24: import sys
25: from .crackfortran import markoutercomma
26: from . import cb_rules
27: 
28: # The eviroment provided by auxfuncs.py is needed for some calls to eval.
29: # As the needed functions cannot be determined by static inspection of the
30: # code, it is safest to use import * pending a major refactoring of f2py.
31: from .auxfuncs import *
32: 
33: __all__ = [
34:     'getctype', 'getstrlength', 'getarrdims', 'getpydocsign',
35:     'getarrdocsign', 'getinit', 'sign2map', 'routsign2map', 'modsign2map',
36:     'cb_sign2map', 'cb_routsign2map', 'common_sign2map'
37: ]
38: 
39: 
40: # Numarray and Numeric users should set this False
41: using_newcore = True
42: 
43: depargs = []
44: lcb_map = {}
45: lcb2_map = {}
46: # forced casting: mainly caused by the fact that Python or Numeric
47: #                 C/APIs do not support the corresponding C types.
48: c2py_map = {'double': 'float',
49:             'float': 'float',                          # forced casting
50:             'long_double': 'float',                    # forced casting
51:             'char': 'int',                             # forced casting
52:             'signed_char': 'int',                      # forced casting
53:             'unsigned_char': 'int',                    # forced casting
54:             'short': 'int',                            # forced casting
55:             'unsigned_short': 'int',                   # forced casting
56:             'int': 'int',                              # (forced casting)
57:             'long': 'int',
58:             'long_long': 'long',
59:             'unsigned': 'int',                         # forced casting
60:             'complex_float': 'complex',                # forced casting
61:             'complex_double': 'complex',
62:             'complex_long_double': 'complex',          # forced casting
63:             'string': 'string',
64:             }
65: c2capi_map = {'double': 'NPY_DOUBLE',
66:               'float': 'NPY_FLOAT',
67:               'long_double': 'NPY_DOUBLE',           # forced casting
68:               'char': 'NPY_CHAR',
69:               'unsigned_char': 'NPY_UBYTE',
70:               'signed_char': 'NPY_BYTE',
71:               'short': 'NPY_SHORT',
72:               'unsigned_short': 'NPY_USHORT',
73:               'int': 'NPY_INT',
74:               'unsigned': 'NPY_UINT',
75:               'long': 'NPY_LONG',
76:               'long_long': 'NPY_LONG',                # forced casting
77:               'complex_float': 'NPY_CFLOAT',
78:               'complex_double': 'NPY_CDOUBLE',
79:               'complex_long_double': 'NPY_CDOUBLE',   # forced casting
80:               'string': 'NPY_CHAR'}
81: 
82: # These new maps aren't used anyhere yet, but should be by default
83: #  unless building numeric or numarray extensions.
84: if using_newcore:
85:     c2capi_map = {'double': 'NPY_DOUBLE',
86:                   'float': 'NPY_FLOAT',
87:                   'long_double': 'NPY_LONGDOUBLE',
88:                   'char': 'NPY_BYTE',
89:                   'unsigned_char': 'NPY_UBYTE',
90:                   'signed_char': 'NPY_BYTE',
91:                   'short': 'NPY_SHORT',
92:                   'unsigned_short': 'NPY_USHORT',
93:                   'int': 'NPY_INT',
94:                   'unsigned': 'NPY_UINT',
95:                   'long': 'NPY_LONG',
96:                   'unsigned_long': 'NPY_ULONG',
97:                   'long_long': 'NPY_LONGLONG',
98:                   'unsigned_long_long': 'NPY_ULONGLONG',
99:                   'complex_float': 'NPY_CFLOAT',
100:                   'complex_double': 'NPY_CDOUBLE',
101:                   'complex_long_double': 'NPY_CDOUBLE',
102:                   # f2py 2e is not ready for NPY_STRING (must set itemisize
103:                   # etc)
104:                   'string': 'NPY_CHAR',
105:                   #'string':'NPY_STRING'
106: 
107:                   }
108: c2pycode_map = {'double': 'd',
109:                 'float': 'f',
110:                 'long_double': 'd',                       # forced casting
111:                 'char': '1',
112:                 'signed_char': '1',
113:                 'unsigned_char': 'b',
114:                 'short': 's',
115:                 'unsigned_short': 'w',
116:                 'int': 'i',
117:                 'unsigned': 'u',
118:                 'long': 'l',
119:                 'long_long': 'L',
120:                 'complex_float': 'F',
121:                 'complex_double': 'D',
122:                 'complex_long_double': 'D',               # forced casting
123:                 'string': 'c'
124:                 }
125: if using_newcore:
126:     c2pycode_map = {'double': 'd',
127:                     'float': 'f',
128:                     'long_double': 'g',
129:                     'char': 'b',
130:                     'unsigned_char': 'B',
131:                     'signed_char': 'b',
132:                     'short': 'h',
133:                     'unsigned_short': 'H',
134:                     'int': 'i',
135:                     'unsigned': 'I',
136:                     'long': 'l',
137:                     'unsigned_long': 'L',
138:                     'long_long': 'q',
139:                     'unsigned_long_long': 'Q',
140:                     'complex_float': 'F',
141:                     'complex_double': 'D',
142:                     'complex_long_double': 'G',
143:                     'string': 'S'}
144: c2buildvalue_map = {'double': 'd',
145:                     'float': 'f',
146:                     'char': 'b',
147:                     'signed_char': 'b',
148:                     'short': 'h',
149:                     'int': 'i',
150:                     'long': 'l',
151:                     'long_long': 'L',
152:                     'complex_float': 'N',
153:                     'complex_double': 'N',
154:                     'complex_long_double': 'N',
155:                     'string': 'z'}
156: 
157: if sys.version_info[0] >= 3:
158:     # Bytes, not Unicode strings
159:     c2buildvalue_map['string'] = 'y'
160: 
161: if using_newcore:
162:     # c2buildvalue_map=???
163:     pass
164: 
165: f2cmap_all = {'real': {'': 'float', '4': 'float', '8': 'double',
166:                        '12': 'long_double', '16': 'long_double'},
167:               'integer': {'': 'int', '1': 'signed_char', '2': 'short',
168:                           '4': 'int', '8': 'long_long',
169:                           '-1': 'unsigned_char', '-2': 'unsigned_short',
170:                           '-4': 'unsigned', '-8': 'unsigned_long_long'},
171:               'complex': {'': 'complex_float', '8': 'complex_float',
172:                           '16': 'complex_double', '24': 'complex_long_double',
173:                           '32': 'complex_long_double'},
174:               'complexkind': {'': 'complex_float', '4': 'complex_float',
175:                               '8': 'complex_double', '12': 'complex_long_double',
176:                               '16': 'complex_long_double'},
177:               'logical': {'': 'int', '1': 'char', '2': 'short', '4': 'int',
178:                           '8': 'long_long'},
179:               'double complex': {'': 'complex_double'},
180:               'double precision': {'': 'double'},
181:               'byte': {'': 'char'},
182:               'character': {'': 'string'}
183:               }
184: 
185: if os.path.isfile('.f2py_f2cmap'):
186:     # User defined additions to f2cmap_all.
187:     # .f2py_f2cmap must contain a dictionary of dictionaries, only.  For
188:     # example, {'real':{'low':'float'}} means that Fortran 'real(low)' is
189:     # interpreted as C 'float'.  This feature is useful for F90/95 users if
190:     # they use PARAMETERSs in type specifications.
191:     try:
192:         outmess('Reading .f2py_f2cmap ...\n')
193:         f = open('.f2py_f2cmap', 'r')
194:         d = eval(f.read(), {}, {})
195:         f.close()
196:         for k, d1 in list(d.items()):
197:             for k1 in list(d1.keys()):
198:                 d1[k1.lower()] = d1[k1]
199:             d[k.lower()] = d[k]
200:         for k in list(d.keys()):
201:             if k not in f2cmap_all:
202:                 f2cmap_all[k] = {}
203:             for k1 in list(d[k].keys()):
204:                 if d[k][k1] in c2py_map:
205:                     if k1 in f2cmap_all[k]:
206:                         outmess(
207:                             "\tWarning: redefinition of {'%s':{'%s':'%s'->'%s'}}\n" % (k, k1, f2cmap_all[k][k1], d[k][k1]))
208:                     f2cmap_all[k][k1] = d[k][k1]
209:                     outmess('\tMapping "%s(kind=%s)" to "%s"\n' %
210:                             (k, k1, d[k][k1]))
211:                 else:
212:                     errmess("\tIgnoring map {'%s':{'%s':'%s'}}: '%s' must be in %s\n" % (
213:                         k, k1, d[k][k1], d[k][k1], list(c2py_map.keys())))
214:         outmess('Succesfully applied user defined changes from .f2py_f2cmap\n')
215:     except Exception as msg:
216:         errmess(
217:             'Failed to apply user defined changes from .f2py_f2cmap: %s. Skipping.\n' % (msg))
218: 
219: cformat_map = {'double': '%g',
220:                'float': '%g',
221:                'long_double': '%Lg',
222:                'char': '%d',
223:                'signed_char': '%d',
224:                'unsigned_char': '%hhu',
225:                'short': '%hd',
226:                'unsigned_short': '%hu',
227:                'int': '%d',
228:                'unsigned': '%u',
229:                'long': '%ld',
230:                'unsigned_long': '%lu',
231:                'long_long': '%ld',
232:                'complex_float': '(%g,%g)',
233:                'complex_double': '(%g,%g)',
234:                'complex_long_double': '(%Lg,%Lg)',
235:                'string': '%s',
236:                }
237: 
238: # Auxiliary functions
239: 
240: 
241: def getctype(var):
242:     '''
243:     Determines C type
244:     '''
245:     ctype = 'void'
246:     if isfunction(var):
247:         if 'result' in var:
248:             a = var['result']
249:         else:
250:             a = var['name']
251:         if a in var['vars']:
252:             return getctype(var['vars'][a])
253:         else:
254:             errmess('getctype: function %s has no return value?!\n' % a)
255:     elif issubroutine(var):
256:         return ctype
257:     elif 'typespec' in var and var['typespec'].lower() in f2cmap_all:
258:         typespec = var['typespec'].lower()
259:         f2cmap = f2cmap_all[typespec]
260:         ctype = f2cmap['']  # default type
261:         if 'kindselector' in var:
262:             if '*' in var['kindselector']:
263:                 try:
264:                     ctype = f2cmap[var['kindselector']['*']]
265:                 except KeyError:
266:                     errmess('getctype: "%s %s %s" not supported.\n' %
267:                             (var['typespec'], '*', var['kindselector']['*']))
268:             elif 'kind' in var['kindselector']:
269:                 if typespec + 'kind' in f2cmap_all:
270:                     f2cmap = f2cmap_all[typespec + 'kind']
271:                 try:
272:                     ctype = f2cmap[var['kindselector']['kind']]
273:                 except KeyError:
274:                     if typespec in f2cmap_all:
275:                         f2cmap = f2cmap_all[typespec]
276:                     try:
277:                         ctype = f2cmap[str(var['kindselector']['kind'])]
278:                     except KeyError:
279:                         errmess('getctype: "%s(kind=%s)" is mapped to C "%s" (to override define dict(%s = dict(%s="<C typespec>")) in %s/.f2py_f2cmap file).\n'
280:                                 % (typespec, var['kindselector']['kind'], ctype,
281:                                    typespec, var['kindselector']['kind'], os.getcwd()))
282: 
283:     else:
284:         if not isexternal(var):
285:             errmess(
286:                 'getctype: No C-type found in "%s", assuming void.\n' % var)
287:     return ctype
288: 
289: 
290: def getstrlength(var):
291:     if isstringfunction(var):
292:         if 'result' in var:
293:             a = var['result']
294:         else:
295:             a = var['name']
296:         if a in var['vars']:
297:             return getstrlength(var['vars'][a])
298:         else:
299:             errmess('getstrlength: function %s has no return value?!\n' % a)
300:     if not isstring(var):
301:         errmess(
302:             'getstrlength: expected a signature of a string but got: %s\n' % (repr(var)))
303:     len = '1'
304:     if 'charselector' in var:
305:         a = var['charselector']
306:         if '*' in a:
307:             len = a['*']
308:         elif 'len' in a:
309:             len = a['len']
310:     if re.match(r'\(\s*([*]|[:])\s*\)', len) or re.match(r'([*]|[:])', len):
311:         if isintent_hide(var):
312:             errmess('getstrlength:intent(hide): expected a string with defined length but got: %s\n' % (
313:                 repr(var)))
314:         len = '-1'
315:     return len
316: 
317: 
318: def getarrdims(a, var, verbose=0):
319:     global depargs
320:     ret = {}
321:     if isstring(var) and not isarray(var):
322:         ret['dims'] = getstrlength(var)
323:         ret['size'] = ret['dims']
324:         ret['rank'] = '1'
325:     elif isscalar(var):
326:         ret['size'] = '1'
327:         ret['rank'] = '0'
328:         ret['dims'] = ''
329:     elif isarray(var):
330:         dim = copy.copy(var['dimension'])
331:         ret['size'] = '*'.join(dim)
332:         try:
333:             ret['size'] = repr(eval(ret['size']))
334:         except:
335:             pass
336:         ret['dims'] = ','.join(dim)
337:         ret['rank'] = repr(len(dim))
338:         ret['rank*[-1]'] = repr(len(dim) * [-1])[1:-1]
339:         for i in range(len(dim)):  # solve dim for dependecies
340:             v = []
341:             if dim[i] in depargs:
342:                 v = [dim[i]]
343:             else:
344:                 for va in depargs:
345:                     if re.match(r'.*?\b%s\b.*' % va, dim[i]):
346:                         v.append(va)
347:             for va in v:
348:                 if depargs.index(va) > depargs.index(a):
349:                     dim[i] = '*'
350:                     break
351:         ret['setdims'], i = '', -1
352:         for d in dim:
353:             i = i + 1
354:             if d not in ['*', ':', '(*)', '(:)']:
355:                 ret['setdims'] = '%s#varname#_Dims[%d]=%s,' % (
356:                     ret['setdims'], i, d)
357:         if ret['setdims']:
358:             ret['setdims'] = ret['setdims'][:-1]
359:         ret['cbsetdims'], i = '', -1
360:         for d in var['dimension']:
361:             i = i + 1
362:             if d not in ['*', ':', '(*)', '(:)']:
363:                 ret['cbsetdims'] = '%s#varname#_Dims[%d]=%s,' % (
364:                     ret['cbsetdims'], i, d)
365:             elif isintent_in(var):
366:                 outmess('getarrdims:warning: assumed shape array, using 0 instead of %r\n'
367:                         % (d))
368:                 ret['cbsetdims'] = '%s#varname#_Dims[%d]=%s,' % (
369:                     ret['cbsetdims'], i, 0)
370:             elif verbose:
371:                 errmess(
372:                     'getarrdims: If in call-back function: array argument %s must have bounded dimensions: got %s\n' % (repr(a), repr(d)))
373:         if ret['cbsetdims']:
374:             ret['cbsetdims'] = ret['cbsetdims'][:-1]
375: #         if not isintent_c(var):
376: #             var['dimension'].reverse()
377:     return ret
378: 
379: 
380: def getpydocsign(a, var):
381:     global lcb_map
382:     if isfunction(var):
383:         if 'result' in var:
384:             af = var['result']
385:         else:
386:             af = var['name']
387:         if af in var['vars']:
388:             return getpydocsign(af, var['vars'][af])
389:         else:
390:             errmess('getctype: function %s has no return value?!\n' % af)
391:         return '', ''
392:     sig, sigout = a, a
393:     opt = ''
394:     if isintent_in(var):
395:         opt = 'input'
396:     elif isintent_inout(var):
397:         opt = 'in/output'
398:     out_a = a
399:     if isintent_out(var):
400:         for k in var['intent']:
401:             if k[:4] == 'out=':
402:                 out_a = k[4:]
403:                 break
404:     init = ''
405:     ctype = getctype(var)
406: 
407:     if hasinitvalue(var):
408:         init, showinit = getinit(a, var)
409:         init = ', optional\\n    Default: %s' % showinit
410:     if isscalar(var):
411:         if isintent_inout(var):
412:             sig = '%s : %s rank-0 array(%s,\'%s\')%s' % (a, opt, c2py_map[ctype],
413:                                                          c2pycode_map[ctype], init)
414:         else:
415:             sig = '%s : %s %s%s' % (a, opt, c2py_map[ctype], init)
416:         sigout = '%s : %s' % (out_a, c2py_map[ctype])
417:     elif isstring(var):
418:         if isintent_inout(var):
419:             sig = '%s : %s rank-0 array(string(len=%s),\'c\')%s' % (
420:                 a, opt, getstrlength(var), init)
421:         else:
422:             sig = '%s : %s string(len=%s)%s' % (
423:                 a, opt, getstrlength(var), init)
424:         sigout = '%s : string(len=%s)' % (out_a, getstrlength(var))
425:     elif isarray(var):
426:         dim = var['dimension']
427:         rank = repr(len(dim))
428:         sig = '%s : %s rank-%s array(\'%s\') with bounds (%s)%s' % (a, opt, rank,
429:                                                                     c2pycode_map[
430:                                                                         ctype],
431:                                                                     ','.join(dim), init)
432:         if a == out_a:
433:             sigout = '%s : rank-%s array(\'%s\') with bounds (%s)'\
434:                 % (a, rank, c2pycode_map[ctype], ','.join(dim))
435:         else:
436:             sigout = '%s : rank-%s array(\'%s\') with bounds (%s) and %s storage'\
437:                 % (out_a, rank, c2pycode_map[ctype], ','.join(dim), a)
438:     elif isexternal(var):
439:         ua = ''
440:         if a in lcb_map and lcb_map[a] in lcb2_map and 'argname' in lcb2_map[lcb_map[a]]:
441:             ua = lcb2_map[lcb_map[a]]['argname']
442:             if not ua == a:
443:                 ua = ' => %s' % ua
444:             else:
445:                 ua = ''
446:         sig = '%s : call-back function%s' % (a, ua)
447:         sigout = sig
448:     else:
449:         errmess(
450:             'getpydocsign: Could not resolve docsignature for "%s".\\n' % a)
451:     return sig, sigout
452: 
453: 
454: def getarrdocsign(a, var):
455:     ctype = getctype(var)
456:     if isstring(var) and (not isarray(var)):
457:         sig = '%s : rank-0 array(string(len=%s),\'c\')' % (a,
458:                                                            getstrlength(var))
459:     elif isscalar(var):
460:         sig = '%s : rank-0 array(%s,\'%s\')' % (a, c2py_map[ctype],
461:                                                 c2pycode_map[ctype],)
462:     elif isarray(var):
463:         dim = var['dimension']
464:         rank = repr(len(dim))
465:         sig = '%s : rank-%s array(\'%s\') with bounds (%s)' % (a, rank,
466:                                                                c2pycode_map[
467:                                                                    ctype],
468:                                                                ','.join(dim))
469:     return sig
470: 
471: 
472: def getinit(a, var):
473:     if isstring(var):
474:         init, showinit = '""', "''"
475:     else:
476:         init, showinit = '', ''
477:     if hasinitvalue(var):
478:         init = var['=']
479:         showinit = init
480:         if iscomplex(var) or iscomplexarray(var):
481:             ret = {}
482: 
483:             try:
484:                 v = var["="]
485:                 if ',' in v:
486:                     ret['init.r'], ret['init.i'] = markoutercomma(
487:                         v[1:-1]).split('@,@')
488:                 else:
489:                     v = eval(v, {}, {})
490:                     ret['init.r'], ret['init.i'] = str(v.real), str(v.imag)
491:             except:
492:                 raise ValueError(
493:                     'getinit: expected complex number `(r,i)\' but got `%s\' as initial value of %r.' % (init, a))
494:             if isarray(var):
495:                 init = '(capi_c.r=%s,capi_c.i=%s,capi_c)' % (
496:                     ret['init.r'], ret['init.i'])
497:         elif isstring(var):
498:             if not init:
499:                 init, showinit = '""', "''"
500:             if init[0] == "'":
501:                 init = '"%s"' % (init[1:-1].replace('"', '\\"'))
502:             if init[0] == '"':
503:                 showinit = "'%s'" % (init[1:-1])
504:     return init, showinit
505: 
506: 
507: def sign2map(a, var):
508:     '''
509:     varname,ctype,atype
510:     init,init.r,init.i,pytype
511:     vardebuginfo,vardebugshowvalue,varshowvalue
512:     varrfromat
513:     intent
514:     '''
515:     global lcb_map, cb_map
516:     out_a = a
517:     if isintent_out(var):
518:         for k in var['intent']:
519:             if k[:4] == 'out=':
520:                 out_a = k[4:]
521:                 break
522:     ret = {'varname': a, 'outvarname': out_a, 'ctype': getctype(var)}
523:     intent_flags = []
524:     for f, s in isintent_dict.items():
525:         if f(var):
526:             intent_flags.append('F2PY_%s' % s)
527:     if intent_flags:
528:         # XXX: Evaluate intent_flags here.
529:         ret['intent'] = '|'.join(intent_flags)
530:     else:
531:         ret['intent'] = 'F2PY_INTENT_IN'
532:     if isarray(var):
533:         ret['varrformat'] = 'N'
534:     elif ret['ctype'] in c2buildvalue_map:
535:         ret['varrformat'] = c2buildvalue_map[ret['ctype']]
536:     else:
537:         ret['varrformat'] = 'O'
538:     ret['init'], ret['showinit'] = getinit(a, var)
539:     if hasinitvalue(var) and iscomplex(var) and not isarray(var):
540:         ret['init.r'], ret['init.i'] = markoutercomma(
541:             ret['init'][1:-1]).split('@,@')
542:     if isexternal(var):
543:         ret['cbnamekey'] = a
544:         if a in lcb_map:
545:             ret['cbname'] = lcb_map[a]
546:             ret['maxnofargs'] = lcb2_map[lcb_map[a]]['maxnofargs']
547:             ret['nofoptargs'] = lcb2_map[lcb_map[a]]['nofoptargs']
548:             ret['cbdocstr'] = lcb2_map[lcb_map[a]]['docstr']
549:             ret['cblatexdocstr'] = lcb2_map[lcb_map[a]]['latexdocstr']
550:         else:
551:             ret['cbname'] = a
552:             errmess('sign2map: Confused: external %s is not in lcb_map%s.\n' % (
553:                 a, list(lcb_map.keys())))
554:     if isstring(var):
555:         ret['length'] = getstrlength(var)
556:     if isarray(var):
557:         ret = dictappend(ret, getarrdims(a, var))
558:         dim = copy.copy(var['dimension'])
559:     if ret['ctype'] in c2capi_map:
560:         ret['atype'] = c2capi_map[ret['ctype']]
561:     # Debug info
562:     if debugcapi(var):
563:         il = [isintent_in, 'input', isintent_out, 'output',
564:               isintent_inout, 'inoutput', isrequired, 'required',
565:               isoptional, 'optional', isintent_hide, 'hidden',
566:               iscomplex, 'complex scalar',
567:               l_and(isscalar, l_not(iscomplex)), 'scalar',
568:               isstring, 'string', isarray, 'array',
569:               iscomplexarray, 'complex array', isstringarray, 'string array',
570:               iscomplexfunction, 'complex function',
571:               l_and(isfunction, l_not(iscomplexfunction)), 'function',
572:               isexternal, 'callback',
573:               isintent_callback, 'callback',
574:               isintent_aux, 'auxiliary',
575:               ]
576:         rl = []
577:         for i in range(0, len(il), 2):
578:             if il[i](var):
579:                 rl.append(il[i + 1])
580:         if isstring(var):
581:             rl.append('slen(%s)=%s' % (a, ret['length']))
582:         if isarray(var):
583:             ddim = ','.join(
584:                 map(lambda x, y: '%s|%s' % (x, y), var['dimension'], dim))
585:             rl.append('dims(%s)' % ddim)
586:         if isexternal(var):
587:             ret['vardebuginfo'] = 'debug-capi:%s=>%s:%s' % (
588:                 a, ret['cbname'], ','.join(rl))
589:         else:
590:             ret['vardebuginfo'] = 'debug-capi:%s %s=%s:%s' % (
591:                 ret['ctype'], a, ret['showinit'], ','.join(rl))
592:         if isscalar(var):
593:             if ret['ctype'] in cformat_map:
594:                 ret['vardebugshowvalue'] = 'debug-capi:%s=%s' % (
595:                     a, cformat_map[ret['ctype']])
596:         if isstring(var):
597:             ret['vardebugshowvalue'] = 'debug-capi:slen(%s)=%%d %s=\\"%%s\\"' % (
598:                 a, a)
599:         if isexternal(var):
600:             ret['vardebugshowvalue'] = 'debug-capi:%s=%%p' % (a)
601:     if ret['ctype'] in cformat_map:
602:         ret['varshowvalue'] = '#name#:%s=%s' % (a, cformat_map[ret['ctype']])
603:         ret['showvalueformat'] = '%s' % (cformat_map[ret['ctype']])
604:     if isstring(var):
605:         ret['varshowvalue'] = '#name#:slen(%s)=%%d %s=\\"%%s\\"' % (a, a)
606:     ret['pydocsign'], ret['pydocsignout'] = getpydocsign(a, var)
607:     if hasnote(var):
608:         ret['note'] = var['note']
609:     return ret
610: 
611: 
612: def routsign2map(rout):
613:     '''
614:     name,NAME,begintitle,endtitle
615:     rname,ctype,rformat
616:     routdebugshowvalue
617:     '''
618:     global lcb_map
619:     name = rout['name']
620:     fname = getfortranname(rout)
621:     ret = {'name': name,
622:            'texname': name.replace('_', '\\_'),
623:            'name_lower': name.lower(),
624:            'NAME': name.upper(),
625:            'begintitle': gentitle(name),
626:            'endtitle': gentitle('end of %s' % name),
627:            'fortranname': fname,
628:            'FORTRANNAME': fname.upper(),
629:            'callstatement': getcallstatement(rout) or '',
630:            'usercode': getusercode(rout) or '',
631:            'usercode1': getusercode1(rout) or '',
632:            }
633:     if '_' in fname:
634:         ret['F_FUNC'] = 'F_FUNC_US'
635:     else:
636:         ret['F_FUNC'] = 'F_FUNC'
637:     if '_' in name:
638:         ret['F_WRAPPEDFUNC'] = 'F_WRAPPEDFUNC_US'
639:     else:
640:         ret['F_WRAPPEDFUNC'] = 'F_WRAPPEDFUNC'
641:     lcb_map = {}
642:     if 'use' in rout:
643:         for u in rout['use'].keys():
644:             if u in cb_rules.cb_map:
645:                 for un in cb_rules.cb_map[u]:
646:                     ln = un[0]
647:                     if 'map' in rout['use'][u]:
648:                         for k in rout['use'][u]['map'].keys():
649:                             if rout['use'][u]['map'][k] == un[0]:
650:                                 ln = k
651:                                 break
652:                     lcb_map[ln] = un[1]
653:     elif 'externals' in rout and rout['externals']:
654:         errmess('routsign2map: Confused: function %s has externals %s but no "use" statement.\n' % (
655:             ret['name'], repr(rout['externals'])))
656:     ret['callprotoargument'] = getcallprotoargument(rout, lcb_map) or ''
657:     if isfunction(rout):
658:         if 'result' in rout:
659:             a = rout['result']
660:         else:
661:             a = rout['name']
662:         ret['rname'] = a
663:         ret['pydocsign'], ret['pydocsignout'] = getpydocsign(a, rout)
664:         ret['ctype'] = getctype(rout['vars'][a])
665:         if hasresultnote(rout):
666:             ret['resultnote'] = rout['vars'][a]['note']
667:             rout['vars'][a]['note'] = ['See elsewhere.']
668:         if ret['ctype'] in c2buildvalue_map:
669:             ret['rformat'] = c2buildvalue_map[ret['ctype']]
670:         else:
671:             ret['rformat'] = 'O'
672:             errmess('routsign2map: no c2buildvalue key for type %s\n' %
673:                     (repr(ret['ctype'])))
674:         if debugcapi(rout):
675:             if ret['ctype'] in cformat_map:
676:                 ret['routdebugshowvalue'] = 'debug-capi:%s=%s' % (
677:                     a, cformat_map[ret['ctype']])
678:             if isstringfunction(rout):
679:                 ret['routdebugshowvalue'] = 'debug-capi:slen(%s)=%%d %s=\\"%%s\\"' % (
680:                     a, a)
681:         if isstringfunction(rout):
682:             ret['rlength'] = getstrlength(rout['vars'][a])
683:             if ret['rlength'] == '-1':
684:                 errmess('routsign2map: expected explicit specification of the length of the string returned by the fortran function %s; taking 10.\n' % (
685:                     repr(rout['name'])))
686:                 ret['rlength'] = '10'
687:     if hasnote(rout):
688:         ret['note'] = rout['note']
689:         rout['note'] = ['See elsewhere.']
690:     return ret
691: 
692: 
693: def modsign2map(m):
694:     '''
695:     modulename
696:     '''
697:     if ismodule(m):
698:         ret = {'f90modulename': m['name'],
699:                'F90MODULENAME': m['name'].upper(),
700:                'texf90modulename': m['name'].replace('_', '\\_')}
701:     else:
702:         ret = {'modulename': m['name'],
703:                'MODULENAME': m['name'].upper(),
704:                'texmodulename': m['name'].replace('_', '\\_')}
705:     ret['restdoc'] = getrestdoc(m) or []
706:     if hasnote(m):
707:         ret['note'] = m['note']
708:     ret['usercode'] = getusercode(m) or ''
709:     ret['usercode1'] = getusercode1(m) or ''
710:     if m['body']:
711:         ret['interface_usercode'] = getusercode(m['body'][0]) or ''
712:     else:
713:         ret['interface_usercode'] = ''
714:     ret['pymethoddef'] = getpymethoddef(m) or ''
715:     if 'coutput' in m:
716:         ret['coutput'] = m['coutput']
717:     if 'f2py_wrapper_output' in m:
718:         ret['f2py_wrapper_output'] = m['f2py_wrapper_output']
719:     return ret
720: 
721: 
722: def cb_sign2map(a, var, index=None):
723:     ret = {'varname': a}
724:     if index is None or 1:  # disable 7712 patch
725:         ret['varname_i'] = ret['varname']
726:     else:
727:         ret['varname_i'] = ret['varname'] + '_' + str(index)
728:     ret['ctype'] = getctype(var)
729:     if ret['ctype'] in c2capi_map:
730:         ret['atype'] = c2capi_map[ret['ctype']]
731:     if ret['ctype'] in cformat_map:
732:         ret['showvalueformat'] = '%s' % (cformat_map[ret['ctype']])
733:     if isarray(var):
734:         ret = dictappend(ret, getarrdims(a, var))
735:     ret['pydocsign'], ret['pydocsignout'] = getpydocsign(a, var)
736:     if hasnote(var):
737:         ret['note'] = var['note']
738:         var['note'] = ['See elsewhere.']
739:     return ret
740: 
741: 
742: def cb_routsign2map(rout, um):
743:     '''
744:     name,begintitle,endtitle,argname
745:     ctype,rctype,maxnofargs,nofoptargs,returncptr
746:     '''
747:     ret = {'name': 'cb_%s_in_%s' % (rout['name'], um),
748:            'returncptr': ''}
749:     if isintent_callback(rout):
750:         if '_' in rout['name']:
751:             F_FUNC = 'F_FUNC_US'
752:         else:
753:             F_FUNC = 'F_FUNC'
754:         ret['callbackname'] = '%s(%s,%s)' \
755:                               % (F_FUNC,
756:                                  rout['name'].lower(),
757:                                  rout['name'].upper(),
758:                                  )
759:         ret['static'] = 'extern'
760:     else:
761:         ret['callbackname'] = ret['name']
762:         ret['static'] = 'static'
763:     ret['argname'] = rout['name']
764:     ret['begintitle'] = gentitle(ret['name'])
765:     ret['endtitle'] = gentitle('end of %s' % ret['name'])
766:     ret['ctype'] = getctype(rout)
767:     ret['rctype'] = 'void'
768:     if ret['ctype'] == 'string':
769:         ret['rctype'] = 'void'
770:     else:
771:         ret['rctype'] = ret['ctype']
772:     if ret['rctype'] != 'void':
773:         if iscomplexfunction(rout):
774:             ret['returncptr'] = '''
775: #ifdef F2PY_CB_RETURNCOMPLEX
776: return_value=
777: #endif
778: '''
779:         else:
780:             ret['returncptr'] = 'return_value='
781:     if ret['ctype'] in cformat_map:
782:         ret['showvalueformat'] = '%s' % (cformat_map[ret['ctype']])
783:     if isstringfunction(rout):
784:         ret['strlength'] = getstrlength(rout)
785:     if isfunction(rout):
786:         if 'result' in rout:
787:             a = rout['result']
788:         else:
789:             a = rout['name']
790:         if hasnote(rout['vars'][a]):
791:             ret['note'] = rout['vars'][a]['note']
792:             rout['vars'][a]['note'] = ['See elsewhere.']
793:         ret['rname'] = a
794:         ret['pydocsign'], ret['pydocsignout'] = getpydocsign(a, rout)
795:         if iscomplexfunction(rout):
796:             ret['rctype'] = '''
797: #ifdef F2PY_CB_RETURNCOMPLEX
798: #ctype#
799: #else
800: void
801: #endif
802: '''
803:     else:
804:         if hasnote(rout):
805:             ret['note'] = rout['note']
806:             rout['note'] = ['See elsewhere.']
807:     nofargs = 0
808:     nofoptargs = 0
809:     if 'args' in rout and 'vars' in rout:
810:         for a in rout['args']:
811:             var = rout['vars'][a]
812:             if l_or(isintent_in, isintent_inout)(var):
813:                 nofargs = nofargs + 1
814:                 if isoptional(var):
815:                     nofoptargs = nofoptargs + 1
816:     ret['maxnofargs'] = repr(nofargs)
817:     ret['nofoptargs'] = repr(nofoptargs)
818:     if hasnote(rout) and isfunction(rout) and 'result' in rout:
819:         ret['routnote'] = rout['note']
820:         rout['note'] = ['See elsewhere.']
821:     return ret
822: 
823: 
824: def common_sign2map(a, var):  # obsolute
825:     ret = {'varname': a, 'ctype': getctype(var)}
826:     if isstringarray(var):
827:         ret['ctype'] = 'char'
828:     if ret['ctype'] in c2capi_map:
829:         ret['atype'] = c2capi_map[ret['ctype']]
830:     if ret['ctype'] in cformat_map:
831:         ret['showvalueformat'] = '%s' % (cformat_map[ret['ctype']])
832:     if isarray(var):
833:         ret = dictappend(ret, getarrdims(a, var))
834:     elif isstring(var):
835:         ret['size'] = getstrlength(var)
836:         ret['rank'] = '1'
837:     ret['pydocsign'], ret['pydocsignout'] = getpydocsign(a, var)
838:     if hasnote(var):
839:         ret['note'] = var['note']
840:         var['note'] = ['See elsewhere.']
841:     # for strings this returns 0-rank but actually is 1-rank
842:     ret['arrdocstr'] = getarrdocsign(a, var)
843:     return ret
844: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_69384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, (-1)), 'str', '\n\nCopyright 1999,2000 Pearu Peterson all rights reserved,\nPearu Peterson <pearu@ioc.ee>\nPermission to use, modify, and distribute this software is given under the\nterms of the NumPy License.\n\nNO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.\n$Date: 2005/05/06 10:57:33 $\nPearu Peterson\n\n')

# Assigning a Subscript to a Name (line 16):

# Assigning a Subscript to a Name (line 16):

# Obtaining the type of the subscript
int_69385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 34), 'int')
int_69386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 37), 'int')
slice_69387 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 16, 14), int_69385, int_69386, None)
str_69388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 14), 'str', '$Revision: 1.60 $')
# Obtaining the member '__getitem__' of a type (line 16)
getitem___69389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 14), str_69388, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 16)
subscript_call_result_69390 = invoke(stypy.reporting.localization.Localization(__file__, 16, 14), getitem___69389, slice_69387)

# Assigning a type to the variable '__version__' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), '__version__', subscript_call_result_69390)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from numpy.f2py import __version__' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_69391 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.f2py')

if (type(import_69391) is not StypyTypeError):

    if (import_69391 != 'pyd_module'):
        __import__(import_69391)
        sys_modules_69392 = sys.modules[import_69391]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.f2py', sys_modules_69392.module_type_store, module_type_store, ['__version__'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_69392, sys_modules_69392.module_type_store, module_type_store)
    else:
        from numpy.f2py import __version__

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.f2py', None, module_type_store, ['__version__'], [__version__])

else:
    # Assigning a type to the variable 'numpy.f2py' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.f2py', import_69391)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')


# Assigning a Attribute to a Name (line 19):

# Assigning a Attribute to a Name (line 19):
# Getting the type of '__version__' (line 19)
version___69393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), '__version__')
# Obtaining the member 'version' of a type (line 19)
version_69394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 15), version___69393, 'version')
# Assigning a type to the variable 'f2py_version' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'f2py_version', version_69394)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'import copy' statement (line 21)
import copy

import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'copy', copy, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'import re' statement (line 22)
import re

import_module(stypy.reporting.localization.Localization(__file__, 22, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'import os' statement (line 23)
import os

import_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'import sys' statement (line 24)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from numpy.f2py.crackfortran import markoutercomma' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_69395 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.f2py.crackfortran')

if (type(import_69395) is not StypyTypeError):

    if (import_69395 != 'pyd_module'):
        __import__(import_69395)
        sys_modules_69396 = sys.modules[import_69395]
        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.f2py.crackfortran', sys_modules_69396.module_type_store, module_type_store, ['markoutercomma'])
        nest_module(stypy.reporting.localization.Localization(__file__, 25, 0), __file__, sys_modules_69396, sys_modules_69396.module_type_store, module_type_store)
    else:
        from numpy.f2py.crackfortran import markoutercomma

        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.f2py.crackfortran', None, module_type_store, ['markoutercomma'], [markoutercomma])

else:
    # Assigning a type to the variable 'numpy.f2py.crackfortran' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.f2py.crackfortran', import_69395)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from numpy.f2py import cb_rules' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_69397 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.f2py')

if (type(import_69397) is not StypyTypeError):

    if (import_69397 != 'pyd_module'):
        __import__(import_69397)
        sys_modules_69398 = sys.modules[import_69397]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.f2py', sys_modules_69398.module_type_store, module_type_store, ['cb_rules'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_69398, sys_modules_69398.module_type_store, module_type_store)
    else:
        from numpy.f2py import cb_rules

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.f2py', None, module_type_store, ['cb_rules'], [cb_rules])

else:
    # Assigning a type to the variable 'numpy.f2py' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.f2py', import_69397)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 31, 0))

# 'from numpy.f2py.auxfuncs import ' statement (line 31)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_69399 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'numpy.f2py.auxfuncs')

if (type(import_69399) is not StypyTypeError):

    if (import_69399 != 'pyd_module'):
        __import__(import_69399)
        sys_modules_69400 = sys.modules[import_69399]
        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'numpy.f2py.auxfuncs', sys_modules_69400.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 31, 0), __file__, sys_modules_69400, sys_modules_69400.module_type_store, module_type_store)
    else:
        from numpy.f2py.auxfuncs import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'numpy.f2py.auxfuncs', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.f2py.auxfuncs' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'numpy.f2py.auxfuncs', import_69399)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')


# Assigning a List to a Name (line 33):

# Assigning a List to a Name (line 33):
__all__ = ['getctype', 'getstrlength', 'getarrdims', 'getpydocsign', 'getarrdocsign', 'getinit', 'sign2map', 'routsign2map', 'modsign2map', 'cb_sign2map', 'cb_routsign2map', 'common_sign2map']
module_type_store.set_exportable_members(['getctype', 'getstrlength', 'getarrdims', 'getpydocsign', 'getarrdocsign', 'getinit', 'sign2map', 'routsign2map', 'modsign2map', 'cb_sign2map', 'cb_routsign2map', 'common_sign2map'])

# Obtaining an instance of the builtin type 'list' (line 33)
list_69401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 33)
# Adding element type (line 33)
str_69402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 4), 'str', 'getctype')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 10), list_69401, str_69402)
# Adding element type (line 33)
str_69403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 16), 'str', 'getstrlength')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 10), list_69401, str_69403)
# Adding element type (line 33)
str_69404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 32), 'str', 'getarrdims')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 10), list_69401, str_69404)
# Adding element type (line 33)
str_69405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 46), 'str', 'getpydocsign')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 10), list_69401, str_69405)
# Adding element type (line 33)
str_69406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 4), 'str', 'getarrdocsign')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 10), list_69401, str_69406)
# Adding element type (line 33)
str_69407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 21), 'str', 'getinit')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 10), list_69401, str_69407)
# Adding element type (line 33)
str_69408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 32), 'str', 'sign2map')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 10), list_69401, str_69408)
# Adding element type (line 33)
str_69409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 44), 'str', 'routsign2map')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 10), list_69401, str_69409)
# Adding element type (line 33)
str_69410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 60), 'str', 'modsign2map')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 10), list_69401, str_69410)
# Adding element type (line 33)
str_69411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 4), 'str', 'cb_sign2map')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 10), list_69401, str_69411)
# Adding element type (line 33)
str_69412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 19), 'str', 'cb_routsign2map')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 10), list_69401, str_69412)
# Adding element type (line 33)
str_69413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 38), 'str', 'common_sign2map')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 10), list_69401, str_69413)

# Assigning a type to the variable '__all__' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), '__all__', list_69401)

# Assigning a Name to a Name (line 41):

# Assigning a Name to a Name (line 41):
# Getting the type of 'True' (line 41)
True_69414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'True')
# Assigning a type to the variable 'using_newcore' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'using_newcore', True_69414)

# Assigning a List to a Name (line 43):

# Assigning a List to a Name (line 43):

# Obtaining an instance of the builtin type 'list' (line 43)
list_69415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 43)

# Assigning a type to the variable 'depargs' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'depargs', list_69415)

# Assigning a Dict to a Name (line 44):

# Assigning a Dict to a Name (line 44):

# Obtaining an instance of the builtin type 'dict' (line 44)
dict_69416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 10), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 44)

# Assigning a type to the variable 'lcb_map' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'lcb_map', dict_69416)

# Assigning a Dict to a Name (line 45):

# Assigning a Dict to a Name (line 45):

# Obtaining an instance of the builtin type 'dict' (line 45)
dict_69417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 11), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 45)

# Assigning a type to the variable 'lcb2_map' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'lcb2_map', dict_69417)

# Assigning a Dict to a Name (line 48):

# Assigning a Dict to a Name (line 48):

# Obtaining an instance of the builtin type 'dict' (line 48)
dict_69418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 11), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 48)
# Adding element type (key, value) (line 48)
str_69419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 12), 'str', 'double')
str_69420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 22), 'str', 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 11), dict_69418, (str_69419, str_69420))
# Adding element type (key, value) (line 48)
str_69421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 12), 'str', 'float')
str_69422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 21), 'str', 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 11), dict_69418, (str_69421, str_69422))
# Adding element type (key, value) (line 48)
str_69423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 12), 'str', 'long_double')
str_69424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 27), 'str', 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 11), dict_69418, (str_69423, str_69424))
# Adding element type (key, value) (line 48)
str_69425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 12), 'str', 'char')
str_69426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 20), 'str', 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 11), dict_69418, (str_69425, str_69426))
# Adding element type (key, value) (line 48)
str_69427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 12), 'str', 'signed_char')
str_69428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 27), 'str', 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 11), dict_69418, (str_69427, str_69428))
# Adding element type (key, value) (line 48)
str_69429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 12), 'str', 'unsigned_char')
str_69430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 29), 'str', 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 11), dict_69418, (str_69429, str_69430))
# Adding element type (key, value) (line 48)
str_69431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 12), 'str', 'short')
str_69432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 21), 'str', 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 11), dict_69418, (str_69431, str_69432))
# Adding element type (key, value) (line 48)
str_69433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 12), 'str', 'unsigned_short')
str_69434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 30), 'str', 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 11), dict_69418, (str_69433, str_69434))
# Adding element type (key, value) (line 48)
str_69435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 12), 'str', 'int')
str_69436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 19), 'str', 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 11), dict_69418, (str_69435, str_69436))
# Adding element type (key, value) (line 48)
str_69437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 12), 'str', 'long')
str_69438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 20), 'str', 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 11), dict_69418, (str_69437, str_69438))
# Adding element type (key, value) (line 48)
str_69439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 12), 'str', 'long_long')
str_69440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 25), 'str', 'long')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 11), dict_69418, (str_69439, str_69440))
# Adding element type (key, value) (line 48)
str_69441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 12), 'str', 'unsigned')
str_69442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 24), 'str', 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 11), dict_69418, (str_69441, str_69442))
# Adding element type (key, value) (line 48)
str_69443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 12), 'str', 'complex_float')
str_69444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 29), 'str', 'complex')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 11), dict_69418, (str_69443, str_69444))
# Adding element type (key, value) (line 48)
str_69445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 12), 'str', 'complex_double')
str_69446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 30), 'str', 'complex')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 11), dict_69418, (str_69445, str_69446))
# Adding element type (key, value) (line 48)
str_69447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 12), 'str', 'complex_long_double')
str_69448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 35), 'str', 'complex')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 11), dict_69418, (str_69447, str_69448))
# Adding element type (key, value) (line 48)
str_69449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 12), 'str', 'string')
str_69450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 22), 'str', 'string')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 11), dict_69418, (str_69449, str_69450))

# Assigning a type to the variable 'c2py_map' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'c2py_map', dict_69418)

# Assigning a Dict to a Name (line 65):

# Assigning a Dict to a Name (line 65):

# Obtaining an instance of the builtin type 'dict' (line 65)
dict_69451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 65)
# Adding element type (key, value) (line 65)
str_69452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 14), 'str', 'double')
str_69453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 24), 'str', 'NPY_DOUBLE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 13), dict_69451, (str_69452, str_69453))
# Adding element type (key, value) (line 65)
str_69454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 14), 'str', 'float')
str_69455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 23), 'str', 'NPY_FLOAT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 13), dict_69451, (str_69454, str_69455))
# Adding element type (key, value) (line 65)
str_69456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 14), 'str', 'long_double')
str_69457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 29), 'str', 'NPY_DOUBLE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 13), dict_69451, (str_69456, str_69457))
# Adding element type (key, value) (line 65)
str_69458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 14), 'str', 'char')
str_69459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 22), 'str', 'NPY_CHAR')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 13), dict_69451, (str_69458, str_69459))
# Adding element type (key, value) (line 65)
str_69460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 14), 'str', 'unsigned_char')
str_69461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 31), 'str', 'NPY_UBYTE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 13), dict_69451, (str_69460, str_69461))
# Adding element type (key, value) (line 65)
str_69462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 14), 'str', 'signed_char')
str_69463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 29), 'str', 'NPY_BYTE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 13), dict_69451, (str_69462, str_69463))
# Adding element type (key, value) (line 65)
str_69464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 14), 'str', 'short')
str_69465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 23), 'str', 'NPY_SHORT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 13), dict_69451, (str_69464, str_69465))
# Adding element type (key, value) (line 65)
str_69466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 14), 'str', 'unsigned_short')
str_69467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 32), 'str', 'NPY_USHORT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 13), dict_69451, (str_69466, str_69467))
# Adding element type (key, value) (line 65)
str_69468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 14), 'str', 'int')
str_69469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 21), 'str', 'NPY_INT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 13), dict_69451, (str_69468, str_69469))
# Adding element type (key, value) (line 65)
str_69470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 14), 'str', 'unsigned')
str_69471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 26), 'str', 'NPY_UINT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 13), dict_69451, (str_69470, str_69471))
# Adding element type (key, value) (line 65)
str_69472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 14), 'str', 'long')
str_69473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 22), 'str', 'NPY_LONG')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 13), dict_69451, (str_69472, str_69473))
# Adding element type (key, value) (line 65)
str_69474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 14), 'str', 'long_long')
str_69475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 27), 'str', 'NPY_LONG')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 13), dict_69451, (str_69474, str_69475))
# Adding element type (key, value) (line 65)
str_69476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 14), 'str', 'complex_float')
str_69477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 31), 'str', 'NPY_CFLOAT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 13), dict_69451, (str_69476, str_69477))
# Adding element type (key, value) (line 65)
str_69478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 14), 'str', 'complex_double')
str_69479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 32), 'str', 'NPY_CDOUBLE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 13), dict_69451, (str_69478, str_69479))
# Adding element type (key, value) (line 65)
str_69480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 14), 'str', 'complex_long_double')
str_69481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 37), 'str', 'NPY_CDOUBLE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 13), dict_69451, (str_69480, str_69481))
# Adding element type (key, value) (line 65)
str_69482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 14), 'str', 'string')
str_69483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 24), 'str', 'NPY_CHAR')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 13), dict_69451, (str_69482, str_69483))

# Assigning a type to the variable 'c2capi_map' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'c2capi_map', dict_69451)

# Getting the type of 'using_newcore' (line 84)
using_newcore_69484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 3), 'using_newcore')
# Testing the type of an if condition (line 84)
if_condition_69485 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 0), using_newcore_69484)
# Assigning a type to the variable 'if_condition_69485' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'if_condition_69485', if_condition_69485)
# SSA begins for if statement (line 84)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Dict to a Name (line 85):

# Assigning a Dict to a Name (line 85):

# Obtaining an instance of the builtin type 'dict' (line 85)
dict_69486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 85)
# Adding element type (key, value) (line 85)
str_69487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 18), 'str', 'double')
str_69488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 28), 'str', 'NPY_DOUBLE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), dict_69486, (str_69487, str_69488))
# Adding element type (key, value) (line 85)
str_69489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 18), 'str', 'float')
str_69490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 27), 'str', 'NPY_FLOAT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), dict_69486, (str_69489, str_69490))
# Adding element type (key, value) (line 85)
str_69491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 18), 'str', 'long_double')
str_69492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 33), 'str', 'NPY_LONGDOUBLE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), dict_69486, (str_69491, str_69492))
# Adding element type (key, value) (line 85)
str_69493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 18), 'str', 'char')
str_69494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 26), 'str', 'NPY_BYTE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), dict_69486, (str_69493, str_69494))
# Adding element type (key, value) (line 85)
str_69495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 18), 'str', 'unsigned_char')
str_69496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 35), 'str', 'NPY_UBYTE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), dict_69486, (str_69495, str_69496))
# Adding element type (key, value) (line 85)
str_69497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 18), 'str', 'signed_char')
str_69498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 33), 'str', 'NPY_BYTE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), dict_69486, (str_69497, str_69498))
# Adding element type (key, value) (line 85)
str_69499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 18), 'str', 'short')
str_69500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 27), 'str', 'NPY_SHORT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), dict_69486, (str_69499, str_69500))
# Adding element type (key, value) (line 85)
str_69501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 18), 'str', 'unsigned_short')
str_69502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 36), 'str', 'NPY_USHORT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), dict_69486, (str_69501, str_69502))
# Adding element type (key, value) (line 85)
str_69503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 18), 'str', 'int')
str_69504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 25), 'str', 'NPY_INT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), dict_69486, (str_69503, str_69504))
# Adding element type (key, value) (line 85)
str_69505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 18), 'str', 'unsigned')
str_69506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 30), 'str', 'NPY_UINT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), dict_69486, (str_69505, str_69506))
# Adding element type (key, value) (line 85)
str_69507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 18), 'str', 'long')
str_69508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 26), 'str', 'NPY_LONG')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), dict_69486, (str_69507, str_69508))
# Adding element type (key, value) (line 85)
str_69509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 18), 'str', 'unsigned_long')
str_69510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 35), 'str', 'NPY_ULONG')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), dict_69486, (str_69509, str_69510))
# Adding element type (key, value) (line 85)
str_69511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 18), 'str', 'long_long')
str_69512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 31), 'str', 'NPY_LONGLONG')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), dict_69486, (str_69511, str_69512))
# Adding element type (key, value) (line 85)
str_69513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 18), 'str', 'unsigned_long_long')
str_69514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 40), 'str', 'NPY_ULONGLONG')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), dict_69486, (str_69513, str_69514))
# Adding element type (key, value) (line 85)
str_69515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 18), 'str', 'complex_float')
str_69516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 35), 'str', 'NPY_CFLOAT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), dict_69486, (str_69515, str_69516))
# Adding element type (key, value) (line 85)
str_69517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 18), 'str', 'complex_double')
str_69518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 36), 'str', 'NPY_CDOUBLE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), dict_69486, (str_69517, str_69518))
# Adding element type (key, value) (line 85)
str_69519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 18), 'str', 'complex_long_double')
str_69520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 41), 'str', 'NPY_CDOUBLE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), dict_69486, (str_69519, str_69520))
# Adding element type (key, value) (line 85)
str_69521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 18), 'str', 'string')
str_69522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 28), 'str', 'NPY_CHAR')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), dict_69486, (str_69521, str_69522))

# Assigning a type to the variable 'c2capi_map' (line 85)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'c2capi_map', dict_69486)
# SSA join for if statement (line 84)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Dict to a Name (line 108):

# Assigning a Dict to a Name (line 108):

# Obtaining an instance of the builtin type 'dict' (line 108)
dict_69523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 15), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 108)
# Adding element type (key, value) (line 108)
str_69524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 16), 'str', 'double')
str_69525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 26), 'str', 'd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), dict_69523, (str_69524, str_69525))
# Adding element type (key, value) (line 108)
str_69526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 16), 'str', 'float')
str_69527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 25), 'str', 'f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), dict_69523, (str_69526, str_69527))
# Adding element type (key, value) (line 108)
str_69528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 16), 'str', 'long_double')
str_69529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 31), 'str', 'd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), dict_69523, (str_69528, str_69529))
# Adding element type (key, value) (line 108)
str_69530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 16), 'str', 'char')
str_69531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 24), 'str', '1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), dict_69523, (str_69530, str_69531))
# Adding element type (key, value) (line 108)
str_69532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 16), 'str', 'signed_char')
str_69533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 31), 'str', '1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), dict_69523, (str_69532, str_69533))
# Adding element type (key, value) (line 108)
str_69534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 16), 'str', 'unsigned_char')
str_69535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 33), 'str', 'b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), dict_69523, (str_69534, str_69535))
# Adding element type (key, value) (line 108)
str_69536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 16), 'str', 'short')
str_69537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 25), 'str', 's')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), dict_69523, (str_69536, str_69537))
# Adding element type (key, value) (line 108)
str_69538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 16), 'str', 'unsigned_short')
str_69539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 34), 'str', 'w')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), dict_69523, (str_69538, str_69539))
# Adding element type (key, value) (line 108)
str_69540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 16), 'str', 'int')
str_69541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 23), 'str', 'i')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), dict_69523, (str_69540, str_69541))
# Adding element type (key, value) (line 108)
str_69542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 16), 'str', 'unsigned')
str_69543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 28), 'str', 'u')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), dict_69523, (str_69542, str_69543))
# Adding element type (key, value) (line 108)
str_69544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 16), 'str', 'long')
str_69545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 24), 'str', 'l')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), dict_69523, (str_69544, str_69545))
# Adding element type (key, value) (line 108)
str_69546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 16), 'str', 'long_long')
str_69547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 29), 'str', 'L')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), dict_69523, (str_69546, str_69547))
# Adding element type (key, value) (line 108)
str_69548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 16), 'str', 'complex_float')
str_69549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 33), 'str', 'F')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), dict_69523, (str_69548, str_69549))
# Adding element type (key, value) (line 108)
str_69550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 16), 'str', 'complex_double')
str_69551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 34), 'str', 'D')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), dict_69523, (str_69550, str_69551))
# Adding element type (key, value) (line 108)
str_69552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 16), 'str', 'complex_long_double')
str_69553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 39), 'str', 'D')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), dict_69523, (str_69552, str_69553))
# Adding element type (key, value) (line 108)
str_69554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 16), 'str', 'string')
str_69555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 26), 'str', 'c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), dict_69523, (str_69554, str_69555))

# Assigning a type to the variable 'c2pycode_map' (line 108)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'c2pycode_map', dict_69523)

# Getting the type of 'using_newcore' (line 125)
using_newcore_69556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 3), 'using_newcore')
# Testing the type of an if condition (line 125)
if_condition_69557 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 0), using_newcore_69556)
# Assigning a type to the variable 'if_condition_69557' (line 125)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 0), 'if_condition_69557', if_condition_69557)
# SSA begins for if statement (line 125)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Dict to a Name (line 126):

# Assigning a Dict to a Name (line 126):

# Obtaining an instance of the builtin type 'dict' (line 126)
dict_69558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 126)
# Adding element type (key, value) (line 126)
str_69559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 20), 'str', 'double')
str_69560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 30), 'str', 'd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 19), dict_69558, (str_69559, str_69560))
# Adding element type (key, value) (line 126)
str_69561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 20), 'str', 'float')
str_69562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 29), 'str', 'f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 19), dict_69558, (str_69561, str_69562))
# Adding element type (key, value) (line 126)
str_69563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 20), 'str', 'long_double')
str_69564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 35), 'str', 'g')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 19), dict_69558, (str_69563, str_69564))
# Adding element type (key, value) (line 126)
str_69565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 20), 'str', 'char')
str_69566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 28), 'str', 'b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 19), dict_69558, (str_69565, str_69566))
# Adding element type (key, value) (line 126)
str_69567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 20), 'str', 'unsigned_char')
str_69568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 37), 'str', 'B')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 19), dict_69558, (str_69567, str_69568))
# Adding element type (key, value) (line 126)
str_69569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 20), 'str', 'signed_char')
str_69570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 35), 'str', 'b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 19), dict_69558, (str_69569, str_69570))
# Adding element type (key, value) (line 126)
str_69571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 20), 'str', 'short')
str_69572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 29), 'str', 'h')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 19), dict_69558, (str_69571, str_69572))
# Adding element type (key, value) (line 126)
str_69573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 20), 'str', 'unsigned_short')
str_69574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 38), 'str', 'H')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 19), dict_69558, (str_69573, str_69574))
# Adding element type (key, value) (line 126)
str_69575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 20), 'str', 'int')
str_69576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 27), 'str', 'i')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 19), dict_69558, (str_69575, str_69576))
# Adding element type (key, value) (line 126)
str_69577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 20), 'str', 'unsigned')
str_69578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 32), 'str', 'I')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 19), dict_69558, (str_69577, str_69578))
# Adding element type (key, value) (line 126)
str_69579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 20), 'str', 'long')
str_69580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 28), 'str', 'l')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 19), dict_69558, (str_69579, str_69580))
# Adding element type (key, value) (line 126)
str_69581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 20), 'str', 'unsigned_long')
str_69582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 37), 'str', 'L')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 19), dict_69558, (str_69581, str_69582))
# Adding element type (key, value) (line 126)
str_69583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 20), 'str', 'long_long')
str_69584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 33), 'str', 'q')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 19), dict_69558, (str_69583, str_69584))
# Adding element type (key, value) (line 126)
str_69585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 20), 'str', 'unsigned_long_long')
str_69586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 42), 'str', 'Q')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 19), dict_69558, (str_69585, str_69586))
# Adding element type (key, value) (line 126)
str_69587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 20), 'str', 'complex_float')
str_69588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 37), 'str', 'F')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 19), dict_69558, (str_69587, str_69588))
# Adding element type (key, value) (line 126)
str_69589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 20), 'str', 'complex_double')
str_69590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 38), 'str', 'D')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 19), dict_69558, (str_69589, str_69590))
# Adding element type (key, value) (line 126)
str_69591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 20), 'str', 'complex_long_double')
str_69592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 43), 'str', 'G')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 19), dict_69558, (str_69591, str_69592))
# Adding element type (key, value) (line 126)
str_69593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 20), 'str', 'string')
str_69594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 30), 'str', 'S')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 19), dict_69558, (str_69593, str_69594))

# Assigning a type to the variable 'c2pycode_map' (line 126)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'c2pycode_map', dict_69558)
# SSA join for if statement (line 125)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Dict to a Name (line 144):

# Assigning a Dict to a Name (line 144):

# Obtaining an instance of the builtin type 'dict' (line 144)
dict_69595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 144)
# Adding element type (key, value) (line 144)
str_69596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 20), 'str', 'double')
str_69597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 30), 'str', 'd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 19), dict_69595, (str_69596, str_69597))
# Adding element type (key, value) (line 144)
str_69598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 20), 'str', 'float')
str_69599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 29), 'str', 'f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 19), dict_69595, (str_69598, str_69599))
# Adding element type (key, value) (line 144)
str_69600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 20), 'str', 'char')
str_69601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 28), 'str', 'b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 19), dict_69595, (str_69600, str_69601))
# Adding element type (key, value) (line 144)
str_69602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 20), 'str', 'signed_char')
str_69603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 35), 'str', 'b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 19), dict_69595, (str_69602, str_69603))
# Adding element type (key, value) (line 144)
str_69604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 20), 'str', 'short')
str_69605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 29), 'str', 'h')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 19), dict_69595, (str_69604, str_69605))
# Adding element type (key, value) (line 144)
str_69606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 20), 'str', 'int')
str_69607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 27), 'str', 'i')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 19), dict_69595, (str_69606, str_69607))
# Adding element type (key, value) (line 144)
str_69608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 20), 'str', 'long')
str_69609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 28), 'str', 'l')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 19), dict_69595, (str_69608, str_69609))
# Adding element type (key, value) (line 144)
str_69610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 20), 'str', 'long_long')
str_69611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 33), 'str', 'L')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 19), dict_69595, (str_69610, str_69611))
# Adding element type (key, value) (line 144)
str_69612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 20), 'str', 'complex_float')
str_69613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 37), 'str', 'N')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 19), dict_69595, (str_69612, str_69613))
# Adding element type (key, value) (line 144)
str_69614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 20), 'str', 'complex_double')
str_69615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 38), 'str', 'N')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 19), dict_69595, (str_69614, str_69615))
# Adding element type (key, value) (line 144)
str_69616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 20), 'str', 'complex_long_double')
str_69617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 43), 'str', 'N')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 19), dict_69595, (str_69616, str_69617))
# Adding element type (key, value) (line 144)
str_69618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 20), 'str', 'string')
str_69619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 30), 'str', 'z')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 19), dict_69595, (str_69618, str_69619))

# Assigning a type to the variable 'c2buildvalue_map' (line 144)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 0), 'c2buildvalue_map', dict_69595)



# Obtaining the type of the subscript
int_69620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 20), 'int')
# Getting the type of 'sys' (line 157)
sys_69621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 157)
version_info_69622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 3), sys_69621, 'version_info')
# Obtaining the member '__getitem__' of a type (line 157)
getitem___69623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 3), version_info_69622, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 157)
subscript_call_result_69624 = invoke(stypy.reporting.localization.Localization(__file__, 157, 3), getitem___69623, int_69620)

int_69625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 26), 'int')
# Applying the binary operator '>=' (line 157)
result_ge_69626 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 3), '>=', subscript_call_result_69624, int_69625)

# Testing the type of an if condition (line 157)
if_condition_69627 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 0), result_ge_69626)
# Assigning a type to the variable 'if_condition_69627' (line 157)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'if_condition_69627', if_condition_69627)
# SSA begins for if statement (line 157)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Str to a Subscript (line 159):

# Assigning a Str to a Subscript (line 159):
str_69628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 33), 'str', 'y')
# Getting the type of 'c2buildvalue_map' (line 159)
c2buildvalue_map_69629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'c2buildvalue_map')
str_69630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 21), 'str', 'string')
# Storing an element on a container (line 159)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 4), c2buildvalue_map_69629, (str_69630, str_69628))
# SSA join for if statement (line 157)
module_type_store = module_type_store.join_ssa_context()


# Getting the type of 'using_newcore' (line 161)
using_newcore_69631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 3), 'using_newcore')
# Testing the type of an if condition (line 161)
if_condition_69632 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 0), using_newcore_69631)
# Assigning a type to the variable 'if_condition_69632' (line 161)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'if_condition_69632', if_condition_69632)
# SSA begins for if statement (line 161)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
pass
# SSA join for if statement (line 161)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Dict to a Name (line 165):

# Assigning a Dict to a Name (line 165):

# Obtaining an instance of the builtin type 'dict' (line 165)
dict_69633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 165)
# Adding element type (key, value) (line 165)
str_69634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 14), 'str', 'real')

# Obtaining an instance of the builtin type 'dict' (line 165)
dict_69635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 165)
# Adding element type (key, value) (line 165)
str_69636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 23), 'str', '')
str_69637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 27), 'str', 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 22), dict_69635, (str_69636, str_69637))
# Adding element type (key, value) (line 165)
str_69638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 36), 'str', '4')
str_69639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 41), 'str', 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 22), dict_69635, (str_69638, str_69639))
# Adding element type (key, value) (line 165)
str_69640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 50), 'str', '8')
str_69641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 55), 'str', 'double')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 22), dict_69635, (str_69640, str_69641))
# Adding element type (key, value) (line 165)
str_69642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 23), 'str', '12')
str_69643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 29), 'str', 'long_double')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 22), dict_69635, (str_69642, str_69643))
# Adding element type (key, value) (line 165)
str_69644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 44), 'str', '16')
str_69645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 50), 'str', 'long_double')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 22), dict_69635, (str_69644, str_69645))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 13), dict_69633, (str_69634, dict_69635))
# Adding element type (key, value) (line 165)
str_69646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 14), 'str', 'integer')

# Obtaining an instance of the builtin type 'dict' (line 167)
dict_69647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 25), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 167)
# Adding element type (key, value) (line 167)
str_69648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 26), 'str', '')
str_69649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 30), 'str', 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 25), dict_69647, (str_69648, str_69649))
# Adding element type (key, value) (line 167)
str_69650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 37), 'str', '1')
str_69651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 42), 'str', 'signed_char')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 25), dict_69647, (str_69650, str_69651))
# Adding element type (key, value) (line 167)
str_69652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 57), 'str', '2')
str_69653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 62), 'str', 'short')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 25), dict_69647, (str_69652, str_69653))
# Adding element type (key, value) (line 167)
str_69654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 26), 'str', '4')
str_69655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 31), 'str', 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 25), dict_69647, (str_69654, str_69655))
# Adding element type (key, value) (line 167)
str_69656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 38), 'str', '8')
str_69657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 43), 'str', 'long_long')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 25), dict_69647, (str_69656, str_69657))
# Adding element type (key, value) (line 167)
str_69658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 26), 'str', '-1')
str_69659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 32), 'str', 'unsigned_char')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 25), dict_69647, (str_69658, str_69659))
# Adding element type (key, value) (line 167)
str_69660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 49), 'str', '-2')
str_69661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 55), 'str', 'unsigned_short')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 25), dict_69647, (str_69660, str_69661))
# Adding element type (key, value) (line 167)
str_69662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 26), 'str', '-4')
str_69663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 32), 'str', 'unsigned')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 25), dict_69647, (str_69662, str_69663))
# Adding element type (key, value) (line 167)
str_69664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 44), 'str', '-8')
str_69665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 50), 'str', 'unsigned_long_long')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 25), dict_69647, (str_69664, str_69665))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 13), dict_69633, (str_69646, dict_69647))
# Adding element type (key, value) (line 165)
str_69666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 14), 'str', 'complex')

# Obtaining an instance of the builtin type 'dict' (line 171)
dict_69667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 25), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 171)
# Adding element type (key, value) (line 171)
str_69668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 26), 'str', '')
str_69669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 30), 'str', 'complex_float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 25), dict_69667, (str_69668, str_69669))
# Adding element type (key, value) (line 171)
str_69670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 47), 'str', '8')
str_69671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 52), 'str', 'complex_float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 25), dict_69667, (str_69670, str_69671))
# Adding element type (key, value) (line 171)
str_69672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 26), 'str', '16')
str_69673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 32), 'str', 'complex_double')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 25), dict_69667, (str_69672, str_69673))
# Adding element type (key, value) (line 171)
str_69674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 50), 'str', '24')
str_69675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 56), 'str', 'complex_long_double')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 25), dict_69667, (str_69674, str_69675))
# Adding element type (key, value) (line 171)
str_69676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 26), 'str', '32')
str_69677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 32), 'str', 'complex_long_double')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 25), dict_69667, (str_69676, str_69677))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 13), dict_69633, (str_69666, dict_69667))
# Adding element type (key, value) (line 165)
str_69678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 14), 'str', 'complexkind')

# Obtaining an instance of the builtin type 'dict' (line 174)
dict_69679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 29), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 174)
# Adding element type (key, value) (line 174)
str_69680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 30), 'str', '')
str_69681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 34), 'str', 'complex_float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 29), dict_69679, (str_69680, str_69681))
# Adding element type (key, value) (line 174)
str_69682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 51), 'str', '4')
str_69683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 56), 'str', 'complex_float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 29), dict_69679, (str_69682, str_69683))
# Adding element type (key, value) (line 174)
str_69684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 30), 'str', '8')
str_69685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 35), 'str', 'complex_double')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 29), dict_69679, (str_69684, str_69685))
# Adding element type (key, value) (line 174)
str_69686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 53), 'str', '12')
str_69687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 59), 'str', 'complex_long_double')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 29), dict_69679, (str_69686, str_69687))
# Adding element type (key, value) (line 174)
str_69688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 30), 'str', '16')
str_69689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 36), 'str', 'complex_long_double')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 29), dict_69679, (str_69688, str_69689))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 13), dict_69633, (str_69678, dict_69679))
# Adding element type (key, value) (line 165)
str_69690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 14), 'str', 'logical')

# Obtaining an instance of the builtin type 'dict' (line 177)
dict_69691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 25), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 177)
# Adding element type (key, value) (line 177)
str_69692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 26), 'str', '')
str_69693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 30), 'str', 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 25), dict_69691, (str_69692, str_69693))
# Adding element type (key, value) (line 177)
str_69694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 37), 'str', '1')
str_69695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 42), 'str', 'char')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 25), dict_69691, (str_69694, str_69695))
# Adding element type (key, value) (line 177)
str_69696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 50), 'str', '2')
str_69697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 55), 'str', 'short')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 25), dict_69691, (str_69696, str_69697))
# Adding element type (key, value) (line 177)
str_69698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 64), 'str', '4')
str_69699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 69), 'str', 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 25), dict_69691, (str_69698, str_69699))
# Adding element type (key, value) (line 177)
str_69700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 26), 'str', '8')
str_69701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 31), 'str', 'long_long')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 25), dict_69691, (str_69700, str_69701))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 13), dict_69633, (str_69690, dict_69691))
# Adding element type (key, value) (line 165)
str_69702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 14), 'str', 'double complex')

# Obtaining an instance of the builtin type 'dict' (line 179)
dict_69703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 32), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 179)
# Adding element type (key, value) (line 179)
str_69704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 33), 'str', '')
str_69705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 37), 'str', 'complex_double')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 32), dict_69703, (str_69704, str_69705))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 13), dict_69633, (str_69702, dict_69703))
# Adding element type (key, value) (line 165)
str_69706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 14), 'str', 'double precision')

# Obtaining an instance of the builtin type 'dict' (line 180)
dict_69707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 34), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 180)
# Adding element type (key, value) (line 180)
str_69708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 35), 'str', '')
str_69709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 39), 'str', 'double')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 34), dict_69707, (str_69708, str_69709))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 13), dict_69633, (str_69706, dict_69707))
# Adding element type (key, value) (line 165)
str_69710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 14), 'str', 'byte')

# Obtaining an instance of the builtin type 'dict' (line 181)
dict_69711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 181)
# Adding element type (key, value) (line 181)
str_69712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 23), 'str', '')
str_69713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 27), 'str', 'char')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 22), dict_69711, (str_69712, str_69713))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 13), dict_69633, (str_69710, dict_69711))
# Adding element type (key, value) (line 165)
str_69714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 14), 'str', 'character')

# Obtaining an instance of the builtin type 'dict' (line 182)
dict_69715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 27), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 182)
# Adding element type (key, value) (line 182)
str_69716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 28), 'str', '')
str_69717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 32), 'str', 'string')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 27), dict_69715, (str_69716, str_69717))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 13), dict_69633, (str_69714, dict_69715))

# Assigning a type to the variable 'f2cmap_all' (line 165)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 0), 'f2cmap_all', dict_69633)


# Call to isfile(...): (line 185)
# Processing the call arguments (line 185)
str_69721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 18), 'str', '.f2py_f2cmap')
# Processing the call keyword arguments (line 185)
kwargs_69722 = {}
# Getting the type of 'os' (line 185)
os_69718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 3), 'os', False)
# Obtaining the member 'path' of a type (line 185)
path_69719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 3), os_69718, 'path')
# Obtaining the member 'isfile' of a type (line 185)
isfile_69720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 3), path_69719, 'isfile')
# Calling isfile(args, kwargs) (line 185)
isfile_call_result_69723 = invoke(stypy.reporting.localization.Localization(__file__, 185, 3), isfile_69720, *[str_69721], **kwargs_69722)

# Testing the type of an if condition (line 185)
if_condition_69724 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 0), isfile_call_result_69723)
# Assigning a type to the variable 'if_condition_69724' (line 185)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 0), 'if_condition_69724', if_condition_69724)
# SSA begins for if statement (line 185)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')


# SSA begins for try-except statement (line 191)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Call to outmess(...): (line 192)
# Processing the call arguments (line 192)
str_69726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 16), 'str', 'Reading .f2py_f2cmap ...\n')
# Processing the call keyword arguments (line 192)
kwargs_69727 = {}
# Getting the type of 'outmess' (line 192)
outmess_69725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'outmess', False)
# Calling outmess(args, kwargs) (line 192)
outmess_call_result_69728 = invoke(stypy.reporting.localization.Localization(__file__, 192, 8), outmess_69725, *[str_69726], **kwargs_69727)


# Assigning a Call to a Name (line 193):

# Assigning a Call to a Name (line 193):

# Call to open(...): (line 193)
# Processing the call arguments (line 193)
str_69730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 17), 'str', '.f2py_f2cmap')
str_69731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 33), 'str', 'r')
# Processing the call keyword arguments (line 193)
kwargs_69732 = {}
# Getting the type of 'open' (line 193)
open_69729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'open', False)
# Calling open(args, kwargs) (line 193)
open_call_result_69733 = invoke(stypy.reporting.localization.Localization(__file__, 193, 12), open_69729, *[str_69730, str_69731], **kwargs_69732)

# Assigning a type to the variable 'f' (line 193)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'f', open_call_result_69733)

# Assigning a Call to a Name (line 194):

# Assigning a Call to a Name (line 194):

# Call to eval(...): (line 194)
# Processing the call arguments (line 194)

# Call to read(...): (line 194)
# Processing the call keyword arguments (line 194)
kwargs_69737 = {}
# Getting the type of 'f' (line 194)
f_69735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 17), 'f', False)
# Obtaining the member 'read' of a type (line 194)
read_69736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 17), f_69735, 'read')
# Calling read(args, kwargs) (line 194)
read_call_result_69738 = invoke(stypy.reporting.localization.Localization(__file__, 194, 17), read_69736, *[], **kwargs_69737)


# Obtaining an instance of the builtin type 'dict' (line 194)
dict_69739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 27), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 194)


# Obtaining an instance of the builtin type 'dict' (line 194)
dict_69740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 31), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 194)

# Processing the call keyword arguments (line 194)
kwargs_69741 = {}
# Getting the type of 'eval' (line 194)
eval_69734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'eval', False)
# Calling eval(args, kwargs) (line 194)
eval_call_result_69742 = invoke(stypy.reporting.localization.Localization(__file__, 194, 12), eval_69734, *[read_call_result_69738, dict_69739, dict_69740], **kwargs_69741)

# Assigning a type to the variable 'd' (line 194)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'd', eval_call_result_69742)

# Call to close(...): (line 195)
# Processing the call keyword arguments (line 195)
kwargs_69745 = {}
# Getting the type of 'f' (line 195)
f_69743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'f', False)
# Obtaining the member 'close' of a type (line 195)
close_69744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 8), f_69743, 'close')
# Calling close(args, kwargs) (line 195)
close_call_result_69746 = invoke(stypy.reporting.localization.Localization(__file__, 195, 8), close_69744, *[], **kwargs_69745)



# Call to list(...): (line 196)
# Processing the call arguments (line 196)

# Call to items(...): (line 196)
# Processing the call keyword arguments (line 196)
kwargs_69750 = {}
# Getting the type of 'd' (line 196)
d_69748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 26), 'd', False)
# Obtaining the member 'items' of a type (line 196)
items_69749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 26), d_69748, 'items')
# Calling items(args, kwargs) (line 196)
items_call_result_69751 = invoke(stypy.reporting.localization.Localization(__file__, 196, 26), items_69749, *[], **kwargs_69750)

# Processing the call keyword arguments (line 196)
kwargs_69752 = {}
# Getting the type of 'list' (line 196)
list_69747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 21), 'list', False)
# Calling list(args, kwargs) (line 196)
list_call_result_69753 = invoke(stypy.reporting.localization.Localization(__file__, 196, 21), list_69747, *[items_call_result_69751], **kwargs_69752)

# Testing the type of a for loop iterable (line 196)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 196, 8), list_call_result_69753)
# Getting the type of the for loop variable (line 196)
for_loop_var_69754 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 196, 8), list_call_result_69753)
# Assigning a type to the variable 'k' (line 196)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 8), for_loop_var_69754))
# Assigning a type to the variable 'd1' (line 196)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'd1', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 8), for_loop_var_69754))
# SSA begins for a for statement (line 196)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')


# Call to list(...): (line 197)
# Processing the call arguments (line 197)

# Call to keys(...): (line 197)
# Processing the call keyword arguments (line 197)
kwargs_69758 = {}
# Getting the type of 'd1' (line 197)
d1_69756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 27), 'd1', False)
# Obtaining the member 'keys' of a type (line 197)
keys_69757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 27), d1_69756, 'keys')
# Calling keys(args, kwargs) (line 197)
keys_call_result_69759 = invoke(stypy.reporting.localization.Localization(__file__, 197, 27), keys_69757, *[], **kwargs_69758)

# Processing the call keyword arguments (line 197)
kwargs_69760 = {}
# Getting the type of 'list' (line 197)
list_69755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 22), 'list', False)
# Calling list(args, kwargs) (line 197)
list_call_result_69761 = invoke(stypy.reporting.localization.Localization(__file__, 197, 22), list_69755, *[keys_call_result_69759], **kwargs_69760)

# Testing the type of a for loop iterable (line 197)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 197, 12), list_call_result_69761)
# Getting the type of the for loop variable (line 197)
for_loop_var_69762 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 197, 12), list_call_result_69761)
# Assigning a type to the variable 'k1' (line 197)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'k1', for_loop_var_69762)
# SSA begins for a for statement (line 197)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a Subscript to a Subscript (line 198):

# Assigning a Subscript to a Subscript (line 198):

# Obtaining the type of the subscript
# Getting the type of 'k1' (line 198)
k1_69763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 36), 'k1')
# Getting the type of 'd1' (line 198)
d1_69764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 33), 'd1')
# Obtaining the member '__getitem__' of a type (line 198)
getitem___69765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 33), d1_69764, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 198)
subscript_call_result_69766 = invoke(stypy.reporting.localization.Localization(__file__, 198, 33), getitem___69765, k1_69763)

# Getting the type of 'd1' (line 198)
d1_69767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'd1')

# Call to lower(...): (line 198)
# Processing the call keyword arguments (line 198)
kwargs_69770 = {}
# Getting the type of 'k1' (line 198)
k1_69768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 19), 'k1', False)
# Obtaining the member 'lower' of a type (line 198)
lower_69769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 19), k1_69768, 'lower')
# Calling lower(args, kwargs) (line 198)
lower_call_result_69771 = invoke(stypy.reporting.localization.Localization(__file__, 198, 19), lower_69769, *[], **kwargs_69770)

# Storing an element on a container (line 198)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 16), d1_69767, (lower_call_result_69771, subscript_call_result_69766))
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# Assigning a Subscript to a Subscript (line 199):

# Assigning a Subscript to a Subscript (line 199):

# Obtaining the type of the subscript
# Getting the type of 'k' (line 199)
k_69772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 29), 'k')
# Getting the type of 'd' (line 199)
d_69773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 27), 'd')
# Obtaining the member '__getitem__' of a type (line 199)
getitem___69774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 27), d_69773, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 199)
subscript_call_result_69775 = invoke(stypy.reporting.localization.Localization(__file__, 199, 27), getitem___69774, k_69772)

# Getting the type of 'd' (line 199)
d_69776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'd')

# Call to lower(...): (line 199)
# Processing the call keyword arguments (line 199)
kwargs_69779 = {}
# Getting the type of 'k' (line 199)
k_69777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 14), 'k', False)
# Obtaining the member 'lower' of a type (line 199)
lower_69778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 14), k_69777, 'lower')
# Calling lower(args, kwargs) (line 199)
lower_call_result_69780 = invoke(stypy.reporting.localization.Localization(__file__, 199, 14), lower_69778, *[], **kwargs_69779)

# Storing an element on a container (line 199)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 12), d_69776, (lower_call_result_69780, subscript_call_result_69775))
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()



# Call to list(...): (line 200)
# Processing the call arguments (line 200)

# Call to keys(...): (line 200)
# Processing the call keyword arguments (line 200)
kwargs_69784 = {}
# Getting the type of 'd' (line 200)
d_69782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 22), 'd', False)
# Obtaining the member 'keys' of a type (line 200)
keys_69783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 22), d_69782, 'keys')
# Calling keys(args, kwargs) (line 200)
keys_call_result_69785 = invoke(stypy.reporting.localization.Localization(__file__, 200, 22), keys_69783, *[], **kwargs_69784)

# Processing the call keyword arguments (line 200)
kwargs_69786 = {}
# Getting the type of 'list' (line 200)
list_69781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 17), 'list', False)
# Calling list(args, kwargs) (line 200)
list_call_result_69787 = invoke(stypy.reporting.localization.Localization(__file__, 200, 17), list_69781, *[keys_call_result_69785], **kwargs_69786)

# Testing the type of a for loop iterable (line 200)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 200, 8), list_call_result_69787)
# Getting the type of the for loop variable (line 200)
for_loop_var_69788 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 200, 8), list_call_result_69787)
# Assigning a type to the variable 'k' (line 200)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'k', for_loop_var_69788)
# SSA begins for a for statement (line 200)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')


# Getting the type of 'k' (line 201)
k_69789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 15), 'k')
# Getting the type of 'f2cmap_all' (line 201)
f2cmap_all_69790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 24), 'f2cmap_all')
# Applying the binary operator 'notin' (line 201)
result_contains_69791 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 15), 'notin', k_69789, f2cmap_all_69790)

# Testing the type of an if condition (line 201)
if_condition_69792 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 12), result_contains_69791)
# Assigning a type to the variable 'if_condition_69792' (line 201)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'if_condition_69792', if_condition_69792)
# SSA begins for if statement (line 201)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Dict to a Subscript (line 202):

# Assigning a Dict to a Subscript (line 202):

# Obtaining an instance of the builtin type 'dict' (line 202)
dict_69793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 32), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 202)

# Getting the type of 'f2cmap_all' (line 202)
f2cmap_all_69794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'f2cmap_all')
# Getting the type of 'k' (line 202)
k_69795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 27), 'k')
# Storing an element on a container (line 202)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 16), f2cmap_all_69794, (k_69795, dict_69793))
# SSA join for if statement (line 201)
module_type_store = module_type_store.join_ssa_context()



# Call to list(...): (line 203)
# Processing the call arguments (line 203)

# Call to keys(...): (line 203)
# Processing the call keyword arguments (line 203)
kwargs_69802 = {}

# Obtaining the type of the subscript
# Getting the type of 'k' (line 203)
k_69797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 29), 'k', False)
# Getting the type of 'd' (line 203)
d_69798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 27), 'd', False)
# Obtaining the member '__getitem__' of a type (line 203)
getitem___69799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 27), d_69798, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 203)
subscript_call_result_69800 = invoke(stypy.reporting.localization.Localization(__file__, 203, 27), getitem___69799, k_69797)

# Obtaining the member 'keys' of a type (line 203)
keys_69801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 27), subscript_call_result_69800, 'keys')
# Calling keys(args, kwargs) (line 203)
keys_call_result_69803 = invoke(stypy.reporting.localization.Localization(__file__, 203, 27), keys_69801, *[], **kwargs_69802)

# Processing the call keyword arguments (line 203)
kwargs_69804 = {}
# Getting the type of 'list' (line 203)
list_69796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 22), 'list', False)
# Calling list(args, kwargs) (line 203)
list_call_result_69805 = invoke(stypy.reporting.localization.Localization(__file__, 203, 22), list_69796, *[keys_call_result_69803], **kwargs_69804)

# Testing the type of a for loop iterable (line 203)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 203, 12), list_call_result_69805)
# Getting the type of the for loop variable (line 203)
for_loop_var_69806 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 203, 12), list_call_result_69805)
# Assigning a type to the variable 'k1' (line 203)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'k1', for_loop_var_69806)
# SSA begins for a for statement (line 203)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')



# Obtaining the type of the subscript
# Getting the type of 'k1' (line 204)
k1_69807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 24), 'k1')

# Obtaining the type of the subscript
# Getting the type of 'k' (line 204)
k_69808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 21), 'k')
# Getting the type of 'd' (line 204)
d_69809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 19), 'd')
# Obtaining the member '__getitem__' of a type (line 204)
getitem___69810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 19), d_69809, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 204)
subscript_call_result_69811 = invoke(stypy.reporting.localization.Localization(__file__, 204, 19), getitem___69810, k_69808)

# Obtaining the member '__getitem__' of a type (line 204)
getitem___69812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 19), subscript_call_result_69811, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 204)
subscript_call_result_69813 = invoke(stypy.reporting.localization.Localization(__file__, 204, 19), getitem___69812, k1_69807)

# Getting the type of 'c2py_map' (line 204)
c2py_map_69814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 31), 'c2py_map')
# Applying the binary operator 'in' (line 204)
result_contains_69815 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 19), 'in', subscript_call_result_69813, c2py_map_69814)

# Testing the type of an if condition (line 204)
if_condition_69816 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 204, 16), result_contains_69815)
# Assigning a type to the variable 'if_condition_69816' (line 204)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 16), 'if_condition_69816', if_condition_69816)
# SSA begins for if statement (line 204)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')


# Getting the type of 'k1' (line 205)
k1_69817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 23), 'k1')

# Obtaining the type of the subscript
# Getting the type of 'k' (line 205)
k_69818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 40), 'k')
# Getting the type of 'f2cmap_all' (line 205)
f2cmap_all_69819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 29), 'f2cmap_all')
# Obtaining the member '__getitem__' of a type (line 205)
getitem___69820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 29), f2cmap_all_69819, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 205)
subscript_call_result_69821 = invoke(stypy.reporting.localization.Localization(__file__, 205, 29), getitem___69820, k_69818)

# Applying the binary operator 'in' (line 205)
result_contains_69822 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 23), 'in', k1_69817, subscript_call_result_69821)

# Testing the type of an if condition (line 205)
if_condition_69823 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 20), result_contains_69822)
# Assigning a type to the variable 'if_condition_69823' (line 205)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'if_condition_69823', if_condition_69823)
# SSA begins for if statement (line 205)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to outmess(...): (line 206)
# Processing the call arguments (line 206)
str_69825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 28), 'str', "\tWarning: redefinition of {'%s':{'%s':'%s'->'%s'}}\n")

# Obtaining an instance of the builtin type 'tuple' (line 207)
tuple_69826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 87), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 207)
# Adding element type (line 207)
# Getting the type of 'k' (line 207)
k_69827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 87), 'k', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 87), tuple_69826, k_69827)
# Adding element type (line 207)
# Getting the type of 'k1' (line 207)
k1_69828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 90), 'k1', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 87), tuple_69826, k1_69828)
# Adding element type (line 207)

# Obtaining the type of the subscript
# Getting the type of 'k1' (line 207)
k1_69829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 108), 'k1', False)

# Obtaining the type of the subscript
# Getting the type of 'k' (line 207)
k_69830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 105), 'k', False)
# Getting the type of 'f2cmap_all' (line 207)
f2cmap_all_69831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 94), 'f2cmap_all', False)
# Obtaining the member '__getitem__' of a type (line 207)
getitem___69832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 94), f2cmap_all_69831, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 207)
subscript_call_result_69833 = invoke(stypy.reporting.localization.Localization(__file__, 207, 94), getitem___69832, k_69830)

# Obtaining the member '__getitem__' of a type (line 207)
getitem___69834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 94), subscript_call_result_69833, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 207)
subscript_call_result_69835 = invoke(stypy.reporting.localization.Localization(__file__, 207, 94), getitem___69834, k1_69829)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 87), tuple_69826, subscript_call_result_69835)
# Adding element type (line 207)

# Obtaining the type of the subscript
# Getting the type of 'k1' (line 207)
k1_69836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 118), 'k1', False)

# Obtaining the type of the subscript
# Getting the type of 'k' (line 207)
k_69837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 115), 'k', False)
# Getting the type of 'd' (line 207)
d_69838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 113), 'd', False)
# Obtaining the member '__getitem__' of a type (line 207)
getitem___69839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 113), d_69838, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 207)
subscript_call_result_69840 = invoke(stypy.reporting.localization.Localization(__file__, 207, 113), getitem___69839, k_69837)

# Obtaining the member '__getitem__' of a type (line 207)
getitem___69841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 113), subscript_call_result_69840, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 207)
subscript_call_result_69842 = invoke(stypy.reporting.localization.Localization(__file__, 207, 113), getitem___69841, k1_69836)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 87), tuple_69826, subscript_call_result_69842)

# Applying the binary operator '%' (line 207)
result_mod_69843 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 28), '%', str_69825, tuple_69826)

# Processing the call keyword arguments (line 206)
kwargs_69844 = {}
# Getting the type of 'outmess' (line 206)
outmess_69824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 24), 'outmess', False)
# Calling outmess(args, kwargs) (line 206)
outmess_call_result_69845 = invoke(stypy.reporting.localization.Localization(__file__, 206, 24), outmess_69824, *[result_mod_69843], **kwargs_69844)

# SSA join for if statement (line 205)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Subscript to a Subscript (line 208):

# Assigning a Subscript to a Subscript (line 208):

# Obtaining the type of the subscript
# Getting the type of 'k1' (line 208)
k1_69846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 45), 'k1')

# Obtaining the type of the subscript
# Getting the type of 'k' (line 208)
k_69847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 42), 'k')
# Getting the type of 'd' (line 208)
d_69848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 40), 'd')
# Obtaining the member '__getitem__' of a type (line 208)
getitem___69849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 40), d_69848, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 208)
subscript_call_result_69850 = invoke(stypy.reporting.localization.Localization(__file__, 208, 40), getitem___69849, k_69847)

# Obtaining the member '__getitem__' of a type (line 208)
getitem___69851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 40), subscript_call_result_69850, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 208)
subscript_call_result_69852 = invoke(stypy.reporting.localization.Localization(__file__, 208, 40), getitem___69851, k1_69846)


# Obtaining the type of the subscript
# Getting the type of 'k' (line 208)
k_69853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 31), 'k')
# Getting the type of 'f2cmap_all' (line 208)
f2cmap_all_69854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 20), 'f2cmap_all')
# Obtaining the member '__getitem__' of a type (line 208)
getitem___69855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 20), f2cmap_all_69854, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 208)
subscript_call_result_69856 = invoke(stypy.reporting.localization.Localization(__file__, 208, 20), getitem___69855, k_69853)

# Getting the type of 'k1' (line 208)
k1_69857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 34), 'k1')
# Storing an element on a container (line 208)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 20), subscript_call_result_69856, (k1_69857, subscript_call_result_69852))

# Call to outmess(...): (line 209)
# Processing the call arguments (line 209)
str_69859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 28), 'str', '\tMapping "%s(kind=%s)" to "%s"\n')

# Obtaining an instance of the builtin type 'tuple' (line 210)
tuple_69860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 29), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 210)
# Adding element type (line 210)
# Getting the type of 'k' (line 210)
k_69861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 29), 'k', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 29), tuple_69860, k_69861)
# Adding element type (line 210)
# Getting the type of 'k1' (line 210)
k1_69862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 32), 'k1', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 29), tuple_69860, k1_69862)
# Adding element type (line 210)

# Obtaining the type of the subscript
# Getting the type of 'k1' (line 210)
k1_69863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 41), 'k1', False)

# Obtaining the type of the subscript
# Getting the type of 'k' (line 210)
k_69864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 38), 'k', False)
# Getting the type of 'd' (line 210)
d_69865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 36), 'd', False)
# Obtaining the member '__getitem__' of a type (line 210)
getitem___69866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 36), d_69865, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 210)
subscript_call_result_69867 = invoke(stypy.reporting.localization.Localization(__file__, 210, 36), getitem___69866, k_69864)

# Obtaining the member '__getitem__' of a type (line 210)
getitem___69868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 36), subscript_call_result_69867, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 210)
subscript_call_result_69869 = invoke(stypy.reporting.localization.Localization(__file__, 210, 36), getitem___69868, k1_69863)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 29), tuple_69860, subscript_call_result_69869)

# Applying the binary operator '%' (line 209)
result_mod_69870 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 28), '%', str_69859, tuple_69860)

# Processing the call keyword arguments (line 209)
kwargs_69871 = {}
# Getting the type of 'outmess' (line 209)
outmess_69858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 20), 'outmess', False)
# Calling outmess(args, kwargs) (line 209)
outmess_call_result_69872 = invoke(stypy.reporting.localization.Localization(__file__, 209, 20), outmess_69858, *[result_mod_69870], **kwargs_69871)

# SSA branch for the else part of an if statement (line 204)
module_type_store.open_ssa_branch('else')

# Call to errmess(...): (line 212)
# Processing the call arguments (line 212)
str_69874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 28), 'str', "\tIgnoring map {'%s':{'%s':'%s'}}: '%s' must be in %s\n")

# Obtaining an instance of the builtin type 'tuple' (line 213)
tuple_69875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 213)
# Adding element type (line 213)
# Getting the type of 'k' (line 213)
k_69876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 24), 'k', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 24), tuple_69875, k_69876)
# Adding element type (line 213)
# Getting the type of 'k1' (line 213)
k1_69877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 27), 'k1', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 24), tuple_69875, k1_69877)
# Adding element type (line 213)

# Obtaining the type of the subscript
# Getting the type of 'k1' (line 213)
k1_69878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 36), 'k1', False)

# Obtaining the type of the subscript
# Getting the type of 'k' (line 213)
k_69879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 33), 'k', False)
# Getting the type of 'd' (line 213)
d_69880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 31), 'd', False)
# Obtaining the member '__getitem__' of a type (line 213)
getitem___69881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 31), d_69880, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 213)
subscript_call_result_69882 = invoke(stypy.reporting.localization.Localization(__file__, 213, 31), getitem___69881, k_69879)

# Obtaining the member '__getitem__' of a type (line 213)
getitem___69883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 31), subscript_call_result_69882, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 213)
subscript_call_result_69884 = invoke(stypy.reporting.localization.Localization(__file__, 213, 31), getitem___69883, k1_69878)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 24), tuple_69875, subscript_call_result_69884)
# Adding element type (line 213)

# Obtaining the type of the subscript
# Getting the type of 'k1' (line 213)
k1_69885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 46), 'k1', False)

# Obtaining the type of the subscript
# Getting the type of 'k' (line 213)
k_69886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 43), 'k', False)
# Getting the type of 'd' (line 213)
d_69887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 41), 'd', False)
# Obtaining the member '__getitem__' of a type (line 213)
getitem___69888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 41), d_69887, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 213)
subscript_call_result_69889 = invoke(stypy.reporting.localization.Localization(__file__, 213, 41), getitem___69888, k_69886)

# Obtaining the member '__getitem__' of a type (line 213)
getitem___69890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 41), subscript_call_result_69889, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 213)
subscript_call_result_69891 = invoke(stypy.reporting.localization.Localization(__file__, 213, 41), getitem___69890, k1_69885)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 24), tuple_69875, subscript_call_result_69891)
# Adding element type (line 213)

# Call to list(...): (line 213)
# Processing the call arguments (line 213)

# Call to keys(...): (line 213)
# Processing the call keyword arguments (line 213)
kwargs_69895 = {}
# Getting the type of 'c2py_map' (line 213)
c2py_map_69893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 56), 'c2py_map', False)
# Obtaining the member 'keys' of a type (line 213)
keys_69894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 56), c2py_map_69893, 'keys')
# Calling keys(args, kwargs) (line 213)
keys_call_result_69896 = invoke(stypy.reporting.localization.Localization(__file__, 213, 56), keys_69894, *[], **kwargs_69895)

# Processing the call keyword arguments (line 213)
kwargs_69897 = {}
# Getting the type of 'list' (line 213)
list_69892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 51), 'list', False)
# Calling list(args, kwargs) (line 213)
list_call_result_69898 = invoke(stypy.reporting.localization.Localization(__file__, 213, 51), list_69892, *[keys_call_result_69896], **kwargs_69897)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 24), tuple_69875, list_call_result_69898)

# Applying the binary operator '%' (line 212)
result_mod_69899 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 28), '%', str_69874, tuple_69875)

# Processing the call keyword arguments (line 212)
kwargs_69900 = {}
# Getting the type of 'errmess' (line 212)
errmess_69873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 20), 'errmess', False)
# Calling errmess(args, kwargs) (line 212)
errmess_call_result_69901 = invoke(stypy.reporting.localization.Localization(__file__, 212, 20), errmess_69873, *[result_mod_69899], **kwargs_69900)

# SSA join for if statement (line 204)
module_type_store = module_type_store.join_ssa_context()

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# Call to outmess(...): (line 214)
# Processing the call arguments (line 214)
str_69903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 16), 'str', 'Succesfully applied user defined changes from .f2py_f2cmap\n')
# Processing the call keyword arguments (line 214)
kwargs_69904 = {}
# Getting the type of 'outmess' (line 214)
outmess_69902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'outmess', False)
# Calling outmess(args, kwargs) (line 214)
outmess_call_result_69905 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), outmess_69902, *[str_69903], **kwargs_69904)

# SSA branch for the except part of a try statement (line 191)
# SSA branch for the except 'Exception' branch of a try statement (line 191)
# Storing handler type
module_type_store.open_ssa_branch('except')
# Getting the type of 'Exception' (line 215)
Exception_69906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 11), 'Exception')
# Assigning a type to the variable 'msg' (line 215)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'msg', Exception_69906)

# Call to errmess(...): (line 216)
# Processing the call arguments (line 216)
str_69908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 12), 'str', 'Failed to apply user defined changes from .f2py_f2cmap: %s. Skipping.\n')
# Getting the type of 'msg' (line 217)
msg_69909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 89), 'msg', False)
# Applying the binary operator '%' (line 217)
result_mod_69910 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 12), '%', str_69908, msg_69909)

# Processing the call keyword arguments (line 216)
kwargs_69911 = {}
# Getting the type of 'errmess' (line 216)
errmess_69907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'errmess', False)
# Calling errmess(args, kwargs) (line 216)
errmess_call_result_69912 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), errmess_69907, *[result_mod_69910], **kwargs_69911)

# SSA join for try-except statement (line 191)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 185)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Dict to a Name (line 219):

# Assigning a Dict to a Name (line 219):

# Obtaining an instance of the builtin type 'dict' (line 219)
dict_69913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 219)
# Adding element type (key, value) (line 219)
str_69914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 15), 'str', 'double')
str_69915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 25), 'str', '%g')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 14), dict_69913, (str_69914, str_69915))
# Adding element type (key, value) (line 219)
str_69916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 15), 'str', 'float')
str_69917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 24), 'str', '%g')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 14), dict_69913, (str_69916, str_69917))
# Adding element type (key, value) (line 219)
str_69918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 15), 'str', 'long_double')
str_69919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 30), 'str', '%Lg')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 14), dict_69913, (str_69918, str_69919))
# Adding element type (key, value) (line 219)
str_69920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 15), 'str', 'char')
str_69921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 23), 'str', '%d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 14), dict_69913, (str_69920, str_69921))
# Adding element type (key, value) (line 219)
str_69922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 15), 'str', 'signed_char')
str_69923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 30), 'str', '%d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 14), dict_69913, (str_69922, str_69923))
# Adding element type (key, value) (line 219)
str_69924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 15), 'str', 'unsigned_char')
str_69925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 32), 'str', '%hhu')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 14), dict_69913, (str_69924, str_69925))
# Adding element type (key, value) (line 219)
str_69926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 15), 'str', 'short')
str_69927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 24), 'str', '%hd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 14), dict_69913, (str_69926, str_69927))
# Adding element type (key, value) (line 219)
str_69928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 15), 'str', 'unsigned_short')
str_69929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 33), 'str', '%hu')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 14), dict_69913, (str_69928, str_69929))
# Adding element type (key, value) (line 219)
str_69930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 15), 'str', 'int')
str_69931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 22), 'str', '%d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 14), dict_69913, (str_69930, str_69931))
# Adding element type (key, value) (line 219)
str_69932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 15), 'str', 'unsigned')
str_69933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 27), 'str', '%u')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 14), dict_69913, (str_69932, str_69933))
# Adding element type (key, value) (line 219)
str_69934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 15), 'str', 'long')
str_69935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 23), 'str', '%ld')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 14), dict_69913, (str_69934, str_69935))
# Adding element type (key, value) (line 219)
str_69936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 15), 'str', 'unsigned_long')
str_69937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 32), 'str', '%lu')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 14), dict_69913, (str_69936, str_69937))
# Adding element type (key, value) (line 219)
str_69938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 15), 'str', 'long_long')
str_69939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 28), 'str', '%ld')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 14), dict_69913, (str_69938, str_69939))
# Adding element type (key, value) (line 219)
str_69940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 15), 'str', 'complex_float')
str_69941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 32), 'str', '(%g,%g)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 14), dict_69913, (str_69940, str_69941))
# Adding element type (key, value) (line 219)
str_69942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 15), 'str', 'complex_double')
str_69943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 33), 'str', '(%g,%g)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 14), dict_69913, (str_69942, str_69943))
# Adding element type (key, value) (line 219)
str_69944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 15), 'str', 'complex_long_double')
str_69945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 38), 'str', '(%Lg,%Lg)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 14), dict_69913, (str_69944, str_69945))
# Adding element type (key, value) (line 219)
str_69946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 15), 'str', 'string')
str_69947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 25), 'str', '%s')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 14), dict_69913, (str_69946, str_69947))

# Assigning a type to the variable 'cformat_map' (line 219)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 0), 'cformat_map', dict_69913)

@norecursion
def getctype(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getctype'
    module_type_store = module_type_store.open_function_context('getctype', 241, 0, False)
    
    # Passed parameters checking function
    getctype.stypy_localization = localization
    getctype.stypy_type_of_self = None
    getctype.stypy_type_store = module_type_store
    getctype.stypy_function_name = 'getctype'
    getctype.stypy_param_names_list = ['var']
    getctype.stypy_varargs_param_name = None
    getctype.stypy_kwargs_param_name = None
    getctype.stypy_call_defaults = defaults
    getctype.stypy_call_varargs = varargs
    getctype.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getctype', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getctype', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getctype(...)' code ##################

    str_69948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, (-1)), 'str', '\n    Determines C type\n    ')
    
    # Assigning a Str to a Name (line 245):
    
    # Assigning a Str to a Name (line 245):
    str_69949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 12), 'str', 'void')
    # Assigning a type to the variable 'ctype' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'ctype', str_69949)
    
    
    # Call to isfunction(...): (line 246)
    # Processing the call arguments (line 246)
    # Getting the type of 'var' (line 246)
    var_69951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 18), 'var', False)
    # Processing the call keyword arguments (line 246)
    kwargs_69952 = {}
    # Getting the type of 'isfunction' (line 246)
    isfunction_69950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 7), 'isfunction', False)
    # Calling isfunction(args, kwargs) (line 246)
    isfunction_call_result_69953 = invoke(stypy.reporting.localization.Localization(__file__, 246, 7), isfunction_69950, *[var_69951], **kwargs_69952)
    
    # Testing the type of an if condition (line 246)
    if_condition_69954 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 4), isfunction_call_result_69953)
    # Assigning a type to the variable 'if_condition_69954' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'if_condition_69954', if_condition_69954)
    # SSA begins for if statement (line 246)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    str_69955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 11), 'str', 'result')
    # Getting the type of 'var' (line 247)
    var_69956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 23), 'var')
    # Applying the binary operator 'in' (line 247)
    result_contains_69957 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 11), 'in', str_69955, var_69956)
    
    # Testing the type of an if condition (line 247)
    if_condition_69958 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 247, 8), result_contains_69957)
    # Assigning a type to the variable 'if_condition_69958' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'if_condition_69958', if_condition_69958)
    # SSA begins for if statement (line 247)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 248):
    
    # Assigning a Subscript to a Name (line 248):
    
    # Obtaining the type of the subscript
    str_69959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 20), 'str', 'result')
    # Getting the type of 'var' (line 248)
    var_69960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'var')
    # Obtaining the member '__getitem__' of a type (line 248)
    getitem___69961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 16), var_69960, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 248)
    subscript_call_result_69962 = invoke(stypy.reporting.localization.Localization(__file__, 248, 16), getitem___69961, str_69959)
    
    # Assigning a type to the variable 'a' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'a', subscript_call_result_69962)
    # SSA branch for the else part of an if statement (line 247)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 250):
    
    # Assigning a Subscript to a Name (line 250):
    
    # Obtaining the type of the subscript
    str_69963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 20), 'str', 'name')
    # Getting the type of 'var' (line 250)
    var_69964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'var')
    # Obtaining the member '__getitem__' of a type (line 250)
    getitem___69965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 16), var_69964, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 250)
    subscript_call_result_69966 = invoke(stypy.reporting.localization.Localization(__file__, 250, 16), getitem___69965, str_69963)
    
    # Assigning a type to the variable 'a' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'a', subscript_call_result_69966)
    # SSA join for if statement (line 247)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'a' (line 251)
    a_69967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 11), 'a')
    
    # Obtaining the type of the subscript
    str_69968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 20), 'str', 'vars')
    # Getting the type of 'var' (line 251)
    var_69969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 16), 'var')
    # Obtaining the member '__getitem__' of a type (line 251)
    getitem___69970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 16), var_69969, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 251)
    subscript_call_result_69971 = invoke(stypy.reporting.localization.Localization(__file__, 251, 16), getitem___69970, str_69968)
    
    # Applying the binary operator 'in' (line 251)
    result_contains_69972 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 11), 'in', a_69967, subscript_call_result_69971)
    
    # Testing the type of an if condition (line 251)
    if_condition_69973 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 251, 8), result_contains_69972)
    # Assigning a type to the variable 'if_condition_69973' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'if_condition_69973', if_condition_69973)
    # SSA begins for if statement (line 251)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to getctype(...): (line 252)
    # Processing the call arguments (line 252)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 252)
    a_69975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 40), 'a', False)
    
    # Obtaining the type of the subscript
    str_69976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 32), 'str', 'vars')
    # Getting the type of 'var' (line 252)
    var_69977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 28), 'var', False)
    # Obtaining the member '__getitem__' of a type (line 252)
    getitem___69978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 28), var_69977, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 252)
    subscript_call_result_69979 = invoke(stypy.reporting.localization.Localization(__file__, 252, 28), getitem___69978, str_69976)
    
    # Obtaining the member '__getitem__' of a type (line 252)
    getitem___69980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 28), subscript_call_result_69979, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 252)
    subscript_call_result_69981 = invoke(stypy.reporting.localization.Localization(__file__, 252, 28), getitem___69980, a_69975)
    
    # Processing the call keyword arguments (line 252)
    kwargs_69982 = {}
    # Getting the type of 'getctype' (line 252)
    getctype_69974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 19), 'getctype', False)
    # Calling getctype(args, kwargs) (line 252)
    getctype_call_result_69983 = invoke(stypy.reporting.localization.Localization(__file__, 252, 19), getctype_69974, *[subscript_call_result_69981], **kwargs_69982)
    
    # Assigning a type to the variable 'stypy_return_type' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'stypy_return_type', getctype_call_result_69983)
    # SSA branch for the else part of an if statement (line 251)
    module_type_store.open_ssa_branch('else')
    
    # Call to errmess(...): (line 254)
    # Processing the call arguments (line 254)
    str_69985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 20), 'str', 'getctype: function %s has no return value?!\n')
    # Getting the type of 'a' (line 254)
    a_69986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 70), 'a', False)
    # Applying the binary operator '%' (line 254)
    result_mod_69987 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 20), '%', str_69985, a_69986)
    
    # Processing the call keyword arguments (line 254)
    kwargs_69988 = {}
    # Getting the type of 'errmess' (line 254)
    errmess_69984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'errmess', False)
    # Calling errmess(args, kwargs) (line 254)
    errmess_call_result_69989 = invoke(stypy.reporting.localization.Localization(__file__, 254, 12), errmess_69984, *[result_mod_69987], **kwargs_69988)
    
    # SSA join for if statement (line 251)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 246)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to issubroutine(...): (line 255)
    # Processing the call arguments (line 255)
    # Getting the type of 'var' (line 255)
    var_69991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 22), 'var', False)
    # Processing the call keyword arguments (line 255)
    kwargs_69992 = {}
    # Getting the type of 'issubroutine' (line 255)
    issubroutine_69990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 9), 'issubroutine', False)
    # Calling issubroutine(args, kwargs) (line 255)
    issubroutine_call_result_69993 = invoke(stypy.reporting.localization.Localization(__file__, 255, 9), issubroutine_69990, *[var_69991], **kwargs_69992)
    
    # Testing the type of an if condition (line 255)
    if_condition_69994 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 9), issubroutine_call_result_69993)
    # Assigning a type to the variable 'if_condition_69994' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 9), 'if_condition_69994', if_condition_69994)
    # SSA begins for if statement (line 255)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'ctype' (line 256)
    ctype_69995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 15), 'ctype')
    # Assigning a type to the variable 'stypy_return_type' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'stypy_return_type', ctype_69995)
    # SSA branch for the else part of an if statement (line 255)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    str_69996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 9), 'str', 'typespec')
    # Getting the type of 'var' (line 257)
    var_69997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 23), 'var')
    # Applying the binary operator 'in' (line 257)
    result_contains_69998 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 9), 'in', str_69996, var_69997)
    
    
    
    # Call to lower(...): (line 257)
    # Processing the call keyword arguments (line 257)
    kwargs_70004 = {}
    
    # Obtaining the type of the subscript
    str_69999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 35), 'str', 'typespec')
    # Getting the type of 'var' (line 257)
    var_70000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 31), 'var', False)
    # Obtaining the member '__getitem__' of a type (line 257)
    getitem___70001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 31), var_70000, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 257)
    subscript_call_result_70002 = invoke(stypy.reporting.localization.Localization(__file__, 257, 31), getitem___70001, str_69999)
    
    # Obtaining the member 'lower' of a type (line 257)
    lower_70003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 31), subscript_call_result_70002, 'lower')
    # Calling lower(args, kwargs) (line 257)
    lower_call_result_70005 = invoke(stypy.reporting.localization.Localization(__file__, 257, 31), lower_70003, *[], **kwargs_70004)
    
    # Getting the type of 'f2cmap_all' (line 257)
    f2cmap_all_70006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 58), 'f2cmap_all')
    # Applying the binary operator 'in' (line 257)
    result_contains_70007 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 31), 'in', lower_call_result_70005, f2cmap_all_70006)
    
    # Applying the binary operator 'and' (line 257)
    result_and_keyword_70008 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 9), 'and', result_contains_69998, result_contains_70007)
    
    # Testing the type of an if condition (line 257)
    if_condition_70009 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 257, 9), result_and_keyword_70008)
    # Assigning a type to the variable 'if_condition_70009' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 9), 'if_condition_70009', if_condition_70009)
    # SSA begins for if statement (line 257)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 258):
    
    # Assigning a Call to a Name (line 258):
    
    # Call to lower(...): (line 258)
    # Processing the call keyword arguments (line 258)
    kwargs_70015 = {}
    
    # Obtaining the type of the subscript
    str_70010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 23), 'str', 'typespec')
    # Getting the type of 'var' (line 258)
    var_70011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 19), 'var', False)
    # Obtaining the member '__getitem__' of a type (line 258)
    getitem___70012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 19), var_70011, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 258)
    subscript_call_result_70013 = invoke(stypy.reporting.localization.Localization(__file__, 258, 19), getitem___70012, str_70010)
    
    # Obtaining the member 'lower' of a type (line 258)
    lower_70014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 19), subscript_call_result_70013, 'lower')
    # Calling lower(args, kwargs) (line 258)
    lower_call_result_70016 = invoke(stypy.reporting.localization.Localization(__file__, 258, 19), lower_70014, *[], **kwargs_70015)
    
    # Assigning a type to the variable 'typespec' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'typespec', lower_call_result_70016)
    
    # Assigning a Subscript to a Name (line 259):
    
    # Assigning a Subscript to a Name (line 259):
    
    # Obtaining the type of the subscript
    # Getting the type of 'typespec' (line 259)
    typespec_70017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'typespec')
    # Getting the type of 'f2cmap_all' (line 259)
    f2cmap_all_70018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 17), 'f2cmap_all')
    # Obtaining the member '__getitem__' of a type (line 259)
    getitem___70019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 17), f2cmap_all_70018, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 259)
    subscript_call_result_70020 = invoke(stypy.reporting.localization.Localization(__file__, 259, 17), getitem___70019, typespec_70017)
    
    # Assigning a type to the variable 'f2cmap' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'f2cmap', subscript_call_result_70020)
    
    # Assigning a Subscript to a Name (line 260):
    
    # Assigning a Subscript to a Name (line 260):
    
    # Obtaining the type of the subscript
    str_70021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 23), 'str', '')
    # Getting the type of 'f2cmap' (line 260)
    f2cmap_70022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 16), 'f2cmap')
    # Obtaining the member '__getitem__' of a type (line 260)
    getitem___70023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 16), f2cmap_70022, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 260)
    subscript_call_result_70024 = invoke(stypy.reporting.localization.Localization(__file__, 260, 16), getitem___70023, str_70021)
    
    # Assigning a type to the variable 'ctype' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'ctype', subscript_call_result_70024)
    
    
    str_70025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 11), 'str', 'kindselector')
    # Getting the type of 'var' (line 261)
    var_70026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 29), 'var')
    # Applying the binary operator 'in' (line 261)
    result_contains_70027 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 11), 'in', str_70025, var_70026)
    
    # Testing the type of an if condition (line 261)
    if_condition_70028 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 261, 8), result_contains_70027)
    # Assigning a type to the variable 'if_condition_70028' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'if_condition_70028', if_condition_70028)
    # SSA begins for if statement (line 261)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    str_70029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 15), 'str', '*')
    
    # Obtaining the type of the subscript
    str_70030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 26), 'str', 'kindselector')
    # Getting the type of 'var' (line 262)
    var_70031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 22), 'var')
    # Obtaining the member '__getitem__' of a type (line 262)
    getitem___70032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 22), var_70031, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 262)
    subscript_call_result_70033 = invoke(stypy.reporting.localization.Localization(__file__, 262, 22), getitem___70032, str_70030)
    
    # Applying the binary operator 'in' (line 262)
    result_contains_70034 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 15), 'in', str_70029, subscript_call_result_70033)
    
    # Testing the type of an if condition (line 262)
    if_condition_70035 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 262, 12), result_contains_70034)
    # Assigning a type to the variable 'if_condition_70035' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'if_condition_70035', if_condition_70035)
    # SSA begins for if statement (line 262)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 263)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 264):
    
    # Assigning a Subscript to a Name (line 264):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    str_70036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 55), 'str', '*')
    
    # Obtaining the type of the subscript
    str_70037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 39), 'str', 'kindselector')
    # Getting the type of 'var' (line 264)
    var_70038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 35), 'var')
    # Obtaining the member '__getitem__' of a type (line 264)
    getitem___70039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 35), var_70038, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 264)
    subscript_call_result_70040 = invoke(stypy.reporting.localization.Localization(__file__, 264, 35), getitem___70039, str_70037)
    
    # Obtaining the member '__getitem__' of a type (line 264)
    getitem___70041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 35), subscript_call_result_70040, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 264)
    subscript_call_result_70042 = invoke(stypy.reporting.localization.Localization(__file__, 264, 35), getitem___70041, str_70036)
    
    # Getting the type of 'f2cmap' (line 264)
    f2cmap_70043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 28), 'f2cmap')
    # Obtaining the member '__getitem__' of a type (line 264)
    getitem___70044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 28), f2cmap_70043, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 264)
    subscript_call_result_70045 = invoke(stypy.reporting.localization.Localization(__file__, 264, 28), getitem___70044, subscript_call_result_70042)
    
    # Assigning a type to the variable 'ctype' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 20), 'ctype', subscript_call_result_70045)
    # SSA branch for the except part of a try statement (line 263)
    # SSA branch for the except 'KeyError' branch of a try statement (line 263)
    module_type_store.open_ssa_branch('except')
    
    # Call to errmess(...): (line 266)
    # Processing the call arguments (line 266)
    str_70047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 28), 'str', 'getctype: "%s %s %s" not supported.\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 267)
    tuple_70048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 267)
    # Adding element type (line 267)
    
    # Obtaining the type of the subscript
    str_70049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 33), 'str', 'typespec')
    # Getting the type of 'var' (line 267)
    var_70050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 29), 'var', False)
    # Obtaining the member '__getitem__' of a type (line 267)
    getitem___70051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 29), var_70050, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 267)
    subscript_call_result_70052 = invoke(stypy.reporting.localization.Localization(__file__, 267, 29), getitem___70051, str_70049)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 29), tuple_70048, subscript_call_result_70052)
    # Adding element type (line 267)
    str_70053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 46), 'str', '*')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 29), tuple_70048, str_70053)
    # Adding element type (line 267)
    
    # Obtaining the type of the subscript
    str_70054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 71), 'str', '*')
    
    # Obtaining the type of the subscript
    str_70055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 55), 'str', 'kindselector')
    # Getting the type of 'var' (line 267)
    var_70056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 51), 'var', False)
    # Obtaining the member '__getitem__' of a type (line 267)
    getitem___70057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 51), var_70056, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 267)
    subscript_call_result_70058 = invoke(stypy.reporting.localization.Localization(__file__, 267, 51), getitem___70057, str_70055)
    
    # Obtaining the member '__getitem__' of a type (line 267)
    getitem___70059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 51), subscript_call_result_70058, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 267)
    subscript_call_result_70060 = invoke(stypy.reporting.localization.Localization(__file__, 267, 51), getitem___70059, str_70054)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 29), tuple_70048, subscript_call_result_70060)
    
    # Applying the binary operator '%' (line 266)
    result_mod_70061 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 28), '%', str_70047, tuple_70048)
    
    # Processing the call keyword arguments (line 266)
    kwargs_70062 = {}
    # Getting the type of 'errmess' (line 266)
    errmess_70046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 20), 'errmess', False)
    # Calling errmess(args, kwargs) (line 266)
    errmess_call_result_70063 = invoke(stypy.reporting.localization.Localization(__file__, 266, 20), errmess_70046, *[result_mod_70061], **kwargs_70062)
    
    # SSA join for try-except statement (line 263)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 262)
    module_type_store.open_ssa_branch('else')
    
    
    str_70064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 17), 'str', 'kind')
    
    # Obtaining the type of the subscript
    str_70065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 31), 'str', 'kindselector')
    # Getting the type of 'var' (line 268)
    var_70066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 27), 'var')
    # Obtaining the member '__getitem__' of a type (line 268)
    getitem___70067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 27), var_70066, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 268)
    subscript_call_result_70068 = invoke(stypy.reporting.localization.Localization(__file__, 268, 27), getitem___70067, str_70065)
    
    # Applying the binary operator 'in' (line 268)
    result_contains_70069 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 17), 'in', str_70064, subscript_call_result_70068)
    
    # Testing the type of an if condition (line 268)
    if_condition_70070 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 268, 17), result_contains_70069)
    # Assigning a type to the variable 'if_condition_70070' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 17), 'if_condition_70070', if_condition_70070)
    # SSA begins for if statement (line 268)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'typespec' (line 269)
    typespec_70071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 19), 'typespec')
    str_70072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 30), 'str', 'kind')
    # Applying the binary operator '+' (line 269)
    result_add_70073 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 19), '+', typespec_70071, str_70072)
    
    # Getting the type of 'f2cmap_all' (line 269)
    f2cmap_all_70074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 40), 'f2cmap_all')
    # Applying the binary operator 'in' (line 269)
    result_contains_70075 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 19), 'in', result_add_70073, f2cmap_all_70074)
    
    # Testing the type of an if condition (line 269)
    if_condition_70076 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 269, 16), result_contains_70075)
    # Assigning a type to the variable 'if_condition_70076' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 16), 'if_condition_70076', if_condition_70076)
    # SSA begins for if statement (line 269)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 270):
    
    # Assigning a Subscript to a Name (line 270):
    
    # Obtaining the type of the subscript
    # Getting the type of 'typespec' (line 270)
    typespec_70077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 40), 'typespec')
    str_70078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 51), 'str', 'kind')
    # Applying the binary operator '+' (line 270)
    result_add_70079 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 40), '+', typespec_70077, str_70078)
    
    # Getting the type of 'f2cmap_all' (line 270)
    f2cmap_all_70080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 29), 'f2cmap_all')
    # Obtaining the member '__getitem__' of a type (line 270)
    getitem___70081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 29), f2cmap_all_70080, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 270)
    subscript_call_result_70082 = invoke(stypy.reporting.localization.Localization(__file__, 270, 29), getitem___70081, result_add_70079)
    
    # Assigning a type to the variable 'f2cmap' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 20), 'f2cmap', subscript_call_result_70082)
    # SSA join for if statement (line 269)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 271)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 272):
    
    # Assigning a Subscript to a Name (line 272):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    str_70083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 55), 'str', 'kind')
    
    # Obtaining the type of the subscript
    str_70084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 39), 'str', 'kindselector')
    # Getting the type of 'var' (line 272)
    var_70085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 35), 'var')
    # Obtaining the member '__getitem__' of a type (line 272)
    getitem___70086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 35), var_70085, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 272)
    subscript_call_result_70087 = invoke(stypy.reporting.localization.Localization(__file__, 272, 35), getitem___70086, str_70084)
    
    # Obtaining the member '__getitem__' of a type (line 272)
    getitem___70088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 35), subscript_call_result_70087, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 272)
    subscript_call_result_70089 = invoke(stypy.reporting.localization.Localization(__file__, 272, 35), getitem___70088, str_70083)
    
    # Getting the type of 'f2cmap' (line 272)
    f2cmap_70090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 28), 'f2cmap')
    # Obtaining the member '__getitem__' of a type (line 272)
    getitem___70091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 28), f2cmap_70090, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 272)
    subscript_call_result_70092 = invoke(stypy.reporting.localization.Localization(__file__, 272, 28), getitem___70091, subscript_call_result_70089)
    
    # Assigning a type to the variable 'ctype' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 20), 'ctype', subscript_call_result_70092)
    # SSA branch for the except part of a try statement (line 271)
    # SSA branch for the except 'KeyError' branch of a try statement (line 271)
    module_type_store.open_ssa_branch('except')
    
    
    # Getting the type of 'typespec' (line 274)
    typespec_70093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 23), 'typespec')
    # Getting the type of 'f2cmap_all' (line 274)
    f2cmap_all_70094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 35), 'f2cmap_all')
    # Applying the binary operator 'in' (line 274)
    result_contains_70095 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 23), 'in', typespec_70093, f2cmap_all_70094)
    
    # Testing the type of an if condition (line 274)
    if_condition_70096 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 274, 20), result_contains_70095)
    # Assigning a type to the variable 'if_condition_70096' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 20), 'if_condition_70096', if_condition_70096)
    # SSA begins for if statement (line 274)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 275):
    
    # Assigning a Subscript to a Name (line 275):
    
    # Obtaining the type of the subscript
    # Getting the type of 'typespec' (line 275)
    typespec_70097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 44), 'typespec')
    # Getting the type of 'f2cmap_all' (line 275)
    f2cmap_all_70098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 33), 'f2cmap_all')
    # Obtaining the member '__getitem__' of a type (line 275)
    getitem___70099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 33), f2cmap_all_70098, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 275)
    subscript_call_result_70100 = invoke(stypy.reporting.localization.Localization(__file__, 275, 33), getitem___70099, typespec_70097)
    
    # Assigning a type to the variable 'f2cmap' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 24), 'f2cmap', subscript_call_result_70100)
    # SSA join for if statement (line 274)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 276)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 277):
    
    # Assigning a Subscript to a Name (line 277):
    
    # Obtaining the type of the subscript
    
    # Call to str(...): (line 277)
    # Processing the call arguments (line 277)
    
    # Obtaining the type of the subscript
    str_70102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 63), 'str', 'kind')
    
    # Obtaining the type of the subscript
    str_70103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 47), 'str', 'kindselector')
    # Getting the type of 'var' (line 277)
    var_70104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 43), 'var', False)
    # Obtaining the member '__getitem__' of a type (line 277)
    getitem___70105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 43), var_70104, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 277)
    subscript_call_result_70106 = invoke(stypy.reporting.localization.Localization(__file__, 277, 43), getitem___70105, str_70103)
    
    # Obtaining the member '__getitem__' of a type (line 277)
    getitem___70107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 43), subscript_call_result_70106, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 277)
    subscript_call_result_70108 = invoke(stypy.reporting.localization.Localization(__file__, 277, 43), getitem___70107, str_70102)
    
    # Processing the call keyword arguments (line 277)
    kwargs_70109 = {}
    # Getting the type of 'str' (line 277)
    str_70101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 39), 'str', False)
    # Calling str(args, kwargs) (line 277)
    str_call_result_70110 = invoke(stypy.reporting.localization.Localization(__file__, 277, 39), str_70101, *[subscript_call_result_70108], **kwargs_70109)
    
    # Getting the type of 'f2cmap' (line 277)
    f2cmap_70111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 32), 'f2cmap')
    # Obtaining the member '__getitem__' of a type (line 277)
    getitem___70112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 32), f2cmap_70111, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 277)
    subscript_call_result_70113 = invoke(stypy.reporting.localization.Localization(__file__, 277, 32), getitem___70112, str_call_result_70110)
    
    # Assigning a type to the variable 'ctype' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 24), 'ctype', subscript_call_result_70113)
    # SSA branch for the except part of a try statement (line 276)
    # SSA branch for the except 'KeyError' branch of a try statement (line 276)
    module_type_store.open_ssa_branch('except')
    
    # Call to errmess(...): (line 279)
    # Processing the call arguments (line 279)
    str_70115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 32), 'str', 'getctype: "%s(kind=%s)" is mapped to C "%s" (to override define dict(%s = dict(%s="<C typespec>")) in %s/.f2py_f2cmap file).\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 280)
    tuple_70116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 280)
    # Adding element type (line 280)
    # Getting the type of 'typespec' (line 280)
    typespec_70117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 35), 'typespec', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 35), tuple_70116, typespec_70117)
    # Adding element type (line 280)
    
    # Obtaining the type of the subscript
    str_70118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 65), 'str', 'kind')
    
    # Obtaining the type of the subscript
    str_70119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 49), 'str', 'kindselector')
    # Getting the type of 'var' (line 280)
    var_70120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 45), 'var', False)
    # Obtaining the member '__getitem__' of a type (line 280)
    getitem___70121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 45), var_70120, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 280)
    subscript_call_result_70122 = invoke(stypy.reporting.localization.Localization(__file__, 280, 45), getitem___70121, str_70119)
    
    # Obtaining the member '__getitem__' of a type (line 280)
    getitem___70123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 45), subscript_call_result_70122, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 280)
    subscript_call_result_70124 = invoke(stypy.reporting.localization.Localization(__file__, 280, 45), getitem___70123, str_70118)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 35), tuple_70116, subscript_call_result_70124)
    # Adding element type (line 280)
    # Getting the type of 'ctype' (line 280)
    ctype_70125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 74), 'ctype', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 35), tuple_70116, ctype_70125)
    # Adding element type (line 280)
    # Getting the type of 'typespec' (line 281)
    typespec_70126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 35), 'typespec', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 35), tuple_70116, typespec_70126)
    # Adding element type (line 280)
    
    # Obtaining the type of the subscript
    str_70127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 65), 'str', 'kind')
    
    # Obtaining the type of the subscript
    str_70128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 49), 'str', 'kindselector')
    # Getting the type of 'var' (line 281)
    var_70129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 45), 'var', False)
    # Obtaining the member '__getitem__' of a type (line 281)
    getitem___70130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 45), var_70129, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 281)
    subscript_call_result_70131 = invoke(stypy.reporting.localization.Localization(__file__, 281, 45), getitem___70130, str_70128)
    
    # Obtaining the member '__getitem__' of a type (line 281)
    getitem___70132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 45), subscript_call_result_70131, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 281)
    subscript_call_result_70133 = invoke(stypy.reporting.localization.Localization(__file__, 281, 45), getitem___70132, str_70127)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 35), tuple_70116, subscript_call_result_70133)
    # Adding element type (line 280)
    
    # Call to getcwd(...): (line 281)
    # Processing the call keyword arguments (line 281)
    kwargs_70136 = {}
    # Getting the type of 'os' (line 281)
    os_70134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 74), 'os', False)
    # Obtaining the member 'getcwd' of a type (line 281)
    getcwd_70135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 74), os_70134, 'getcwd')
    # Calling getcwd(args, kwargs) (line 281)
    getcwd_call_result_70137 = invoke(stypy.reporting.localization.Localization(__file__, 281, 74), getcwd_70135, *[], **kwargs_70136)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 35), tuple_70116, getcwd_call_result_70137)
    
    # Applying the binary operator '%' (line 279)
    result_mod_70138 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 32), '%', str_70115, tuple_70116)
    
    # Processing the call keyword arguments (line 279)
    kwargs_70139 = {}
    # Getting the type of 'errmess' (line 279)
    errmess_70114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 24), 'errmess', False)
    # Calling errmess(args, kwargs) (line 279)
    errmess_call_result_70140 = invoke(stypy.reporting.localization.Localization(__file__, 279, 24), errmess_70114, *[result_mod_70138], **kwargs_70139)
    
    # SSA join for try-except statement (line 276)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 271)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 268)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 262)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 261)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 257)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to isexternal(...): (line 284)
    # Processing the call arguments (line 284)
    # Getting the type of 'var' (line 284)
    var_70142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 26), 'var', False)
    # Processing the call keyword arguments (line 284)
    kwargs_70143 = {}
    # Getting the type of 'isexternal' (line 284)
    isexternal_70141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 15), 'isexternal', False)
    # Calling isexternal(args, kwargs) (line 284)
    isexternal_call_result_70144 = invoke(stypy.reporting.localization.Localization(__file__, 284, 15), isexternal_70141, *[var_70142], **kwargs_70143)
    
    # Applying the 'not' unary operator (line 284)
    result_not__70145 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 11), 'not', isexternal_call_result_70144)
    
    # Testing the type of an if condition (line 284)
    if_condition_70146 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 8), result_not__70145)
    # Assigning a type to the variable 'if_condition_70146' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'if_condition_70146', if_condition_70146)
    # SSA begins for if statement (line 284)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to errmess(...): (line 285)
    # Processing the call arguments (line 285)
    str_70148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 16), 'str', 'getctype: No C-type found in "%s", assuming void.\n')
    # Getting the type of 'var' (line 286)
    var_70149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 72), 'var', False)
    # Applying the binary operator '%' (line 286)
    result_mod_70150 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 16), '%', str_70148, var_70149)
    
    # Processing the call keyword arguments (line 285)
    kwargs_70151 = {}
    # Getting the type of 'errmess' (line 285)
    errmess_70147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'errmess', False)
    # Calling errmess(args, kwargs) (line 285)
    errmess_call_result_70152 = invoke(stypy.reporting.localization.Localization(__file__, 285, 12), errmess_70147, *[result_mod_70150], **kwargs_70151)
    
    # SSA join for if statement (line 284)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 257)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 255)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 246)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ctype' (line 287)
    ctype_70153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 11), 'ctype')
    # Assigning a type to the variable 'stypy_return_type' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'stypy_return_type', ctype_70153)
    
    # ################# End of 'getctype(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getctype' in the type store
    # Getting the type of 'stypy_return_type' (line 241)
    stypy_return_type_70154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_70154)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getctype'
    return stypy_return_type_70154

# Assigning a type to the variable 'getctype' (line 241)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 0), 'getctype', getctype)

@norecursion
def getstrlength(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getstrlength'
    module_type_store = module_type_store.open_function_context('getstrlength', 290, 0, False)
    
    # Passed parameters checking function
    getstrlength.stypy_localization = localization
    getstrlength.stypy_type_of_self = None
    getstrlength.stypy_type_store = module_type_store
    getstrlength.stypy_function_name = 'getstrlength'
    getstrlength.stypy_param_names_list = ['var']
    getstrlength.stypy_varargs_param_name = None
    getstrlength.stypy_kwargs_param_name = None
    getstrlength.stypy_call_defaults = defaults
    getstrlength.stypy_call_varargs = varargs
    getstrlength.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getstrlength', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getstrlength', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getstrlength(...)' code ##################

    
    
    # Call to isstringfunction(...): (line 291)
    # Processing the call arguments (line 291)
    # Getting the type of 'var' (line 291)
    var_70156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 24), 'var', False)
    # Processing the call keyword arguments (line 291)
    kwargs_70157 = {}
    # Getting the type of 'isstringfunction' (line 291)
    isstringfunction_70155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 7), 'isstringfunction', False)
    # Calling isstringfunction(args, kwargs) (line 291)
    isstringfunction_call_result_70158 = invoke(stypy.reporting.localization.Localization(__file__, 291, 7), isstringfunction_70155, *[var_70156], **kwargs_70157)
    
    # Testing the type of an if condition (line 291)
    if_condition_70159 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 291, 4), isstringfunction_call_result_70158)
    # Assigning a type to the variable 'if_condition_70159' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'if_condition_70159', if_condition_70159)
    # SSA begins for if statement (line 291)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    str_70160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 11), 'str', 'result')
    # Getting the type of 'var' (line 292)
    var_70161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 23), 'var')
    # Applying the binary operator 'in' (line 292)
    result_contains_70162 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 11), 'in', str_70160, var_70161)
    
    # Testing the type of an if condition (line 292)
    if_condition_70163 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 292, 8), result_contains_70162)
    # Assigning a type to the variable 'if_condition_70163' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'if_condition_70163', if_condition_70163)
    # SSA begins for if statement (line 292)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 293):
    
    # Assigning a Subscript to a Name (line 293):
    
    # Obtaining the type of the subscript
    str_70164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 20), 'str', 'result')
    # Getting the type of 'var' (line 293)
    var_70165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 16), 'var')
    # Obtaining the member '__getitem__' of a type (line 293)
    getitem___70166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 16), var_70165, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 293)
    subscript_call_result_70167 = invoke(stypy.reporting.localization.Localization(__file__, 293, 16), getitem___70166, str_70164)
    
    # Assigning a type to the variable 'a' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'a', subscript_call_result_70167)
    # SSA branch for the else part of an if statement (line 292)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 295):
    
    # Assigning a Subscript to a Name (line 295):
    
    # Obtaining the type of the subscript
    str_70168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 20), 'str', 'name')
    # Getting the type of 'var' (line 295)
    var_70169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'var')
    # Obtaining the member '__getitem__' of a type (line 295)
    getitem___70170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 16), var_70169, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 295)
    subscript_call_result_70171 = invoke(stypy.reporting.localization.Localization(__file__, 295, 16), getitem___70170, str_70168)
    
    # Assigning a type to the variable 'a' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'a', subscript_call_result_70171)
    # SSA join for if statement (line 292)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'a' (line 296)
    a_70172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 11), 'a')
    
    # Obtaining the type of the subscript
    str_70173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 20), 'str', 'vars')
    # Getting the type of 'var' (line 296)
    var_70174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 16), 'var')
    # Obtaining the member '__getitem__' of a type (line 296)
    getitem___70175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 16), var_70174, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 296)
    subscript_call_result_70176 = invoke(stypy.reporting.localization.Localization(__file__, 296, 16), getitem___70175, str_70173)
    
    # Applying the binary operator 'in' (line 296)
    result_contains_70177 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 11), 'in', a_70172, subscript_call_result_70176)
    
    # Testing the type of an if condition (line 296)
    if_condition_70178 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 296, 8), result_contains_70177)
    # Assigning a type to the variable 'if_condition_70178' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'if_condition_70178', if_condition_70178)
    # SSA begins for if statement (line 296)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to getstrlength(...): (line 297)
    # Processing the call arguments (line 297)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 297)
    a_70180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 44), 'a', False)
    
    # Obtaining the type of the subscript
    str_70181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 36), 'str', 'vars')
    # Getting the type of 'var' (line 297)
    var_70182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 32), 'var', False)
    # Obtaining the member '__getitem__' of a type (line 297)
    getitem___70183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 32), var_70182, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 297)
    subscript_call_result_70184 = invoke(stypy.reporting.localization.Localization(__file__, 297, 32), getitem___70183, str_70181)
    
    # Obtaining the member '__getitem__' of a type (line 297)
    getitem___70185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 32), subscript_call_result_70184, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 297)
    subscript_call_result_70186 = invoke(stypy.reporting.localization.Localization(__file__, 297, 32), getitem___70185, a_70180)
    
    # Processing the call keyword arguments (line 297)
    kwargs_70187 = {}
    # Getting the type of 'getstrlength' (line 297)
    getstrlength_70179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 19), 'getstrlength', False)
    # Calling getstrlength(args, kwargs) (line 297)
    getstrlength_call_result_70188 = invoke(stypy.reporting.localization.Localization(__file__, 297, 19), getstrlength_70179, *[subscript_call_result_70186], **kwargs_70187)
    
    # Assigning a type to the variable 'stypy_return_type' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'stypy_return_type', getstrlength_call_result_70188)
    # SSA branch for the else part of an if statement (line 296)
    module_type_store.open_ssa_branch('else')
    
    # Call to errmess(...): (line 299)
    # Processing the call arguments (line 299)
    str_70190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 20), 'str', 'getstrlength: function %s has no return value?!\n')
    # Getting the type of 'a' (line 299)
    a_70191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 74), 'a', False)
    # Applying the binary operator '%' (line 299)
    result_mod_70192 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 20), '%', str_70190, a_70191)
    
    # Processing the call keyword arguments (line 299)
    kwargs_70193 = {}
    # Getting the type of 'errmess' (line 299)
    errmess_70189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'errmess', False)
    # Calling errmess(args, kwargs) (line 299)
    errmess_call_result_70194 = invoke(stypy.reporting.localization.Localization(__file__, 299, 12), errmess_70189, *[result_mod_70192], **kwargs_70193)
    
    # SSA join for if statement (line 296)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 291)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to isstring(...): (line 300)
    # Processing the call arguments (line 300)
    # Getting the type of 'var' (line 300)
    var_70196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 20), 'var', False)
    # Processing the call keyword arguments (line 300)
    kwargs_70197 = {}
    # Getting the type of 'isstring' (line 300)
    isstring_70195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 11), 'isstring', False)
    # Calling isstring(args, kwargs) (line 300)
    isstring_call_result_70198 = invoke(stypy.reporting.localization.Localization(__file__, 300, 11), isstring_70195, *[var_70196], **kwargs_70197)
    
    # Applying the 'not' unary operator (line 300)
    result_not__70199 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 7), 'not', isstring_call_result_70198)
    
    # Testing the type of an if condition (line 300)
    if_condition_70200 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 300, 4), result_not__70199)
    # Assigning a type to the variable 'if_condition_70200' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'if_condition_70200', if_condition_70200)
    # SSA begins for if statement (line 300)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to errmess(...): (line 301)
    # Processing the call arguments (line 301)
    str_70202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 12), 'str', 'getstrlength: expected a signature of a string but got: %s\n')
    
    # Call to repr(...): (line 302)
    # Processing the call arguments (line 302)
    # Getting the type of 'var' (line 302)
    var_70204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 83), 'var', False)
    # Processing the call keyword arguments (line 302)
    kwargs_70205 = {}
    # Getting the type of 'repr' (line 302)
    repr_70203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 78), 'repr', False)
    # Calling repr(args, kwargs) (line 302)
    repr_call_result_70206 = invoke(stypy.reporting.localization.Localization(__file__, 302, 78), repr_70203, *[var_70204], **kwargs_70205)
    
    # Applying the binary operator '%' (line 302)
    result_mod_70207 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 12), '%', str_70202, repr_call_result_70206)
    
    # Processing the call keyword arguments (line 301)
    kwargs_70208 = {}
    # Getting the type of 'errmess' (line 301)
    errmess_70201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'errmess', False)
    # Calling errmess(args, kwargs) (line 301)
    errmess_call_result_70209 = invoke(stypy.reporting.localization.Localization(__file__, 301, 8), errmess_70201, *[result_mod_70207], **kwargs_70208)
    
    # SSA join for if statement (line 300)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Str to a Name (line 303):
    
    # Assigning a Str to a Name (line 303):
    str_70210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 10), 'str', '1')
    # Assigning a type to the variable 'len' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'len', str_70210)
    
    
    str_70211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 7), 'str', 'charselector')
    # Getting the type of 'var' (line 304)
    var_70212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 25), 'var')
    # Applying the binary operator 'in' (line 304)
    result_contains_70213 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 7), 'in', str_70211, var_70212)
    
    # Testing the type of an if condition (line 304)
    if_condition_70214 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 304, 4), result_contains_70213)
    # Assigning a type to the variable 'if_condition_70214' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'if_condition_70214', if_condition_70214)
    # SSA begins for if statement (line 304)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 305):
    
    # Assigning a Subscript to a Name (line 305):
    
    # Obtaining the type of the subscript
    str_70215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 16), 'str', 'charselector')
    # Getting the type of 'var' (line 305)
    var_70216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'var')
    # Obtaining the member '__getitem__' of a type (line 305)
    getitem___70217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 12), var_70216, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 305)
    subscript_call_result_70218 = invoke(stypy.reporting.localization.Localization(__file__, 305, 12), getitem___70217, str_70215)
    
    # Assigning a type to the variable 'a' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'a', subscript_call_result_70218)
    
    
    str_70219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 11), 'str', '*')
    # Getting the type of 'a' (line 306)
    a_70220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 18), 'a')
    # Applying the binary operator 'in' (line 306)
    result_contains_70221 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 11), 'in', str_70219, a_70220)
    
    # Testing the type of an if condition (line 306)
    if_condition_70222 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 306, 8), result_contains_70221)
    # Assigning a type to the variable 'if_condition_70222' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'if_condition_70222', if_condition_70222)
    # SSA begins for if statement (line 306)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 307):
    
    # Assigning a Subscript to a Name (line 307):
    
    # Obtaining the type of the subscript
    str_70223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 20), 'str', '*')
    # Getting the type of 'a' (line 307)
    a_70224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 18), 'a')
    # Obtaining the member '__getitem__' of a type (line 307)
    getitem___70225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 18), a_70224, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 307)
    subscript_call_result_70226 = invoke(stypy.reporting.localization.Localization(__file__, 307, 18), getitem___70225, str_70223)
    
    # Assigning a type to the variable 'len' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'len', subscript_call_result_70226)
    # SSA branch for the else part of an if statement (line 306)
    module_type_store.open_ssa_branch('else')
    
    
    str_70227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 13), 'str', 'len')
    # Getting the type of 'a' (line 308)
    a_70228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 22), 'a')
    # Applying the binary operator 'in' (line 308)
    result_contains_70229 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 13), 'in', str_70227, a_70228)
    
    # Testing the type of an if condition (line 308)
    if_condition_70230 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 308, 13), result_contains_70229)
    # Assigning a type to the variable 'if_condition_70230' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 13), 'if_condition_70230', if_condition_70230)
    # SSA begins for if statement (line 308)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 309):
    
    # Assigning a Subscript to a Name (line 309):
    
    # Obtaining the type of the subscript
    str_70231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 20), 'str', 'len')
    # Getting the type of 'a' (line 309)
    a_70232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 18), 'a')
    # Obtaining the member '__getitem__' of a type (line 309)
    getitem___70233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 18), a_70232, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 309)
    subscript_call_result_70234 = invoke(stypy.reporting.localization.Localization(__file__, 309, 18), getitem___70233, str_70231)
    
    # Assigning a type to the variable 'len' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'len', subscript_call_result_70234)
    # SSA join for if statement (line 308)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 306)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 304)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Call to match(...): (line 310)
    # Processing the call arguments (line 310)
    str_70237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 16), 'str', '\\(\\s*([*]|[:])\\s*\\)')
    # Getting the type of 'len' (line 310)
    len_70238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 40), 'len', False)
    # Processing the call keyword arguments (line 310)
    kwargs_70239 = {}
    # Getting the type of 're' (line 310)
    re_70235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 7), 're', False)
    # Obtaining the member 'match' of a type (line 310)
    match_70236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 7), re_70235, 'match')
    # Calling match(args, kwargs) (line 310)
    match_call_result_70240 = invoke(stypy.reporting.localization.Localization(__file__, 310, 7), match_70236, *[str_70237, len_70238], **kwargs_70239)
    
    
    # Call to match(...): (line 310)
    # Processing the call arguments (line 310)
    str_70243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 57), 'str', '([*]|[:])')
    # Getting the type of 'len' (line 310)
    len_70244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 71), 'len', False)
    # Processing the call keyword arguments (line 310)
    kwargs_70245 = {}
    # Getting the type of 're' (line 310)
    re_70241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 48), 're', False)
    # Obtaining the member 'match' of a type (line 310)
    match_70242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 48), re_70241, 'match')
    # Calling match(args, kwargs) (line 310)
    match_call_result_70246 = invoke(stypy.reporting.localization.Localization(__file__, 310, 48), match_70242, *[str_70243, len_70244], **kwargs_70245)
    
    # Applying the binary operator 'or' (line 310)
    result_or_keyword_70247 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 7), 'or', match_call_result_70240, match_call_result_70246)
    
    # Testing the type of an if condition (line 310)
    if_condition_70248 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 310, 4), result_or_keyword_70247)
    # Assigning a type to the variable 'if_condition_70248' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'if_condition_70248', if_condition_70248)
    # SSA begins for if statement (line 310)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to isintent_hide(...): (line 311)
    # Processing the call arguments (line 311)
    # Getting the type of 'var' (line 311)
    var_70250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 25), 'var', False)
    # Processing the call keyword arguments (line 311)
    kwargs_70251 = {}
    # Getting the type of 'isintent_hide' (line 311)
    isintent_hide_70249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 11), 'isintent_hide', False)
    # Calling isintent_hide(args, kwargs) (line 311)
    isintent_hide_call_result_70252 = invoke(stypy.reporting.localization.Localization(__file__, 311, 11), isintent_hide_70249, *[var_70250], **kwargs_70251)
    
    # Testing the type of an if condition (line 311)
    if_condition_70253 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 311, 8), isintent_hide_call_result_70252)
    # Assigning a type to the variable 'if_condition_70253' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'if_condition_70253', if_condition_70253)
    # SSA begins for if statement (line 311)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to errmess(...): (line 312)
    # Processing the call arguments (line 312)
    str_70255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 20), 'str', 'getstrlength:intent(hide): expected a string with defined length but got: %s\n')
    
    # Call to repr(...): (line 313)
    # Processing the call arguments (line 313)
    # Getting the type of 'var' (line 313)
    var_70257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 21), 'var', False)
    # Processing the call keyword arguments (line 313)
    kwargs_70258 = {}
    # Getting the type of 'repr' (line 313)
    repr_70256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 16), 'repr', False)
    # Calling repr(args, kwargs) (line 313)
    repr_call_result_70259 = invoke(stypy.reporting.localization.Localization(__file__, 313, 16), repr_70256, *[var_70257], **kwargs_70258)
    
    # Applying the binary operator '%' (line 312)
    result_mod_70260 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 20), '%', str_70255, repr_call_result_70259)
    
    # Processing the call keyword arguments (line 312)
    kwargs_70261 = {}
    # Getting the type of 'errmess' (line 312)
    errmess_70254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'errmess', False)
    # Calling errmess(args, kwargs) (line 312)
    errmess_call_result_70262 = invoke(stypy.reporting.localization.Localization(__file__, 312, 12), errmess_70254, *[result_mod_70260], **kwargs_70261)
    
    # SSA join for if statement (line 311)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Str to a Name (line 314):
    
    # Assigning a Str to a Name (line 314):
    str_70263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 14), 'str', '-1')
    # Assigning a type to the variable 'len' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'len', str_70263)
    # SSA join for if statement (line 310)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'len' (line 315)
    len_70264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 11), 'len')
    # Assigning a type to the variable 'stypy_return_type' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'stypy_return_type', len_70264)
    
    # ################# End of 'getstrlength(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getstrlength' in the type store
    # Getting the type of 'stypy_return_type' (line 290)
    stypy_return_type_70265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_70265)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getstrlength'
    return stypy_return_type_70265

# Assigning a type to the variable 'getstrlength' (line 290)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 0), 'getstrlength', getstrlength)

@norecursion
def getarrdims(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_70266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 31), 'int')
    defaults = [int_70266]
    # Create a new context for function 'getarrdims'
    module_type_store = module_type_store.open_function_context('getarrdims', 318, 0, False)
    
    # Passed parameters checking function
    getarrdims.stypy_localization = localization
    getarrdims.stypy_type_of_self = None
    getarrdims.stypy_type_store = module_type_store
    getarrdims.stypy_function_name = 'getarrdims'
    getarrdims.stypy_param_names_list = ['a', 'var', 'verbose']
    getarrdims.stypy_varargs_param_name = None
    getarrdims.stypy_kwargs_param_name = None
    getarrdims.stypy_call_defaults = defaults
    getarrdims.stypy_call_varargs = varargs
    getarrdims.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getarrdims', ['a', 'var', 'verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getarrdims', localization, ['a', 'var', 'verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getarrdims(...)' code ##################

    # Marking variables as global (line 319)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 319, 4), 'depargs')
    
    # Assigning a Dict to a Name (line 320):
    
    # Assigning a Dict to a Name (line 320):
    
    # Obtaining an instance of the builtin type 'dict' (line 320)
    dict_70267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 10), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 320)
    
    # Assigning a type to the variable 'ret' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'ret', dict_70267)
    
    
    # Evaluating a boolean operation
    
    # Call to isstring(...): (line 321)
    # Processing the call arguments (line 321)
    # Getting the type of 'var' (line 321)
    var_70269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 16), 'var', False)
    # Processing the call keyword arguments (line 321)
    kwargs_70270 = {}
    # Getting the type of 'isstring' (line 321)
    isstring_70268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 7), 'isstring', False)
    # Calling isstring(args, kwargs) (line 321)
    isstring_call_result_70271 = invoke(stypy.reporting.localization.Localization(__file__, 321, 7), isstring_70268, *[var_70269], **kwargs_70270)
    
    
    
    # Call to isarray(...): (line 321)
    # Processing the call arguments (line 321)
    # Getting the type of 'var' (line 321)
    var_70273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 37), 'var', False)
    # Processing the call keyword arguments (line 321)
    kwargs_70274 = {}
    # Getting the type of 'isarray' (line 321)
    isarray_70272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 29), 'isarray', False)
    # Calling isarray(args, kwargs) (line 321)
    isarray_call_result_70275 = invoke(stypy.reporting.localization.Localization(__file__, 321, 29), isarray_70272, *[var_70273], **kwargs_70274)
    
    # Applying the 'not' unary operator (line 321)
    result_not__70276 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 25), 'not', isarray_call_result_70275)
    
    # Applying the binary operator 'and' (line 321)
    result_and_keyword_70277 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 7), 'and', isstring_call_result_70271, result_not__70276)
    
    # Testing the type of an if condition (line 321)
    if_condition_70278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 321, 4), result_and_keyword_70277)
    # Assigning a type to the variable 'if_condition_70278' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'if_condition_70278', if_condition_70278)
    # SSA begins for if statement (line 321)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 322):
    
    # Assigning a Call to a Subscript (line 322):
    
    # Call to getstrlength(...): (line 322)
    # Processing the call arguments (line 322)
    # Getting the type of 'var' (line 322)
    var_70280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 35), 'var', False)
    # Processing the call keyword arguments (line 322)
    kwargs_70281 = {}
    # Getting the type of 'getstrlength' (line 322)
    getstrlength_70279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 22), 'getstrlength', False)
    # Calling getstrlength(args, kwargs) (line 322)
    getstrlength_call_result_70282 = invoke(stypy.reporting.localization.Localization(__file__, 322, 22), getstrlength_70279, *[var_70280], **kwargs_70281)
    
    # Getting the type of 'ret' (line 322)
    ret_70283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'ret')
    str_70284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 12), 'str', 'dims')
    # Storing an element on a container (line 322)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 8), ret_70283, (str_70284, getstrlength_call_result_70282))
    
    # Assigning a Subscript to a Subscript (line 323):
    
    # Assigning a Subscript to a Subscript (line 323):
    
    # Obtaining the type of the subscript
    str_70285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 26), 'str', 'dims')
    # Getting the type of 'ret' (line 323)
    ret_70286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 22), 'ret')
    # Obtaining the member '__getitem__' of a type (line 323)
    getitem___70287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 22), ret_70286, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 323)
    subscript_call_result_70288 = invoke(stypy.reporting.localization.Localization(__file__, 323, 22), getitem___70287, str_70285)
    
    # Getting the type of 'ret' (line 323)
    ret_70289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'ret')
    str_70290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 12), 'str', 'size')
    # Storing an element on a container (line 323)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 8), ret_70289, (str_70290, subscript_call_result_70288))
    
    # Assigning a Str to a Subscript (line 324):
    
    # Assigning a Str to a Subscript (line 324):
    str_70291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 22), 'str', '1')
    # Getting the type of 'ret' (line 324)
    ret_70292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'ret')
    str_70293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 12), 'str', 'rank')
    # Storing an element on a container (line 324)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 8), ret_70292, (str_70293, str_70291))
    # SSA branch for the else part of an if statement (line 321)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isscalar(...): (line 325)
    # Processing the call arguments (line 325)
    # Getting the type of 'var' (line 325)
    var_70295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 18), 'var', False)
    # Processing the call keyword arguments (line 325)
    kwargs_70296 = {}
    # Getting the type of 'isscalar' (line 325)
    isscalar_70294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 9), 'isscalar', False)
    # Calling isscalar(args, kwargs) (line 325)
    isscalar_call_result_70297 = invoke(stypy.reporting.localization.Localization(__file__, 325, 9), isscalar_70294, *[var_70295], **kwargs_70296)
    
    # Testing the type of an if condition (line 325)
    if_condition_70298 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 325, 9), isscalar_call_result_70297)
    # Assigning a type to the variable 'if_condition_70298' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 9), 'if_condition_70298', if_condition_70298)
    # SSA begins for if statement (line 325)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Subscript (line 326):
    
    # Assigning a Str to a Subscript (line 326):
    str_70299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 22), 'str', '1')
    # Getting the type of 'ret' (line 326)
    ret_70300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'ret')
    str_70301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 12), 'str', 'size')
    # Storing an element on a container (line 326)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 8), ret_70300, (str_70301, str_70299))
    
    # Assigning a Str to a Subscript (line 327):
    
    # Assigning a Str to a Subscript (line 327):
    str_70302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 22), 'str', '0')
    # Getting the type of 'ret' (line 327)
    ret_70303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'ret')
    str_70304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 12), 'str', 'rank')
    # Storing an element on a container (line 327)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 8), ret_70303, (str_70304, str_70302))
    
    # Assigning a Str to a Subscript (line 328):
    
    # Assigning a Str to a Subscript (line 328):
    str_70305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 22), 'str', '')
    # Getting the type of 'ret' (line 328)
    ret_70306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'ret')
    str_70307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 12), 'str', 'dims')
    # Storing an element on a container (line 328)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 8), ret_70306, (str_70307, str_70305))
    # SSA branch for the else part of an if statement (line 325)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isarray(...): (line 329)
    # Processing the call arguments (line 329)
    # Getting the type of 'var' (line 329)
    var_70309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 17), 'var', False)
    # Processing the call keyword arguments (line 329)
    kwargs_70310 = {}
    # Getting the type of 'isarray' (line 329)
    isarray_70308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 9), 'isarray', False)
    # Calling isarray(args, kwargs) (line 329)
    isarray_call_result_70311 = invoke(stypy.reporting.localization.Localization(__file__, 329, 9), isarray_70308, *[var_70309], **kwargs_70310)
    
    # Testing the type of an if condition (line 329)
    if_condition_70312 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 329, 9), isarray_call_result_70311)
    # Assigning a type to the variable 'if_condition_70312' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 9), 'if_condition_70312', if_condition_70312)
    # SSA begins for if statement (line 329)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 330):
    
    # Assigning a Call to a Name (line 330):
    
    # Call to copy(...): (line 330)
    # Processing the call arguments (line 330)
    
    # Obtaining the type of the subscript
    str_70315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 28), 'str', 'dimension')
    # Getting the type of 'var' (line 330)
    var_70316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 24), 'var', False)
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___70317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 24), var_70316, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 330)
    subscript_call_result_70318 = invoke(stypy.reporting.localization.Localization(__file__, 330, 24), getitem___70317, str_70315)
    
    # Processing the call keyword arguments (line 330)
    kwargs_70319 = {}
    # Getting the type of 'copy' (line 330)
    copy_70313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 14), 'copy', False)
    # Obtaining the member 'copy' of a type (line 330)
    copy_70314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 14), copy_70313, 'copy')
    # Calling copy(args, kwargs) (line 330)
    copy_call_result_70320 = invoke(stypy.reporting.localization.Localization(__file__, 330, 14), copy_70314, *[subscript_call_result_70318], **kwargs_70319)
    
    # Assigning a type to the variable 'dim' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'dim', copy_call_result_70320)
    
    # Assigning a Call to a Subscript (line 331):
    
    # Assigning a Call to a Subscript (line 331):
    
    # Call to join(...): (line 331)
    # Processing the call arguments (line 331)
    # Getting the type of 'dim' (line 331)
    dim_70323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 31), 'dim', False)
    # Processing the call keyword arguments (line 331)
    kwargs_70324 = {}
    str_70321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 22), 'str', '*')
    # Obtaining the member 'join' of a type (line 331)
    join_70322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 22), str_70321, 'join')
    # Calling join(args, kwargs) (line 331)
    join_call_result_70325 = invoke(stypy.reporting.localization.Localization(__file__, 331, 22), join_70322, *[dim_70323], **kwargs_70324)
    
    # Getting the type of 'ret' (line 331)
    ret_70326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'ret')
    str_70327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 12), 'str', 'size')
    # Storing an element on a container (line 331)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 8), ret_70326, (str_70327, join_call_result_70325))
    
    
    # SSA begins for try-except statement (line 332)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Subscript (line 333):
    
    # Assigning a Call to a Subscript (line 333):
    
    # Call to repr(...): (line 333)
    # Processing the call arguments (line 333)
    
    # Call to eval(...): (line 333)
    # Processing the call arguments (line 333)
    
    # Obtaining the type of the subscript
    str_70330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 40), 'str', 'size')
    # Getting the type of 'ret' (line 333)
    ret_70331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 36), 'ret', False)
    # Obtaining the member '__getitem__' of a type (line 333)
    getitem___70332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 36), ret_70331, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 333)
    subscript_call_result_70333 = invoke(stypy.reporting.localization.Localization(__file__, 333, 36), getitem___70332, str_70330)
    
    # Processing the call keyword arguments (line 333)
    kwargs_70334 = {}
    # Getting the type of 'eval' (line 333)
    eval_70329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 31), 'eval', False)
    # Calling eval(args, kwargs) (line 333)
    eval_call_result_70335 = invoke(stypy.reporting.localization.Localization(__file__, 333, 31), eval_70329, *[subscript_call_result_70333], **kwargs_70334)
    
    # Processing the call keyword arguments (line 333)
    kwargs_70336 = {}
    # Getting the type of 'repr' (line 333)
    repr_70328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 26), 'repr', False)
    # Calling repr(args, kwargs) (line 333)
    repr_call_result_70337 = invoke(stypy.reporting.localization.Localization(__file__, 333, 26), repr_70328, *[eval_call_result_70335], **kwargs_70336)
    
    # Getting the type of 'ret' (line 333)
    ret_70338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'ret')
    str_70339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 16), 'str', 'size')
    # Storing an element on a container (line 333)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 12), ret_70338, (str_70339, repr_call_result_70337))
    # SSA branch for the except part of a try statement (line 332)
    # SSA branch for the except '<any exception>' branch of a try statement (line 332)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 332)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Subscript (line 336):
    
    # Assigning a Call to a Subscript (line 336):
    
    # Call to join(...): (line 336)
    # Processing the call arguments (line 336)
    # Getting the type of 'dim' (line 336)
    dim_70342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 31), 'dim', False)
    # Processing the call keyword arguments (line 336)
    kwargs_70343 = {}
    str_70340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 22), 'str', ',')
    # Obtaining the member 'join' of a type (line 336)
    join_70341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 22), str_70340, 'join')
    # Calling join(args, kwargs) (line 336)
    join_call_result_70344 = invoke(stypy.reporting.localization.Localization(__file__, 336, 22), join_70341, *[dim_70342], **kwargs_70343)
    
    # Getting the type of 'ret' (line 336)
    ret_70345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'ret')
    str_70346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 12), 'str', 'dims')
    # Storing an element on a container (line 336)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 8), ret_70345, (str_70346, join_call_result_70344))
    
    # Assigning a Call to a Subscript (line 337):
    
    # Assigning a Call to a Subscript (line 337):
    
    # Call to repr(...): (line 337)
    # Processing the call arguments (line 337)
    
    # Call to len(...): (line 337)
    # Processing the call arguments (line 337)
    # Getting the type of 'dim' (line 337)
    dim_70349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 31), 'dim', False)
    # Processing the call keyword arguments (line 337)
    kwargs_70350 = {}
    # Getting the type of 'len' (line 337)
    len_70348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 27), 'len', False)
    # Calling len(args, kwargs) (line 337)
    len_call_result_70351 = invoke(stypy.reporting.localization.Localization(__file__, 337, 27), len_70348, *[dim_70349], **kwargs_70350)
    
    # Processing the call keyword arguments (line 337)
    kwargs_70352 = {}
    # Getting the type of 'repr' (line 337)
    repr_70347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 22), 'repr', False)
    # Calling repr(args, kwargs) (line 337)
    repr_call_result_70353 = invoke(stypy.reporting.localization.Localization(__file__, 337, 22), repr_70347, *[len_call_result_70351], **kwargs_70352)
    
    # Getting the type of 'ret' (line 337)
    ret_70354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'ret')
    str_70355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 12), 'str', 'rank')
    # Storing an element on a container (line 337)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 8), ret_70354, (str_70355, repr_call_result_70353))
    
    # Assigning a Subscript to a Subscript (line 338):
    
    # Assigning a Subscript to a Subscript (line 338):
    
    # Obtaining the type of the subscript
    int_70356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 49), 'int')
    int_70357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 51), 'int')
    slice_70358 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 338, 27), int_70356, int_70357, None)
    
    # Call to repr(...): (line 338)
    # Processing the call arguments (line 338)
    
    # Call to len(...): (line 338)
    # Processing the call arguments (line 338)
    # Getting the type of 'dim' (line 338)
    dim_70361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 36), 'dim', False)
    # Processing the call keyword arguments (line 338)
    kwargs_70362 = {}
    # Getting the type of 'len' (line 338)
    len_70360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 32), 'len', False)
    # Calling len(args, kwargs) (line 338)
    len_call_result_70363 = invoke(stypy.reporting.localization.Localization(__file__, 338, 32), len_70360, *[dim_70361], **kwargs_70362)
    
    
    # Obtaining an instance of the builtin type 'list' (line 338)
    list_70364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 338)
    # Adding element type (line 338)
    int_70365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 43), list_70364, int_70365)
    
    # Applying the binary operator '*' (line 338)
    result_mul_70366 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 32), '*', len_call_result_70363, list_70364)
    
    # Processing the call keyword arguments (line 338)
    kwargs_70367 = {}
    # Getting the type of 'repr' (line 338)
    repr_70359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 27), 'repr', False)
    # Calling repr(args, kwargs) (line 338)
    repr_call_result_70368 = invoke(stypy.reporting.localization.Localization(__file__, 338, 27), repr_70359, *[result_mul_70366], **kwargs_70367)
    
    # Obtaining the member '__getitem__' of a type (line 338)
    getitem___70369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 27), repr_call_result_70368, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 338)
    subscript_call_result_70370 = invoke(stypy.reporting.localization.Localization(__file__, 338, 27), getitem___70369, slice_70358)
    
    # Getting the type of 'ret' (line 338)
    ret_70371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'ret')
    str_70372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 12), 'str', 'rank*[-1]')
    # Storing an element on a container (line 338)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 8), ret_70371, (str_70372, subscript_call_result_70370))
    
    
    # Call to range(...): (line 339)
    # Processing the call arguments (line 339)
    
    # Call to len(...): (line 339)
    # Processing the call arguments (line 339)
    # Getting the type of 'dim' (line 339)
    dim_70375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 27), 'dim', False)
    # Processing the call keyword arguments (line 339)
    kwargs_70376 = {}
    # Getting the type of 'len' (line 339)
    len_70374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 23), 'len', False)
    # Calling len(args, kwargs) (line 339)
    len_call_result_70377 = invoke(stypy.reporting.localization.Localization(__file__, 339, 23), len_70374, *[dim_70375], **kwargs_70376)
    
    # Processing the call keyword arguments (line 339)
    kwargs_70378 = {}
    # Getting the type of 'range' (line 339)
    range_70373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 17), 'range', False)
    # Calling range(args, kwargs) (line 339)
    range_call_result_70379 = invoke(stypy.reporting.localization.Localization(__file__, 339, 17), range_70373, *[len_call_result_70377], **kwargs_70378)
    
    # Testing the type of a for loop iterable (line 339)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 339, 8), range_call_result_70379)
    # Getting the type of the for loop variable (line 339)
    for_loop_var_70380 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 339, 8), range_call_result_70379)
    # Assigning a type to the variable 'i' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'i', for_loop_var_70380)
    # SSA begins for a for statement (line 339)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a List to a Name (line 340):
    
    # Assigning a List to a Name (line 340):
    
    # Obtaining an instance of the builtin type 'list' (line 340)
    list_70381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 340)
    
    # Assigning a type to the variable 'v' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 12), 'v', list_70381)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 341)
    i_70382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 19), 'i')
    # Getting the type of 'dim' (line 341)
    dim_70383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 15), 'dim')
    # Obtaining the member '__getitem__' of a type (line 341)
    getitem___70384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 15), dim_70383, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 341)
    subscript_call_result_70385 = invoke(stypy.reporting.localization.Localization(__file__, 341, 15), getitem___70384, i_70382)
    
    # Getting the type of 'depargs' (line 341)
    depargs_70386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 25), 'depargs')
    # Applying the binary operator 'in' (line 341)
    result_contains_70387 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 15), 'in', subscript_call_result_70385, depargs_70386)
    
    # Testing the type of an if condition (line 341)
    if_condition_70388 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 341, 12), result_contains_70387)
    # Assigning a type to the variable 'if_condition_70388' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'if_condition_70388', if_condition_70388)
    # SSA begins for if statement (line 341)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 342):
    
    # Assigning a List to a Name (line 342):
    
    # Obtaining an instance of the builtin type 'list' (line 342)
    list_70389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 342)
    # Adding element type (line 342)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 342)
    i_70390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 25), 'i')
    # Getting the type of 'dim' (line 342)
    dim_70391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 21), 'dim')
    # Obtaining the member '__getitem__' of a type (line 342)
    getitem___70392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 21), dim_70391, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 342)
    subscript_call_result_70393 = invoke(stypy.reporting.localization.Localization(__file__, 342, 21), getitem___70392, i_70390)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 20), list_70389, subscript_call_result_70393)
    
    # Assigning a type to the variable 'v' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 16), 'v', list_70389)
    # SSA branch for the else part of an if statement (line 341)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'depargs' (line 344)
    depargs_70394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 26), 'depargs')
    # Testing the type of a for loop iterable (line 344)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 344, 16), depargs_70394)
    # Getting the type of the for loop variable (line 344)
    for_loop_var_70395 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 344, 16), depargs_70394)
    # Assigning a type to the variable 'va' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 16), 'va', for_loop_var_70395)
    # SSA begins for a for statement (line 344)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to match(...): (line 345)
    # Processing the call arguments (line 345)
    str_70398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 32), 'str', '.*?\\b%s\\b.*')
    # Getting the type of 'va' (line 345)
    va_70399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 49), 'va', False)
    # Applying the binary operator '%' (line 345)
    result_mod_70400 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 32), '%', str_70398, va_70399)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 345)
    i_70401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 57), 'i', False)
    # Getting the type of 'dim' (line 345)
    dim_70402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 53), 'dim', False)
    # Obtaining the member '__getitem__' of a type (line 345)
    getitem___70403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 53), dim_70402, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 345)
    subscript_call_result_70404 = invoke(stypy.reporting.localization.Localization(__file__, 345, 53), getitem___70403, i_70401)
    
    # Processing the call keyword arguments (line 345)
    kwargs_70405 = {}
    # Getting the type of 're' (line 345)
    re_70396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 23), 're', False)
    # Obtaining the member 'match' of a type (line 345)
    match_70397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 23), re_70396, 'match')
    # Calling match(args, kwargs) (line 345)
    match_call_result_70406 = invoke(stypy.reporting.localization.Localization(__file__, 345, 23), match_70397, *[result_mod_70400, subscript_call_result_70404], **kwargs_70405)
    
    # Testing the type of an if condition (line 345)
    if_condition_70407 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 345, 20), match_call_result_70406)
    # Assigning a type to the variable 'if_condition_70407' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 20), 'if_condition_70407', if_condition_70407)
    # SSA begins for if statement (line 345)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 346)
    # Processing the call arguments (line 346)
    # Getting the type of 'va' (line 346)
    va_70410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 33), 'va', False)
    # Processing the call keyword arguments (line 346)
    kwargs_70411 = {}
    # Getting the type of 'v' (line 346)
    v_70408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 24), 'v', False)
    # Obtaining the member 'append' of a type (line 346)
    append_70409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 24), v_70408, 'append')
    # Calling append(args, kwargs) (line 346)
    append_call_result_70412 = invoke(stypy.reporting.localization.Localization(__file__, 346, 24), append_70409, *[va_70410], **kwargs_70411)
    
    # SSA join for if statement (line 345)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 341)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'v' (line 347)
    v_70413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 22), 'v')
    # Testing the type of a for loop iterable (line 347)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 347, 12), v_70413)
    # Getting the type of the for loop variable (line 347)
    for_loop_var_70414 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 347, 12), v_70413)
    # Assigning a type to the variable 'va' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'va', for_loop_var_70414)
    # SSA begins for a for statement (line 347)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Call to index(...): (line 348)
    # Processing the call arguments (line 348)
    # Getting the type of 'va' (line 348)
    va_70417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 33), 'va', False)
    # Processing the call keyword arguments (line 348)
    kwargs_70418 = {}
    # Getting the type of 'depargs' (line 348)
    depargs_70415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 19), 'depargs', False)
    # Obtaining the member 'index' of a type (line 348)
    index_70416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 19), depargs_70415, 'index')
    # Calling index(args, kwargs) (line 348)
    index_call_result_70419 = invoke(stypy.reporting.localization.Localization(__file__, 348, 19), index_70416, *[va_70417], **kwargs_70418)
    
    
    # Call to index(...): (line 348)
    # Processing the call arguments (line 348)
    # Getting the type of 'a' (line 348)
    a_70422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 53), 'a', False)
    # Processing the call keyword arguments (line 348)
    kwargs_70423 = {}
    # Getting the type of 'depargs' (line 348)
    depargs_70420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 39), 'depargs', False)
    # Obtaining the member 'index' of a type (line 348)
    index_70421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 39), depargs_70420, 'index')
    # Calling index(args, kwargs) (line 348)
    index_call_result_70424 = invoke(stypy.reporting.localization.Localization(__file__, 348, 39), index_70421, *[a_70422], **kwargs_70423)
    
    # Applying the binary operator '>' (line 348)
    result_gt_70425 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 19), '>', index_call_result_70419, index_call_result_70424)
    
    # Testing the type of an if condition (line 348)
    if_condition_70426 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 348, 16), result_gt_70425)
    # Assigning a type to the variable 'if_condition_70426' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 16), 'if_condition_70426', if_condition_70426)
    # SSA begins for if statement (line 348)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Subscript (line 349):
    
    # Assigning a Str to a Subscript (line 349):
    str_70427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 29), 'str', '*')
    # Getting the type of 'dim' (line 349)
    dim_70428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 20), 'dim')
    # Getting the type of 'i' (line 349)
    i_70429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 24), 'i')
    # Storing an element on a container (line 349)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 20), dim_70428, (i_70429, str_70427))
    # SSA join for if statement (line 348)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Tuple (line 351):
    
    # Assigning a Str to a Name (line 351):
    str_70430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 28), 'str', '')
    # Assigning a type to the variable 'tuple_assignment_69343' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'tuple_assignment_69343', str_70430)
    
    # Assigning a Num to a Name (line 351):
    int_70431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 32), 'int')
    # Assigning a type to the variable 'tuple_assignment_69344' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'tuple_assignment_69344', int_70431)
    
    # Assigning a Name to a Subscript (line 351):
    # Getting the type of 'tuple_assignment_69343' (line 351)
    tuple_assignment_69343_70432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'tuple_assignment_69343')
    # Getting the type of 'ret' (line 351)
    ret_70433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'ret')
    str_70434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 12), 'str', 'setdims')
    # Storing an element on a container (line 351)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 8), ret_70433, (str_70434, tuple_assignment_69343_70432))
    
    # Assigning a Name to a Name (line 351):
    # Getting the type of 'tuple_assignment_69344' (line 351)
    tuple_assignment_69344_70435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'tuple_assignment_69344')
    # Assigning a type to the variable 'i' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 24), 'i', tuple_assignment_69344_70435)
    
    # Getting the type of 'dim' (line 352)
    dim_70436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 17), 'dim')
    # Testing the type of a for loop iterable (line 352)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 352, 8), dim_70436)
    # Getting the type of the for loop variable (line 352)
    for_loop_var_70437 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 352, 8), dim_70436)
    # Assigning a type to the variable 'd' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'd', for_loop_var_70437)
    # SSA begins for a for statement (line 352)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 353):
    
    # Assigning a BinOp to a Name (line 353):
    # Getting the type of 'i' (line 353)
    i_70438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 16), 'i')
    int_70439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 20), 'int')
    # Applying the binary operator '+' (line 353)
    result_add_70440 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 16), '+', i_70438, int_70439)
    
    # Assigning a type to the variable 'i' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'i', result_add_70440)
    
    
    # Getting the type of 'd' (line 354)
    d_70441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 15), 'd')
    
    # Obtaining an instance of the builtin type 'list' (line 354)
    list_70442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 354)
    # Adding element type (line 354)
    str_70443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 25), 'str', '*')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 24), list_70442, str_70443)
    # Adding element type (line 354)
    str_70444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 30), 'str', ':')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 24), list_70442, str_70444)
    # Adding element type (line 354)
    str_70445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 35), 'str', '(*)')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 24), list_70442, str_70445)
    # Adding element type (line 354)
    str_70446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 42), 'str', '(:)')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 24), list_70442, str_70446)
    
    # Applying the binary operator 'notin' (line 354)
    result_contains_70447 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 15), 'notin', d_70441, list_70442)
    
    # Testing the type of an if condition (line 354)
    if_condition_70448 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 354, 12), result_contains_70447)
    # Assigning a type to the variable 'if_condition_70448' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), 'if_condition_70448', if_condition_70448)
    # SSA begins for if statement (line 354)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 355):
    
    # Assigning a BinOp to a Subscript (line 355):
    str_70449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 33), 'str', '%s#varname#_Dims[%d]=%s,')
    
    # Obtaining an instance of the builtin type 'tuple' (line 356)
    tuple_70450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 356)
    # Adding element type (line 356)
    
    # Obtaining the type of the subscript
    str_70451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 24), 'str', 'setdims')
    # Getting the type of 'ret' (line 356)
    ret_70452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 20), 'ret')
    # Obtaining the member '__getitem__' of a type (line 356)
    getitem___70453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 20), ret_70452, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 356)
    subscript_call_result_70454 = invoke(stypy.reporting.localization.Localization(__file__, 356, 20), getitem___70453, str_70451)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 20), tuple_70450, subscript_call_result_70454)
    # Adding element type (line 356)
    # Getting the type of 'i' (line 356)
    i_70455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 36), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 20), tuple_70450, i_70455)
    # Adding element type (line 356)
    # Getting the type of 'd' (line 356)
    d_70456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 39), 'd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 20), tuple_70450, d_70456)
    
    # Applying the binary operator '%' (line 355)
    result_mod_70457 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 33), '%', str_70449, tuple_70450)
    
    # Getting the type of 'ret' (line 355)
    ret_70458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 16), 'ret')
    str_70459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 20), 'str', 'setdims')
    # Storing an element on a container (line 355)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 16), ret_70458, (str_70459, result_mod_70457))
    # SSA join for if statement (line 354)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining the type of the subscript
    str_70460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 15), 'str', 'setdims')
    # Getting the type of 'ret' (line 357)
    ret_70461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 11), 'ret')
    # Obtaining the member '__getitem__' of a type (line 357)
    getitem___70462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 11), ret_70461, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 357)
    subscript_call_result_70463 = invoke(stypy.reporting.localization.Localization(__file__, 357, 11), getitem___70462, str_70460)
    
    # Testing the type of an if condition (line 357)
    if_condition_70464 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 357, 8), subscript_call_result_70463)
    # Assigning a type to the variable 'if_condition_70464' (line 357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'if_condition_70464', if_condition_70464)
    # SSA begins for if statement (line 357)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 358):
    
    # Assigning a Subscript to a Subscript (line 358):
    
    # Obtaining the type of the subscript
    int_70465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 45), 'int')
    slice_70466 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 358, 29), None, int_70465, None)
    
    # Obtaining the type of the subscript
    str_70467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 33), 'str', 'setdims')
    # Getting the type of 'ret' (line 358)
    ret_70468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 29), 'ret')
    # Obtaining the member '__getitem__' of a type (line 358)
    getitem___70469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 29), ret_70468, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 358)
    subscript_call_result_70470 = invoke(stypy.reporting.localization.Localization(__file__, 358, 29), getitem___70469, str_70467)
    
    # Obtaining the member '__getitem__' of a type (line 358)
    getitem___70471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 29), subscript_call_result_70470, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 358)
    subscript_call_result_70472 = invoke(stypy.reporting.localization.Localization(__file__, 358, 29), getitem___70471, slice_70466)
    
    # Getting the type of 'ret' (line 358)
    ret_70473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'ret')
    str_70474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 16), 'str', 'setdims')
    # Storing an element on a container (line 358)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), ret_70473, (str_70474, subscript_call_result_70472))
    # SSA join for if statement (line 357)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Tuple (line 359):
    
    # Assigning a Str to a Name (line 359):
    str_70475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 30), 'str', '')
    # Assigning a type to the variable 'tuple_assignment_69345' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'tuple_assignment_69345', str_70475)
    
    # Assigning a Num to a Name (line 359):
    int_70476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 34), 'int')
    # Assigning a type to the variable 'tuple_assignment_69346' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'tuple_assignment_69346', int_70476)
    
    # Assigning a Name to a Subscript (line 359):
    # Getting the type of 'tuple_assignment_69345' (line 359)
    tuple_assignment_69345_70477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'tuple_assignment_69345')
    # Getting the type of 'ret' (line 359)
    ret_70478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'ret')
    str_70479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 12), 'str', 'cbsetdims')
    # Storing an element on a container (line 359)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 8), ret_70478, (str_70479, tuple_assignment_69345_70477))
    
    # Assigning a Name to a Name (line 359):
    # Getting the type of 'tuple_assignment_69346' (line 359)
    tuple_assignment_69346_70480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'tuple_assignment_69346')
    # Assigning a type to the variable 'i' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 26), 'i', tuple_assignment_69346_70480)
    
    
    # Obtaining the type of the subscript
    str_70481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 21), 'str', 'dimension')
    # Getting the type of 'var' (line 360)
    var_70482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 17), 'var')
    # Obtaining the member '__getitem__' of a type (line 360)
    getitem___70483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 17), var_70482, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 360)
    subscript_call_result_70484 = invoke(stypy.reporting.localization.Localization(__file__, 360, 17), getitem___70483, str_70481)
    
    # Testing the type of a for loop iterable (line 360)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 360, 8), subscript_call_result_70484)
    # Getting the type of the for loop variable (line 360)
    for_loop_var_70485 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 360, 8), subscript_call_result_70484)
    # Assigning a type to the variable 'd' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'd', for_loop_var_70485)
    # SSA begins for a for statement (line 360)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 361):
    
    # Assigning a BinOp to a Name (line 361):
    # Getting the type of 'i' (line 361)
    i_70486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 16), 'i')
    int_70487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 20), 'int')
    # Applying the binary operator '+' (line 361)
    result_add_70488 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 16), '+', i_70486, int_70487)
    
    # Assigning a type to the variable 'i' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'i', result_add_70488)
    
    
    # Getting the type of 'd' (line 362)
    d_70489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 15), 'd')
    
    # Obtaining an instance of the builtin type 'list' (line 362)
    list_70490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 362)
    # Adding element type (line 362)
    str_70491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 25), 'str', '*')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 24), list_70490, str_70491)
    # Adding element type (line 362)
    str_70492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 30), 'str', ':')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 24), list_70490, str_70492)
    # Adding element type (line 362)
    str_70493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 35), 'str', '(*)')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 24), list_70490, str_70493)
    # Adding element type (line 362)
    str_70494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 42), 'str', '(:)')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 24), list_70490, str_70494)
    
    # Applying the binary operator 'notin' (line 362)
    result_contains_70495 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 15), 'notin', d_70489, list_70490)
    
    # Testing the type of an if condition (line 362)
    if_condition_70496 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 362, 12), result_contains_70495)
    # Assigning a type to the variable 'if_condition_70496' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'if_condition_70496', if_condition_70496)
    # SSA begins for if statement (line 362)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 363):
    
    # Assigning a BinOp to a Subscript (line 363):
    str_70497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 35), 'str', '%s#varname#_Dims[%d]=%s,')
    
    # Obtaining an instance of the builtin type 'tuple' (line 364)
    tuple_70498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 364)
    # Adding element type (line 364)
    
    # Obtaining the type of the subscript
    str_70499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 24), 'str', 'cbsetdims')
    # Getting the type of 'ret' (line 364)
    ret_70500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 20), 'ret')
    # Obtaining the member '__getitem__' of a type (line 364)
    getitem___70501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 20), ret_70500, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 364)
    subscript_call_result_70502 = invoke(stypy.reporting.localization.Localization(__file__, 364, 20), getitem___70501, str_70499)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 20), tuple_70498, subscript_call_result_70502)
    # Adding element type (line 364)
    # Getting the type of 'i' (line 364)
    i_70503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 38), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 20), tuple_70498, i_70503)
    # Adding element type (line 364)
    # Getting the type of 'd' (line 364)
    d_70504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 41), 'd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 20), tuple_70498, d_70504)
    
    # Applying the binary operator '%' (line 363)
    result_mod_70505 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 35), '%', str_70497, tuple_70498)
    
    # Getting the type of 'ret' (line 363)
    ret_70506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 16), 'ret')
    str_70507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 20), 'str', 'cbsetdims')
    # Storing an element on a container (line 363)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 16), ret_70506, (str_70507, result_mod_70505))
    # SSA branch for the else part of an if statement (line 362)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isintent_in(...): (line 365)
    # Processing the call arguments (line 365)
    # Getting the type of 'var' (line 365)
    var_70509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 29), 'var', False)
    # Processing the call keyword arguments (line 365)
    kwargs_70510 = {}
    # Getting the type of 'isintent_in' (line 365)
    isintent_in_70508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 17), 'isintent_in', False)
    # Calling isintent_in(args, kwargs) (line 365)
    isintent_in_call_result_70511 = invoke(stypy.reporting.localization.Localization(__file__, 365, 17), isintent_in_70508, *[var_70509], **kwargs_70510)
    
    # Testing the type of an if condition (line 365)
    if_condition_70512 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 365, 17), isintent_in_call_result_70511)
    # Assigning a type to the variable 'if_condition_70512' (line 365)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 17), 'if_condition_70512', if_condition_70512)
    # SSA begins for if statement (line 365)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to outmess(...): (line 366)
    # Processing the call arguments (line 366)
    str_70514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 24), 'str', 'getarrdims:warning: assumed shape array, using 0 instead of %r\n')
    # Getting the type of 'd' (line 367)
    d_70515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 27), 'd', False)
    # Applying the binary operator '%' (line 366)
    result_mod_70516 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 24), '%', str_70514, d_70515)
    
    # Processing the call keyword arguments (line 366)
    kwargs_70517 = {}
    # Getting the type of 'outmess' (line 366)
    outmess_70513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 16), 'outmess', False)
    # Calling outmess(args, kwargs) (line 366)
    outmess_call_result_70518 = invoke(stypy.reporting.localization.Localization(__file__, 366, 16), outmess_70513, *[result_mod_70516], **kwargs_70517)
    
    
    # Assigning a BinOp to a Subscript (line 368):
    
    # Assigning a BinOp to a Subscript (line 368):
    str_70519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 35), 'str', '%s#varname#_Dims[%d]=%s,')
    
    # Obtaining an instance of the builtin type 'tuple' (line 369)
    tuple_70520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 369)
    # Adding element type (line 369)
    
    # Obtaining the type of the subscript
    str_70521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 24), 'str', 'cbsetdims')
    # Getting the type of 'ret' (line 369)
    ret_70522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 20), 'ret')
    # Obtaining the member '__getitem__' of a type (line 369)
    getitem___70523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 20), ret_70522, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 369)
    subscript_call_result_70524 = invoke(stypy.reporting.localization.Localization(__file__, 369, 20), getitem___70523, str_70521)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 20), tuple_70520, subscript_call_result_70524)
    # Adding element type (line 369)
    # Getting the type of 'i' (line 369)
    i_70525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 38), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 20), tuple_70520, i_70525)
    # Adding element type (line 369)
    int_70526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 20), tuple_70520, int_70526)
    
    # Applying the binary operator '%' (line 368)
    result_mod_70527 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 35), '%', str_70519, tuple_70520)
    
    # Getting the type of 'ret' (line 368)
    ret_70528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 16), 'ret')
    str_70529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 20), 'str', 'cbsetdims')
    # Storing an element on a container (line 368)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 16), ret_70528, (str_70529, result_mod_70527))
    # SSA branch for the else part of an if statement (line 365)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'verbose' (line 370)
    verbose_70530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 17), 'verbose')
    # Testing the type of an if condition (line 370)
    if_condition_70531 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 370, 17), verbose_70530)
    # Assigning a type to the variable 'if_condition_70531' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 17), 'if_condition_70531', if_condition_70531)
    # SSA begins for if statement (line 370)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to errmess(...): (line 371)
    # Processing the call arguments (line 371)
    str_70533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 20), 'str', 'getarrdims: If in call-back function: array argument %s must have bounded dimensions: got %s\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 372)
    tuple_70534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 120), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 372)
    # Adding element type (line 372)
    
    # Call to repr(...): (line 372)
    # Processing the call arguments (line 372)
    # Getting the type of 'a' (line 372)
    a_70536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 125), 'a', False)
    # Processing the call keyword arguments (line 372)
    kwargs_70537 = {}
    # Getting the type of 'repr' (line 372)
    repr_70535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 120), 'repr', False)
    # Calling repr(args, kwargs) (line 372)
    repr_call_result_70538 = invoke(stypy.reporting.localization.Localization(__file__, 372, 120), repr_70535, *[a_70536], **kwargs_70537)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 120), tuple_70534, repr_call_result_70538)
    # Adding element type (line 372)
    
    # Call to repr(...): (line 372)
    # Processing the call arguments (line 372)
    # Getting the type of 'd' (line 372)
    d_70540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 134), 'd', False)
    # Processing the call keyword arguments (line 372)
    kwargs_70541 = {}
    # Getting the type of 'repr' (line 372)
    repr_70539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 129), 'repr', False)
    # Calling repr(args, kwargs) (line 372)
    repr_call_result_70542 = invoke(stypy.reporting.localization.Localization(__file__, 372, 129), repr_70539, *[d_70540], **kwargs_70541)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 120), tuple_70534, repr_call_result_70542)
    
    # Applying the binary operator '%' (line 372)
    result_mod_70543 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 20), '%', str_70533, tuple_70534)
    
    # Processing the call keyword arguments (line 371)
    kwargs_70544 = {}
    # Getting the type of 'errmess' (line 371)
    errmess_70532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 16), 'errmess', False)
    # Calling errmess(args, kwargs) (line 371)
    errmess_call_result_70545 = invoke(stypy.reporting.localization.Localization(__file__, 371, 16), errmess_70532, *[result_mod_70543], **kwargs_70544)
    
    # SSA join for if statement (line 370)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 365)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 362)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining the type of the subscript
    str_70546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 15), 'str', 'cbsetdims')
    # Getting the type of 'ret' (line 373)
    ret_70547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 11), 'ret')
    # Obtaining the member '__getitem__' of a type (line 373)
    getitem___70548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 11), ret_70547, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 373)
    subscript_call_result_70549 = invoke(stypy.reporting.localization.Localization(__file__, 373, 11), getitem___70548, str_70546)
    
    # Testing the type of an if condition (line 373)
    if_condition_70550 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 373, 8), subscript_call_result_70549)
    # Assigning a type to the variable 'if_condition_70550' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'if_condition_70550', if_condition_70550)
    # SSA begins for if statement (line 373)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 374):
    
    # Assigning a Subscript to a Subscript (line 374):
    
    # Obtaining the type of the subscript
    int_70551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 49), 'int')
    slice_70552 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 374, 31), None, int_70551, None)
    
    # Obtaining the type of the subscript
    str_70553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 35), 'str', 'cbsetdims')
    # Getting the type of 'ret' (line 374)
    ret_70554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 31), 'ret')
    # Obtaining the member '__getitem__' of a type (line 374)
    getitem___70555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 31), ret_70554, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 374)
    subscript_call_result_70556 = invoke(stypy.reporting.localization.Localization(__file__, 374, 31), getitem___70555, str_70553)
    
    # Obtaining the member '__getitem__' of a type (line 374)
    getitem___70557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 31), subscript_call_result_70556, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 374)
    subscript_call_result_70558 = invoke(stypy.reporting.localization.Localization(__file__, 374, 31), getitem___70557, slice_70552)
    
    # Getting the type of 'ret' (line 374)
    ret_70559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'ret')
    str_70560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 16), 'str', 'cbsetdims')
    # Storing an element on a container (line 374)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 12), ret_70559, (str_70560, subscript_call_result_70558))
    # SSA join for if statement (line 373)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 329)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 325)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 321)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ret' (line 377)
    ret_70561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 11), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'stypy_return_type', ret_70561)
    
    # ################# End of 'getarrdims(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getarrdims' in the type store
    # Getting the type of 'stypy_return_type' (line 318)
    stypy_return_type_70562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_70562)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getarrdims'
    return stypy_return_type_70562

# Assigning a type to the variable 'getarrdims' (line 318)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 0), 'getarrdims', getarrdims)

@norecursion
def getpydocsign(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getpydocsign'
    module_type_store = module_type_store.open_function_context('getpydocsign', 380, 0, False)
    
    # Passed parameters checking function
    getpydocsign.stypy_localization = localization
    getpydocsign.stypy_type_of_self = None
    getpydocsign.stypy_type_store = module_type_store
    getpydocsign.stypy_function_name = 'getpydocsign'
    getpydocsign.stypy_param_names_list = ['a', 'var']
    getpydocsign.stypy_varargs_param_name = None
    getpydocsign.stypy_kwargs_param_name = None
    getpydocsign.stypy_call_defaults = defaults
    getpydocsign.stypy_call_varargs = varargs
    getpydocsign.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getpydocsign', ['a', 'var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getpydocsign', localization, ['a', 'var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getpydocsign(...)' code ##################

    # Marking variables as global (line 381)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 381, 4), 'lcb_map')
    
    
    # Call to isfunction(...): (line 382)
    # Processing the call arguments (line 382)
    # Getting the type of 'var' (line 382)
    var_70564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 18), 'var', False)
    # Processing the call keyword arguments (line 382)
    kwargs_70565 = {}
    # Getting the type of 'isfunction' (line 382)
    isfunction_70563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 7), 'isfunction', False)
    # Calling isfunction(args, kwargs) (line 382)
    isfunction_call_result_70566 = invoke(stypy.reporting.localization.Localization(__file__, 382, 7), isfunction_70563, *[var_70564], **kwargs_70565)
    
    # Testing the type of an if condition (line 382)
    if_condition_70567 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 382, 4), isfunction_call_result_70566)
    # Assigning a type to the variable 'if_condition_70567' (line 382)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'if_condition_70567', if_condition_70567)
    # SSA begins for if statement (line 382)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    str_70568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 11), 'str', 'result')
    # Getting the type of 'var' (line 383)
    var_70569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 23), 'var')
    # Applying the binary operator 'in' (line 383)
    result_contains_70570 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 11), 'in', str_70568, var_70569)
    
    # Testing the type of an if condition (line 383)
    if_condition_70571 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 383, 8), result_contains_70570)
    # Assigning a type to the variable 'if_condition_70571' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'if_condition_70571', if_condition_70571)
    # SSA begins for if statement (line 383)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 384):
    
    # Assigning a Subscript to a Name (line 384):
    
    # Obtaining the type of the subscript
    str_70572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 21), 'str', 'result')
    # Getting the type of 'var' (line 384)
    var_70573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 17), 'var')
    # Obtaining the member '__getitem__' of a type (line 384)
    getitem___70574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 17), var_70573, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 384)
    subscript_call_result_70575 = invoke(stypy.reporting.localization.Localization(__file__, 384, 17), getitem___70574, str_70572)
    
    # Assigning a type to the variable 'af' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'af', subscript_call_result_70575)
    # SSA branch for the else part of an if statement (line 383)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 386):
    
    # Assigning a Subscript to a Name (line 386):
    
    # Obtaining the type of the subscript
    str_70576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 21), 'str', 'name')
    # Getting the type of 'var' (line 386)
    var_70577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 17), 'var')
    # Obtaining the member '__getitem__' of a type (line 386)
    getitem___70578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 17), var_70577, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
    subscript_call_result_70579 = invoke(stypy.reporting.localization.Localization(__file__, 386, 17), getitem___70578, str_70576)
    
    # Assigning a type to the variable 'af' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'af', subscript_call_result_70579)
    # SSA join for if statement (line 383)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'af' (line 387)
    af_70580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 11), 'af')
    
    # Obtaining the type of the subscript
    str_70581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 21), 'str', 'vars')
    # Getting the type of 'var' (line 387)
    var_70582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 17), 'var')
    # Obtaining the member '__getitem__' of a type (line 387)
    getitem___70583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 17), var_70582, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 387)
    subscript_call_result_70584 = invoke(stypy.reporting.localization.Localization(__file__, 387, 17), getitem___70583, str_70581)
    
    # Applying the binary operator 'in' (line 387)
    result_contains_70585 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 11), 'in', af_70580, subscript_call_result_70584)
    
    # Testing the type of an if condition (line 387)
    if_condition_70586 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 387, 8), result_contains_70585)
    # Assigning a type to the variable 'if_condition_70586' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'if_condition_70586', if_condition_70586)
    # SSA begins for if statement (line 387)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to getpydocsign(...): (line 388)
    # Processing the call arguments (line 388)
    # Getting the type of 'af' (line 388)
    af_70588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 32), 'af', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'af' (line 388)
    af_70589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 48), 'af', False)
    
    # Obtaining the type of the subscript
    str_70590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 40), 'str', 'vars')
    # Getting the type of 'var' (line 388)
    var_70591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 36), 'var', False)
    # Obtaining the member '__getitem__' of a type (line 388)
    getitem___70592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 36), var_70591, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 388)
    subscript_call_result_70593 = invoke(stypy.reporting.localization.Localization(__file__, 388, 36), getitem___70592, str_70590)
    
    # Obtaining the member '__getitem__' of a type (line 388)
    getitem___70594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 36), subscript_call_result_70593, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 388)
    subscript_call_result_70595 = invoke(stypy.reporting.localization.Localization(__file__, 388, 36), getitem___70594, af_70589)
    
    # Processing the call keyword arguments (line 388)
    kwargs_70596 = {}
    # Getting the type of 'getpydocsign' (line 388)
    getpydocsign_70587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 19), 'getpydocsign', False)
    # Calling getpydocsign(args, kwargs) (line 388)
    getpydocsign_call_result_70597 = invoke(stypy.reporting.localization.Localization(__file__, 388, 19), getpydocsign_70587, *[af_70588, subscript_call_result_70595], **kwargs_70596)
    
    # Assigning a type to the variable 'stypy_return_type' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'stypy_return_type', getpydocsign_call_result_70597)
    # SSA branch for the else part of an if statement (line 387)
    module_type_store.open_ssa_branch('else')
    
    # Call to errmess(...): (line 390)
    # Processing the call arguments (line 390)
    str_70599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 20), 'str', 'getctype: function %s has no return value?!\n')
    # Getting the type of 'af' (line 390)
    af_70600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 70), 'af', False)
    # Applying the binary operator '%' (line 390)
    result_mod_70601 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 20), '%', str_70599, af_70600)
    
    # Processing the call keyword arguments (line 390)
    kwargs_70602 = {}
    # Getting the type of 'errmess' (line 390)
    errmess_70598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'errmess', False)
    # Calling errmess(args, kwargs) (line 390)
    errmess_call_result_70603 = invoke(stypy.reporting.localization.Localization(__file__, 390, 12), errmess_70598, *[result_mod_70601], **kwargs_70602)
    
    # SSA join for if statement (line 387)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 391)
    tuple_70604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 391)
    # Adding element type (line 391)
    str_70605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 15), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 15), tuple_70604, str_70605)
    # Adding element type (line 391)
    str_70606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 19), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 15), tuple_70604, str_70606)
    
    # Assigning a type to the variable 'stypy_return_type' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'stypy_return_type', tuple_70604)
    # SSA join for if statement (line 382)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Tuple (line 392):
    
    # Assigning a Name to a Name (line 392):
    # Getting the type of 'a' (line 392)
    a_70607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 18), 'a')
    # Assigning a type to the variable 'tuple_assignment_69347' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'tuple_assignment_69347', a_70607)
    
    # Assigning a Name to a Name (line 392):
    # Getting the type of 'a' (line 392)
    a_70608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 21), 'a')
    # Assigning a type to the variable 'tuple_assignment_69348' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'tuple_assignment_69348', a_70608)
    
    # Assigning a Name to a Name (line 392):
    # Getting the type of 'tuple_assignment_69347' (line 392)
    tuple_assignment_69347_70609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'tuple_assignment_69347')
    # Assigning a type to the variable 'sig' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'sig', tuple_assignment_69347_70609)
    
    # Assigning a Name to a Name (line 392):
    # Getting the type of 'tuple_assignment_69348' (line 392)
    tuple_assignment_69348_70610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'tuple_assignment_69348')
    # Assigning a type to the variable 'sigout' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 9), 'sigout', tuple_assignment_69348_70610)
    
    # Assigning a Str to a Name (line 393):
    
    # Assigning a Str to a Name (line 393):
    str_70611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 10), 'str', '')
    # Assigning a type to the variable 'opt' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'opt', str_70611)
    
    
    # Call to isintent_in(...): (line 394)
    # Processing the call arguments (line 394)
    # Getting the type of 'var' (line 394)
    var_70613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 19), 'var', False)
    # Processing the call keyword arguments (line 394)
    kwargs_70614 = {}
    # Getting the type of 'isintent_in' (line 394)
    isintent_in_70612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 7), 'isintent_in', False)
    # Calling isintent_in(args, kwargs) (line 394)
    isintent_in_call_result_70615 = invoke(stypy.reporting.localization.Localization(__file__, 394, 7), isintent_in_70612, *[var_70613], **kwargs_70614)
    
    # Testing the type of an if condition (line 394)
    if_condition_70616 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 394, 4), isintent_in_call_result_70615)
    # Assigning a type to the variable 'if_condition_70616' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'if_condition_70616', if_condition_70616)
    # SSA begins for if statement (line 394)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 395):
    
    # Assigning a Str to a Name (line 395):
    str_70617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 14), 'str', 'input')
    # Assigning a type to the variable 'opt' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'opt', str_70617)
    # SSA branch for the else part of an if statement (line 394)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isintent_inout(...): (line 396)
    # Processing the call arguments (line 396)
    # Getting the type of 'var' (line 396)
    var_70619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 24), 'var', False)
    # Processing the call keyword arguments (line 396)
    kwargs_70620 = {}
    # Getting the type of 'isintent_inout' (line 396)
    isintent_inout_70618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 9), 'isintent_inout', False)
    # Calling isintent_inout(args, kwargs) (line 396)
    isintent_inout_call_result_70621 = invoke(stypy.reporting.localization.Localization(__file__, 396, 9), isintent_inout_70618, *[var_70619], **kwargs_70620)
    
    # Testing the type of an if condition (line 396)
    if_condition_70622 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 396, 9), isintent_inout_call_result_70621)
    # Assigning a type to the variable 'if_condition_70622' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 9), 'if_condition_70622', if_condition_70622)
    # SSA begins for if statement (line 396)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 397):
    
    # Assigning a Str to a Name (line 397):
    str_70623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 14), 'str', 'in/output')
    # Assigning a type to the variable 'opt' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'opt', str_70623)
    # SSA join for if statement (line 396)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 394)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 398):
    
    # Assigning a Name to a Name (line 398):
    # Getting the type of 'a' (line 398)
    a_70624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'a')
    # Assigning a type to the variable 'out_a' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'out_a', a_70624)
    
    
    # Call to isintent_out(...): (line 399)
    # Processing the call arguments (line 399)
    # Getting the type of 'var' (line 399)
    var_70626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 20), 'var', False)
    # Processing the call keyword arguments (line 399)
    kwargs_70627 = {}
    # Getting the type of 'isintent_out' (line 399)
    isintent_out_70625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 7), 'isintent_out', False)
    # Calling isintent_out(args, kwargs) (line 399)
    isintent_out_call_result_70628 = invoke(stypy.reporting.localization.Localization(__file__, 399, 7), isintent_out_70625, *[var_70626], **kwargs_70627)
    
    # Testing the type of an if condition (line 399)
    if_condition_70629 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 399, 4), isintent_out_call_result_70628)
    # Assigning a type to the variable 'if_condition_70629' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'if_condition_70629', if_condition_70629)
    # SSA begins for if statement (line 399)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Obtaining the type of the subscript
    str_70630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 21), 'str', 'intent')
    # Getting the type of 'var' (line 400)
    var_70631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 17), 'var')
    # Obtaining the member '__getitem__' of a type (line 400)
    getitem___70632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 17), var_70631, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 400)
    subscript_call_result_70633 = invoke(stypy.reporting.localization.Localization(__file__, 400, 17), getitem___70632, str_70630)
    
    # Testing the type of a for loop iterable (line 400)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 400, 8), subscript_call_result_70633)
    # Getting the type of the for loop variable (line 400)
    for_loop_var_70634 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 400, 8), subscript_call_result_70633)
    # Assigning a type to the variable 'k' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'k', for_loop_var_70634)
    # SSA begins for a for statement (line 400)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    int_70635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 18), 'int')
    slice_70636 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 401, 15), None, int_70635, None)
    # Getting the type of 'k' (line 401)
    k_70637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 15), 'k')
    # Obtaining the member '__getitem__' of a type (line 401)
    getitem___70638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 15), k_70637, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 401)
    subscript_call_result_70639 = invoke(stypy.reporting.localization.Localization(__file__, 401, 15), getitem___70638, slice_70636)
    
    str_70640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 24), 'str', 'out=')
    # Applying the binary operator '==' (line 401)
    result_eq_70641 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 15), '==', subscript_call_result_70639, str_70640)
    
    # Testing the type of an if condition (line 401)
    if_condition_70642 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 401, 12), result_eq_70641)
    # Assigning a type to the variable 'if_condition_70642' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'if_condition_70642', if_condition_70642)
    # SSA begins for if statement (line 401)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 402):
    
    # Assigning a Subscript to a Name (line 402):
    
    # Obtaining the type of the subscript
    int_70643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 26), 'int')
    slice_70644 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 402, 24), int_70643, None, None)
    # Getting the type of 'k' (line 402)
    k_70645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 24), 'k')
    # Obtaining the member '__getitem__' of a type (line 402)
    getitem___70646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 24), k_70645, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 402)
    subscript_call_result_70647 = invoke(stypy.reporting.localization.Localization(__file__, 402, 24), getitem___70646, slice_70644)
    
    # Assigning a type to the variable 'out_a' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 16), 'out_a', subscript_call_result_70647)
    # SSA join for if statement (line 401)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 399)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Str to a Name (line 404):
    
    # Assigning a Str to a Name (line 404):
    str_70648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 11), 'str', '')
    # Assigning a type to the variable 'init' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'init', str_70648)
    
    # Assigning a Call to a Name (line 405):
    
    # Assigning a Call to a Name (line 405):
    
    # Call to getctype(...): (line 405)
    # Processing the call arguments (line 405)
    # Getting the type of 'var' (line 405)
    var_70650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 21), 'var', False)
    # Processing the call keyword arguments (line 405)
    kwargs_70651 = {}
    # Getting the type of 'getctype' (line 405)
    getctype_70649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'getctype', False)
    # Calling getctype(args, kwargs) (line 405)
    getctype_call_result_70652 = invoke(stypy.reporting.localization.Localization(__file__, 405, 12), getctype_70649, *[var_70650], **kwargs_70651)
    
    # Assigning a type to the variable 'ctype' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'ctype', getctype_call_result_70652)
    
    
    # Call to hasinitvalue(...): (line 407)
    # Processing the call arguments (line 407)
    # Getting the type of 'var' (line 407)
    var_70654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 20), 'var', False)
    # Processing the call keyword arguments (line 407)
    kwargs_70655 = {}
    # Getting the type of 'hasinitvalue' (line 407)
    hasinitvalue_70653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 7), 'hasinitvalue', False)
    # Calling hasinitvalue(args, kwargs) (line 407)
    hasinitvalue_call_result_70656 = invoke(stypy.reporting.localization.Localization(__file__, 407, 7), hasinitvalue_70653, *[var_70654], **kwargs_70655)
    
    # Testing the type of an if condition (line 407)
    if_condition_70657 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 407, 4), hasinitvalue_call_result_70656)
    # Assigning a type to the variable 'if_condition_70657' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'if_condition_70657', if_condition_70657)
    # SSA begins for if statement (line 407)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 408):
    
    # Assigning a Call to a Name:
    
    # Call to getinit(...): (line 408)
    # Processing the call arguments (line 408)
    # Getting the type of 'a' (line 408)
    a_70659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 33), 'a', False)
    # Getting the type of 'var' (line 408)
    var_70660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 36), 'var', False)
    # Processing the call keyword arguments (line 408)
    kwargs_70661 = {}
    # Getting the type of 'getinit' (line 408)
    getinit_70658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 25), 'getinit', False)
    # Calling getinit(args, kwargs) (line 408)
    getinit_call_result_70662 = invoke(stypy.reporting.localization.Localization(__file__, 408, 25), getinit_70658, *[a_70659, var_70660], **kwargs_70661)
    
    # Assigning a type to the variable 'call_assignment_69349' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'call_assignment_69349', getinit_call_result_70662)
    
    # Assigning a Call to a Name (line 408):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_70665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 8), 'int')
    # Processing the call keyword arguments
    kwargs_70666 = {}
    # Getting the type of 'call_assignment_69349' (line 408)
    call_assignment_69349_70663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'call_assignment_69349', False)
    # Obtaining the member '__getitem__' of a type (line 408)
    getitem___70664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 8), call_assignment_69349_70663, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_70667 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___70664, *[int_70665], **kwargs_70666)
    
    # Assigning a type to the variable 'call_assignment_69350' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'call_assignment_69350', getitem___call_result_70667)
    
    # Assigning a Name to a Name (line 408):
    # Getting the type of 'call_assignment_69350' (line 408)
    call_assignment_69350_70668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'call_assignment_69350')
    # Assigning a type to the variable 'init' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'init', call_assignment_69350_70668)
    
    # Assigning a Call to a Name (line 408):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_70671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 8), 'int')
    # Processing the call keyword arguments
    kwargs_70672 = {}
    # Getting the type of 'call_assignment_69349' (line 408)
    call_assignment_69349_70669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'call_assignment_69349', False)
    # Obtaining the member '__getitem__' of a type (line 408)
    getitem___70670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 8), call_assignment_69349_70669, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_70673 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___70670, *[int_70671], **kwargs_70672)
    
    # Assigning a type to the variable 'call_assignment_69351' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'call_assignment_69351', getitem___call_result_70673)
    
    # Assigning a Name to a Name (line 408):
    # Getting the type of 'call_assignment_69351' (line 408)
    call_assignment_69351_70674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'call_assignment_69351')
    # Assigning a type to the variable 'showinit' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 14), 'showinit', call_assignment_69351_70674)
    
    # Assigning a BinOp to a Name (line 409):
    
    # Assigning a BinOp to a Name (line 409):
    str_70675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 15), 'str', ', optional\\n    Default: %s')
    # Getting the type of 'showinit' (line 409)
    showinit_70676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 48), 'showinit')
    # Applying the binary operator '%' (line 409)
    result_mod_70677 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 15), '%', str_70675, showinit_70676)
    
    # Assigning a type to the variable 'init' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'init', result_mod_70677)
    # SSA join for if statement (line 407)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isscalar(...): (line 410)
    # Processing the call arguments (line 410)
    # Getting the type of 'var' (line 410)
    var_70679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 16), 'var', False)
    # Processing the call keyword arguments (line 410)
    kwargs_70680 = {}
    # Getting the type of 'isscalar' (line 410)
    isscalar_70678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 7), 'isscalar', False)
    # Calling isscalar(args, kwargs) (line 410)
    isscalar_call_result_70681 = invoke(stypy.reporting.localization.Localization(__file__, 410, 7), isscalar_70678, *[var_70679], **kwargs_70680)
    
    # Testing the type of an if condition (line 410)
    if_condition_70682 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 410, 4), isscalar_call_result_70681)
    # Assigning a type to the variable 'if_condition_70682' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'if_condition_70682', if_condition_70682)
    # SSA begins for if statement (line 410)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to isintent_inout(...): (line 411)
    # Processing the call arguments (line 411)
    # Getting the type of 'var' (line 411)
    var_70684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 26), 'var', False)
    # Processing the call keyword arguments (line 411)
    kwargs_70685 = {}
    # Getting the type of 'isintent_inout' (line 411)
    isintent_inout_70683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 11), 'isintent_inout', False)
    # Calling isintent_inout(args, kwargs) (line 411)
    isintent_inout_call_result_70686 = invoke(stypy.reporting.localization.Localization(__file__, 411, 11), isintent_inout_70683, *[var_70684], **kwargs_70685)
    
    # Testing the type of an if condition (line 411)
    if_condition_70687 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 411, 8), isintent_inout_call_result_70686)
    # Assigning a type to the variable 'if_condition_70687' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'if_condition_70687', if_condition_70687)
    # SSA begins for if statement (line 411)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 412):
    
    # Assigning a BinOp to a Name (line 412):
    str_70688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 18), 'str', "%s : %s rank-0 array(%s,'%s')%s")
    
    # Obtaining an instance of the builtin type 'tuple' (line 412)
    tuple_70689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 57), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 412)
    # Adding element type (line 412)
    # Getting the type of 'a' (line 412)
    a_70690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 57), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 57), tuple_70689, a_70690)
    # Adding element type (line 412)
    # Getting the type of 'opt' (line 412)
    opt_70691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 60), 'opt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 57), tuple_70689, opt_70691)
    # Adding element type (line 412)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ctype' (line 412)
    ctype_70692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 74), 'ctype')
    # Getting the type of 'c2py_map' (line 412)
    c2py_map_70693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 65), 'c2py_map')
    # Obtaining the member '__getitem__' of a type (line 412)
    getitem___70694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 65), c2py_map_70693, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 412)
    subscript_call_result_70695 = invoke(stypy.reporting.localization.Localization(__file__, 412, 65), getitem___70694, ctype_70692)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 57), tuple_70689, subscript_call_result_70695)
    # Adding element type (line 412)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ctype' (line 413)
    ctype_70696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 70), 'ctype')
    # Getting the type of 'c2pycode_map' (line 413)
    c2pycode_map_70697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 57), 'c2pycode_map')
    # Obtaining the member '__getitem__' of a type (line 413)
    getitem___70698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 57), c2pycode_map_70697, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 413)
    subscript_call_result_70699 = invoke(stypy.reporting.localization.Localization(__file__, 413, 57), getitem___70698, ctype_70696)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 57), tuple_70689, subscript_call_result_70699)
    # Adding element type (line 412)
    # Getting the type of 'init' (line 413)
    init_70700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 78), 'init')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 57), tuple_70689, init_70700)
    
    # Applying the binary operator '%' (line 412)
    result_mod_70701 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 18), '%', str_70688, tuple_70689)
    
    # Assigning a type to the variable 'sig' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 12), 'sig', result_mod_70701)
    # SSA branch for the else part of an if statement (line 411)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 415):
    
    # Assigning a BinOp to a Name (line 415):
    str_70702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 18), 'str', '%s : %s %s%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 415)
    tuple_70703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 415)
    # Adding element type (line 415)
    # Getting the type of 'a' (line 415)
    a_70704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 36), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 36), tuple_70703, a_70704)
    # Adding element type (line 415)
    # Getting the type of 'opt' (line 415)
    opt_70705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 39), 'opt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 36), tuple_70703, opt_70705)
    # Adding element type (line 415)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ctype' (line 415)
    ctype_70706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 53), 'ctype')
    # Getting the type of 'c2py_map' (line 415)
    c2py_map_70707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 44), 'c2py_map')
    # Obtaining the member '__getitem__' of a type (line 415)
    getitem___70708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 44), c2py_map_70707, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 415)
    subscript_call_result_70709 = invoke(stypy.reporting.localization.Localization(__file__, 415, 44), getitem___70708, ctype_70706)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 36), tuple_70703, subscript_call_result_70709)
    # Adding element type (line 415)
    # Getting the type of 'init' (line 415)
    init_70710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 61), 'init')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 36), tuple_70703, init_70710)
    
    # Applying the binary operator '%' (line 415)
    result_mod_70711 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 18), '%', str_70702, tuple_70703)
    
    # Assigning a type to the variable 'sig' (line 415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 12), 'sig', result_mod_70711)
    # SSA join for if statement (line 411)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 416):
    
    # Assigning a BinOp to a Name (line 416):
    str_70712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 17), 'str', '%s : %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 416)
    tuple_70713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 416)
    # Adding element type (line 416)
    # Getting the type of 'out_a' (line 416)
    out_a_70714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 30), 'out_a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 30), tuple_70713, out_a_70714)
    # Adding element type (line 416)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ctype' (line 416)
    ctype_70715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 46), 'ctype')
    # Getting the type of 'c2py_map' (line 416)
    c2py_map_70716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 37), 'c2py_map')
    # Obtaining the member '__getitem__' of a type (line 416)
    getitem___70717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 37), c2py_map_70716, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 416)
    subscript_call_result_70718 = invoke(stypy.reporting.localization.Localization(__file__, 416, 37), getitem___70717, ctype_70715)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 30), tuple_70713, subscript_call_result_70718)
    
    # Applying the binary operator '%' (line 416)
    result_mod_70719 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 17), '%', str_70712, tuple_70713)
    
    # Assigning a type to the variable 'sigout' (line 416)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'sigout', result_mod_70719)
    # SSA branch for the else part of an if statement (line 410)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isstring(...): (line 417)
    # Processing the call arguments (line 417)
    # Getting the type of 'var' (line 417)
    var_70721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 18), 'var', False)
    # Processing the call keyword arguments (line 417)
    kwargs_70722 = {}
    # Getting the type of 'isstring' (line 417)
    isstring_70720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 9), 'isstring', False)
    # Calling isstring(args, kwargs) (line 417)
    isstring_call_result_70723 = invoke(stypy.reporting.localization.Localization(__file__, 417, 9), isstring_70720, *[var_70721], **kwargs_70722)
    
    # Testing the type of an if condition (line 417)
    if_condition_70724 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 417, 9), isstring_call_result_70723)
    # Assigning a type to the variable 'if_condition_70724' (line 417)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 9), 'if_condition_70724', if_condition_70724)
    # SSA begins for if statement (line 417)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to isintent_inout(...): (line 418)
    # Processing the call arguments (line 418)
    # Getting the type of 'var' (line 418)
    var_70726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 26), 'var', False)
    # Processing the call keyword arguments (line 418)
    kwargs_70727 = {}
    # Getting the type of 'isintent_inout' (line 418)
    isintent_inout_70725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 11), 'isintent_inout', False)
    # Calling isintent_inout(args, kwargs) (line 418)
    isintent_inout_call_result_70728 = invoke(stypy.reporting.localization.Localization(__file__, 418, 11), isintent_inout_70725, *[var_70726], **kwargs_70727)
    
    # Testing the type of an if condition (line 418)
    if_condition_70729 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 418, 8), isintent_inout_call_result_70728)
    # Assigning a type to the variable 'if_condition_70729' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'if_condition_70729', if_condition_70729)
    # SSA begins for if statement (line 418)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 419):
    
    # Assigning a BinOp to a Name (line 419):
    str_70730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 18), 'str', "%s : %s rank-0 array(string(len=%s),'c')%s")
    
    # Obtaining an instance of the builtin type 'tuple' (line 420)
    tuple_70731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 420)
    # Adding element type (line 420)
    # Getting the type of 'a' (line 420)
    a_70732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 16), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 16), tuple_70731, a_70732)
    # Adding element type (line 420)
    # Getting the type of 'opt' (line 420)
    opt_70733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 19), 'opt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 16), tuple_70731, opt_70733)
    # Adding element type (line 420)
    
    # Call to getstrlength(...): (line 420)
    # Processing the call arguments (line 420)
    # Getting the type of 'var' (line 420)
    var_70735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 37), 'var', False)
    # Processing the call keyword arguments (line 420)
    kwargs_70736 = {}
    # Getting the type of 'getstrlength' (line 420)
    getstrlength_70734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 24), 'getstrlength', False)
    # Calling getstrlength(args, kwargs) (line 420)
    getstrlength_call_result_70737 = invoke(stypy.reporting.localization.Localization(__file__, 420, 24), getstrlength_70734, *[var_70735], **kwargs_70736)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 16), tuple_70731, getstrlength_call_result_70737)
    # Adding element type (line 420)
    # Getting the type of 'init' (line 420)
    init_70738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 43), 'init')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 16), tuple_70731, init_70738)
    
    # Applying the binary operator '%' (line 419)
    result_mod_70739 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 18), '%', str_70730, tuple_70731)
    
    # Assigning a type to the variable 'sig' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 12), 'sig', result_mod_70739)
    # SSA branch for the else part of an if statement (line 418)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 422):
    
    # Assigning a BinOp to a Name (line 422):
    str_70740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 18), 'str', '%s : %s string(len=%s)%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 423)
    tuple_70741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 423)
    # Adding element type (line 423)
    # Getting the type of 'a' (line 423)
    a_70742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 16), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 16), tuple_70741, a_70742)
    # Adding element type (line 423)
    # Getting the type of 'opt' (line 423)
    opt_70743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 19), 'opt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 16), tuple_70741, opt_70743)
    # Adding element type (line 423)
    
    # Call to getstrlength(...): (line 423)
    # Processing the call arguments (line 423)
    # Getting the type of 'var' (line 423)
    var_70745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 37), 'var', False)
    # Processing the call keyword arguments (line 423)
    kwargs_70746 = {}
    # Getting the type of 'getstrlength' (line 423)
    getstrlength_70744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 24), 'getstrlength', False)
    # Calling getstrlength(args, kwargs) (line 423)
    getstrlength_call_result_70747 = invoke(stypy.reporting.localization.Localization(__file__, 423, 24), getstrlength_70744, *[var_70745], **kwargs_70746)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 16), tuple_70741, getstrlength_call_result_70747)
    # Adding element type (line 423)
    # Getting the type of 'init' (line 423)
    init_70748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 43), 'init')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 16), tuple_70741, init_70748)
    
    # Applying the binary operator '%' (line 422)
    result_mod_70749 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 18), '%', str_70740, tuple_70741)
    
    # Assigning a type to the variable 'sig' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 12), 'sig', result_mod_70749)
    # SSA join for if statement (line 418)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 424):
    
    # Assigning a BinOp to a Name (line 424):
    str_70750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 17), 'str', '%s : string(len=%s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 424)
    tuple_70751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 424)
    # Adding element type (line 424)
    # Getting the type of 'out_a' (line 424)
    out_a_70752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 42), 'out_a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 42), tuple_70751, out_a_70752)
    # Adding element type (line 424)
    
    # Call to getstrlength(...): (line 424)
    # Processing the call arguments (line 424)
    # Getting the type of 'var' (line 424)
    var_70754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 62), 'var', False)
    # Processing the call keyword arguments (line 424)
    kwargs_70755 = {}
    # Getting the type of 'getstrlength' (line 424)
    getstrlength_70753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 49), 'getstrlength', False)
    # Calling getstrlength(args, kwargs) (line 424)
    getstrlength_call_result_70756 = invoke(stypy.reporting.localization.Localization(__file__, 424, 49), getstrlength_70753, *[var_70754], **kwargs_70755)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 42), tuple_70751, getstrlength_call_result_70756)
    
    # Applying the binary operator '%' (line 424)
    result_mod_70757 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 17), '%', str_70750, tuple_70751)
    
    # Assigning a type to the variable 'sigout' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'sigout', result_mod_70757)
    # SSA branch for the else part of an if statement (line 417)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isarray(...): (line 425)
    # Processing the call arguments (line 425)
    # Getting the type of 'var' (line 425)
    var_70759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 17), 'var', False)
    # Processing the call keyword arguments (line 425)
    kwargs_70760 = {}
    # Getting the type of 'isarray' (line 425)
    isarray_70758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 9), 'isarray', False)
    # Calling isarray(args, kwargs) (line 425)
    isarray_call_result_70761 = invoke(stypy.reporting.localization.Localization(__file__, 425, 9), isarray_70758, *[var_70759], **kwargs_70760)
    
    # Testing the type of an if condition (line 425)
    if_condition_70762 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 425, 9), isarray_call_result_70761)
    # Assigning a type to the variable 'if_condition_70762' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 9), 'if_condition_70762', if_condition_70762)
    # SSA begins for if statement (line 425)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 426):
    
    # Assigning a Subscript to a Name (line 426):
    
    # Obtaining the type of the subscript
    str_70763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 18), 'str', 'dimension')
    # Getting the type of 'var' (line 426)
    var_70764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 14), 'var')
    # Obtaining the member '__getitem__' of a type (line 426)
    getitem___70765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 14), var_70764, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 426)
    subscript_call_result_70766 = invoke(stypy.reporting.localization.Localization(__file__, 426, 14), getitem___70765, str_70763)
    
    # Assigning a type to the variable 'dim' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'dim', subscript_call_result_70766)
    
    # Assigning a Call to a Name (line 427):
    
    # Assigning a Call to a Name (line 427):
    
    # Call to repr(...): (line 427)
    # Processing the call arguments (line 427)
    
    # Call to len(...): (line 427)
    # Processing the call arguments (line 427)
    # Getting the type of 'dim' (line 427)
    dim_70769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 24), 'dim', False)
    # Processing the call keyword arguments (line 427)
    kwargs_70770 = {}
    # Getting the type of 'len' (line 427)
    len_70768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 20), 'len', False)
    # Calling len(args, kwargs) (line 427)
    len_call_result_70771 = invoke(stypy.reporting.localization.Localization(__file__, 427, 20), len_70768, *[dim_70769], **kwargs_70770)
    
    # Processing the call keyword arguments (line 427)
    kwargs_70772 = {}
    # Getting the type of 'repr' (line 427)
    repr_70767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 15), 'repr', False)
    # Calling repr(args, kwargs) (line 427)
    repr_call_result_70773 = invoke(stypy.reporting.localization.Localization(__file__, 427, 15), repr_70767, *[len_call_result_70771], **kwargs_70772)
    
    # Assigning a type to the variable 'rank' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'rank', repr_call_result_70773)
    
    # Assigning a BinOp to a Name (line 428):
    
    # Assigning a BinOp to a Name (line 428):
    str_70774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 14), 'str', "%s : %s rank-%s array('%s') with bounds (%s)%s")
    
    # Obtaining an instance of the builtin type 'tuple' (line 428)
    tuple_70775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 68), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 428)
    # Adding element type (line 428)
    # Getting the type of 'a' (line 428)
    a_70776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 68), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 68), tuple_70775, a_70776)
    # Adding element type (line 428)
    # Getting the type of 'opt' (line 428)
    opt_70777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 71), 'opt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 68), tuple_70775, opt_70777)
    # Adding element type (line 428)
    # Getting the type of 'rank' (line 428)
    rank_70778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 76), 'rank')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 68), tuple_70775, rank_70778)
    # Adding element type (line 428)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ctype' (line 430)
    ctype_70779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 72), 'ctype')
    # Getting the type of 'c2pycode_map' (line 429)
    c2pycode_map_70780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 68), 'c2pycode_map')
    # Obtaining the member '__getitem__' of a type (line 429)
    getitem___70781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 68), c2pycode_map_70780, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 429)
    subscript_call_result_70782 = invoke(stypy.reporting.localization.Localization(__file__, 429, 68), getitem___70781, ctype_70779)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 68), tuple_70775, subscript_call_result_70782)
    # Adding element type (line 428)
    
    # Call to join(...): (line 431)
    # Processing the call arguments (line 431)
    # Getting the type of 'dim' (line 431)
    dim_70785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 77), 'dim', False)
    # Processing the call keyword arguments (line 431)
    kwargs_70786 = {}
    str_70783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 68), 'str', ',')
    # Obtaining the member 'join' of a type (line 431)
    join_70784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 68), str_70783, 'join')
    # Calling join(args, kwargs) (line 431)
    join_call_result_70787 = invoke(stypy.reporting.localization.Localization(__file__, 431, 68), join_70784, *[dim_70785], **kwargs_70786)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 68), tuple_70775, join_call_result_70787)
    # Adding element type (line 428)
    # Getting the type of 'init' (line 431)
    init_70788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 83), 'init')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 68), tuple_70775, init_70788)
    
    # Applying the binary operator '%' (line 428)
    result_mod_70789 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 14), '%', str_70774, tuple_70775)
    
    # Assigning a type to the variable 'sig' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'sig', result_mod_70789)
    
    
    # Getting the type of 'a' (line 432)
    a_70790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 11), 'a')
    # Getting the type of 'out_a' (line 432)
    out_a_70791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 16), 'out_a')
    # Applying the binary operator '==' (line 432)
    result_eq_70792 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 11), '==', a_70790, out_a_70791)
    
    # Testing the type of an if condition (line 432)
    if_condition_70793 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 432, 8), result_eq_70792)
    # Assigning a type to the variable 'if_condition_70793' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'if_condition_70793', if_condition_70793)
    # SSA begins for if statement (line 432)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 433):
    
    # Assigning a BinOp to a Name (line 433):
    str_70794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 21), 'str', "%s : rank-%s array('%s') with bounds (%s)")
    
    # Obtaining an instance of the builtin type 'tuple' (line 434)
    tuple_70795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 434)
    # Adding element type (line 434)
    # Getting the type of 'a' (line 434)
    a_70796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 19), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 19), tuple_70795, a_70796)
    # Adding element type (line 434)
    # Getting the type of 'rank' (line 434)
    rank_70797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 22), 'rank')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 19), tuple_70795, rank_70797)
    # Adding element type (line 434)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ctype' (line 434)
    ctype_70798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 41), 'ctype')
    # Getting the type of 'c2pycode_map' (line 434)
    c2pycode_map_70799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 28), 'c2pycode_map')
    # Obtaining the member '__getitem__' of a type (line 434)
    getitem___70800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 28), c2pycode_map_70799, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 434)
    subscript_call_result_70801 = invoke(stypy.reporting.localization.Localization(__file__, 434, 28), getitem___70800, ctype_70798)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 19), tuple_70795, subscript_call_result_70801)
    # Adding element type (line 434)
    
    # Call to join(...): (line 434)
    # Processing the call arguments (line 434)
    # Getting the type of 'dim' (line 434)
    dim_70804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 58), 'dim', False)
    # Processing the call keyword arguments (line 434)
    kwargs_70805 = {}
    str_70802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 49), 'str', ',')
    # Obtaining the member 'join' of a type (line 434)
    join_70803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 49), str_70802, 'join')
    # Calling join(args, kwargs) (line 434)
    join_call_result_70806 = invoke(stypy.reporting.localization.Localization(__file__, 434, 49), join_70803, *[dim_70804], **kwargs_70805)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 19), tuple_70795, join_call_result_70806)
    
    # Applying the binary operator '%' (line 433)
    result_mod_70807 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 21), '%', str_70794, tuple_70795)
    
    # Assigning a type to the variable 'sigout' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 12), 'sigout', result_mod_70807)
    # SSA branch for the else part of an if statement (line 432)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 436):
    
    # Assigning a BinOp to a Name (line 436):
    str_70808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 21), 'str', "%s : rank-%s array('%s') with bounds (%s) and %s storage")
    
    # Obtaining an instance of the builtin type 'tuple' (line 437)
    tuple_70809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 437)
    # Adding element type (line 437)
    # Getting the type of 'out_a' (line 437)
    out_a_70810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 19), 'out_a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 19), tuple_70809, out_a_70810)
    # Adding element type (line 437)
    # Getting the type of 'rank' (line 437)
    rank_70811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 26), 'rank')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 19), tuple_70809, rank_70811)
    # Adding element type (line 437)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ctype' (line 437)
    ctype_70812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 45), 'ctype')
    # Getting the type of 'c2pycode_map' (line 437)
    c2pycode_map_70813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 32), 'c2pycode_map')
    # Obtaining the member '__getitem__' of a type (line 437)
    getitem___70814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 32), c2pycode_map_70813, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 437)
    subscript_call_result_70815 = invoke(stypy.reporting.localization.Localization(__file__, 437, 32), getitem___70814, ctype_70812)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 19), tuple_70809, subscript_call_result_70815)
    # Adding element type (line 437)
    
    # Call to join(...): (line 437)
    # Processing the call arguments (line 437)
    # Getting the type of 'dim' (line 437)
    dim_70818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 62), 'dim', False)
    # Processing the call keyword arguments (line 437)
    kwargs_70819 = {}
    str_70816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 53), 'str', ',')
    # Obtaining the member 'join' of a type (line 437)
    join_70817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 53), str_70816, 'join')
    # Calling join(args, kwargs) (line 437)
    join_call_result_70820 = invoke(stypy.reporting.localization.Localization(__file__, 437, 53), join_70817, *[dim_70818], **kwargs_70819)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 19), tuple_70809, join_call_result_70820)
    # Adding element type (line 437)
    # Getting the type of 'a' (line 437)
    a_70821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 68), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 19), tuple_70809, a_70821)
    
    # Applying the binary operator '%' (line 436)
    result_mod_70822 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 21), '%', str_70808, tuple_70809)
    
    # Assigning a type to the variable 'sigout' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'sigout', result_mod_70822)
    # SSA join for if statement (line 432)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 425)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isexternal(...): (line 438)
    # Processing the call arguments (line 438)
    # Getting the type of 'var' (line 438)
    var_70824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 20), 'var', False)
    # Processing the call keyword arguments (line 438)
    kwargs_70825 = {}
    # Getting the type of 'isexternal' (line 438)
    isexternal_70823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 9), 'isexternal', False)
    # Calling isexternal(args, kwargs) (line 438)
    isexternal_call_result_70826 = invoke(stypy.reporting.localization.Localization(__file__, 438, 9), isexternal_70823, *[var_70824], **kwargs_70825)
    
    # Testing the type of an if condition (line 438)
    if_condition_70827 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 438, 9), isexternal_call_result_70826)
    # Assigning a type to the variable 'if_condition_70827' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 9), 'if_condition_70827', if_condition_70827)
    # SSA begins for if statement (line 438)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 439):
    
    # Assigning a Str to a Name (line 439):
    str_70828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 13), 'str', '')
    # Assigning a type to the variable 'ua' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'ua', str_70828)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'a' (line 440)
    a_70829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 11), 'a')
    # Getting the type of 'lcb_map' (line 440)
    lcb_map_70830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 16), 'lcb_map')
    # Applying the binary operator 'in' (line 440)
    result_contains_70831 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 11), 'in', a_70829, lcb_map_70830)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 440)
    a_70832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 36), 'a')
    # Getting the type of 'lcb_map' (line 440)
    lcb_map_70833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 28), 'lcb_map')
    # Obtaining the member '__getitem__' of a type (line 440)
    getitem___70834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 28), lcb_map_70833, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 440)
    subscript_call_result_70835 = invoke(stypy.reporting.localization.Localization(__file__, 440, 28), getitem___70834, a_70832)
    
    # Getting the type of 'lcb2_map' (line 440)
    lcb2_map_70836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 42), 'lcb2_map')
    # Applying the binary operator 'in' (line 440)
    result_contains_70837 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 28), 'in', subscript_call_result_70835, lcb2_map_70836)
    
    # Applying the binary operator 'and' (line 440)
    result_and_keyword_70838 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 11), 'and', result_contains_70831, result_contains_70837)
    
    str_70839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 55), 'str', 'argname')
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 440)
    a_70840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 85), 'a')
    # Getting the type of 'lcb_map' (line 440)
    lcb_map_70841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 77), 'lcb_map')
    # Obtaining the member '__getitem__' of a type (line 440)
    getitem___70842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 77), lcb_map_70841, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 440)
    subscript_call_result_70843 = invoke(stypy.reporting.localization.Localization(__file__, 440, 77), getitem___70842, a_70840)
    
    # Getting the type of 'lcb2_map' (line 440)
    lcb2_map_70844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 68), 'lcb2_map')
    # Obtaining the member '__getitem__' of a type (line 440)
    getitem___70845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 68), lcb2_map_70844, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 440)
    subscript_call_result_70846 = invoke(stypy.reporting.localization.Localization(__file__, 440, 68), getitem___70845, subscript_call_result_70843)
    
    # Applying the binary operator 'in' (line 440)
    result_contains_70847 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 55), 'in', str_70839, subscript_call_result_70846)
    
    # Applying the binary operator 'and' (line 440)
    result_and_keyword_70848 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 11), 'and', result_and_keyword_70838, result_contains_70847)
    
    # Testing the type of an if condition (line 440)
    if_condition_70849 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 440, 8), result_and_keyword_70848)
    # Assigning a type to the variable 'if_condition_70849' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'if_condition_70849', if_condition_70849)
    # SSA begins for if statement (line 440)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 441):
    
    # Assigning a Subscript to a Name (line 441):
    
    # Obtaining the type of the subscript
    str_70850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 38), 'str', 'argname')
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 441)
    a_70851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 34), 'a')
    # Getting the type of 'lcb_map' (line 441)
    lcb_map_70852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 26), 'lcb_map')
    # Obtaining the member '__getitem__' of a type (line 441)
    getitem___70853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 26), lcb_map_70852, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 441)
    subscript_call_result_70854 = invoke(stypy.reporting.localization.Localization(__file__, 441, 26), getitem___70853, a_70851)
    
    # Getting the type of 'lcb2_map' (line 441)
    lcb2_map_70855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 17), 'lcb2_map')
    # Obtaining the member '__getitem__' of a type (line 441)
    getitem___70856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 17), lcb2_map_70855, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 441)
    subscript_call_result_70857 = invoke(stypy.reporting.localization.Localization(__file__, 441, 17), getitem___70856, subscript_call_result_70854)
    
    # Obtaining the member '__getitem__' of a type (line 441)
    getitem___70858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 17), subscript_call_result_70857, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 441)
    subscript_call_result_70859 = invoke(stypy.reporting.localization.Localization(__file__, 441, 17), getitem___70858, str_70850)
    
    # Assigning a type to the variable 'ua' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 'ua', subscript_call_result_70859)
    
    
    
    # Getting the type of 'ua' (line 442)
    ua_70860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 19), 'ua')
    # Getting the type of 'a' (line 442)
    a_70861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 25), 'a')
    # Applying the binary operator '==' (line 442)
    result_eq_70862 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 19), '==', ua_70860, a_70861)
    
    # Applying the 'not' unary operator (line 442)
    result_not__70863 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 15), 'not', result_eq_70862)
    
    # Testing the type of an if condition (line 442)
    if_condition_70864 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 442, 12), result_not__70863)
    # Assigning a type to the variable 'if_condition_70864' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'if_condition_70864', if_condition_70864)
    # SSA begins for if statement (line 442)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 443):
    
    # Assigning a BinOp to a Name (line 443):
    str_70865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 21), 'str', ' => %s')
    # Getting the type of 'ua' (line 443)
    ua_70866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 32), 'ua')
    # Applying the binary operator '%' (line 443)
    result_mod_70867 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 21), '%', str_70865, ua_70866)
    
    # Assigning a type to the variable 'ua' (line 443)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 16), 'ua', result_mod_70867)
    # SSA branch for the else part of an if statement (line 442)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 445):
    
    # Assigning a Str to a Name (line 445):
    str_70868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 21), 'str', '')
    # Assigning a type to the variable 'ua' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 16), 'ua', str_70868)
    # SSA join for if statement (line 442)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 440)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 446):
    
    # Assigning a BinOp to a Name (line 446):
    str_70869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 14), 'str', '%s : call-back function%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 446)
    tuple_70870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 45), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 446)
    # Adding element type (line 446)
    # Getting the type of 'a' (line 446)
    a_70871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 45), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 45), tuple_70870, a_70871)
    # Adding element type (line 446)
    # Getting the type of 'ua' (line 446)
    ua_70872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 48), 'ua')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 45), tuple_70870, ua_70872)
    
    # Applying the binary operator '%' (line 446)
    result_mod_70873 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 14), '%', str_70869, tuple_70870)
    
    # Assigning a type to the variable 'sig' (line 446)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'sig', result_mod_70873)
    
    # Assigning a Name to a Name (line 447):
    
    # Assigning a Name to a Name (line 447):
    # Getting the type of 'sig' (line 447)
    sig_70874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 17), 'sig')
    # Assigning a type to the variable 'sigout' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'sigout', sig_70874)
    # SSA branch for the else part of an if statement (line 438)
    module_type_store.open_ssa_branch('else')
    
    # Call to errmess(...): (line 449)
    # Processing the call arguments (line 449)
    str_70876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 12), 'str', 'getpydocsign: Could not resolve docsignature for "%s".\\n')
    # Getting the type of 'a' (line 450)
    a_70877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 74), 'a', False)
    # Applying the binary operator '%' (line 450)
    result_mod_70878 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 12), '%', str_70876, a_70877)
    
    # Processing the call keyword arguments (line 449)
    kwargs_70879 = {}
    # Getting the type of 'errmess' (line 449)
    errmess_70875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'errmess', False)
    # Calling errmess(args, kwargs) (line 449)
    errmess_call_result_70880 = invoke(stypy.reporting.localization.Localization(__file__, 449, 8), errmess_70875, *[result_mod_70878], **kwargs_70879)
    
    # SSA join for if statement (line 438)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 425)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 417)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 410)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 451)
    tuple_70881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 451)
    # Adding element type (line 451)
    # Getting the type of 'sig' (line 451)
    sig_70882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 11), 'sig')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 11), tuple_70881, sig_70882)
    # Adding element type (line 451)
    # Getting the type of 'sigout' (line 451)
    sigout_70883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 16), 'sigout')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 11), tuple_70881, sigout_70883)
    
    # Assigning a type to the variable 'stypy_return_type' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'stypy_return_type', tuple_70881)
    
    # ################# End of 'getpydocsign(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getpydocsign' in the type store
    # Getting the type of 'stypy_return_type' (line 380)
    stypy_return_type_70884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_70884)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getpydocsign'
    return stypy_return_type_70884

# Assigning a type to the variable 'getpydocsign' (line 380)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 0), 'getpydocsign', getpydocsign)

@norecursion
def getarrdocsign(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getarrdocsign'
    module_type_store = module_type_store.open_function_context('getarrdocsign', 454, 0, False)
    
    # Passed parameters checking function
    getarrdocsign.stypy_localization = localization
    getarrdocsign.stypy_type_of_self = None
    getarrdocsign.stypy_type_store = module_type_store
    getarrdocsign.stypy_function_name = 'getarrdocsign'
    getarrdocsign.stypy_param_names_list = ['a', 'var']
    getarrdocsign.stypy_varargs_param_name = None
    getarrdocsign.stypy_kwargs_param_name = None
    getarrdocsign.stypy_call_defaults = defaults
    getarrdocsign.stypy_call_varargs = varargs
    getarrdocsign.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getarrdocsign', ['a', 'var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getarrdocsign', localization, ['a', 'var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getarrdocsign(...)' code ##################

    
    # Assigning a Call to a Name (line 455):
    
    # Assigning a Call to a Name (line 455):
    
    # Call to getctype(...): (line 455)
    # Processing the call arguments (line 455)
    # Getting the type of 'var' (line 455)
    var_70886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 21), 'var', False)
    # Processing the call keyword arguments (line 455)
    kwargs_70887 = {}
    # Getting the type of 'getctype' (line 455)
    getctype_70885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 12), 'getctype', False)
    # Calling getctype(args, kwargs) (line 455)
    getctype_call_result_70888 = invoke(stypy.reporting.localization.Localization(__file__, 455, 12), getctype_70885, *[var_70886], **kwargs_70887)
    
    # Assigning a type to the variable 'ctype' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'ctype', getctype_call_result_70888)
    
    
    # Evaluating a boolean operation
    
    # Call to isstring(...): (line 456)
    # Processing the call arguments (line 456)
    # Getting the type of 'var' (line 456)
    var_70890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'var', False)
    # Processing the call keyword arguments (line 456)
    kwargs_70891 = {}
    # Getting the type of 'isstring' (line 456)
    isstring_70889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 7), 'isstring', False)
    # Calling isstring(args, kwargs) (line 456)
    isstring_call_result_70892 = invoke(stypy.reporting.localization.Localization(__file__, 456, 7), isstring_70889, *[var_70890], **kwargs_70891)
    
    
    
    # Call to isarray(...): (line 456)
    # Processing the call arguments (line 456)
    # Getting the type of 'var' (line 456)
    var_70894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 38), 'var', False)
    # Processing the call keyword arguments (line 456)
    kwargs_70895 = {}
    # Getting the type of 'isarray' (line 456)
    isarray_70893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 30), 'isarray', False)
    # Calling isarray(args, kwargs) (line 456)
    isarray_call_result_70896 = invoke(stypy.reporting.localization.Localization(__file__, 456, 30), isarray_70893, *[var_70894], **kwargs_70895)
    
    # Applying the 'not' unary operator (line 456)
    result_not__70897 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 26), 'not', isarray_call_result_70896)
    
    # Applying the binary operator 'and' (line 456)
    result_and_keyword_70898 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 7), 'and', isstring_call_result_70892, result_not__70897)
    
    # Testing the type of an if condition (line 456)
    if_condition_70899 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 456, 4), result_and_keyword_70898)
    # Assigning a type to the variable 'if_condition_70899' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'if_condition_70899', if_condition_70899)
    # SSA begins for if statement (line 456)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 457):
    
    # Assigning a BinOp to a Name (line 457):
    str_70900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 14), 'str', "%s : rank-0 array(string(len=%s),'c')")
    
    # Obtaining an instance of the builtin type 'tuple' (line 457)
    tuple_70901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 59), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 457)
    # Adding element type (line 457)
    # Getting the type of 'a' (line 457)
    a_70902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 59), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 59), tuple_70901, a_70902)
    # Adding element type (line 457)
    
    # Call to getstrlength(...): (line 458)
    # Processing the call arguments (line 458)
    # Getting the type of 'var' (line 458)
    var_70904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 72), 'var', False)
    # Processing the call keyword arguments (line 458)
    kwargs_70905 = {}
    # Getting the type of 'getstrlength' (line 458)
    getstrlength_70903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 59), 'getstrlength', False)
    # Calling getstrlength(args, kwargs) (line 458)
    getstrlength_call_result_70906 = invoke(stypy.reporting.localization.Localization(__file__, 458, 59), getstrlength_70903, *[var_70904], **kwargs_70905)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 59), tuple_70901, getstrlength_call_result_70906)
    
    # Applying the binary operator '%' (line 457)
    result_mod_70907 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 14), '%', str_70900, tuple_70901)
    
    # Assigning a type to the variable 'sig' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'sig', result_mod_70907)
    # SSA branch for the else part of an if statement (line 456)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isscalar(...): (line 459)
    # Processing the call arguments (line 459)
    # Getting the type of 'var' (line 459)
    var_70909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 18), 'var', False)
    # Processing the call keyword arguments (line 459)
    kwargs_70910 = {}
    # Getting the type of 'isscalar' (line 459)
    isscalar_70908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 9), 'isscalar', False)
    # Calling isscalar(args, kwargs) (line 459)
    isscalar_call_result_70911 = invoke(stypy.reporting.localization.Localization(__file__, 459, 9), isscalar_70908, *[var_70909], **kwargs_70910)
    
    # Testing the type of an if condition (line 459)
    if_condition_70912 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 459, 9), isscalar_call_result_70911)
    # Assigning a type to the variable 'if_condition_70912' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 9), 'if_condition_70912', if_condition_70912)
    # SSA begins for if statement (line 459)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 460):
    
    # Assigning a BinOp to a Name (line 460):
    str_70913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 14), 'str', "%s : rank-0 array(%s,'%s')")
    
    # Obtaining an instance of the builtin type 'tuple' (line 460)
    tuple_70914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 460)
    # Adding element type (line 460)
    # Getting the type of 'a' (line 460)
    a_70915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 48), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 48), tuple_70914, a_70915)
    # Adding element type (line 460)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ctype' (line 460)
    ctype_70916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 60), 'ctype')
    # Getting the type of 'c2py_map' (line 460)
    c2py_map_70917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 51), 'c2py_map')
    # Obtaining the member '__getitem__' of a type (line 460)
    getitem___70918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 51), c2py_map_70917, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 460)
    subscript_call_result_70919 = invoke(stypy.reporting.localization.Localization(__file__, 460, 51), getitem___70918, ctype_70916)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 48), tuple_70914, subscript_call_result_70919)
    # Adding element type (line 460)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ctype' (line 461)
    ctype_70920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 61), 'ctype')
    # Getting the type of 'c2pycode_map' (line 461)
    c2pycode_map_70921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 48), 'c2pycode_map')
    # Obtaining the member '__getitem__' of a type (line 461)
    getitem___70922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 48), c2pycode_map_70921, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 461)
    subscript_call_result_70923 = invoke(stypy.reporting.localization.Localization(__file__, 461, 48), getitem___70922, ctype_70920)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 48), tuple_70914, subscript_call_result_70923)
    
    # Applying the binary operator '%' (line 460)
    result_mod_70924 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 14), '%', str_70913, tuple_70914)
    
    # Assigning a type to the variable 'sig' (line 460)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'sig', result_mod_70924)
    # SSA branch for the else part of an if statement (line 459)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isarray(...): (line 462)
    # Processing the call arguments (line 462)
    # Getting the type of 'var' (line 462)
    var_70926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 17), 'var', False)
    # Processing the call keyword arguments (line 462)
    kwargs_70927 = {}
    # Getting the type of 'isarray' (line 462)
    isarray_70925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 9), 'isarray', False)
    # Calling isarray(args, kwargs) (line 462)
    isarray_call_result_70928 = invoke(stypy.reporting.localization.Localization(__file__, 462, 9), isarray_70925, *[var_70926], **kwargs_70927)
    
    # Testing the type of an if condition (line 462)
    if_condition_70929 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 462, 9), isarray_call_result_70928)
    # Assigning a type to the variable 'if_condition_70929' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 9), 'if_condition_70929', if_condition_70929)
    # SSA begins for if statement (line 462)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 463):
    
    # Assigning a Subscript to a Name (line 463):
    
    # Obtaining the type of the subscript
    str_70930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 18), 'str', 'dimension')
    # Getting the type of 'var' (line 463)
    var_70931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 14), 'var')
    # Obtaining the member '__getitem__' of a type (line 463)
    getitem___70932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 14), var_70931, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 463)
    subscript_call_result_70933 = invoke(stypy.reporting.localization.Localization(__file__, 463, 14), getitem___70932, str_70930)
    
    # Assigning a type to the variable 'dim' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'dim', subscript_call_result_70933)
    
    # Assigning a Call to a Name (line 464):
    
    # Assigning a Call to a Name (line 464):
    
    # Call to repr(...): (line 464)
    # Processing the call arguments (line 464)
    
    # Call to len(...): (line 464)
    # Processing the call arguments (line 464)
    # Getting the type of 'dim' (line 464)
    dim_70936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 24), 'dim', False)
    # Processing the call keyword arguments (line 464)
    kwargs_70937 = {}
    # Getting the type of 'len' (line 464)
    len_70935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 20), 'len', False)
    # Calling len(args, kwargs) (line 464)
    len_call_result_70938 = invoke(stypy.reporting.localization.Localization(__file__, 464, 20), len_70935, *[dim_70936], **kwargs_70937)
    
    # Processing the call keyword arguments (line 464)
    kwargs_70939 = {}
    # Getting the type of 'repr' (line 464)
    repr_70934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 15), 'repr', False)
    # Calling repr(args, kwargs) (line 464)
    repr_call_result_70940 = invoke(stypy.reporting.localization.Localization(__file__, 464, 15), repr_70934, *[len_call_result_70938], **kwargs_70939)
    
    # Assigning a type to the variable 'rank' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'rank', repr_call_result_70940)
    
    # Assigning a BinOp to a Name (line 465):
    
    # Assigning a BinOp to a Name (line 465):
    str_70941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 14), 'str', "%s : rank-%s array('%s') with bounds (%s)")
    
    # Obtaining an instance of the builtin type 'tuple' (line 465)
    tuple_70942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 63), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 465)
    # Adding element type (line 465)
    # Getting the type of 'a' (line 465)
    a_70943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 63), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 63), tuple_70942, a_70943)
    # Adding element type (line 465)
    # Getting the type of 'rank' (line 465)
    rank_70944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 66), 'rank')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 63), tuple_70942, rank_70944)
    # Adding element type (line 465)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ctype' (line 467)
    ctype_70945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 67), 'ctype')
    # Getting the type of 'c2pycode_map' (line 466)
    c2pycode_map_70946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 63), 'c2pycode_map')
    # Obtaining the member '__getitem__' of a type (line 466)
    getitem___70947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 63), c2pycode_map_70946, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 466)
    subscript_call_result_70948 = invoke(stypy.reporting.localization.Localization(__file__, 466, 63), getitem___70947, ctype_70945)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 63), tuple_70942, subscript_call_result_70948)
    # Adding element type (line 465)
    
    # Call to join(...): (line 468)
    # Processing the call arguments (line 468)
    # Getting the type of 'dim' (line 468)
    dim_70951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 72), 'dim', False)
    # Processing the call keyword arguments (line 468)
    kwargs_70952 = {}
    str_70949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 63), 'str', ',')
    # Obtaining the member 'join' of a type (line 468)
    join_70950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 63), str_70949, 'join')
    # Calling join(args, kwargs) (line 468)
    join_call_result_70953 = invoke(stypy.reporting.localization.Localization(__file__, 468, 63), join_70950, *[dim_70951], **kwargs_70952)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 63), tuple_70942, join_call_result_70953)
    
    # Applying the binary operator '%' (line 465)
    result_mod_70954 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 14), '%', str_70941, tuple_70942)
    
    # Assigning a type to the variable 'sig' (line 465)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'sig', result_mod_70954)
    # SSA join for if statement (line 462)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 459)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 456)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'sig' (line 469)
    sig_70955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 11), 'sig')
    # Assigning a type to the variable 'stypy_return_type' (line 469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 4), 'stypy_return_type', sig_70955)
    
    # ################# End of 'getarrdocsign(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getarrdocsign' in the type store
    # Getting the type of 'stypy_return_type' (line 454)
    stypy_return_type_70956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_70956)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getarrdocsign'
    return stypy_return_type_70956

# Assigning a type to the variable 'getarrdocsign' (line 454)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 0), 'getarrdocsign', getarrdocsign)

@norecursion
def getinit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getinit'
    module_type_store = module_type_store.open_function_context('getinit', 472, 0, False)
    
    # Passed parameters checking function
    getinit.stypy_localization = localization
    getinit.stypy_type_of_self = None
    getinit.stypy_type_store = module_type_store
    getinit.stypy_function_name = 'getinit'
    getinit.stypy_param_names_list = ['a', 'var']
    getinit.stypy_varargs_param_name = None
    getinit.stypy_kwargs_param_name = None
    getinit.stypy_call_defaults = defaults
    getinit.stypy_call_varargs = varargs
    getinit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getinit', ['a', 'var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getinit', localization, ['a', 'var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getinit(...)' code ##################

    
    
    # Call to isstring(...): (line 473)
    # Processing the call arguments (line 473)
    # Getting the type of 'var' (line 473)
    var_70958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 16), 'var', False)
    # Processing the call keyword arguments (line 473)
    kwargs_70959 = {}
    # Getting the type of 'isstring' (line 473)
    isstring_70957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 7), 'isstring', False)
    # Calling isstring(args, kwargs) (line 473)
    isstring_call_result_70960 = invoke(stypy.reporting.localization.Localization(__file__, 473, 7), isstring_70957, *[var_70958], **kwargs_70959)
    
    # Testing the type of an if condition (line 473)
    if_condition_70961 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 473, 4), isstring_call_result_70960)
    # Assigning a type to the variable 'if_condition_70961' (line 473)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 4), 'if_condition_70961', if_condition_70961)
    # SSA begins for if statement (line 473)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Tuple (line 474):
    
    # Assigning a Str to a Name (line 474):
    str_70962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 25), 'str', '""')
    # Assigning a type to the variable 'tuple_assignment_69352' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'tuple_assignment_69352', str_70962)
    
    # Assigning a Str to a Name (line 474):
    str_70963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 31), 'str', "''")
    # Assigning a type to the variable 'tuple_assignment_69353' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'tuple_assignment_69353', str_70963)
    
    # Assigning a Name to a Name (line 474):
    # Getting the type of 'tuple_assignment_69352' (line 474)
    tuple_assignment_69352_70964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'tuple_assignment_69352')
    # Assigning a type to the variable 'init' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'init', tuple_assignment_69352_70964)
    
    # Assigning a Name to a Name (line 474):
    # Getting the type of 'tuple_assignment_69353' (line 474)
    tuple_assignment_69353_70965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'tuple_assignment_69353')
    # Assigning a type to the variable 'showinit' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 14), 'showinit', tuple_assignment_69353_70965)
    # SSA branch for the else part of an if statement (line 473)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Tuple to a Tuple (line 476):
    
    # Assigning a Str to a Name (line 476):
    str_70966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 25), 'str', '')
    # Assigning a type to the variable 'tuple_assignment_69354' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'tuple_assignment_69354', str_70966)
    
    # Assigning a Str to a Name (line 476):
    str_70967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 29), 'str', '')
    # Assigning a type to the variable 'tuple_assignment_69355' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'tuple_assignment_69355', str_70967)
    
    # Assigning a Name to a Name (line 476):
    # Getting the type of 'tuple_assignment_69354' (line 476)
    tuple_assignment_69354_70968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'tuple_assignment_69354')
    # Assigning a type to the variable 'init' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'init', tuple_assignment_69354_70968)
    
    # Assigning a Name to a Name (line 476):
    # Getting the type of 'tuple_assignment_69355' (line 476)
    tuple_assignment_69355_70969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'tuple_assignment_69355')
    # Assigning a type to the variable 'showinit' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 14), 'showinit', tuple_assignment_69355_70969)
    # SSA join for if statement (line 473)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to hasinitvalue(...): (line 477)
    # Processing the call arguments (line 477)
    # Getting the type of 'var' (line 477)
    var_70971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 20), 'var', False)
    # Processing the call keyword arguments (line 477)
    kwargs_70972 = {}
    # Getting the type of 'hasinitvalue' (line 477)
    hasinitvalue_70970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 7), 'hasinitvalue', False)
    # Calling hasinitvalue(args, kwargs) (line 477)
    hasinitvalue_call_result_70973 = invoke(stypy.reporting.localization.Localization(__file__, 477, 7), hasinitvalue_70970, *[var_70971], **kwargs_70972)
    
    # Testing the type of an if condition (line 477)
    if_condition_70974 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 477, 4), hasinitvalue_call_result_70973)
    # Assigning a type to the variable 'if_condition_70974' (line 477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'if_condition_70974', if_condition_70974)
    # SSA begins for if statement (line 477)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 478):
    
    # Assigning a Subscript to a Name (line 478):
    
    # Obtaining the type of the subscript
    str_70975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 19), 'str', '=')
    # Getting the type of 'var' (line 478)
    var_70976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 15), 'var')
    # Obtaining the member '__getitem__' of a type (line 478)
    getitem___70977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 15), var_70976, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 478)
    subscript_call_result_70978 = invoke(stypy.reporting.localization.Localization(__file__, 478, 15), getitem___70977, str_70975)
    
    # Assigning a type to the variable 'init' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'init', subscript_call_result_70978)
    
    # Assigning a Name to a Name (line 479):
    
    # Assigning a Name to a Name (line 479):
    # Getting the type of 'init' (line 479)
    init_70979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 19), 'init')
    # Assigning a type to the variable 'showinit' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'showinit', init_70979)
    
    
    # Evaluating a boolean operation
    
    # Call to iscomplex(...): (line 480)
    # Processing the call arguments (line 480)
    # Getting the type of 'var' (line 480)
    var_70981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 21), 'var', False)
    # Processing the call keyword arguments (line 480)
    kwargs_70982 = {}
    # Getting the type of 'iscomplex' (line 480)
    iscomplex_70980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 11), 'iscomplex', False)
    # Calling iscomplex(args, kwargs) (line 480)
    iscomplex_call_result_70983 = invoke(stypy.reporting.localization.Localization(__file__, 480, 11), iscomplex_70980, *[var_70981], **kwargs_70982)
    
    
    # Call to iscomplexarray(...): (line 480)
    # Processing the call arguments (line 480)
    # Getting the type of 'var' (line 480)
    var_70985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 44), 'var', False)
    # Processing the call keyword arguments (line 480)
    kwargs_70986 = {}
    # Getting the type of 'iscomplexarray' (line 480)
    iscomplexarray_70984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 29), 'iscomplexarray', False)
    # Calling iscomplexarray(args, kwargs) (line 480)
    iscomplexarray_call_result_70987 = invoke(stypy.reporting.localization.Localization(__file__, 480, 29), iscomplexarray_70984, *[var_70985], **kwargs_70986)
    
    # Applying the binary operator 'or' (line 480)
    result_or_keyword_70988 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 11), 'or', iscomplex_call_result_70983, iscomplexarray_call_result_70987)
    
    # Testing the type of an if condition (line 480)
    if_condition_70989 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 480, 8), result_or_keyword_70988)
    # Assigning a type to the variable 'if_condition_70989' (line 480)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'if_condition_70989', if_condition_70989)
    # SSA begins for if statement (line 480)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Dict to a Name (line 481):
    
    # Assigning a Dict to a Name (line 481):
    
    # Obtaining an instance of the builtin type 'dict' (line 481)
    dict_70990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 18), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 481)
    
    # Assigning a type to the variable 'ret' (line 481)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'ret', dict_70990)
    
    
    # SSA begins for try-except statement (line 483)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 484):
    
    # Assigning a Subscript to a Name (line 484):
    
    # Obtaining the type of the subscript
    str_70991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 24), 'str', '=')
    # Getting the type of 'var' (line 484)
    var_70992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 20), 'var')
    # Obtaining the member '__getitem__' of a type (line 484)
    getitem___70993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 20), var_70992, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 484)
    subscript_call_result_70994 = invoke(stypy.reporting.localization.Localization(__file__, 484, 20), getitem___70993, str_70991)
    
    # Assigning a type to the variable 'v' (line 484)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 16), 'v', subscript_call_result_70994)
    
    
    str_70995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 19), 'str', ',')
    # Getting the type of 'v' (line 485)
    v_70996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 26), 'v')
    # Applying the binary operator 'in' (line 485)
    result_contains_70997 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 19), 'in', str_70995, v_70996)
    
    # Testing the type of an if condition (line 485)
    if_condition_70998 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 485, 16), result_contains_70997)
    # Assigning a type to the variable 'if_condition_70998' (line 485)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 16), 'if_condition_70998', if_condition_70998)
    # SSA begins for if statement (line 485)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 486):
    
    # Assigning a Call to a Name:
    
    # Call to split(...): (line 486)
    # Processing the call arguments (line 486)
    str_71009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 39), 'str', '@,@')
    # Processing the call keyword arguments (line 486)
    kwargs_71010 = {}
    
    # Call to markoutercomma(...): (line 486)
    # Processing the call arguments (line 486)
    
    # Obtaining the type of the subscript
    int_71000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 26), 'int')
    int_71001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 28), 'int')
    slice_71002 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 487, 24), int_71000, int_71001, None)
    # Getting the type of 'v' (line 487)
    v_71003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 24), 'v', False)
    # Obtaining the member '__getitem__' of a type (line 487)
    getitem___71004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 24), v_71003, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 487)
    subscript_call_result_71005 = invoke(stypy.reporting.localization.Localization(__file__, 487, 24), getitem___71004, slice_71002)
    
    # Processing the call keyword arguments (line 486)
    kwargs_71006 = {}
    # Getting the type of 'markoutercomma' (line 486)
    markoutercomma_70999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 51), 'markoutercomma', False)
    # Calling markoutercomma(args, kwargs) (line 486)
    markoutercomma_call_result_71007 = invoke(stypy.reporting.localization.Localization(__file__, 486, 51), markoutercomma_70999, *[subscript_call_result_71005], **kwargs_71006)
    
    # Obtaining the member 'split' of a type (line 486)
    split_71008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 51), markoutercomma_call_result_71007, 'split')
    # Calling split(args, kwargs) (line 486)
    split_call_result_71011 = invoke(stypy.reporting.localization.Localization(__file__, 486, 51), split_71008, *[str_71009], **kwargs_71010)
    
    # Assigning a type to the variable 'call_assignment_69356' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 20), 'call_assignment_69356', split_call_result_71011)
    
    # Assigning a Call to a Name (line 486):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_71014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 20), 'int')
    # Processing the call keyword arguments
    kwargs_71015 = {}
    # Getting the type of 'call_assignment_69356' (line 486)
    call_assignment_69356_71012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 20), 'call_assignment_69356', False)
    # Obtaining the member '__getitem__' of a type (line 486)
    getitem___71013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 20), call_assignment_69356_71012, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_71016 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___71013, *[int_71014], **kwargs_71015)
    
    # Assigning a type to the variable 'call_assignment_69357' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 20), 'call_assignment_69357', getitem___call_result_71016)
    
    # Assigning a Name to a Subscript (line 486):
    # Getting the type of 'call_assignment_69357' (line 486)
    call_assignment_69357_71017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 20), 'call_assignment_69357')
    # Getting the type of 'ret' (line 486)
    ret_71018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 20), 'ret')
    str_71019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 24), 'str', 'init.r')
    # Storing an element on a container (line 486)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 486, 20), ret_71018, (str_71019, call_assignment_69357_71017))
    
    # Assigning a Call to a Name (line 486):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_71022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 20), 'int')
    # Processing the call keyword arguments
    kwargs_71023 = {}
    # Getting the type of 'call_assignment_69356' (line 486)
    call_assignment_69356_71020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 20), 'call_assignment_69356', False)
    # Obtaining the member '__getitem__' of a type (line 486)
    getitem___71021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 20), call_assignment_69356_71020, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_71024 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___71021, *[int_71022], **kwargs_71023)
    
    # Assigning a type to the variable 'call_assignment_69358' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 20), 'call_assignment_69358', getitem___call_result_71024)
    
    # Assigning a Name to a Subscript (line 486):
    # Getting the type of 'call_assignment_69358' (line 486)
    call_assignment_69358_71025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 20), 'call_assignment_69358')
    # Getting the type of 'ret' (line 486)
    ret_71026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 35), 'ret')
    str_71027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 39), 'str', 'init.i')
    # Storing an element on a container (line 486)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 486, 35), ret_71026, (str_71027, call_assignment_69358_71025))
    # SSA branch for the else part of an if statement (line 485)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 489):
    
    # Assigning a Call to a Name (line 489):
    
    # Call to eval(...): (line 489)
    # Processing the call arguments (line 489)
    # Getting the type of 'v' (line 489)
    v_71029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 29), 'v', False)
    
    # Obtaining an instance of the builtin type 'dict' (line 489)
    dict_71030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 32), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 489)
    
    
    # Obtaining an instance of the builtin type 'dict' (line 489)
    dict_71031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 36), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 489)
    
    # Processing the call keyword arguments (line 489)
    kwargs_71032 = {}
    # Getting the type of 'eval' (line 489)
    eval_71028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 24), 'eval', False)
    # Calling eval(args, kwargs) (line 489)
    eval_call_result_71033 = invoke(stypy.reporting.localization.Localization(__file__, 489, 24), eval_71028, *[v_71029, dict_71030, dict_71031], **kwargs_71032)
    
    # Assigning a type to the variable 'v' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 20), 'v', eval_call_result_71033)
    
    # Assigning a Tuple to a Tuple (line 490):
    
    # Assigning a Call to a Name (line 490):
    
    # Call to str(...): (line 490)
    # Processing the call arguments (line 490)
    # Getting the type of 'v' (line 490)
    v_71035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 55), 'v', False)
    # Obtaining the member 'real' of a type (line 490)
    real_71036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 55), v_71035, 'real')
    # Processing the call keyword arguments (line 490)
    kwargs_71037 = {}
    # Getting the type of 'str' (line 490)
    str_71034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 51), 'str', False)
    # Calling str(args, kwargs) (line 490)
    str_call_result_71038 = invoke(stypy.reporting.localization.Localization(__file__, 490, 51), str_71034, *[real_71036], **kwargs_71037)
    
    # Assigning a type to the variable 'tuple_assignment_69359' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 20), 'tuple_assignment_69359', str_call_result_71038)
    
    # Assigning a Call to a Name (line 490):
    
    # Call to str(...): (line 490)
    # Processing the call arguments (line 490)
    # Getting the type of 'v' (line 490)
    v_71040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 68), 'v', False)
    # Obtaining the member 'imag' of a type (line 490)
    imag_71041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 68), v_71040, 'imag')
    # Processing the call keyword arguments (line 490)
    kwargs_71042 = {}
    # Getting the type of 'str' (line 490)
    str_71039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 64), 'str', False)
    # Calling str(args, kwargs) (line 490)
    str_call_result_71043 = invoke(stypy.reporting.localization.Localization(__file__, 490, 64), str_71039, *[imag_71041], **kwargs_71042)
    
    # Assigning a type to the variable 'tuple_assignment_69360' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 20), 'tuple_assignment_69360', str_call_result_71043)
    
    # Assigning a Name to a Subscript (line 490):
    # Getting the type of 'tuple_assignment_69359' (line 490)
    tuple_assignment_69359_71044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 20), 'tuple_assignment_69359')
    # Getting the type of 'ret' (line 490)
    ret_71045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 20), 'ret')
    str_71046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 24), 'str', 'init.r')
    # Storing an element on a container (line 490)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 20), ret_71045, (str_71046, tuple_assignment_69359_71044))
    
    # Assigning a Name to a Subscript (line 490):
    # Getting the type of 'tuple_assignment_69360' (line 490)
    tuple_assignment_69360_71047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 20), 'tuple_assignment_69360')
    # Getting the type of 'ret' (line 490)
    ret_71048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 35), 'ret')
    str_71049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 39), 'str', 'init.i')
    # Storing an element on a container (line 490)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 35), ret_71048, (str_71049, tuple_assignment_69360_71047))
    # SSA join for if statement (line 485)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 483)
    # SSA branch for the except '<any exception>' branch of a try statement (line 483)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 492)
    # Processing the call arguments (line 492)
    str_71051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 20), 'str', "getinit: expected complex number `(r,i)' but got `%s' as initial value of %r.")
    
    # Obtaining an instance of the builtin type 'tuple' (line 493)
    tuple_71052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 105), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 493)
    # Adding element type (line 493)
    # Getting the type of 'init' (line 493)
    init_71053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 105), 'init', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 105), tuple_71052, init_71053)
    # Adding element type (line 493)
    # Getting the type of 'a' (line 493)
    a_71054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 111), 'a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 105), tuple_71052, a_71054)
    
    # Applying the binary operator '%' (line 493)
    result_mod_71055 = python_operator(stypy.reporting.localization.Localization(__file__, 493, 20), '%', str_71051, tuple_71052)
    
    # Processing the call keyword arguments (line 492)
    kwargs_71056 = {}
    # Getting the type of 'ValueError' (line 492)
    ValueError_71050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 492)
    ValueError_call_result_71057 = invoke(stypy.reporting.localization.Localization(__file__, 492, 22), ValueError_71050, *[result_mod_71055], **kwargs_71056)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 492, 16), ValueError_call_result_71057, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 483)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isarray(...): (line 494)
    # Processing the call arguments (line 494)
    # Getting the type of 'var' (line 494)
    var_71059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 23), 'var', False)
    # Processing the call keyword arguments (line 494)
    kwargs_71060 = {}
    # Getting the type of 'isarray' (line 494)
    isarray_71058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 15), 'isarray', False)
    # Calling isarray(args, kwargs) (line 494)
    isarray_call_result_71061 = invoke(stypy.reporting.localization.Localization(__file__, 494, 15), isarray_71058, *[var_71059], **kwargs_71060)
    
    # Testing the type of an if condition (line 494)
    if_condition_71062 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 494, 12), isarray_call_result_71061)
    # Assigning a type to the variable 'if_condition_71062' (line 494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'if_condition_71062', if_condition_71062)
    # SSA begins for if statement (line 494)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 495):
    
    # Assigning a BinOp to a Name (line 495):
    str_71063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 23), 'str', '(capi_c.r=%s,capi_c.i=%s,capi_c)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 496)
    tuple_71064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 496)
    # Adding element type (line 496)
    
    # Obtaining the type of the subscript
    str_71065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 24), 'str', 'init.r')
    # Getting the type of 'ret' (line 496)
    ret_71066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 20), 'ret')
    # Obtaining the member '__getitem__' of a type (line 496)
    getitem___71067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 20), ret_71066, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 496)
    subscript_call_result_71068 = invoke(stypy.reporting.localization.Localization(__file__, 496, 20), getitem___71067, str_71065)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 20), tuple_71064, subscript_call_result_71068)
    # Adding element type (line 496)
    
    # Obtaining the type of the subscript
    str_71069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 39), 'str', 'init.i')
    # Getting the type of 'ret' (line 496)
    ret_71070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 35), 'ret')
    # Obtaining the member '__getitem__' of a type (line 496)
    getitem___71071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 35), ret_71070, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 496)
    subscript_call_result_71072 = invoke(stypy.reporting.localization.Localization(__file__, 496, 35), getitem___71071, str_71069)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 20), tuple_71064, subscript_call_result_71072)
    
    # Applying the binary operator '%' (line 495)
    result_mod_71073 = python_operator(stypy.reporting.localization.Localization(__file__, 495, 23), '%', str_71063, tuple_71064)
    
    # Assigning a type to the variable 'init' (line 495)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 16), 'init', result_mod_71073)
    # SSA join for if statement (line 494)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 480)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isstring(...): (line 497)
    # Processing the call arguments (line 497)
    # Getting the type of 'var' (line 497)
    var_71075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 22), 'var', False)
    # Processing the call keyword arguments (line 497)
    kwargs_71076 = {}
    # Getting the type of 'isstring' (line 497)
    isstring_71074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 13), 'isstring', False)
    # Calling isstring(args, kwargs) (line 497)
    isstring_call_result_71077 = invoke(stypy.reporting.localization.Localization(__file__, 497, 13), isstring_71074, *[var_71075], **kwargs_71076)
    
    # Testing the type of an if condition (line 497)
    if_condition_71078 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 497, 13), isstring_call_result_71077)
    # Assigning a type to the variable 'if_condition_71078' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 13), 'if_condition_71078', if_condition_71078)
    # SSA begins for if statement (line 497)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'init' (line 498)
    init_71079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 19), 'init')
    # Applying the 'not' unary operator (line 498)
    result_not__71080 = python_operator(stypy.reporting.localization.Localization(__file__, 498, 15), 'not', init_71079)
    
    # Testing the type of an if condition (line 498)
    if_condition_71081 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 498, 12), result_not__71080)
    # Assigning a type to the variable 'if_condition_71081' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 12), 'if_condition_71081', if_condition_71081)
    # SSA begins for if statement (line 498)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Tuple (line 499):
    
    # Assigning a Str to a Name (line 499):
    str_71082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 33), 'str', '""')
    # Assigning a type to the variable 'tuple_assignment_69361' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 16), 'tuple_assignment_69361', str_71082)
    
    # Assigning a Str to a Name (line 499):
    str_71083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 39), 'str', "''")
    # Assigning a type to the variable 'tuple_assignment_69362' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 16), 'tuple_assignment_69362', str_71083)
    
    # Assigning a Name to a Name (line 499):
    # Getting the type of 'tuple_assignment_69361' (line 499)
    tuple_assignment_69361_71084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 16), 'tuple_assignment_69361')
    # Assigning a type to the variable 'init' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 16), 'init', tuple_assignment_69361_71084)
    
    # Assigning a Name to a Name (line 499):
    # Getting the type of 'tuple_assignment_69362' (line 499)
    tuple_assignment_69362_71085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 16), 'tuple_assignment_69362')
    # Assigning a type to the variable 'showinit' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 22), 'showinit', tuple_assignment_69362_71085)
    # SSA join for if statement (line 498)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_71086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 20), 'int')
    # Getting the type of 'init' (line 500)
    init_71087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 15), 'init')
    # Obtaining the member '__getitem__' of a type (line 500)
    getitem___71088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 15), init_71087, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 500)
    subscript_call_result_71089 = invoke(stypy.reporting.localization.Localization(__file__, 500, 15), getitem___71088, int_71086)
    
    str_71090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 26), 'str', "'")
    # Applying the binary operator '==' (line 500)
    result_eq_71091 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 15), '==', subscript_call_result_71089, str_71090)
    
    # Testing the type of an if condition (line 500)
    if_condition_71092 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 500, 12), result_eq_71091)
    # Assigning a type to the variable 'if_condition_71092' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 12), 'if_condition_71092', if_condition_71092)
    # SSA begins for if statement (line 500)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 501):
    
    # Assigning a BinOp to a Name (line 501):
    str_71093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 23), 'str', '"%s"')
    
    # Call to replace(...): (line 501)
    # Processing the call arguments (line 501)
    str_71101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 52), 'str', '"')
    str_71102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 57), 'str', '\\"')
    # Processing the call keyword arguments (line 501)
    kwargs_71103 = {}
    
    # Obtaining the type of the subscript
    int_71094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 38), 'int')
    int_71095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 40), 'int')
    slice_71096 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 501, 33), int_71094, int_71095, None)
    # Getting the type of 'init' (line 501)
    init_71097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 33), 'init', False)
    # Obtaining the member '__getitem__' of a type (line 501)
    getitem___71098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 33), init_71097, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 501)
    subscript_call_result_71099 = invoke(stypy.reporting.localization.Localization(__file__, 501, 33), getitem___71098, slice_71096)
    
    # Obtaining the member 'replace' of a type (line 501)
    replace_71100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 33), subscript_call_result_71099, 'replace')
    # Calling replace(args, kwargs) (line 501)
    replace_call_result_71104 = invoke(stypy.reporting.localization.Localization(__file__, 501, 33), replace_71100, *[str_71101, str_71102], **kwargs_71103)
    
    # Applying the binary operator '%' (line 501)
    result_mod_71105 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 23), '%', str_71093, replace_call_result_71104)
    
    # Assigning a type to the variable 'init' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 16), 'init', result_mod_71105)
    # SSA join for if statement (line 500)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_71106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 20), 'int')
    # Getting the type of 'init' (line 502)
    init_71107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 15), 'init')
    # Obtaining the member '__getitem__' of a type (line 502)
    getitem___71108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 15), init_71107, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 502)
    subscript_call_result_71109 = invoke(stypy.reporting.localization.Localization(__file__, 502, 15), getitem___71108, int_71106)
    
    str_71110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 26), 'str', '"')
    # Applying the binary operator '==' (line 502)
    result_eq_71111 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 15), '==', subscript_call_result_71109, str_71110)
    
    # Testing the type of an if condition (line 502)
    if_condition_71112 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 502, 12), result_eq_71111)
    # Assigning a type to the variable 'if_condition_71112' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 12), 'if_condition_71112', if_condition_71112)
    # SSA begins for if statement (line 502)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 503):
    
    # Assigning a BinOp to a Name (line 503):
    str_71113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 27), 'str', "'%s'")
    
    # Obtaining the type of the subscript
    int_71114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 42), 'int')
    int_71115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 44), 'int')
    slice_71116 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 503, 37), int_71114, int_71115, None)
    # Getting the type of 'init' (line 503)
    init_71117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 37), 'init')
    # Obtaining the member '__getitem__' of a type (line 503)
    getitem___71118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 37), init_71117, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 503)
    subscript_call_result_71119 = invoke(stypy.reporting.localization.Localization(__file__, 503, 37), getitem___71118, slice_71116)
    
    # Applying the binary operator '%' (line 503)
    result_mod_71120 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 27), '%', str_71113, subscript_call_result_71119)
    
    # Assigning a type to the variable 'showinit' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 16), 'showinit', result_mod_71120)
    # SSA join for if statement (line 502)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 497)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 480)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 477)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 504)
    tuple_71121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 504)
    # Adding element type (line 504)
    # Getting the type of 'init' (line 504)
    init_71122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 11), 'init')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 11), tuple_71121, init_71122)
    # Adding element type (line 504)
    # Getting the type of 'showinit' (line 504)
    showinit_71123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 17), 'showinit')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 11), tuple_71121, showinit_71123)
    
    # Assigning a type to the variable 'stypy_return_type' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), 'stypy_return_type', tuple_71121)
    
    # ################# End of 'getinit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getinit' in the type store
    # Getting the type of 'stypy_return_type' (line 472)
    stypy_return_type_71124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_71124)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getinit'
    return stypy_return_type_71124

# Assigning a type to the variable 'getinit' (line 472)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 0), 'getinit', getinit)

@norecursion
def sign2map(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sign2map'
    module_type_store = module_type_store.open_function_context('sign2map', 507, 0, False)
    
    # Passed parameters checking function
    sign2map.stypy_localization = localization
    sign2map.stypy_type_of_self = None
    sign2map.stypy_type_store = module_type_store
    sign2map.stypy_function_name = 'sign2map'
    sign2map.stypy_param_names_list = ['a', 'var']
    sign2map.stypy_varargs_param_name = None
    sign2map.stypy_kwargs_param_name = None
    sign2map.stypy_call_defaults = defaults
    sign2map.stypy_call_varargs = varargs
    sign2map.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sign2map', ['a', 'var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sign2map', localization, ['a', 'var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sign2map(...)' code ##################

    str_71125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, (-1)), 'str', '\n    varname,ctype,atype\n    init,init.r,init.i,pytype\n    vardebuginfo,vardebugshowvalue,varshowvalue\n    varrfromat\n    intent\n    ')
    # Marking variables as global (line 515)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 515, 4), 'lcb_map')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 515, 4), 'cb_map')
    
    # Assigning a Name to a Name (line 516):
    
    # Assigning a Name to a Name (line 516):
    # Getting the type of 'a' (line 516)
    a_71126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 12), 'a')
    # Assigning a type to the variable 'out_a' (line 516)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'out_a', a_71126)
    
    
    # Call to isintent_out(...): (line 517)
    # Processing the call arguments (line 517)
    # Getting the type of 'var' (line 517)
    var_71128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 20), 'var', False)
    # Processing the call keyword arguments (line 517)
    kwargs_71129 = {}
    # Getting the type of 'isintent_out' (line 517)
    isintent_out_71127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 7), 'isintent_out', False)
    # Calling isintent_out(args, kwargs) (line 517)
    isintent_out_call_result_71130 = invoke(stypy.reporting.localization.Localization(__file__, 517, 7), isintent_out_71127, *[var_71128], **kwargs_71129)
    
    # Testing the type of an if condition (line 517)
    if_condition_71131 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 517, 4), isintent_out_call_result_71130)
    # Assigning a type to the variable 'if_condition_71131' (line 517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 4), 'if_condition_71131', if_condition_71131)
    # SSA begins for if statement (line 517)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Obtaining the type of the subscript
    str_71132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 21), 'str', 'intent')
    # Getting the type of 'var' (line 518)
    var_71133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 17), 'var')
    # Obtaining the member '__getitem__' of a type (line 518)
    getitem___71134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 17), var_71133, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 518)
    subscript_call_result_71135 = invoke(stypy.reporting.localization.Localization(__file__, 518, 17), getitem___71134, str_71132)
    
    # Testing the type of a for loop iterable (line 518)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 518, 8), subscript_call_result_71135)
    # Getting the type of the for loop variable (line 518)
    for_loop_var_71136 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 518, 8), subscript_call_result_71135)
    # Assigning a type to the variable 'k' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'k', for_loop_var_71136)
    # SSA begins for a for statement (line 518)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    int_71137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 18), 'int')
    slice_71138 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 519, 15), None, int_71137, None)
    # Getting the type of 'k' (line 519)
    k_71139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 15), 'k')
    # Obtaining the member '__getitem__' of a type (line 519)
    getitem___71140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 15), k_71139, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 519)
    subscript_call_result_71141 = invoke(stypy.reporting.localization.Localization(__file__, 519, 15), getitem___71140, slice_71138)
    
    str_71142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 24), 'str', 'out=')
    # Applying the binary operator '==' (line 519)
    result_eq_71143 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 15), '==', subscript_call_result_71141, str_71142)
    
    # Testing the type of an if condition (line 519)
    if_condition_71144 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 519, 12), result_eq_71143)
    # Assigning a type to the variable 'if_condition_71144' (line 519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 12), 'if_condition_71144', if_condition_71144)
    # SSA begins for if statement (line 519)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 520):
    
    # Assigning a Subscript to a Name (line 520):
    
    # Obtaining the type of the subscript
    int_71145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 26), 'int')
    slice_71146 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 520, 24), int_71145, None, None)
    # Getting the type of 'k' (line 520)
    k_71147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 24), 'k')
    # Obtaining the member '__getitem__' of a type (line 520)
    getitem___71148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 24), k_71147, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 520)
    subscript_call_result_71149 = invoke(stypy.reporting.localization.Localization(__file__, 520, 24), getitem___71148, slice_71146)
    
    # Assigning a type to the variable 'out_a' (line 520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 16), 'out_a', subscript_call_result_71149)
    # SSA join for if statement (line 519)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 517)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Name (line 522):
    
    # Assigning a Dict to a Name (line 522):
    
    # Obtaining an instance of the builtin type 'dict' (line 522)
    dict_71150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 10), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 522)
    # Adding element type (key, value) (line 522)
    str_71151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 11), 'str', 'varname')
    # Getting the type of 'a' (line 522)
    a_71152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 22), 'a')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 10), dict_71150, (str_71151, a_71152))
    # Adding element type (key, value) (line 522)
    str_71153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 25), 'str', 'outvarname')
    # Getting the type of 'out_a' (line 522)
    out_a_71154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 39), 'out_a')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 10), dict_71150, (str_71153, out_a_71154))
    # Adding element type (key, value) (line 522)
    str_71155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 46), 'str', 'ctype')
    
    # Call to getctype(...): (line 522)
    # Processing the call arguments (line 522)
    # Getting the type of 'var' (line 522)
    var_71157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 64), 'var', False)
    # Processing the call keyword arguments (line 522)
    kwargs_71158 = {}
    # Getting the type of 'getctype' (line 522)
    getctype_71156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 55), 'getctype', False)
    # Calling getctype(args, kwargs) (line 522)
    getctype_call_result_71159 = invoke(stypy.reporting.localization.Localization(__file__, 522, 55), getctype_71156, *[var_71157], **kwargs_71158)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 10), dict_71150, (str_71155, getctype_call_result_71159))
    
    # Assigning a type to the variable 'ret' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), 'ret', dict_71150)
    
    # Assigning a List to a Name (line 523):
    
    # Assigning a List to a Name (line 523):
    
    # Obtaining an instance of the builtin type 'list' (line 523)
    list_71160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 523)
    
    # Assigning a type to the variable 'intent_flags' (line 523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), 'intent_flags', list_71160)
    
    
    # Call to items(...): (line 524)
    # Processing the call keyword arguments (line 524)
    kwargs_71163 = {}
    # Getting the type of 'isintent_dict' (line 524)
    isintent_dict_71161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 16), 'isintent_dict', False)
    # Obtaining the member 'items' of a type (line 524)
    items_71162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 16), isintent_dict_71161, 'items')
    # Calling items(args, kwargs) (line 524)
    items_call_result_71164 = invoke(stypy.reporting.localization.Localization(__file__, 524, 16), items_71162, *[], **kwargs_71163)
    
    # Testing the type of a for loop iterable (line 524)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 524, 4), items_call_result_71164)
    # Getting the type of the for loop variable (line 524)
    for_loop_var_71165 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 524, 4), items_call_result_71164)
    # Assigning a type to the variable 'f' (line 524)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 4), 'f', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 4), for_loop_var_71165))
    # Assigning a type to the variable 's' (line 524)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 4), 's', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 4), for_loop_var_71165))
    # SSA begins for a for statement (line 524)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to f(...): (line 525)
    # Processing the call arguments (line 525)
    # Getting the type of 'var' (line 525)
    var_71167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 13), 'var', False)
    # Processing the call keyword arguments (line 525)
    kwargs_71168 = {}
    # Getting the type of 'f' (line 525)
    f_71166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 11), 'f', False)
    # Calling f(args, kwargs) (line 525)
    f_call_result_71169 = invoke(stypy.reporting.localization.Localization(__file__, 525, 11), f_71166, *[var_71167], **kwargs_71168)
    
    # Testing the type of an if condition (line 525)
    if_condition_71170 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 525, 8), f_call_result_71169)
    # Assigning a type to the variable 'if_condition_71170' (line 525)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'if_condition_71170', if_condition_71170)
    # SSA begins for if statement (line 525)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 526)
    # Processing the call arguments (line 526)
    str_71173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 32), 'str', 'F2PY_%s')
    # Getting the type of 's' (line 526)
    s_71174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 44), 's', False)
    # Applying the binary operator '%' (line 526)
    result_mod_71175 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 32), '%', str_71173, s_71174)
    
    # Processing the call keyword arguments (line 526)
    kwargs_71176 = {}
    # Getting the type of 'intent_flags' (line 526)
    intent_flags_71171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 12), 'intent_flags', False)
    # Obtaining the member 'append' of a type (line 526)
    append_71172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 12), intent_flags_71171, 'append')
    # Calling append(args, kwargs) (line 526)
    append_call_result_71177 = invoke(stypy.reporting.localization.Localization(__file__, 526, 12), append_71172, *[result_mod_71175], **kwargs_71176)
    
    # SSA join for if statement (line 525)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'intent_flags' (line 527)
    intent_flags_71178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 7), 'intent_flags')
    # Testing the type of an if condition (line 527)
    if_condition_71179 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 527, 4), intent_flags_71178)
    # Assigning a type to the variable 'if_condition_71179' (line 527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 4), 'if_condition_71179', if_condition_71179)
    # SSA begins for if statement (line 527)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 529):
    
    # Assigning a Call to a Subscript (line 529):
    
    # Call to join(...): (line 529)
    # Processing the call arguments (line 529)
    # Getting the type of 'intent_flags' (line 529)
    intent_flags_71182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 33), 'intent_flags', False)
    # Processing the call keyword arguments (line 529)
    kwargs_71183 = {}
    str_71180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 24), 'str', '|')
    # Obtaining the member 'join' of a type (line 529)
    join_71181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 24), str_71180, 'join')
    # Calling join(args, kwargs) (line 529)
    join_call_result_71184 = invoke(stypy.reporting.localization.Localization(__file__, 529, 24), join_71181, *[intent_flags_71182], **kwargs_71183)
    
    # Getting the type of 'ret' (line 529)
    ret_71185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'ret')
    str_71186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 12), 'str', 'intent')
    # Storing an element on a container (line 529)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 8), ret_71185, (str_71186, join_call_result_71184))
    # SSA branch for the else part of an if statement (line 527)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Subscript (line 531):
    
    # Assigning a Str to a Subscript (line 531):
    str_71187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 24), 'str', 'F2PY_INTENT_IN')
    # Getting the type of 'ret' (line 531)
    ret_71188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 8), 'ret')
    str_71189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 12), 'str', 'intent')
    # Storing an element on a container (line 531)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 8), ret_71188, (str_71189, str_71187))
    # SSA join for if statement (line 527)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isarray(...): (line 532)
    # Processing the call arguments (line 532)
    # Getting the type of 'var' (line 532)
    var_71191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 15), 'var', False)
    # Processing the call keyword arguments (line 532)
    kwargs_71192 = {}
    # Getting the type of 'isarray' (line 532)
    isarray_71190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 7), 'isarray', False)
    # Calling isarray(args, kwargs) (line 532)
    isarray_call_result_71193 = invoke(stypy.reporting.localization.Localization(__file__, 532, 7), isarray_71190, *[var_71191], **kwargs_71192)
    
    # Testing the type of an if condition (line 532)
    if_condition_71194 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 532, 4), isarray_call_result_71193)
    # Assigning a type to the variable 'if_condition_71194' (line 532)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 4), 'if_condition_71194', if_condition_71194)
    # SSA begins for if statement (line 532)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Subscript (line 533):
    
    # Assigning a Str to a Subscript (line 533):
    str_71195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 28), 'str', 'N')
    # Getting the type of 'ret' (line 533)
    ret_71196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'ret')
    str_71197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 12), 'str', 'varrformat')
    # Storing an element on a container (line 533)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 8), ret_71196, (str_71197, str_71195))
    # SSA branch for the else part of an if statement (line 532)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    str_71198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 13), 'str', 'ctype')
    # Getting the type of 'ret' (line 534)
    ret_71199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 9), 'ret')
    # Obtaining the member '__getitem__' of a type (line 534)
    getitem___71200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 9), ret_71199, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 534)
    subscript_call_result_71201 = invoke(stypy.reporting.localization.Localization(__file__, 534, 9), getitem___71200, str_71198)
    
    # Getting the type of 'c2buildvalue_map' (line 534)
    c2buildvalue_map_71202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 25), 'c2buildvalue_map')
    # Applying the binary operator 'in' (line 534)
    result_contains_71203 = python_operator(stypy.reporting.localization.Localization(__file__, 534, 9), 'in', subscript_call_result_71201, c2buildvalue_map_71202)
    
    # Testing the type of an if condition (line 534)
    if_condition_71204 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 534, 9), result_contains_71203)
    # Assigning a type to the variable 'if_condition_71204' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 9), 'if_condition_71204', if_condition_71204)
    # SSA begins for if statement (line 534)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 535):
    
    # Assigning a Subscript to a Subscript (line 535):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    str_71205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 49), 'str', 'ctype')
    # Getting the type of 'ret' (line 535)
    ret_71206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 45), 'ret')
    # Obtaining the member '__getitem__' of a type (line 535)
    getitem___71207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 45), ret_71206, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 535)
    subscript_call_result_71208 = invoke(stypy.reporting.localization.Localization(__file__, 535, 45), getitem___71207, str_71205)
    
    # Getting the type of 'c2buildvalue_map' (line 535)
    c2buildvalue_map_71209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 28), 'c2buildvalue_map')
    # Obtaining the member '__getitem__' of a type (line 535)
    getitem___71210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 28), c2buildvalue_map_71209, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 535)
    subscript_call_result_71211 = invoke(stypy.reporting.localization.Localization(__file__, 535, 28), getitem___71210, subscript_call_result_71208)
    
    # Getting the type of 'ret' (line 535)
    ret_71212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'ret')
    str_71213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 12), 'str', 'varrformat')
    # Storing an element on a container (line 535)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 8), ret_71212, (str_71213, subscript_call_result_71211))
    # SSA branch for the else part of an if statement (line 534)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Subscript (line 537):
    
    # Assigning a Str to a Subscript (line 537):
    str_71214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 28), 'str', 'O')
    # Getting the type of 'ret' (line 537)
    ret_71215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'ret')
    str_71216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 12), 'str', 'varrformat')
    # Storing an element on a container (line 537)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 8), ret_71215, (str_71216, str_71214))
    # SSA join for if statement (line 534)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 532)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 538):
    
    # Assigning a Call to a Name:
    
    # Call to getinit(...): (line 538)
    # Processing the call arguments (line 538)
    # Getting the type of 'a' (line 538)
    a_71218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 43), 'a', False)
    # Getting the type of 'var' (line 538)
    var_71219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 46), 'var', False)
    # Processing the call keyword arguments (line 538)
    kwargs_71220 = {}
    # Getting the type of 'getinit' (line 538)
    getinit_71217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 35), 'getinit', False)
    # Calling getinit(args, kwargs) (line 538)
    getinit_call_result_71221 = invoke(stypy.reporting.localization.Localization(__file__, 538, 35), getinit_71217, *[a_71218, var_71219], **kwargs_71220)
    
    # Assigning a type to the variable 'call_assignment_69363' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 4), 'call_assignment_69363', getinit_call_result_71221)
    
    # Assigning a Call to a Name (line 538):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_71224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 4), 'int')
    # Processing the call keyword arguments
    kwargs_71225 = {}
    # Getting the type of 'call_assignment_69363' (line 538)
    call_assignment_69363_71222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 4), 'call_assignment_69363', False)
    # Obtaining the member '__getitem__' of a type (line 538)
    getitem___71223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 4), call_assignment_69363_71222, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_71226 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___71223, *[int_71224], **kwargs_71225)
    
    # Assigning a type to the variable 'call_assignment_69364' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 4), 'call_assignment_69364', getitem___call_result_71226)
    
    # Assigning a Name to a Subscript (line 538):
    # Getting the type of 'call_assignment_69364' (line 538)
    call_assignment_69364_71227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 4), 'call_assignment_69364')
    # Getting the type of 'ret' (line 538)
    ret_71228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 4), 'ret')
    str_71229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 8), 'str', 'init')
    # Storing an element on a container (line 538)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 538, 4), ret_71228, (str_71229, call_assignment_69364_71227))
    
    # Assigning a Call to a Name (line 538):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_71232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 4), 'int')
    # Processing the call keyword arguments
    kwargs_71233 = {}
    # Getting the type of 'call_assignment_69363' (line 538)
    call_assignment_69363_71230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 4), 'call_assignment_69363', False)
    # Obtaining the member '__getitem__' of a type (line 538)
    getitem___71231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 4), call_assignment_69363_71230, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_71234 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___71231, *[int_71232], **kwargs_71233)
    
    # Assigning a type to the variable 'call_assignment_69365' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 4), 'call_assignment_69365', getitem___call_result_71234)
    
    # Assigning a Name to a Subscript (line 538):
    # Getting the type of 'call_assignment_69365' (line 538)
    call_assignment_69365_71235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 4), 'call_assignment_69365')
    # Getting the type of 'ret' (line 538)
    ret_71236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 17), 'ret')
    str_71237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 21), 'str', 'showinit')
    # Storing an element on a container (line 538)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 538, 17), ret_71236, (str_71237, call_assignment_69365_71235))
    
    
    # Evaluating a boolean operation
    
    # Call to hasinitvalue(...): (line 539)
    # Processing the call arguments (line 539)
    # Getting the type of 'var' (line 539)
    var_71239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 20), 'var', False)
    # Processing the call keyword arguments (line 539)
    kwargs_71240 = {}
    # Getting the type of 'hasinitvalue' (line 539)
    hasinitvalue_71238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 7), 'hasinitvalue', False)
    # Calling hasinitvalue(args, kwargs) (line 539)
    hasinitvalue_call_result_71241 = invoke(stypy.reporting.localization.Localization(__file__, 539, 7), hasinitvalue_71238, *[var_71239], **kwargs_71240)
    
    
    # Call to iscomplex(...): (line 539)
    # Processing the call arguments (line 539)
    # Getting the type of 'var' (line 539)
    var_71243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 39), 'var', False)
    # Processing the call keyword arguments (line 539)
    kwargs_71244 = {}
    # Getting the type of 'iscomplex' (line 539)
    iscomplex_71242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 29), 'iscomplex', False)
    # Calling iscomplex(args, kwargs) (line 539)
    iscomplex_call_result_71245 = invoke(stypy.reporting.localization.Localization(__file__, 539, 29), iscomplex_71242, *[var_71243], **kwargs_71244)
    
    # Applying the binary operator 'and' (line 539)
    result_and_keyword_71246 = python_operator(stypy.reporting.localization.Localization(__file__, 539, 7), 'and', hasinitvalue_call_result_71241, iscomplex_call_result_71245)
    
    
    # Call to isarray(...): (line 539)
    # Processing the call arguments (line 539)
    # Getting the type of 'var' (line 539)
    var_71248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 60), 'var', False)
    # Processing the call keyword arguments (line 539)
    kwargs_71249 = {}
    # Getting the type of 'isarray' (line 539)
    isarray_71247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 52), 'isarray', False)
    # Calling isarray(args, kwargs) (line 539)
    isarray_call_result_71250 = invoke(stypy.reporting.localization.Localization(__file__, 539, 52), isarray_71247, *[var_71248], **kwargs_71249)
    
    # Applying the 'not' unary operator (line 539)
    result_not__71251 = python_operator(stypy.reporting.localization.Localization(__file__, 539, 48), 'not', isarray_call_result_71250)
    
    # Applying the binary operator 'and' (line 539)
    result_and_keyword_71252 = python_operator(stypy.reporting.localization.Localization(__file__, 539, 7), 'and', result_and_keyword_71246, result_not__71251)
    
    # Testing the type of an if condition (line 539)
    if_condition_71253 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 539, 4), result_and_keyword_71252)
    # Assigning a type to the variable 'if_condition_71253' (line 539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'if_condition_71253', if_condition_71253)
    # SSA begins for if statement (line 539)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 540):
    
    # Assigning a Call to a Name:
    
    # Call to split(...): (line 540)
    # Processing the call arguments (line 540)
    str_71267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 37), 'str', '@,@')
    # Processing the call keyword arguments (line 540)
    kwargs_71268 = {}
    
    # Call to markoutercomma(...): (line 540)
    # Processing the call arguments (line 540)
    
    # Obtaining the type of the subscript
    int_71255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 24), 'int')
    int_71256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 26), 'int')
    slice_71257 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 541, 12), int_71255, int_71256, None)
    
    # Obtaining the type of the subscript
    str_71258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 16), 'str', 'init')
    # Getting the type of 'ret' (line 541)
    ret_71259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 12), 'ret', False)
    # Obtaining the member '__getitem__' of a type (line 541)
    getitem___71260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 12), ret_71259, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 541)
    subscript_call_result_71261 = invoke(stypy.reporting.localization.Localization(__file__, 541, 12), getitem___71260, str_71258)
    
    # Obtaining the member '__getitem__' of a type (line 541)
    getitem___71262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 12), subscript_call_result_71261, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 541)
    subscript_call_result_71263 = invoke(stypy.reporting.localization.Localization(__file__, 541, 12), getitem___71262, slice_71257)
    
    # Processing the call keyword arguments (line 540)
    kwargs_71264 = {}
    # Getting the type of 'markoutercomma' (line 540)
    markoutercomma_71254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 39), 'markoutercomma', False)
    # Calling markoutercomma(args, kwargs) (line 540)
    markoutercomma_call_result_71265 = invoke(stypy.reporting.localization.Localization(__file__, 540, 39), markoutercomma_71254, *[subscript_call_result_71263], **kwargs_71264)
    
    # Obtaining the member 'split' of a type (line 540)
    split_71266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 39), markoutercomma_call_result_71265, 'split')
    # Calling split(args, kwargs) (line 540)
    split_call_result_71269 = invoke(stypy.reporting.localization.Localization(__file__, 540, 39), split_71266, *[str_71267], **kwargs_71268)
    
    # Assigning a type to the variable 'call_assignment_69366' (line 540)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'call_assignment_69366', split_call_result_71269)
    
    # Assigning a Call to a Name (line 540):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_71272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 8), 'int')
    # Processing the call keyword arguments
    kwargs_71273 = {}
    # Getting the type of 'call_assignment_69366' (line 540)
    call_assignment_69366_71270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'call_assignment_69366', False)
    # Obtaining the member '__getitem__' of a type (line 540)
    getitem___71271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 8), call_assignment_69366_71270, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_71274 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___71271, *[int_71272], **kwargs_71273)
    
    # Assigning a type to the variable 'call_assignment_69367' (line 540)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'call_assignment_69367', getitem___call_result_71274)
    
    # Assigning a Name to a Subscript (line 540):
    # Getting the type of 'call_assignment_69367' (line 540)
    call_assignment_69367_71275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'call_assignment_69367')
    # Getting the type of 'ret' (line 540)
    ret_71276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'ret')
    str_71277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 12), 'str', 'init.r')
    # Storing an element on a container (line 540)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 540, 8), ret_71276, (str_71277, call_assignment_69367_71275))
    
    # Assigning a Call to a Name (line 540):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_71280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 8), 'int')
    # Processing the call keyword arguments
    kwargs_71281 = {}
    # Getting the type of 'call_assignment_69366' (line 540)
    call_assignment_69366_71278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'call_assignment_69366', False)
    # Obtaining the member '__getitem__' of a type (line 540)
    getitem___71279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 8), call_assignment_69366_71278, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_71282 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___71279, *[int_71280], **kwargs_71281)
    
    # Assigning a type to the variable 'call_assignment_69368' (line 540)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'call_assignment_69368', getitem___call_result_71282)
    
    # Assigning a Name to a Subscript (line 540):
    # Getting the type of 'call_assignment_69368' (line 540)
    call_assignment_69368_71283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'call_assignment_69368')
    # Getting the type of 'ret' (line 540)
    ret_71284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 23), 'ret')
    str_71285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 27), 'str', 'init.i')
    # Storing an element on a container (line 540)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 540, 23), ret_71284, (str_71285, call_assignment_69368_71283))
    # SSA join for if statement (line 539)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isexternal(...): (line 542)
    # Processing the call arguments (line 542)
    # Getting the type of 'var' (line 542)
    var_71287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 18), 'var', False)
    # Processing the call keyword arguments (line 542)
    kwargs_71288 = {}
    # Getting the type of 'isexternal' (line 542)
    isexternal_71286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 7), 'isexternal', False)
    # Calling isexternal(args, kwargs) (line 542)
    isexternal_call_result_71289 = invoke(stypy.reporting.localization.Localization(__file__, 542, 7), isexternal_71286, *[var_71287], **kwargs_71288)
    
    # Testing the type of an if condition (line 542)
    if_condition_71290 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 542, 4), isexternal_call_result_71289)
    # Assigning a type to the variable 'if_condition_71290' (line 542)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 4), 'if_condition_71290', if_condition_71290)
    # SSA begins for if statement (line 542)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 543):
    
    # Assigning a Name to a Subscript (line 543):
    # Getting the type of 'a' (line 543)
    a_71291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 27), 'a')
    # Getting the type of 'ret' (line 543)
    ret_71292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'ret')
    str_71293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 12), 'str', 'cbnamekey')
    # Storing an element on a container (line 543)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 543, 8), ret_71292, (str_71293, a_71291))
    
    
    # Getting the type of 'a' (line 544)
    a_71294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 11), 'a')
    # Getting the type of 'lcb_map' (line 544)
    lcb_map_71295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 16), 'lcb_map')
    # Applying the binary operator 'in' (line 544)
    result_contains_71296 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 11), 'in', a_71294, lcb_map_71295)
    
    # Testing the type of an if condition (line 544)
    if_condition_71297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 544, 8), result_contains_71296)
    # Assigning a type to the variable 'if_condition_71297' (line 544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 8), 'if_condition_71297', if_condition_71297)
    # SSA begins for if statement (line 544)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 545):
    
    # Assigning a Subscript to a Subscript (line 545):
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 545)
    a_71298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 36), 'a')
    # Getting the type of 'lcb_map' (line 545)
    lcb_map_71299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 28), 'lcb_map')
    # Obtaining the member '__getitem__' of a type (line 545)
    getitem___71300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 28), lcb_map_71299, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 545)
    subscript_call_result_71301 = invoke(stypy.reporting.localization.Localization(__file__, 545, 28), getitem___71300, a_71298)
    
    # Getting the type of 'ret' (line 545)
    ret_71302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 12), 'ret')
    str_71303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 16), 'str', 'cbname')
    # Storing an element on a container (line 545)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 545, 12), ret_71302, (str_71303, subscript_call_result_71301))
    
    # Assigning a Subscript to a Subscript (line 546):
    
    # Assigning a Subscript to a Subscript (line 546):
    
    # Obtaining the type of the subscript
    str_71304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 53), 'str', 'maxnofargs')
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 546)
    a_71305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 49), 'a')
    # Getting the type of 'lcb_map' (line 546)
    lcb_map_71306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 41), 'lcb_map')
    # Obtaining the member '__getitem__' of a type (line 546)
    getitem___71307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 41), lcb_map_71306, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 546)
    subscript_call_result_71308 = invoke(stypy.reporting.localization.Localization(__file__, 546, 41), getitem___71307, a_71305)
    
    # Getting the type of 'lcb2_map' (line 546)
    lcb2_map_71309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 32), 'lcb2_map')
    # Obtaining the member '__getitem__' of a type (line 546)
    getitem___71310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 32), lcb2_map_71309, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 546)
    subscript_call_result_71311 = invoke(stypy.reporting.localization.Localization(__file__, 546, 32), getitem___71310, subscript_call_result_71308)
    
    # Obtaining the member '__getitem__' of a type (line 546)
    getitem___71312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 32), subscript_call_result_71311, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 546)
    subscript_call_result_71313 = invoke(stypy.reporting.localization.Localization(__file__, 546, 32), getitem___71312, str_71304)
    
    # Getting the type of 'ret' (line 546)
    ret_71314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 12), 'ret')
    str_71315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 16), 'str', 'maxnofargs')
    # Storing an element on a container (line 546)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 12), ret_71314, (str_71315, subscript_call_result_71313))
    
    # Assigning a Subscript to a Subscript (line 547):
    
    # Assigning a Subscript to a Subscript (line 547):
    
    # Obtaining the type of the subscript
    str_71316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 53), 'str', 'nofoptargs')
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 547)
    a_71317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 49), 'a')
    # Getting the type of 'lcb_map' (line 547)
    lcb_map_71318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 41), 'lcb_map')
    # Obtaining the member '__getitem__' of a type (line 547)
    getitem___71319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 41), lcb_map_71318, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 547)
    subscript_call_result_71320 = invoke(stypy.reporting.localization.Localization(__file__, 547, 41), getitem___71319, a_71317)
    
    # Getting the type of 'lcb2_map' (line 547)
    lcb2_map_71321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 32), 'lcb2_map')
    # Obtaining the member '__getitem__' of a type (line 547)
    getitem___71322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 32), lcb2_map_71321, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 547)
    subscript_call_result_71323 = invoke(stypy.reporting.localization.Localization(__file__, 547, 32), getitem___71322, subscript_call_result_71320)
    
    # Obtaining the member '__getitem__' of a type (line 547)
    getitem___71324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 32), subscript_call_result_71323, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 547)
    subscript_call_result_71325 = invoke(stypy.reporting.localization.Localization(__file__, 547, 32), getitem___71324, str_71316)
    
    # Getting the type of 'ret' (line 547)
    ret_71326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 12), 'ret')
    str_71327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 16), 'str', 'nofoptargs')
    # Storing an element on a container (line 547)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 547, 12), ret_71326, (str_71327, subscript_call_result_71325))
    
    # Assigning a Subscript to a Subscript (line 548):
    
    # Assigning a Subscript to a Subscript (line 548):
    
    # Obtaining the type of the subscript
    str_71328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 51), 'str', 'docstr')
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 548)
    a_71329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 47), 'a')
    # Getting the type of 'lcb_map' (line 548)
    lcb_map_71330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 39), 'lcb_map')
    # Obtaining the member '__getitem__' of a type (line 548)
    getitem___71331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 39), lcb_map_71330, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 548)
    subscript_call_result_71332 = invoke(stypy.reporting.localization.Localization(__file__, 548, 39), getitem___71331, a_71329)
    
    # Getting the type of 'lcb2_map' (line 548)
    lcb2_map_71333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 30), 'lcb2_map')
    # Obtaining the member '__getitem__' of a type (line 548)
    getitem___71334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 30), lcb2_map_71333, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 548)
    subscript_call_result_71335 = invoke(stypy.reporting.localization.Localization(__file__, 548, 30), getitem___71334, subscript_call_result_71332)
    
    # Obtaining the member '__getitem__' of a type (line 548)
    getitem___71336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 30), subscript_call_result_71335, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 548)
    subscript_call_result_71337 = invoke(stypy.reporting.localization.Localization(__file__, 548, 30), getitem___71336, str_71328)
    
    # Getting the type of 'ret' (line 548)
    ret_71338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 12), 'ret')
    str_71339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 16), 'str', 'cbdocstr')
    # Storing an element on a container (line 548)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 548, 12), ret_71338, (str_71339, subscript_call_result_71337))
    
    # Assigning a Subscript to a Subscript (line 549):
    
    # Assigning a Subscript to a Subscript (line 549):
    
    # Obtaining the type of the subscript
    str_71340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 56), 'str', 'latexdocstr')
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 549)
    a_71341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 52), 'a')
    # Getting the type of 'lcb_map' (line 549)
    lcb_map_71342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 44), 'lcb_map')
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___71343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 44), lcb_map_71342, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 549)
    subscript_call_result_71344 = invoke(stypy.reporting.localization.Localization(__file__, 549, 44), getitem___71343, a_71341)
    
    # Getting the type of 'lcb2_map' (line 549)
    lcb2_map_71345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 35), 'lcb2_map')
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___71346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 35), lcb2_map_71345, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 549)
    subscript_call_result_71347 = invoke(stypy.reporting.localization.Localization(__file__, 549, 35), getitem___71346, subscript_call_result_71344)
    
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___71348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 35), subscript_call_result_71347, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 549)
    subscript_call_result_71349 = invoke(stypy.reporting.localization.Localization(__file__, 549, 35), getitem___71348, str_71340)
    
    # Getting the type of 'ret' (line 549)
    ret_71350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'ret')
    str_71351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 16), 'str', 'cblatexdocstr')
    # Storing an element on a container (line 549)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 549, 12), ret_71350, (str_71351, subscript_call_result_71349))
    # SSA branch for the else part of an if statement (line 544)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Subscript (line 551):
    
    # Assigning a Name to a Subscript (line 551):
    # Getting the type of 'a' (line 551)
    a_71352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 28), 'a')
    # Getting the type of 'ret' (line 551)
    ret_71353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), 'ret')
    str_71354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 16), 'str', 'cbname')
    # Storing an element on a container (line 551)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 551, 12), ret_71353, (str_71354, a_71352))
    
    # Call to errmess(...): (line 552)
    # Processing the call arguments (line 552)
    str_71356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 20), 'str', 'sign2map: Confused: external %s is not in lcb_map%s.\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 553)
    tuple_71357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 553)
    # Adding element type (line 553)
    # Getting the type of 'a' (line 553)
    a_71358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 16), 'a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 553, 16), tuple_71357, a_71358)
    # Adding element type (line 553)
    
    # Call to list(...): (line 553)
    # Processing the call arguments (line 553)
    
    # Call to keys(...): (line 553)
    # Processing the call keyword arguments (line 553)
    kwargs_71362 = {}
    # Getting the type of 'lcb_map' (line 553)
    lcb_map_71360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 24), 'lcb_map', False)
    # Obtaining the member 'keys' of a type (line 553)
    keys_71361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 24), lcb_map_71360, 'keys')
    # Calling keys(args, kwargs) (line 553)
    keys_call_result_71363 = invoke(stypy.reporting.localization.Localization(__file__, 553, 24), keys_71361, *[], **kwargs_71362)
    
    # Processing the call keyword arguments (line 553)
    kwargs_71364 = {}
    # Getting the type of 'list' (line 553)
    list_71359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 19), 'list', False)
    # Calling list(args, kwargs) (line 553)
    list_call_result_71365 = invoke(stypy.reporting.localization.Localization(__file__, 553, 19), list_71359, *[keys_call_result_71363], **kwargs_71364)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 553, 16), tuple_71357, list_call_result_71365)
    
    # Applying the binary operator '%' (line 552)
    result_mod_71366 = python_operator(stypy.reporting.localization.Localization(__file__, 552, 20), '%', str_71356, tuple_71357)
    
    # Processing the call keyword arguments (line 552)
    kwargs_71367 = {}
    # Getting the type of 'errmess' (line 552)
    errmess_71355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 12), 'errmess', False)
    # Calling errmess(args, kwargs) (line 552)
    errmess_call_result_71368 = invoke(stypy.reporting.localization.Localization(__file__, 552, 12), errmess_71355, *[result_mod_71366], **kwargs_71367)
    
    # SSA join for if statement (line 544)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 542)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isstring(...): (line 554)
    # Processing the call arguments (line 554)
    # Getting the type of 'var' (line 554)
    var_71370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 16), 'var', False)
    # Processing the call keyword arguments (line 554)
    kwargs_71371 = {}
    # Getting the type of 'isstring' (line 554)
    isstring_71369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 7), 'isstring', False)
    # Calling isstring(args, kwargs) (line 554)
    isstring_call_result_71372 = invoke(stypy.reporting.localization.Localization(__file__, 554, 7), isstring_71369, *[var_71370], **kwargs_71371)
    
    # Testing the type of an if condition (line 554)
    if_condition_71373 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 554, 4), isstring_call_result_71372)
    # Assigning a type to the variable 'if_condition_71373' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'if_condition_71373', if_condition_71373)
    # SSA begins for if statement (line 554)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 555):
    
    # Assigning a Call to a Subscript (line 555):
    
    # Call to getstrlength(...): (line 555)
    # Processing the call arguments (line 555)
    # Getting the type of 'var' (line 555)
    var_71375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 37), 'var', False)
    # Processing the call keyword arguments (line 555)
    kwargs_71376 = {}
    # Getting the type of 'getstrlength' (line 555)
    getstrlength_71374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 24), 'getstrlength', False)
    # Calling getstrlength(args, kwargs) (line 555)
    getstrlength_call_result_71377 = invoke(stypy.reporting.localization.Localization(__file__, 555, 24), getstrlength_71374, *[var_71375], **kwargs_71376)
    
    # Getting the type of 'ret' (line 555)
    ret_71378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 8), 'ret')
    str_71379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 12), 'str', 'length')
    # Storing an element on a container (line 555)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 555, 8), ret_71378, (str_71379, getstrlength_call_result_71377))
    # SSA join for if statement (line 554)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isarray(...): (line 556)
    # Processing the call arguments (line 556)
    # Getting the type of 'var' (line 556)
    var_71381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 15), 'var', False)
    # Processing the call keyword arguments (line 556)
    kwargs_71382 = {}
    # Getting the type of 'isarray' (line 556)
    isarray_71380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 7), 'isarray', False)
    # Calling isarray(args, kwargs) (line 556)
    isarray_call_result_71383 = invoke(stypy.reporting.localization.Localization(__file__, 556, 7), isarray_71380, *[var_71381], **kwargs_71382)
    
    # Testing the type of an if condition (line 556)
    if_condition_71384 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 556, 4), isarray_call_result_71383)
    # Assigning a type to the variable 'if_condition_71384' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 4), 'if_condition_71384', if_condition_71384)
    # SSA begins for if statement (line 556)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 557):
    
    # Assigning a Call to a Name (line 557):
    
    # Call to dictappend(...): (line 557)
    # Processing the call arguments (line 557)
    # Getting the type of 'ret' (line 557)
    ret_71386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 25), 'ret', False)
    
    # Call to getarrdims(...): (line 557)
    # Processing the call arguments (line 557)
    # Getting the type of 'a' (line 557)
    a_71388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 41), 'a', False)
    # Getting the type of 'var' (line 557)
    var_71389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 44), 'var', False)
    # Processing the call keyword arguments (line 557)
    kwargs_71390 = {}
    # Getting the type of 'getarrdims' (line 557)
    getarrdims_71387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 30), 'getarrdims', False)
    # Calling getarrdims(args, kwargs) (line 557)
    getarrdims_call_result_71391 = invoke(stypy.reporting.localization.Localization(__file__, 557, 30), getarrdims_71387, *[a_71388, var_71389], **kwargs_71390)
    
    # Processing the call keyword arguments (line 557)
    kwargs_71392 = {}
    # Getting the type of 'dictappend' (line 557)
    dictappend_71385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 14), 'dictappend', False)
    # Calling dictappend(args, kwargs) (line 557)
    dictappend_call_result_71393 = invoke(stypy.reporting.localization.Localization(__file__, 557, 14), dictappend_71385, *[ret_71386, getarrdims_call_result_71391], **kwargs_71392)
    
    # Assigning a type to the variable 'ret' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'ret', dictappend_call_result_71393)
    
    # Assigning a Call to a Name (line 558):
    
    # Assigning a Call to a Name (line 558):
    
    # Call to copy(...): (line 558)
    # Processing the call arguments (line 558)
    
    # Obtaining the type of the subscript
    str_71396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 28), 'str', 'dimension')
    # Getting the type of 'var' (line 558)
    var_71397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 24), 'var', False)
    # Obtaining the member '__getitem__' of a type (line 558)
    getitem___71398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 24), var_71397, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 558)
    subscript_call_result_71399 = invoke(stypy.reporting.localization.Localization(__file__, 558, 24), getitem___71398, str_71396)
    
    # Processing the call keyword arguments (line 558)
    kwargs_71400 = {}
    # Getting the type of 'copy' (line 558)
    copy_71394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 14), 'copy', False)
    # Obtaining the member 'copy' of a type (line 558)
    copy_71395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 14), copy_71394, 'copy')
    # Calling copy(args, kwargs) (line 558)
    copy_call_result_71401 = invoke(stypy.reporting.localization.Localization(__file__, 558, 14), copy_71395, *[subscript_call_result_71399], **kwargs_71400)
    
    # Assigning a type to the variable 'dim' (line 558)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'dim', copy_call_result_71401)
    # SSA join for if statement (line 556)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    str_71402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 11), 'str', 'ctype')
    # Getting the type of 'ret' (line 559)
    ret_71403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 7), 'ret')
    # Obtaining the member '__getitem__' of a type (line 559)
    getitem___71404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 7), ret_71403, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 559)
    subscript_call_result_71405 = invoke(stypy.reporting.localization.Localization(__file__, 559, 7), getitem___71404, str_71402)
    
    # Getting the type of 'c2capi_map' (line 559)
    c2capi_map_71406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 23), 'c2capi_map')
    # Applying the binary operator 'in' (line 559)
    result_contains_71407 = python_operator(stypy.reporting.localization.Localization(__file__, 559, 7), 'in', subscript_call_result_71405, c2capi_map_71406)
    
    # Testing the type of an if condition (line 559)
    if_condition_71408 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 559, 4), result_contains_71407)
    # Assigning a type to the variable 'if_condition_71408' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'if_condition_71408', if_condition_71408)
    # SSA begins for if statement (line 559)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 560):
    
    # Assigning a Subscript to a Subscript (line 560):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    str_71409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 38), 'str', 'ctype')
    # Getting the type of 'ret' (line 560)
    ret_71410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 34), 'ret')
    # Obtaining the member '__getitem__' of a type (line 560)
    getitem___71411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 34), ret_71410, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 560)
    subscript_call_result_71412 = invoke(stypy.reporting.localization.Localization(__file__, 560, 34), getitem___71411, str_71409)
    
    # Getting the type of 'c2capi_map' (line 560)
    c2capi_map_71413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 23), 'c2capi_map')
    # Obtaining the member '__getitem__' of a type (line 560)
    getitem___71414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 23), c2capi_map_71413, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 560)
    subscript_call_result_71415 = invoke(stypy.reporting.localization.Localization(__file__, 560, 23), getitem___71414, subscript_call_result_71412)
    
    # Getting the type of 'ret' (line 560)
    ret_71416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'ret')
    str_71417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 12), 'str', 'atype')
    # Storing an element on a container (line 560)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 8), ret_71416, (str_71417, subscript_call_result_71415))
    # SSA join for if statement (line 559)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to debugcapi(...): (line 562)
    # Processing the call arguments (line 562)
    # Getting the type of 'var' (line 562)
    var_71419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 17), 'var', False)
    # Processing the call keyword arguments (line 562)
    kwargs_71420 = {}
    # Getting the type of 'debugcapi' (line 562)
    debugcapi_71418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 7), 'debugcapi', False)
    # Calling debugcapi(args, kwargs) (line 562)
    debugcapi_call_result_71421 = invoke(stypy.reporting.localization.Localization(__file__, 562, 7), debugcapi_71418, *[var_71419], **kwargs_71420)
    
    # Testing the type of an if condition (line 562)
    if_condition_71422 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 562, 4), debugcapi_call_result_71421)
    # Assigning a type to the variable 'if_condition_71422' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'if_condition_71422', if_condition_71422)
    # SSA begins for if statement (line 562)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 563):
    
    # Assigning a List to a Name (line 563):
    
    # Obtaining an instance of the builtin type 'list' (line 563)
    list_71423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 563)
    # Adding element type (line 563)
    # Getting the type of 'isintent_in' (line 563)
    isintent_in_71424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 14), 'isintent_in')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, isintent_in_71424)
    # Adding element type (line 563)
    str_71425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 27), 'str', 'input')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, str_71425)
    # Adding element type (line 563)
    # Getting the type of 'isintent_out' (line 563)
    isintent_out_71426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 36), 'isintent_out')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, isintent_out_71426)
    # Adding element type (line 563)
    str_71427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 50), 'str', 'output')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, str_71427)
    # Adding element type (line 563)
    # Getting the type of 'isintent_inout' (line 564)
    isintent_inout_71428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 14), 'isintent_inout')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, isintent_inout_71428)
    # Adding element type (line 563)
    str_71429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 30), 'str', 'inoutput')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, str_71429)
    # Adding element type (line 563)
    # Getting the type of 'isrequired' (line 564)
    isrequired_71430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 42), 'isrequired')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, isrequired_71430)
    # Adding element type (line 563)
    str_71431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 54), 'str', 'required')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, str_71431)
    # Adding element type (line 563)
    # Getting the type of 'isoptional' (line 565)
    isoptional_71432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 14), 'isoptional')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, isoptional_71432)
    # Adding element type (line 563)
    str_71433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 26), 'str', 'optional')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, str_71433)
    # Adding element type (line 563)
    # Getting the type of 'isintent_hide' (line 565)
    isintent_hide_71434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 38), 'isintent_hide')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, isintent_hide_71434)
    # Adding element type (line 563)
    str_71435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 53), 'str', 'hidden')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, str_71435)
    # Adding element type (line 563)
    # Getting the type of 'iscomplex' (line 566)
    iscomplex_71436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 14), 'iscomplex')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, iscomplex_71436)
    # Adding element type (line 563)
    str_71437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 25), 'str', 'complex scalar')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, str_71437)
    # Adding element type (line 563)
    
    # Call to l_and(...): (line 567)
    # Processing the call arguments (line 567)
    # Getting the type of 'isscalar' (line 567)
    isscalar_71439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 20), 'isscalar', False)
    
    # Call to l_not(...): (line 567)
    # Processing the call arguments (line 567)
    # Getting the type of 'iscomplex' (line 567)
    iscomplex_71441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 36), 'iscomplex', False)
    # Processing the call keyword arguments (line 567)
    kwargs_71442 = {}
    # Getting the type of 'l_not' (line 567)
    l_not_71440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 30), 'l_not', False)
    # Calling l_not(args, kwargs) (line 567)
    l_not_call_result_71443 = invoke(stypy.reporting.localization.Localization(__file__, 567, 30), l_not_71440, *[iscomplex_71441], **kwargs_71442)
    
    # Processing the call keyword arguments (line 567)
    kwargs_71444 = {}
    # Getting the type of 'l_and' (line 567)
    l_and_71438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 14), 'l_and', False)
    # Calling l_and(args, kwargs) (line 567)
    l_and_call_result_71445 = invoke(stypy.reporting.localization.Localization(__file__, 567, 14), l_and_71438, *[isscalar_71439, l_not_call_result_71443], **kwargs_71444)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, l_and_call_result_71445)
    # Adding element type (line 563)
    str_71446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 49), 'str', 'scalar')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, str_71446)
    # Adding element type (line 563)
    # Getting the type of 'isstring' (line 568)
    isstring_71447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 14), 'isstring')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, isstring_71447)
    # Adding element type (line 563)
    str_71448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 24), 'str', 'string')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, str_71448)
    # Adding element type (line 563)
    # Getting the type of 'isarray' (line 568)
    isarray_71449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 34), 'isarray')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, isarray_71449)
    # Adding element type (line 563)
    str_71450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 43), 'str', 'array')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, str_71450)
    # Adding element type (line 563)
    # Getting the type of 'iscomplexarray' (line 569)
    iscomplexarray_71451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 14), 'iscomplexarray')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, iscomplexarray_71451)
    # Adding element type (line 563)
    str_71452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 30), 'str', 'complex array')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, str_71452)
    # Adding element type (line 563)
    # Getting the type of 'isstringarray' (line 569)
    isstringarray_71453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 47), 'isstringarray')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, isstringarray_71453)
    # Adding element type (line 563)
    str_71454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 62), 'str', 'string array')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, str_71454)
    # Adding element type (line 563)
    # Getting the type of 'iscomplexfunction' (line 570)
    iscomplexfunction_71455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 14), 'iscomplexfunction')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, iscomplexfunction_71455)
    # Adding element type (line 563)
    str_71456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 33), 'str', 'complex function')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, str_71456)
    # Adding element type (line 563)
    
    # Call to l_and(...): (line 571)
    # Processing the call arguments (line 571)
    # Getting the type of 'isfunction' (line 571)
    isfunction_71458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 20), 'isfunction', False)
    
    # Call to l_not(...): (line 571)
    # Processing the call arguments (line 571)
    # Getting the type of 'iscomplexfunction' (line 571)
    iscomplexfunction_71460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 38), 'iscomplexfunction', False)
    # Processing the call keyword arguments (line 571)
    kwargs_71461 = {}
    # Getting the type of 'l_not' (line 571)
    l_not_71459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 32), 'l_not', False)
    # Calling l_not(args, kwargs) (line 571)
    l_not_call_result_71462 = invoke(stypy.reporting.localization.Localization(__file__, 571, 32), l_not_71459, *[iscomplexfunction_71460], **kwargs_71461)
    
    # Processing the call keyword arguments (line 571)
    kwargs_71463 = {}
    # Getting the type of 'l_and' (line 571)
    l_and_71457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 14), 'l_and', False)
    # Calling l_and(args, kwargs) (line 571)
    l_and_call_result_71464 = invoke(stypy.reporting.localization.Localization(__file__, 571, 14), l_and_71457, *[isfunction_71458, l_not_call_result_71462], **kwargs_71463)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, l_and_call_result_71464)
    # Adding element type (line 563)
    str_71465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 59), 'str', 'function')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, str_71465)
    # Adding element type (line 563)
    # Getting the type of 'isexternal' (line 572)
    isexternal_71466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 14), 'isexternal')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, isexternal_71466)
    # Adding element type (line 563)
    str_71467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 26), 'str', 'callback')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, str_71467)
    # Adding element type (line 563)
    # Getting the type of 'isintent_callback' (line 573)
    isintent_callback_71468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 14), 'isintent_callback')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, isintent_callback_71468)
    # Adding element type (line 563)
    str_71469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 33), 'str', 'callback')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, str_71469)
    # Adding element type (line 563)
    # Getting the type of 'isintent_aux' (line 574)
    isintent_aux_71470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 14), 'isintent_aux')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, isintent_aux_71470)
    # Adding element type (line 563)
    str_71471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 28), 'str', 'auxiliary')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 13), list_71423, str_71471)
    
    # Assigning a type to the variable 'il' (line 563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), 'il', list_71423)
    
    # Assigning a List to a Name (line 576):
    
    # Assigning a List to a Name (line 576):
    
    # Obtaining an instance of the builtin type 'list' (line 576)
    list_71472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 576)
    
    # Assigning a type to the variable 'rl' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 8), 'rl', list_71472)
    
    
    # Call to range(...): (line 577)
    # Processing the call arguments (line 577)
    int_71474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 23), 'int')
    
    # Call to len(...): (line 577)
    # Processing the call arguments (line 577)
    # Getting the type of 'il' (line 577)
    il_71476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 30), 'il', False)
    # Processing the call keyword arguments (line 577)
    kwargs_71477 = {}
    # Getting the type of 'len' (line 577)
    len_71475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 26), 'len', False)
    # Calling len(args, kwargs) (line 577)
    len_call_result_71478 = invoke(stypy.reporting.localization.Localization(__file__, 577, 26), len_71475, *[il_71476], **kwargs_71477)
    
    int_71479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 35), 'int')
    # Processing the call keyword arguments (line 577)
    kwargs_71480 = {}
    # Getting the type of 'range' (line 577)
    range_71473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 17), 'range', False)
    # Calling range(args, kwargs) (line 577)
    range_call_result_71481 = invoke(stypy.reporting.localization.Localization(__file__, 577, 17), range_71473, *[int_71474, len_call_result_71478, int_71479], **kwargs_71480)
    
    # Testing the type of a for loop iterable (line 577)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 577, 8), range_call_result_71481)
    # Getting the type of the for loop variable (line 577)
    for_loop_var_71482 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 577, 8), range_call_result_71481)
    # Assigning a type to the variable 'i' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 8), 'i', for_loop_var_71482)
    # SSA begins for a for statement (line 577)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to (...): (line 578)
    # Processing the call arguments (line 578)
    # Getting the type of 'var' (line 578)
    var_71487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 21), 'var', False)
    # Processing the call keyword arguments (line 578)
    kwargs_71488 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 578)
    i_71483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 18), 'i', False)
    # Getting the type of 'il' (line 578)
    il_71484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 15), 'il', False)
    # Obtaining the member '__getitem__' of a type (line 578)
    getitem___71485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 15), il_71484, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 578)
    subscript_call_result_71486 = invoke(stypy.reporting.localization.Localization(__file__, 578, 15), getitem___71485, i_71483)
    
    # Calling (args, kwargs) (line 578)
    _call_result_71489 = invoke(stypy.reporting.localization.Localization(__file__, 578, 15), subscript_call_result_71486, *[var_71487], **kwargs_71488)
    
    # Testing the type of an if condition (line 578)
    if_condition_71490 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 578, 12), _call_result_71489)
    # Assigning a type to the variable 'if_condition_71490' (line 578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 12), 'if_condition_71490', if_condition_71490)
    # SSA begins for if statement (line 578)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 579)
    # Processing the call arguments (line 579)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 579)
    i_71493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 29), 'i', False)
    int_71494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 33), 'int')
    # Applying the binary operator '+' (line 579)
    result_add_71495 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 29), '+', i_71493, int_71494)
    
    # Getting the type of 'il' (line 579)
    il_71496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 26), 'il', False)
    # Obtaining the member '__getitem__' of a type (line 579)
    getitem___71497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 26), il_71496, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 579)
    subscript_call_result_71498 = invoke(stypy.reporting.localization.Localization(__file__, 579, 26), getitem___71497, result_add_71495)
    
    # Processing the call keyword arguments (line 579)
    kwargs_71499 = {}
    # Getting the type of 'rl' (line 579)
    rl_71491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 16), 'rl', False)
    # Obtaining the member 'append' of a type (line 579)
    append_71492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 16), rl_71491, 'append')
    # Calling append(args, kwargs) (line 579)
    append_call_result_71500 = invoke(stypy.reporting.localization.Localization(__file__, 579, 16), append_71492, *[subscript_call_result_71498], **kwargs_71499)
    
    # SSA join for if statement (line 578)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isstring(...): (line 580)
    # Processing the call arguments (line 580)
    # Getting the type of 'var' (line 580)
    var_71502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 20), 'var', False)
    # Processing the call keyword arguments (line 580)
    kwargs_71503 = {}
    # Getting the type of 'isstring' (line 580)
    isstring_71501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 11), 'isstring', False)
    # Calling isstring(args, kwargs) (line 580)
    isstring_call_result_71504 = invoke(stypy.reporting.localization.Localization(__file__, 580, 11), isstring_71501, *[var_71502], **kwargs_71503)
    
    # Testing the type of an if condition (line 580)
    if_condition_71505 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 580, 8), isstring_call_result_71504)
    # Assigning a type to the variable 'if_condition_71505' (line 580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 8), 'if_condition_71505', if_condition_71505)
    # SSA begins for if statement (line 580)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 581)
    # Processing the call arguments (line 581)
    str_71508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 22), 'str', 'slen(%s)=%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 581)
    tuple_71509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 581)
    # Adding element type (line 581)
    # Getting the type of 'a' (line 581)
    a_71510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 39), 'a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 581, 39), tuple_71509, a_71510)
    # Adding element type (line 581)
    
    # Obtaining the type of the subscript
    str_71511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 46), 'str', 'length')
    # Getting the type of 'ret' (line 581)
    ret_71512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 42), 'ret', False)
    # Obtaining the member '__getitem__' of a type (line 581)
    getitem___71513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 42), ret_71512, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 581)
    subscript_call_result_71514 = invoke(stypy.reporting.localization.Localization(__file__, 581, 42), getitem___71513, str_71511)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 581, 39), tuple_71509, subscript_call_result_71514)
    
    # Applying the binary operator '%' (line 581)
    result_mod_71515 = python_operator(stypy.reporting.localization.Localization(__file__, 581, 22), '%', str_71508, tuple_71509)
    
    # Processing the call keyword arguments (line 581)
    kwargs_71516 = {}
    # Getting the type of 'rl' (line 581)
    rl_71506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 12), 'rl', False)
    # Obtaining the member 'append' of a type (line 581)
    append_71507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 12), rl_71506, 'append')
    # Calling append(args, kwargs) (line 581)
    append_call_result_71517 = invoke(stypy.reporting.localization.Localization(__file__, 581, 12), append_71507, *[result_mod_71515], **kwargs_71516)
    
    # SSA join for if statement (line 580)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isarray(...): (line 582)
    # Processing the call arguments (line 582)
    # Getting the type of 'var' (line 582)
    var_71519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 19), 'var', False)
    # Processing the call keyword arguments (line 582)
    kwargs_71520 = {}
    # Getting the type of 'isarray' (line 582)
    isarray_71518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 11), 'isarray', False)
    # Calling isarray(args, kwargs) (line 582)
    isarray_call_result_71521 = invoke(stypy.reporting.localization.Localization(__file__, 582, 11), isarray_71518, *[var_71519], **kwargs_71520)
    
    # Testing the type of an if condition (line 582)
    if_condition_71522 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 582, 8), isarray_call_result_71521)
    # Assigning a type to the variable 'if_condition_71522' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 8), 'if_condition_71522', if_condition_71522)
    # SSA begins for if statement (line 582)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 583):
    
    # Assigning a Call to a Name (line 583):
    
    # Call to join(...): (line 583)
    # Processing the call arguments (line 583)
    
    # Call to map(...): (line 584)
    # Processing the call arguments (line 584)

    @norecursion
    def _stypy_temp_lambda_24(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_24'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_24', 584, 20, True)
        # Passed parameters checking function
        _stypy_temp_lambda_24.stypy_localization = localization
        _stypy_temp_lambda_24.stypy_type_of_self = None
        _stypy_temp_lambda_24.stypy_type_store = module_type_store
        _stypy_temp_lambda_24.stypy_function_name = '_stypy_temp_lambda_24'
        _stypy_temp_lambda_24.stypy_param_names_list = ['x', 'y']
        _stypy_temp_lambda_24.stypy_varargs_param_name = None
        _stypy_temp_lambda_24.stypy_kwargs_param_name = None
        _stypy_temp_lambda_24.stypy_call_defaults = defaults
        _stypy_temp_lambda_24.stypy_call_varargs = varargs
        _stypy_temp_lambda_24.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_24', ['x', 'y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_24', ['x', 'y'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        str_71526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 33), 'str', '%s|%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 584)
        tuple_71527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 584)
        # Adding element type (line 584)
        # Getting the type of 'x' (line 584)
        x_71528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 44), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 584, 44), tuple_71527, x_71528)
        # Adding element type (line 584)
        # Getting the type of 'y' (line 584)
        y_71529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 47), 'y', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 584, 44), tuple_71527, y_71529)
        
        # Applying the binary operator '%' (line 584)
        result_mod_71530 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 33), '%', str_71526, tuple_71527)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 584)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 20), 'stypy_return_type', result_mod_71530)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_24' in the type store
        # Getting the type of 'stypy_return_type' (line 584)
        stypy_return_type_71531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 20), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_71531)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_24'
        return stypy_return_type_71531

    # Assigning a type to the variable '_stypy_temp_lambda_24' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 20), '_stypy_temp_lambda_24', _stypy_temp_lambda_24)
    # Getting the type of '_stypy_temp_lambda_24' (line 584)
    _stypy_temp_lambda_24_71532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 20), '_stypy_temp_lambda_24')
    
    # Obtaining the type of the subscript
    str_71533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 55), 'str', 'dimension')
    # Getting the type of 'var' (line 584)
    var_71534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 51), 'var', False)
    # Obtaining the member '__getitem__' of a type (line 584)
    getitem___71535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 51), var_71534, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 584)
    subscript_call_result_71536 = invoke(stypy.reporting.localization.Localization(__file__, 584, 51), getitem___71535, str_71533)
    
    # Getting the type of 'dim' (line 584)
    dim_71537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 69), 'dim', False)
    # Processing the call keyword arguments (line 584)
    kwargs_71538 = {}
    # Getting the type of 'map' (line 584)
    map_71525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 16), 'map', False)
    # Calling map(args, kwargs) (line 584)
    map_call_result_71539 = invoke(stypy.reporting.localization.Localization(__file__, 584, 16), map_71525, *[_stypy_temp_lambda_24_71532, subscript_call_result_71536, dim_71537], **kwargs_71538)
    
    # Processing the call keyword arguments (line 583)
    kwargs_71540 = {}
    str_71523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 19), 'str', ',')
    # Obtaining the member 'join' of a type (line 583)
    join_71524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 19), str_71523, 'join')
    # Calling join(args, kwargs) (line 583)
    join_call_result_71541 = invoke(stypy.reporting.localization.Localization(__file__, 583, 19), join_71524, *[map_call_result_71539], **kwargs_71540)
    
    # Assigning a type to the variable 'ddim' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 12), 'ddim', join_call_result_71541)
    
    # Call to append(...): (line 585)
    # Processing the call arguments (line 585)
    str_71544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 22), 'str', 'dims(%s)')
    # Getting the type of 'ddim' (line 585)
    ddim_71545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 35), 'ddim', False)
    # Applying the binary operator '%' (line 585)
    result_mod_71546 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 22), '%', str_71544, ddim_71545)
    
    # Processing the call keyword arguments (line 585)
    kwargs_71547 = {}
    # Getting the type of 'rl' (line 585)
    rl_71542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 12), 'rl', False)
    # Obtaining the member 'append' of a type (line 585)
    append_71543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 12), rl_71542, 'append')
    # Calling append(args, kwargs) (line 585)
    append_call_result_71548 = invoke(stypy.reporting.localization.Localization(__file__, 585, 12), append_71543, *[result_mod_71546], **kwargs_71547)
    
    # SSA join for if statement (line 582)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isexternal(...): (line 586)
    # Processing the call arguments (line 586)
    # Getting the type of 'var' (line 586)
    var_71550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 22), 'var', False)
    # Processing the call keyword arguments (line 586)
    kwargs_71551 = {}
    # Getting the type of 'isexternal' (line 586)
    isexternal_71549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 11), 'isexternal', False)
    # Calling isexternal(args, kwargs) (line 586)
    isexternal_call_result_71552 = invoke(stypy.reporting.localization.Localization(__file__, 586, 11), isexternal_71549, *[var_71550], **kwargs_71551)
    
    # Testing the type of an if condition (line 586)
    if_condition_71553 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 586, 8), isexternal_call_result_71552)
    # Assigning a type to the variable 'if_condition_71553' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 8), 'if_condition_71553', if_condition_71553)
    # SSA begins for if statement (line 586)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 587):
    
    # Assigning a BinOp to a Subscript (line 587):
    str_71554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 34), 'str', 'debug-capi:%s=>%s:%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 588)
    tuple_71555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 588)
    # Adding element type (line 588)
    # Getting the type of 'a' (line 588)
    a_71556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 16), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 588, 16), tuple_71555, a_71556)
    # Adding element type (line 588)
    
    # Obtaining the type of the subscript
    str_71557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 23), 'str', 'cbname')
    # Getting the type of 'ret' (line 588)
    ret_71558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 19), 'ret')
    # Obtaining the member '__getitem__' of a type (line 588)
    getitem___71559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 19), ret_71558, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 588)
    subscript_call_result_71560 = invoke(stypy.reporting.localization.Localization(__file__, 588, 19), getitem___71559, str_71557)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 588, 16), tuple_71555, subscript_call_result_71560)
    # Adding element type (line 588)
    
    # Call to join(...): (line 588)
    # Processing the call arguments (line 588)
    # Getting the type of 'rl' (line 588)
    rl_71563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 43), 'rl', False)
    # Processing the call keyword arguments (line 588)
    kwargs_71564 = {}
    str_71561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 34), 'str', ',')
    # Obtaining the member 'join' of a type (line 588)
    join_71562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 34), str_71561, 'join')
    # Calling join(args, kwargs) (line 588)
    join_call_result_71565 = invoke(stypy.reporting.localization.Localization(__file__, 588, 34), join_71562, *[rl_71563], **kwargs_71564)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 588, 16), tuple_71555, join_call_result_71565)
    
    # Applying the binary operator '%' (line 587)
    result_mod_71566 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 34), '%', str_71554, tuple_71555)
    
    # Getting the type of 'ret' (line 587)
    ret_71567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 12), 'ret')
    str_71568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 16), 'str', 'vardebuginfo')
    # Storing an element on a container (line 587)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 587, 12), ret_71567, (str_71568, result_mod_71566))
    # SSA branch for the else part of an if statement (line 586)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Subscript (line 590):
    
    # Assigning a BinOp to a Subscript (line 590):
    str_71569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 34), 'str', 'debug-capi:%s %s=%s:%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 591)
    tuple_71570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 591)
    # Adding element type (line 591)
    
    # Obtaining the type of the subscript
    str_71571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 20), 'str', 'ctype')
    # Getting the type of 'ret' (line 591)
    ret_71572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 16), 'ret')
    # Obtaining the member '__getitem__' of a type (line 591)
    getitem___71573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 16), ret_71572, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 591)
    subscript_call_result_71574 = invoke(stypy.reporting.localization.Localization(__file__, 591, 16), getitem___71573, str_71571)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 16), tuple_71570, subscript_call_result_71574)
    # Adding element type (line 591)
    # Getting the type of 'a' (line 591)
    a_71575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 30), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 16), tuple_71570, a_71575)
    # Adding element type (line 591)
    
    # Obtaining the type of the subscript
    str_71576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 37), 'str', 'showinit')
    # Getting the type of 'ret' (line 591)
    ret_71577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 33), 'ret')
    # Obtaining the member '__getitem__' of a type (line 591)
    getitem___71578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 33), ret_71577, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 591)
    subscript_call_result_71579 = invoke(stypy.reporting.localization.Localization(__file__, 591, 33), getitem___71578, str_71576)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 16), tuple_71570, subscript_call_result_71579)
    # Adding element type (line 591)
    
    # Call to join(...): (line 591)
    # Processing the call arguments (line 591)
    # Getting the type of 'rl' (line 591)
    rl_71582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 59), 'rl', False)
    # Processing the call keyword arguments (line 591)
    kwargs_71583 = {}
    str_71580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 50), 'str', ',')
    # Obtaining the member 'join' of a type (line 591)
    join_71581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 50), str_71580, 'join')
    # Calling join(args, kwargs) (line 591)
    join_call_result_71584 = invoke(stypy.reporting.localization.Localization(__file__, 591, 50), join_71581, *[rl_71582], **kwargs_71583)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 16), tuple_71570, join_call_result_71584)
    
    # Applying the binary operator '%' (line 590)
    result_mod_71585 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 34), '%', str_71569, tuple_71570)
    
    # Getting the type of 'ret' (line 590)
    ret_71586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 12), 'ret')
    str_71587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 16), 'str', 'vardebuginfo')
    # Storing an element on a container (line 590)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 12), ret_71586, (str_71587, result_mod_71585))
    # SSA join for if statement (line 586)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isscalar(...): (line 592)
    # Processing the call arguments (line 592)
    # Getting the type of 'var' (line 592)
    var_71589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 20), 'var', False)
    # Processing the call keyword arguments (line 592)
    kwargs_71590 = {}
    # Getting the type of 'isscalar' (line 592)
    isscalar_71588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 11), 'isscalar', False)
    # Calling isscalar(args, kwargs) (line 592)
    isscalar_call_result_71591 = invoke(stypy.reporting.localization.Localization(__file__, 592, 11), isscalar_71588, *[var_71589], **kwargs_71590)
    
    # Testing the type of an if condition (line 592)
    if_condition_71592 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 592, 8), isscalar_call_result_71591)
    # Assigning a type to the variable 'if_condition_71592' (line 592)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'if_condition_71592', if_condition_71592)
    # SSA begins for if statement (line 592)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    str_71593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 19), 'str', 'ctype')
    # Getting the type of 'ret' (line 593)
    ret_71594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 15), 'ret')
    # Obtaining the member '__getitem__' of a type (line 593)
    getitem___71595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 15), ret_71594, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 593)
    subscript_call_result_71596 = invoke(stypy.reporting.localization.Localization(__file__, 593, 15), getitem___71595, str_71593)
    
    # Getting the type of 'cformat_map' (line 593)
    cformat_map_71597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 31), 'cformat_map')
    # Applying the binary operator 'in' (line 593)
    result_contains_71598 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 15), 'in', subscript_call_result_71596, cformat_map_71597)
    
    # Testing the type of an if condition (line 593)
    if_condition_71599 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 593, 12), result_contains_71598)
    # Assigning a type to the variable 'if_condition_71599' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 12), 'if_condition_71599', if_condition_71599)
    # SSA begins for if statement (line 593)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 594):
    
    # Assigning a BinOp to a Subscript (line 594):
    str_71600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 43), 'str', 'debug-capi:%s=%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 595)
    tuple_71601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 595)
    # Adding element type (line 595)
    # Getting the type of 'a' (line 595)
    a_71602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 20), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 595, 20), tuple_71601, a_71602)
    # Adding element type (line 595)
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    str_71603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 39), 'str', 'ctype')
    # Getting the type of 'ret' (line 595)
    ret_71604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 35), 'ret')
    # Obtaining the member '__getitem__' of a type (line 595)
    getitem___71605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 35), ret_71604, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 595)
    subscript_call_result_71606 = invoke(stypy.reporting.localization.Localization(__file__, 595, 35), getitem___71605, str_71603)
    
    # Getting the type of 'cformat_map' (line 595)
    cformat_map_71607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 23), 'cformat_map')
    # Obtaining the member '__getitem__' of a type (line 595)
    getitem___71608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 23), cformat_map_71607, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 595)
    subscript_call_result_71609 = invoke(stypy.reporting.localization.Localization(__file__, 595, 23), getitem___71608, subscript_call_result_71606)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 595, 20), tuple_71601, subscript_call_result_71609)
    
    # Applying the binary operator '%' (line 594)
    result_mod_71610 = python_operator(stypy.reporting.localization.Localization(__file__, 594, 43), '%', str_71600, tuple_71601)
    
    # Getting the type of 'ret' (line 594)
    ret_71611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 16), 'ret')
    str_71612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 20), 'str', 'vardebugshowvalue')
    # Storing an element on a container (line 594)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 16), ret_71611, (str_71612, result_mod_71610))
    # SSA join for if statement (line 593)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 592)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isstring(...): (line 596)
    # Processing the call arguments (line 596)
    # Getting the type of 'var' (line 596)
    var_71614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 20), 'var', False)
    # Processing the call keyword arguments (line 596)
    kwargs_71615 = {}
    # Getting the type of 'isstring' (line 596)
    isstring_71613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 11), 'isstring', False)
    # Calling isstring(args, kwargs) (line 596)
    isstring_call_result_71616 = invoke(stypy.reporting.localization.Localization(__file__, 596, 11), isstring_71613, *[var_71614], **kwargs_71615)
    
    # Testing the type of an if condition (line 596)
    if_condition_71617 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 596, 8), isstring_call_result_71616)
    # Assigning a type to the variable 'if_condition_71617' (line 596)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'if_condition_71617', if_condition_71617)
    # SSA begins for if statement (line 596)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 597):
    
    # Assigning a BinOp to a Subscript (line 597):
    str_71618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 39), 'str', 'debug-capi:slen(%s)=%%d %s=\\"%%s\\"')
    
    # Obtaining an instance of the builtin type 'tuple' (line 598)
    tuple_71619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 598)
    # Adding element type (line 598)
    # Getting the type of 'a' (line 598)
    a_71620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 16), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 16), tuple_71619, a_71620)
    # Adding element type (line 598)
    # Getting the type of 'a' (line 598)
    a_71621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 19), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 16), tuple_71619, a_71621)
    
    # Applying the binary operator '%' (line 597)
    result_mod_71622 = python_operator(stypy.reporting.localization.Localization(__file__, 597, 39), '%', str_71618, tuple_71619)
    
    # Getting the type of 'ret' (line 597)
    ret_71623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 12), 'ret')
    str_71624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 16), 'str', 'vardebugshowvalue')
    # Storing an element on a container (line 597)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 597, 12), ret_71623, (str_71624, result_mod_71622))
    # SSA join for if statement (line 596)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isexternal(...): (line 599)
    # Processing the call arguments (line 599)
    # Getting the type of 'var' (line 599)
    var_71626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 22), 'var', False)
    # Processing the call keyword arguments (line 599)
    kwargs_71627 = {}
    # Getting the type of 'isexternal' (line 599)
    isexternal_71625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 11), 'isexternal', False)
    # Calling isexternal(args, kwargs) (line 599)
    isexternal_call_result_71628 = invoke(stypy.reporting.localization.Localization(__file__, 599, 11), isexternal_71625, *[var_71626], **kwargs_71627)
    
    # Testing the type of an if condition (line 599)
    if_condition_71629 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 599, 8), isexternal_call_result_71628)
    # Assigning a type to the variable 'if_condition_71629' (line 599)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'if_condition_71629', if_condition_71629)
    # SSA begins for if statement (line 599)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 600):
    
    # Assigning a BinOp to a Subscript (line 600):
    str_71630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 39), 'str', 'debug-capi:%s=%%p')
    # Getting the type of 'a' (line 600)
    a_71631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 62), 'a')
    # Applying the binary operator '%' (line 600)
    result_mod_71632 = python_operator(stypy.reporting.localization.Localization(__file__, 600, 39), '%', str_71630, a_71631)
    
    # Getting the type of 'ret' (line 600)
    ret_71633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 12), 'ret')
    str_71634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 16), 'str', 'vardebugshowvalue')
    # Storing an element on a container (line 600)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 600, 12), ret_71633, (str_71634, result_mod_71632))
    # SSA join for if statement (line 599)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 562)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    str_71635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 11), 'str', 'ctype')
    # Getting the type of 'ret' (line 601)
    ret_71636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 7), 'ret')
    # Obtaining the member '__getitem__' of a type (line 601)
    getitem___71637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 7), ret_71636, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 601)
    subscript_call_result_71638 = invoke(stypy.reporting.localization.Localization(__file__, 601, 7), getitem___71637, str_71635)
    
    # Getting the type of 'cformat_map' (line 601)
    cformat_map_71639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 23), 'cformat_map')
    # Applying the binary operator 'in' (line 601)
    result_contains_71640 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 7), 'in', subscript_call_result_71638, cformat_map_71639)
    
    # Testing the type of an if condition (line 601)
    if_condition_71641 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 601, 4), result_contains_71640)
    # Assigning a type to the variable 'if_condition_71641' (line 601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 4), 'if_condition_71641', if_condition_71641)
    # SSA begins for if statement (line 601)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 602):
    
    # Assigning a BinOp to a Subscript (line 602):
    str_71642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 30), 'str', '#name#:%s=%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 602)
    tuple_71643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 602)
    # Adding element type (line 602)
    # Getting the type of 'a' (line 602)
    a_71644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 48), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 602, 48), tuple_71643, a_71644)
    # Adding element type (line 602)
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    str_71645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 67), 'str', 'ctype')
    # Getting the type of 'ret' (line 602)
    ret_71646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 63), 'ret')
    # Obtaining the member '__getitem__' of a type (line 602)
    getitem___71647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 63), ret_71646, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 602)
    subscript_call_result_71648 = invoke(stypy.reporting.localization.Localization(__file__, 602, 63), getitem___71647, str_71645)
    
    # Getting the type of 'cformat_map' (line 602)
    cformat_map_71649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 51), 'cformat_map')
    # Obtaining the member '__getitem__' of a type (line 602)
    getitem___71650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 51), cformat_map_71649, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 602)
    subscript_call_result_71651 = invoke(stypy.reporting.localization.Localization(__file__, 602, 51), getitem___71650, subscript_call_result_71648)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 602, 48), tuple_71643, subscript_call_result_71651)
    
    # Applying the binary operator '%' (line 602)
    result_mod_71652 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 30), '%', str_71642, tuple_71643)
    
    # Getting the type of 'ret' (line 602)
    ret_71653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 8), 'ret')
    str_71654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 12), 'str', 'varshowvalue')
    # Storing an element on a container (line 602)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 602, 8), ret_71653, (str_71654, result_mod_71652))
    
    # Assigning a BinOp to a Subscript (line 603):
    
    # Assigning a BinOp to a Subscript (line 603):
    str_71655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 33), 'str', '%s')
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    str_71656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 57), 'str', 'ctype')
    # Getting the type of 'ret' (line 603)
    ret_71657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 53), 'ret')
    # Obtaining the member '__getitem__' of a type (line 603)
    getitem___71658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 53), ret_71657, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 603)
    subscript_call_result_71659 = invoke(stypy.reporting.localization.Localization(__file__, 603, 53), getitem___71658, str_71656)
    
    # Getting the type of 'cformat_map' (line 603)
    cformat_map_71660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 41), 'cformat_map')
    # Obtaining the member '__getitem__' of a type (line 603)
    getitem___71661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 41), cformat_map_71660, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 603)
    subscript_call_result_71662 = invoke(stypy.reporting.localization.Localization(__file__, 603, 41), getitem___71661, subscript_call_result_71659)
    
    # Applying the binary operator '%' (line 603)
    result_mod_71663 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 33), '%', str_71655, subscript_call_result_71662)
    
    # Getting the type of 'ret' (line 603)
    ret_71664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 8), 'ret')
    str_71665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 12), 'str', 'showvalueformat')
    # Storing an element on a container (line 603)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 603, 8), ret_71664, (str_71665, result_mod_71663))
    # SSA join for if statement (line 601)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isstring(...): (line 604)
    # Processing the call arguments (line 604)
    # Getting the type of 'var' (line 604)
    var_71667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 16), 'var', False)
    # Processing the call keyword arguments (line 604)
    kwargs_71668 = {}
    # Getting the type of 'isstring' (line 604)
    isstring_71666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 7), 'isstring', False)
    # Calling isstring(args, kwargs) (line 604)
    isstring_call_result_71669 = invoke(stypy.reporting.localization.Localization(__file__, 604, 7), isstring_71666, *[var_71667], **kwargs_71668)
    
    # Testing the type of an if condition (line 604)
    if_condition_71670 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 604, 4), isstring_call_result_71669)
    # Assigning a type to the variable 'if_condition_71670' (line 604)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 4), 'if_condition_71670', if_condition_71670)
    # SSA begins for if statement (line 604)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 605):
    
    # Assigning a BinOp to a Subscript (line 605):
    str_71671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 30), 'str', '#name#:slen(%s)=%%d %s=\\"%%s\\"')
    
    # Obtaining an instance of the builtin type 'tuple' (line 605)
    tuple_71672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 68), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 605)
    # Adding element type (line 605)
    # Getting the type of 'a' (line 605)
    a_71673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 68), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 68), tuple_71672, a_71673)
    # Adding element type (line 605)
    # Getting the type of 'a' (line 605)
    a_71674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 71), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 68), tuple_71672, a_71674)
    
    # Applying the binary operator '%' (line 605)
    result_mod_71675 = python_operator(stypy.reporting.localization.Localization(__file__, 605, 30), '%', str_71671, tuple_71672)
    
    # Getting the type of 'ret' (line 605)
    ret_71676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'ret')
    str_71677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 12), 'str', 'varshowvalue')
    # Storing an element on a container (line 605)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 8), ret_71676, (str_71677, result_mod_71675))
    # SSA join for if statement (line 604)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 606):
    
    # Assigning a Call to a Name:
    
    # Call to getpydocsign(...): (line 606)
    # Processing the call arguments (line 606)
    # Getting the type of 'a' (line 606)
    a_71679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 57), 'a', False)
    # Getting the type of 'var' (line 606)
    var_71680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 60), 'var', False)
    # Processing the call keyword arguments (line 606)
    kwargs_71681 = {}
    # Getting the type of 'getpydocsign' (line 606)
    getpydocsign_71678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 44), 'getpydocsign', False)
    # Calling getpydocsign(args, kwargs) (line 606)
    getpydocsign_call_result_71682 = invoke(stypy.reporting.localization.Localization(__file__, 606, 44), getpydocsign_71678, *[a_71679, var_71680], **kwargs_71681)
    
    # Assigning a type to the variable 'call_assignment_69369' (line 606)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 4), 'call_assignment_69369', getpydocsign_call_result_71682)
    
    # Assigning a Call to a Name (line 606):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_71685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 4), 'int')
    # Processing the call keyword arguments
    kwargs_71686 = {}
    # Getting the type of 'call_assignment_69369' (line 606)
    call_assignment_69369_71683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 4), 'call_assignment_69369', False)
    # Obtaining the member '__getitem__' of a type (line 606)
    getitem___71684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 4), call_assignment_69369_71683, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_71687 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___71684, *[int_71685], **kwargs_71686)
    
    # Assigning a type to the variable 'call_assignment_69370' (line 606)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 4), 'call_assignment_69370', getitem___call_result_71687)
    
    # Assigning a Name to a Subscript (line 606):
    # Getting the type of 'call_assignment_69370' (line 606)
    call_assignment_69370_71688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 4), 'call_assignment_69370')
    # Getting the type of 'ret' (line 606)
    ret_71689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 4), 'ret')
    str_71690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 8), 'str', 'pydocsign')
    # Storing an element on a container (line 606)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 606, 4), ret_71689, (str_71690, call_assignment_69370_71688))
    
    # Assigning a Call to a Name (line 606):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_71693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 4), 'int')
    # Processing the call keyword arguments
    kwargs_71694 = {}
    # Getting the type of 'call_assignment_69369' (line 606)
    call_assignment_69369_71691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 4), 'call_assignment_69369', False)
    # Obtaining the member '__getitem__' of a type (line 606)
    getitem___71692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 4), call_assignment_69369_71691, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_71695 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___71692, *[int_71693], **kwargs_71694)
    
    # Assigning a type to the variable 'call_assignment_69371' (line 606)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 4), 'call_assignment_69371', getitem___call_result_71695)
    
    # Assigning a Name to a Subscript (line 606):
    # Getting the type of 'call_assignment_69371' (line 606)
    call_assignment_69371_71696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 4), 'call_assignment_69371')
    # Getting the type of 'ret' (line 606)
    ret_71697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 22), 'ret')
    str_71698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 26), 'str', 'pydocsignout')
    # Storing an element on a container (line 606)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 606, 22), ret_71697, (str_71698, call_assignment_69371_71696))
    
    
    # Call to hasnote(...): (line 607)
    # Processing the call arguments (line 607)
    # Getting the type of 'var' (line 607)
    var_71700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 15), 'var', False)
    # Processing the call keyword arguments (line 607)
    kwargs_71701 = {}
    # Getting the type of 'hasnote' (line 607)
    hasnote_71699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 7), 'hasnote', False)
    # Calling hasnote(args, kwargs) (line 607)
    hasnote_call_result_71702 = invoke(stypy.reporting.localization.Localization(__file__, 607, 7), hasnote_71699, *[var_71700], **kwargs_71701)
    
    # Testing the type of an if condition (line 607)
    if_condition_71703 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 607, 4), hasnote_call_result_71702)
    # Assigning a type to the variable 'if_condition_71703' (line 607)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 4), 'if_condition_71703', if_condition_71703)
    # SSA begins for if statement (line 607)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 608):
    
    # Assigning a Subscript to a Subscript (line 608):
    
    # Obtaining the type of the subscript
    str_71704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 26), 'str', 'note')
    # Getting the type of 'var' (line 608)
    var_71705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 22), 'var')
    # Obtaining the member '__getitem__' of a type (line 608)
    getitem___71706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 22), var_71705, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 608)
    subscript_call_result_71707 = invoke(stypy.reporting.localization.Localization(__file__, 608, 22), getitem___71706, str_71704)
    
    # Getting the type of 'ret' (line 608)
    ret_71708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 8), 'ret')
    str_71709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 12), 'str', 'note')
    # Storing an element on a container (line 608)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 608, 8), ret_71708, (str_71709, subscript_call_result_71707))
    # SSA join for if statement (line 607)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ret' (line 609)
    ret_71710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 11), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 609)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 4), 'stypy_return_type', ret_71710)
    
    # ################# End of 'sign2map(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sign2map' in the type store
    # Getting the type of 'stypy_return_type' (line 507)
    stypy_return_type_71711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_71711)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sign2map'
    return stypy_return_type_71711

# Assigning a type to the variable 'sign2map' (line 507)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 0), 'sign2map', sign2map)

@norecursion
def routsign2map(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'routsign2map'
    module_type_store = module_type_store.open_function_context('routsign2map', 612, 0, False)
    
    # Passed parameters checking function
    routsign2map.stypy_localization = localization
    routsign2map.stypy_type_of_self = None
    routsign2map.stypy_type_store = module_type_store
    routsign2map.stypy_function_name = 'routsign2map'
    routsign2map.stypy_param_names_list = ['rout']
    routsign2map.stypy_varargs_param_name = None
    routsign2map.stypy_kwargs_param_name = None
    routsign2map.stypy_call_defaults = defaults
    routsign2map.stypy_call_varargs = varargs
    routsign2map.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'routsign2map', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'routsign2map', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'routsign2map(...)' code ##################

    str_71712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, (-1)), 'str', '\n    name,NAME,begintitle,endtitle\n    rname,ctype,rformat\n    routdebugshowvalue\n    ')
    # Marking variables as global (line 618)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 618, 4), 'lcb_map')
    
    # Assigning a Subscript to a Name (line 619):
    
    # Assigning a Subscript to a Name (line 619):
    
    # Obtaining the type of the subscript
    str_71713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 16), 'str', 'name')
    # Getting the type of 'rout' (line 619)
    rout_71714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 11), 'rout')
    # Obtaining the member '__getitem__' of a type (line 619)
    getitem___71715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 11), rout_71714, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 619)
    subscript_call_result_71716 = invoke(stypy.reporting.localization.Localization(__file__, 619, 11), getitem___71715, str_71713)
    
    # Assigning a type to the variable 'name' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 4), 'name', subscript_call_result_71716)
    
    # Assigning a Call to a Name (line 620):
    
    # Assigning a Call to a Name (line 620):
    
    # Call to getfortranname(...): (line 620)
    # Processing the call arguments (line 620)
    # Getting the type of 'rout' (line 620)
    rout_71718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 27), 'rout', False)
    # Processing the call keyword arguments (line 620)
    kwargs_71719 = {}
    # Getting the type of 'getfortranname' (line 620)
    getfortranname_71717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 12), 'getfortranname', False)
    # Calling getfortranname(args, kwargs) (line 620)
    getfortranname_call_result_71720 = invoke(stypy.reporting.localization.Localization(__file__, 620, 12), getfortranname_71717, *[rout_71718], **kwargs_71719)
    
    # Assigning a type to the variable 'fname' (line 620)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 4), 'fname', getfortranname_call_result_71720)
    
    # Assigning a Dict to a Name (line 621):
    
    # Assigning a Dict to a Name (line 621):
    
    # Obtaining an instance of the builtin type 'dict' (line 621)
    dict_71721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 10), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 621)
    # Adding element type (key, value) (line 621)
    str_71722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 11), 'str', 'name')
    # Getting the type of 'name' (line 621)
    name_71723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 19), 'name')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 621, 10), dict_71721, (str_71722, name_71723))
    # Adding element type (key, value) (line 621)
    str_71724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 11), 'str', 'texname')
    
    # Call to replace(...): (line 622)
    # Processing the call arguments (line 622)
    str_71727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 35), 'str', '_')
    str_71728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 40), 'str', '\\_')
    # Processing the call keyword arguments (line 622)
    kwargs_71729 = {}
    # Getting the type of 'name' (line 622)
    name_71725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 22), 'name', False)
    # Obtaining the member 'replace' of a type (line 622)
    replace_71726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 22), name_71725, 'replace')
    # Calling replace(args, kwargs) (line 622)
    replace_call_result_71730 = invoke(stypy.reporting.localization.Localization(__file__, 622, 22), replace_71726, *[str_71727, str_71728], **kwargs_71729)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 621, 10), dict_71721, (str_71724, replace_call_result_71730))
    # Adding element type (key, value) (line 621)
    str_71731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 11), 'str', 'name_lower')
    
    # Call to lower(...): (line 623)
    # Processing the call keyword arguments (line 623)
    kwargs_71734 = {}
    # Getting the type of 'name' (line 623)
    name_71732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 25), 'name', False)
    # Obtaining the member 'lower' of a type (line 623)
    lower_71733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 25), name_71732, 'lower')
    # Calling lower(args, kwargs) (line 623)
    lower_call_result_71735 = invoke(stypy.reporting.localization.Localization(__file__, 623, 25), lower_71733, *[], **kwargs_71734)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 621, 10), dict_71721, (str_71731, lower_call_result_71735))
    # Adding element type (key, value) (line 621)
    str_71736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 11), 'str', 'NAME')
    
    # Call to upper(...): (line 624)
    # Processing the call keyword arguments (line 624)
    kwargs_71739 = {}
    # Getting the type of 'name' (line 624)
    name_71737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 19), 'name', False)
    # Obtaining the member 'upper' of a type (line 624)
    upper_71738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 19), name_71737, 'upper')
    # Calling upper(args, kwargs) (line 624)
    upper_call_result_71740 = invoke(stypy.reporting.localization.Localization(__file__, 624, 19), upper_71738, *[], **kwargs_71739)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 621, 10), dict_71721, (str_71736, upper_call_result_71740))
    # Adding element type (key, value) (line 621)
    str_71741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 11), 'str', 'begintitle')
    
    # Call to gentitle(...): (line 625)
    # Processing the call arguments (line 625)
    # Getting the type of 'name' (line 625)
    name_71743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 34), 'name', False)
    # Processing the call keyword arguments (line 625)
    kwargs_71744 = {}
    # Getting the type of 'gentitle' (line 625)
    gentitle_71742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 25), 'gentitle', False)
    # Calling gentitle(args, kwargs) (line 625)
    gentitle_call_result_71745 = invoke(stypy.reporting.localization.Localization(__file__, 625, 25), gentitle_71742, *[name_71743], **kwargs_71744)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 621, 10), dict_71721, (str_71741, gentitle_call_result_71745))
    # Adding element type (key, value) (line 621)
    str_71746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 11), 'str', 'endtitle')
    
    # Call to gentitle(...): (line 626)
    # Processing the call arguments (line 626)
    str_71748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 32), 'str', 'end of %s')
    # Getting the type of 'name' (line 626)
    name_71749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 46), 'name', False)
    # Applying the binary operator '%' (line 626)
    result_mod_71750 = python_operator(stypy.reporting.localization.Localization(__file__, 626, 32), '%', str_71748, name_71749)
    
    # Processing the call keyword arguments (line 626)
    kwargs_71751 = {}
    # Getting the type of 'gentitle' (line 626)
    gentitle_71747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 23), 'gentitle', False)
    # Calling gentitle(args, kwargs) (line 626)
    gentitle_call_result_71752 = invoke(stypy.reporting.localization.Localization(__file__, 626, 23), gentitle_71747, *[result_mod_71750], **kwargs_71751)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 621, 10), dict_71721, (str_71746, gentitle_call_result_71752))
    # Adding element type (key, value) (line 621)
    str_71753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 11), 'str', 'fortranname')
    # Getting the type of 'fname' (line 627)
    fname_71754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 26), 'fname')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 621, 10), dict_71721, (str_71753, fname_71754))
    # Adding element type (key, value) (line 621)
    str_71755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 11), 'str', 'FORTRANNAME')
    
    # Call to upper(...): (line 628)
    # Processing the call keyword arguments (line 628)
    kwargs_71758 = {}
    # Getting the type of 'fname' (line 628)
    fname_71756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 26), 'fname', False)
    # Obtaining the member 'upper' of a type (line 628)
    upper_71757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 26), fname_71756, 'upper')
    # Calling upper(args, kwargs) (line 628)
    upper_call_result_71759 = invoke(stypy.reporting.localization.Localization(__file__, 628, 26), upper_71757, *[], **kwargs_71758)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 621, 10), dict_71721, (str_71755, upper_call_result_71759))
    # Adding element type (key, value) (line 621)
    str_71760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 11), 'str', 'callstatement')
    
    # Evaluating a boolean operation
    
    # Call to getcallstatement(...): (line 629)
    # Processing the call arguments (line 629)
    # Getting the type of 'rout' (line 629)
    rout_71762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 45), 'rout', False)
    # Processing the call keyword arguments (line 629)
    kwargs_71763 = {}
    # Getting the type of 'getcallstatement' (line 629)
    getcallstatement_71761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 28), 'getcallstatement', False)
    # Calling getcallstatement(args, kwargs) (line 629)
    getcallstatement_call_result_71764 = invoke(stypy.reporting.localization.Localization(__file__, 629, 28), getcallstatement_71761, *[rout_71762], **kwargs_71763)
    
    str_71765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 54), 'str', '')
    # Applying the binary operator 'or' (line 629)
    result_or_keyword_71766 = python_operator(stypy.reporting.localization.Localization(__file__, 629, 28), 'or', getcallstatement_call_result_71764, str_71765)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 621, 10), dict_71721, (str_71760, result_or_keyword_71766))
    # Adding element type (key, value) (line 621)
    str_71767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 11), 'str', 'usercode')
    
    # Evaluating a boolean operation
    
    # Call to getusercode(...): (line 630)
    # Processing the call arguments (line 630)
    # Getting the type of 'rout' (line 630)
    rout_71769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 35), 'rout', False)
    # Processing the call keyword arguments (line 630)
    kwargs_71770 = {}
    # Getting the type of 'getusercode' (line 630)
    getusercode_71768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 23), 'getusercode', False)
    # Calling getusercode(args, kwargs) (line 630)
    getusercode_call_result_71771 = invoke(stypy.reporting.localization.Localization(__file__, 630, 23), getusercode_71768, *[rout_71769], **kwargs_71770)
    
    str_71772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 44), 'str', '')
    # Applying the binary operator 'or' (line 630)
    result_or_keyword_71773 = python_operator(stypy.reporting.localization.Localization(__file__, 630, 23), 'or', getusercode_call_result_71771, str_71772)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 621, 10), dict_71721, (str_71767, result_or_keyword_71773))
    # Adding element type (key, value) (line 621)
    str_71774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 11), 'str', 'usercode1')
    
    # Evaluating a boolean operation
    
    # Call to getusercode1(...): (line 631)
    # Processing the call arguments (line 631)
    # Getting the type of 'rout' (line 631)
    rout_71776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 37), 'rout', False)
    # Processing the call keyword arguments (line 631)
    kwargs_71777 = {}
    # Getting the type of 'getusercode1' (line 631)
    getusercode1_71775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 24), 'getusercode1', False)
    # Calling getusercode1(args, kwargs) (line 631)
    getusercode1_call_result_71778 = invoke(stypy.reporting.localization.Localization(__file__, 631, 24), getusercode1_71775, *[rout_71776], **kwargs_71777)
    
    str_71779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 46), 'str', '')
    # Applying the binary operator 'or' (line 631)
    result_or_keyword_71780 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 24), 'or', getusercode1_call_result_71778, str_71779)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 621, 10), dict_71721, (str_71774, result_or_keyword_71780))
    
    # Assigning a type to the variable 'ret' (line 621)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 4), 'ret', dict_71721)
    
    
    str_71781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 7), 'str', '_')
    # Getting the type of 'fname' (line 633)
    fname_71782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 14), 'fname')
    # Applying the binary operator 'in' (line 633)
    result_contains_71783 = python_operator(stypy.reporting.localization.Localization(__file__, 633, 7), 'in', str_71781, fname_71782)
    
    # Testing the type of an if condition (line 633)
    if_condition_71784 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 633, 4), result_contains_71783)
    # Assigning a type to the variable 'if_condition_71784' (line 633)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 4), 'if_condition_71784', if_condition_71784)
    # SSA begins for if statement (line 633)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Subscript (line 634):
    
    # Assigning a Str to a Subscript (line 634):
    str_71785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 24), 'str', 'F_FUNC_US')
    # Getting the type of 'ret' (line 634)
    ret_71786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 8), 'ret')
    str_71787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 12), 'str', 'F_FUNC')
    # Storing an element on a container (line 634)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 634, 8), ret_71786, (str_71787, str_71785))
    # SSA branch for the else part of an if statement (line 633)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Subscript (line 636):
    
    # Assigning a Str to a Subscript (line 636):
    str_71788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 24), 'str', 'F_FUNC')
    # Getting the type of 'ret' (line 636)
    ret_71789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 8), 'ret')
    str_71790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 12), 'str', 'F_FUNC')
    # Storing an element on a container (line 636)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 636, 8), ret_71789, (str_71790, str_71788))
    # SSA join for if statement (line 633)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_71791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 7), 'str', '_')
    # Getting the type of 'name' (line 637)
    name_71792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 14), 'name')
    # Applying the binary operator 'in' (line 637)
    result_contains_71793 = python_operator(stypy.reporting.localization.Localization(__file__, 637, 7), 'in', str_71791, name_71792)
    
    # Testing the type of an if condition (line 637)
    if_condition_71794 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 637, 4), result_contains_71793)
    # Assigning a type to the variable 'if_condition_71794' (line 637)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 4), 'if_condition_71794', if_condition_71794)
    # SSA begins for if statement (line 637)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Subscript (line 638):
    
    # Assigning a Str to a Subscript (line 638):
    str_71795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 31), 'str', 'F_WRAPPEDFUNC_US')
    # Getting the type of 'ret' (line 638)
    ret_71796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 8), 'ret')
    str_71797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 12), 'str', 'F_WRAPPEDFUNC')
    # Storing an element on a container (line 638)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 638, 8), ret_71796, (str_71797, str_71795))
    # SSA branch for the else part of an if statement (line 637)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Subscript (line 640):
    
    # Assigning a Str to a Subscript (line 640):
    str_71798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 31), 'str', 'F_WRAPPEDFUNC')
    # Getting the type of 'ret' (line 640)
    ret_71799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'ret')
    str_71800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 12), 'str', 'F_WRAPPEDFUNC')
    # Storing an element on a container (line 640)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 640, 8), ret_71799, (str_71800, str_71798))
    # SSA join for if statement (line 637)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Name (line 641):
    
    # Assigning a Dict to a Name (line 641):
    
    # Obtaining an instance of the builtin type 'dict' (line 641)
    dict_71801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 14), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 641)
    
    # Assigning a type to the variable 'lcb_map' (line 641)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 4), 'lcb_map', dict_71801)
    
    
    str_71802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 7), 'str', 'use')
    # Getting the type of 'rout' (line 642)
    rout_71803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 16), 'rout')
    # Applying the binary operator 'in' (line 642)
    result_contains_71804 = python_operator(stypy.reporting.localization.Localization(__file__, 642, 7), 'in', str_71802, rout_71803)
    
    # Testing the type of an if condition (line 642)
    if_condition_71805 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 642, 4), result_contains_71804)
    # Assigning a type to the variable 'if_condition_71805' (line 642)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 4), 'if_condition_71805', if_condition_71805)
    # SSA begins for if statement (line 642)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to keys(...): (line 643)
    # Processing the call keyword arguments (line 643)
    kwargs_71811 = {}
    
    # Obtaining the type of the subscript
    str_71806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 22), 'str', 'use')
    # Getting the type of 'rout' (line 643)
    rout_71807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 17), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 643)
    getitem___71808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 17), rout_71807, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 643)
    subscript_call_result_71809 = invoke(stypy.reporting.localization.Localization(__file__, 643, 17), getitem___71808, str_71806)
    
    # Obtaining the member 'keys' of a type (line 643)
    keys_71810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 17), subscript_call_result_71809, 'keys')
    # Calling keys(args, kwargs) (line 643)
    keys_call_result_71812 = invoke(stypy.reporting.localization.Localization(__file__, 643, 17), keys_71810, *[], **kwargs_71811)
    
    # Testing the type of a for loop iterable (line 643)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 643, 8), keys_call_result_71812)
    # Getting the type of the for loop variable (line 643)
    for_loop_var_71813 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 643, 8), keys_call_result_71812)
    # Assigning a type to the variable 'u' (line 643)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'u', for_loop_var_71813)
    # SSA begins for a for statement (line 643)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'u' (line 644)
    u_71814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 15), 'u')
    # Getting the type of 'cb_rules' (line 644)
    cb_rules_71815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 20), 'cb_rules')
    # Obtaining the member 'cb_map' of a type (line 644)
    cb_map_71816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 20), cb_rules_71815, 'cb_map')
    # Applying the binary operator 'in' (line 644)
    result_contains_71817 = python_operator(stypy.reporting.localization.Localization(__file__, 644, 15), 'in', u_71814, cb_map_71816)
    
    # Testing the type of an if condition (line 644)
    if_condition_71818 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 644, 12), result_contains_71817)
    # Assigning a type to the variable 'if_condition_71818' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 12), 'if_condition_71818', if_condition_71818)
    # SSA begins for if statement (line 644)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'u' (line 645)
    u_71819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 42), 'u')
    # Getting the type of 'cb_rules' (line 645)
    cb_rules_71820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 26), 'cb_rules')
    # Obtaining the member 'cb_map' of a type (line 645)
    cb_map_71821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 26), cb_rules_71820, 'cb_map')
    # Obtaining the member '__getitem__' of a type (line 645)
    getitem___71822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 26), cb_map_71821, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 645)
    subscript_call_result_71823 = invoke(stypy.reporting.localization.Localization(__file__, 645, 26), getitem___71822, u_71819)
    
    # Testing the type of a for loop iterable (line 645)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 645, 16), subscript_call_result_71823)
    # Getting the type of the for loop variable (line 645)
    for_loop_var_71824 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 645, 16), subscript_call_result_71823)
    # Assigning a type to the variable 'un' (line 645)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 16), 'un', for_loop_var_71824)
    # SSA begins for a for statement (line 645)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 646):
    
    # Assigning a Subscript to a Name (line 646):
    
    # Obtaining the type of the subscript
    int_71825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 28), 'int')
    # Getting the type of 'un' (line 646)
    un_71826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 25), 'un')
    # Obtaining the member '__getitem__' of a type (line 646)
    getitem___71827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 646, 25), un_71826, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 646)
    subscript_call_result_71828 = invoke(stypy.reporting.localization.Localization(__file__, 646, 25), getitem___71827, int_71825)
    
    # Assigning a type to the variable 'ln' (line 646)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 20), 'ln', subscript_call_result_71828)
    
    
    str_71829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 23), 'str', 'map')
    
    # Obtaining the type of the subscript
    # Getting the type of 'u' (line 647)
    u_71830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 44), 'u')
    
    # Obtaining the type of the subscript
    str_71831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 37), 'str', 'use')
    # Getting the type of 'rout' (line 647)
    rout_71832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 32), 'rout')
    # Obtaining the member '__getitem__' of a type (line 647)
    getitem___71833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 32), rout_71832, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 647)
    subscript_call_result_71834 = invoke(stypy.reporting.localization.Localization(__file__, 647, 32), getitem___71833, str_71831)
    
    # Obtaining the member '__getitem__' of a type (line 647)
    getitem___71835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 32), subscript_call_result_71834, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 647)
    subscript_call_result_71836 = invoke(stypy.reporting.localization.Localization(__file__, 647, 32), getitem___71835, u_71830)
    
    # Applying the binary operator 'in' (line 647)
    result_contains_71837 = python_operator(stypy.reporting.localization.Localization(__file__, 647, 23), 'in', str_71829, subscript_call_result_71836)
    
    # Testing the type of an if condition (line 647)
    if_condition_71838 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 647, 20), result_contains_71837)
    # Assigning a type to the variable 'if_condition_71838' (line 647)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 20), 'if_condition_71838', if_condition_71838)
    # SSA begins for if statement (line 647)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to keys(...): (line 648)
    # Processing the call keyword arguments (line 648)
    kwargs_71850 = {}
    
    # Obtaining the type of the subscript
    str_71839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 48), 'str', 'map')
    
    # Obtaining the type of the subscript
    # Getting the type of 'u' (line 648)
    u_71840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 45), 'u', False)
    
    # Obtaining the type of the subscript
    str_71841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 38), 'str', 'use')
    # Getting the type of 'rout' (line 648)
    rout_71842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 33), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 648)
    getitem___71843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 33), rout_71842, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 648)
    subscript_call_result_71844 = invoke(stypy.reporting.localization.Localization(__file__, 648, 33), getitem___71843, str_71841)
    
    # Obtaining the member '__getitem__' of a type (line 648)
    getitem___71845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 33), subscript_call_result_71844, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 648)
    subscript_call_result_71846 = invoke(stypy.reporting.localization.Localization(__file__, 648, 33), getitem___71845, u_71840)
    
    # Obtaining the member '__getitem__' of a type (line 648)
    getitem___71847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 33), subscript_call_result_71846, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 648)
    subscript_call_result_71848 = invoke(stypy.reporting.localization.Localization(__file__, 648, 33), getitem___71847, str_71839)
    
    # Obtaining the member 'keys' of a type (line 648)
    keys_71849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 33), subscript_call_result_71848, 'keys')
    # Calling keys(args, kwargs) (line 648)
    keys_call_result_71851 = invoke(stypy.reporting.localization.Localization(__file__, 648, 33), keys_71849, *[], **kwargs_71850)
    
    # Testing the type of a for loop iterable (line 648)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 648, 24), keys_call_result_71851)
    # Getting the type of the for loop variable (line 648)
    for_loop_var_71852 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 648, 24), keys_call_result_71851)
    # Assigning a type to the variable 'k' (line 648)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 24), 'k', for_loop_var_71852)
    # SSA begins for a for statement (line 648)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 649)
    k_71853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 53), 'k')
    
    # Obtaining the type of the subscript
    str_71854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 46), 'str', 'map')
    
    # Obtaining the type of the subscript
    # Getting the type of 'u' (line 649)
    u_71855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 43), 'u')
    
    # Obtaining the type of the subscript
    str_71856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 36), 'str', 'use')
    # Getting the type of 'rout' (line 649)
    rout_71857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 31), 'rout')
    # Obtaining the member '__getitem__' of a type (line 649)
    getitem___71858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 31), rout_71857, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 649)
    subscript_call_result_71859 = invoke(stypy.reporting.localization.Localization(__file__, 649, 31), getitem___71858, str_71856)
    
    # Obtaining the member '__getitem__' of a type (line 649)
    getitem___71860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 31), subscript_call_result_71859, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 649)
    subscript_call_result_71861 = invoke(stypy.reporting.localization.Localization(__file__, 649, 31), getitem___71860, u_71855)
    
    # Obtaining the member '__getitem__' of a type (line 649)
    getitem___71862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 31), subscript_call_result_71861, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 649)
    subscript_call_result_71863 = invoke(stypy.reporting.localization.Localization(__file__, 649, 31), getitem___71862, str_71854)
    
    # Obtaining the member '__getitem__' of a type (line 649)
    getitem___71864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 31), subscript_call_result_71863, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 649)
    subscript_call_result_71865 = invoke(stypy.reporting.localization.Localization(__file__, 649, 31), getitem___71864, k_71853)
    
    
    # Obtaining the type of the subscript
    int_71866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 62), 'int')
    # Getting the type of 'un' (line 649)
    un_71867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 59), 'un')
    # Obtaining the member '__getitem__' of a type (line 649)
    getitem___71868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 59), un_71867, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 649)
    subscript_call_result_71869 = invoke(stypy.reporting.localization.Localization(__file__, 649, 59), getitem___71868, int_71866)
    
    # Applying the binary operator '==' (line 649)
    result_eq_71870 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 31), '==', subscript_call_result_71865, subscript_call_result_71869)
    
    # Testing the type of an if condition (line 649)
    if_condition_71871 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 649, 28), result_eq_71870)
    # Assigning a type to the variable 'if_condition_71871' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 28), 'if_condition_71871', if_condition_71871)
    # SSA begins for if statement (line 649)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 650):
    
    # Assigning a Name to a Name (line 650):
    # Getting the type of 'k' (line 650)
    k_71872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 37), 'k')
    # Assigning a type to the variable 'ln' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 32), 'ln', k_71872)
    # SSA join for if statement (line 649)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 647)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Subscript (line 652):
    
    # Assigning a Subscript to a Subscript (line 652):
    
    # Obtaining the type of the subscript
    int_71873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 37), 'int')
    # Getting the type of 'un' (line 652)
    un_71874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 34), 'un')
    # Obtaining the member '__getitem__' of a type (line 652)
    getitem___71875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 34), un_71874, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 652)
    subscript_call_result_71876 = invoke(stypy.reporting.localization.Localization(__file__, 652, 34), getitem___71875, int_71873)
    
    # Getting the type of 'lcb_map' (line 652)
    lcb_map_71877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 20), 'lcb_map')
    # Getting the type of 'ln' (line 652)
    ln_71878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 28), 'ln')
    # Storing an element on a container (line 652)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 652, 20), lcb_map_71877, (ln_71878, subscript_call_result_71876))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 644)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 642)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    str_71879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 9), 'str', 'externals')
    # Getting the type of 'rout' (line 653)
    rout_71880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 24), 'rout')
    # Applying the binary operator 'in' (line 653)
    result_contains_71881 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 9), 'in', str_71879, rout_71880)
    
    
    # Obtaining the type of the subscript
    str_71882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 38), 'str', 'externals')
    # Getting the type of 'rout' (line 653)
    rout_71883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 33), 'rout')
    # Obtaining the member '__getitem__' of a type (line 653)
    getitem___71884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 33), rout_71883, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 653)
    subscript_call_result_71885 = invoke(stypy.reporting.localization.Localization(__file__, 653, 33), getitem___71884, str_71882)
    
    # Applying the binary operator 'and' (line 653)
    result_and_keyword_71886 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 9), 'and', result_contains_71881, subscript_call_result_71885)
    
    # Testing the type of an if condition (line 653)
    if_condition_71887 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 653, 9), result_and_keyword_71886)
    # Assigning a type to the variable 'if_condition_71887' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 9), 'if_condition_71887', if_condition_71887)
    # SSA begins for if statement (line 653)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to errmess(...): (line 654)
    # Processing the call arguments (line 654)
    str_71889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 16), 'str', 'routsign2map: Confused: function %s has externals %s but no "use" statement.\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 655)
    tuple_71890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 655)
    # Adding element type (line 655)
    
    # Obtaining the type of the subscript
    str_71891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 16), 'str', 'name')
    # Getting the type of 'ret' (line 655)
    ret_71892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 12), 'ret', False)
    # Obtaining the member '__getitem__' of a type (line 655)
    getitem___71893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 12), ret_71892, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 655)
    subscript_call_result_71894 = invoke(stypy.reporting.localization.Localization(__file__, 655, 12), getitem___71893, str_71891)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 655, 12), tuple_71890, subscript_call_result_71894)
    # Adding element type (line 655)
    
    # Call to repr(...): (line 655)
    # Processing the call arguments (line 655)
    
    # Obtaining the type of the subscript
    str_71896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 35), 'str', 'externals')
    # Getting the type of 'rout' (line 655)
    rout_71897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 30), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 655)
    getitem___71898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 30), rout_71897, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 655)
    subscript_call_result_71899 = invoke(stypy.reporting.localization.Localization(__file__, 655, 30), getitem___71898, str_71896)
    
    # Processing the call keyword arguments (line 655)
    kwargs_71900 = {}
    # Getting the type of 'repr' (line 655)
    repr_71895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 25), 'repr', False)
    # Calling repr(args, kwargs) (line 655)
    repr_call_result_71901 = invoke(stypy.reporting.localization.Localization(__file__, 655, 25), repr_71895, *[subscript_call_result_71899], **kwargs_71900)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 655, 12), tuple_71890, repr_call_result_71901)
    
    # Applying the binary operator '%' (line 654)
    result_mod_71902 = python_operator(stypy.reporting.localization.Localization(__file__, 654, 16), '%', str_71889, tuple_71890)
    
    # Processing the call keyword arguments (line 654)
    kwargs_71903 = {}
    # Getting the type of 'errmess' (line 654)
    errmess_71888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 8), 'errmess', False)
    # Calling errmess(args, kwargs) (line 654)
    errmess_call_result_71904 = invoke(stypy.reporting.localization.Localization(__file__, 654, 8), errmess_71888, *[result_mod_71902], **kwargs_71903)
    
    # SSA join for if statement (line 653)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 642)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Subscript (line 656):
    
    # Assigning a BoolOp to a Subscript (line 656):
    
    # Evaluating a boolean operation
    
    # Call to getcallprotoargument(...): (line 656)
    # Processing the call arguments (line 656)
    # Getting the type of 'rout' (line 656)
    rout_71906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 52), 'rout', False)
    # Getting the type of 'lcb_map' (line 656)
    lcb_map_71907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 58), 'lcb_map', False)
    # Processing the call keyword arguments (line 656)
    kwargs_71908 = {}
    # Getting the type of 'getcallprotoargument' (line 656)
    getcallprotoargument_71905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 31), 'getcallprotoargument', False)
    # Calling getcallprotoargument(args, kwargs) (line 656)
    getcallprotoargument_call_result_71909 = invoke(stypy.reporting.localization.Localization(__file__, 656, 31), getcallprotoargument_71905, *[rout_71906, lcb_map_71907], **kwargs_71908)
    
    str_71910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 70), 'str', '')
    # Applying the binary operator 'or' (line 656)
    result_or_keyword_71911 = python_operator(stypy.reporting.localization.Localization(__file__, 656, 31), 'or', getcallprotoargument_call_result_71909, str_71910)
    
    # Getting the type of 'ret' (line 656)
    ret_71912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 4), 'ret')
    str_71913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 8), 'str', 'callprotoargument')
    # Storing an element on a container (line 656)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 656, 4), ret_71912, (str_71913, result_or_keyword_71911))
    
    
    # Call to isfunction(...): (line 657)
    # Processing the call arguments (line 657)
    # Getting the type of 'rout' (line 657)
    rout_71915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 18), 'rout', False)
    # Processing the call keyword arguments (line 657)
    kwargs_71916 = {}
    # Getting the type of 'isfunction' (line 657)
    isfunction_71914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 7), 'isfunction', False)
    # Calling isfunction(args, kwargs) (line 657)
    isfunction_call_result_71917 = invoke(stypy.reporting.localization.Localization(__file__, 657, 7), isfunction_71914, *[rout_71915], **kwargs_71916)
    
    # Testing the type of an if condition (line 657)
    if_condition_71918 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 657, 4), isfunction_call_result_71917)
    # Assigning a type to the variable 'if_condition_71918' (line 657)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 4), 'if_condition_71918', if_condition_71918)
    # SSA begins for if statement (line 657)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    str_71919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 11), 'str', 'result')
    # Getting the type of 'rout' (line 658)
    rout_71920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 23), 'rout')
    # Applying the binary operator 'in' (line 658)
    result_contains_71921 = python_operator(stypy.reporting.localization.Localization(__file__, 658, 11), 'in', str_71919, rout_71920)
    
    # Testing the type of an if condition (line 658)
    if_condition_71922 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 658, 8), result_contains_71921)
    # Assigning a type to the variable 'if_condition_71922' (line 658)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 8), 'if_condition_71922', if_condition_71922)
    # SSA begins for if statement (line 658)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 659):
    
    # Assigning a Subscript to a Name (line 659):
    
    # Obtaining the type of the subscript
    str_71923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 21), 'str', 'result')
    # Getting the type of 'rout' (line 659)
    rout_71924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 16), 'rout')
    # Obtaining the member '__getitem__' of a type (line 659)
    getitem___71925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 16), rout_71924, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 659)
    subscript_call_result_71926 = invoke(stypy.reporting.localization.Localization(__file__, 659, 16), getitem___71925, str_71923)
    
    # Assigning a type to the variable 'a' (line 659)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 12), 'a', subscript_call_result_71926)
    # SSA branch for the else part of an if statement (line 658)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 661):
    
    # Assigning a Subscript to a Name (line 661):
    
    # Obtaining the type of the subscript
    str_71927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 21), 'str', 'name')
    # Getting the type of 'rout' (line 661)
    rout_71928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 16), 'rout')
    # Obtaining the member '__getitem__' of a type (line 661)
    getitem___71929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 16), rout_71928, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 661)
    subscript_call_result_71930 = invoke(stypy.reporting.localization.Localization(__file__, 661, 16), getitem___71929, str_71927)
    
    # Assigning a type to the variable 'a' (line 661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 12), 'a', subscript_call_result_71930)
    # SSA join for if statement (line 658)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 662):
    
    # Assigning a Name to a Subscript (line 662):
    # Getting the type of 'a' (line 662)
    a_71931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 23), 'a')
    # Getting the type of 'ret' (line 662)
    ret_71932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 8), 'ret')
    str_71933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 12), 'str', 'rname')
    # Storing an element on a container (line 662)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 662, 8), ret_71932, (str_71933, a_71931))
    
    # Assigning a Call to a Tuple (line 663):
    
    # Assigning a Call to a Name:
    
    # Call to getpydocsign(...): (line 663)
    # Processing the call arguments (line 663)
    # Getting the type of 'a' (line 663)
    a_71935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 61), 'a', False)
    # Getting the type of 'rout' (line 663)
    rout_71936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 64), 'rout', False)
    # Processing the call keyword arguments (line 663)
    kwargs_71937 = {}
    # Getting the type of 'getpydocsign' (line 663)
    getpydocsign_71934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 48), 'getpydocsign', False)
    # Calling getpydocsign(args, kwargs) (line 663)
    getpydocsign_call_result_71938 = invoke(stypy.reporting.localization.Localization(__file__, 663, 48), getpydocsign_71934, *[a_71935, rout_71936], **kwargs_71937)
    
    # Assigning a type to the variable 'call_assignment_69372' (line 663)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'call_assignment_69372', getpydocsign_call_result_71938)
    
    # Assigning a Call to a Name (line 663):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_71941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 8), 'int')
    # Processing the call keyword arguments
    kwargs_71942 = {}
    # Getting the type of 'call_assignment_69372' (line 663)
    call_assignment_69372_71939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'call_assignment_69372', False)
    # Obtaining the member '__getitem__' of a type (line 663)
    getitem___71940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 8), call_assignment_69372_71939, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_71943 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___71940, *[int_71941], **kwargs_71942)
    
    # Assigning a type to the variable 'call_assignment_69373' (line 663)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'call_assignment_69373', getitem___call_result_71943)
    
    # Assigning a Name to a Subscript (line 663):
    # Getting the type of 'call_assignment_69373' (line 663)
    call_assignment_69373_71944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'call_assignment_69373')
    # Getting the type of 'ret' (line 663)
    ret_71945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'ret')
    str_71946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 12), 'str', 'pydocsign')
    # Storing an element on a container (line 663)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 663, 8), ret_71945, (str_71946, call_assignment_69373_71944))
    
    # Assigning a Call to a Name (line 663):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_71949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 8), 'int')
    # Processing the call keyword arguments
    kwargs_71950 = {}
    # Getting the type of 'call_assignment_69372' (line 663)
    call_assignment_69372_71947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'call_assignment_69372', False)
    # Obtaining the member '__getitem__' of a type (line 663)
    getitem___71948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 8), call_assignment_69372_71947, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_71951 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___71948, *[int_71949], **kwargs_71950)
    
    # Assigning a type to the variable 'call_assignment_69374' (line 663)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'call_assignment_69374', getitem___call_result_71951)
    
    # Assigning a Name to a Subscript (line 663):
    # Getting the type of 'call_assignment_69374' (line 663)
    call_assignment_69374_71952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'call_assignment_69374')
    # Getting the type of 'ret' (line 663)
    ret_71953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 26), 'ret')
    str_71954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 30), 'str', 'pydocsignout')
    # Storing an element on a container (line 663)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 663, 26), ret_71953, (str_71954, call_assignment_69374_71952))
    
    # Assigning a Call to a Subscript (line 664):
    
    # Assigning a Call to a Subscript (line 664):
    
    # Call to getctype(...): (line 664)
    # Processing the call arguments (line 664)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 664)
    a_71956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 45), 'a', False)
    
    # Obtaining the type of the subscript
    str_71957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 37), 'str', 'vars')
    # Getting the type of 'rout' (line 664)
    rout_71958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 32), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 664)
    getitem___71959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 32), rout_71958, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 664)
    subscript_call_result_71960 = invoke(stypy.reporting.localization.Localization(__file__, 664, 32), getitem___71959, str_71957)
    
    # Obtaining the member '__getitem__' of a type (line 664)
    getitem___71961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 32), subscript_call_result_71960, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 664)
    subscript_call_result_71962 = invoke(stypy.reporting.localization.Localization(__file__, 664, 32), getitem___71961, a_71956)
    
    # Processing the call keyword arguments (line 664)
    kwargs_71963 = {}
    # Getting the type of 'getctype' (line 664)
    getctype_71955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 23), 'getctype', False)
    # Calling getctype(args, kwargs) (line 664)
    getctype_call_result_71964 = invoke(stypy.reporting.localization.Localization(__file__, 664, 23), getctype_71955, *[subscript_call_result_71962], **kwargs_71963)
    
    # Getting the type of 'ret' (line 664)
    ret_71965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 8), 'ret')
    str_71966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 12), 'str', 'ctype')
    # Storing an element on a container (line 664)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 664, 8), ret_71965, (str_71966, getctype_call_result_71964))
    
    
    # Call to hasresultnote(...): (line 665)
    # Processing the call arguments (line 665)
    # Getting the type of 'rout' (line 665)
    rout_71968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 25), 'rout', False)
    # Processing the call keyword arguments (line 665)
    kwargs_71969 = {}
    # Getting the type of 'hasresultnote' (line 665)
    hasresultnote_71967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 11), 'hasresultnote', False)
    # Calling hasresultnote(args, kwargs) (line 665)
    hasresultnote_call_result_71970 = invoke(stypy.reporting.localization.Localization(__file__, 665, 11), hasresultnote_71967, *[rout_71968], **kwargs_71969)
    
    # Testing the type of an if condition (line 665)
    if_condition_71971 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 665, 8), hasresultnote_call_result_71970)
    # Assigning a type to the variable 'if_condition_71971' (line 665)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 8), 'if_condition_71971', if_condition_71971)
    # SSA begins for if statement (line 665)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 666):
    
    # Assigning a Subscript to a Subscript (line 666):
    
    # Obtaining the type of the subscript
    str_71972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 48), 'str', 'note')
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 666)
    a_71973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 45), 'a')
    
    # Obtaining the type of the subscript
    str_71974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 37), 'str', 'vars')
    # Getting the type of 'rout' (line 666)
    rout_71975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 32), 'rout')
    # Obtaining the member '__getitem__' of a type (line 666)
    getitem___71976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 32), rout_71975, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 666)
    subscript_call_result_71977 = invoke(stypy.reporting.localization.Localization(__file__, 666, 32), getitem___71976, str_71974)
    
    # Obtaining the member '__getitem__' of a type (line 666)
    getitem___71978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 32), subscript_call_result_71977, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 666)
    subscript_call_result_71979 = invoke(stypy.reporting.localization.Localization(__file__, 666, 32), getitem___71978, a_71973)
    
    # Obtaining the member '__getitem__' of a type (line 666)
    getitem___71980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 32), subscript_call_result_71979, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 666)
    subscript_call_result_71981 = invoke(stypy.reporting.localization.Localization(__file__, 666, 32), getitem___71980, str_71972)
    
    # Getting the type of 'ret' (line 666)
    ret_71982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 12), 'ret')
    str_71983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 16), 'str', 'resultnote')
    # Storing an element on a container (line 666)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 12), ret_71982, (str_71983, subscript_call_result_71981))
    
    # Assigning a List to a Subscript (line 667):
    
    # Assigning a List to a Subscript (line 667):
    
    # Obtaining an instance of the builtin type 'list' (line 667)
    list_71984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 667)
    # Adding element type (line 667)
    str_71985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 39), 'str', 'See elsewhere.')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 38), list_71984, str_71985)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 667)
    a_71986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 25), 'a')
    
    # Obtaining the type of the subscript
    str_71987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 17), 'str', 'vars')
    # Getting the type of 'rout' (line 667)
    rout_71988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 12), 'rout')
    # Obtaining the member '__getitem__' of a type (line 667)
    getitem___71989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 12), rout_71988, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 667)
    subscript_call_result_71990 = invoke(stypy.reporting.localization.Localization(__file__, 667, 12), getitem___71989, str_71987)
    
    # Obtaining the member '__getitem__' of a type (line 667)
    getitem___71991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 12), subscript_call_result_71990, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 667)
    subscript_call_result_71992 = invoke(stypy.reporting.localization.Localization(__file__, 667, 12), getitem___71991, a_71986)
    
    str_71993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 28), 'str', 'note')
    # Storing an element on a container (line 667)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 12), subscript_call_result_71992, (str_71993, list_71984))
    # SSA join for if statement (line 665)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    str_71994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 15), 'str', 'ctype')
    # Getting the type of 'ret' (line 668)
    ret_71995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 11), 'ret')
    # Obtaining the member '__getitem__' of a type (line 668)
    getitem___71996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 11), ret_71995, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 668)
    subscript_call_result_71997 = invoke(stypy.reporting.localization.Localization(__file__, 668, 11), getitem___71996, str_71994)
    
    # Getting the type of 'c2buildvalue_map' (line 668)
    c2buildvalue_map_71998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 27), 'c2buildvalue_map')
    # Applying the binary operator 'in' (line 668)
    result_contains_71999 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 11), 'in', subscript_call_result_71997, c2buildvalue_map_71998)
    
    # Testing the type of an if condition (line 668)
    if_condition_72000 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 668, 8), result_contains_71999)
    # Assigning a type to the variable 'if_condition_72000' (line 668)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 8), 'if_condition_72000', if_condition_72000)
    # SSA begins for if statement (line 668)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 669):
    
    # Assigning a Subscript to a Subscript (line 669):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    str_72001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 50), 'str', 'ctype')
    # Getting the type of 'ret' (line 669)
    ret_72002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 46), 'ret')
    # Obtaining the member '__getitem__' of a type (line 669)
    getitem___72003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 46), ret_72002, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 669)
    subscript_call_result_72004 = invoke(stypy.reporting.localization.Localization(__file__, 669, 46), getitem___72003, str_72001)
    
    # Getting the type of 'c2buildvalue_map' (line 669)
    c2buildvalue_map_72005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 29), 'c2buildvalue_map')
    # Obtaining the member '__getitem__' of a type (line 669)
    getitem___72006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 29), c2buildvalue_map_72005, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 669)
    subscript_call_result_72007 = invoke(stypy.reporting.localization.Localization(__file__, 669, 29), getitem___72006, subscript_call_result_72004)
    
    # Getting the type of 'ret' (line 669)
    ret_72008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 12), 'ret')
    str_72009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 16), 'str', 'rformat')
    # Storing an element on a container (line 669)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 669, 12), ret_72008, (str_72009, subscript_call_result_72007))
    # SSA branch for the else part of an if statement (line 668)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Subscript (line 671):
    
    # Assigning a Str to a Subscript (line 671):
    str_72010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 29), 'str', 'O')
    # Getting the type of 'ret' (line 671)
    ret_72011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 12), 'ret')
    str_72012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 16), 'str', 'rformat')
    # Storing an element on a container (line 671)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 671, 12), ret_72011, (str_72012, str_72010))
    
    # Call to errmess(...): (line 672)
    # Processing the call arguments (line 672)
    str_72014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 20), 'str', 'routsign2map: no c2buildvalue key for type %s\n')
    
    # Call to repr(...): (line 673)
    # Processing the call arguments (line 673)
    
    # Obtaining the type of the subscript
    str_72016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 30), 'str', 'ctype')
    # Getting the type of 'ret' (line 673)
    ret_72017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 26), 'ret', False)
    # Obtaining the member '__getitem__' of a type (line 673)
    getitem___72018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 26), ret_72017, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 673)
    subscript_call_result_72019 = invoke(stypy.reporting.localization.Localization(__file__, 673, 26), getitem___72018, str_72016)
    
    # Processing the call keyword arguments (line 673)
    kwargs_72020 = {}
    # Getting the type of 'repr' (line 673)
    repr_72015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 21), 'repr', False)
    # Calling repr(args, kwargs) (line 673)
    repr_call_result_72021 = invoke(stypy.reporting.localization.Localization(__file__, 673, 21), repr_72015, *[subscript_call_result_72019], **kwargs_72020)
    
    # Applying the binary operator '%' (line 672)
    result_mod_72022 = python_operator(stypy.reporting.localization.Localization(__file__, 672, 20), '%', str_72014, repr_call_result_72021)
    
    # Processing the call keyword arguments (line 672)
    kwargs_72023 = {}
    # Getting the type of 'errmess' (line 672)
    errmess_72013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 12), 'errmess', False)
    # Calling errmess(args, kwargs) (line 672)
    errmess_call_result_72024 = invoke(stypy.reporting.localization.Localization(__file__, 672, 12), errmess_72013, *[result_mod_72022], **kwargs_72023)
    
    # SSA join for if statement (line 668)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to debugcapi(...): (line 674)
    # Processing the call arguments (line 674)
    # Getting the type of 'rout' (line 674)
    rout_72026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 21), 'rout', False)
    # Processing the call keyword arguments (line 674)
    kwargs_72027 = {}
    # Getting the type of 'debugcapi' (line 674)
    debugcapi_72025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 11), 'debugcapi', False)
    # Calling debugcapi(args, kwargs) (line 674)
    debugcapi_call_result_72028 = invoke(stypy.reporting.localization.Localization(__file__, 674, 11), debugcapi_72025, *[rout_72026], **kwargs_72027)
    
    # Testing the type of an if condition (line 674)
    if_condition_72029 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 674, 8), debugcapi_call_result_72028)
    # Assigning a type to the variable 'if_condition_72029' (line 674)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 8), 'if_condition_72029', if_condition_72029)
    # SSA begins for if statement (line 674)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    str_72030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 19), 'str', 'ctype')
    # Getting the type of 'ret' (line 675)
    ret_72031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 15), 'ret')
    # Obtaining the member '__getitem__' of a type (line 675)
    getitem___72032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 15), ret_72031, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 675)
    subscript_call_result_72033 = invoke(stypy.reporting.localization.Localization(__file__, 675, 15), getitem___72032, str_72030)
    
    # Getting the type of 'cformat_map' (line 675)
    cformat_map_72034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 31), 'cformat_map')
    # Applying the binary operator 'in' (line 675)
    result_contains_72035 = python_operator(stypy.reporting.localization.Localization(__file__, 675, 15), 'in', subscript_call_result_72033, cformat_map_72034)
    
    # Testing the type of an if condition (line 675)
    if_condition_72036 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 675, 12), result_contains_72035)
    # Assigning a type to the variable 'if_condition_72036' (line 675)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 12), 'if_condition_72036', if_condition_72036)
    # SSA begins for if statement (line 675)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 676):
    
    # Assigning a BinOp to a Subscript (line 676):
    str_72037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 44), 'str', 'debug-capi:%s=%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 677)
    tuple_72038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 677)
    # Adding element type (line 677)
    # Getting the type of 'a' (line 677)
    a_72039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 20), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 677, 20), tuple_72038, a_72039)
    # Adding element type (line 677)
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    str_72040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 39), 'str', 'ctype')
    # Getting the type of 'ret' (line 677)
    ret_72041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 35), 'ret')
    # Obtaining the member '__getitem__' of a type (line 677)
    getitem___72042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 35), ret_72041, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 677)
    subscript_call_result_72043 = invoke(stypy.reporting.localization.Localization(__file__, 677, 35), getitem___72042, str_72040)
    
    # Getting the type of 'cformat_map' (line 677)
    cformat_map_72044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 23), 'cformat_map')
    # Obtaining the member '__getitem__' of a type (line 677)
    getitem___72045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 23), cformat_map_72044, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 677)
    subscript_call_result_72046 = invoke(stypy.reporting.localization.Localization(__file__, 677, 23), getitem___72045, subscript_call_result_72043)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 677, 20), tuple_72038, subscript_call_result_72046)
    
    # Applying the binary operator '%' (line 676)
    result_mod_72047 = python_operator(stypy.reporting.localization.Localization(__file__, 676, 44), '%', str_72037, tuple_72038)
    
    # Getting the type of 'ret' (line 676)
    ret_72048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'ret')
    str_72049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 20), 'str', 'routdebugshowvalue')
    # Storing an element on a container (line 676)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 16), ret_72048, (str_72049, result_mod_72047))
    # SSA join for if statement (line 675)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isstringfunction(...): (line 678)
    # Processing the call arguments (line 678)
    # Getting the type of 'rout' (line 678)
    rout_72051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 32), 'rout', False)
    # Processing the call keyword arguments (line 678)
    kwargs_72052 = {}
    # Getting the type of 'isstringfunction' (line 678)
    isstringfunction_72050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 15), 'isstringfunction', False)
    # Calling isstringfunction(args, kwargs) (line 678)
    isstringfunction_call_result_72053 = invoke(stypy.reporting.localization.Localization(__file__, 678, 15), isstringfunction_72050, *[rout_72051], **kwargs_72052)
    
    # Testing the type of an if condition (line 678)
    if_condition_72054 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 678, 12), isstringfunction_call_result_72053)
    # Assigning a type to the variable 'if_condition_72054' (line 678)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 12), 'if_condition_72054', if_condition_72054)
    # SSA begins for if statement (line 678)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 679):
    
    # Assigning a BinOp to a Subscript (line 679):
    str_72055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 44), 'str', 'debug-capi:slen(%s)=%%d %s=\\"%%s\\"')
    
    # Obtaining an instance of the builtin type 'tuple' (line 680)
    tuple_72056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 680)
    # Adding element type (line 680)
    # Getting the type of 'a' (line 680)
    a_72057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 20), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 680, 20), tuple_72056, a_72057)
    # Adding element type (line 680)
    # Getting the type of 'a' (line 680)
    a_72058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 23), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 680, 20), tuple_72056, a_72058)
    
    # Applying the binary operator '%' (line 679)
    result_mod_72059 = python_operator(stypy.reporting.localization.Localization(__file__, 679, 44), '%', str_72055, tuple_72056)
    
    # Getting the type of 'ret' (line 679)
    ret_72060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 16), 'ret')
    str_72061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 20), 'str', 'routdebugshowvalue')
    # Storing an element on a container (line 679)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 679, 16), ret_72060, (str_72061, result_mod_72059))
    # SSA join for if statement (line 678)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 674)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isstringfunction(...): (line 681)
    # Processing the call arguments (line 681)
    # Getting the type of 'rout' (line 681)
    rout_72063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 28), 'rout', False)
    # Processing the call keyword arguments (line 681)
    kwargs_72064 = {}
    # Getting the type of 'isstringfunction' (line 681)
    isstringfunction_72062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 11), 'isstringfunction', False)
    # Calling isstringfunction(args, kwargs) (line 681)
    isstringfunction_call_result_72065 = invoke(stypy.reporting.localization.Localization(__file__, 681, 11), isstringfunction_72062, *[rout_72063], **kwargs_72064)
    
    # Testing the type of an if condition (line 681)
    if_condition_72066 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 681, 8), isstringfunction_call_result_72065)
    # Assigning a type to the variable 'if_condition_72066' (line 681)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 8), 'if_condition_72066', if_condition_72066)
    # SSA begins for if statement (line 681)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 682):
    
    # Assigning a Call to a Subscript (line 682):
    
    # Call to getstrlength(...): (line 682)
    # Processing the call arguments (line 682)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 682)
    a_72068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 55), 'a', False)
    
    # Obtaining the type of the subscript
    str_72069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 47), 'str', 'vars')
    # Getting the type of 'rout' (line 682)
    rout_72070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 42), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 682)
    getitem___72071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 42), rout_72070, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 682)
    subscript_call_result_72072 = invoke(stypy.reporting.localization.Localization(__file__, 682, 42), getitem___72071, str_72069)
    
    # Obtaining the member '__getitem__' of a type (line 682)
    getitem___72073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 42), subscript_call_result_72072, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 682)
    subscript_call_result_72074 = invoke(stypy.reporting.localization.Localization(__file__, 682, 42), getitem___72073, a_72068)
    
    # Processing the call keyword arguments (line 682)
    kwargs_72075 = {}
    # Getting the type of 'getstrlength' (line 682)
    getstrlength_72067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 29), 'getstrlength', False)
    # Calling getstrlength(args, kwargs) (line 682)
    getstrlength_call_result_72076 = invoke(stypy.reporting.localization.Localization(__file__, 682, 29), getstrlength_72067, *[subscript_call_result_72074], **kwargs_72075)
    
    # Getting the type of 'ret' (line 682)
    ret_72077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 12), 'ret')
    str_72078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 16), 'str', 'rlength')
    # Storing an element on a container (line 682)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 682, 12), ret_72077, (str_72078, getstrlength_call_result_72076))
    
    
    
    # Obtaining the type of the subscript
    str_72079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 19), 'str', 'rlength')
    # Getting the type of 'ret' (line 683)
    ret_72080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 15), 'ret')
    # Obtaining the member '__getitem__' of a type (line 683)
    getitem___72081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 683, 15), ret_72080, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 683)
    subscript_call_result_72082 = invoke(stypy.reporting.localization.Localization(__file__, 683, 15), getitem___72081, str_72079)
    
    str_72083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 33), 'str', '-1')
    # Applying the binary operator '==' (line 683)
    result_eq_72084 = python_operator(stypy.reporting.localization.Localization(__file__, 683, 15), '==', subscript_call_result_72082, str_72083)
    
    # Testing the type of an if condition (line 683)
    if_condition_72085 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 683, 12), result_eq_72084)
    # Assigning a type to the variable 'if_condition_72085' (line 683)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 12), 'if_condition_72085', if_condition_72085)
    # SSA begins for if statement (line 683)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to errmess(...): (line 684)
    # Processing the call arguments (line 684)
    str_72087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 24), 'str', 'routsign2map: expected explicit specification of the length of the string returned by the fortran function %s; taking 10.\n')
    
    # Call to repr(...): (line 685)
    # Processing the call arguments (line 685)
    
    # Obtaining the type of the subscript
    str_72089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 30), 'str', 'name')
    # Getting the type of 'rout' (line 685)
    rout_72090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 25), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 685)
    getitem___72091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 685, 25), rout_72090, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 685)
    subscript_call_result_72092 = invoke(stypy.reporting.localization.Localization(__file__, 685, 25), getitem___72091, str_72089)
    
    # Processing the call keyword arguments (line 685)
    kwargs_72093 = {}
    # Getting the type of 'repr' (line 685)
    repr_72088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 20), 'repr', False)
    # Calling repr(args, kwargs) (line 685)
    repr_call_result_72094 = invoke(stypy.reporting.localization.Localization(__file__, 685, 20), repr_72088, *[subscript_call_result_72092], **kwargs_72093)
    
    # Applying the binary operator '%' (line 684)
    result_mod_72095 = python_operator(stypy.reporting.localization.Localization(__file__, 684, 24), '%', str_72087, repr_call_result_72094)
    
    # Processing the call keyword arguments (line 684)
    kwargs_72096 = {}
    # Getting the type of 'errmess' (line 684)
    errmess_72086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 16), 'errmess', False)
    # Calling errmess(args, kwargs) (line 684)
    errmess_call_result_72097 = invoke(stypy.reporting.localization.Localization(__file__, 684, 16), errmess_72086, *[result_mod_72095], **kwargs_72096)
    
    
    # Assigning a Str to a Subscript (line 686):
    
    # Assigning a Str to a Subscript (line 686):
    str_72098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 33), 'str', '10')
    # Getting the type of 'ret' (line 686)
    ret_72099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 16), 'ret')
    str_72100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 20), 'str', 'rlength')
    # Storing an element on a container (line 686)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 686, 16), ret_72099, (str_72100, str_72098))
    # SSA join for if statement (line 683)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 681)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 657)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to hasnote(...): (line 687)
    # Processing the call arguments (line 687)
    # Getting the type of 'rout' (line 687)
    rout_72102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 15), 'rout', False)
    # Processing the call keyword arguments (line 687)
    kwargs_72103 = {}
    # Getting the type of 'hasnote' (line 687)
    hasnote_72101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 7), 'hasnote', False)
    # Calling hasnote(args, kwargs) (line 687)
    hasnote_call_result_72104 = invoke(stypy.reporting.localization.Localization(__file__, 687, 7), hasnote_72101, *[rout_72102], **kwargs_72103)
    
    # Testing the type of an if condition (line 687)
    if_condition_72105 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 687, 4), hasnote_call_result_72104)
    # Assigning a type to the variable 'if_condition_72105' (line 687)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 4), 'if_condition_72105', if_condition_72105)
    # SSA begins for if statement (line 687)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 688):
    
    # Assigning a Subscript to a Subscript (line 688):
    
    # Obtaining the type of the subscript
    str_72106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 27), 'str', 'note')
    # Getting the type of 'rout' (line 688)
    rout_72107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 22), 'rout')
    # Obtaining the member '__getitem__' of a type (line 688)
    getitem___72108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 22), rout_72107, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 688)
    subscript_call_result_72109 = invoke(stypy.reporting.localization.Localization(__file__, 688, 22), getitem___72108, str_72106)
    
    # Getting the type of 'ret' (line 688)
    ret_72110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 8), 'ret')
    str_72111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 12), 'str', 'note')
    # Storing an element on a container (line 688)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 688, 8), ret_72110, (str_72111, subscript_call_result_72109))
    
    # Assigning a List to a Subscript (line 689):
    
    # Assigning a List to a Subscript (line 689):
    
    # Obtaining an instance of the builtin type 'list' (line 689)
    list_72112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 689)
    # Adding element type (line 689)
    str_72113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 24), 'str', 'See elsewhere.')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 689, 23), list_72112, str_72113)
    
    # Getting the type of 'rout' (line 689)
    rout_72114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 8), 'rout')
    str_72115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 13), 'str', 'note')
    # Storing an element on a container (line 689)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 689, 8), rout_72114, (str_72115, list_72112))
    # SSA join for if statement (line 687)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ret' (line 690)
    ret_72116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 11), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'stypy_return_type', ret_72116)
    
    # ################# End of 'routsign2map(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'routsign2map' in the type store
    # Getting the type of 'stypy_return_type' (line 612)
    stypy_return_type_72117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_72117)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'routsign2map'
    return stypy_return_type_72117

# Assigning a type to the variable 'routsign2map' (line 612)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 0), 'routsign2map', routsign2map)

@norecursion
def modsign2map(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'modsign2map'
    module_type_store = module_type_store.open_function_context('modsign2map', 693, 0, False)
    
    # Passed parameters checking function
    modsign2map.stypy_localization = localization
    modsign2map.stypy_type_of_self = None
    modsign2map.stypy_type_store = module_type_store
    modsign2map.stypy_function_name = 'modsign2map'
    modsign2map.stypy_param_names_list = ['m']
    modsign2map.stypy_varargs_param_name = None
    modsign2map.stypy_kwargs_param_name = None
    modsign2map.stypy_call_defaults = defaults
    modsign2map.stypy_call_varargs = varargs
    modsign2map.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'modsign2map', ['m'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'modsign2map', localization, ['m'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'modsign2map(...)' code ##################

    str_72118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, (-1)), 'str', '\n    modulename\n    ')
    
    
    # Call to ismodule(...): (line 697)
    # Processing the call arguments (line 697)
    # Getting the type of 'm' (line 697)
    m_72120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 16), 'm', False)
    # Processing the call keyword arguments (line 697)
    kwargs_72121 = {}
    # Getting the type of 'ismodule' (line 697)
    ismodule_72119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 7), 'ismodule', False)
    # Calling ismodule(args, kwargs) (line 697)
    ismodule_call_result_72122 = invoke(stypy.reporting.localization.Localization(__file__, 697, 7), ismodule_72119, *[m_72120], **kwargs_72121)
    
    # Testing the type of an if condition (line 697)
    if_condition_72123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 697, 4), ismodule_call_result_72122)
    # Assigning a type to the variable 'if_condition_72123' (line 697)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 4), 'if_condition_72123', if_condition_72123)
    # SSA begins for if statement (line 697)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Dict to a Name (line 698):
    
    # Assigning a Dict to a Name (line 698):
    
    # Obtaining an instance of the builtin type 'dict' (line 698)
    dict_72124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 14), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 698)
    # Adding element type (key, value) (line 698)
    str_72125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 15), 'str', 'f90modulename')
    
    # Obtaining the type of the subscript
    str_72126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 34), 'str', 'name')
    # Getting the type of 'm' (line 698)
    m_72127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 32), 'm')
    # Obtaining the member '__getitem__' of a type (line 698)
    getitem___72128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 32), m_72127, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 698)
    subscript_call_result_72129 = invoke(stypy.reporting.localization.Localization(__file__, 698, 32), getitem___72128, str_72126)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 698, 14), dict_72124, (str_72125, subscript_call_result_72129))
    # Adding element type (key, value) (line 698)
    str_72130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 15), 'str', 'F90MODULENAME')
    
    # Call to upper(...): (line 699)
    # Processing the call keyword arguments (line 699)
    kwargs_72136 = {}
    
    # Obtaining the type of the subscript
    str_72131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 34), 'str', 'name')
    # Getting the type of 'm' (line 699)
    m_72132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 32), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 699)
    getitem___72133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 32), m_72132, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 699)
    subscript_call_result_72134 = invoke(stypy.reporting.localization.Localization(__file__, 699, 32), getitem___72133, str_72131)
    
    # Obtaining the member 'upper' of a type (line 699)
    upper_72135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 32), subscript_call_result_72134, 'upper')
    # Calling upper(args, kwargs) (line 699)
    upper_call_result_72137 = invoke(stypy.reporting.localization.Localization(__file__, 699, 32), upper_72135, *[], **kwargs_72136)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 698, 14), dict_72124, (str_72130, upper_call_result_72137))
    # Adding element type (key, value) (line 698)
    str_72138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 15), 'str', 'texf90modulename')
    
    # Call to replace(...): (line 700)
    # Processing the call arguments (line 700)
    str_72144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 53), 'str', '_')
    str_72145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 58), 'str', '\\_')
    # Processing the call keyword arguments (line 700)
    kwargs_72146 = {}
    
    # Obtaining the type of the subscript
    str_72139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 37), 'str', 'name')
    # Getting the type of 'm' (line 700)
    m_72140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 35), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 700)
    getitem___72141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 35), m_72140, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 700)
    subscript_call_result_72142 = invoke(stypy.reporting.localization.Localization(__file__, 700, 35), getitem___72141, str_72139)
    
    # Obtaining the member 'replace' of a type (line 700)
    replace_72143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 35), subscript_call_result_72142, 'replace')
    # Calling replace(args, kwargs) (line 700)
    replace_call_result_72147 = invoke(stypy.reporting.localization.Localization(__file__, 700, 35), replace_72143, *[str_72144, str_72145], **kwargs_72146)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 698, 14), dict_72124, (str_72138, replace_call_result_72147))
    
    # Assigning a type to the variable 'ret' (line 698)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 8), 'ret', dict_72124)
    # SSA branch for the else part of an if statement (line 697)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Dict to a Name (line 702):
    
    # Assigning a Dict to a Name (line 702):
    
    # Obtaining an instance of the builtin type 'dict' (line 702)
    dict_72148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 14), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 702)
    # Adding element type (key, value) (line 702)
    str_72149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 15), 'str', 'modulename')
    
    # Obtaining the type of the subscript
    str_72150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 31), 'str', 'name')
    # Getting the type of 'm' (line 702)
    m_72151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 29), 'm')
    # Obtaining the member '__getitem__' of a type (line 702)
    getitem___72152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 29), m_72151, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 702)
    subscript_call_result_72153 = invoke(stypy.reporting.localization.Localization(__file__, 702, 29), getitem___72152, str_72150)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 702, 14), dict_72148, (str_72149, subscript_call_result_72153))
    # Adding element type (key, value) (line 702)
    str_72154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, 15), 'str', 'MODULENAME')
    
    # Call to upper(...): (line 703)
    # Processing the call keyword arguments (line 703)
    kwargs_72160 = {}
    
    # Obtaining the type of the subscript
    str_72155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, 31), 'str', 'name')
    # Getting the type of 'm' (line 703)
    m_72156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 29), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 703)
    getitem___72157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 29), m_72156, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 703)
    subscript_call_result_72158 = invoke(stypy.reporting.localization.Localization(__file__, 703, 29), getitem___72157, str_72155)
    
    # Obtaining the member 'upper' of a type (line 703)
    upper_72159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 29), subscript_call_result_72158, 'upper')
    # Calling upper(args, kwargs) (line 703)
    upper_call_result_72161 = invoke(stypy.reporting.localization.Localization(__file__, 703, 29), upper_72159, *[], **kwargs_72160)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 702, 14), dict_72148, (str_72154, upper_call_result_72161))
    # Adding element type (key, value) (line 702)
    str_72162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 15), 'str', 'texmodulename')
    
    # Call to replace(...): (line 704)
    # Processing the call arguments (line 704)
    str_72168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 50), 'str', '_')
    str_72169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 55), 'str', '\\_')
    # Processing the call keyword arguments (line 704)
    kwargs_72170 = {}
    
    # Obtaining the type of the subscript
    str_72163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 34), 'str', 'name')
    # Getting the type of 'm' (line 704)
    m_72164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 32), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 704)
    getitem___72165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 32), m_72164, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 704)
    subscript_call_result_72166 = invoke(stypy.reporting.localization.Localization(__file__, 704, 32), getitem___72165, str_72163)
    
    # Obtaining the member 'replace' of a type (line 704)
    replace_72167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 32), subscript_call_result_72166, 'replace')
    # Calling replace(args, kwargs) (line 704)
    replace_call_result_72171 = invoke(stypy.reporting.localization.Localization(__file__, 704, 32), replace_72167, *[str_72168, str_72169], **kwargs_72170)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 702, 14), dict_72148, (str_72162, replace_call_result_72171))
    
    # Assigning a type to the variable 'ret' (line 702)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 8), 'ret', dict_72148)
    # SSA join for if statement (line 697)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Subscript (line 705):
    
    # Assigning a BoolOp to a Subscript (line 705):
    
    # Evaluating a boolean operation
    
    # Call to getrestdoc(...): (line 705)
    # Processing the call arguments (line 705)
    # Getting the type of 'm' (line 705)
    m_72173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 32), 'm', False)
    # Processing the call keyword arguments (line 705)
    kwargs_72174 = {}
    # Getting the type of 'getrestdoc' (line 705)
    getrestdoc_72172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 21), 'getrestdoc', False)
    # Calling getrestdoc(args, kwargs) (line 705)
    getrestdoc_call_result_72175 = invoke(stypy.reporting.localization.Localization(__file__, 705, 21), getrestdoc_72172, *[m_72173], **kwargs_72174)
    
    
    # Obtaining an instance of the builtin type 'list' (line 705)
    list_72176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 705)
    
    # Applying the binary operator 'or' (line 705)
    result_or_keyword_72177 = python_operator(stypy.reporting.localization.Localization(__file__, 705, 21), 'or', getrestdoc_call_result_72175, list_72176)
    
    # Getting the type of 'ret' (line 705)
    ret_72178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'ret')
    str_72179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 8), 'str', 'restdoc')
    # Storing an element on a container (line 705)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 705, 4), ret_72178, (str_72179, result_or_keyword_72177))
    
    
    # Call to hasnote(...): (line 706)
    # Processing the call arguments (line 706)
    # Getting the type of 'm' (line 706)
    m_72181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 15), 'm', False)
    # Processing the call keyword arguments (line 706)
    kwargs_72182 = {}
    # Getting the type of 'hasnote' (line 706)
    hasnote_72180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 7), 'hasnote', False)
    # Calling hasnote(args, kwargs) (line 706)
    hasnote_call_result_72183 = invoke(stypy.reporting.localization.Localization(__file__, 706, 7), hasnote_72180, *[m_72181], **kwargs_72182)
    
    # Testing the type of an if condition (line 706)
    if_condition_72184 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 706, 4), hasnote_call_result_72183)
    # Assigning a type to the variable 'if_condition_72184' (line 706)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 4), 'if_condition_72184', if_condition_72184)
    # SSA begins for if statement (line 706)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 707):
    
    # Assigning a Subscript to a Subscript (line 707):
    
    # Obtaining the type of the subscript
    str_72185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 24), 'str', 'note')
    # Getting the type of 'm' (line 707)
    m_72186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 22), 'm')
    # Obtaining the member '__getitem__' of a type (line 707)
    getitem___72187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 22), m_72186, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 707)
    subscript_call_result_72188 = invoke(stypy.reporting.localization.Localization(__file__, 707, 22), getitem___72187, str_72185)
    
    # Getting the type of 'ret' (line 707)
    ret_72189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 8), 'ret')
    str_72190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 12), 'str', 'note')
    # Storing an element on a container (line 707)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 707, 8), ret_72189, (str_72190, subscript_call_result_72188))
    # SSA join for if statement (line 706)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Subscript (line 708):
    
    # Assigning a BoolOp to a Subscript (line 708):
    
    # Evaluating a boolean operation
    
    # Call to getusercode(...): (line 708)
    # Processing the call arguments (line 708)
    # Getting the type of 'm' (line 708)
    m_72192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 34), 'm', False)
    # Processing the call keyword arguments (line 708)
    kwargs_72193 = {}
    # Getting the type of 'getusercode' (line 708)
    getusercode_72191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 22), 'getusercode', False)
    # Calling getusercode(args, kwargs) (line 708)
    getusercode_call_result_72194 = invoke(stypy.reporting.localization.Localization(__file__, 708, 22), getusercode_72191, *[m_72192], **kwargs_72193)
    
    str_72195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 40), 'str', '')
    # Applying the binary operator 'or' (line 708)
    result_or_keyword_72196 = python_operator(stypy.reporting.localization.Localization(__file__, 708, 22), 'or', getusercode_call_result_72194, str_72195)
    
    # Getting the type of 'ret' (line 708)
    ret_72197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 4), 'ret')
    str_72198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 8), 'str', 'usercode')
    # Storing an element on a container (line 708)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 4), ret_72197, (str_72198, result_or_keyword_72196))
    
    # Assigning a BoolOp to a Subscript (line 709):
    
    # Assigning a BoolOp to a Subscript (line 709):
    
    # Evaluating a boolean operation
    
    # Call to getusercode1(...): (line 709)
    # Processing the call arguments (line 709)
    # Getting the type of 'm' (line 709)
    m_72200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 36), 'm', False)
    # Processing the call keyword arguments (line 709)
    kwargs_72201 = {}
    # Getting the type of 'getusercode1' (line 709)
    getusercode1_72199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 23), 'getusercode1', False)
    # Calling getusercode1(args, kwargs) (line 709)
    getusercode1_call_result_72202 = invoke(stypy.reporting.localization.Localization(__file__, 709, 23), getusercode1_72199, *[m_72200], **kwargs_72201)
    
    str_72203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 42), 'str', '')
    # Applying the binary operator 'or' (line 709)
    result_or_keyword_72204 = python_operator(stypy.reporting.localization.Localization(__file__, 709, 23), 'or', getusercode1_call_result_72202, str_72203)
    
    # Getting the type of 'ret' (line 709)
    ret_72205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 4), 'ret')
    str_72206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 8), 'str', 'usercode1')
    # Storing an element on a container (line 709)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 709, 4), ret_72205, (str_72206, result_or_keyword_72204))
    
    
    # Obtaining the type of the subscript
    str_72207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 9), 'str', 'body')
    # Getting the type of 'm' (line 710)
    m_72208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 7), 'm')
    # Obtaining the member '__getitem__' of a type (line 710)
    getitem___72209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 710, 7), m_72208, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 710)
    subscript_call_result_72210 = invoke(stypy.reporting.localization.Localization(__file__, 710, 7), getitem___72209, str_72207)
    
    # Testing the type of an if condition (line 710)
    if_condition_72211 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 710, 4), subscript_call_result_72210)
    # Assigning a type to the variable 'if_condition_72211' (line 710)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 710, 4), 'if_condition_72211', if_condition_72211)
    # SSA begins for if statement (line 710)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BoolOp to a Subscript (line 711):
    
    # Assigning a BoolOp to a Subscript (line 711):
    
    # Evaluating a boolean operation
    
    # Call to getusercode(...): (line 711)
    # Processing the call arguments (line 711)
    
    # Obtaining the type of the subscript
    int_72213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 58), 'int')
    
    # Obtaining the type of the subscript
    str_72214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 50), 'str', 'body')
    # Getting the type of 'm' (line 711)
    m_72215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 48), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 711)
    getitem___72216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 48), m_72215, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 711)
    subscript_call_result_72217 = invoke(stypy.reporting.localization.Localization(__file__, 711, 48), getitem___72216, str_72214)
    
    # Obtaining the member '__getitem__' of a type (line 711)
    getitem___72218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 48), subscript_call_result_72217, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 711)
    subscript_call_result_72219 = invoke(stypy.reporting.localization.Localization(__file__, 711, 48), getitem___72218, int_72213)
    
    # Processing the call keyword arguments (line 711)
    kwargs_72220 = {}
    # Getting the type of 'getusercode' (line 711)
    getusercode_72212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 36), 'getusercode', False)
    # Calling getusercode(args, kwargs) (line 711)
    getusercode_call_result_72221 = invoke(stypy.reporting.localization.Localization(__file__, 711, 36), getusercode_72212, *[subscript_call_result_72219], **kwargs_72220)
    
    str_72222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 65), 'str', '')
    # Applying the binary operator 'or' (line 711)
    result_or_keyword_72223 = python_operator(stypy.reporting.localization.Localization(__file__, 711, 36), 'or', getusercode_call_result_72221, str_72222)
    
    # Getting the type of 'ret' (line 711)
    ret_72224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 8), 'ret')
    str_72225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 12), 'str', 'interface_usercode')
    # Storing an element on a container (line 711)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 8), ret_72224, (str_72225, result_or_keyword_72223))
    # SSA branch for the else part of an if statement (line 710)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Subscript (line 713):
    
    # Assigning a Str to a Subscript (line 713):
    str_72226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 36), 'str', '')
    # Getting the type of 'ret' (line 713)
    ret_72227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 8), 'ret')
    str_72228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 12), 'str', 'interface_usercode')
    # Storing an element on a container (line 713)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 713, 8), ret_72227, (str_72228, str_72226))
    # SSA join for if statement (line 710)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Subscript (line 714):
    
    # Assigning a BoolOp to a Subscript (line 714):
    
    # Evaluating a boolean operation
    
    # Call to getpymethoddef(...): (line 714)
    # Processing the call arguments (line 714)
    # Getting the type of 'm' (line 714)
    m_72230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 40), 'm', False)
    # Processing the call keyword arguments (line 714)
    kwargs_72231 = {}
    # Getting the type of 'getpymethoddef' (line 714)
    getpymethoddef_72229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 25), 'getpymethoddef', False)
    # Calling getpymethoddef(args, kwargs) (line 714)
    getpymethoddef_call_result_72232 = invoke(stypy.reporting.localization.Localization(__file__, 714, 25), getpymethoddef_72229, *[m_72230], **kwargs_72231)
    
    str_72233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 46), 'str', '')
    # Applying the binary operator 'or' (line 714)
    result_or_keyword_72234 = python_operator(stypy.reporting.localization.Localization(__file__, 714, 25), 'or', getpymethoddef_call_result_72232, str_72233)
    
    # Getting the type of 'ret' (line 714)
    ret_72235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 4), 'ret')
    str_72236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 8), 'str', 'pymethoddef')
    # Storing an element on a container (line 714)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 714, 4), ret_72235, (str_72236, result_or_keyword_72234))
    
    
    str_72237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 7), 'str', 'coutput')
    # Getting the type of 'm' (line 715)
    m_72238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 20), 'm')
    # Applying the binary operator 'in' (line 715)
    result_contains_72239 = python_operator(stypy.reporting.localization.Localization(__file__, 715, 7), 'in', str_72237, m_72238)
    
    # Testing the type of an if condition (line 715)
    if_condition_72240 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 715, 4), result_contains_72239)
    # Assigning a type to the variable 'if_condition_72240' (line 715)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 4), 'if_condition_72240', if_condition_72240)
    # SSA begins for if statement (line 715)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 716):
    
    # Assigning a Subscript to a Subscript (line 716):
    
    # Obtaining the type of the subscript
    str_72241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 27), 'str', 'coutput')
    # Getting the type of 'm' (line 716)
    m_72242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 25), 'm')
    # Obtaining the member '__getitem__' of a type (line 716)
    getitem___72243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 25), m_72242, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 716)
    subscript_call_result_72244 = invoke(stypy.reporting.localization.Localization(__file__, 716, 25), getitem___72243, str_72241)
    
    # Getting the type of 'ret' (line 716)
    ret_72245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 8), 'ret')
    str_72246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 12), 'str', 'coutput')
    # Storing an element on a container (line 716)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 716, 8), ret_72245, (str_72246, subscript_call_result_72244))
    # SSA join for if statement (line 715)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_72247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, 7), 'str', 'f2py_wrapper_output')
    # Getting the type of 'm' (line 717)
    m_72248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 32), 'm')
    # Applying the binary operator 'in' (line 717)
    result_contains_72249 = python_operator(stypy.reporting.localization.Localization(__file__, 717, 7), 'in', str_72247, m_72248)
    
    # Testing the type of an if condition (line 717)
    if_condition_72250 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 717, 4), result_contains_72249)
    # Assigning a type to the variable 'if_condition_72250' (line 717)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 4), 'if_condition_72250', if_condition_72250)
    # SSA begins for if statement (line 717)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 718):
    
    # Assigning a Subscript to a Subscript (line 718):
    
    # Obtaining the type of the subscript
    str_72251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 39), 'str', 'f2py_wrapper_output')
    # Getting the type of 'm' (line 718)
    m_72252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 37), 'm')
    # Obtaining the member '__getitem__' of a type (line 718)
    getitem___72253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 37), m_72252, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 718)
    subscript_call_result_72254 = invoke(stypy.reporting.localization.Localization(__file__, 718, 37), getitem___72253, str_72251)
    
    # Getting the type of 'ret' (line 718)
    ret_72255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 8), 'ret')
    str_72256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 12), 'str', 'f2py_wrapper_output')
    # Storing an element on a container (line 718)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 8), ret_72255, (str_72256, subscript_call_result_72254))
    # SSA join for if statement (line 717)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ret' (line 719)
    ret_72257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 11), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 719)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 4), 'stypy_return_type', ret_72257)
    
    # ################# End of 'modsign2map(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'modsign2map' in the type store
    # Getting the type of 'stypy_return_type' (line 693)
    stypy_return_type_72258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_72258)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'modsign2map'
    return stypy_return_type_72258

# Assigning a type to the variable 'modsign2map' (line 693)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 0), 'modsign2map', modsign2map)

@norecursion
def cb_sign2map(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 722)
    None_72259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 30), 'None')
    defaults = [None_72259]
    # Create a new context for function 'cb_sign2map'
    module_type_store = module_type_store.open_function_context('cb_sign2map', 722, 0, False)
    
    # Passed parameters checking function
    cb_sign2map.stypy_localization = localization
    cb_sign2map.stypy_type_of_self = None
    cb_sign2map.stypy_type_store = module_type_store
    cb_sign2map.stypy_function_name = 'cb_sign2map'
    cb_sign2map.stypy_param_names_list = ['a', 'var', 'index']
    cb_sign2map.stypy_varargs_param_name = None
    cb_sign2map.stypy_kwargs_param_name = None
    cb_sign2map.stypy_call_defaults = defaults
    cb_sign2map.stypy_call_varargs = varargs
    cb_sign2map.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cb_sign2map', ['a', 'var', 'index'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cb_sign2map', localization, ['a', 'var', 'index'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cb_sign2map(...)' code ##################

    
    # Assigning a Dict to a Name (line 723):
    
    # Assigning a Dict to a Name (line 723):
    
    # Obtaining an instance of the builtin type 'dict' (line 723)
    dict_72260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 10), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 723)
    # Adding element type (key, value) (line 723)
    str_72261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 11), 'str', 'varname')
    # Getting the type of 'a' (line 723)
    a_72262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 22), 'a')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 723, 10), dict_72260, (str_72261, a_72262))
    
    # Assigning a type to the variable 'ret' (line 723)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 4), 'ret', dict_72260)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'index' (line 724)
    index_72263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 7), 'index')
    # Getting the type of 'None' (line 724)
    None_72264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 16), 'None')
    # Applying the binary operator 'is' (line 724)
    result_is__72265 = python_operator(stypy.reporting.localization.Localization(__file__, 724, 7), 'is', index_72263, None_72264)
    
    int_72266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 24), 'int')
    # Applying the binary operator 'or' (line 724)
    result_or_keyword_72267 = python_operator(stypy.reporting.localization.Localization(__file__, 724, 7), 'or', result_is__72265, int_72266)
    
    # Testing the type of an if condition (line 724)
    if_condition_72268 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 724, 4), result_or_keyword_72267)
    # Assigning a type to the variable 'if_condition_72268' (line 724)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 4), 'if_condition_72268', if_condition_72268)
    # SSA begins for if statement (line 724)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 725):
    
    # Assigning a Subscript to a Subscript (line 725):
    
    # Obtaining the type of the subscript
    str_72269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 31), 'str', 'varname')
    # Getting the type of 'ret' (line 725)
    ret_72270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 27), 'ret')
    # Obtaining the member '__getitem__' of a type (line 725)
    getitem___72271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 27), ret_72270, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 725)
    subscript_call_result_72272 = invoke(stypy.reporting.localization.Localization(__file__, 725, 27), getitem___72271, str_72269)
    
    # Getting the type of 'ret' (line 725)
    ret_72273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 8), 'ret')
    str_72274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 12), 'str', 'varname_i')
    # Storing an element on a container (line 725)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 8), ret_72273, (str_72274, subscript_call_result_72272))
    # SSA branch for the else part of an if statement (line 724)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Subscript (line 727):
    
    # Assigning a BinOp to a Subscript (line 727):
    
    # Obtaining the type of the subscript
    str_72275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 31), 'str', 'varname')
    # Getting the type of 'ret' (line 727)
    ret_72276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 27), 'ret')
    # Obtaining the member '__getitem__' of a type (line 727)
    getitem___72277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 727, 27), ret_72276, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 727)
    subscript_call_result_72278 = invoke(stypy.reporting.localization.Localization(__file__, 727, 27), getitem___72277, str_72275)
    
    str_72279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 44), 'str', '_')
    # Applying the binary operator '+' (line 727)
    result_add_72280 = python_operator(stypy.reporting.localization.Localization(__file__, 727, 27), '+', subscript_call_result_72278, str_72279)
    
    
    # Call to str(...): (line 727)
    # Processing the call arguments (line 727)
    # Getting the type of 'index' (line 727)
    index_72282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 54), 'index', False)
    # Processing the call keyword arguments (line 727)
    kwargs_72283 = {}
    # Getting the type of 'str' (line 727)
    str_72281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 50), 'str', False)
    # Calling str(args, kwargs) (line 727)
    str_call_result_72284 = invoke(stypy.reporting.localization.Localization(__file__, 727, 50), str_72281, *[index_72282], **kwargs_72283)
    
    # Applying the binary operator '+' (line 727)
    result_add_72285 = python_operator(stypy.reporting.localization.Localization(__file__, 727, 48), '+', result_add_72280, str_call_result_72284)
    
    # Getting the type of 'ret' (line 727)
    ret_72286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 8), 'ret')
    str_72287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 12), 'str', 'varname_i')
    # Storing an element on a container (line 727)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 727, 8), ret_72286, (str_72287, result_add_72285))
    # SSA join for if statement (line 724)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Subscript (line 728):
    
    # Assigning a Call to a Subscript (line 728):
    
    # Call to getctype(...): (line 728)
    # Processing the call arguments (line 728)
    # Getting the type of 'var' (line 728)
    var_72289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 28), 'var', False)
    # Processing the call keyword arguments (line 728)
    kwargs_72290 = {}
    # Getting the type of 'getctype' (line 728)
    getctype_72288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 19), 'getctype', False)
    # Calling getctype(args, kwargs) (line 728)
    getctype_call_result_72291 = invoke(stypy.reporting.localization.Localization(__file__, 728, 19), getctype_72288, *[var_72289], **kwargs_72290)
    
    # Getting the type of 'ret' (line 728)
    ret_72292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 4), 'ret')
    str_72293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 8), 'str', 'ctype')
    # Storing an element on a container (line 728)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 728, 4), ret_72292, (str_72293, getctype_call_result_72291))
    
    
    
    # Obtaining the type of the subscript
    str_72294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 11), 'str', 'ctype')
    # Getting the type of 'ret' (line 729)
    ret_72295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 7), 'ret')
    # Obtaining the member '__getitem__' of a type (line 729)
    getitem___72296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 7), ret_72295, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 729)
    subscript_call_result_72297 = invoke(stypy.reporting.localization.Localization(__file__, 729, 7), getitem___72296, str_72294)
    
    # Getting the type of 'c2capi_map' (line 729)
    c2capi_map_72298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 23), 'c2capi_map')
    # Applying the binary operator 'in' (line 729)
    result_contains_72299 = python_operator(stypy.reporting.localization.Localization(__file__, 729, 7), 'in', subscript_call_result_72297, c2capi_map_72298)
    
    # Testing the type of an if condition (line 729)
    if_condition_72300 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 729, 4), result_contains_72299)
    # Assigning a type to the variable 'if_condition_72300' (line 729)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 4), 'if_condition_72300', if_condition_72300)
    # SSA begins for if statement (line 729)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 730):
    
    # Assigning a Subscript to a Subscript (line 730):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    str_72301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 38), 'str', 'ctype')
    # Getting the type of 'ret' (line 730)
    ret_72302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 34), 'ret')
    # Obtaining the member '__getitem__' of a type (line 730)
    getitem___72303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 34), ret_72302, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 730)
    subscript_call_result_72304 = invoke(stypy.reporting.localization.Localization(__file__, 730, 34), getitem___72303, str_72301)
    
    # Getting the type of 'c2capi_map' (line 730)
    c2capi_map_72305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 23), 'c2capi_map')
    # Obtaining the member '__getitem__' of a type (line 730)
    getitem___72306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 23), c2capi_map_72305, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 730)
    subscript_call_result_72307 = invoke(stypy.reporting.localization.Localization(__file__, 730, 23), getitem___72306, subscript_call_result_72304)
    
    # Getting the type of 'ret' (line 730)
    ret_72308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 8), 'ret')
    str_72309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 12), 'str', 'atype')
    # Storing an element on a container (line 730)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 730, 8), ret_72308, (str_72309, subscript_call_result_72307))
    # SSA join for if statement (line 729)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    str_72310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 11), 'str', 'ctype')
    # Getting the type of 'ret' (line 731)
    ret_72311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 7), 'ret')
    # Obtaining the member '__getitem__' of a type (line 731)
    getitem___72312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 731, 7), ret_72311, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 731)
    subscript_call_result_72313 = invoke(stypy.reporting.localization.Localization(__file__, 731, 7), getitem___72312, str_72310)
    
    # Getting the type of 'cformat_map' (line 731)
    cformat_map_72314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 23), 'cformat_map')
    # Applying the binary operator 'in' (line 731)
    result_contains_72315 = python_operator(stypy.reporting.localization.Localization(__file__, 731, 7), 'in', subscript_call_result_72313, cformat_map_72314)
    
    # Testing the type of an if condition (line 731)
    if_condition_72316 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 731, 4), result_contains_72315)
    # Assigning a type to the variable 'if_condition_72316' (line 731)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 4), 'if_condition_72316', if_condition_72316)
    # SSA begins for if statement (line 731)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 732):
    
    # Assigning a BinOp to a Subscript (line 732):
    str_72317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 33), 'str', '%s')
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    str_72318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 57), 'str', 'ctype')
    # Getting the type of 'ret' (line 732)
    ret_72319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 53), 'ret')
    # Obtaining the member '__getitem__' of a type (line 732)
    getitem___72320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 732, 53), ret_72319, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 732)
    subscript_call_result_72321 = invoke(stypy.reporting.localization.Localization(__file__, 732, 53), getitem___72320, str_72318)
    
    # Getting the type of 'cformat_map' (line 732)
    cformat_map_72322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 41), 'cformat_map')
    # Obtaining the member '__getitem__' of a type (line 732)
    getitem___72323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 732, 41), cformat_map_72322, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 732)
    subscript_call_result_72324 = invoke(stypy.reporting.localization.Localization(__file__, 732, 41), getitem___72323, subscript_call_result_72321)
    
    # Applying the binary operator '%' (line 732)
    result_mod_72325 = python_operator(stypy.reporting.localization.Localization(__file__, 732, 33), '%', str_72317, subscript_call_result_72324)
    
    # Getting the type of 'ret' (line 732)
    ret_72326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 8), 'ret')
    str_72327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 12), 'str', 'showvalueformat')
    # Storing an element on a container (line 732)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 732, 8), ret_72326, (str_72327, result_mod_72325))
    # SSA join for if statement (line 731)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isarray(...): (line 733)
    # Processing the call arguments (line 733)
    # Getting the type of 'var' (line 733)
    var_72329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 15), 'var', False)
    # Processing the call keyword arguments (line 733)
    kwargs_72330 = {}
    # Getting the type of 'isarray' (line 733)
    isarray_72328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 7), 'isarray', False)
    # Calling isarray(args, kwargs) (line 733)
    isarray_call_result_72331 = invoke(stypy.reporting.localization.Localization(__file__, 733, 7), isarray_72328, *[var_72329], **kwargs_72330)
    
    # Testing the type of an if condition (line 733)
    if_condition_72332 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 733, 4), isarray_call_result_72331)
    # Assigning a type to the variable 'if_condition_72332' (line 733)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 4), 'if_condition_72332', if_condition_72332)
    # SSA begins for if statement (line 733)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 734):
    
    # Assigning a Call to a Name (line 734):
    
    # Call to dictappend(...): (line 734)
    # Processing the call arguments (line 734)
    # Getting the type of 'ret' (line 734)
    ret_72334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 25), 'ret', False)
    
    # Call to getarrdims(...): (line 734)
    # Processing the call arguments (line 734)
    # Getting the type of 'a' (line 734)
    a_72336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 41), 'a', False)
    # Getting the type of 'var' (line 734)
    var_72337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 44), 'var', False)
    # Processing the call keyword arguments (line 734)
    kwargs_72338 = {}
    # Getting the type of 'getarrdims' (line 734)
    getarrdims_72335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 30), 'getarrdims', False)
    # Calling getarrdims(args, kwargs) (line 734)
    getarrdims_call_result_72339 = invoke(stypy.reporting.localization.Localization(__file__, 734, 30), getarrdims_72335, *[a_72336, var_72337], **kwargs_72338)
    
    # Processing the call keyword arguments (line 734)
    kwargs_72340 = {}
    # Getting the type of 'dictappend' (line 734)
    dictappend_72333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 14), 'dictappend', False)
    # Calling dictappend(args, kwargs) (line 734)
    dictappend_call_result_72341 = invoke(stypy.reporting.localization.Localization(__file__, 734, 14), dictappend_72333, *[ret_72334, getarrdims_call_result_72339], **kwargs_72340)
    
    # Assigning a type to the variable 'ret' (line 734)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 734, 8), 'ret', dictappend_call_result_72341)
    # SSA join for if statement (line 733)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 735):
    
    # Assigning a Call to a Name:
    
    # Call to getpydocsign(...): (line 735)
    # Processing the call arguments (line 735)
    # Getting the type of 'a' (line 735)
    a_72343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 57), 'a', False)
    # Getting the type of 'var' (line 735)
    var_72344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 60), 'var', False)
    # Processing the call keyword arguments (line 735)
    kwargs_72345 = {}
    # Getting the type of 'getpydocsign' (line 735)
    getpydocsign_72342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 44), 'getpydocsign', False)
    # Calling getpydocsign(args, kwargs) (line 735)
    getpydocsign_call_result_72346 = invoke(stypy.reporting.localization.Localization(__file__, 735, 44), getpydocsign_72342, *[a_72343, var_72344], **kwargs_72345)
    
    # Assigning a type to the variable 'call_assignment_69375' (line 735)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'call_assignment_69375', getpydocsign_call_result_72346)
    
    # Assigning a Call to a Name (line 735):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_72349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 4), 'int')
    # Processing the call keyword arguments
    kwargs_72350 = {}
    # Getting the type of 'call_assignment_69375' (line 735)
    call_assignment_69375_72347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'call_assignment_69375', False)
    # Obtaining the member '__getitem__' of a type (line 735)
    getitem___72348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 4), call_assignment_69375_72347, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_72351 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___72348, *[int_72349], **kwargs_72350)
    
    # Assigning a type to the variable 'call_assignment_69376' (line 735)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'call_assignment_69376', getitem___call_result_72351)
    
    # Assigning a Name to a Subscript (line 735):
    # Getting the type of 'call_assignment_69376' (line 735)
    call_assignment_69376_72352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'call_assignment_69376')
    # Getting the type of 'ret' (line 735)
    ret_72353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'ret')
    str_72354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 8), 'str', 'pydocsign')
    # Storing an element on a container (line 735)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 735, 4), ret_72353, (str_72354, call_assignment_69376_72352))
    
    # Assigning a Call to a Name (line 735):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_72357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 4), 'int')
    # Processing the call keyword arguments
    kwargs_72358 = {}
    # Getting the type of 'call_assignment_69375' (line 735)
    call_assignment_69375_72355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'call_assignment_69375', False)
    # Obtaining the member '__getitem__' of a type (line 735)
    getitem___72356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 4), call_assignment_69375_72355, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_72359 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___72356, *[int_72357], **kwargs_72358)
    
    # Assigning a type to the variable 'call_assignment_69377' (line 735)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'call_assignment_69377', getitem___call_result_72359)
    
    # Assigning a Name to a Subscript (line 735):
    # Getting the type of 'call_assignment_69377' (line 735)
    call_assignment_69377_72360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'call_assignment_69377')
    # Getting the type of 'ret' (line 735)
    ret_72361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 22), 'ret')
    str_72362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 26), 'str', 'pydocsignout')
    # Storing an element on a container (line 735)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 735, 22), ret_72361, (str_72362, call_assignment_69377_72360))
    
    
    # Call to hasnote(...): (line 736)
    # Processing the call arguments (line 736)
    # Getting the type of 'var' (line 736)
    var_72364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 15), 'var', False)
    # Processing the call keyword arguments (line 736)
    kwargs_72365 = {}
    # Getting the type of 'hasnote' (line 736)
    hasnote_72363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 7), 'hasnote', False)
    # Calling hasnote(args, kwargs) (line 736)
    hasnote_call_result_72366 = invoke(stypy.reporting.localization.Localization(__file__, 736, 7), hasnote_72363, *[var_72364], **kwargs_72365)
    
    # Testing the type of an if condition (line 736)
    if_condition_72367 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 736, 4), hasnote_call_result_72366)
    # Assigning a type to the variable 'if_condition_72367' (line 736)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 4), 'if_condition_72367', if_condition_72367)
    # SSA begins for if statement (line 736)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 737):
    
    # Assigning a Subscript to a Subscript (line 737):
    
    # Obtaining the type of the subscript
    str_72368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 26), 'str', 'note')
    # Getting the type of 'var' (line 737)
    var_72369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 22), 'var')
    # Obtaining the member '__getitem__' of a type (line 737)
    getitem___72370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 737, 22), var_72369, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 737)
    subscript_call_result_72371 = invoke(stypy.reporting.localization.Localization(__file__, 737, 22), getitem___72370, str_72368)
    
    # Getting the type of 'ret' (line 737)
    ret_72372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 8), 'ret')
    str_72373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 12), 'str', 'note')
    # Storing an element on a container (line 737)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 737, 8), ret_72372, (str_72373, subscript_call_result_72371))
    
    # Assigning a List to a Subscript (line 738):
    
    # Assigning a List to a Subscript (line 738):
    
    # Obtaining an instance of the builtin type 'list' (line 738)
    list_72374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 738)
    # Adding element type (line 738)
    str_72375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 23), 'str', 'See elsewhere.')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 738, 22), list_72374, str_72375)
    
    # Getting the type of 'var' (line 738)
    var_72376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 8), 'var')
    str_72377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 12), 'str', 'note')
    # Storing an element on a container (line 738)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 738, 8), var_72376, (str_72377, list_72374))
    # SSA join for if statement (line 736)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ret' (line 739)
    ret_72378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 11), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 739)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 4), 'stypy_return_type', ret_72378)
    
    # ################# End of 'cb_sign2map(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cb_sign2map' in the type store
    # Getting the type of 'stypy_return_type' (line 722)
    stypy_return_type_72379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_72379)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cb_sign2map'
    return stypy_return_type_72379

# Assigning a type to the variable 'cb_sign2map' (line 722)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 0), 'cb_sign2map', cb_sign2map)

@norecursion
def cb_routsign2map(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'cb_routsign2map'
    module_type_store = module_type_store.open_function_context('cb_routsign2map', 742, 0, False)
    
    # Passed parameters checking function
    cb_routsign2map.stypy_localization = localization
    cb_routsign2map.stypy_type_of_self = None
    cb_routsign2map.stypy_type_store = module_type_store
    cb_routsign2map.stypy_function_name = 'cb_routsign2map'
    cb_routsign2map.stypy_param_names_list = ['rout', 'um']
    cb_routsign2map.stypy_varargs_param_name = None
    cb_routsign2map.stypy_kwargs_param_name = None
    cb_routsign2map.stypy_call_defaults = defaults
    cb_routsign2map.stypy_call_varargs = varargs
    cb_routsign2map.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cb_routsign2map', ['rout', 'um'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cb_routsign2map', localization, ['rout', 'um'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cb_routsign2map(...)' code ##################

    str_72380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, (-1)), 'str', '\n    name,begintitle,endtitle,argname\n    ctype,rctype,maxnofargs,nofoptargs,returncptr\n    ')
    
    # Assigning a Dict to a Name (line 747):
    
    # Assigning a Dict to a Name (line 747):
    
    # Obtaining an instance of the builtin type 'dict' (line 747)
    dict_72381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 10), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 747)
    # Adding element type (key, value) (line 747)
    str_72382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 11), 'str', 'name')
    str_72383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 19), 'str', 'cb_%s_in_%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 747)
    tuple_72384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 747)
    # Adding element type (line 747)
    
    # Obtaining the type of the subscript
    str_72385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 41), 'str', 'name')
    # Getting the type of 'rout' (line 747)
    rout_72386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 36), 'rout')
    # Obtaining the member '__getitem__' of a type (line 747)
    getitem___72387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 36), rout_72386, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 747)
    subscript_call_result_72388 = invoke(stypy.reporting.localization.Localization(__file__, 747, 36), getitem___72387, str_72385)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 747, 36), tuple_72384, subscript_call_result_72388)
    # Adding element type (line 747)
    # Getting the type of 'um' (line 747)
    um_72389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 50), 'um')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 747, 36), tuple_72384, um_72389)
    
    # Applying the binary operator '%' (line 747)
    result_mod_72390 = python_operator(stypy.reporting.localization.Localization(__file__, 747, 19), '%', str_72383, tuple_72384)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 747, 10), dict_72381, (str_72382, result_mod_72390))
    # Adding element type (key, value) (line 747)
    str_72391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 11), 'str', 'returncptr')
    str_72392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 25), 'str', '')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 747, 10), dict_72381, (str_72391, str_72392))
    
    # Assigning a type to the variable 'ret' (line 747)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 4), 'ret', dict_72381)
    
    
    # Call to isintent_callback(...): (line 749)
    # Processing the call arguments (line 749)
    # Getting the type of 'rout' (line 749)
    rout_72394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 25), 'rout', False)
    # Processing the call keyword arguments (line 749)
    kwargs_72395 = {}
    # Getting the type of 'isintent_callback' (line 749)
    isintent_callback_72393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 7), 'isintent_callback', False)
    # Calling isintent_callback(args, kwargs) (line 749)
    isintent_callback_call_result_72396 = invoke(stypy.reporting.localization.Localization(__file__, 749, 7), isintent_callback_72393, *[rout_72394], **kwargs_72395)
    
    # Testing the type of an if condition (line 749)
    if_condition_72397 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 749, 4), isintent_callback_call_result_72396)
    # Assigning a type to the variable 'if_condition_72397' (line 749)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 4), 'if_condition_72397', if_condition_72397)
    # SSA begins for if statement (line 749)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    str_72398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 11), 'str', '_')
    
    # Obtaining the type of the subscript
    str_72399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 23), 'str', 'name')
    # Getting the type of 'rout' (line 750)
    rout_72400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 18), 'rout')
    # Obtaining the member '__getitem__' of a type (line 750)
    getitem___72401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 18), rout_72400, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 750)
    subscript_call_result_72402 = invoke(stypy.reporting.localization.Localization(__file__, 750, 18), getitem___72401, str_72399)
    
    # Applying the binary operator 'in' (line 750)
    result_contains_72403 = python_operator(stypy.reporting.localization.Localization(__file__, 750, 11), 'in', str_72398, subscript_call_result_72402)
    
    # Testing the type of an if condition (line 750)
    if_condition_72404 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 750, 8), result_contains_72403)
    # Assigning a type to the variable 'if_condition_72404' (line 750)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 8), 'if_condition_72404', if_condition_72404)
    # SSA begins for if statement (line 750)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 751):
    
    # Assigning a Str to a Name (line 751):
    str_72405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 21), 'str', 'F_FUNC_US')
    # Assigning a type to the variable 'F_FUNC' (line 751)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 12), 'F_FUNC', str_72405)
    # SSA branch for the else part of an if statement (line 750)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 753):
    
    # Assigning a Str to a Name (line 753):
    str_72406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, 21), 'str', 'F_FUNC')
    # Assigning a type to the variable 'F_FUNC' (line 753)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 753, 12), 'F_FUNC', str_72406)
    # SSA join for if statement (line 750)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Subscript (line 754):
    
    # Assigning a BinOp to a Subscript (line 754):
    str_72407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 30), 'str', '%s(%s,%s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 755)
    tuple_72408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 755)
    # Adding element type (line 755)
    # Getting the type of 'F_FUNC' (line 755)
    F_FUNC_72409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 33), 'F_FUNC')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 33), tuple_72408, F_FUNC_72409)
    # Adding element type (line 755)
    
    # Call to lower(...): (line 756)
    # Processing the call keyword arguments (line 756)
    kwargs_72415 = {}
    
    # Obtaining the type of the subscript
    str_72410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 38), 'str', 'name')
    # Getting the type of 'rout' (line 756)
    rout_72411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 33), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 756)
    getitem___72412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 33), rout_72411, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 756)
    subscript_call_result_72413 = invoke(stypy.reporting.localization.Localization(__file__, 756, 33), getitem___72412, str_72410)
    
    # Obtaining the member 'lower' of a type (line 756)
    lower_72414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 33), subscript_call_result_72413, 'lower')
    # Calling lower(args, kwargs) (line 756)
    lower_call_result_72416 = invoke(stypy.reporting.localization.Localization(__file__, 756, 33), lower_72414, *[], **kwargs_72415)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 33), tuple_72408, lower_call_result_72416)
    # Adding element type (line 755)
    
    # Call to upper(...): (line 757)
    # Processing the call keyword arguments (line 757)
    kwargs_72422 = {}
    
    # Obtaining the type of the subscript
    str_72417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 38), 'str', 'name')
    # Getting the type of 'rout' (line 757)
    rout_72418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 33), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 757)
    getitem___72419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 757, 33), rout_72418, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 757)
    subscript_call_result_72420 = invoke(stypy.reporting.localization.Localization(__file__, 757, 33), getitem___72419, str_72417)
    
    # Obtaining the member 'upper' of a type (line 757)
    upper_72421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 757, 33), subscript_call_result_72420, 'upper')
    # Calling upper(args, kwargs) (line 757)
    upper_call_result_72423 = invoke(stypy.reporting.localization.Localization(__file__, 757, 33), upper_72421, *[], **kwargs_72422)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 33), tuple_72408, upper_call_result_72423)
    
    # Applying the binary operator '%' (line 754)
    result_mod_72424 = python_operator(stypy.reporting.localization.Localization(__file__, 754, 30), '%', str_72407, tuple_72408)
    
    # Getting the type of 'ret' (line 754)
    ret_72425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 8), 'ret')
    str_72426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 12), 'str', 'callbackname')
    # Storing an element on a container (line 754)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 754, 8), ret_72425, (str_72426, result_mod_72424))
    
    # Assigning a Str to a Subscript (line 759):
    
    # Assigning a Str to a Subscript (line 759):
    str_72427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 24), 'str', 'extern')
    # Getting the type of 'ret' (line 759)
    ret_72428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 8), 'ret')
    str_72429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 12), 'str', 'static')
    # Storing an element on a container (line 759)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 759, 8), ret_72428, (str_72429, str_72427))
    # SSA branch for the else part of an if statement (line 749)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Subscript (line 761):
    
    # Assigning a Subscript to a Subscript (line 761):
    
    # Obtaining the type of the subscript
    str_72430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 34), 'str', 'name')
    # Getting the type of 'ret' (line 761)
    ret_72431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 30), 'ret')
    # Obtaining the member '__getitem__' of a type (line 761)
    getitem___72432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 30), ret_72431, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 761)
    subscript_call_result_72433 = invoke(stypy.reporting.localization.Localization(__file__, 761, 30), getitem___72432, str_72430)
    
    # Getting the type of 'ret' (line 761)
    ret_72434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 8), 'ret')
    str_72435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 12), 'str', 'callbackname')
    # Storing an element on a container (line 761)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 761, 8), ret_72434, (str_72435, subscript_call_result_72433))
    
    # Assigning a Str to a Subscript (line 762):
    
    # Assigning a Str to a Subscript (line 762):
    str_72436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 24), 'str', 'static')
    # Getting the type of 'ret' (line 762)
    ret_72437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 8), 'ret')
    str_72438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 12), 'str', 'static')
    # Storing an element on a container (line 762)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 762, 8), ret_72437, (str_72438, str_72436))
    # SSA join for if statement (line 749)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Subscript (line 763):
    
    # Assigning a Subscript to a Subscript (line 763):
    
    # Obtaining the type of the subscript
    str_72439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 26), 'str', 'name')
    # Getting the type of 'rout' (line 763)
    rout_72440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 21), 'rout')
    # Obtaining the member '__getitem__' of a type (line 763)
    getitem___72441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 21), rout_72440, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 763)
    subscript_call_result_72442 = invoke(stypy.reporting.localization.Localization(__file__, 763, 21), getitem___72441, str_72439)
    
    # Getting the type of 'ret' (line 763)
    ret_72443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 4), 'ret')
    str_72444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 8), 'str', 'argname')
    # Storing an element on a container (line 763)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 4), ret_72443, (str_72444, subscript_call_result_72442))
    
    # Assigning a Call to a Subscript (line 764):
    
    # Assigning a Call to a Subscript (line 764):
    
    # Call to gentitle(...): (line 764)
    # Processing the call arguments (line 764)
    
    # Obtaining the type of the subscript
    str_72446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 37), 'str', 'name')
    # Getting the type of 'ret' (line 764)
    ret_72447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 33), 'ret', False)
    # Obtaining the member '__getitem__' of a type (line 764)
    getitem___72448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 33), ret_72447, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 764)
    subscript_call_result_72449 = invoke(stypy.reporting.localization.Localization(__file__, 764, 33), getitem___72448, str_72446)
    
    # Processing the call keyword arguments (line 764)
    kwargs_72450 = {}
    # Getting the type of 'gentitle' (line 764)
    gentitle_72445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 24), 'gentitle', False)
    # Calling gentitle(args, kwargs) (line 764)
    gentitle_call_result_72451 = invoke(stypy.reporting.localization.Localization(__file__, 764, 24), gentitle_72445, *[subscript_call_result_72449], **kwargs_72450)
    
    # Getting the type of 'ret' (line 764)
    ret_72452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 4), 'ret')
    str_72453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 8), 'str', 'begintitle')
    # Storing an element on a container (line 764)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 764, 4), ret_72452, (str_72453, gentitle_call_result_72451))
    
    # Assigning a Call to a Subscript (line 765):
    
    # Assigning a Call to a Subscript (line 765):
    
    # Call to gentitle(...): (line 765)
    # Processing the call arguments (line 765)
    str_72455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 765, 31), 'str', 'end of %s')
    
    # Obtaining the type of the subscript
    str_72456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 765, 49), 'str', 'name')
    # Getting the type of 'ret' (line 765)
    ret_72457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 45), 'ret', False)
    # Obtaining the member '__getitem__' of a type (line 765)
    getitem___72458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 45), ret_72457, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 765)
    subscript_call_result_72459 = invoke(stypy.reporting.localization.Localization(__file__, 765, 45), getitem___72458, str_72456)
    
    # Applying the binary operator '%' (line 765)
    result_mod_72460 = python_operator(stypy.reporting.localization.Localization(__file__, 765, 31), '%', str_72455, subscript_call_result_72459)
    
    # Processing the call keyword arguments (line 765)
    kwargs_72461 = {}
    # Getting the type of 'gentitle' (line 765)
    gentitle_72454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 22), 'gentitle', False)
    # Calling gentitle(args, kwargs) (line 765)
    gentitle_call_result_72462 = invoke(stypy.reporting.localization.Localization(__file__, 765, 22), gentitle_72454, *[result_mod_72460], **kwargs_72461)
    
    # Getting the type of 'ret' (line 765)
    ret_72463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 4), 'ret')
    str_72464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 765, 8), 'str', 'endtitle')
    # Storing an element on a container (line 765)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 765, 4), ret_72463, (str_72464, gentitle_call_result_72462))
    
    # Assigning a Call to a Subscript (line 766):
    
    # Assigning a Call to a Subscript (line 766):
    
    # Call to getctype(...): (line 766)
    # Processing the call arguments (line 766)
    # Getting the type of 'rout' (line 766)
    rout_72466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 28), 'rout', False)
    # Processing the call keyword arguments (line 766)
    kwargs_72467 = {}
    # Getting the type of 'getctype' (line 766)
    getctype_72465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 19), 'getctype', False)
    # Calling getctype(args, kwargs) (line 766)
    getctype_call_result_72468 = invoke(stypy.reporting.localization.Localization(__file__, 766, 19), getctype_72465, *[rout_72466], **kwargs_72467)
    
    # Getting the type of 'ret' (line 766)
    ret_72469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 4), 'ret')
    str_72470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 766, 8), 'str', 'ctype')
    # Storing an element on a container (line 766)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 766, 4), ret_72469, (str_72470, getctype_call_result_72468))
    
    # Assigning a Str to a Subscript (line 767):
    
    # Assigning a Str to a Subscript (line 767):
    str_72471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 20), 'str', 'void')
    # Getting the type of 'ret' (line 767)
    ret_72472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 4), 'ret')
    str_72473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 8), 'str', 'rctype')
    # Storing an element on a container (line 767)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 767, 4), ret_72472, (str_72473, str_72471))
    
    
    
    # Obtaining the type of the subscript
    str_72474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 11), 'str', 'ctype')
    # Getting the type of 'ret' (line 768)
    ret_72475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 7), 'ret')
    # Obtaining the member '__getitem__' of a type (line 768)
    getitem___72476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 768, 7), ret_72475, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 768)
    subscript_call_result_72477 = invoke(stypy.reporting.localization.Localization(__file__, 768, 7), getitem___72476, str_72474)
    
    str_72478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 23), 'str', 'string')
    # Applying the binary operator '==' (line 768)
    result_eq_72479 = python_operator(stypy.reporting.localization.Localization(__file__, 768, 7), '==', subscript_call_result_72477, str_72478)
    
    # Testing the type of an if condition (line 768)
    if_condition_72480 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 768, 4), result_eq_72479)
    # Assigning a type to the variable 'if_condition_72480' (line 768)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 4), 'if_condition_72480', if_condition_72480)
    # SSA begins for if statement (line 768)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Subscript (line 769):
    
    # Assigning a Str to a Subscript (line 769):
    str_72481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 24), 'str', 'void')
    # Getting the type of 'ret' (line 769)
    ret_72482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 8), 'ret')
    str_72483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 12), 'str', 'rctype')
    # Storing an element on a container (line 769)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 8), ret_72482, (str_72483, str_72481))
    # SSA branch for the else part of an if statement (line 768)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Subscript (line 771):
    
    # Assigning a Subscript to a Subscript (line 771):
    
    # Obtaining the type of the subscript
    str_72484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 28), 'str', 'ctype')
    # Getting the type of 'ret' (line 771)
    ret_72485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 24), 'ret')
    # Obtaining the member '__getitem__' of a type (line 771)
    getitem___72486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 24), ret_72485, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 771)
    subscript_call_result_72487 = invoke(stypy.reporting.localization.Localization(__file__, 771, 24), getitem___72486, str_72484)
    
    # Getting the type of 'ret' (line 771)
    ret_72488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 8), 'ret')
    str_72489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 12), 'str', 'rctype')
    # Storing an element on a container (line 771)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 771, 8), ret_72488, (str_72489, subscript_call_result_72487))
    # SSA join for if statement (line 768)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    str_72490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 11), 'str', 'rctype')
    # Getting the type of 'ret' (line 772)
    ret_72491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 7), 'ret')
    # Obtaining the member '__getitem__' of a type (line 772)
    getitem___72492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 772, 7), ret_72491, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 772)
    subscript_call_result_72493 = invoke(stypy.reporting.localization.Localization(__file__, 772, 7), getitem___72492, str_72490)
    
    str_72494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 24), 'str', 'void')
    # Applying the binary operator '!=' (line 772)
    result_ne_72495 = python_operator(stypy.reporting.localization.Localization(__file__, 772, 7), '!=', subscript_call_result_72493, str_72494)
    
    # Testing the type of an if condition (line 772)
    if_condition_72496 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 772, 4), result_ne_72495)
    # Assigning a type to the variable 'if_condition_72496' (line 772)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 4), 'if_condition_72496', if_condition_72496)
    # SSA begins for if statement (line 772)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to iscomplexfunction(...): (line 773)
    # Processing the call arguments (line 773)
    # Getting the type of 'rout' (line 773)
    rout_72498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 29), 'rout', False)
    # Processing the call keyword arguments (line 773)
    kwargs_72499 = {}
    # Getting the type of 'iscomplexfunction' (line 773)
    iscomplexfunction_72497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 11), 'iscomplexfunction', False)
    # Calling iscomplexfunction(args, kwargs) (line 773)
    iscomplexfunction_call_result_72500 = invoke(stypy.reporting.localization.Localization(__file__, 773, 11), iscomplexfunction_72497, *[rout_72498], **kwargs_72499)
    
    # Testing the type of an if condition (line 773)
    if_condition_72501 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 773, 8), iscomplexfunction_call_result_72500)
    # Assigning a type to the variable 'if_condition_72501' (line 773)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 8), 'if_condition_72501', if_condition_72501)
    # SSA begins for if statement (line 773)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Subscript (line 774):
    
    # Assigning a Str to a Subscript (line 774):
    str_72502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, (-1)), 'str', '\n#ifdef F2PY_CB_RETURNCOMPLEX\nreturn_value=\n#endif\n')
    # Getting the type of 'ret' (line 774)
    ret_72503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 12), 'ret')
    str_72504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 16), 'str', 'returncptr')
    # Storing an element on a container (line 774)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 774, 12), ret_72503, (str_72504, str_72502))
    # SSA branch for the else part of an if statement (line 773)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Subscript (line 780):
    
    # Assigning a Str to a Subscript (line 780):
    str_72505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 32), 'str', 'return_value=')
    # Getting the type of 'ret' (line 780)
    ret_72506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 12), 'ret')
    str_72507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 16), 'str', 'returncptr')
    # Storing an element on a container (line 780)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 780, 12), ret_72506, (str_72507, str_72505))
    # SSA join for if statement (line 773)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 772)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    str_72508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, 11), 'str', 'ctype')
    # Getting the type of 'ret' (line 781)
    ret_72509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 7), 'ret')
    # Obtaining the member '__getitem__' of a type (line 781)
    getitem___72510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 7), ret_72509, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 781)
    subscript_call_result_72511 = invoke(stypy.reporting.localization.Localization(__file__, 781, 7), getitem___72510, str_72508)
    
    # Getting the type of 'cformat_map' (line 781)
    cformat_map_72512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 23), 'cformat_map')
    # Applying the binary operator 'in' (line 781)
    result_contains_72513 = python_operator(stypy.reporting.localization.Localization(__file__, 781, 7), 'in', subscript_call_result_72511, cformat_map_72512)
    
    # Testing the type of an if condition (line 781)
    if_condition_72514 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 781, 4), result_contains_72513)
    # Assigning a type to the variable 'if_condition_72514' (line 781)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 781, 4), 'if_condition_72514', if_condition_72514)
    # SSA begins for if statement (line 781)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 782):
    
    # Assigning a BinOp to a Subscript (line 782):
    str_72515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 782, 33), 'str', '%s')
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    str_72516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 782, 57), 'str', 'ctype')
    # Getting the type of 'ret' (line 782)
    ret_72517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 53), 'ret')
    # Obtaining the member '__getitem__' of a type (line 782)
    getitem___72518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 53), ret_72517, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 782)
    subscript_call_result_72519 = invoke(stypy.reporting.localization.Localization(__file__, 782, 53), getitem___72518, str_72516)
    
    # Getting the type of 'cformat_map' (line 782)
    cformat_map_72520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 41), 'cformat_map')
    # Obtaining the member '__getitem__' of a type (line 782)
    getitem___72521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 41), cformat_map_72520, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 782)
    subscript_call_result_72522 = invoke(stypy.reporting.localization.Localization(__file__, 782, 41), getitem___72521, subscript_call_result_72519)
    
    # Applying the binary operator '%' (line 782)
    result_mod_72523 = python_operator(stypy.reporting.localization.Localization(__file__, 782, 33), '%', str_72515, subscript_call_result_72522)
    
    # Getting the type of 'ret' (line 782)
    ret_72524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 8), 'ret')
    str_72525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 782, 12), 'str', 'showvalueformat')
    # Storing an element on a container (line 782)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 782, 8), ret_72524, (str_72525, result_mod_72523))
    # SSA join for if statement (line 781)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isstringfunction(...): (line 783)
    # Processing the call arguments (line 783)
    # Getting the type of 'rout' (line 783)
    rout_72527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 24), 'rout', False)
    # Processing the call keyword arguments (line 783)
    kwargs_72528 = {}
    # Getting the type of 'isstringfunction' (line 783)
    isstringfunction_72526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 7), 'isstringfunction', False)
    # Calling isstringfunction(args, kwargs) (line 783)
    isstringfunction_call_result_72529 = invoke(stypy.reporting.localization.Localization(__file__, 783, 7), isstringfunction_72526, *[rout_72527], **kwargs_72528)
    
    # Testing the type of an if condition (line 783)
    if_condition_72530 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 783, 4), isstringfunction_call_result_72529)
    # Assigning a type to the variable 'if_condition_72530' (line 783)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 4), 'if_condition_72530', if_condition_72530)
    # SSA begins for if statement (line 783)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 784):
    
    # Assigning a Call to a Subscript (line 784):
    
    # Call to getstrlength(...): (line 784)
    # Processing the call arguments (line 784)
    # Getting the type of 'rout' (line 784)
    rout_72532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 40), 'rout', False)
    # Processing the call keyword arguments (line 784)
    kwargs_72533 = {}
    # Getting the type of 'getstrlength' (line 784)
    getstrlength_72531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 27), 'getstrlength', False)
    # Calling getstrlength(args, kwargs) (line 784)
    getstrlength_call_result_72534 = invoke(stypy.reporting.localization.Localization(__file__, 784, 27), getstrlength_72531, *[rout_72532], **kwargs_72533)
    
    # Getting the type of 'ret' (line 784)
    ret_72535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 8), 'ret')
    str_72536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 12), 'str', 'strlength')
    # Storing an element on a container (line 784)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 784, 8), ret_72535, (str_72536, getstrlength_call_result_72534))
    # SSA join for if statement (line 783)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isfunction(...): (line 785)
    # Processing the call arguments (line 785)
    # Getting the type of 'rout' (line 785)
    rout_72538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 18), 'rout', False)
    # Processing the call keyword arguments (line 785)
    kwargs_72539 = {}
    # Getting the type of 'isfunction' (line 785)
    isfunction_72537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 7), 'isfunction', False)
    # Calling isfunction(args, kwargs) (line 785)
    isfunction_call_result_72540 = invoke(stypy.reporting.localization.Localization(__file__, 785, 7), isfunction_72537, *[rout_72538], **kwargs_72539)
    
    # Testing the type of an if condition (line 785)
    if_condition_72541 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 785, 4), isfunction_call_result_72540)
    # Assigning a type to the variable 'if_condition_72541' (line 785)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 4), 'if_condition_72541', if_condition_72541)
    # SSA begins for if statement (line 785)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    str_72542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, 11), 'str', 'result')
    # Getting the type of 'rout' (line 786)
    rout_72543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 23), 'rout')
    # Applying the binary operator 'in' (line 786)
    result_contains_72544 = python_operator(stypy.reporting.localization.Localization(__file__, 786, 11), 'in', str_72542, rout_72543)
    
    # Testing the type of an if condition (line 786)
    if_condition_72545 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 786, 8), result_contains_72544)
    # Assigning a type to the variable 'if_condition_72545' (line 786)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 786, 8), 'if_condition_72545', if_condition_72545)
    # SSA begins for if statement (line 786)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 787):
    
    # Assigning a Subscript to a Name (line 787):
    
    # Obtaining the type of the subscript
    str_72546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 787, 21), 'str', 'result')
    # Getting the type of 'rout' (line 787)
    rout_72547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 16), 'rout')
    # Obtaining the member '__getitem__' of a type (line 787)
    getitem___72548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 16), rout_72547, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 787)
    subscript_call_result_72549 = invoke(stypy.reporting.localization.Localization(__file__, 787, 16), getitem___72548, str_72546)
    
    # Assigning a type to the variable 'a' (line 787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 12), 'a', subscript_call_result_72549)
    # SSA branch for the else part of an if statement (line 786)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 789):
    
    # Assigning a Subscript to a Name (line 789):
    
    # Obtaining the type of the subscript
    str_72550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 789, 21), 'str', 'name')
    # Getting the type of 'rout' (line 789)
    rout_72551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 16), 'rout')
    # Obtaining the member '__getitem__' of a type (line 789)
    getitem___72552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 789, 16), rout_72551, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 789)
    subscript_call_result_72553 = invoke(stypy.reporting.localization.Localization(__file__, 789, 16), getitem___72552, str_72550)
    
    # Assigning a type to the variable 'a' (line 789)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 789, 12), 'a', subscript_call_result_72553)
    # SSA join for if statement (line 786)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to hasnote(...): (line 790)
    # Processing the call arguments (line 790)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 790)
    a_72555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 32), 'a', False)
    
    # Obtaining the type of the subscript
    str_72556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 790, 24), 'str', 'vars')
    # Getting the type of 'rout' (line 790)
    rout_72557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 19), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 790)
    getitem___72558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 790, 19), rout_72557, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 790)
    subscript_call_result_72559 = invoke(stypy.reporting.localization.Localization(__file__, 790, 19), getitem___72558, str_72556)
    
    # Obtaining the member '__getitem__' of a type (line 790)
    getitem___72560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 790, 19), subscript_call_result_72559, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 790)
    subscript_call_result_72561 = invoke(stypy.reporting.localization.Localization(__file__, 790, 19), getitem___72560, a_72555)
    
    # Processing the call keyword arguments (line 790)
    kwargs_72562 = {}
    # Getting the type of 'hasnote' (line 790)
    hasnote_72554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 11), 'hasnote', False)
    # Calling hasnote(args, kwargs) (line 790)
    hasnote_call_result_72563 = invoke(stypy.reporting.localization.Localization(__file__, 790, 11), hasnote_72554, *[subscript_call_result_72561], **kwargs_72562)
    
    # Testing the type of an if condition (line 790)
    if_condition_72564 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 790, 8), hasnote_call_result_72563)
    # Assigning a type to the variable 'if_condition_72564' (line 790)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 790, 8), 'if_condition_72564', if_condition_72564)
    # SSA begins for if statement (line 790)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 791):
    
    # Assigning a Subscript to a Subscript (line 791):
    
    # Obtaining the type of the subscript
    str_72565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 791, 42), 'str', 'note')
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 791)
    a_72566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 39), 'a')
    
    # Obtaining the type of the subscript
    str_72567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 791, 31), 'str', 'vars')
    # Getting the type of 'rout' (line 791)
    rout_72568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 26), 'rout')
    # Obtaining the member '__getitem__' of a type (line 791)
    getitem___72569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 26), rout_72568, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 791)
    subscript_call_result_72570 = invoke(stypy.reporting.localization.Localization(__file__, 791, 26), getitem___72569, str_72567)
    
    # Obtaining the member '__getitem__' of a type (line 791)
    getitem___72571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 26), subscript_call_result_72570, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 791)
    subscript_call_result_72572 = invoke(stypy.reporting.localization.Localization(__file__, 791, 26), getitem___72571, a_72566)
    
    # Obtaining the member '__getitem__' of a type (line 791)
    getitem___72573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 26), subscript_call_result_72572, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 791)
    subscript_call_result_72574 = invoke(stypy.reporting.localization.Localization(__file__, 791, 26), getitem___72573, str_72565)
    
    # Getting the type of 'ret' (line 791)
    ret_72575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 12), 'ret')
    str_72576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 791, 16), 'str', 'note')
    # Storing an element on a container (line 791)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 791, 12), ret_72575, (str_72576, subscript_call_result_72574))
    
    # Assigning a List to a Subscript (line 792):
    
    # Assigning a List to a Subscript (line 792):
    
    # Obtaining an instance of the builtin type 'list' (line 792)
    list_72577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 792)
    # Adding element type (line 792)
    str_72578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, 39), 'str', 'See elsewhere.')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 792, 38), list_72577, str_72578)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 792)
    a_72579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 25), 'a')
    
    # Obtaining the type of the subscript
    str_72580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, 17), 'str', 'vars')
    # Getting the type of 'rout' (line 792)
    rout_72581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 12), 'rout')
    # Obtaining the member '__getitem__' of a type (line 792)
    getitem___72582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 12), rout_72581, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 792)
    subscript_call_result_72583 = invoke(stypy.reporting.localization.Localization(__file__, 792, 12), getitem___72582, str_72580)
    
    # Obtaining the member '__getitem__' of a type (line 792)
    getitem___72584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 12), subscript_call_result_72583, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 792)
    subscript_call_result_72585 = invoke(stypy.reporting.localization.Localization(__file__, 792, 12), getitem___72584, a_72579)
    
    str_72586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, 28), 'str', 'note')
    # Storing an element on a container (line 792)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 792, 12), subscript_call_result_72585, (str_72586, list_72577))
    # SSA join for if statement (line 790)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 793):
    
    # Assigning a Name to a Subscript (line 793):
    # Getting the type of 'a' (line 793)
    a_72587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 23), 'a')
    # Getting the type of 'ret' (line 793)
    ret_72588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 8), 'ret')
    str_72589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 793, 12), 'str', 'rname')
    # Storing an element on a container (line 793)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 793, 8), ret_72588, (str_72589, a_72587))
    
    # Assigning a Call to a Tuple (line 794):
    
    # Assigning a Call to a Name:
    
    # Call to getpydocsign(...): (line 794)
    # Processing the call arguments (line 794)
    # Getting the type of 'a' (line 794)
    a_72591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 61), 'a', False)
    # Getting the type of 'rout' (line 794)
    rout_72592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 64), 'rout', False)
    # Processing the call keyword arguments (line 794)
    kwargs_72593 = {}
    # Getting the type of 'getpydocsign' (line 794)
    getpydocsign_72590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 48), 'getpydocsign', False)
    # Calling getpydocsign(args, kwargs) (line 794)
    getpydocsign_call_result_72594 = invoke(stypy.reporting.localization.Localization(__file__, 794, 48), getpydocsign_72590, *[a_72591, rout_72592], **kwargs_72593)
    
    # Assigning a type to the variable 'call_assignment_69378' (line 794)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 794, 8), 'call_assignment_69378', getpydocsign_call_result_72594)
    
    # Assigning a Call to a Name (line 794):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_72597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 794, 8), 'int')
    # Processing the call keyword arguments
    kwargs_72598 = {}
    # Getting the type of 'call_assignment_69378' (line 794)
    call_assignment_69378_72595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 8), 'call_assignment_69378', False)
    # Obtaining the member '__getitem__' of a type (line 794)
    getitem___72596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 8), call_assignment_69378_72595, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_72599 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___72596, *[int_72597], **kwargs_72598)
    
    # Assigning a type to the variable 'call_assignment_69379' (line 794)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 794, 8), 'call_assignment_69379', getitem___call_result_72599)
    
    # Assigning a Name to a Subscript (line 794):
    # Getting the type of 'call_assignment_69379' (line 794)
    call_assignment_69379_72600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 8), 'call_assignment_69379')
    # Getting the type of 'ret' (line 794)
    ret_72601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 8), 'ret')
    str_72602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 794, 12), 'str', 'pydocsign')
    # Storing an element on a container (line 794)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 794, 8), ret_72601, (str_72602, call_assignment_69379_72600))
    
    # Assigning a Call to a Name (line 794):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_72605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 794, 8), 'int')
    # Processing the call keyword arguments
    kwargs_72606 = {}
    # Getting the type of 'call_assignment_69378' (line 794)
    call_assignment_69378_72603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 8), 'call_assignment_69378', False)
    # Obtaining the member '__getitem__' of a type (line 794)
    getitem___72604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 8), call_assignment_69378_72603, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_72607 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___72604, *[int_72605], **kwargs_72606)
    
    # Assigning a type to the variable 'call_assignment_69380' (line 794)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 794, 8), 'call_assignment_69380', getitem___call_result_72607)
    
    # Assigning a Name to a Subscript (line 794):
    # Getting the type of 'call_assignment_69380' (line 794)
    call_assignment_69380_72608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 8), 'call_assignment_69380')
    # Getting the type of 'ret' (line 794)
    ret_72609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 26), 'ret')
    str_72610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 794, 30), 'str', 'pydocsignout')
    # Storing an element on a container (line 794)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 794, 26), ret_72609, (str_72610, call_assignment_69380_72608))
    
    
    # Call to iscomplexfunction(...): (line 795)
    # Processing the call arguments (line 795)
    # Getting the type of 'rout' (line 795)
    rout_72612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 29), 'rout', False)
    # Processing the call keyword arguments (line 795)
    kwargs_72613 = {}
    # Getting the type of 'iscomplexfunction' (line 795)
    iscomplexfunction_72611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 11), 'iscomplexfunction', False)
    # Calling iscomplexfunction(args, kwargs) (line 795)
    iscomplexfunction_call_result_72614 = invoke(stypy.reporting.localization.Localization(__file__, 795, 11), iscomplexfunction_72611, *[rout_72612], **kwargs_72613)
    
    # Testing the type of an if condition (line 795)
    if_condition_72615 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 795, 8), iscomplexfunction_call_result_72614)
    # Assigning a type to the variable 'if_condition_72615' (line 795)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 795, 8), 'if_condition_72615', if_condition_72615)
    # SSA begins for if statement (line 795)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Subscript (line 796):
    
    # Assigning a Str to a Subscript (line 796):
    str_72616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 802, (-1)), 'str', '\n#ifdef F2PY_CB_RETURNCOMPLEX\n#ctype#\n#else\nvoid\n#endif\n')
    # Getting the type of 'ret' (line 796)
    ret_72617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 12), 'ret')
    str_72618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 796, 16), 'str', 'rctype')
    # Storing an element on a container (line 796)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 796, 12), ret_72617, (str_72618, str_72616))
    # SSA join for if statement (line 795)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 785)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to hasnote(...): (line 804)
    # Processing the call arguments (line 804)
    # Getting the type of 'rout' (line 804)
    rout_72620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 19), 'rout', False)
    # Processing the call keyword arguments (line 804)
    kwargs_72621 = {}
    # Getting the type of 'hasnote' (line 804)
    hasnote_72619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 11), 'hasnote', False)
    # Calling hasnote(args, kwargs) (line 804)
    hasnote_call_result_72622 = invoke(stypy.reporting.localization.Localization(__file__, 804, 11), hasnote_72619, *[rout_72620], **kwargs_72621)
    
    # Testing the type of an if condition (line 804)
    if_condition_72623 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 804, 8), hasnote_call_result_72622)
    # Assigning a type to the variable 'if_condition_72623' (line 804)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 804, 8), 'if_condition_72623', if_condition_72623)
    # SSA begins for if statement (line 804)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 805):
    
    # Assigning a Subscript to a Subscript (line 805):
    
    # Obtaining the type of the subscript
    str_72624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, 31), 'str', 'note')
    # Getting the type of 'rout' (line 805)
    rout_72625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 26), 'rout')
    # Obtaining the member '__getitem__' of a type (line 805)
    getitem___72626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 26), rout_72625, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 805)
    subscript_call_result_72627 = invoke(stypy.reporting.localization.Localization(__file__, 805, 26), getitem___72626, str_72624)
    
    # Getting the type of 'ret' (line 805)
    ret_72628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 12), 'ret')
    str_72629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, 16), 'str', 'note')
    # Storing an element on a container (line 805)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 805, 12), ret_72628, (str_72629, subscript_call_result_72627))
    
    # Assigning a List to a Subscript (line 806):
    
    # Assigning a List to a Subscript (line 806):
    
    # Obtaining an instance of the builtin type 'list' (line 806)
    list_72630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 806)
    # Adding element type (line 806)
    str_72631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 28), 'str', 'See elsewhere.')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 806, 27), list_72630, str_72631)
    
    # Getting the type of 'rout' (line 806)
    rout_72632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 12), 'rout')
    str_72633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 17), 'str', 'note')
    # Storing an element on a container (line 806)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 806, 12), rout_72632, (str_72633, list_72630))
    # SSA join for if statement (line 804)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 785)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 807):
    
    # Assigning a Num to a Name (line 807):
    int_72634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 14), 'int')
    # Assigning a type to the variable 'nofargs' (line 807)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 807, 4), 'nofargs', int_72634)
    
    # Assigning a Num to a Name (line 808):
    
    # Assigning a Num to a Name (line 808):
    int_72635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 17), 'int')
    # Assigning a type to the variable 'nofoptargs' (line 808)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 808, 4), 'nofoptargs', int_72635)
    
    
    # Evaluating a boolean operation
    
    str_72636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 809, 7), 'str', 'args')
    # Getting the type of 'rout' (line 809)
    rout_72637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 17), 'rout')
    # Applying the binary operator 'in' (line 809)
    result_contains_72638 = python_operator(stypy.reporting.localization.Localization(__file__, 809, 7), 'in', str_72636, rout_72637)
    
    
    str_72639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 809, 26), 'str', 'vars')
    # Getting the type of 'rout' (line 809)
    rout_72640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 36), 'rout')
    # Applying the binary operator 'in' (line 809)
    result_contains_72641 = python_operator(stypy.reporting.localization.Localization(__file__, 809, 26), 'in', str_72639, rout_72640)
    
    # Applying the binary operator 'and' (line 809)
    result_and_keyword_72642 = python_operator(stypy.reporting.localization.Localization(__file__, 809, 7), 'and', result_contains_72638, result_contains_72641)
    
    # Testing the type of an if condition (line 809)
    if_condition_72643 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 809, 4), result_and_keyword_72642)
    # Assigning a type to the variable 'if_condition_72643' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 4), 'if_condition_72643', if_condition_72643)
    # SSA begins for if statement (line 809)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Obtaining the type of the subscript
    str_72644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 810, 22), 'str', 'args')
    # Getting the type of 'rout' (line 810)
    rout_72645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 17), 'rout')
    # Obtaining the member '__getitem__' of a type (line 810)
    getitem___72646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 810, 17), rout_72645, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 810)
    subscript_call_result_72647 = invoke(stypy.reporting.localization.Localization(__file__, 810, 17), getitem___72646, str_72644)
    
    # Testing the type of a for loop iterable (line 810)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 810, 8), subscript_call_result_72647)
    # Getting the type of the for loop variable (line 810)
    for_loop_var_72648 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 810, 8), subscript_call_result_72647)
    # Assigning a type to the variable 'a' (line 810)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 8), 'a', for_loop_var_72648)
    # SSA begins for a for statement (line 810)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 811):
    
    # Assigning a Subscript to a Name (line 811):
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 811)
    a_72649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 31), 'a')
    
    # Obtaining the type of the subscript
    str_72650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 811, 23), 'str', 'vars')
    # Getting the type of 'rout' (line 811)
    rout_72651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 18), 'rout')
    # Obtaining the member '__getitem__' of a type (line 811)
    getitem___72652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 18), rout_72651, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 811)
    subscript_call_result_72653 = invoke(stypy.reporting.localization.Localization(__file__, 811, 18), getitem___72652, str_72650)
    
    # Obtaining the member '__getitem__' of a type (line 811)
    getitem___72654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 18), subscript_call_result_72653, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 811)
    subscript_call_result_72655 = invoke(stypy.reporting.localization.Localization(__file__, 811, 18), getitem___72654, a_72649)
    
    # Assigning a type to the variable 'var' (line 811)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 811, 12), 'var', subscript_call_result_72655)
    
    
    # Call to (...): (line 812)
    # Processing the call arguments (line 812)
    # Getting the type of 'var' (line 812)
    var_72661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 49), 'var', False)
    # Processing the call keyword arguments (line 812)
    kwargs_72662 = {}
    
    # Call to l_or(...): (line 812)
    # Processing the call arguments (line 812)
    # Getting the type of 'isintent_in' (line 812)
    isintent_in_72657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 20), 'isintent_in', False)
    # Getting the type of 'isintent_inout' (line 812)
    isintent_inout_72658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 33), 'isintent_inout', False)
    # Processing the call keyword arguments (line 812)
    kwargs_72659 = {}
    # Getting the type of 'l_or' (line 812)
    l_or_72656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 15), 'l_or', False)
    # Calling l_or(args, kwargs) (line 812)
    l_or_call_result_72660 = invoke(stypy.reporting.localization.Localization(__file__, 812, 15), l_or_72656, *[isintent_in_72657, isintent_inout_72658], **kwargs_72659)
    
    # Calling (args, kwargs) (line 812)
    _call_result_72663 = invoke(stypy.reporting.localization.Localization(__file__, 812, 15), l_or_call_result_72660, *[var_72661], **kwargs_72662)
    
    # Testing the type of an if condition (line 812)
    if_condition_72664 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 812, 12), _call_result_72663)
    # Assigning a type to the variable 'if_condition_72664' (line 812)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 12), 'if_condition_72664', if_condition_72664)
    # SSA begins for if statement (line 812)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 813):
    
    # Assigning a BinOp to a Name (line 813):
    # Getting the type of 'nofargs' (line 813)
    nofargs_72665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 26), 'nofargs')
    int_72666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 36), 'int')
    # Applying the binary operator '+' (line 813)
    result_add_72667 = python_operator(stypy.reporting.localization.Localization(__file__, 813, 26), '+', nofargs_72665, int_72666)
    
    # Assigning a type to the variable 'nofargs' (line 813)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 813, 16), 'nofargs', result_add_72667)
    
    
    # Call to isoptional(...): (line 814)
    # Processing the call arguments (line 814)
    # Getting the type of 'var' (line 814)
    var_72669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 30), 'var', False)
    # Processing the call keyword arguments (line 814)
    kwargs_72670 = {}
    # Getting the type of 'isoptional' (line 814)
    isoptional_72668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 19), 'isoptional', False)
    # Calling isoptional(args, kwargs) (line 814)
    isoptional_call_result_72671 = invoke(stypy.reporting.localization.Localization(__file__, 814, 19), isoptional_72668, *[var_72669], **kwargs_72670)
    
    # Testing the type of an if condition (line 814)
    if_condition_72672 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 814, 16), isoptional_call_result_72671)
    # Assigning a type to the variable 'if_condition_72672' (line 814)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 814, 16), 'if_condition_72672', if_condition_72672)
    # SSA begins for if statement (line 814)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 815):
    
    # Assigning a BinOp to a Name (line 815):
    # Getting the type of 'nofoptargs' (line 815)
    nofoptargs_72673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 33), 'nofoptargs')
    int_72674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 46), 'int')
    # Applying the binary operator '+' (line 815)
    result_add_72675 = python_operator(stypy.reporting.localization.Localization(__file__, 815, 33), '+', nofoptargs_72673, int_72674)
    
    # Assigning a type to the variable 'nofoptargs' (line 815)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 20), 'nofoptargs', result_add_72675)
    # SSA join for if statement (line 814)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 812)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 809)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Subscript (line 816):
    
    # Assigning a Call to a Subscript (line 816):
    
    # Call to repr(...): (line 816)
    # Processing the call arguments (line 816)
    # Getting the type of 'nofargs' (line 816)
    nofargs_72677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 29), 'nofargs', False)
    # Processing the call keyword arguments (line 816)
    kwargs_72678 = {}
    # Getting the type of 'repr' (line 816)
    repr_72676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 24), 'repr', False)
    # Calling repr(args, kwargs) (line 816)
    repr_call_result_72679 = invoke(stypy.reporting.localization.Localization(__file__, 816, 24), repr_72676, *[nofargs_72677], **kwargs_72678)
    
    # Getting the type of 'ret' (line 816)
    ret_72680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 4), 'ret')
    str_72681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 816, 8), 'str', 'maxnofargs')
    # Storing an element on a container (line 816)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 816, 4), ret_72680, (str_72681, repr_call_result_72679))
    
    # Assigning a Call to a Subscript (line 817):
    
    # Assigning a Call to a Subscript (line 817):
    
    # Call to repr(...): (line 817)
    # Processing the call arguments (line 817)
    # Getting the type of 'nofoptargs' (line 817)
    nofoptargs_72683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 29), 'nofoptargs', False)
    # Processing the call keyword arguments (line 817)
    kwargs_72684 = {}
    # Getting the type of 'repr' (line 817)
    repr_72682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 24), 'repr', False)
    # Calling repr(args, kwargs) (line 817)
    repr_call_result_72685 = invoke(stypy.reporting.localization.Localization(__file__, 817, 24), repr_72682, *[nofoptargs_72683], **kwargs_72684)
    
    # Getting the type of 'ret' (line 817)
    ret_72686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 4), 'ret')
    str_72687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 8), 'str', 'nofoptargs')
    # Storing an element on a container (line 817)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 817, 4), ret_72686, (str_72687, repr_call_result_72685))
    
    
    # Evaluating a boolean operation
    
    # Call to hasnote(...): (line 818)
    # Processing the call arguments (line 818)
    # Getting the type of 'rout' (line 818)
    rout_72689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 15), 'rout', False)
    # Processing the call keyword arguments (line 818)
    kwargs_72690 = {}
    # Getting the type of 'hasnote' (line 818)
    hasnote_72688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 7), 'hasnote', False)
    # Calling hasnote(args, kwargs) (line 818)
    hasnote_call_result_72691 = invoke(stypy.reporting.localization.Localization(__file__, 818, 7), hasnote_72688, *[rout_72689], **kwargs_72690)
    
    
    # Call to isfunction(...): (line 818)
    # Processing the call arguments (line 818)
    # Getting the type of 'rout' (line 818)
    rout_72693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 36), 'rout', False)
    # Processing the call keyword arguments (line 818)
    kwargs_72694 = {}
    # Getting the type of 'isfunction' (line 818)
    isfunction_72692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 25), 'isfunction', False)
    # Calling isfunction(args, kwargs) (line 818)
    isfunction_call_result_72695 = invoke(stypy.reporting.localization.Localization(__file__, 818, 25), isfunction_72692, *[rout_72693], **kwargs_72694)
    
    # Applying the binary operator 'and' (line 818)
    result_and_keyword_72696 = python_operator(stypy.reporting.localization.Localization(__file__, 818, 7), 'and', hasnote_call_result_72691, isfunction_call_result_72695)
    
    str_72697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 46), 'str', 'result')
    # Getting the type of 'rout' (line 818)
    rout_72698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 58), 'rout')
    # Applying the binary operator 'in' (line 818)
    result_contains_72699 = python_operator(stypy.reporting.localization.Localization(__file__, 818, 46), 'in', str_72697, rout_72698)
    
    # Applying the binary operator 'and' (line 818)
    result_and_keyword_72700 = python_operator(stypy.reporting.localization.Localization(__file__, 818, 7), 'and', result_and_keyword_72696, result_contains_72699)
    
    # Testing the type of an if condition (line 818)
    if_condition_72701 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 818, 4), result_and_keyword_72700)
    # Assigning a type to the variable 'if_condition_72701' (line 818)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 818, 4), 'if_condition_72701', if_condition_72701)
    # SSA begins for if statement (line 818)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 819):
    
    # Assigning a Subscript to a Subscript (line 819):
    
    # Obtaining the type of the subscript
    str_72702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 31), 'str', 'note')
    # Getting the type of 'rout' (line 819)
    rout_72703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 26), 'rout')
    # Obtaining the member '__getitem__' of a type (line 819)
    getitem___72704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 819, 26), rout_72703, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 819)
    subscript_call_result_72705 = invoke(stypy.reporting.localization.Localization(__file__, 819, 26), getitem___72704, str_72702)
    
    # Getting the type of 'ret' (line 819)
    ret_72706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 8), 'ret')
    str_72707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 12), 'str', 'routnote')
    # Storing an element on a container (line 819)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 819, 8), ret_72706, (str_72707, subscript_call_result_72705))
    
    # Assigning a List to a Subscript (line 820):
    
    # Assigning a List to a Subscript (line 820):
    
    # Obtaining an instance of the builtin type 'list' (line 820)
    list_72708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 820)
    # Adding element type (line 820)
    str_72709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, 24), 'str', 'See elsewhere.')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 820, 23), list_72708, str_72709)
    
    # Getting the type of 'rout' (line 820)
    rout_72710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 8), 'rout')
    str_72711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, 13), 'str', 'note')
    # Storing an element on a container (line 820)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 820, 8), rout_72710, (str_72711, list_72708))
    # SSA join for if statement (line 818)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ret' (line 821)
    ret_72712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 11), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 821)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 821, 4), 'stypy_return_type', ret_72712)
    
    # ################# End of 'cb_routsign2map(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cb_routsign2map' in the type store
    # Getting the type of 'stypy_return_type' (line 742)
    stypy_return_type_72713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_72713)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cb_routsign2map'
    return stypy_return_type_72713

# Assigning a type to the variable 'cb_routsign2map' (line 742)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 0), 'cb_routsign2map', cb_routsign2map)

@norecursion
def common_sign2map(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'common_sign2map'
    module_type_store = module_type_store.open_function_context('common_sign2map', 824, 0, False)
    
    # Passed parameters checking function
    common_sign2map.stypy_localization = localization
    common_sign2map.stypy_type_of_self = None
    common_sign2map.stypy_type_store = module_type_store
    common_sign2map.stypy_function_name = 'common_sign2map'
    common_sign2map.stypy_param_names_list = ['a', 'var']
    common_sign2map.stypy_varargs_param_name = None
    common_sign2map.stypy_kwargs_param_name = None
    common_sign2map.stypy_call_defaults = defaults
    common_sign2map.stypy_call_varargs = varargs
    common_sign2map.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'common_sign2map', ['a', 'var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'common_sign2map', localization, ['a', 'var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'common_sign2map(...)' code ##################

    
    # Assigning a Dict to a Name (line 825):
    
    # Assigning a Dict to a Name (line 825):
    
    # Obtaining an instance of the builtin type 'dict' (line 825)
    dict_72714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 825, 10), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 825)
    # Adding element type (key, value) (line 825)
    str_72715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 825, 11), 'str', 'varname')
    # Getting the type of 'a' (line 825)
    a_72716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 22), 'a')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 825, 10), dict_72714, (str_72715, a_72716))
    # Adding element type (key, value) (line 825)
    str_72717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 825, 25), 'str', 'ctype')
    
    # Call to getctype(...): (line 825)
    # Processing the call arguments (line 825)
    # Getting the type of 'var' (line 825)
    var_72719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 43), 'var', False)
    # Processing the call keyword arguments (line 825)
    kwargs_72720 = {}
    # Getting the type of 'getctype' (line 825)
    getctype_72718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 34), 'getctype', False)
    # Calling getctype(args, kwargs) (line 825)
    getctype_call_result_72721 = invoke(stypy.reporting.localization.Localization(__file__, 825, 34), getctype_72718, *[var_72719], **kwargs_72720)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 825, 10), dict_72714, (str_72717, getctype_call_result_72721))
    
    # Assigning a type to the variable 'ret' (line 825)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 825, 4), 'ret', dict_72714)
    
    
    # Call to isstringarray(...): (line 826)
    # Processing the call arguments (line 826)
    # Getting the type of 'var' (line 826)
    var_72723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 21), 'var', False)
    # Processing the call keyword arguments (line 826)
    kwargs_72724 = {}
    # Getting the type of 'isstringarray' (line 826)
    isstringarray_72722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 7), 'isstringarray', False)
    # Calling isstringarray(args, kwargs) (line 826)
    isstringarray_call_result_72725 = invoke(stypy.reporting.localization.Localization(__file__, 826, 7), isstringarray_72722, *[var_72723], **kwargs_72724)
    
    # Testing the type of an if condition (line 826)
    if_condition_72726 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 826, 4), isstringarray_call_result_72725)
    # Assigning a type to the variable 'if_condition_72726' (line 826)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 826, 4), 'if_condition_72726', if_condition_72726)
    # SSA begins for if statement (line 826)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Subscript (line 827):
    
    # Assigning a Str to a Subscript (line 827):
    str_72727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 827, 23), 'str', 'char')
    # Getting the type of 'ret' (line 827)
    ret_72728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 8), 'ret')
    str_72729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 827, 12), 'str', 'ctype')
    # Storing an element on a container (line 827)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 827, 8), ret_72728, (str_72729, str_72727))
    # SSA join for if statement (line 826)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    str_72730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 828, 11), 'str', 'ctype')
    # Getting the type of 'ret' (line 828)
    ret_72731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 7), 'ret')
    # Obtaining the member '__getitem__' of a type (line 828)
    getitem___72732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 828, 7), ret_72731, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 828)
    subscript_call_result_72733 = invoke(stypy.reporting.localization.Localization(__file__, 828, 7), getitem___72732, str_72730)
    
    # Getting the type of 'c2capi_map' (line 828)
    c2capi_map_72734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 23), 'c2capi_map')
    # Applying the binary operator 'in' (line 828)
    result_contains_72735 = python_operator(stypy.reporting.localization.Localization(__file__, 828, 7), 'in', subscript_call_result_72733, c2capi_map_72734)
    
    # Testing the type of an if condition (line 828)
    if_condition_72736 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 828, 4), result_contains_72735)
    # Assigning a type to the variable 'if_condition_72736' (line 828)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 828, 4), 'if_condition_72736', if_condition_72736)
    # SSA begins for if statement (line 828)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 829):
    
    # Assigning a Subscript to a Subscript (line 829):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    str_72737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 38), 'str', 'ctype')
    # Getting the type of 'ret' (line 829)
    ret_72738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 34), 'ret')
    # Obtaining the member '__getitem__' of a type (line 829)
    getitem___72739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 34), ret_72738, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 829)
    subscript_call_result_72740 = invoke(stypy.reporting.localization.Localization(__file__, 829, 34), getitem___72739, str_72737)
    
    # Getting the type of 'c2capi_map' (line 829)
    c2capi_map_72741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 23), 'c2capi_map')
    # Obtaining the member '__getitem__' of a type (line 829)
    getitem___72742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 23), c2capi_map_72741, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 829)
    subscript_call_result_72743 = invoke(stypy.reporting.localization.Localization(__file__, 829, 23), getitem___72742, subscript_call_result_72740)
    
    # Getting the type of 'ret' (line 829)
    ret_72744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 8), 'ret')
    str_72745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 12), 'str', 'atype')
    # Storing an element on a container (line 829)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 829, 8), ret_72744, (str_72745, subscript_call_result_72743))
    # SSA join for if statement (line 828)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    str_72746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 11), 'str', 'ctype')
    # Getting the type of 'ret' (line 830)
    ret_72747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 7), 'ret')
    # Obtaining the member '__getitem__' of a type (line 830)
    getitem___72748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 830, 7), ret_72747, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 830)
    subscript_call_result_72749 = invoke(stypy.reporting.localization.Localization(__file__, 830, 7), getitem___72748, str_72746)
    
    # Getting the type of 'cformat_map' (line 830)
    cformat_map_72750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 23), 'cformat_map')
    # Applying the binary operator 'in' (line 830)
    result_contains_72751 = python_operator(stypy.reporting.localization.Localization(__file__, 830, 7), 'in', subscript_call_result_72749, cformat_map_72750)
    
    # Testing the type of an if condition (line 830)
    if_condition_72752 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 830, 4), result_contains_72751)
    # Assigning a type to the variable 'if_condition_72752' (line 830)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 830, 4), 'if_condition_72752', if_condition_72752)
    # SSA begins for if statement (line 830)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 831):
    
    # Assigning a BinOp to a Subscript (line 831):
    str_72753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, 33), 'str', '%s')
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    str_72754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, 57), 'str', 'ctype')
    # Getting the type of 'ret' (line 831)
    ret_72755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 53), 'ret')
    # Obtaining the member '__getitem__' of a type (line 831)
    getitem___72756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 831, 53), ret_72755, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 831)
    subscript_call_result_72757 = invoke(stypy.reporting.localization.Localization(__file__, 831, 53), getitem___72756, str_72754)
    
    # Getting the type of 'cformat_map' (line 831)
    cformat_map_72758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 41), 'cformat_map')
    # Obtaining the member '__getitem__' of a type (line 831)
    getitem___72759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 831, 41), cformat_map_72758, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 831)
    subscript_call_result_72760 = invoke(stypy.reporting.localization.Localization(__file__, 831, 41), getitem___72759, subscript_call_result_72757)
    
    # Applying the binary operator '%' (line 831)
    result_mod_72761 = python_operator(stypy.reporting.localization.Localization(__file__, 831, 33), '%', str_72753, subscript_call_result_72760)
    
    # Getting the type of 'ret' (line 831)
    ret_72762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 8), 'ret')
    str_72763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, 12), 'str', 'showvalueformat')
    # Storing an element on a container (line 831)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 831, 8), ret_72762, (str_72763, result_mod_72761))
    # SSA join for if statement (line 830)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isarray(...): (line 832)
    # Processing the call arguments (line 832)
    # Getting the type of 'var' (line 832)
    var_72765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 15), 'var', False)
    # Processing the call keyword arguments (line 832)
    kwargs_72766 = {}
    # Getting the type of 'isarray' (line 832)
    isarray_72764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 7), 'isarray', False)
    # Calling isarray(args, kwargs) (line 832)
    isarray_call_result_72767 = invoke(stypy.reporting.localization.Localization(__file__, 832, 7), isarray_72764, *[var_72765], **kwargs_72766)
    
    # Testing the type of an if condition (line 832)
    if_condition_72768 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 832, 4), isarray_call_result_72767)
    # Assigning a type to the variable 'if_condition_72768' (line 832)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 832, 4), 'if_condition_72768', if_condition_72768)
    # SSA begins for if statement (line 832)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 833):
    
    # Assigning a Call to a Name (line 833):
    
    # Call to dictappend(...): (line 833)
    # Processing the call arguments (line 833)
    # Getting the type of 'ret' (line 833)
    ret_72770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 25), 'ret', False)
    
    # Call to getarrdims(...): (line 833)
    # Processing the call arguments (line 833)
    # Getting the type of 'a' (line 833)
    a_72772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 41), 'a', False)
    # Getting the type of 'var' (line 833)
    var_72773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 44), 'var', False)
    # Processing the call keyword arguments (line 833)
    kwargs_72774 = {}
    # Getting the type of 'getarrdims' (line 833)
    getarrdims_72771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 30), 'getarrdims', False)
    # Calling getarrdims(args, kwargs) (line 833)
    getarrdims_call_result_72775 = invoke(stypy.reporting.localization.Localization(__file__, 833, 30), getarrdims_72771, *[a_72772, var_72773], **kwargs_72774)
    
    # Processing the call keyword arguments (line 833)
    kwargs_72776 = {}
    # Getting the type of 'dictappend' (line 833)
    dictappend_72769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 14), 'dictappend', False)
    # Calling dictappend(args, kwargs) (line 833)
    dictappend_call_result_72777 = invoke(stypy.reporting.localization.Localization(__file__, 833, 14), dictappend_72769, *[ret_72770, getarrdims_call_result_72775], **kwargs_72776)
    
    # Assigning a type to the variable 'ret' (line 833)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 833, 8), 'ret', dictappend_call_result_72777)
    # SSA branch for the else part of an if statement (line 832)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isstring(...): (line 834)
    # Processing the call arguments (line 834)
    # Getting the type of 'var' (line 834)
    var_72779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 18), 'var', False)
    # Processing the call keyword arguments (line 834)
    kwargs_72780 = {}
    # Getting the type of 'isstring' (line 834)
    isstring_72778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 9), 'isstring', False)
    # Calling isstring(args, kwargs) (line 834)
    isstring_call_result_72781 = invoke(stypy.reporting.localization.Localization(__file__, 834, 9), isstring_72778, *[var_72779], **kwargs_72780)
    
    # Testing the type of an if condition (line 834)
    if_condition_72782 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 834, 9), isstring_call_result_72781)
    # Assigning a type to the variable 'if_condition_72782' (line 834)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 834, 9), 'if_condition_72782', if_condition_72782)
    # SSA begins for if statement (line 834)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 835):
    
    # Assigning a Call to a Subscript (line 835):
    
    # Call to getstrlength(...): (line 835)
    # Processing the call arguments (line 835)
    # Getting the type of 'var' (line 835)
    var_72784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 35), 'var', False)
    # Processing the call keyword arguments (line 835)
    kwargs_72785 = {}
    # Getting the type of 'getstrlength' (line 835)
    getstrlength_72783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 22), 'getstrlength', False)
    # Calling getstrlength(args, kwargs) (line 835)
    getstrlength_call_result_72786 = invoke(stypy.reporting.localization.Localization(__file__, 835, 22), getstrlength_72783, *[var_72784], **kwargs_72785)
    
    # Getting the type of 'ret' (line 835)
    ret_72787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 8), 'ret')
    str_72788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 835, 12), 'str', 'size')
    # Storing an element on a container (line 835)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 835, 8), ret_72787, (str_72788, getstrlength_call_result_72786))
    
    # Assigning a Str to a Subscript (line 836):
    
    # Assigning a Str to a Subscript (line 836):
    str_72789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 836, 22), 'str', '1')
    # Getting the type of 'ret' (line 836)
    ret_72790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 8), 'ret')
    str_72791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 836, 12), 'str', 'rank')
    # Storing an element on a container (line 836)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 836, 8), ret_72790, (str_72791, str_72789))
    # SSA join for if statement (line 834)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 832)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 837):
    
    # Assigning a Call to a Name:
    
    # Call to getpydocsign(...): (line 837)
    # Processing the call arguments (line 837)
    # Getting the type of 'a' (line 837)
    a_72793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 57), 'a', False)
    # Getting the type of 'var' (line 837)
    var_72794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 60), 'var', False)
    # Processing the call keyword arguments (line 837)
    kwargs_72795 = {}
    # Getting the type of 'getpydocsign' (line 837)
    getpydocsign_72792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 44), 'getpydocsign', False)
    # Calling getpydocsign(args, kwargs) (line 837)
    getpydocsign_call_result_72796 = invoke(stypy.reporting.localization.Localization(__file__, 837, 44), getpydocsign_72792, *[a_72793, var_72794], **kwargs_72795)
    
    # Assigning a type to the variable 'call_assignment_69381' (line 837)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 837, 4), 'call_assignment_69381', getpydocsign_call_result_72796)
    
    # Assigning a Call to a Name (line 837):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_72799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 837, 4), 'int')
    # Processing the call keyword arguments
    kwargs_72800 = {}
    # Getting the type of 'call_assignment_69381' (line 837)
    call_assignment_69381_72797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 4), 'call_assignment_69381', False)
    # Obtaining the member '__getitem__' of a type (line 837)
    getitem___72798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 4), call_assignment_69381_72797, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_72801 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___72798, *[int_72799], **kwargs_72800)
    
    # Assigning a type to the variable 'call_assignment_69382' (line 837)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 837, 4), 'call_assignment_69382', getitem___call_result_72801)
    
    # Assigning a Name to a Subscript (line 837):
    # Getting the type of 'call_assignment_69382' (line 837)
    call_assignment_69382_72802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 4), 'call_assignment_69382')
    # Getting the type of 'ret' (line 837)
    ret_72803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 4), 'ret')
    str_72804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 837, 8), 'str', 'pydocsign')
    # Storing an element on a container (line 837)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 837, 4), ret_72803, (str_72804, call_assignment_69382_72802))
    
    # Assigning a Call to a Name (line 837):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_72807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 837, 4), 'int')
    # Processing the call keyword arguments
    kwargs_72808 = {}
    # Getting the type of 'call_assignment_69381' (line 837)
    call_assignment_69381_72805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 4), 'call_assignment_69381', False)
    # Obtaining the member '__getitem__' of a type (line 837)
    getitem___72806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 4), call_assignment_69381_72805, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_72809 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___72806, *[int_72807], **kwargs_72808)
    
    # Assigning a type to the variable 'call_assignment_69383' (line 837)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 837, 4), 'call_assignment_69383', getitem___call_result_72809)
    
    # Assigning a Name to a Subscript (line 837):
    # Getting the type of 'call_assignment_69383' (line 837)
    call_assignment_69383_72810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 4), 'call_assignment_69383')
    # Getting the type of 'ret' (line 837)
    ret_72811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 22), 'ret')
    str_72812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 837, 26), 'str', 'pydocsignout')
    # Storing an element on a container (line 837)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 837, 22), ret_72811, (str_72812, call_assignment_69383_72810))
    
    
    # Call to hasnote(...): (line 838)
    # Processing the call arguments (line 838)
    # Getting the type of 'var' (line 838)
    var_72814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 15), 'var', False)
    # Processing the call keyword arguments (line 838)
    kwargs_72815 = {}
    # Getting the type of 'hasnote' (line 838)
    hasnote_72813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 7), 'hasnote', False)
    # Calling hasnote(args, kwargs) (line 838)
    hasnote_call_result_72816 = invoke(stypy.reporting.localization.Localization(__file__, 838, 7), hasnote_72813, *[var_72814], **kwargs_72815)
    
    # Testing the type of an if condition (line 838)
    if_condition_72817 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 838, 4), hasnote_call_result_72816)
    # Assigning a type to the variable 'if_condition_72817' (line 838)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 4), 'if_condition_72817', if_condition_72817)
    # SSA begins for if statement (line 838)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 839):
    
    # Assigning a Subscript to a Subscript (line 839):
    
    # Obtaining the type of the subscript
    str_72818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, 26), 'str', 'note')
    # Getting the type of 'var' (line 839)
    var_72819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 22), 'var')
    # Obtaining the member '__getitem__' of a type (line 839)
    getitem___72820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 22), var_72819, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 839)
    subscript_call_result_72821 = invoke(stypy.reporting.localization.Localization(__file__, 839, 22), getitem___72820, str_72818)
    
    # Getting the type of 'ret' (line 839)
    ret_72822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 8), 'ret')
    str_72823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, 12), 'str', 'note')
    # Storing an element on a container (line 839)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 839, 8), ret_72822, (str_72823, subscript_call_result_72821))
    
    # Assigning a List to a Subscript (line 840):
    
    # Assigning a List to a Subscript (line 840):
    
    # Obtaining an instance of the builtin type 'list' (line 840)
    list_72824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 840, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 840)
    # Adding element type (line 840)
    str_72825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 840, 23), 'str', 'See elsewhere.')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 840, 22), list_72824, str_72825)
    
    # Getting the type of 'var' (line 840)
    var_72826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 8), 'var')
    str_72827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 840, 12), 'str', 'note')
    # Storing an element on a container (line 840)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 840, 8), var_72826, (str_72827, list_72824))
    # SSA join for if statement (line 838)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Subscript (line 842):
    
    # Assigning a Call to a Subscript (line 842):
    
    # Call to getarrdocsign(...): (line 842)
    # Processing the call arguments (line 842)
    # Getting the type of 'a' (line 842)
    a_72829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 37), 'a', False)
    # Getting the type of 'var' (line 842)
    var_72830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 40), 'var', False)
    # Processing the call keyword arguments (line 842)
    kwargs_72831 = {}
    # Getting the type of 'getarrdocsign' (line 842)
    getarrdocsign_72828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 23), 'getarrdocsign', False)
    # Calling getarrdocsign(args, kwargs) (line 842)
    getarrdocsign_call_result_72832 = invoke(stypy.reporting.localization.Localization(__file__, 842, 23), getarrdocsign_72828, *[a_72829, var_72830], **kwargs_72831)
    
    # Getting the type of 'ret' (line 842)
    ret_72833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 4), 'ret')
    str_72834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 8), 'str', 'arrdocstr')
    # Storing an element on a container (line 842)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 842, 4), ret_72833, (str_72834, getarrdocsign_call_result_72832))
    # Getting the type of 'ret' (line 843)
    ret_72835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 11), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 843)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 843, 4), 'stypy_return_type', ret_72835)
    
    # ################# End of 'common_sign2map(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'common_sign2map' in the type store
    # Getting the type of 'stypy_return_type' (line 824)
    stypy_return_type_72836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_72836)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'common_sign2map'
    return stypy_return_type_72836

# Assigning a type to the variable 'common_sign2map' (line 824)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 0), 'common_sign2map', common_sign2map)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
